# 06 — `runtime/organism/` (gobernanza "T5 soberanía")

4542 LOC, 24 archivos. La capa **arquitectónicamente más coherente y moderna** del proyecto: el
organismo como entidad persistente por encima de los episodios, con constitución, viabilidad,
riesgo bayesiano, trayectoria, linaje y auto-modificación certificada.

**Núcleo:**
- `state.py`: `OrganismState` (frozen) + 5 sub-estados (belief/policy/identity/viability/modification).
- `constitution.py`: 7 invariantes hard + 4 soft con umbrales; `validate()` → verdict
  valid/quarantine/rollback; componentes mutables vs inmutables.
- `trajectory_state_machine.py`: transición nativa `Ψ(T_t, r_t, o_t, u_t)` (la transición T5 real;
  `state.transition_organism_state` es adapter legacy hacia aquí).
- `viability.py` (`ViabilityKernel`, T5, estado) + `viability_kernel.py`
  (`TrajectoryViabilityKernel`, T4, trayectoria).
- `risk.py`: `compute_constitutional_posterior` (posteriores bayesianos + LCB Agresti-Coull).
- `court_runtime.py`: `ConstitutionalCourtRuntime` — corte T5 que ingiere episodios y persiste el
  chain completo (snapshot/window/flow/renorm/risk/failure).
- `self_modification.py`: pipeline sandbox de auto-modificación.
- Soporte: `trajectory`, `lineage`, `transport`, `regime_model`, `constitution_flow`,
  `regime_renormalization`, `risk_process`, `failure_atlas`, `snapshot`, `invariants`,
  `t5_mode`/`t4_mode`, adapters `*_runtime`.

**Cableado:** `get_t5_mode()` **por defecto = "on"** (t5_mode.py: lee `RNFE_T5_MODE`, si no
`RNFE_T4_MODE`, default `"on"`). `ConstitutionalCourtRuntime` se invoca desde
`certification/promotion_gate.py` y `transfer_posterior.py` → **el chain T5 corre por defecto** en
el pipeline vivo de certificación.

---

## Hallazgos

### [RIESGO] El sandbox de auto-modificación es un no-op sin `apply_fn` (self_modification.py:182-186)
```python
if apply_fn is not None:
    simulated = apply_fn(current_state, proposal)
else:
    simulated = current_state  # ← identidad: no aplica el cambio
```
Si no se pasa `apply_fn`, `sandbox_simulate` evalúa el **estado actual sin modificar**. La decisión
(accept/quarantine/reject) se basa entonces en la salud actual, no en el efecto de la propuesta →
puede "aceptar" una modificación que **nunca simuló**. El pipeline solo es real si el llamador
provee `apply_fn`; el default es inseguro para una etapa que se llama "sandbox".

### [DISEÑO] `composite_health` y `drift_identity` se miden contra una identidad por defecto
- `OrganismState.composite_health` (state.py:261) usa `1.0 - identity.identity_distance(IdentityState())`.
- `risk.compute_constitutional_posterior` (risk.py:205) usa
  `identity_distance(OrganismState().identity)` como `drift_identity`.
Ambos comparan contra una `IdentityState` **fresca/genesis** (lineage="genesis", invariantes
vacíos, hash ""). Un organismo con identidad real (lineage + hash) tiene distancia alta respecto al
default → `composite_health` se penaliza y `drift_identity` se infla por el mero hecho de **tener
identidad**, no por derivar de su propio baseline. En `risk` es defendible (drift desde génesis);
en `composite_health` es un sesgo no intencionado.

### [DISEÑO] Dos kernels de viabilidad con nombres archivo/clase cruzados
- `viability.py` → clase `ViabilityKernel` (basada en estado, T5).
- `viability_kernel.py` → clase `TrajectoryViabilityKernel` (basada en trayectoria, T4).
El archivo `viability_kernel.py` **no** contiene `ViabilityKernel`. Fácil de confundir; ambos
coexisten (T4 trayectorial usado por `court_runtime`, T5 estatal usado por `self_modification`).

### [DISEÑO] `constitution.validate` evalúa los hard checks dos veces (constitution.py:320-345)
Una vez en el bucle principal y otra al computar `margin_to_threshold`. Doble evaluación de las 7
comprobaciones por validación. Además dos invariantes **soft** (`policy_stability` max 0.50 y
`drift_tolerance` max 0.60) miden el **mismo** campo `policy.accumulated_drift` con umbrales
distintos — solapamiento.

### [DISEÑO] `t4_mode` es alias delgado de `t5_mode`; T5 activo por defecto
`t4_mode.get_t4_mode()` simplemente delega en `get_t5_mode()`. Y el default es `"on"`: sin
variables de entorno, **el organismo T5 gobierna**. Conviene tenerlo claro: los benchmarks corren
con la corte constitucional activa salvo que se ponga `RNFE_T5_MODE=off`.

### [RIESGO] Amplificación de escritura por episodio en la corte (court_runtime.py:235-360)
`ingest_episode` persiste por **cada** episodio: organism_snapshot + trajectory_window +
trajectory_flow_report + (renorm_event si cambia régimen) + **N** constitutional_risk_states
(organism/modification/inheritance/edge) + failure_atlas_events. Combinado con el `EventBus.emit`
(fichero+DB por evento) y el `materialize_artifact` del runner, un episodio dispara muchas
escrituras a storage. Relevante para el bug de bloqueo SQLite ([02_storage.md](02_storage.md)).

### [DISEÑO] Posteriores bayesianos heurísticos
`risk.py` combina prior·likelihood en forma de odds (`p/(p+ (1-p)(1-l))`), con priors/pesos
ad-hoc (0.80/0.20, etc.). Es un esquema "bayesiano-ish" razonable como proxy, no una verosimilitud
calibrada. El LCB Agresti-Coull con `n=6+n_historical` es correcto pero el `n` base de 6 es
arbitrario.

---

## Aspectos positivos
- **Modelo de gobernanza coherente y completo**: estado constitucional, kernel de viabilidad como
  región `K = {x | C(x)=valid ∧ M(x)≥0}`, posteriores por scope (local/transfer/modification/
  inheritance), corte que materializa todo el chain con `prev_state_id`/`step_index`
  (encaja con el esquema de storage y su migración de columnas).
- `trajectory.py`: histories + digest + window + invariant report bien factorizados.
- `lineage.py`: reglas de herencia, genesis/modification/rollback/divergence, consistency score.
- `transport.py` / `regime_renormalization.py`: proyección belief/policy entre regímenes con
  residual + incertidumbre + coste de recuperación.
- `constitution_flow.py`: umbrales por cuantil (calibrables) sobre la trayectoria.
- Todo en frozen dataclasses; `t5_mode` con off/shadow/on (permite correr T5 en sombra).

## Veredicto
La capa de mejor diseño del repo. Riesgo concreto a corregir: el **sandbox no-op** de
auto-modificación (peligroso para una etapa de seguridad). Deuda menor: doble cómputo en
`validate`, identidad-vs-default en `composite_health`, naming cruzado de los dos kernels de
viabilidad, y la amplificación de escritura por episodio (que agrava el bug de SQLite).
