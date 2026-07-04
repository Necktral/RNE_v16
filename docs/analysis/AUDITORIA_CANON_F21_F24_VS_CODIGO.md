# Auditoría: canon matemático RNFE (f2.1–f2.4) ↔ código vivo

**Fecha:** 2026-06-10 · **Rama auditada:** `work/external-reasoner-latency-checkpoint` (post-unificación de arquitecturas, storage en PostgreSQL).

**Fuentes del canon** (documentos de diseño aportados por el autor):
1. *Motores de razonamiento reales* — catálogo de motores por familia (Z3, Prolog, clingo, ProbLog, DoWhy, …).
2. *RNFE‑f2.1 — Addendum de Robustificación* — B_safe, Edge 2.1/RQA, S‑I‑E 2.0 con CVaR, MPC robusto, certificados 𝔠ₜ, kill‑switch.
3. *RNFE f2.2* — H‑Net jerárquico con dynamic chunking, Hctrl, ratio loss, PAI, F→M→S.
4. *RNFE f2.3* — mejoras fractales: MFE, TFI, memoria IFS‑FDE, perceptores fractales, política fractal de modos.
5. *f2.4 / blueprint v0.2* — estado total Xₜ, kernel de viabilidad 𝒱, IoC*, continuidad C^cont, morfogénesis ρₜ, linajes μₜ, scheduler semi‑Markov, disipación Dₜ, funcional maestro.

**Metodología:** la documentación .md histórica del repo está desfasada; **la única fuente de verdad es el código**. Cada veredicto está anclado a `archivo:línea` y fue verificado por lectura directa + grep. "Camino vivo" = lo alcanzable desde `runtime/world/scenario_runner.py::run_episode()` (el lazo episódico real). El legacy en cuarentena (`runtime/core/`, `runtime/evolution/`, `runtime/control/homeostasis/`, `runtime/agents/`) NO cuenta como implementación.

## Escala de veredictos

| Veredicto | Significado |
|---|---|
| ✅ **REAL** | Computa la fórmula (o un equivalente fiel) |
| 🟡 **PARCIAL‑HEURÍSTICO** | Aproxima la idea con heurística más simple |
| 🏷️ **SOLO‑NOMBRE** | El nombre existe; la matemática no |
| 💤 **IMPLEMENTADO‑INERTE** | Código completo pero nadie lo invoca en el camino vivo |
| 🧪 **SOLO‑EN‑TESTS** | Existe solo en el laboratorio de tests/benchmarks |
| ☣️ **LEGACY** | Solo en la cuarentena (no cuenta) |
| ❌ **NO‑EXISTE** | Cero rastro en el repo |

---

## Resumen ejecutivo

| Pieza del canon | Doc | Veredicto | Ancla principal |
|---|---|---|---|
| Estado total Xₜ | f2.4 §1 | 🟡 PARCIAL | `runtime/organism/state.py:232` |
| Dinámica híbrida (4 acciones uₜ) | f2.4 §2 | 🟡 PARCIAL (2 de 4) | `runtime/world/scenario_runner.py:169` |
| Kernel de viabilidad 𝒱 (mirando‑adelante) | f2.4 §3 | 🟡 PARCIAL (retrospectivo) | `runtime/organism/viability.py` |
| IoC* = IoC − λΩ·Ωₜ | f2.4 §4 | 🏷️ SOLO‑NOMBRE (sin Ωₜ) | `runtime/certification/ioc_proxy.py:1` |
| Continuidad identitaria C^cont | f2.4 §5 | 🟡 PARCIAL | `runtime/reality/continuity.py:79` |
| Certificado ampliado 𝔠ₜ⁺ | f2.4 §5 / f2.1 | 🟡 PARCIAL (sin CVaR/B_safe/Edge) | `runtime/storage/records.py:99` |
| Morfogénesis ρₜ (reescritura tipada) | f2.4 §6 | 💤 INERTE | `runtime/organism/self_modification.py` |
| Linajes μₜ (replicador–mutador, Z_stable) | f2.4 §7 | 🟡 PARCIAL‑PASIVO | `runtime/organism/lineage.py` |
| Scheduler semi‑Markov V(x) | f2.4 §8 | 🟡 HEURÍSTICO | `runtime/reasoning/scheduler_meta/` |
| Disipación medida Dₜ (7 términos) | f2.4 §9 | 🟡 PARCIAL (2 de 7) | `runtime/control/msrc/vram_sampler.py` |
| Funcional maestro (max E Σ γᵗ[…]) | f2.4 §10 | ❌ NO‑EXISTE | — |
| Barrera B_safe (log‑barrier) | f2.1 | 🏷️ SOLO‑NOMBRE (umbrales sueltos) | `runtime/control/msrc/` |
| Edge 2.1 / RQA (RR, DET, FD, ρ(J)) | f2.1 | ❌ NO‑EXISTE en runtime | `runtime/reality/edge_benchmark.py` |
| S‑I‑E 2.0 (ACEPTAR/BUFFER/RECHAZAR + CVaR) | f2.1 | 🟡 PARCIAL (LCB bayesiano, sin CVaR) | `runtime/certification/transfer_posterior.py` |
| CVaR_α | f2.1/f2.4 | 🧪 SOLO‑EN‑TESTS | `tests/reasoning_stress/fractal_geometries.py:1163` |
| MPC robusto Hctrl | f2.1/f2.2 | 🟡 PARCIAL (FSM reactiva, no MPC) | `runtime/control/msrc/controller.py:56` |
| Causalidad invariante por entornos | f2.1 | ❌ NO‑EXISTE | — |
| κtop / κgeo | f2.1 | ❌ NO‑EXISTE | — |
| Pool multi‑agente log‑opinión | f2.1 | ❌ NO‑EXISTE | — |
| EWC causal | f2.1 | ❌ NO‑EXISTE (no hay θ que proteger) | — |
| Tests metamórficos T_P | f2.1 | ❌ NO‑EXISTE | — |
| Kill‑switch cognitivo | f2.1 | ☣️ LEGACY (roto) | `runtime/control/homeostasis/shutdown_logic.py` |
| H‑Net dynamic chunking | f2.2 | ✅ REAL pero 💤 ISLA | `engines/hnet/modules/dc.py:47` |
| Ratio loss L_ratio | f2.2 | ✅ REAL pero 💤 sin uso | `engines/hnet/utils/train.py:13` |
| PAI (acción informacional) | f2.2 | ❌ NO‑EXISTE | — |
| Pipeline F→M→S | f2.2 | 🟡 MINIATURA REAL | `scenario_runner.py:197,227,250` |
| MFE (mapa fractal de entrenabilidad) | f2.3 | ❌ NO‑EXISTE (box‑counting solo en tests) | `tests/reasoning_stress/fractal_geometries.py` |
| TFI (κ_info por jacobianos) | f2.3 | ❌ NO‑EXISTE | — |
| Memoria IFS‑FDE | f2.3 | ❌ NO‑EXISTE (MFM = Jaccard) | `runtime/memory/mfm_lite/retrieval.py:115` |
| Perceptores fractales | f2.3 | ❌ NO‑EXISTE | — |
| Política fractal de modos πₜ(r) | f2.3 | ❌ NO‑EXISTE | `runtime/organism/regime_renormalization.py` |
| Familias DED/ABD/ANA/CAU/CTF/PROB | doc 1 | ✅ REAL (DED con Z3) | `runtime/reasoning/families/` |
| Familias IND/PLAN/OPT/NESY/EVO | doc 1 | 🏷️ STUB PURO | `families/{ind,plan,opt,nesy,evo_search}/__init__.py:7` |
| Motores externos (DoWhy, pgmpy, networkx, causal‑learn) | doc 1 | 🏷️ DEPS FANTASMA (0 imports) | `requirements.reasoning-core-causal.txt` |
| Razonador LLM gated | — | ✅ REAL (advisory, opt‑in) | `runtime/reasoning/external_models/gating.py` |

---

## §1 — f2.4/v0.2: el organismo viable, morfogénico y heredable

### 1.1 Estado total Xₜ = (Gₜ, Σₜ, Fₜ, Wₜ, Mₜ, θₜ, φₜ, μₜ, 𝔠ₜ, 𝒯ₜ) — 🟡 PARCIAL

Lo que existe es `OrganismState` ([runtime/organism/state.py:232](../../runtime/organism/state.py#L232)): 5 sub‑estados frozen (belief, policy, identity, viability, modification) que se transicionan cada episodio vía `TrajectoryStateMachine.advance_state` ([trajectory_state_machine.py:21](../../runtime/organism/trajectory_state_machine.py#L21)) e instanciados en el camino vivo ([scenario_runner.py:105-120](../../runtime/world/scenario_runner.py#L105-L120)).

Componente por componente del Xₜ formal:

| Componente | Estado | Evidencia |
|---|---|---|
| Gₜ grafo tipado del organismo | ❌ NO‑EXISTE | No hay grafo de módulos/agentes/enlaces. El único grafo es el SMG episódico (proposiciones), no estructura del organismo. |
| Σₜ (SMG) | 🟡 existe pero DESACOPLADO | `SMGMin` ([runtime/smg/smg_min.py](../../runtime/smg/smg_min.py)): observaciones + signos + relaciones support/contradiction. Se snapshotea por episodio, no es campo de `OrganismState`. |
| Fₜ (LOT‑F) | 🟡 existe pero efímero | `LOTFMin` parsea/chequea la fórmula por episodio ([scenario_runner.py:197-199](../../runtime/world/scenario_runner.py#L197-L199)); no persiste como estado. |
| Wₜ (C‑GWM) | 🟡 existe en miniatura | `CGWMMin` ([runtime/world/cgwm_min.py](../../runtime/world/cgwm_min.py)): transición determinista de 3 variables. |
| Mₜ (memoria MFM/VFD) | 🟡 existe fuera del estado | `EpisodeMemoryStore` + retrieval; no es campo de Xₜ. |
| θₜ/φₜ parámetros rápidos/lentos | ❌ NO‑EXISTE | No hay un solo parámetro aprendible en el camino vivo. Todos los pesos son constantes a mano (p.ej. los 0.20/0.25/0.15 de `composite_confidence`, [state.py:53-58](../../runtime/organism/state.py#L53-L58)). |
| μₜ medida de linajes | 🟡 historial, no medida | Ver §1.7. |
| 𝔠ₜ certificado | 🟡 existe en storage | Ver §1.5. |
| 𝒯ₜ telemetría | 🟡 existe en silo | nvidia‑smi real en MSRC (§2.4) pero no entra al estado. |

**Gap esencial:** Xₜ del canon es UNA entidad con 10 componentes; en el código son ~6 piezas reales pero **desacopladas** entre sí, y faltan por completo Gₜ y θₜ/φₜ (no hay nada que aprenda ni grafo de sí mismo).

### 1.2 Dinámica híbrida X_{t+1} ∈ 𝓕(Xₜ, uₜ), uₜ = (a^env, o^raz, a^ctrl, ρₜ) — 🟡 PARCIAL (2 de 4)

- **a^env (intervención sobre el mundo):** ✅ REAL — `scenario.select_intervention` + `factual_transition` ([scenario_runner.py:214-237](../../runtime/world/scenario_runner.py#L214-L237)).
- **o^raz (opción de razonamiento activa):** ✅ REAL (heurístico) — el MetaScheduler selecciona secuencia de familias por régimen ([scheduler_meta/policy.py](../../runtime/reasoning/scheduler_meta/policy.py)).
- **a^ctrl (acción homeostática Hctrl):** 💤 fuera del lazo — MSRC existe y decide acciones, pero **no se invoca dentro de run_episode**; solo lo usa el benchmark `runtime/reality/msrc_policy_benchmark.py:20`.
- **ρₜ (reescritura estructural):** 💤 INERTE — ver §1.6.

El episodio vivo es por tanto un sistema dinámico de **2 acciones**, no 4.

### 1.3 Kernel de viabilidad 𝒱 = {x ∈ safe : ∃π ∀k X_{t+k} ∈ safe ∧ IoC* ≥ ι ∧ C^cont ≥ c} — 🟡 PARCIAL (retrospectivo)

El canon define "vivo" como **existencia de política futura** que preserve seguridad+cierre+continuidad. Lo implementado:

- `ViabilityKernel.assess(state, previous_state)` ([runtime/organism/viability.py](../../runtime/organism/viability.py)) computa un margen como combinación lineal del estado **actual** (≈ 30% margen + 25% belief + 25% degradación + 20% recuperación) más validación constitucional.
- `TrajectoryViabilityKernel` ([viability_kernel.py:20-48](../../runtime/organism/viability_kernel.py#L20-L48)) agrega drift/histéresis/recuperación sobre una ventana **pasada** de la trayectoria.

**Gap:** ambos son *scores retrospectivos*. No hay horizonte H, no hay búsqueda de política, no hay predicción. El sistema sabe si *estuvo* bien, no si *puede seguir* vivo. La distinción canónica "estar seguro vs estar vivo" no está implementada.

### 1.4 IoC* = IoC − λΩ·Ωₜ (coherencia global multi‑contexto) — 🏷️ SOLO‑NOMBRE

`IoCProxy.compute` ([ioc_proxy.py:7-28](../../runtime/certification/ioc_proxy.py#L7-L28)) — cuyo docstring se autodenomina "IoC* proxy operativo" — es:

```
ioc = 0.45·continuity + 0.25·closure + 0.20·trace − 0.06·uncertainty − 0.14·collapse
```

Es un proxy lineal razonable de **cierre local**. Pero la pieza dura del canon — la obstrucción Ωₜ que suma divergencias d_S/d_F/d_W entre secciones locales en solapamientos de contextos, más el error del ciclo Φ_{M→S}Φ_{F→M}Φ_{S→F} − I — **no existe**: no hay cobertura de contextos 𝔘ₜ, no hay restricciones r_ij, no hay medición del ciclo F→M→S como operador. El riesgo que Ωₜ ataca (puntuar alto en cierre local mientras se deriva semánticamente a escala global) está sin guardia.

### 1.5 C^cont y certificado ampliado 𝔠ₜ⁺ — 🟡 PARCIAL

**C^cont** del canon: ω₁·sim(Σₜ,Σₜ₋₁) + ω₂·sim(Gₜ,Gₜ₋₁) + ω₃·sim(Mₜ,Mₜ₋₁) + ω₄·𝟙[rollback recuperable]. Lo implementado ([continuity.py:79-109](../../runtime/reality/continuity.py#L79-L109)):

```
score = 0.40·Jaccard(signos) + 0.30·consistencia_causal + 0.20·estabilidad_secuencia + 0.10·integridad_traza
```

- sim(Σ) ≈ Jaccard de proposiciones ✅ (aproximación honesta).
- sim(G) ❌ (no hay Gₜ). sim(M) ❌ (memoria fuera del estado). 𝟙[rollback] 🟡 — existe el booleano `rollback_readiness` pero **no hay mecanismo ejecutable de rollback** en el camino vivo.

**𝔠ₜ⁺** del canon: (IoC*, B_safe, L_edge2.1, CVaR_α[−ΔIoC*], C^cont). Campos reales de `EpisodeCertificateRecord` ([records.py:99-114](../../runtime/storage/records.py#L99-L114)): `continuity_score, ioc_proxy, risk_score, verdict, rollback_ready, promotion_candidate` + artefactos SMG/LOTF/world.

| Componente formal | ¿En el certificado real? |
|---|---|
| IoC* | 🟡 `ioc_proxy` (sin Ω) |
| B_safe | ❌ |
| L_edge2.1 (RQA) | ❌ |
| CVaR_α[−ΔIoC*] | ❌ (`risk_score` es lineal: ≈ 0.55·(1−cont) + 0.30·(1−ioc) + …, [certificate_builder.py](../../runtime/certification/certificate_builder.py)) |
| C^cont | 🟡 `continuity_score` (parcial, ver arriba) |

Es decir: del quíntuple formal, hay 2 proxies parciales y faltan los 3 componentes de riesgo/estabilidad.

### 1.6 Morfogénesis ρₜ: reescritura tipada L⇒R admisible — 💤 IMPLEMENTADO‑INERTE

Existe un pipeline completo: `SelfModificationPipeline` ([runtime/organism/self_modification.py](../../runtime/organism/self_modification.py)) con etapas proposal → precheck constitucional → sandbox → stress de edge → estimación de posterior → accept/quarantine/reject, y separación mutable/inmutable (transport_parameters, selection_policy, memory_scoring_weights vs invariantes constitucionales). Esto **mapea sorprendentemente bien** al esquema de admisibilidad 𝒜(Xₜ) del canon (viabilidad + ΔIoC ≥ ε + riesgo ≤ τ + invariantes).

**Pero:** el único llamador en todo el repo es el test `tests/organism/test_self_modification.py`. `scenario_runner` arrastra `modification=current.modification` sin tocarla jamás ([trajectory_state_machine.py:134](../../runtime/organism/trajectory_state_machine.py#L134)). Además:
- No hay reescritura de **grafo** L⇒R (no hay Gₜ que reescribir): las "modificaciones" son cambios de parámetros de configuración.
- La condición CVaR del canon no puede evaluarse (no hay CVaR, §2.3).
- No hay tests metamórficos T_P.

**Este es el hueco #1 para "autoevolutivo": el órgano de auto‑modificación existe y está desconectado.**

### 1.7 Linajes μₜ: medida + replicador–mutador + Z_stable — 🟡 PARCIAL‑PASIVO

Existe `LineageState` con historial tipado (genesis, modification_accepted/rejected, transport_inherited, rollback, divergence) y `InheritanceRule` con condiciones (certified_safe, constitution_consistent, baseline_preserved, no_contamination) — [runtime/organism/lineage.py](../../runtime/organism/lineage.py). El `lineage_id` se propaga en `IdentityState` ([scenario_runner.py:111](../../runtime/world/scenario_runner.py#L111)).

**Gap:** (a) μₜ del canon es una **medida de probabilidad sobre un espacio de motivos 𝒵** con evolución replicador–mutador μₜ₊₁ ∝ e^{βF}·𝒦μₜ; lo implementado es un **log lineal de eventos**. (b) No hay espacio de motivos, ni kernel de mutación, ni fitness F(ζ;Xₜ), ni criterio Z_stable de reaparición robusta entre semillas/entornos. (c) Las `InheritanceRule` jamás se evalúan en el camino vivo: el linaje registra, no hereda.

### 1.8 Scheduler de razonamientos como control semi‑Markov — 🟡 HEURÍSTICO

Canon: opciones o con política π_o, tiempo de parada τ_o, V(x) = max_o E[Σ γᵏ r + γ^τ V(X_τ)], con r = ΔIoC* − λE·ΔE − λD·𝒟 − λB·B_safe.

Implementado ([runtime/reasoning/scheduler_meta/](../../runtime/reasoning/scheduler_meta/)):
- **Budget heurístico** ([budgeting.py](../../runtime/reasoning/scheduler_meta/budgeting.py)): base 6 pasos ± bonos por umbrales de uncertainty/contradiction/causal_risk, clamp [4,10]; risk_budget lineal.
- **Selección por régimen** ([policy.py](../../runtime/reasoning/scheduler_meta/policy.py)): resuelve regime_label (viability_edge, heterogeneous_*, vram_favorable, homogeneous_safe) → pisos obligatorios de familias (`MANDATORY_FAMILY_FLOORS`) → perfil + overlays en orden fijo.
- **Ejecución secuencial** con early‑stop solo en modo adaptive.

No hay función de valor, ni opciones con parada, ni modelo de recompensa. Es un **selector determinista por reglas** — reproducible y auditable (virtud real), pero no es el control óptimo del canon. Nota: los ingredientes para entrenarlo offline ya se persisten (traces de razonamiento + certificados en PG).

### 1.9 Disipación medida Dₜ (7 términos) — 🟡 PARCIAL (≈2 de 7)

Canon: Dₜ = α₁·ΔVRAM⁺ + α₂·ΔTEMP⁺ + α₃·ReLU(ρ(J)−(1−ε)) + α₄·φ_band(RR) + α₅·φ_band(DET) + α₆·ΔH_ruta⁺ + α₇·ΔH_Σ⁺.

- VRAM/TEMP: ✅ se **miden de verdad** — `NvidiaVRAMSampler` lee `nvidia-smi` (memory.used/total, temperature.gpu) con timeout y computa presión/headroom/oportunidad ([vram_sampler.py:43-139](../../runtime/control/msrc/vram_sampler.py#L43-L139)). Pero como valores absolutos con umbrales, no como deltas positivos ponderados.
- ρ(J): ❌ no se computa ningún jacobiano (el único "lyapunov" del repo está en el legacy [runtime/core/loss.py:96](../../runtime/core/loss.py#L96), cuarentena).
- RR/DET (RQA): ❌ (§2.2). Entropías de ruta/SMG: ❌ (hay "entropy" en telemetría legacy como mix CPU/mem, no entropía de trayectoria).
- Dₜ como cantidad agregada que entre en decisiones/certificado: ❌.

### 1.10 Funcional maestro — ❌ NO‑EXISTE

No hay ningún lugar donde se optimice max_{π,ρ,μ} E Σ γᵗ[IoC* − λ_safe·B_safe − λ_edge·L_edge − λ_D·D + λ_Λ·R_stable]. Tampoco podría: faltan 4 de sus 5 términos y los 3 espacios de decisión (π aprendible, ρ activa, μ como medida). Es la "función objetivo del ser" y hoy el sistema no optimiza nada — ejecuta.

---

## §2 — f2.1: Robustificación

### 2.1 Barrera unificada B_safe (log‑barrier sobre VRAM/TEMP/ρ(J)/FD) — 🏷️ SOLO‑NOMBRE

No hay φ_bar(x;δ) = −log(δ+(1−x)) en ninguna parte. Lo que hay: umbrales sueltos (presión VRAM > 0.85 descuenta oportunidad en [vram_sampler.py:131-139](../../runtime/control/msrc/vram_sampler.py#L131-L139); fallback térmico 40 °C en telemetría). De las 4 variables de la barrera, 2 se miden (VRAM, TEMP) y 2 no existen (ρ(J), FD). No hay término λ_safe·B_safe en ninguna pérdida porque no hay pérdidas (no hay entrenamiento).

### 2.2 Edge 2.1: RQA (RR, DET) + espectro + fractalidad — ❌ NO‑EXISTE en runtime

Búsqueda exhaustiva de recurrence/RR/DET/spectral/jacobian en `runtime/`: vacío (salvo legacy). [runtime/reality/edge_benchmark.py](../../runtime/reality/edge_benchmark.py) menciona "Edge 2.1" en su docstring pero implementa otra cosa (valiosa pero distinta): un stress‑test A→B→A entre escenarios con clasificación de bordes (compatible/homomorphic/analogical/adversarial). No hay matriz de recurrencia R_ij, ni P(l) de diagonales, ni dimensión fractal sobre trayectorias del organismo. El box‑counting existe — pero en el laboratorio de tests (§4.1).

### 2.3 S‑I‑E 2.0: ACEPTAR/BUFFER/RECHAZAR con Pr(ΔIoC≥0) y CVaR — 🟡 PARCIAL (pariente honesto)

Lo más cercano en vivo es el par `PromotionGate` + `TransferPosterior`:
- `PromotionGate.process_episode` ([promotion_gate.py:81-87](../../runtime/certification/promotion_gate.py#L81-L87)) → `assess_transfer` → posterior bayesiano con **LCB Agresti‑Coull** ([transfer_posterior.py:44-58](../../runtime/certification/transfer_posterior.py#L44-L58)) y alcances `local_only / compatible_transfer / analogical_hint_only / blocked`.
- Correspondencia conceptual: ACEPTAR ≈ certified+promotion_candidate; BUFFER ≈ local_only/analogical_hint_only; RECHAZAR ≈ blocked/rejected.

**Gaps:** (a) el posterior es sobre P(transferencia segura), **no** sobre ΔIoC; (b) **CVaR_α[−ΔIoC] no existe en runtime** — la única implementación de CVaR del repo es `compute_cvar()` en el harness de tests ([tests/reasoning_stress/fractal_geometries.py:1163](../../tests/reasoning_stress/fractal_geometries.py#L1163), con `cvar_alpha=0.95` en sus MetaParameters), correcta pero nunca usada por la certificación; (c) la regla se aplica a episodios/transferencias, no a **candidatos de auto‑modificación** (que es el uso central en el canon), porque ρₜ está inerte.

### 2.4 HNet 2.1: MPC robusto de recursos — 🟡 PARCIAL (FSM reactiva)

`MSRCController.step()` ([controller.py:56-191](../../runtime/control/msrc/controller.py#L56-L191)) hace: muestrear VRAM real → estimar escala → decidir acción discreta (stay/fork_probe/commit/discard/recover/escalate) → ejecutar transición → auditar oscilación. Es un controlador **reactivo por reglas con sondas** — sin horizonte H, sin conjunto de incertidumbre 𝒰, sin min‑max. El fallback con histéresis del canon tiene un eco en la auditoría de oscilación, pero el "modo seguro" forzado no existe como mecanismo. Además **no corre dentro del episodio** (solo en `msrc_policy_benchmark.py`).

### 2.5 Resto de f2.1 — ❌

- **Causalidad invariante por entornos** (∂E[Y|X_S,e] independiente de e, Var_e[ACE]): nada. La causalidad viva es la firma declarativa por escenario ([causal_signature.py](../../runtime/world/causal_signature.py)) — un DAG **escrito a mano**, no aprendido ni validado entre entornos.
- **κtop/κgeo:** nada.
- **Pool multi‑agente log‑opinión** p_pool ∝ Π p_a^{w_a}: nada vivo (el "FenixAgent" es legacy roto; los "pool" del repo son ThreadPool/dedup).
- **EWC causal:** sin sentido hoy — no hay parámetros θ que proteger del olvido.
- **Pruebas metamórficas T_P:** nada.
- **Kill‑switch cognitivo:** existe el diseño completo (PhasedShutdown con 4 niveles y protocolos reversibles) pero en ☣️ legacy con imports rotos ([shutdown_logic.py](../../runtime/control/homeostasis/shutdown_logic.py)). En el camino vivo no hay HALT.

---

## §3 — f2.2: H‑Net jerárquico

### 3.1 Renombrado Hctrl vs H‑Net — ✅ cumplido de facto

La separación del canon se respeta estructuralmente: el control homeostático vivo es MSRC (`runtime/control/msrc/`) y la red jerárquica vive aparte en `engines/hnet/`. (El legacy `runtime/control/homeostasis/` usa la nomenclatura vieja, pero está en cuarentena.)

### 3.2 Dynamic chunking — ✅ REAL… pero 💤 ISLA total

Hallazgo notable: **la matemática de f2.2 sí está implementada**, fielmente, en `engines/hnet/`:
- Router con score de frontera por similitud coseno entre tokens consecutivos: `boundary_prob = (1−cos)/2` — `RoutingModule` ([engines/hnet/modules/dc.py:47-139](../../engines/hnet/modules/dc.py#L47-L139)).
- STE (straight‑through estimator) para las fronteras duras b_t ([dc.py:20-31](../../engines/hnet/modules/dc.py#L20-L31)).
- `ChunkLayer`/`DeChunkLayer` = down/upsampling físicos por boundary_mask, con suavizado.
- Jerarquía recursiva de stages encoder→routing→main→dechunk→decoder ([engines/hnet/models/hnet.py](../../engines/hnet/models/hnet.py)).
- Ratio loss ≈ L_ratio: `load_balancing_loss` ([engines/hnet/utils/train.py:13-40](../../engines/hnet/utils/train.py#L13-L40)).

**Pero:** cero imports desde `runtime/` o `scripts/` (verificado por grep). No hay loop de entrenamiento (el propio train.py declara "not used inside the HNet package"). Los encoders del canon serían Mamba; el Mamba vendoreado (`engines/mamba_vendor/`) tampoco está integrado. H‑Net es un **motor terminado esperando un fuselaje**.

### 3.3 PAI, F→M→S, espectro fractal jerárquico — ❌ / 🟡

- **PAI (acción informacional 𝒜[Θ]):** ❌ — no hay funcional, no hay Euler‑Lagrange, no hay Θ.
- **Pipeline F→M→S:** 🟡 existe en miniatura y por episodio: fórmula LOT‑F parseada/tipada ([scenario_runner.py:197-199](../../runtime/world/scenario_runner.py#L197-L199)) → transición C‑GWM ([:227-237](../../runtime/world/scenario_runner.py#L227-L237)) → signos/relaciones SMG ([:240-262](../../runtime/world/scenario_runner.py#L240-L262)). Es el ciclo S→F→M→S conceptual con piezas mínimas (parser booleano, mundo de 3 variables, grafo plano), sin operadores Φ medibles ni error de ciclo.
- **Espectro fractal jerárquico {𝓕^(s)} y asignación χ de modos a niveles:** ❌ — no hay niveles (H‑Net es isla) ni espectro.

---

## §4 — f2.3: mejoras fractales

### 4.1 MFE, box‑counting, frontera de entrenabilidad — 🧪 SOLO‑EN‑TESTS

`estimate_fractal_dimension_boxcount` (normalización a cubo unidad + ajuste en región de escalado) y `measure_multiscale_boundary` viven en [tests/reasoning_stress/fractal_geometries.py](../../tests/reasoning_stress/fractal_geometries.py) y [fractal_utils.py](../../tests/reasoning_stress/fractal_utils.py), junto con un catálogo de ~12 familias de geometrías (Sierpinski, Cantor, Lorenz, Julia, …). Es un **laboratorio de estrés del scheduler**, valioso, pero: (a) la "frontera" que mide es de geometrías sintéticas, no la frontera de entrenabilidad en espacio de hiperparámetros Λ del MFE (no hay entrenamiento, así que no puede haber Y_m(λ)); (b) cero imports desde runtime/.

### 4.2 TFI, IFS‑FDE, perceptores, política fractal πₜ(r) — ❌

- **TFI:** sin jacobianos por nivel no hay κ_info; nada.
- **Memoria IFS‑FDE:** la memoria real es `mfm_lite`: scoring por **Jaccard de tokens** ([retrieval.py:115-128](../../runtime/memory/mfm_lite/retrieval.py#L115-L128)) + condensación micro/meso/macro tipo GROUP‑BY ([condenser.py](../../runtime/memory/mfm_lite/condenser.py)). Ni IFS, ni interpolación fractal, ni FDE, ni embeddings.
- **Perceptores fractales:** nada.
- **πₜ(r) ∝ exp(−gₜ(r)):** lo más cercano es `regime_renormalization.py` (factor de asimetría acotado [0.25, 2.5] que reescala constraints) — útil pero sin relación con la softmax de desajuste FD/κ del canon.

---

## §5 — Motores de razonamiento (doc 1) vs realidad

### 5.1 Familias vivas

| Familia | Realidad | Veredicto |
|---|---|---|
| **DED** | **Z3 de verdad**: parser LOT‑F → traducción a AST Z3 → Solver con SAT/UNSAT, literales entailed por push/pop, unsat core ([families/ded/engine.py:16](../../runtime/reasoning/families/ded/engine.py#L16), translator.py) | ✅ REAL — única familia con motor del catálogo |
| **ABD** | Ranking determinista de intervenciones contra la firma causal (dirección, magnitud, rol semántico, alarma) — [core_inference.py:220-269](../../runtime/reasoning/families/core_inference.py#L220-L269) | ✅ REAL (propio, no motor externo) |
| **CAU** | Efecto = factual − contrafactual vs dirección esperada de la firma; strength, direction_match — [core_inference.py:143-187](../../runtime/reasoning/families/core_inference.py#L143-L187) | ✅ REAL (propio) |
| **CTF** | Comparación factual≥contrafactual según optimization_direction; agreement con relation_kind — [core_inference.py:192-215](../../runtime/reasoning/families/core_inference.py#L192-L215) | ✅ REAL (propio) |
| **ANA** | Top‑hit de memoria por alignment_score; fallback overlap de vocabulario — [core_inference.py:274-306](../../runtime/reasoning/families/core_inference.py#L274-L306) | ✅ REAL (simple) |
| **PROB** | Fusión de evidencia ponderada + **LCB Agresti‑Coull** — [core_inference.py:311-374](../../runtime/reasoning/families/core_inference.py#L311-L374) | ✅ REAL (propio) |
| HEUR / FAL_GUARD / DIA_ADV | Heurísticas de 1‑2 umbrales (triage, riesgo de falacia, desafío adversarial) | 🟡 overlays mínimos |
| EML_SR | Regresión simbólica real (tree search prof.≤3 + R² + estabilidad + validez de dominio) en shadow opt‑in — [runtime/symbolic/eml/](../../runtime/symbolic/eml/) | ✅ REAL (shadow) |
| **IND, PLAN, OPT, NESY, EVO_SEARCH** | `{"status": "idle", "confidence": 0.0}` — [families/ind/__init__.py:7-9](../../runtime/reasoning/families/ind/__init__.py#L7-L9) e idénticos | 🏷️ STUB PURO |

### 5.2 Deps fantasma — el catálogo está "instalado" pero no conectado

`requirements.reasoning-core-causal.txt` lista **dowhy==0.14, causal-learn==0.1.4.5, pgmpy==1.1.0, networkx==3.4.2** (+ cvxpy, statsmodels, scikit-learn) y **z3-solver==4.16.0.0**. Grep de imports en runtime/scripts/tests/engines: **solo z3 se importa**. Es decir, del shortlist del doc 1 para CAU/CTF (DoWhy/pgmpy) y ANA (networkx/VF2), las librerías están declaradas pero con **cero uso** — la inferencia causal viva es la firma declarativa hecha a mano, no descubrimiento causal.

### 5.3 Razonador externo LLM — ✅ REAL, gated, advisory

`ExternalReasonerGate` ([gating.py](../../runtime/reasoning/external_models/gating.py)): llama al LLM solo ante (a) régimen de conflicto causal/contrafactual explícito, (b) conflicto estructural CAU↔CTF, o (c) core riesgoso con historial de correcciones ≥20%. Cliente `llama-cli` (OpenThinker3‑7B), máx. 1 llamada/episodio, opt‑in por `RNFE_CORE_FAMILIES_LLM=1`, y el resultado es **evidencia advisoria que no altera la decisión simbólica** ([core_inference.py:409-496](../../runtime/reasoning/families/core_inference.py#L409-L496)). Diseño sano y alineado con el canon (LLM como módulo, no como núcleo).

### 5.4 Motor de morfismos — ✅ REAL pero solo en laboratorio

`DirectedScenarioMorphism` con scores compuestos (semántico/control/efectos/contrafactual + penalización direccional → clase isomorphic…incompatible) en [runtime/world/morphism_engine.py](../../runtime/world/morphism_engine.py). Lo usan `edge_benchmark.py:24`, `ablation_lab.py:23` y `transition_matrix.py:35` — pero en el episodio vivo `promotion_gate` llama `assess_transfer` **sin morfismo** ([promotion_gate.py:83-87](../../runtime/certification/promotion_gate.py#L83-L87), parámetro default None). El transporte formal entre escenarios existe y no participa de la certificación cotidiana.

---

## §6 — Veredicto: ¿es ya el ser del canon?

**No todavía — y la distancia está medida.** Lo que existe hoy es un **organismo certificador determinista de lazo episódico**: percibe → razona (6 familias reales, una con Z3) → interviene → simula contrafactual → se certifica (posterior bayesiano) → transiciona un estado constitucional con viabilidad y linaje. Eso ya es cibernética genuina y es la parte del canon mejor servida (≈ el esqueleto de Xₜ, C^cont, 𝔠ₜ, S‑I‑E aproximado).

Lo que el canon llama "estar vivo" y "autoevolucionar" está ausente o dormido, en cuatro carencias estructurales:

1. **No se modifica a sí mismo:** ρₜ (SelfModificationPipeline) está completo e inerte; las InheritanceRule nunca se evalúan; el rollback es un booleano sin mecanismo.
2. **No gestiona riesgo de cola:** sin CVaR, sin B_safe formal, sin Ωₜ — los gates deciden con proxies lineales; justo la matemática que f2.1/f2.4 añadieron para que evolucionar sea seguro.
3. **No aprende ni optimiza:** sin θₜ/φₜ, sin recompensa, sin funcional maestro; el scheduler es reglas; H‑Net (la única red real, funcional) es una isla sin entrenamiento ni invocación.
4. **No vive en el tiempo:** sin life‑loop, sin Hctrl dentro del lazo, sin kill‑switch vivo; el organismo solo existe mientras un benchmark lo ejecuta.

Patrón general honesto: **el andamiaje de gobierno (certificar, validar, registrar) está al ~70% del canon; la sustancia que gobierna (aprender, mutar, heredar, medir riesgo profundo) está al ~10–15%.** La buena noticia: varios órganos críticos ya están construidos y solo desconectados (ρₜ, morfismos, H‑Net, MSRC, CVaR de tests), lo que abarata el cierre del gap.

---

## §7 — Roadmap recomendado hacia el ser cibernético autoevolutivo

Orden elegido por dependencias: *primero el gate matemático que hace segura la evolución, luego la evolución misma, luego la economía del razonamiento, luego la telemetría profunda.*

### R1 — Corazón de riesgo del certificado (f2.1/f2.4) — prerequisito de todo
- **CVaR_α[−ΔIoC] empírico**: promover `compute_cvar` del laboratorio (fractal_geometries.py:1163) a `runtime/certification/`, calculado sobre la serie de `ioc_proxy` por run/linaje. Hay **3.841 certificados históricos en PostgreSQL** para calibrar α y umbrales τ con datos reales.
- **B_safe formal**: barrera log φ_bar sobre la telemetría que MSRC ya mide (presión VRAM, temperatura), expuesta como señal continua.
- **𝔠ₜ⁺ completo**: añadir ambos al certificado (campos nuevos en `EpisodeCertificateRecord` + builder), con S‑I‑E 2.0 explícito: ACEPTAR / BUFFER / RECHAZAR.
- *Por qué primero:* la admisibilidad 𝒜(Xₜ) de la auto‑modificación (R2) se define con estas señales; sin ellas ρₜ no tiene gate matemático y "autoevolutivo" sería "auto‑mutante sin frenos".

### R2 — Vida: cablear ρₜ + μₜ + life‑loop — el salto a "ser"
- Invocar `SelfModificationPipeline` desde el camino vivo cuando viabilidad/drift crucen umbral, con la regla S‑I‑E de R1 como gate de aceptación; hacer ejecutable el rollback (snapshot→restore, los snapshots ya existen).
- Activar herencia: evaluar `InheritanceRule` al cerrar episodios/runs, registrar entradas de linaje reales (accepted/rejected/rollback).
- **Life‑loop**: daemon de episodios continuos multi‑régimen (alternando escenarios/perturbaciones) persistiendo trayectoria y certificados a PG — el organismo existe en el tiempo, no por invocación.

### R3 — Scheduler semi‑Markov + familias stub con motores reales
- Recompensa r = ΔIoC − λE·coste − λB·B_safe estimable offline desde los traces ya persistidos; primero como *scoring* de secuencias (bandit/semi‑Markov tabular), no deep RL.
- Implementar IND/PLAN/OPT con los motores ya declarados en requirements (pgmpy para inducción de estructura, networkx para matching/planning ligero; OR‑Tools opcional para OPT) — convertir las deps fantasma en deps reales o quitarlas.

### R4 — Edge 2.1/RQA + Dₜ medida
- RR/DET/dimensión fractal sobre `OrganismTrajectory` (promover el box‑counting de tests a runtime); ρ(J) por diferencias finitas sobre la dinámica del mundo/estado.
- Dₜ con los términos medibles (ΔVRAM⁺, ΔTEMP⁺, bandas RR/DET, entropía de ruta del scheduler) → entra al certificado y a las decisiones de MSRC (que además debe entrar al lazo del episodio como a^ctrl).

### R5 — Integración H‑Net / fractal (largo plazo)
- Conectar `engines/hnet` como módulo de Forma cuando exista un flujo de datos secuencial real que lo justifique (mundos más ricos / percepción); entonces aparecen θₜ/φₜ de verdad, PAI deja de ser vacío y MFE/TFI se vuelven medibles. Antes de eso sería integración decorativa.

---

*Documento generado por auditoría línea‑por‑línea del código vivo (3 pasadas de exploración + verificación por grep/lectura directa). Cualquier discrepancia futura entre este documento y el código se resuelve a favor del código.*
