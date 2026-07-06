# A11 — Agente de Imaginación (foresight multi-horizonte) — Blueprint

**Fecha:** 2026-07-06 · **Estado:** diseño (pre-código) · **Ola:** 2.2 (facultad "imaginar/predecir", hoy ✗)
**Flag:** `RNFE_IMAGINATION_DEEP` · **Familia:** `IMAGINATION` (opcional, shadow-first)

---

## 1. Contexto y por qué

La tabla de facultades de aeon-01 marca **imaginar/predecir = ✗** ([carta de crecimiento](2026-07-05_carta_de_crecimiento_aeon01.md)). Hoy el organismo **reacciona**: las familias de razonamiento (ABD/ANA/CAU/CTF/DED/PROB + overlays) computan sobre el estado *presente*. Ninguna simula sistemáticamente el *futuro* antes de actuar.

A11 le da al ser **previsión**: imaginar hacia adelante varias trayectorias hipotéticas, detectar trampas temporales, y ofrecer esa visión como **evidencia advisory** — expandiendo lo que el ser puede considerar, sin quitarle la decisión.

### Restricción de diseño que viene de datos reales

La campaña graduada `aeon_graded_deep_v1` (2026-07-06, deep ON, 5 regímenes × 8 bloques) dictaminó: **"hubo mejora estructural pero no cognitiva"** — cero regímenes fuertes, cero familias con señal positiva, y recomendación literal: *"podar familias neutrales y volver a medir antes de añadir más complejidad."*

Lectura: en los mundos térmicos **greedy-resolubles** (la acción reactiva de 1 paso ya es casi óptima) la profundidad de razonamiento no paga. **A11 no puede ser "otro planificador"**: PLAN, OPT y EVO_SEARCH ya hacen rollout multi-paso sobre el mismo effect-model ([core_inference.py:826](../../runtime/reasoning/families/core_inference.py#L826), [evo_search:63](../../runtime/reasoning/families/evo_search/__init__.py#L63)). Duplicar eso sería exactamente la complejidad neutral que el veredicto pide evitar.

**A11 solo se justifica como EJE NUEVO**, y solo se prueba donde la previsión paga: escenarios con **trampa temporal** (`causal_counterfactual_conflict` o uno de consecuencia diferida), contra una **baseline podada/limpia**. No en los mundos greedy.

---

## 2. Principio: facultad expansiva + red, no jaula

A11 **añade** una facultad (prever). El "gate" no la limita: es la red que le permite ser **audaz**. Sin red, un mal salto imaginativo que sesgue una decisión colapsa al ser; con red (shadow→advisory→gated + rollback), puede proponer futuros arriesgados porque lo malo se atrapa. La red **sube el techo de la audacia**. Y medir su ganancia no es vigilarla — es **ver crecer al ser** ("crecer se mide, no se declara").

---

## 3. El mecanismo (tres sub-motores, todos deterministas)

**Primitiva compartida** (la misma transición de 1 paso que usan CTF/EVO/PLAN/OPT, byte-idéntica, reproducible):

```python
from runtime.reasoning.families import core_inference as ci
from runtime.reasoning.families import _deep_common as dc

model     = ci._effect_model(state)                          # {intervención: Δ con signo}, determinista
mv        = ci.main_variable(state)
direction = ci.optimization_direction(state, ci.resolve_signature(state))
threshold = dc.safe_dict(state.get("scenario_metadata")).get("alarm_threshold")
x0        = dc.num(dc.safe_dict(state.get("observation")).get(mv))
actions   = sorted(model)                                     # espacio de acción canónico

def step(x, iv):                                             # == evo:64 / plan:826
    return min(1.0, max(0.0, x + model[iv]))

def value(x):                                               # menor = mejor
    return x if direction == "minimize" else 1.0 - x
```

Rollout de H pasos comprometiendo una primera acción `a`, con continuación greedy determinista:

```python
def rollout(a, H):
    x, traj = step(x0, a), []
    for _ in range(H - 1):
        x = step(x, min(actions, key=step_value(x)))         # continuación greedy determinista
        traj.append(x)
    return x, traj      # (terminal, trayectoria)
```

### M1 — Fusión multi-horizonte + detección de miopía  *(la señal estrella)*

Para cada horizonte `h ∈ {1, 5, 20}` elegí la mejor primera acción `a*_h = argmin_a value(rollout(a, h).terminal)`.

- Si **`a*_1 ≠ a*_H`** (la mejor a corto plazo difiere de la mejor a largo): **trampa temporal detectada**. El ser recibe: *"lo que parece mejor ahora NO es lo mejor a largo plazo."* Ninguna familia de 1 horizonte ve esto.
- `imagination_regret` = `value(rollout(a*_1, H)) − value(rollout(a*_H, H))` — cuánto cuesta la miopía.

### M2 — Robustez bajo incertidumbre del effect-model  *(imaginar mundos, no solo acciones)*

Reproducí la selección de `a*_H` bajo variantes **deterministas** del modelo (`0.8×`, `1.0×`, `1.2×` sobre las magnitudes). Si `a*_H` es estable en las 3 → `imagination_robustness` alto; si cambia → `imagination_fragile = True`. Es "imaginar que mi modelo puede estar equivocado" — eje que PLAN no tiene (PLAN confía en el modelo).

### M3 — Trayectoria de riesgo futura de la acción elegida  *(previsión advisory)*

Rollout de la intervención **ya elegida** (`state["intervention"]`) H pasos: ¿cruza `alarm_threshold` dentro de H? ¿en qué paso? Emite `imagination_chosen_breaches_at` (índice o `None`) y la trayectoria como artifact. Le da al ser "presiento que en 5 pasos esto rompe viabilidad" — algo que una familia reactiva no puede ver.

**Coste:** H≤20, |acciones| chico, 3 perturbaciones → milisegundos, CPU, determinista. Cabe trivial en 8 GB. **No usa** rssm_lite2/predictive_coder (torch estocástico) porque romperían la reproducibilidad byte-idéntica.

---

## 4. Contrato de salida

`execute(state) -> dict` (el scheduler lo normaliza vía `normalize_family_result`, [family_result.py:47](../../runtime/reasoning/contracts/family_result.py#L47)).

**OFF** (`not ci.family_deep_enabled("IMAGINATION")`) — byte-idéntico, idle:
```python
{"family": "IMAGINATION", "status": "idle", "state_delta": {}, "confidence": 0.0, "cost": 0.0}
```

**ON** — advisory, claves `imagination_*` en `state_delta`:
```python
{"family": "IMAGINATION", "status": "ok",              # "warn" si fragile o breach inminente
 "state_delta": {
     "imagination_recommended_intervention": a_long,    # a*_H
     "imagination_myopia_detected": a_short != a_long,
     "imagination_short_action": a_short, "imagination_long_action": a_long,
     "imagination_regret": round(regret, 4),
     "imagination_robustness": round(robustness, 4), "imagination_fragile": bool,
     "imagination_chosen_breaches_at": int_or_None,
     "imagination_horizon_values": {1: v1, 5: v5, 20: v20},
 },
 "confidence": round(robustness, 4), "cost": 0.6,
 "recommended_next_family": "PROB",
 "failure_mode": None,                                  # o "imagination_no_model_or_observation"
 "artifacts": {"chosen_trajectory": [...], "per_action_terminals": {...}}}
```

Guardrail idle-con-señal (patrón evo_search): si falta `model`/`observation`, `status:"idle"`, `confidence:0.2`, `failure_mode:"imagination_no_model_or_observation"`.

---

## 5. Superficie de integración (cambios mínimos, exactos)

1. **Paquete nuevo** `runtime/reasoning/families/imagination/__init__.py` — `FAMILY_ID="IMAGINATION"`, `execute(state)`, patrón shadow de [eml_sr](../../runtime/reasoning/families/eml_sr/__init__.py#L68). Reutiliza `ci._effect_model / main_variable / optimization_direction / resolve_signature / _goal_reached` y `dc.safe_dict / num / clamp`.
2. **Flag:** cero cambios en core_inference — `ci.family_deep_enabled("IMAGINATION")` ya lee `RNFE_IMAGINATION_DEEP` (o el maestro `RNFE_REASONING_DEEP`), [core_inference.py:157](../../runtime/reasoning/families/core_inference.py#L157).
3. **Registro:** agregar `"imagination"` a `CONDITIONAL_SHADOW_FAMILIES` ([family_profiles.py:12](../../runtime/reasoning/scheduler_meta/family_profiles.py#L12)) — hogar idiomático de una familia shadow; fluye a `OPTIONAL_FAMILIES`/`TRACKED_OPTIONAL_FAMILIES`. Opcional: perfil aislado `core_plus_imagination` (espejo de `core_plus_ind`) para medirla sola.
4. **Inyección shadow** (opcional, espejo de eml_sr): rama en [meta_scheduler.py:222](../../runtime/reasoning/scheduler_meta/meta_scheduler.py#L222) — `if family == "imagination": state["imagination_mode"] = "shadow" if is_imagination_experimental_enabled() else "disabled"`, con `is_imagination_experimental_enabled()` en policy.py (espejo de [policy.py:1105](../../runtime/reasoning/scheduler_meta/policy.py#L1105)).
5. **Advisory por construcción:** **NO** agregar `EvidenceItem.create(kind="imagination", canonical=True, ...)` en [conjunction/service.py](../../runtime/conjunction/service.py). Sus claves viajan por el trace y el `reasoning_governance.v1` envelope ([governance.py:184](../../runtime/reasoning/scheduler_meta/governance.py#L184)), nunca como directiva vinculante. La decisión de OperationalConjunction sigue derivando de vitals/causal-signature/memory/checkpoint.

---

## 6. Señal de crecimiento (cómo se prueba que hizo al ser más listo)

**Hipótesis:** en escenarios con trampa temporal, el ser que **considera** la recomendación de largo horizonte de A11 interviene mejor que la baseline reactiva.

- **Métricas:** `intervention_precision`, `viability_margin`, ΔIoC*, y # de cruces de alarma — vía el mismo arnés de campaña (`run_adaptive_v2_intelligence_campaign`).
- **Dónde:** régimen `causal_counterfactual_conflict` (y un mundo nuevo de consecuencia diferida, ver §8), **no** los térmicos greedy.
- **Contra qué:** baseline **podada/limpia** (Fase 0). A11 es la primera familia diseñada desde el día 1 para medir ganancia cognitiva real, no para acumularse.
- **Criterio de vida (A8 honestidad):** si A11 no produce ganancia cognitiva medible en el escenario-trampa, **se queda en shadow/lab**. No entra al tronco vivo por intuición.

---

## 7. Plan por fases (cada una con criterio de aceptación)

| Fase | Qué hace | Aceptación |
|---|---|---|
| **0 · Podar baseline** | Identificar familias neutrales de la campaña; fijar baseline limpia | Baseline reproducible con familias no-neutrales; suite verde |
| **1 · Shadow** | A11 emite `imagination_*`; 0 decisiones cambiadas; OFF byte-idéntico | N episodios sin crash; determinista (re-run `a==a`); **0 decisiones modificadas** |
| **2 · Advisory** | La recomendación aparece en governance/trace; un sesgador *puede* mirarla; conjunction intacta | Miopía dispara cuando toca en el escenario-trampa; recomendación correlaciona con mejor outcome |
| **3 · Gated** | La recomendación de largo horizonte puede sesgar la intervención **solo si** robustez alta + checkpoint sano + riesgo bajo + R1 permite | ΔIoC* certificado > baseline en trampa; **sin regresión** en mundos greedy; rollback listo |

---

### Resultado Fase 0 — baseline podada (2026-07-06) ✅

Analizado desde `data/benchmarks/cognitive_gain/aeon_graded_deep_v1_prompt1/` (sin re-correr).

- **Baseline limpia = `core_only`** = `CORE_SEQUENCE` = ABD → ANA → CAU → CTF → DED → PROB.
- **Podadas (sobrecoste neutral en estos regímenes):** HEUR, DIA_ADV, FAL_GUARD (`AUGMENTER_FAMILIES`). En los 5 regímenes: ΔPrecisión = 0.0000, ΔViability = 0.0000, ΔIVC-R negativo (−0.006 a −0.057), 0 regímenes positivos. `core_only` gana como mejor perfil **y** mejor baseline fijo en todos.
- **Alcance honesto:** la neutralidad está probada para los *augmenters* en mundos greedy. Las deliberativas (plan/opt) y shadow (ind/eml_sr) no se aislaron en esta campaña; no están en la baseline nominal de todos modos.
- **Uso:** A11 se mide contra `core_only` en el escenario-trampa. Debe producir ganancia cognitiva que `core_only` no puede — o se queda en shadow.

---

### Resultado Fase 1 — motor de imaginación en shadow (2026-07-06) ✅

Entregado: [`runtime/reasoning/families/imagination/__init__.py`](../../runtime/reasoning/families/imagination/__init__.py) + [`tests/reasoning_stress/test_imagination_a11.py`](../../tests/reasoning_stress/test_imagination_a11.py) — **9 tests verde**. Baseline intacta: no se tocaron perfiles; `reasoning_stress` colecta **303 tests sin error** (294 previos + 9).

**Hallazgo de implementación (honesto, revisa §3):** el effect-model lineal (Δ fijo) es *incapaz* de miopía o consecuencia diferida — con Δ constantes la mejor acción a 1 y a H pasos coincide. Y en `cgwm_min` **enfriar siempre domina**, así que la *miopía* pura tampoco dispara. Lo que **sí** aporta valor no-redundante es la **previsión de consecuencia diferida**: A11 reconstruye un mundo térmico con estado (`cooling_active` persistente + deriva ambiental) y detecta que "desactivar" —que el Δ lineal ve como neutral (Δ=0)— cruza la alarma en N pasos. PLAN/EVO/CTF (lineales, 1-paso) son ciegos a eso.

**Qué expone A11 (siempre advisory):** `imagination_recommended_intervention`, `imagination_chosen_breaches_at`, `imagination_disagrees_with_choice`, trayectoria imaginada. Marcado `imagination_speculative=True` (deriva asumida `_ASSUMED_DRIFT=0.03`), confianza 0.35, nunca canónico/vinculante. Motor puro `imagine()` agnóstico del mundo → testeable con mundos sintéticos y, en Fase 2, con la trampa real.

**Diferido a Fase 2 (necesita mundo-trampa real):**
- **Miopía y robustez** sólo tienen dientes con una acción "buena-ahora-mala-después"; ningún mundo actual la tiene → se construyen sobre el escenario-trampa (§8.1).
- **Calibrar la deriva** contra el escenario real (o extender la firma causal con la deriva declarada) para quitar `speculative`.
- **Registrar** `imagination` en un perfil (`core_plus_imagination`) y medir ganancia cognitiva vs `core_only` en la trampa. Sin ganancia medible → queda en lab.

### Resultado Fase 2a — mundo-trampa construido y verificado (2026-07-06) ✅

Entregado: [`runtime/world/deferred_load_scenario.py`](../../runtime/world/deferred_load_scenario.py) (registrado como `deferred_load_trap`) + [`tests/reasoning_stress/test_deferred_load_trap.py`](../../tests/reasoning_stress/test_deferred_load_trap.py) — **7 tests verde**. Sin regresión: `tests/world` + `tests/miniworlds` + morfismos causales (57 tests) siguen verde.

**La trampa, verificada a nivel de dinámica** (`external_input=0.04`):

| Política | Trayectoria de `load` | Resultado |
|---|---|---|
| `boost_throughput` (reactivo) | `0.67 → 0.72 → 0.85 → 1.0 …` | **cruza alarma en el paso 3** |
| `shed_load` (previsor) | `0.69 → 0.68 → 0.67 → 0.66 …` | seguro indefinidamente |

Clave: en el **paso 1 boost baja más la carga** (0.67 < 0.69) — el effect-model lineal (magnitud correctiva 0.15 > 0.05) lo prefiere y **cae en la trampa**; sólo un lector multi-paso (A11) ve el rebote por deuda. Es el terreno donde la previsión puede ganarle a `core_only` de forma medible.

**Falta (Fase 2b):**
- Enseñar a A11 la dinámica de deuda de este mundo (world-model keyed por escenario, o extender la firma con el efecto diferido declarado) para que su rollout capture el rebote.
- Registrar `imagination` en un perfil `core_plus_imagination`.
- Medir `core_only` vs `core+imagination` en `deferred_load_trap` con el arnés de campaña: ¿menos breaches / mejor viabilidad / ΔIoC*? Con ganancia → gated; sin ganancia → lab.

### Resultado Fase 2b — A11 imagina la deuda, y GANANCIA MEDIDA (2026-07-06) ✅

Entregado:
- **A11 imagina la deuda**: world-model *keyed por escenario* en [`imagination/__init__.py`](../../runtime/reasoning/families/imagination/__init__.py) (`thermal_world` + `deferred_load_world`), política de rollout **"repetir la acción"** (correcta para ambos mundos). Núcleo `imagine()` puro reutilizado por `execute()` y por la medición.
- **Perfil registrado**: `core_plus_imagination` (lab_only) en [`family_profiles.py`](../../runtime/reasoning/scheduler_meta/family_profiles.py).
- **Medición** en [`test_imagination_gain.py`](../../tests/reasoning_stress/test_imagination_gain.py).

**Ganancia medida** (política reactiva vs previsora sobre `deferred_load_trap`, 25 pasos):

| Política | breaches de alarma | carga final | carga media |
|---|---:|---:|---:|
| REACTIVA (greedy lineal → `boost_throughput`) | **23 / 25** | 1.00 | 0.97 |
| PREVISORA (A11 → `shed_load`) | **0 / 25** | 0.45 | 0.57 |

La previsión **elimina los breaches** (23 → 0) y mantiene la carga en la zona segura. Es la evidencia que justifica mover A11 a **ejecución gated** (Fase 3): cuando A11 predice breach diferido y su recomendación discrepa de la elección reactiva, sesgar la decisión — bajo checkpoint sano + riesgo bajo + R1.

**Nota honesta:** hoy A11 es advisory — no cambia la decisión viva; la medición cuantifica la ganancia *si se sigue* su recomendación. El world-model de deuda usa constantes declaradas (`_BOOST_DEBT/_SHED_DEBT`) marcadas `speculative`; en Fase 3 se aprenden/calibran de la experiencia. Un fallo **preexistente** de `test_external_reasoner_profiles.py` (asumía que todo perfil lab-only usa el razonador externo; roto desde que la rama añadió `full_family_deep_v1`) se corrigió a su intención real.

**Verificación:** 9 (A11) + 7 (trampa) + 3 (ganancia) tests verde; regression + perfiles/scheduler verde; baseline `reasoning_stress` sin regresión.

### Resultado Fase 3 — compuerta gated + corrida física (2026-07-06) ✅

Entregado:
- **Compuerta gated** `imagination.gate()` ([`imagination/__init__.py`](../../runtime/reasoning/families/imagination/__init__.py)): abre y sesga la intervención **sólo si** A11 activo + discrepa + predice breach diferido + **checkpoint sano** + **riesgo < techo (0.80)**. Si no, advisory. 6 tests unitarios.
- **Ejecutador de corrida física** [`scripts/run_imagination_gated_trap.py`](../../scripts/run_imagination_gated_trap.py): corre episodios REALES de `deferred_load_trap` a través del A11 real (`execute`) + la compuerta, en 4 modos.

**Corrida física** (4 episodios × 25 pasos, determinista):

| modo | breaches | overrides | carga media |
|---|---:|---:|---:|
| baseline (gate cerrada, advisory) | 90/100 | 0 | 0.97 |
| **gated + checkpoint sano + riesgo bajo** | **0/100** | 100 | 0.55 |
| gated pero sin checkpoint sano | 90/100 | 0 | 0.97 |
| gated pero riesgo alto | 90/100 | 0 | 0.97 |

La compuerta **elimina los breaches (90 → 0)** cuando abre, y **se niega correctamente** (0 overrides → cae en la trampa) sin checkpoint sano o con riesgo alto. Es "red, no jaula": la previsión sesga los 100 pasos *porque* hay estado restaurable que la respalda. Reporte en `data/reports/imagination_gated_trap/`.

**Pendiente real (Ola 3.3, no en este arco):** integrar la compuerta en el `LifeKernel/ScenarioEpisodeRunner` vivo (leer checkpoint/riesgo de las vitales reales en lugar de parámetros de corrida) y aprender las constantes de deuda (`_BOOST_DEBT/_SHED_DEBT`) de la experiencia en vez de declararlas.

---

## 8. Decisiones abiertas (para vos)

1. **Escenario-trampa de prueba:** ¿reusamos `causal_counterfactual_conflict` (ya existe) o construimos un mundo nuevo de **consecuencia diferida** (una acción que ayuda en t pero rompe en t+k) que aísle mejor el valor de la previsión? *(Recomiendo empezar con el existente y añadir el nuevo en Fase 2.)*
2. **Horizontes:** ¿`{1,5,20}` como en tu doc, o calibrarlos al largo típico de episodio (`max_steps=50`)?
3. **Fase 0 (poda):** ¿la corro ya para dejar la baseline limpia antes de escribir el código de A11? *(Recomiendo sí — honra el veredicto y hace medible a A11.)*

---

## Anti-objetivos (qué A11 NO debe ser)

- ❌ Un duplicado de PLAN/OPT/EVO (planificación de un horizonte). Su valor es el **desacuerdo multi-horizonte + robustez + previsión**, no "encontrar el plan".
- ❌ Torch estocástico (rssm/predictive_coder) — rompe la reproducibilidad.
- ❌ `EvidenceItem` canónico / decisión vinculante — es advisory por construcción.
- ❌ Complejidad neutral — si no da ganancia medible en la trampa, no entra (honra el veredicto de la campaña).
