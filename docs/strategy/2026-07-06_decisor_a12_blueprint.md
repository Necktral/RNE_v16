# A12 — Decisor lógico-probabilístico (no-monotonía + Bayes-factor + ACT) — Blueprint

**Fecha:** 2026-07-06 · **Estado:** diseño (pre-código) · **Flag:** `RNFE_A12_DEEP`
**Familia:** `A12` (overlay tardío, shadow-first) · **Compone con:** [A11 imaginación](2026-07-06_imagination_a11_blueprint.md)

---

## 1. Contexto y por qué

El ser hoy **elige reactivo** y las familias son advisory. Falta un **decisor** que integre toda la traza de razonamiento en una decisión coherente con tres capacidades que no existen en el repo (verificado: cero código de Bayes-factor / SPRT / no-monotonía):

- **No-monotonía / defeasible**: sostener la elección por defecto como conclusión tentativa y **retractarla** cuando evidencia posterior la derrota.
- **Adopción por Bayes-factor**: cambiar a una alternativa **sólo si** el peso de la evidencia lo justifica (no por una sola señal).
- **ACT (commit/abstain)**: comprometer la decisión sólo con confianza suficiente; si no, **abstenerse** honestamente.

**Composición con A11:** A12 es una familia *tardía* que lee la traza acumulada — incluida la previsión de A11 (`imagination_chosen_breaches_at`, `imagination_recommended_intervention`). **A11 imagina el futuro; A12 decide con lógica sobre esa imaginación + el resto de la evidencia.**

---

## 2. Entradas (claves reales de la traza acumulada)

A12 corre después de `prob` y lee de `state` (todo mergeado por el scheduler):

- **Default (reactivo):** `state["intervention"]`.
- **Derrotadores del default:**
  - `imagination_chosen_breaches_at is not None` (A11: breach diferido) — el más fuerte.
  - `ctf_checked.supports_choice is False` / `agreement_with_relation_kind is False`.
  - `cau_link.helps_goal is False`.
  - `ded_status == "unsat"` (contradicción deductiva).
- **Testigos (recomiendan intervención):** `opt_intervention`, `plan_first_action`, `ind_best_intervention`, `abd_top_intervention`, `heur_recommended_intervention`, `imagination_recommended_intervention`.
- **Confianza:** `prob_lcb` (Agresti-Coull LCB, conservador), `prob_point`.

Semillas reutilizables: `ci._beta_lcb` (LCB), y el patrón de actuación gated de `intervention_override.py` / `imagination.gate()`.

---

## 3. Mecanismo (determinista)

### 3a. Retracción defeasible
El default es la conclusión tentativa. Se cuentan los **derrotadores** presentes. Si **cero** → el default se sostiene (decisión = default). Si **≥1** → el default queda **derrotado (retractado)** y A12 busca la mejor alternativa.

### 3b. Adopción por Bayes-factor
Para la alternativa candidata `c` (la de mayor consenso entre testigos ≠ default), se acumula el peso de evidencia como producto de *likelihood ratios* (log-aditivos), interpretables:

```
log BF(c vs default) = Σ ln(LR_i)
  derrotadores del default:  imagination_breach 4.0 · ctf_disagree 2.5 · cau_not_help 2.5 · ded_unsat 2.0
  testigos que apoyan c:     por testigo 1.8   (A11 pesa más si además predice breach del default)
  modulación:                escalado por prob_point (traza poco confiable ⇒ BF más débil)
```

Se adopta `c` sólo si `BF ≥ τ_bf` (p.ej. 3.0 "sustancial" / 10 "fuerte", escala de Jeffreys). Se reporta `a12_bayes_factor` / `a12_log_bf`.

### 3c. ACT (commit/abstain)
Aunque el default esté derrotado, A12 **sólo compromete** un cambio si hay confianza:
`act = "commit"` si `prob_lcb ≥ lcb_floor` (p.ej. 0.5) y `#evidencia ≥ mín`; si no, `act = "abstain"` (mantiene el default, marca `a12_default_defeated=True` — honesto: "parece mal pero no estoy seguro de cambiar").

### Decisión final
| default derrotado | ∃ c con BF≥τ | act | → decisión |
|---|---|---|---|
| no | — | commit | default (se sostiene) |
| sí | sí | commit | **c (adopta)** |
| sí | no | commit | default (nada supera el umbral) |
| sí | sí | abstain | default (no confiado) + `default_defeated` |

---

## 4. Contrato de salida (`state_delta`)

OFF (`not ci.family_deep_enabled("A12")`): idle byte-idéntico.
ON (advisory):
```
a12_decision                 # intervención elegida (default o alternativa adoptada)
a12_default_defeated: bool
a12_defeaters: [str]
a12_adopted_alternative: bool
a12_bayes_factor / a12_log_bf
a12_act: "commit" | "abstain"
a12_confidence               # de prob_lcb
a12_witnesses: {familia: intervención}
```
`recommended_next_family=None` (es tardía). Núcleo puro `decide(trace)` testeable con trazas sintéticas.

---

## 5. Integración (shadow → advisory → gated)

- **Fase 1 (shadow):** familia `a12` nueva; OFF byte-idéntico; ON emite `a12_*` (advisory, no decide). Tests: OFF idéntico; retracción/BF/ACT sobre trazas sintéticas.
- **Fase 2 (medir):** perfil `core_plus_a12` (y `core_plus_imagination_a12` para la composición). Medir en `deferred_load_trap`: ¿A12 (leyendo la previsión de A11) retracta `boost` y adopta `shed` por BF, evitando breaches? vs `core_only`.
- **Fase 3 (gated):** exponer `a12_decision` a `intervention_override.py` (añadir a `_CONCRETE_RECOMMENDERS`) o un `a12.gate()` espejo del de A11 — gated por `RNFE_REASONING_ACTUATES` + checkpoint sano + riesgo bajo.

**Señal de crecimiento:** en el escenario-trampa (y `causal_counterfactual_conflict`), A12 debe tomar decisiones mejores que el reactivo — con la disciplina de **abstenerse** cuando no está seguro (que es, en sí, más inteligencia, no menos).

---

---

### Resultado Fase 1 + composición — 2026-07-06 ✅

Entregado: [`runtime/reasoning/families/a12/__init__.py`](../../runtime/reasoning/families/a12/__init__.py) (núcleo puro `decide()` + `execute()`), perfiles `core_plus_a12` y `core_plus_imagination_a12`, y [`tests/reasoning_stress/test_a12_decisor.py`](../../tests/reasoning_stress/test_a12_decisor.py) — **9 tests verde**. OFF byte-idéntico; perfiles nominales intactos.

**No-monotonía demostrada:** con `opt/plan/ind` votando `boost` (consenso lineal = default) pero A11 prediciendo breach del default y recomendando `shed`, A12 **retracta boost y adopta shed** por Bayes-factor (BF ≥ 3). También **se abstiene** honestamente con baja confianza / evidencia débil.

**Composición A11→A12 end-to-end** (escenario real → A11 real → A12 real, 25 pasos): A12 adopta `shed` leyendo la previsión de A11 y **evita la trampa (0 breaches)** — la misma ganancia que A11, ahora tomada como *decisión lógica* integrando toda la evidencia, no como una regla directa.

### Resultado Fase 3 — cableado a la decisión VIVA (2026-07-06) ✅

Entregado:
- **Override de previsión** `evaluate_foresight_override()` en [`intervention_override.py`](../../runtime/world/intervention_override.py): guard de **horizonte** (no de un paso — el guard miope vetaría la acción previsora). Dispara si A12 adoptó una alternativa y A11 certificó el breach diferido del greedy.
- **Cableado en el runner** [`scenario_runner._maybe_override_intervention`](../../runtime/world/scenario_runner.py): corre PRIMERO (antes del override greedy), gated por `RNFE_REASONING_ACTUATES` (sombra OFF ⇒ byte-idéntico). Como el scheduler puede ejecutar A12 antes que otras familias, **A12 se recomputa sobre el estado final completo** en el punto de actuación (robusto a cualquier orden).
- **ACT extendido**: A12 hace *commit* también cuando A11 **certifica por horizonte** (predijo el breach y recomienda la candidata) — un rollout multi-paso determinista, no una conjetura probabilística.

**Cableado vivo, demostrado end-to-end** (`ScenarioEpisodeRunner` real sobre `deferred_load_trap`, perfil `core_plus_imagination_a12`, flags ON):
```
override: fired=True  driver=a12  boost_throughput → shed_load  guard=foresight_horizon
intervención aplicada: shed_load
```
El greedy elige `boost` (trampa); A11 predice el breach; A12 retracta y adopta `shed` por Bayes-factor; el override lo aplica **en vida**. Con actuación OFF (sombra) el episodio aplica el greedy (`boost`) — byte-idéntico. Tests: cableado (13) + episodio vivo (2) verde; override greedy existente intacto.

**Nota:** la traza advisory `a12_*` puede quedar stale si el scheduler corre A12 temprano; la ACTUACIÓN usa el recómputo sobre el estado final. La protección R1 (checkpoint sano / riesgo) la aporta el `OperationalConjunction` que envuelve el life-step.

---

## Anti-objetivos
- ❌ Decidir por una sola señal (por eso Bayes-factor con umbral).
- ❌ Cambiar siempre que el default esté derrotado (por eso ACT/abstain).
- ❌ Vinculante sin gate (advisory hasta Fase 3, bajo `RNFE_REASONING_ACTUATES`).
- ❌ Complejidad neutral — si no mejora la decisión medida, queda en lab.
