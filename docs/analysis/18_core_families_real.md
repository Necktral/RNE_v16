# 18 — Implementación: familias core reales (ABD/ANA/CAU/CTF/PROB)

Cambio aplicado el 2026-06-10: las familias core dejaron de ser stubs y ahora hacen
**inferencia real**. Diseño **híbrido** acordado: núcleo simbólico riguroso determinista +
aumento LLM (OpenThinker3-7B en `/mnt/d`) opcional, gated y opt-in.

## Qué cambió
- **Nuevo:** `runtime/reasoning/families/core_inference.py` — motor de inferencia compartido.
- **Reescrito:** `runtime/reasoning/families/{abd,ana,cau,ctf,prob}/__init__.py` (antes devolvían
  `{"<key>": True}` fijo; ahora delegan en el motor).
- **Intacto:** `ded/` (ya era real con Z3), `policy.py`, `meta_scheduler.py`, contratos, scheduler.

## Inferencia real por familia (núcleo simbólico, determinista, ms)
| Familia | Qué computa ahora (sobre el estado real del episodio) |
|---|---|
| **ABD** | Genera y **rankea hipótesis** intervención→resolución usando `causal_signature.intervention_effects` (dirección/magnitud/rol) + estado de alarma; emite `abd_hypothesis` (top), `abd_hypotheses` (ranking), `abd_top_intervention`. |
| **ANA** | **Mapea** a la memoria recuperada (top por score de overlap) o, sin memoria, al vocabulario de proposiciones de la firma causal; emite `ana_mapping` con `alignment_score`. |
| **CAU** | Infiere el **enlace causal** desde el efecto observado `factual−counterfactual` sobre la variable principal y lo contrasta con la dirección esperada de la firma; emite `cau_link` (helps_goal, direction_match, strength). |
| **CTF** | Usa la transición **contrafactual real** del mundo y verifica si la intervención elegida queda soportada según la dirección de optimización; emite `ctf_checked` (supports_choice, agreement_with_relation_kind). |
| **PROB** | **Calibración bayesiana** real combinando CAU/CTF/DED + belief + incertidumbre, con cota inferior Agresti-Coull; emite `prob_posterior`/`prob_point`/`prob_lcb` (y mantiene `prob_calibrated=True`). |

La firma causal del escenario se obtiene best-effort vía `runtime.world.registry`
(`get_scenario(name).causal_signature`, cacheada). Sin acoplamiento import-time (import perezoso).

## Aumento LLM (gated + opt-in) — el componente "neuronal" desde /mnt/d
- **Apagado por defecto** → comportamiento nominal/benchmarks deterministas **sin cambios**.
- Se activa con **`RNFE_CORE_FAMILIES_LLM=1`** (+ el entorno del razonador externo apuntando al
  modelo de `/mnt/d`: `RNFE_REASONING_GGUF`, `RNFE_LLAMA_CLI_CUDA`/`..._CPU`,
  `RNFE_EXTERNAL_REASONER_BACKEND` — ver `.env.external_reasoner.example` y
  `/mnt/d/rnfe_models/scripts/rnfe_reasoning_models_env.sh`).
- **Gated**: solo llama al LLM cuando hay **conflicto causal/contrafactual o ambigüedad** real
  (reusa `ExternalReasonerGate`), y **a lo sumo UNA vez por episodio** (cacheado en el estado).
- **Advisory**: la augmentación se registra (`core_reasoner_llm`, `<fam>_llm_augmented`) y suma su
  latencia al `cost`, pero **NO altera la decisión simbólica determinista** (el núcleo es
  autoritativo). Degrada con gracia si el modelo no está configurado.
- Tuning: `RNFE_CORE_FAMILIES_LLM_MAX_TOKENS` (def 64), `RNFE_CORE_FAMILIES_LLM_CPU_FALLBACK`.

> Nota de coste: con el LLM activo, cada episodio con conflicto añade **~1 llamada (~60–96 s)**, no
> 5×. El gate `policy.select_sequence` que bloquea la *familia* `ext_open_thinker` en perfiles
> nominales sigue intacto; este aumento es un canal interno distinto, explícitamente opt-in.

## Contrato preservado / compatibilidad
- Las 5 claves (`abd_hypothesis`/`ana_mapping`/`cau_link`/`ctf_checked`/`prob_calibrated`) siguen
  presentes y **truthy** (lo que leen `ext_open_thinker._collect_core_hypotheses` y la validación
  de cierre). Ahora con contenido real (dicts) en vez de `True`.
- Orden/secuencia del cierre sin cambios; `should_early_stop` no depende de la confianza, así que
  las confianzas reales no alteran las secuencias.
- Determinista con el LLM apagado → reproducibilidad de `baseline_fixed` intacta.

## Verificación (2026-06-10)
- Smoke E2E thermal: ABD→`activate_cooling`, CAU `helps_goal=True`, CTF `supports=True`,
  PROB `point≈0.68`. Correcto.
- Suite relevante: **353 tests verdes** (familias, scheduler, closure, miniworlds, organism,
  reality, certification, contracts, integration).
- **0 regresiones introducidas.** Las 2 fallas observadas son **preexistentes** (verificadas con
  `git stash`): `test_dia_adv_fal_guard_contribution` (umbral FAL_GUARD 0.55 vs 0.5 en el test) y
  `test_generates_2x2_matrix` (espera 2 escenarios pero el registro tiene 3: +grid_5x5). Los
  errores de `tests/msrc` son por `pydantic` ausente en el venv (acoplamiento legacy, [08]).

## Estado del hallazgo [07]
El hallazgo "familias core = stubs" de [07_reasoning.md](07_reasoning.md) queda **resuelto** para
ABD/ANA/CAU/CTF/PROB (ahora inferencia real). El cierre triádico ya no es ceremonial: cada familia
aporta cómputo verificable, con DED (Z3) y el razonador externo (gated) como capas adicionales.
