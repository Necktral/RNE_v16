# 15 — `tests/` (suite de pruebas)

~25.8K LOC, 121 archivos, **809 funciones `test_*`, ~2138 asserts**. Buena higiene general.

## Configuración
- `pytest.ini` + `conftest.py`: markers `requires_torch/postgres/cuda/extended_bench` con
  `pytest_runtest_setup` que **salta con gracia** cuando falta la dependencia o el flag de entorno
  (`RNFE_RUN_PG_TESTS`, `RNFE_POSTGRES_DSN`, `RNFE_RUN_EXTENDED_BENCH`). CI-friendly: la suite
  corre en CPU sin GPU/Postgres saltando lo no disponible. **No hay `xfail`** (ningún test marcado
  como fallo conocido).

## Distribución por área
| Área | archivos | LOC | Cobertura |
|---|---|---|---|
| `reasoning_stress` | 14 | 9942 | **40%** — atlas fractal/geometría/box-counting/avalanchas |
| `regression` | 31 | 3517 | meta_scheduler policy, ext_open_thinker, closure, ded_z3, storage |
| `benchmarks` | 8 | 4054 | runners de benchmark + consistencia de métricas |
| `organism` | 13 | 2039 | trajectory/constitution/risk/T5 |
| `reality` | 10 | 1627 | closure/continuity/transición/analógico |
| `world` | 3 | 1051 | escenarios/grid |
| `msrc` | 7 | 732 | políticas de escala |
| `contracts` | 6 | 551 | validación de los JSON schemas |
| `certification` | 3 | 593 | promotion gate |
| `integration` | 5 | 1021 | E2E wiring, external reasoner slow |
| `miniworlds` | 5 | 452 | mundos mínimos |
| `comparison` | 1 | 168 | comparativas |

## Hallazgos
### [DISEÑO] `reasoning_stress` (40% del LOC) son experimentos auto-contenidos, no tests de unidad
Los archivos **definen su propia maquinaria** (`FractalAtlasMetrics`, `build_fractal_atlas`,
`_analyze_fractal_dimensions`, `_analyze_temporal_cascades`, `_analyze_avalanche_statistics`,
módulos `fractal_utils`/`fractal_geometries`) y ejercitan masivamente
`runtime.reasoning.scheduler_meta.policy`/`budgeting`/`context_features`, asertando **propiedades
geométricas/estadísticas** de las decisiones del scheduler a través del espacio de features. Son
**artefactos de investigación** embebidos como tests (marcados `@pytest.mark.slow`). Útiles como
caracterización, pero no son verificación de corrección clásica del organismo.

### [DISEÑO] Acoplamiento de un test a un `.md` (test_metrics_consistency.py:128)
`pytest.skip("PHASE1_COMPLETE.md not found")` — un test depende de la existencia de un doc (que
además está desfasado, ver mandato). Mezcla verificación con documentación.

### Cobertura del pipeline vivo — buena
`contracts` valida los JSON schemas; `regression` cubre el meta-scheduler (policy units, family
profiles, closure profiles), el adaptador `ext_open_thinker`, el motor `ded_z3` y storage
SQLite/Postgres/hybrid; `organism`/`reality`/`msrc`/`certification` cubren el chain T5 y la
certificación; `integration` prueba el wiring E2E (`test_reasoning_wiring_e2e`,
`test_external_reasoner_slow`).

### Observación sobre la validez
Dado que las familias core son **stubs** ([07_reasoning.md](07_reasoning.md)), muchos tests de
"cierre" verifican que la **secuencia** se ejecuta/valida, no que el razonamiento subyacente sea
correcto (excepto `ded_z3`, contratos y el razonador externo). El número alto de asserts no implica
verificación cognitiva profunda; verifica el **andamiaje** (secuencias, contratos, persistencia,
gating).

## Veredicto
Suite **amplia, ordenada y con buena higiene de CI** (skips por dependencia, sin xfail). El 40% es
investigación fractal sobre el scheduler más que verificación de unidad. La cobertura del andamiaje
(contratos, scheduler, storage, certificación, T5) es sólida; la verificación del *contenido*
cognitivo está limitada por los stubs de familias.
