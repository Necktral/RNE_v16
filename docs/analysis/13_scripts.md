# 13 — `scripts/` (campañas de benchmark y diagnóstico)

~6.9K LOC, 16 archivos. Harnesses procedurales que ejercitan el pipeline y generan los artefactos
de `data/benchmarks/*` y `data/reports/*`. **Calidad procedural buena**: cada script tiene docstring
de propósito; el grep de patrones de riesgo (rutas absolutas hardcoded, `eval`/`exec`, `shell=True`,
`except:` desnudo) **no encontró nada**.

## Inventario
| Script | LOC | Propósito |
|---|---|---|
| `intelligence_campaign_lib.py` | 1380 | Librería reusable de campañas de ganancia cognitiva |
| `benchmark_arch_validation_campaign.py` | 1304 | Validación arquitectónica 1x1 vs 5x5 |
| `benchmark_external_reasoner_gain.py` | 1271 | Ganancia del razonador externo (**lab-only**) |
| `benchmark_external_reasoner_latency.py` | 579 | Microbenchmark de latencia EXT_OPEN_THINKER (**lab-only**) |
| `inspect_reasoning_wiring.py` | 497 | Diagnóstico E2E del cableado runtime→storage→reality |
| `benchmark_family_profiles.py` | 433 | Perfil de familias × régimen |
| `benchmark_msrc_policy_comparison.py` | 320 | Comparativa de políticas MSRC |
| `run_tests_with_pg.py` | 320 | Levanta Postgres (docker) y corre pytest |
| `check_external_reasoner_replication.py` | 305 | Reproducibilidad del entorno del razonador externo |
| `inspect_family_ecology.py` | 265 | Ecología de familias por perfil/régimen |
| `render_intelligence_verdict.py`, `run_adaptive_v2_intelligence_campaign.py`, `benchmark_cognitive_gain_v2.py`, `benchmark_family_causal_gain.py` | <60 c/u | CLIs finos sobre la librería |
| `validate_core_existence.py`, `__init__.py` | shims | rutas históricas |

## Hallazgos
- **[DISEÑO] Razonador externo aislado por diseño.** `benchmark_external_reasoner_{gain,latency}`
  declaran explícitamente *"No usa ScenarioEpisodeRunner … la decisión nominal del runner ocurre
  antes"*. Confirma el triple gate de [07_reasoning.md](07_reasoning.md): el LLM se mide en
  laboratorio, **nunca** en el flujo nominal.
- **[DISEÑO] Baseline de regresión hardcodeado** (`benchmark_external_reasoner_latency.py:35-46`):
  un dict `BASELINE` con `latency_mean_s=96.115`, `latency_p95_s=98.953`, `corrected_core_failure_rate=0.875`,
  etc. — el "checkpoint de latencia" que da nombre a la rama. Acopla el código al resultado
  experimental; cambiar el checkpoint exige editar el dict a mano.
- **[DISEÑO] `run_tests_with_pg.py`**: helper de CI que arranca Postgres en docker (puertos
  55432/5433/15432), con password de dev hardcodeada `rnfe_local_dev_only` (aceptable, etiquetada
  como local-only) y `check_prerequisites` (docker/pytest/psycopg). `subprocess.run(..., check=False)`
  (no usa shell). Correcto.
- **[DISEÑO] Acoplamiento código↔artefactos**: la admisión del razonador externo
  (`family_profiles`) referencia rutas concretas bajo `data/benchmarks/external_reasoner_gain/…`.

## Veredicto
Aparato experimental **sólido y bien documentado**, sin defectos de corrección detectables. El
punto a recordar es metodológico: los benchmarks del razonador externo son **lab-only** y los
baselines viven hardcodeados en el código (gestionar como datos versionados, no como constantes).
