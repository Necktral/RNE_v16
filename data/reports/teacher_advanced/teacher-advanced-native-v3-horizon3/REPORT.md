# Campaña docente avanzada — teacher-advanced-native-v3-horizon3

Trials: 27 · pares escenario/seed: 9.

| Variante | N | cambio conducta | Δ severidad | Δ reward | reward acumulado | severidad media | semántica | sin reparación | diversidad | latencia s | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no_teacher | 9 | 0.0 | 0.0 | 0.0 | -0.267572 | 0.132044 | None | None | None | None | None |
| local_7b | 9 | 1.0 | 0.000733 | 0.016389 | -0.232583 | 0.1081 | 0.666667 | 1.0 | 0.777778 | 2.972448 | 47.1 |
| codex_frontier | 9 | 1.0 | 0.000733 | 0.016389 | -0.232583 | 0.1081 | 1.0 | 1.0 | 0.333333 | None | None |

## Consistencia por escenario

| Escenario | Codex-control reward acumulado | Codex-control severidad media |
|---|---:|---:|
| deferred_load_trap | 0.0058 | -0.001134 |
| resource_management | 0.049583 | -0.035266 |
| thermal_homeostasis | 0.049583 | -0.035433 |

Veredicto: **retain_local_7b_as_supervised_student**.

No se autoriza promoción curricular ni entrenamiento. El efecto causal sólo se considera candidato hasta repetición held-out y control de la reparación semántica.
