# Campaña docente avanzada — teacher-heldout-native-v1-10x2-h5

Trials: 180 · pares escenario/seed: 60.

| Variante | N | cambio conducta | Δ severidad | Δ reward | reward acumulado | severidad media | semántica | sin reparación | diversidad | latencia s | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| codex_frontier | 60 | 1.0 | 0.000733 | 0.008194 | -0.417022 | 0.120413 | 1.0 | 1.0 | 0.05 | None | None |
| local_7b | 60 | 1.0 | 0.000733 | 0.008194 | -0.417532 | 0.120445 | 0.8 | 1.0 | 0.433333 | 3.434785 | 45.223333 |
| no_teacher | 60 | 0.0 | 0.0 | 0.0 | -0.431324 | 0.12836 | None | None | None | None | None |

## Consistencia por escenario

| Escenario | Codex-control reward acumulado | Codex-control severidad media |
|---|---:|---:|
| deferred_load_trap | 0.0002 | -0.00227 |
| resource_management | 0.021166 | -0.01005 |
| thermal_homeostasis | 0.021542 | -0.01152 |

Veredicto: **retain_local_7b_as_supervised_student**.

No se autoriza promoción curricular ni entrenamiento. El efecto causal sólo se considera candidato hasta repetición held-out y control de la reparación semántica.
