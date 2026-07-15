# Campaña docente avanzada — teacher-advanced-native-v2-stochastic-3x3

Trials: 27 · pares escenario/seed: 9.

| Variante | N | cambio conducta | Δ severidad | Δ reward | semántica | sin reparación | diversidad | latencia s | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no_teacher | 9 | 0.0 | 0.0 | 0.0 | None | None | None | None | None |
| local_7b | 9 | 1.0 | 0.000733 | 0.016389 | 0.666667 | 1.0 | 0.777778 | 3.165111 | 47.522222 |
| codex_frontier | 9 | 1.0 | 0.000733 | 0.016389 | 1.0 | 1.0 | 0.333333 | None | None |

Veredicto: **retain_local_7b_as_supervised_student**.

No se autoriza promoción curricular ni entrenamiento. El efecto causal sólo se considera candidato hasta repetición held-out y control de la reparación semántica.
