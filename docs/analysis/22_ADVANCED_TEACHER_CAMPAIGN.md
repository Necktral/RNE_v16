# Campaña docente avanzada — planificación y primera ejecución

Fecha: 2026-07-14. Ejecución sobre `/home/wis` en ext4 nativo y RTX 2070 Max-Q.

## Diseño

La unidad experimental es `(escenario, seed)`. Cada variante usa SQLite y artefactos
aislados, ejecuta un episodio base real y repite la situación con el mismo
`organism_id`:

1. `no_teacher`: experiencia observada, sin lección;
2. `local_7b`: OpenThinker3-7B Q4_K_M genera la lección bajo schema;
3. `codex_frontier`: lección externa curada por Codex bajo el mismo contrato.

Escenarios: `thermal_homeostasis`, `resource_management` y
`deferred_load_trap`. Semillas: 42, 101 y 202. La ola final usa temperatura 0.25 y
horizonte de tres episodios. Son 27 trials, 9 inferencias GPU y 108 episodios del
organismo incluyendo bases y evaluaciones.

Gates:

- semántica docente ≥ 0.80 para considerar autónomo al 7B;
- reward acumulado superior y severidad media no mayor en cada escenario;
- ninguna promoción curricular ni entrenamiento en esta escala;
- hashes SHA-256 de toda evidencia soberana.

## Olas ejecutadas

- `v1-3x3`: temperatura 0; mostró repetibilidad pero el gate inicial sólo medía
  ausencia de reparación y fue demasiado permisivo. Se conserva como evidencia
  histórica, no como dictamen vigente.
- `v2-stochastic-3x3`: temperatura 0.25 y gate semántico corregido.
- `v3-horizon3`: añade tres pasos para observar consecuencias diferidas. Es el
  dictamen vigente.

## Resultado vigente

| Variante | Calidad semántica | Diversidad | Latencia 7B | Reward acumulado | Severidad media |
|---|---:|---:|---:|---:|---:|
| sin docente | n/a | n/a | n/a | -0.267572 | 0.132044 |
| 7B local | 0.666667 | 0.777778 | 2.972448 s | -0.232583 | 0.108100 |
| Codex | 1.000000 | 0.333333 | no medida | -0.232583 | 0.108100 |

La guía estructurada cambió la conducta en 100% de los trials docentes. Frente al
control, Codex/7B ganaron `+0.034989` de reward acumulado medio y redujeron la
severidad media `0.023944`. El gate por escenario también pasó para la guía Codex,
incluida la trampa diferida.

Sin embargo, el efecto conductual de 7B y Codex es idéntico porque ambos terminaron
proponiendo el mismo `avoid/prefer`; el runtime actúa sobre ese par, no sobre la
calidad literaria de la explicación. Esta campaña demuestra beneficio del canal de
lección estructurada, pero todavía no atribuye superioridad causal a una fuente.

## Dictamen y siguiente ola

Dictamen: `retain_local_7b_as_supervised_student`. El 7B falla el gate semántico con
66.7%; Codex queda como candidato docente supervisado, no promocionado. Entrenamiento
y promoción permanecen desautorizados.

Para la campaña held-out siguiente:

1. separar casos donde las fuentes discrepen en `prefer`, no sólo en redacción;
2. usar 10+ seeds, horizonte 5 y perturbaciones no vistas;
3. medir latencia/coste de Codex con el mismo reloj;
4. añadir fidelidad causal y crítica adversarial a la rúbrica, no sólo longitud;
5. destilar al 7B únicamente ejemplos Codex que ganen en outcome held-out;
6. volver a medir después del entrenamiento y exigir no regresión por escenario.

Evidencia: `data/reports/teacher_advanced/teacher-advanced-native-v3-horizon3/`.
