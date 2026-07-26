# Experimentos de capacidad cognitiva MCI

Campaña `mci-capability-v1`, ejecutada con actuación habilitada, almacenamiento
aislado por brazo y semillas deterministas. Los JSON enlazados son la evidencia
canónica de cada corrida.

| Experimento | Resultado | Evidencia |
|---|---|---|
| 1 — Regret pareado | Superado. Diferencia media `0.3065016626902441`; IC bootstrap 95% `[0.143738134038, 0.488905917916]`. | [exp1_regret.json](exp1_regret.json) |
| 2 — Aprendizaje estructural | Parcialmente superado. El brazo autónomo no promovió la precondición; el asistido la promovió en el episodio 29 tras desempatar por menor train MAE. | [exp2_learning.json](exp2_learning.json) |
| 3 — Auto-modelo | Superado. Brier `0.003199857351495`, 200 outcomes puntuados y 200 abstenciones. | [exp3_brier.json](exp3_brier.json) |
| 4 — Cambio de régimen | Superado. Cambio detectado en el episodio 33; promoción `0.105` en 44 y refinamiento `0.13125` en 45, con latencia 15. | [exp4_regime.json](exp4_regime.json) |
| 5 — Ciclo cerrado N4–MCI | Gate técnico disponible. El backend de referencia valida integración, pero el gate científico permanece `not_applicable` hasta usar un artefacto entrenado y certificado. | [exp5_n4_closed_loop.json](exp5_n4_closed_loop.json) |
| 6 — Transferencia inter-escenario | La corrida técnica rápida completó los cuatro brazos. El gate científico no se declara: la adquisición autónoma fuente no produjo todavía una precondición promovida y N4 solo dispone del backend de referencia. | [exp6_transfer.json](exp6_transfer.json) |

## Reproducción

```bash
RNFE_EXPERIMENT_WORK_ROOT=/ruta/aislada \
  .venv/bin/python -m scripts.experiments.run_all --root .
```

Para validar únicamente el cableado:

```bash
RNFE_EXPERIMENT_WORK_ROOT=/ruta/aislada \
  .venv/bin/python -m scripts.experiments.run_all --root . --quick
```

Los resultados negativos se conservan como observaciones del sistema actual. No
se modificaron thresholds, datos ni selección de candidatos después de observar
las campañas.

Los archivos `diagnostic_*.json` documentan las corridas previas a las
correcciones y se conservan como evidencia histórica, no como resultados vigentes.
