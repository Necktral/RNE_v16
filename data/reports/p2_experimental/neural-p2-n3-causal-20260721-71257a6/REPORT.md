# P2 — influencia causal de N3 sobre calidad de decisión

## 1. Identidad y prerregistro

Campaña `neural-p2-n3-causal-20260721-71257a6`, basada en el cierre P1 `71257a66cde77f9b4bfb97d4b1d2df9898e8cb52`. El intento desde `ae1f734f...` fue invalidado antes de persistir recibos. La corrida confirmatoria válida se ejecutó desde el prerregistro v2 `47cd37fe36a5728140b0dbc2f51aff6cad922095`.

## 2. Pregunta científica

¿Permutar el mismo conjunto de memorias con N3 reference o N3 trained reduce el regret de la decisión frente al orden canónico?

## 3. Diseño experimental

Tres brazos pareados: canonical, N3 reference y N3 trained. N3 sólo permutó seis candidatos inmutables. IND emitió la recomendación y su hash se selló antes de enumerar outcomes. El risk head trained fue sólo advisory.

## 4. Escenarios y seeds

Se incluyeron los cuatro escenarios registrados: `thermal_homeostasis`, `resource_management`, `grid_thermal_5x5` y `deferred_load_trap`. No hubo exclusiones. Se usaron las 12 seeds SHA-256 prerregistradas y ocho episodios por escenario.

## 5. Contrato de aislamiento

La paridad de estado preacción, conjunto y cantidad de candidatos, seeds y escenarios fue 100%. No hubo escrituras compartidas, entrenamiento, razonador externo, mutaciones del pool ni autoridad live. La primera divergencia permitida fue `memory_reranking`.

## 6. Utilidad y regret

La utilidad procede de `runtime.world.intervention_override.outcome_effectiveness`, aplicada al resultado de `scenario.simulate_counterfactual`. El regret es la diferencia entre la mejor utilidad enumerada y la utilidad de la intervención sellada.

## 7. Resultados reference vs canonical

Gain medio `0.0010416667`; IC95% bootstrap `[-0.0008333333, 0.0025]`; randomización exacta `p=0.369140625`; seeds positivas/cero/negativas `8/2/2`. No pasa el endpoint primario. Regret medio: canonical `0.031875`, reference `0.0308333333`. Optimal-action rate: `0.46875` frente a `0.4817708333`.

## 8. Resultados trained vs canonical

Gain medio `0`; IC95% `[0, 0]`; `p=1`; seeds `0/12/0`. No pasa. Ambos brazos tienen regret medio `0.031875` y optimal-action rate `0.46875`.

## 9. Resultados trained vs reference

Contraste medio `-0.0010416667`; IC95% `[-0.0025, 0.0008333333]`; `p=0.369140625`; seeds `2/2/8`. No hay backend preferido.

## 10. Endpoints secundarios

Reference cambió top-1 en `81.7708%` y acción en `5.46875%`: 13 decisiones mejoraron, 8 empeoraron y 363 quedaron iguales. Trained cambió top-1 en 100%, pero no cambió ninguna acción ni regret. La ganancia de ranking no se transfirió a ganancia decisional. No se observó ganancia operacional. Los diagnósticos de riesgo se conservaron como advisory y no reemplazan el endpoint primario.

## 11. Integridad causal

Todas las paridades prerregistradas son `1.0`; leakage, contaminación cross-arm, escrituras compartidas, training calls, external-reasoner calls y acciones no autorizadas son cero. Los 1.152 recibos están completos.

## 12. Seguridad y cierre

Safety violations: cero. Closure y certification rate registrados por el harness: `1.0` en los tres brazos, sin regresión frente al canonical. No se concedió autoridad live, staging, promotion ni merge.

## 13. Coste computacional

La campaña utilizó CPU; GPU no fue requerida. El artefacto trained tiene 2.631 parámetros. La instrumentación por decisión no midió reloj de pared granular, por lo que los campos `latency_ms=0` significan no medido y no coste cero.

## 14. Limitaciones

- El banco contiene cuatro escenarios pequeños y dos intervenciones por escenario; no prueba generalización sistémica.
- El pool experimental es controlado y balanceado para aislar permutación; no representa toda la distribución histórica de MFM.
- La decisión usa el seam determinista IND, no el conjunto completo de políticas live.
- Ocho episodios por escenario limitan la evolución temporal.
- Latencia granular, nDCG, MRR y calibración de riesgo no fueron reestimadas como inferencia primaria.
- Closure y certificación describen el cierre del sandbox, no autoridad operacional end-to-end.

## 15. Veredicto

`p2_n3_signal_not_transferred_to_decision`, con `preferred_backend=none`. N3 reference mostró una señal pequeña pero estadísticamente insuficiente; N3 trained fue decisionalmente nulo frente a canonical.

## 16. Consecuencia para P3

`N3_DECISIONAL_GAIN=NOT_DEMONSTRATED`, `P3_DESIGN_AUTHORIZED=false`, `LIVE_AUTHORITY=false`, `MAIN_MERGE_AUTHORIZED=false`. P2 no autoriza P3 automáticamente ni modifica el cierre P1.
