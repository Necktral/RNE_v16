# P2-v2 — transferencia causal N3 mediante retrieval real

## Estado

- Base P1: `71257a66cde77f9b4bfb97d4b1d2df9898e8cb52`.
- Prerregistro vigente: `e6c04317d18695937a0f62d3bc8093c120a76388`.
- Ejecución: `55f716e24804ac4ae58f656af8f8d0805b8aa5df`.
- Piloto P2-v1 preservado e invalidado como evidencia confirmatoria.
- Dos ejecuciones de desarrollo fueron invalidadas y no reutilizadas; sus causas están
  registradas en `invalidated-executions.json`.

## Diseño ejecutado

Se ejecutaron 12 semillas, cuatro escenarios y 16 episodios por escenario con tres
brazos, totalizando 2.304 recibos. Cada unidad persistió un banco MFM aislado y
contrabalanceado de 12 memorias. `MemoryRetrieval.retrieve` produjo una sola vez el
pool común. Canonical conservó ese orden; N3-reference y N3-trained solo pudieron
permutarlo. IND recibió los cuatro elementos expuestos y el oracle se abrió después
del sello de decisión.

El brazo trained usó el artefacto congelado `rnfe-mamba2-temporal-lab-v1` en CPU. No
hubo entrenamiento, GPU, razonador externo, escritura al organismo vivo ni autoridad.

## Integridad

- Retrieval real y discriminativo: 100% de unidades con scores no constantes.
- Banco intervención × escala × relación balanceado: 100%.
- Paridad de estado preacción real: 100%.
- Paridad del pool raw y tres brazos completos: 100%.
- Sello antes de oracle: 100%.
- Control positivo de sensibilidad IND: aprobado en los cuatro escenarios.
- Intervenciones no autorizadas, llamadas externas y entrenamiento: cero.
- Auditoría repetida: byte-idéntica.

## Resultados

### Reference frente a canonical

- Ganancia media de regret: `0.0017447917`.
- IC95 bootstrap: `[-0.0000260417, 0.0034375]`.
- p exacta sign-flip: `0.099609375`.
- Semillas positivas/cero/negativas: `8/1/3`.
- Gate de ganancia: no superado.

### Trained frente a canonical

- Ganancia media de regret: `0.0016666667`.
- IC95 bootstrap: `[-0.00046875, 0.0038541667]`.
- p exacta sign-flip: `0.1748046875`.
- Semillas positivas/cero/negativas: `8/0/4`.
- Gate de ganancia: no superado.

### Trained frente a reference

- Contraste medio: `-0.000078125`.
- IC95 bootstrap: `[-0.0007291667, 0.0005729167]`.
- p exacta sign-flip: `0.9375`.
- Ningún backend puede declararse preferido.

## Manipulación y veredicto

La tasa de cambio del conjunto expuesto fue 0% tanto para reference como para
trained. Aunque existieron diferencias numéricas de regret en la matriz, el
tratamiento prerregistrado no fue entregado. Por ello no corresponde concluir
ganancia ni ausencia de ganancia decisional.

Veredicto: `p2_v2_treatment_not_delivered`.

## Limitaciones

Los multiplicadores conservadores definidos por P1 no superaron las separaciones del
score canonical en el corte top-4 de este banco. Modificarlos después de observar los
outcomes sería tuning retrospectivo y está prohibido. Una futura campaña requiere
nuevo diseño y revisión humana; esta evidencia no autoriza P3.

## Autoridad

```text
N3_DECISIONAL_GAIN=NOT_EVALUABLE_TREATMENT_NOT_DELIVERED
P3_DESIGN_AUTHORIZED=false
LIVE_AUTHORITY=false
STAGING_AUTHORIZED=false
PROMOTION_AUTHORIZED=false
MAIN_MERGE_AUTHORIZED=false
```

P2-v2 grants no live authority.

P3 remains blocked pending human review of valid P2-v2 evidence.
