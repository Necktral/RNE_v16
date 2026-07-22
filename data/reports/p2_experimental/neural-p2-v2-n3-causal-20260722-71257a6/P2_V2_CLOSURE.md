# P2-v2 — cierre acotado

## 1. Identidad

Rama `codex/p2-n3-causal-decision-v2`, base P1 `71257a66`, prerregistro
`e6c04317`, ejecución `55f716e` y evidencia congelada `edb79dbc`.

## 2. Evidencia congelada

La auditoría consumió los 2.304 recibos existentes. No se ejecutaron campaña,
retrieval, escenarios, directivas N3 ni backends. Ningún outcome histórico cambió.

## 3. Alcance reconstruible

Se reconstruyeron 768 unidades completas con tres brazos, hashes de estado, pool raw,
órdenes, secuencias y membresías top-4, acciones, utilidades y regret.

## 4. Información no reconstruible

Los recibos no contienen señales micro/meso/macro observadas, multiplicadores ni
scores ajustados. La geometría numérica exacta se conserva como
`NOT_RECONSTRUCTIBLE_FROM_FROZEN_EVIDENCE`; no fue inferida desde el orden.

## 5. Integridad

Pairing, paridad de estado, coincidencia snapshot/estado, paridad del pool,
conservación de candidatos, orden canonical, sello del oracle y regret recomputado:
100% válidos. No hubo autoridad, entrenamiento ni razonador externo.

## 6. Tratamiento ordinal

Reference cambió el orden completo en 713/768 unidades (92.84%), top-1 en 457
(59.51%) y la secuencia top-4 en 579 (75.39%). Trained cambió el orden completo en
766/768 (99.74%), top-1 en 511 (66.54%) y la secuencia top-4 en 675 (87.89%). El
tratamiento ordinal fue observado en ambos brazos.

## 7. Tratamiento de membresía

Ningún brazo cambió la membresía top-4: 0/768, tasa 0%. El tratamiento
confirmatorio prerregistrado no fue entregado.

## 8. Cambios de acción

Reference cambió la acción en 56 unidades (7.29%) y trained en 64 (8.33%), siempre
sin cambio de membresía. Esto demuestra que IND fue sensible al orden de las mismas
cuatro memorias; no reemplaza el gate confirmatorio de membresía.

## 9. Regret exploratorio

Los cambios de regret coinciden con los cambios de acción: 56 unidades reference y
64 trained. La ganancia media descriptiva en esas unidades fue 0.02393 y 0.02000,
respectivamente. Este análisis no fue prerregistrado, no tiene inferencia adicional y
carece de autoridad confirmatoria.

## 10. Limitación de instrumentación

No pueden calcularse scores ajustados ni cruces matemáticos de frontera sin
re-ejecutar backends o inventar valores. Ambos métodos están prohibidos.

## 11. Contrato futuro

`P2_NEXT_OBSERVABILITY_CONTRACT.json` exige persistir señales, multiplicadores,
scores ajustados, transiciones de rango, frontera y hashes suficientes para una
auditoría futura sin re-ejecutar modelos. El contrato no autoriza otro experimento.

## 12. Tests

Suite focalizada: 89 passed. Suite completa por 19 shards: 1.784 passed, 22 skipped,
32 xfailed, un XPASS histórico conocido, seis warnings, cero fallos y cero timeouts.

## 13. Autoridad

P3, live, staging, promoción y merge a main permanecen `false`.

## 14. Cierre

P2-v2 demonstrated observable ordinal influence but did not
deliver the preregistered top-k membership treatment.

Exact numerical treatment geometry is not reconstructible from
the frozen receipts and was not retrospectively regenerated.

No conclusion about N3 decisional gain is authorized.
