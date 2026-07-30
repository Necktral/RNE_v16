---
date: 2026-07-29
description: "Implementación del detector causal híbrido y del pipeline reproducible de evidencia, etiquetado temporal y entrenamiento N4."
tags:
  - work-note
  - neural
  - causal-learning
status: completed
quarter: Q3-2026
---

# N4 Línea 1 — fronteras y pipeline de datos

## Resultado

Se añadió un detector puro de fronteras causales, integrado al generador híbrido
N4 sin modificar la semántica de promoción del MCI. La captura de evidencia ahora
puede observar cada `TransitionEvidence` cuando nace y persistirla incrementalmente
sin depender del buffer circular.

El pipeline incluye etiquetado temporal sin leakage, splits completos por seed,
CLI de dataset, CLI de entrenamiento y comprobación de paridad entre el modelo
exportado y el backend Python del runtime.

## Decisiones

- La identidad persistida es `(run_id, evidence_id)`: el `evidence_id` por sí solo
  colisiona correctamente entre trayectorias deterministas de seeds diferentes.
- El bloque histórico de `CausalLearningEngine.generate_structural_hypothesis`
  no se reemplazó; extraerlo literalmente habría cambiado el MCI y no resolvería
  ruido ni contradicciones.
- Las features usan solo `E≤t`; las etiquetas usan exclusivamente `E>t`.
- El artefacto conserva el orden canónico de seis features y el runtime comparte
  la misma ecuación logística de scoring.

## Validación

- 10 pruebas del detector de fronteras.
- 90 pruebas en `tests/neural`, `tests/symbolic` y `tests/experiments`.
- Smoke completo: 60 evidencias, 237 muestras y artefacto N4 exportado.

## Campaña científica

La campaña inicial en seeds 0–99 produjo un artefacto no promotable por ECE de
holdout (`0.1055`). El resultado quedó sellado y ese holdout no se reutilizó.

Una calibración Platt ajustada solo con validation se validó en seeds descartables
1000–1005. La campaña v2 usó seeds nuevos 100–199 y produjo 3,000 evidencias,
30,399 muestras y un artefacto promotable (`Brier=0.0586`, `ECE=0.00863` en su
única consulta de holdout).

La ablación congelada se ejecutó sobre seeds independientes 200–229. El brazo
entrenado mejoró Recall@1 y MAE gain top-1, pero empeoró Recall@2, MRR y NDCG.
El gate científico final no pasó; Línea 1 no debe declararse cerrada todavía.

## Related

- [[North Star]]
- [[RNFE v16 Project Memory]]
- [[Protocolo de coordinación campaña neural]]
