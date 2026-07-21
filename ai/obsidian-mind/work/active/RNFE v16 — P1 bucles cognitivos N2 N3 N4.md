---
description: "Implementación y experimento SHADOW P1 para cerrar N2, N3 y N4"
tags:
  - rnfe
  - neural
  - p1
  - active
---

# RNFE v16 — P1 bucles cognitivos N2 N3 N4

## Objetivo

Medir si N2, N3 y N4 aportan cognición demostrable sin concederles autoridad
operativa. La política canónica continúa gobernando acción, memoria, mundo y
certificación. Véase [[RNFE v16 — Conectoma neural-simbólico]].

## Contratos implementados

- `MetaScheduler.run_shadow()` aísla estado, persistencia y todo acceso al
  razonador externo, incluidos aumentos indirectos desde familias core.
- El oracle enumera todas las intervenciones desde el mismo estado preacción,
  respeta minimize/maximize, usa epsilon `1e-9` y sella request+outcomes.
- N2 permite exactamente una revisión `core_plus_ind` sólo tras DED+LOT-F válidos
  y NESY rechazado; liga input/candidato/intervención y limita confianza.
- N3 hace retrieval y scheduler alternativos sobre un snapshot MFM atestado, sin
  escrituras ni recibos obligatorios.
- N4-v2 usa un artefacto preacción entrenado; missing, incompatible y OOD abstienen.
  El prior causal sólo es baseline. El entrenamiento verifica splits completos
  24/6/12 y no abre la evaluación sellada.

## Ejecución

- Rehearsal: `p1-ablate --life-steps 8`, tres semillas.
- Final: `p1-ablate --life-steps 32`, doce semillas, diez perfiles, 3.840 pasos
  de vida; 3.250 emitieron episodio y todos cerraron durablemente.
- Evidencia: `rnfe-p1-cognitive-loop-v1` y `p1/matrix.json`.
- Autoridad: siempre `none`; promoción y staging siempre falsos.
- Docencia: Codex docente externo; 7B alumno deshabilitado durante P1.

## Estado

Implementación y experimento final completados. El baseline previo quedó separado
en `d32db2b` y P1 en `06e95c8`. La corrida oficial es
`neural-p1-final-20260721-06e95c8`; su matriz original tiene SHA-256
`30af9f71de23c8b51e0414471a0bc1644ab7408545aba36470d75ea501c297f0`.

La auditoría posterior detectó un error del reporte: `closure_rate` dividía por
todos los pasos de vida e interpretaba `quarantine` sin episodio como cierre
faltante. No faltó ningún `episode.closed`: 3.250/3.250 episodios emitidos cerraron,
sin duplicados ni inesperados. La métrica se separó en integridad de cierre y
`episode_emission_rate`; además se añadió hash/gate de conducta canónica. La
comparación `off` contra nueve perfiles, doce semillas y 32 pasos dio 3.456/3.456
proyecciones idénticas.

Resultado corregido de valor cognitivo:

- N2 no pasa: cero falsos aceptados, pero también cero correcciones puntuables y
  70 rechazos finales.
- N3 pasa su endpoint acotado: delta nDCG@k positivo con IC95% completamente
  positivo y Brier entrenado mejor que reference. MRR no mejora y balanced
  accuracy sigue débil; es señal útil limitada, no autoridad live.
- N4 no pasa: cobertura 91,7% y mejora contra política, pero empata al prior causal;
  aislado abstiene OOD porque depende de señales N3.
- Integridad global corregida: cierre 100%, certificación 100%, cero violaciones y
  conducta canónica idéntica. Ningún resultado autoriza promoción ni staging.

La suite completa después de corregir el reporte cerró con `1715 passed, 22
skipped, 32 xfailed, 1 xpassed` y cero fallos.
