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
- Final: `p1-ablate --life-steps 32`, doce semillas, diez perfiles, 3.840 episodios.
- Evidencia: `rnfe-p1-cognitive-loop-v1` y `p1/matrix.json`.
- Autoridad: siempre `none`; promoción y staging siempre falsos.
- Docencia: Codex docente externo; 7B alumno deshabilitado durante P1.

## Estado

Implementación validada. El baseline previo quedó separado en el commit
`d32db2b`. La suite posterior P1 cerró con `1712 passed, 22 skipped, 32 xfailed,
1 xpassed` y cero fallos. El rehearsal oficial queda como siguiente operación y no
puede otorgar autoridad aunque sus gates pasen.
