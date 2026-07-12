---
title: ADR_NEURAL_CONNECTOME
status: experimental
date: 2026-07-12
---

# Conectoma neural-simbólico RNFE v1

## Decisión

RNFE representa su conectividad N0–N6 mediante un conectoma tipado, versionado y
hasheado. La topología estructural declara puertos y techos de autoridad. El estado
funcional de cada episodio se deriva de candidatos hasheados y `ConsumerReceipt`
validados; un nombre de consumidor o una arista declarada no cuentan como actividad.

La conectómica precede a la activación de Mamba2, H-Net, N6 entrenado y futuras
campañas de aprendizaje: esos sistemas deberán conectarse por puertos explícitos y
producir evidencia compatible con esta frontera.

## Invariantes A-M0

- N1–N6 sólo emiten evidencia o propuestas; ninguna arista neuronal es autoritativa.
- Scheduler, DED, LOT-F, NESY, CAU/CTF/C-GWM, MFM, SMG, certificación, MSRC y
  autoevolución conservan sus autoridades actuales.
- `off` produce cero actividad funcional.
- Un snapshot nunca muta el grafo ni autoriza una acción.
- La plasticidad acumula observaciones únicas y sólo emite deltas propuestos,
  acotados a `[-0.05, 0.05]`, tras al menos tres observaciones.
- Toda propuesta plástica mantiene `apply_authorized=false` y
  `authority_effect=none`; aplicar cambios requerirá sandbox, certificación,
  rollback y una decisión posterior explícita.

## Contratos

- Topología: `rnfe-connectome-v1`.
- Actividad: `rnfe-connectome-activity-v1`.
- Propuesta plástica: `rnfe-connectome-plasticity-v1`.
- Identidad: organismo, linaje, run, episodio, escenario y traza de la simbiosis.
- Integridad: SHA-256 sobre JSON canónico estricto.

## Integración

`SymbioticNeuralCoordinator` mantiene el observador conectómico y adjunta su
snapshot a la traza y al bloque neuronal de certificación. La transición vital ya
ancla esa traza completa mediante `symbiosis_trace_hash`, por lo que el snapshot
queda encadenado sin modificar el runner autoritativo. El coordinador expone puertos
de exportación/restauración del estado conectómico para que el dueño del sustrato
los incorpore al checkpoint en una ventana coordinada. La integración no agrega
hooks de decisión ni modifica grafos causales, memoria, certificación o autoevolución.

## Promoción y rollback

El conectoma permanece experimental hasta demostrar determinismo, trazabilidad,
presupuesto acotado y cero influencia sobre cierre/acción/certificación. El rollback
consiste en retirar el snapshot aditivo; las rutas autoritativas existentes no
dependen de él.
