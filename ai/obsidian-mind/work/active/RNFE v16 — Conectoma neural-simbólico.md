---
date: 2026-07-12
status: active
tags:
  - rnfe
  - neural
  - connectome
---

# RNFE v16 — Conectoma neural-simbólico

## Contexto

La campaña continúa bajo [[North Star]] y A-M0: primero se construye la
conectómica del organismo; Mamba2, H-Net, N6 entrenado y otras tecnologías se
integrarán después mediante puertos explícitos.

## Implementación

- Rama: `integration/connectomic-organism-v1`.
- Base: `main@b56562fb8cb852eac00466c0b10361c01b24cb93`.
- Contratos: topología `rnfe-connectome-v1`, actividad
  `rnfe-connectome-activity-v1`, plasticidad propuesta
  `rnfe-connectome-plasticity-v1` y checkpoint `rnfe-connectome-checkpoint-v1`.
- La actividad sólo nace de candidatos hasheados y `ConsumerReceipt` validados.
- CAU, CTF y C-GWM aparecen como autoridades causales separadas.
- La plasticidad es acotada, deduplicada y no aplicable; no muta el grafo.
- `off` no añade conectómica a la traza viva.
- No se modificaron `runtime/world`, razonamiento, memoria, certificación ni
  autoevolución; se conserva el editor único del sustrato.

## Integración pendiente coordinada

El coordinador expone exportación/restauración del ledger conectómico. Su conexión
al checkpoint soberano corresponde a una ventana con el dueño del sustrato. La
traza conectómica ya queda anclada transitivamente por el
`symbiosis_trace_hash` de la transición vital.

Ver [[RNFE v16 Project Memory]] y
`governance/adr/ADR_NEURAL_CONNECTOME.md`.
