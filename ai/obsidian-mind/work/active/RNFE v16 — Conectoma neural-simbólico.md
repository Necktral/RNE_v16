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

## Capa tecnológica — 2026-07-12

La rama `integration/neural-tech-connectomic-v1` compone sin conflictos los
commits de Claude Code `6c3fad3` (Mamba2 lazy/Turing) y `6ced3b6` (compatibilidad
H-Net/FlashAttention) con el conectoma. Sus 47 pruebas de engine pasaron.

- H-Net: upstream exacto `3ae01de...`, MIT vendorizada; se corrigió la frontera
  nativa a `split_offset`. Loader/trainer N5 listos, artefacto aún ausente porque
  el gate físico rechazó entrenar con otro proceso ocupando 6.38 GiB VRAM.
- Mamba2/N3: SSD-minimal CPU entrenado en laboratorio, 2 631 parámetros,
  pérdida 0.673→0.561; ejecución shadow end-to-end verificada.
- N1: MLP lab de 580 parámetros; ECE 0.316 obliga abstención/no promoción.
- N4: message-passing tipado lab de 64 parámetros; no aprende topología.
- N6: consume propuestas plásticas elegibles, excluye sus propias aristas y no
  dispone de `apply_fn`.

Los artefactos lab locales viven bajo `rnfe_artifacts/neural/` y están ignorados
por Git. Véase `governance/adr/ADR_NEURAL_TRAINED_BINDINGS.md`.

Ver [[RNFE v16 Project Memory]] y
`governance/adr/ADR_NEURAL_CONNECTOME.md`.
