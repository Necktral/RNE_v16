---
title: ADR_NEURAL_TRAINED_BINDINGS
status: experimental
version: 1.0.0
date: 2026-07-12
owner: Codex
---

# ADR — Modelos entrenados sobre el conectoma

## Decisión

N1, N3, N4 y N5 se conectan a sus adaptadores canónicos mediante manifiestos
locales configurados por órgano. N0 verifica ruta, SHA-256, dispositivo, recursos,
licencia y procedencia antes de importar torch, engines o pesos. No hay descargas
en runtime.

Variables de configuración:

- `RNFE_ARTIFACT_ROOT` + namespace `neural` define el artifact root.
- `RNFE_NEURAL_N{1,3,4,5}_MANIFEST` es una ruta relativa dentro de ese root.

En `off` ningún manifiesto se abre. Manifiesto ausente o rechazado mantiene el
adaptador de referencia. Un modelo aceptado sigue con techo `shadow`; su candidato
es evidencia y el fallback de referencia continúa siendo la salida efectiva.

## Evidencia actual

- N1 MLP: 580 parámetros, dataset sintético de validación contractual, ECE 0.316;
  no promocionable.
- N3 Mamba2 SSD-minimal: 2 631 parámetros CPU, pérdida lab 0.673→0.561;
  ejecutado end-to-end en shadow.
- N4 tipado: 64 parámetros, pérdida lab 0.051→0.024; no aprende topología.
- N5: loader, trainer y contrato están listos, pero no se entrenó mientras otro
  proceso consumía 6.38 GiB de VRAM; el gate A-M0 abortó antes de colisionar.
- N6: consume plasticidad conectómica acotada, excluye autorefuerzo y nunca aplica.

Estas métricas prueban cañería, no beneficio cognitivo. Promoción exige datasets
reales, separación por generador/escenario, tres semillas, ECE, intervalos y
`OrganismImpactReport` conforme A-M0.
