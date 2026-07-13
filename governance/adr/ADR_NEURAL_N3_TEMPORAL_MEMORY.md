---
title: ADR_NEURAL_N3_TEMPORAL_MEMORY
status: experimental
version: 1.1.0
date: 2026-07-12
owner: Codex
---

# ADR — N3 Memoria temporal

## Decisión

Puerto SSM version-neutral con estado por organismo+escenario+linaje. Sólo propone
prioridad, importancia, riesgo y continuidad; MFM sigue siendo autoridad y fallback.
La implementación entrenable usa la factorización SSD-minimal de Mamba2 en torch
puro sobre CPU. El vendor fusionado queda disponible para inferencia verificada,
pero su backward Triton no se presenta como entrenable en Turing.

## Hipótesis y coste

Debe superar MFM, Markov/lineal y GRU en recuperación retenida sin contaminación
cross-scenario. El modelo lab actual tiene 2 631 parámetros y cero VRAM.

## Dependencias y rollback

Mamba v2.2.5 queda fijado a `e0761ece...` con Apache-2.0 vendorizada. Sólo se carga
un artefacto local con SHA, evidencia de entrenamiento y manifiesto válido. El
artefacto lab permanece shadow/no promocionable; unload borra estado volátil y MFM
mantiene continuidad.
