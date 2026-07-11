---
title: ADR_NEURAL_N4_CAUSAL_GRAPH
status: experimental
version: 1.0.0
date: 2026-07-10
owner: Codex
---

# ADR — N4 Grafo neural causal

## Decisión

Message passing de tres capas sobre snapshots canónicos. Predice estados, efectos
e incertidumbre, pero no emite mutaciones del grafo ni suplanta CAU/CTF/CGWM.

## Hipótesis y coste

Debe mejorar contrafactuales frente a autoridades existentes con cero divergencias
no trazadas. Objetivo menor a 2 GiB de VRAM.

## Rollback

Shadow/off descarta las predicciones y conserva el grafo canónico sin migración.
