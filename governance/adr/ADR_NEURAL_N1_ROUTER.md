---
title: ADR_NEURAL_N1_ROUTER
status: experimental
version: 1.0.0
date: 2026-07-10
owner: Codex
---

# ADR — N1 Enrutador de familias

## Decisión

Usar un MLP compacto con catálogo versionado y hard masks para proponer familias
opcionales. Nunca altera backbone ni elude validadores. Sólo se entrena con pares
contrafactuales reales, separados por generador y semilla.

## Hipótesis y coste

Debe reducir regret frente al scheduler determinista/reward-guided sin perder más
de 1 pp de cierre. Objetivo menor a un millón de parámetros y 1 GiB de VRAM.

## Dependencias y rollback

P23 antes del hook; P19 y causalidad antes de influencia. Shadow/off restaura la
secuencia autoritativa y el catálogo anterior permanece disponible por versión.
