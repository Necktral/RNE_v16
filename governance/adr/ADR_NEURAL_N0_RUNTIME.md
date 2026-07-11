---
title: ADR_NEURAL_N0_RUNTIME
status: experimental
version: 1.0.0
date: 2026-07-10
owner: Codex
---

# ADR — N0 Runtime neuronal

## Contexto

RNFE no tenía una frontera común que impidiera carga eager, falsa autoridad o
fallos silenciosos de modelos. `lab_only` era metadato, no un gate general.

## Decisión

Crear `runtime/neural/` puro Python con modos off→experimental→shadow→provisional,
manifiestos SHA-256, registro lazy, presupuestos físicos, fallback y eventos.
Todo backend devuelve propuestas; el gate RNFE conserva autoridad.

## Hipótesis falsable y coste

N0 permite ejecutar modelos sin alterar la ruta off y recuperarse de hash inválido,
OOM o presión. N0 añade cero VRAM en off; el perfil local limita residencia a 6 GiB,
reserva 1.5 GiB y corta a 80 %/82 °C.

## Rollback

`RNFE_NEURAL_MODE=off`; retirar el registro del backend. No hay migraciones.
