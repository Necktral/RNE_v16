---
title: ADR_NEURAL_N5_HNET_INGEST
status: experimental
version: 1.2.0
date: 2026-07-12
owner: Codex
---

# ADR — N5 Ingestión H-Net

## Decisión

Puerto H-Net inyectable y servicio real texto/bytes→chunks→sinks SMG/MFM. Sin
modelo certificado se usa un chunker determinista Unicode y no se declara H-Net.
H-Net entrega fronteras de inicio de chunk (`split_offset`) en bytes UTF-8 —el
byte cero siempre es frontera—. N5 conserva esa procedencia,
la convierte a offsets de codepoint sólo en límites válidos y registra ambos
espacios; token offsets requieren un mapa explícito.

## Hipótesis y coste

Mejor F1 de frontera que el chunker, con corpus RNFE retenido y máximo 4 GiB VRAM.

## Dependencias y rollback

H-Net queda fijado al upstream `3ae01de...` y su MIT está vendorizada. Peso, hash,
evidencia de entrenamiento y caller vivo son obligatorios. Cualquier
ausencia vuelve al chunker; memoria recibe candidatos, nunca promoción directa.
El entrenamiento GPU aborta antes de superar 82 °C, 6 GiB residentes o la reserva
de 1.5 GiB. Sin presupuesto físico no se crea ni activa artefacto.
