---
title: ADR_NEURAL_N1_ROUTER
status: experimental
version: 1.3.0
date: 2026-07-15
owner: Codex
---

# ADR — N1 Enrutador de familias

## Decisión

Usar un MLP compacto con catálogo versionado y hard masks para proponer familias
opcionales. Nunca altera backbone ni elude validadores. Sólo se entrena con pares
contrafactuales reales, separados por generador y semilla.
Ranking, activación y presupuesto son salidas distintas. Catálogo v2 incorpora
NESY, EVO_SEARCH, IMAGINATION y A12. Sin utilidad positiva, calibración válida o
incertidumbre aceptable, la decisión obligatoria es `ABSTAIN`.

## Hipótesis y coste

Debe reducir regret frente al scheduler determinista/reward-guided sin perder más
de 1 pp de cierre. Objetivo menor a un millón de parámetros y 1 GiB de VRAM.

## Dependencias y rollback

P23 antes del hook; P19 y causalidad antes de influencia. Shadow/off restaura la
secuencia autoritativa y el catálogo anterior permanece disponible por versión.
El trainer CPU exige como mínimo 300 pares, 50 contextos, 3 generadores y 3
familias. Datos sintéticos sólo validan el contrato: el modelo lab actual tiene
ECE 0.316 y por tanto se abstiene/no es promocionable.

La compuerta de datos también exige al menos 30 pares positivos, 30 negativos y
un rango de utilidad de 0.02. La utilidad incluye efectividad y viabilidad además
de coste, cierre, certificación y continuidad. La campaña nativa v2 superó volumen
y diversidad (360 pares; 108/252), pero el artefacto retenido falló calibración
(ECE validación 0.1545; test 0.1821). Permanece en cuarentena y no reemplaza el
binding preparado.
