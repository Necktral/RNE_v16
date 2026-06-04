---
title: ADR_MSRC_MULTISCALE_RESOLUTION
status: normative
version: 1.0.0
date: 2026-04-20
owner: Codex
type: architecture-decision-record
subject: Controlador Multi-Escala de Resolución (MSRC) para RNFE
---

# ADR — MSRC Multi-Scale Resolution Controller

## 1. Contexto

RNFE requiere selección dinámica de resolución representacional sin romper el contrato runtime.
El estado actual opera con escenarios 1x1 y 5x5, con pipeline de benchmark y análisis estable.
Falta una capa formal para decidir escala de manera auditada y con prioridad cognitiva.

## 2. Decisión

Se incorpora MSRC como subsistema desacoplado en `runtime/control/msrc`, integrado en la capa
`benchmark + reality service` sin modificar `ScenarioEpisodeRunner`.

## 3. Principios normativos

1. Primacía de inteligencia útil con cierre sobre austeridad de costo.
2. Viabilidad dinámica como guardia dura previa a cualquier ahorro operativo.
3. Objetivo lexicográfico: cognición -> viabilidad -> costo meta.
4. VRAM se modela como activo estratégico; solo penaliza en cercanía peligrosa al borde.
5. Proxies (`success_rate`, `viability_margin`) se mantienen explícitos como proxies.

## 4. Alcance de v1

1. Catálogo formal extensible con escalas `1x1,2x2,3x3,5x5,10x10,30x30`.
2. Ejecutables reales en v1: `1x1` y `5x5`.
3. Estimador estructurado con señales de riesgo, heterogeneidad, epistemia y presión operativa.
4. Política con hysteresis, probe mode, downgrade con evidencia y antioscilación.
5. Guardia de memoria cross-scale para bloquear transferencia estructural cruda.
6. Auditoría canónica de decisiones y transiciones en ledger + artifacts JSONL.

## 5. Contratos y trazabilidad

1. `contracts/msrc_scale_decision.schema.json`
2. `contracts/msrc_transition_event.schema.json`
3. Eventos de auditoría:
   - `msrc.decision`
   - `msrc.probe.started`
   - `msrc.probe.completed`
   - `msrc.probe.committed`
   - `msrc.probe.discarded`
   - `msrc.transition`
   - `msrc.rollback`
   - `msrc.regret`
   - `msrc.oscillation`

## 6. Invariantes

1. No modificar contrato de `ScenarioEpisodeRunner`.
2. No reintroducir `max_steps` en runtime.
3. No reintroducir métricas removidas en capas activas.
4. No mezclar memoria densa cross-scale sin abstracción explícita.

## 7. Consecuencias

### Ventajas

- Escalado representacional auditable y extendible.
- Mejor alineación entre demanda cognitiva y resolución.
- Integración natural con `reality_bench_runs` y pipeline analítico existente.

### Costos

- Complejidad de política/hysteresis.
- Riesgo de ajuste de umbrales en primeras campañas.
- Métrica de fragmentación VRAM aproximada, no directa.

## 8. Criterios de verificación

1. Pruebas unitarias de catálogo/estimador/política/transición/guard/logger.
2. Prueba de integración `always_1x1 vs always_5x5 vs adaptive_msrc` con persistencia.
3. Decisiones reconstruibles offline desde JSONL y eventos de ledger.
4. Ausencia de contaminación de memoria estructural cross-scale.
