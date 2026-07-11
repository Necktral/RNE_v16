---
title: Campaña de órganos neuronales N0-N6
status: active
version: 1.0.0
date: 2026-07-10
owner: Codex
principle: A-M0
---

# Campaña N0→N6 — organismo integral y simbiótico

## 1. Ley de diseño

A-M0 gobierna toda decisión de esta campaña: un órgano no se promueve por una
métrica local. Debe mejorar o preservar cierre, certificación, continuidad,
viabilidad, coherencia, seguridad, herencia y presupuesto físico del organismo.
La reparación de Fable y el crecimiento neural son especializaciones simbióticas;
los puertos entre ambas son parte del diseño, no muros.

## 2. Arquitectura

`runtime/neural/` contiene contratos puros, registro lazy, gate de modos,
admisión física, backends y adaptadores. Los modelos sólo producen propuestas.
Kernel, scheduler, mundo, MFM, gates y certificadores existentes conservan
autoridad. `off` no abre artefactos ni emite eventos; `experimental` es de
laboratorio; `shadow` observa la frontera viva sin cambiarla; `provisional`
requiere gate, presupuesto y enlace causal.

La ausencia de P-CADENA no bloquea investigación: marca la inferencia como
`unlinked` y degrada provisional a shadow. Pesos o dependencias ausentes siempre
producen fallback explícito; el runtime no descarga nada.

## 3. Estado real por órgano

| Paquete | Implementado en la campaña | Condición pendiente para frontera viva |
|---|---|---|
| N0 | Contratos, manifiesto/hash, registro lazy, recursos, modos, eventos y reporte A-M0 | Extensión P25/B47 para telemetría GPU absoluta |
| N1 | MLP JSON reproducible, catálogo v1, hard masks, admisión y dataset contrafactual | P23 para aterrizaje; P19 + causalidad para influencia; dataset diverso para entrenar |
| N2 | Recurrencia compartida 4–16 pasos y verificación DED+LOTF por candidato | Hook NESY coordinado y benchmark retenido |
| N3 | Puerto SSM, backend de referencia con estado por organismo/escenario y autoridad MFM | P-CADENA/P21; revisión/licencia/dependencias Mamba2 |
| N4 | Message passing de tres capas, predicción sin mutación del grafo | Dataset causal atestado y hook world coordinado |
| N5 | Chunker Unicode, puerto H-Net y caller real hacia sinks SMG/MFM | Revisión/pesos H-Net y puertos vivos de Fable |
| N6 | KAN exportable, LTC y gate estructural con whitelist/rollback | P29 con `apply_fn` real y certificación de sandbox |

Un backend de referencia valida matemáticamente el contrato, pero no se confunde
con el modelo objetivo. En particular, el SSM de referencia no se declara Mamba2;
`Mamba2Backend` mantiene un stop condition explícito hasta resolver procedencia.

## 4. Datos, artefactos y promoción

N1 sólo acepta pares con igual estado inicial, generador y semilla, donde la única
diferencia sea familia on/off. Los campos históricos `family_delta_*` se rechazan
como etiquetas causales. Train/validation/test se agrupan por generador+semilla.
El umbral mínimo inicial es 300 pares, 50 contextos, 3 generadores y 3 familias.

Pesos viven bajo `RNFE_ARTIFACT_ROOT/neural/`; el repositorio conserva manifiesto,
model card y reporte, no pesos grandes. Cada promoción exige tres semillas, CI95
positivo, pérdida de cierre no mayor a 1 punto porcentual, ECE <= 0.10, cero
violaciones y `OrganismImpactReport` favorable. Tras cada órgano se versiona y
recalibra el catálogo N1; nunca se agregan salidas silenciosamente.

## 5. Coordinación y secuencia

El orden de integración es N0→N1→N2→N3→N4→N5→N6. Codex mantiene adaptadores y
ADR en `governance/adr/`; Fable mantiene sustratos y realiza hooks compartidos en
una ventana sin ediciones simultáneas. Después de cada unión se rebasa reparación,
se ejecuta la suite completa y se compara el impacto global antes de continuar.
