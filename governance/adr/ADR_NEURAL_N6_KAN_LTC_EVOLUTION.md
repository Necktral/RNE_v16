---
title: ADR_NEURAL_N6_KAN_LTC_EVOLUTION
status: experimental
version: 1.1.0
date: 2026-07-12
owner: Codex
---

# ADR — N6 KAN, LTC y evolución estructural

## Decisión

KAN spline exportable a SymPy, LTC estable sobre vitals y mutaciones de whitelist
entregadas a sandbox/certificación. EVO_SEARCH continúa como baseline de intervención.
N6 puede convertir evidencia plástica elegible del conectoma en una propuesta
`parameter_bound`, pero excluye todas sus propias aristas para impedir
autorefuerzo. La propuesta conserva `apply_authorized=false` y no muta el grafo.

## Hipótesis y coste

KAN supera EML-SR/lineal y LTC mejora anticipación vital sin oscilación. N6 completo
no supera 3 GiB VRAM ni aplica cambios sin evidencia global A-M0.

## Dependencias y rollback

P29 exige `apply_fn`, veredicto y rollback reales. Sin ellos el gate devuelve
`applied=false`. Fallo de aplicación invoca el token de rollback inmediatamente.
