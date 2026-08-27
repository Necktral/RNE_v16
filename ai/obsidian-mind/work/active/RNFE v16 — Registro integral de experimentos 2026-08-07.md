---
date: 2026-08-07
description: "Snapshot reproducible y sincronización draft de las líneas experimentales RNFE v16."
tags:
  - work-note
  - rnfe
  - experiments
  - github
status: active
quarter: Q3-2026
---

# RNFE v16 — Registro integral de experimentos 2026-08-07

## Decisión

La evidencia pesada permanece local. GitHub recibe sólo documentación,
inventarios normalizados, metadatos permitidos y SHA-256. Véase
`docs/experiments/README.md` y `docs/experiments/registry.json`.

## Estado

- Se publicaron seis líneas en draft PRs independientes, sin merge ni push a
  `main`.
- ASCG fase 1 permanece `PARTIAL` porque PostgreSQL aislado no fue validado.
- La nocturna `neural-nightly-20260807-d2d54a64` terminó fail-closed por
  `dirty_worktree`; no hubo entrenamiento, staging ni promoción.
- Codex/frontier sigue como docente de referencia; el 7B es alumno/proponente
  supervisado sin autoridad.

## Relaciones

- [[RNFE v16 Project Memory]]
- [[RNFE v16 — Backlog de Reparación]]
- [[North Star]]
- [[Gotchas]]
