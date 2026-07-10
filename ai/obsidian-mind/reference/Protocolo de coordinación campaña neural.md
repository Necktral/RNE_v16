---
date: 2026-07-10
description: "Protocolo permanente de coordinación entre la reparación de RNFE v16 y la campaña neural N0→N6 de Codex: partición por zona, sustrato-antes-que-órgano, cadencia de sync y no-intrusión en runtime/neural/."
tags:
  - reference
  - rnfe
  - coordination
---

# Protocolo de coordinación campaña neural

Regla permanente para el orquestador mientras la **campaña neural** (Codex, N0→N6) corra en paralelo a la reparación ([[RNFE v16 — Backlog de Reparación]]).

## 1. Partición por zona

- **Reparación (dueña):** kernel/gate, storage, experience/memoria-sustrato, contracts, canon.
- **Campaña N (dueña):** `runtime/neural/` y los adaptadores de órganos.
- **Zonas compartidas** (`scheduler_meta`, `world`): **chequeo obligatorio antes de despachar** cualquier paquete que las toque → `git log origin/main` primero; si Codex mergeó algo ahí, **rebase de la rama de reparación + re-correr la suite** antes de seguir.

## 2. Sustrato antes que órgano (dependencias duras)

- **B48 + B39** (gate) ⟶ prerequisito de aterrizaje de **N1**.
- **B41 + B42–B45** (identidad y experiencia) ⟶ prerequisito de aterrizaje de **N3**.

Ambos frentes de sustrato se **adelantan** en la fila; quedan anotados como dependencia dura en el backlog.

## 3. Cadencia de sincronización

Tras cada PR de Codex mergeado a `main`: **rebase** de la rama de reparación **+ baseline**. Los resultados se registran en `RNE_v16_analysis/externa/sync_campana_neural.md` con fecha, PR y zonas tocadas.

## 4. No-intrusión

Nada de reparar dentro de `runtime/neural/`. Si un paquete de reparación encuentra un problema ahí, **se reporta para Codex, no se toca**.

## Related

- [[RNFE v16 — Backlog de Reparación]]
- [[RNFE v16 Project Memory]]
- [[Cierre de adjudicación y reconciliación externa 2026-07-10]]
