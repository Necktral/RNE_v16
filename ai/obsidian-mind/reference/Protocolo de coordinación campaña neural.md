---
date: 2026-07-10
description: "Protocolo permanente de coordinación entre la reparación de RNFE v16 y la campaña neural N0→N6 de Codex: partición por zona, sustrato-antes-que-órgano, cadencia de sync y no-intrusión en runtime/neural/."
tags:
  - reference
  - rnfe
  - coordination
---

# Protocolo de coordinación campaña neural

Regla permanente para el orquestador mientras la **campaña neural** (Codex, N0→N6) y la reparación se **turnan** sobre el mismo repo ([[RNFE v16 — Backlog de Reparación]]).

> [!important] Modelo de ejecución — POR TURNOS, no concurrente
> Codex está **en pausa mientras la reparación trabaja**, y avanza solo cuando la reparación no. Por lo tanto: durante una sesión de reparación **NO hay modificación concurrente** — los worktrees/ramas de Codex están congelados y no cambian bajo los pies del orquestador. El chequeo de sync (`git fetch` + comparar `origin/main` y ramas `codex/*`; rebase de reparación + baseline si Codex avanzó) se hace **al INICIO de cada sesión de reparación** — en los bordes, no a mitad del trabajo. La cadencia de §3 se lee con esta luz.

> [!abstract] Principio rector — un organismo integral y simbiótico
> RNFE es UN organismo (axioma **A-M0** de la cúspide `RNFE_canon_matematico_f2_4_v3_0.md`). Reparación y campaña neural no son proyectos separados por un muro: son **funciones simbióticas de un mismo cuerpo**, integradas con sinergia. Esta partición es división del trabajo, no una frontera adversarial; las dependencias de sustrato (§2) son la simbiosis misma. **Ninguna optimización local de una zona es válida si rompe la sinergia del todo.**

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
