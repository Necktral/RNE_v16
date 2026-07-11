---
description: "Architectural and workflow decisions worth recalling across sessions — each links to its source work note"
tags:
  - brain
---

# Key Decisions

## 2026-07-10 - Install Obsidian Mind as Project-Local Agent Memory

Decision: install Obsidian Mind under `ai/obsidian-mind/` and expose it through root agent bridge files instead of replacing RNFE project root files.

Status: accepted

Rationale: RNFE already has canonical docs, contracts, and runtime memory semantics. A project-local vault gives all agents persistent development memory while avoiding contamination of RNFE runtime architecture.

Related: [[RNFE v16 Project Memory]], [[North Star]]

## 2026-07-10 - Cierre de adjudicación y reconciliación con auditoría externa

Decision: verificar los 16 claims de la auditoría externa contra el árbol real (**21/21 CONFIRMADO**), rutearlos al backlog (`B39–B48` nuevos; C2→B48; convergentes anotan B2/B3/B38/A5), y cerrar el bloque A consolidando los 68 ítems (A1–A20 por recomendación + B1–B48) en **31 paquetes ordenados** (P0–P30).

Status: accepted — **PAUSA antes de despachar P0** (espera confirmación del orden).

Rationale: base de reparación = checkout `feat/reasoning-family-quality-deep` en `pre-repair` (main no contiene pre-repair, sin divergencia). Orden por dependencia + radio-SCC creciente; plan verificado adversarialmente (cobertura 68/68, 0 violaciones).

Related: [[Cierre de adjudicación y reconciliación externa 2026-07-10]], [[RNFE v16 — Backlog de Reparación]], [[RNFE v16 Project Memory]]

## 2026-07-10 - Auditoría y merge de la campaña neural N0–N6 + P-SEG

Decision: auditar la entrega neural de Codex (`7e1e88b`) con contexto fresco (mismo gate que Claude) y mergearla a main como **código inerte** (default OFF) tras **verificación ejecutada** (26 tests verde en el worktree, cero imports en caminos vivos por comando, no por lectura). A-M0 + campaña → `main = 3121b4c`. P-SEG (B48+B39) ratificado (auditor APROBAR) y commiteado en feat.

Status: accepted — gate de promoción por órgano (shadow→provisional) pendiente: tests artifact-missing/schema-inválido + campo "criterio de promoción" en los 7 ADRs.

Rationale: el aislamiento y la no-autoridad se confirman por comando; la cobertura incompleta no bloquea código inerte, define el trabajo de promoción. Principio A-M0: sustrato (P-SEG) y órgano (N1) son simbióticos.

Related: [[Auditoría y merge campaña neural N0-N6 2026-07-10]], [[RNFE v16 — Backlog de Reparación]], [[Protocolo de coordinación campaña neural]]

Architectural or workflow decisions worth recalling. Link to the full [[Decision Record]] when one exists.

-
