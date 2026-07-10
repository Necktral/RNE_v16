---
date: 2026-07-10
description: "Shared project memory for RNFE v16 agent sessions, including canonical sources, guardrails, and active orientation."
tags:
  - work-note
  - rnfe
status: active
quarter: Q3-2026
---

# RNFE v16 Project Memory

## Context

This vault is the shared development memory for [[North Star]] work on the RNFE v16 repository.

Repository root: `/home/wis/Desarrollo/RNE_v16`

## Canonical Sources

- `canon/normative/` is the normative source of truth.
- `docs/analysis/00_INDEX.md` is the current code-reality audit.
- `README.md` is the top-level orientation.
- `docs/history/` contains historical snapshots and may be stale.
- `archive/` is quarantined legacy code.

## Boundaries

Obsidian Mind is persistent memory for development agents. It does not replace RNFE runtime memory, memory compatibility policy, certification, scenario compatibility, or JSON contracts.

## Active Agent Setup

- Root `AGENTS.md`, `CLAUDE.md`, and `GEMINI.md` point agents to this vault.
- Root `.codex/hooks.json`, `.claude/settings.json`, and `.gemini/settings.json` run shared hooks from this vault.
- Codex skills mirrored from Obsidian Mind subagents live in `.agents/skills/`.

## Related

- [[RNFE v16 — Backlog de Reparación]]
- [[Cierre de adjudicación y reconciliación externa 2026-07-10]]
- [[Memories]]
- [[Key Decisions]]
- [[Gotchas]]
- [[Patterns]]
