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

Architectural or workflow decisions worth recalling. Link to the full [[Decision Record]] when one exists.

-
