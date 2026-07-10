# RNFE v16 Agent Memory

This repository uses Obsidian Mind as a shared, project-local memory vault for Codex, Claude Code, Gemini CLI, and other coding agents.

## Shared Vault

- Vault path: `ai/obsidian-mind/`
- Entry point: `ai/obsidian-mind/Home.md`
- Operating manual: `ai/obsidian-mind/CLAUDE.md`
- Shared memory notes: `ai/obsidian-mind/brain/`
- Active work notes: `ai/obsidian-mind/work/active/`

Before substantial work, read:

1. `ai/obsidian-mind/brain/North Star.md`
2. `ai/obsidian-mind/brain/Memories.md`
3. `ai/obsidian-mind/work/Index.md`
4. `README.md`
5. `docs/analysis/00_INDEX.md`

## RNFE Guardrails

- `canon/normative/` is the normative source of truth for RNFE.
- `docs/analysis/00_INDEX.md` is the current code-reality audit.
- `docs/history/` contains historical snapshots and may be outdated.
- `archive/` is quarantined legacy code; do not treat it as live runtime.
- Obsidian Mind is development memory only. It does not replace RNFE runtime memory, certification, scenario compatibility, or canonical contracts.

When durable project context is learned, write it to the vault using Obsidian-style `[[wikilinks]]` and update the relevant index.

## Commands

The Obsidian Mind command prompts live in `ai/obsidian-mind/.claude/commands/`.

- Claude Code and Gemini CLI can use commands such as `/om-standup`, `/om-dump`, and `/om-wrap-up`.
- Codex can use the same commands as plain prompts, for example `om-standup`.

## Codex Skills

Obsidian Mind subagents are mirrored as Codex skills under `.agents/skills/`.
