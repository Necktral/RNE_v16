---
date: 2026-07-10
description: "Auditoría de contexto fresco de la campaña neural N0–N6 de Codex (7e1e88b), ratificación de P-SEG (B48+B39), y merge de la campaña como código inerte a main tras verificación ejecutada."
status: accepted
tags:
  - decision
  - rnfe
  - neural
---

# Decision: Auditoría y merge de la campaña neural N0–N6 + ratificación de P-SEG

## Context

Codex entregó la campaña neural completa N0–N6 en un solo commit (`7e1e88b`, rama `codex/neural-n0-a-m0`, worktree `RNE_v16_worktrees/neural-n0`), construida sobre copias re-hasheadas de P1 y A-M0. El protocolo no distingue autor: lo que entrega Codex pasa por el mismo gate que lo que entrega Claude. En paralelo se ejecutó P-SEG (B48+B39), sustrato pre-N1 del gate del kernel.

## Options Considered

1. **Mergear la campaña como código inerte tras auditoría + verificación ejecutada** — elegida.
2. Mergear por confianza en los disclaimers de Codex — descartada (el protocolo exige verificación, no confianza).
3. Rechazar hasta cobertura completa de los GLOBAL TEST GATES — descartada (la cobertura es trabajo de promoción por órgano, no bloqueador de merge de código inerte OFF).

## Decision

- **Auditoría (3 lotes a `verificador-claims-externos`, contexto fresco): 14 CONFIRMADO / 3 MATIZADO / 0 refutado.** Aislamiento total (cero imports de `runtime.neural` en caminos vivos, verificado por comando), sin autoridad (máx. `BOUNDED_PROPOSAL` vía admission_gate), fail-closed (SHA-256 raise; Mamba2 bloqueado sin vendor commit), default OFF, KAN/LTC no tocan seguridad, N2 aumenta el NESY, benchmark real, `engines/` y la ley intactos.
- **Verificación ejecutada (no inferida) antes de tocar main:** `git show 83d8f34` (identidad = A-M0, mío); **26 tests neurales `26 passed in 2.03s`** en el worktree; grep de comando confirma **cero imports** en los 4 caminos vivos.
- **Merge:** cherry-pick de A-M0 (`83d8f34`→`4ab700c`) y de la campaña (`7e1e88b`→`3121b4c`) a `main` como **código inerte** (default `NeuralMode.OFF`). `main = 3121b4c`.
- **P-SEG (B48+B39):** implementado (Fable), `auditor-reparacion`: APROBAR, suite verde independiente (34 gate tests + 1125 completa), commiteado en `feat` (`b0452cd`), B41/`organism_id` intacto.
- **FASE_VIGENTE = 1** confirmada por el humano (en main).

## Consequences

- `main` (`3121b4c`) tiene la ley simbiótica completa (A-M0) + los órganos neurales inertes. La memoria de coordinación permanece en `feat`, fuera de main.
- **Gate de promoción por órgano (shadow→provisional), definido por los 3 matices:** (1) agregar tests `artifact-missing` + `schema-inválido` y completar la matriz de ~17 categorías por órgano (CC4); (2) agregar el campo explícito **"criterio de promoción"** a los 7 ADRs (CC5). Ningún órgano sale de shadow sin eso.
- **P-SEG** queda como sustrato sano listo: el gate bajo el cual N1 puede salir de shadow. Pendiente su promoción a main.
- `canon-apex-v3.0` sigue en `34cd3a3` (cúspide v3.0.0); A-M0 (v3.1.0) ya está en main sin re-tag (opción abierta: `canon-apex-v3.1`).

## Related

- [[RNFE v16 — Backlog de Reparación]]
- [[Protocolo de coordinación campaña neural]]
- [[Cierre de adjudicación y reconciliación externa 2026-07-10]]
- [[Key Decisions]]
