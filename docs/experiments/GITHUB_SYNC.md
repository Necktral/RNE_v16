# Matriz de sincronización GitHub — 2026-08-07

Todas las publicaciones son draft PRs. No hubo push directo a `main`, merge,
force-push, entrenamiento, nightly, staging ni promoción.

| Línea | Commit remoto | Base | Draft PR | QA / estado |
|---|---|---|---|---|
| `feat/reasoning-family-quality-deep` | `6630258da9d49eccc269f1814ca29efec44f73c6` | `main` | [#5](https://github.com/Necktral/RNE_v16/pull/5) | 59 targeted; 1.311 passed, 15 skipped, 32 xfailed, 1 xpassed |
| `repair/P12` | `cdac5ef10d2acb0c359a65610f67e45de1289b78` | `main` | [#6](https://github.com/Necktral/RNE_v16/pull/6) | 237 targeted; 1.593 passed, 22 skipped, 32 xfailed, 1 xpassed; whitespace histórico documentado |
| `feat/neural-substrate` | `c2b9afae637449d404fadacb913b6c7af0d176b6` | `main` | [#7](https://github.com/Necktral/RNE_v16/pull/7) | 109 targeted, 7 skipped; 1.624 passed, 29 skipped, 32 xfailed, 1 xpassed; whitespace histórico documentado |
| `codex/neural-n0-a-m0` | `7e1e88ba8aca2716d9740ed8f2f814908d90dbbc` | `main` | [#8](https://github.com/Necktral/RNE_v16/pull/8) | Línea histórica: 26 targeted; 1.142 passed, 15 skipped, 32 xfailed, 1 xpassed; riesgo de supersesión |
| `codex/shadow-causal-observability-v1` | `ed45999de3059622120f34239d11beec2b5bca89` | `codex/p1-n3-attribution-audit-v1` | [#9](https://github.com/Necktral/RNE_v16/pull/9) | 6 targeted; 1.740 passed, 22 skipped, 32 xfailed, 1 xpassed; observabilidad sin ampliar autoridad |
| `codex/ascg-v1-1-phase1-ledger` | `c4084bd7047d1f8e538e78135bac5a87ceb2eab2` | `codex/p1-n3-attribution-audit-v1` | [#10](https://github.com/Necktral/RNE_v16/pull/10) | **PARTIAL**: 37 passed, 9 PostgreSQL skipped; global 1.771 passed, 31 skipped, 32 xfailed, 1 xpassed |
| `agent/experiment-registry-sync-20260807` | `77f8d0e5e8d4f308da34a9555d1ce3da4a345829` | `main` | [#11](https://github.com/Necktral/RNE_v16/pull/11) | 12 targeted; 1.514 passed, 22 skipped, 32 xfailed, 1 xpassed; schema y determinismo validados |

Los skips PostgreSQL de ASCG son deliberados: `RNFE_CG_POSTGRES_TEST_DSN` no
estaba configurado. El PR #10 no debe dejar de ser `PARTIAL` hasta repetirlos
contra una base aislada y desechable. El PR #4 existente no fue modificado.

Los worktrees detached y las ramas idénticas al remoto sólo figuran en
[`registry.json`](registry.json); no se crean PRs redundantes.
