# Registro integral de experimentos RNFE

Este directorio publica el índice reproducible de la evidencia local observada el
**2026-08-07**. No contiene modelos, bases de datos, dumps, trazas, prompts ni
payloads. La fuente de verdad sigue siendo el plano local bajo `development://`.

## Snapshot

| Familia | Entradas |
|---|---:|
| Worktrees experimentales | 25 |
| Campañas integrales | 19 |
| Raíces `rnfe_artifacts` | 840 |
| Corridas `data/artifacts` | 480 |
| Familias locales `work/` | 9 |
| Evidencias relevantes `RNE_v16_analysis` | 52 |
| **Total** | **1.425** |

La auditoría previa enumeraba 24 worktrees. El descubrimiento reproducible actual
encuentra 25 directorios reales porque también existe
`p2-n3-causal-decision-v2`; se conserva en el catálogo, pero no se abre un PR
redundante. El worktree temporal que genera este registro queda excluido.

## Archivos

- [`registry.json`](registry.json): snapshot canónico validable por máquina.
- [`GITHUB_SYNC.md`](GITHUB_SYNC.md): ramas, bases, PRs draft, QA y bloqueos.
- [`../../contracts/experiment-registry.v1.schema.json`](../../contracts/experiment-registry.v1.schema.json): contrato JSON Schema 2020-12.
- [`../../scripts/build_experiment_registry.py`](../../scripts/build_experiment_registry.py): generador determinista.

## Reproducción

```bash
RNFE_DEVELOPMENT_ROOT=/path/to/Desarrollo

python scripts/build_experiment_registry.py \
  --development-root "$RNFE_DEVELOPMENT_ROOT" \
  --output docs/experiments/registry.json \
  --snapshot-date 2026-08-07

python scripts/build_experiment_registry.py \
  --development-root "$RNFE_DEVELOPMENT_ROOT" \
  --output docs/experiments/registry.json \
  --snapshot-date 2026-08-07 \
  --check
```

`--check` es read-only y falla si el archivo publicado no coincide con el
filesystem. Dos generaciones con el mismo snapshot deben ser byte a byte
idénticas.

## Semántica y privacidad

- `logical_path` usa exclusivamente URIs `development://`; no publica nombres de
  usuario ni rutas absolutas.
- El SHA-256 de un archivo de análisis cubre su contenido. El de un directorio
  cubre el inventario normalizado (ruta relativa y tamaño) más los SHA-256 de
  manifiestos, checkpoints, supervisores, cuarentenas y veredictos seleccionados.
- Los directorios content-addressed se resumen por cantidad de objetos, bytes,
  extensiones e inventario; sus blobs no se hashean ni se copian individualmente.
- De JSON sólo se extraen IDs, versiones, commits, timestamps, estados,
  veredictos, autoridad y gates. Se excluyen DSN, credenciales, tokens, secretos,
  prompts, contexto y payloads incluso dentro de gates.
- `invalid` identifica JSON de evidencia ilegible; `incomplete` carece de cierre;
  `stale_running` tiene un heartbeat con más de seis horas al final del snapshot;
  y toda cuarentena se representa como `failed_closed`.

## Estado de seguridad relevante

`neural-nightly-20260807-d2d54a64` terminó `failed_closed` porque
`dirty_worktree_blocks_unattended_campaign`. Sólo existen `SUPERVISOR.json` y
`QUARANTINE.json`: no hubo entrenamiento, staging ni promoción, y los flags de
entrenamiento y promoción permanecen en `false`.

Codex/frontier permanece como docente de referencia. El 7B local continúa como
alumno/proponente supervisado sin autoridad de decisión, promoción o actuación.
