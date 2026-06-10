# 16 — `archive/` (cuarentena histórica legacy)

~14.5K LOC, 124 archivos. **Cuarentena técnica bien gobernada**, NO es código vivo.

## Política (archive/README.md + marcador `.READ_ONLY`)
> `archive/` es cuarentena técnica: no tiene poder gobernante, **no se importa desde `runtime/` ni
> `exocortex/`**, solo trazabilidad histórica; rescatar algo exige `PROMOTE`/`REWRITE`.

Verificado: **ningún módulo vivo importa de `archive`** (grep vacío). Hay un fichero `.READ_ONLY`.

## Contenido — tres generaciones legacy
- `archive/deprecated/` (checkpoints, docs, root, utils_legacy).
- `archive/legacy_08-08-25/` — **snapshot completo fechado (2025-08-08)** del sistema antiguo:
  `components/configs/core/data/models/orchestrator/scripts/systems/utils/vitals`.
- `archive/legacy_src2/` — la generación "src2": `cognition/core/data/episteme/evolution/
  homeostasis/optim/persistence/utils`.

## Hallazgos
- **[DISEÑO/MUERTO] Duplica versiones previas de módulos hoy vivos.** Los archivos más grandes son
  copias antiguas de código actual: `episteme.py` (423 LOC, aparece **3 veces**), `life_monitor`
  (377, ×2), `meta_optimizer` (322, ×2), `loss.py` (317), `neurogenesis` (275, ×2),
  `cognitive_self_challenge` (274). Es la **historia de la arquitectura en el árbol**: confirma que
  `runtime/` evolucionó desde estas generaciones (el `runtime/core` legacy es el puente
  superviviente). Redundante con el historial git, pero conservado a propósito.
- **[POSITIVO] Aislamiento correcto**: política explícita + `.READ_ONLY` + cero imports vivos. Es la
  forma correcta de retirar código sin borrarlo.

## Veredicto
Cuarentena legacy **correctamente aislada** (no afecta al sistema vivo). No procede análisis de
corrección: cualquier bug aquí es inerte por política. Su valor es arqueológico — documenta la
transición AEON FENIX (orquestador/episteme/evolution) → arquitectura nueva (organism/reasoning/
reality/world). Recomendación: dado que duplica historia ya en git, podría eliminarse del árbol de
trabajo para reducir 14.5K LOC de ruido, salvo que se quiera referencia offline.
