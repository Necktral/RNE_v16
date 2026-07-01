# runtime/legacy — cuarentena del stack "AEON FENIX-Δ"

Relocación física (2026-07-01) de la cuarentena lógica documentada en
`docs/analysis/LEGACY_QUARANTINE.md`. Reglas (mismas que `archive/`):

- **No se importa desde código vivo** (`runtime/` fuera de este directorio,
  `scripts/`, `contracts/`, `engines/`). El único consumidor permitido es el
  CLI histórico `exocortex/channels/cli/aeon_main_loop.py`.
- No tiene poder gobernante; se conserva por trazabilidad y reversibilidad.

Contenido: orquestador legacy (`module_orchestrator.py`), infraestructura
async (`infrastructure.py` — EventBus/WorkerPool/ConfigLoader; el dataclass
`Event` se extrajo a `runtime/core/events.py` porque sí lo usa código vivo),
stack de entrenamiento (`training_loop.py`, `train.py`, `loss.py`,
`loss_elite.py`), sensores/planificación legacy (`episteme.py`, `planner.py`,
`probabilistic_models.py`) y `homeo_controller.py` (variante core).
