"""Runner del runtime: lifecycle, tareas y shutdown."""

from __future__ import annotations

import asyncio
import tracemalloc
from typing import Optional

from .lifecycle import LifecycleState


class RuntimeRunner:
    def __init__(
        self,
        *,
        orchestrator,
        lifecycle,
        training_loop,
        crisis_router,
        reality_hook=None,
    ):
        self.orchestrator = orchestrator
        self.lifecycle = lifecycle
        self.training_loop = training_loop
        self.crisis_router = crisis_router
        self.reality_hook = reality_hook

    async def run_forever(self) -> None:
        orch = self.orchestrator
        tracemalloc.start()
        orch.logger.info("[DEBUG] runner.run_forever iniciado")
        self.lifecycle.transition(LifecycleState.STARTING)
        await orch.bus.start()
        self.crisis_router.wire_bus_handlers(self.crisis_router.handle_adaptation_pressure)
        self.lifecycle.transition(LifecycleState.RUNNING)

        orch._tasks += [
            asyncio.create_task(self.training_loop.run()),
            asyncio.create_task(self.crisis_router.monitor_vitals()),
        ]

        await orch._shutdown.wait()
        self.lifecycle.begin_shutdown()
        orch.logger.info("[DEBUG] runner: shutdown recibido, cancelando tareas...")
        for task in orch._tasks:
            task.cancel()
        await asyncio.gather(*orch._tasks, return_exceptions=True)
        
        # Ejecutar validación de realidad si el hook está configurado
        if self.reality_hook is not None:
            orch.logger.info("[DEBUG] runner: ejecutando reality validation hook...")
            try:
                run_id = getattr(orch, "run_id", None)
                validation_result = self.reality_hook.on_shutdown(run_id=run_id)
                if validation_result and not validation_result.get("passed", True):
                    orch.logger.warning(
                        f"[REALITY] Validación falló: {validation_result.get('summary', {})}"
                    )
                else:
                    orch.logger.info("[REALITY] Validación completada exitosamente")
            except Exception as exc:
                orch.logger.error(f"[REALITY] Error en validación: {exc}", exc_info=True)
        
        orch.executor.shutdown()
        self.lifecycle.mark_stopped()
        orch.logger.info("[DEBUG] runner.run_forever finalizado")
