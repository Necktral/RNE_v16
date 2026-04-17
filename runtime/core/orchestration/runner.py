"""Runner del runtime: lifecycle, tareas y shutdown."""

from __future__ import annotations

import asyncio
import tracemalloc

from .lifecycle import LifecycleState


class RuntimeRunner:
    def __init__(self, *, orchestrator, lifecycle, training_loop, crisis_router):
        self.orchestrator = orchestrator
        self.lifecycle = lifecycle
        self.training_loop = training_loop
        self.crisis_router = crisis_router

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
        orch.executor.shutdown()
        self.lifecycle.mark_stopped()
        orch.logger.info("[DEBUG] runner.run_forever finalizado")
