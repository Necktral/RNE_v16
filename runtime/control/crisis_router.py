"""Router de crisis y eventos críticos del runtime."""

from __future__ import annotations

import asyncio
from typing import Any

from runtime.core.infrastructure import Event


class CrisisRouter:
    def __init__(self, *, logger, bus, global_event_bus, metrics, shutdown_event):
        self.logger = logger
        self.bus = bus
        self.global_event_bus = global_event_bus
        self.metrics = metrics
        self.shutdown_event = shutdown_event

    def wire_global_handlers(self) -> None:
        self.global_event_bus.on("crisis", self._on_crisis_event)
        self.global_event_bus.on("homeostasis_violation", self._on_homeostasis_violation)
        self.global_event_bus.on("quantum_mutation_applied", self._on_quantum_mutation)
        self.global_event_bus.on("module_pruned", self._on_module_pruned)
        self.global_event_bus.on("module_spawned", self._on_module_spawned)

    def wire_bus_handlers(self, adaptation_handler) -> None:
        self.bus.subscribe("VRAMUsageHigh", adaptation_handler)
        self.bus.subscribe("ThermalAlert", adaptation_handler)
        self.bus.subscribe("EntropyMax", adaptation_handler)
        self.bus.subscribe("StabilityLoss", adaptation_handler)

    def _on_crisis_event(self, payload: Any) -> None:
        self.logger.warning(f"[EventBus] Crisis global detectada | Payload: {payload}")

    def _on_homeostasis_violation(self, payload: Any) -> None:
        self.logger.warning(f"[EventBus] Homeostasis violation detectada | Payload: {payload}")

    def _on_quantum_mutation(self, payload: Any) -> None:
        self.logger.info(f"[EventBus] Mutación cuántica aplicada | Payload: {payload}")

    def _on_module_pruned(self, payload: Any) -> None:
        self.logger.info(f"[EventBus] Módulo podado | Payload: {payload}")

    def _on_module_spawned(self, payload: Any) -> None:
        self.logger.info(f"[EventBus] Nuevo módulo creado | Payload: {payload}")

    async def handle_adaptation_pressure(self, event: Event) -> None:
        self.logger.info(
            f"Evento de adaptación recibido: {event.topic} ({event.severity}). "
            "La adaptación proactiva corre en el training loop."
        )

    async def monitor_vitals(self) -> None:
        self.logger.info("[DEBUG] crisis_router.monitor_vitals iniciado")
        while not self.shutdown_event.is_set():
            m = self.metrics.as_dict()
            if m["Mem"] >= 0.95:
                await self.bus.publish(Event("VRAMUsageHigh", m, "CRITICAL"))
            if m["Temp"] >= 0.9:
                await self.bus.publish(Event("ThermalAlert", m, "CRITICAL"))
            if m["Entropy"] >= 0.98:
                await self.bus.publish(Event("EntropyMax", m, "WARN"))
            if m["Stability"] >= 1e3:
                await self.bus.publish(Event("StabilityLoss", m, "CRITICAL"))
            await asyncio.sleep(5)
        self.logger.info("[DEBUG] crisis_router.monitor_vitals finalizado")
