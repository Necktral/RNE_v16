import asyncio
from types import SimpleNamespace

import pytest

# Nota reorg 2026-07-01: ya no requiere pydantic — `Event` vive en
# runtime/core/events.py (stdlib) y crisis_router no toca la cadena legacy.

from runtime.control.crisis_router import CrisisRouter
from runtime.core.events import Event
from runtime.core.orchestration import LifecycleState, OrchestratorLifecycle, RuntimeRunner


class _DummyLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))


class _DummyBus:
    def __init__(self):
        self.subscribers = {}
        self.events = []
        self.started = False

    async def start(self):
        self.started = True

    def subscribe(self, topic, handler):
        self.subscribers.setdefault(topic, []).append(handler)

    async def publish(self, event):
        self.events.append(event.topic)
        for handler in self.subscribers.get(event.topic, []):
            await handler(event)


class _DummyEventBus:
    def __init__(self):
        self.handlers = {}

    def on(self, topic, callback):
        self.handlers.setdefault(topic, []).append(callback)


class _DummyMetrics:
    def as_dict(self):
        return {"Mem": 0.1, "Temp": 0.1, "Entropy": 0.1, "Stability": 0.1}


class _DummyExecutor:
    def __init__(self):
        self.closed = False

    def shutdown(self):
        self.closed = True


class _DummyTrainingLoop:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    async def run(self):
        await self.orchestrator.bus.publish(
            Event(topic="VRAMUsageHigh", payload={"test": True}, severity="CRITICAL")
        )
        self.orchestrator._shutdown.set()


def test_runner_start_loop_shutdown_sequence():
    shutdown_event = asyncio.Event()
    logger = _DummyLogger()
    bus = _DummyBus()
    global_event_bus = _DummyEventBus()
    metrics = _DummyMetrics()
    lifecycle = OrchestratorLifecycle()

    class _Router(CrisisRouter):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.adaptation_hits = 0

        async def handle_adaptation_pressure(self, event):
            self.adaptation_hits += 1

        async def monitor_vitals(self):
            while not self.shutdown_event.is_set():
                await asyncio.sleep(0.01)

    orchestrator = SimpleNamespace(
        logger=logger,
        bus=bus,
        event_bus=global_event_bus,
        metrics=metrics,
        _shutdown=shutdown_event,
        _tasks=[],
        executor=_DummyExecutor(),
    )
    router = _Router(
        logger=logger,
        bus=bus,
        global_event_bus=global_event_bus,
        metrics=metrics,
        shutdown_event=shutdown_event,
    )
    router.wire_global_handlers()
    training_loop = _DummyTrainingLoop(orchestrator)
    runner = RuntimeRunner(
        orchestrator=orchestrator,
        lifecycle=lifecycle,
        training_loop=training_loop,
        crisis_router=router,
    )

    asyncio.run(runner.run_forever())

    assert lifecycle.state == LifecycleState.STOPPED
    assert orchestrator.executor.closed is True
    assert router.adaptation_hits == 1
