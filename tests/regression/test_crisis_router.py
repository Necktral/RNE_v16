import asyncio

from runtime.control.crisis_router import CrisisRouter
from runtime.core.infrastructure import Event


class _DummyLogger:
    def __init__(self):
        self.logs = []

    def info(self, msg):
        self.logs.append(("info", msg))

    def warning(self, msg):
        self.logs.append(("warning", msg))


class _DummyBus:
    def __init__(self):
        self.subscribers = {}
        self.events = []

    def subscribe(self, topic, handler):
        self.subscribers.setdefault(topic, []).append(handler)

    async def publish(self, event):
        self.events.append(event.topic)
        for handler in self.subscribers.get(event.topic, []):
            await handler(event)


class _DummyGlobalEventBus:
    def __init__(self):
        self.handlers = {}

    def on(self, topic, callback):
        self.handlers.setdefault(topic, []).append(callback)


class _HotMetrics:
    def as_dict(self):
        return {
            "Mem": 0.99,
            "Temp": 0.95,
            "Entropy": 0.99,
            "Stability": 2000,
        }


def test_crisis_router_wires_handlers():
    logger = _DummyLogger()
    bus = _DummyBus()
    global_bus = _DummyGlobalEventBus()
    shutdown = asyncio.Event()

    router = CrisisRouter(
        logger=logger,
        bus=bus,
        global_event_bus=global_bus,
        metrics=_HotMetrics(),
        shutdown_event=shutdown,
    )
    router.wire_global_handlers()
    router.wire_bus_handlers(router.handle_adaptation_pressure)

    assert "crisis" in global_bus.handlers
    assert "VRAMUsageHigh" in bus.subscribers


def test_crisis_router_dispatches_adaptation_handler():
    logger = _DummyLogger()
    bus = _DummyBus()
    global_bus = _DummyGlobalEventBus()
    shutdown = asyncio.Event()

    class _TestRouter(CrisisRouter):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.handled = 0

        async def handle_adaptation_pressure(self, event: Event):
            self.handled += 1

    router = _TestRouter(
        logger=logger,
        bus=bus,
        global_event_bus=global_bus,
        metrics=_HotMetrics(),
        shutdown_event=shutdown,
    )
    router.wire_bus_handlers(router.handle_adaptation_pressure)

    asyncio.run(bus.publish(Event(topic="VRAMUsageHigh", payload={}, severity="CRITICAL")))
    assert router.handled == 1
