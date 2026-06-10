import asyncio
from collections import deque

import pytest

# Soporte del orquestador legacy (cuarentena): adaptation_controller → core/infrastructure → pydantic.
pytest.importorskip("pydantic")

from runtime.control.adaptation_controller import AdaptationController
from runtime.core.metrics import SelfAwarenessMetrics


class _DummyMutator:
    def step(self, context):
        assert "thermal_risk" in context
        return [{"action": "prune", "level": 0.2}]


class _DummyTrainer:
    def __init__(self):
        self.optimizer = {"name": "new_optimizer"}
        self.applied = []

    def apply_adaptation(self, payload):
        self.applied.append(payload)


def test_adaptation_context_has_expected_fields():
    controller = AdaptationController()
    metrics = SelfAwarenessMetrics(
        vram_usage_gb=2.0,
        temperature=45.0,
        entropy=0.4,
        stability=0.2,
    )
    history = deque([0.4, 0.1], maxlen=10)
    context = controller.build_context(
        metrics=metrics,
        history=list(history),
        model=object(),
        optimizer={"lr": 1e-4},
        device="cpu",
    )
    assert context["delta_epist"] == history[-1] - history[-2]
    assert context["vram_usage_gb"] == 2.0
    assert "thermal_risk" in context


def test_apply_adaptations_updates_optimizer():
    controller = AdaptationController()
    trainer = _DummyTrainer()
    mutator = _DummyMutator()
    loop = asyncio.new_event_loop()
    try:
        optimizer = loop.run_until_complete(
            controller.apply_adaptations(
                loop=loop,
                executor=None,
                auto_mutator=mutator,
                trainer=trainer,
                optimizer={"name": "old_optimizer"},
                context={"delta_epist": 0.1, "thermal_risk": 0.2},
                logger=type("L", (), {"info": lambda *a, **kw: None})(),
            )
        )
    finally:
        loop.close()
    assert trainer.applied and trainer.applied[0]["action"] == "prune"
    assert optimizer == trainer.optimizer
