import pytest

from runtime.core.orchestration.lifecycle import LifecycleState, OrchestratorLifecycle


def test_lifecycle_happy_path():
    lifecycle = OrchestratorLifecycle()
    lifecycle.transition(LifecycleState.STARTING)
    lifecycle.transition(LifecycleState.RUNNING)
    lifecycle.begin_shutdown()
    lifecycle.mark_stopped()

    assert lifecycle.state == LifecycleState.STOPPED
    assert lifecycle.history == [
        LifecycleState.INIT,
        LifecycleState.STARTING,
        LifecycleState.RUNNING,
        LifecycleState.SHUTTING_DOWN,
        LifecycleState.STOPPED,
    ]


def test_lifecycle_invalid_transition_raises():
    lifecycle = OrchestratorLifecycle()
    with pytest.raises(ValueError):
        lifecycle.transition(LifecycleState.RUNNING)


def test_lifecycle_degraded_transition():
    lifecycle = OrchestratorLifecycle()
    lifecycle.transition(LifecycleState.STARTING)
    lifecycle.mark_degraded()
    assert lifecycle.state == LifecycleState.DEGRADED
