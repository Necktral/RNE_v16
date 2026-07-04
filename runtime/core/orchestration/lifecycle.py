"""Máquina de estados del ciclo de vida del runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class LifecycleState(str, Enum):
    INIT = "INIT"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    DEGRADED = "DEGRADED"
    SHUTTING_DOWN = "SHUTTING_DOWN"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


_ALLOWED = {
    LifecycleState.INIT: {LifecycleState.STARTING, LifecycleState.FAILED},
    LifecycleState.STARTING: {
        LifecycleState.RUNNING,
        LifecycleState.DEGRADED,
        LifecycleState.FAILED,
    },
    LifecycleState.RUNNING: {
        LifecycleState.DEGRADED,
        LifecycleState.SHUTTING_DOWN,
        LifecycleState.FAILED,
    },
    LifecycleState.DEGRADED: {
        LifecycleState.RUNNING,
        LifecycleState.SHUTTING_DOWN,
        LifecycleState.FAILED,
    },
    LifecycleState.SHUTTING_DOWN: {LifecycleState.STOPPED, LifecycleState.FAILED},
    LifecycleState.STOPPED: set(),
    LifecycleState.FAILED: set(),
}


@dataclass
class OrchestratorLifecycle:
    state: LifecycleState = LifecycleState.INIT
    history: List[LifecycleState] = field(default_factory=lambda: [LifecycleState.INIT])

    def transition(self, new_state: LifecycleState) -> None:
        if new_state not in _ALLOWED[self.state]:
            raise ValueError(f"Transición inválida: {self.state} -> {new_state}")
        self.state = new_state
        self.history.append(new_state)

    def mark_degraded(self) -> None:
        if self.state in {LifecycleState.RUNNING, LifecycleState.STARTING}:
            self.transition(LifecycleState.DEGRADED)

    def begin_shutdown(self) -> None:
        if self.state in {LifecycleState.RUNNING, LifecycleState.DEGRADED}:
            self.transition(LifecycleState.SHUTTING_DOWN)

    def mark_stopped(self) -> None:
        if self.state == LifecycleState.SHUTTING_DOWN:
            self.transition(LifecycleState.STOPPED)

    def mark_failed(self) -> None:
        if self.state != LifecycleState.FAILED:
            if self.state in {
                LifecycleState.INIT,
                LifecycleState.STARTING,
                LifecycleState.RUNNING,
                LifecycleState.DEGRADED,
                LifecycleState.SHUTTING_DOWN,
            }:
                self.state = LifecycleState.FAILED
                self.history.append(LifecycleState.FAILED)
