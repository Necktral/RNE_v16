"""Runtime de self modification T4.

Adaptador compatible hacia pipeline legacy.
"""

from __future__ import annotations

from dataclasses import dataclass

from .self_modification import SelfModificationPipeline


@dataclass
class SelfModificationRuntime:
    pipeline: SelfModificationPipeline

    def evaluate(self, *args, **kwargs):
        return self.pipeline.evaluate_proposal(*args, **kwargs)
