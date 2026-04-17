"""Engine EML-SR para descubrimiento simbólico en shadow mode."""

from .advisory import AdvisoryDecision, evaluate_advisory_promotion
from .benchmark import advisory_from_run, compare_baseline_vs_shadow
from .runner import EMLRunner, EMLRunnerConfig
from .tree import ExprNode

__all__ = [
    "AdvisoryDecision",
    "EMLRunner",
    "EMLRunnerConfig",
    "ExprNode",
    "advisory_from_run",
    "compare_baseline_vs_shadow",
    "evaluate_advisory_promotion",
]
