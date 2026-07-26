"""Módulo de Cognición Integrada."""

from .compiler import TransitionCompiler
from .causal_learning import CausalLearningEngine, TransitionEvidence
from .contracts import (
    BeliefNode,
    CausalOverlay,
    MCICommitReport,
    SMTPlanReport,
    SelfModelReport,
    TransitionSpec,
    Justification,
)
from .jtms import TemporalAssumptionLedger
from .planner import SMTPlanner
from .runtime import MCIRuntime
from .regime_detector import RegimeChangeDetected, RegimeDetector
from .self_model import IncrementalSelfModel
from .specs import deferred_load_spec, resource_spec, thermal_spec

__all__ = [
    "BeliefNode",
    "CausalOverlay",
    "CausalLearningEngine",
    "IncrementalSelfModel",
    "Justification",
    "MCICommitReport",
    "MCIRuntime",
    "RegimeChangeDetected",
    "RegimeDetector",
    "SMTPlanReport",
    "SelfModelReport",
    "SMTPlanner",
    "TemporalAssumptionLedger",
    "TransitionEvidence",
    "TransitionCompiler",
    "TransitionSpec",
    "deferred_load_spec",
    "resource_spec",
    "thermal_spec",
]
