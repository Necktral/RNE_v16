"""Subsistema MSRC (Multi-Scale Resolution Controller)."""

from .contracts import (
    CrossScaleMemoryReport,
    ProbeResult,
    ScaleAction,
    ScaleActionType,
    ScaleDecisionRecord,
    ScaleEstimate,
    ScalePolicyState,
    ScaleSpec,
    ScaleTransitionRecord,
)
from .controller import MSRCController
from .cross_scale_memory_guard import CrossScaleMemoryGuard
from .scale_audit_logger import ScaleAuditLogger
from .scale_catalog import ScaleCatalog
from .scale_estimator import ScaleEstimator
from .scale_policy_engine import ScalePolicyEngine
from .regime_classifier import RegimeClassification, RegimeClassifier
from .scale_transition_manager import ScaleTransitionManager
from .vram_sampler import FixedVRAMSampler, NvidiaVRAMSampler, NullVRAMSampler

__all__ = [
    "CrossScaleMemoryReport",
    "ProbeResult",
    "ScaleAction",
    "ScaleActionType",
    "ScaleDecisionRecord",
    "ScaleEstimate",
    "ScalePolicyState",
    "ScaleSpec",
    "ScaleTransitionRecord",
    "MSRCController",
    "CrossScaleMemoryGuard",
    "ScaleAuditLogger",
    "ScaleCatalog",
    "ScaleEstimator",
    "ScalePolicyEngine",
    "RegimeClassification",
    "RegimeClassifier",
    "ScaleTransitionManager",
    "FixedVRAMSampler",
    "NvidiaVRAMSampler",
    "NullVRAMSampler",
]
