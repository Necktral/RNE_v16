"""Capa constitucional del organismo RNFE.

Reúne estado, constitución, viabilidad, régimen, transporte,
riesgo, auto-modificación y lineage bajo un modelo unificado
del organismo como entidad persistente.
"""

from .t4_mode import T4Mode, get_t4_mode, is_t4_enabled, is_t4_primary
from .t5_mode import T5Mode, get_t5_mode, is_t5_enabled, is_t5_primary
from .snapshot import OrganismSnapshot
from .trajectory import (
    BeliefHistory,
    OrganismTrajectory,
    PolicyHistory,
    TrajectoryDigest,
    TrajectoryInvariantReport,
    TrajectoryWindow,
    ViabilityHistory,
)
from .trajectory_state_machine import TrajectoryStateMachine
from .constitution_flow import ConstitutionalFlowEngine, ConstitutionalFlowResult
from .court_runtime import ConstitutionalCourtRuntime, CourtEpisodeResult
from .regime_renormalization import (
    BeliefProjectionField,
    ConstraintTransform,
    PolicyPhaseTransform,
    RegimeRenormalizationEngine,
    RegimeResidual,
    RenormalizationMap,
    RenormalizationResult,
    RenormalizationUncertainty,
)
from .failure_atlas import FailureAtlas, FailureClass, FailureSignature, detect_failure_atlas
from .risk_process import (
    ConstitutionalRiskProcess,
    EdgeRiskProfile,
    InheritanceRiskProfile,
    ModificationRiskProfile,
    RiskState,
    RiskUpdate,
)
from .viability_kernel import TrajectoryViabilityAssessment, TrajectoryViabilityKernel

__all__ = [
    "BeliefHistory",
    "BeliefProjectionField",
    "ConstitutionalFlowEngine",
    "ConstitutionalFlowResult",
    "ConstitutionalCourtRuntime",
    "ConstitutionalRiskProcess",
    "CourtEpisodeResult",
    "ConstraintTransform",
    "EdgeRiskProfile",
    "FailureAtlas",
    "FailureClass",
    "FailureSignature",
    "InheritanceRiskProfile",
    "ModificationRiskProfile",
    "OrganismSnapshot",
    "OrganismTrajectory",
    "PolicyHistory",
    "PolicyPhaseTransform",
    "RegimeRenormalizationEngine",
    "RegimeResidual",
    "RenormalizationMap",
    "RenormalizationResult",
    "RenormalizationUncertainty",
    "RiskState",
    "RiskUpdate",
    "T4Mode",
    "T5Mode",
    "TrajectoryDigest",
    "TrajectoryInvariantReport",
    "TrajectoryStateMachine",
    "TrajectoryViabilityAssessment",
    "TrajectoryViabilityKernel",
    "TrajectoryWindow",
    "ViabilityHistory",
    "detect_failure_atlas",
    "get_t4_mode",
    "get_t5_mode",
    "is_t4_enabled",
    "is_t4_primary",
    "is_t5_enabled",
    "is_t5_primary",
]
