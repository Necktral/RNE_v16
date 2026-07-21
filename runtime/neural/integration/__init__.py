"""Integración simbiótica N0-N6 con el organismo vivo."""

from .census import ComponentClass, integration_census, validate_active_census
from .adapters import CanonicalOrganAdapter, canonical_adapter_registry
from .contracts import (
    CONSUMER_RECEIPT_SCHEMA_VERSION,
    SYMBIOSIS_TRACE_SCHEMA_VERSION,
    AuthorityEffect,
    ConsumerReceipt,
    ConsumerVerdictClass,
    OrganTrace,
    SymbiosisIdentity,
    SymbiosisTrace,
    validate_consumer_receipt,
)
from runtime.neural.connectome import (
    CONNECTOME_ACTIVITY_SCHEMA_VERSION,
    CONNECTOME_SCHEMA_VERSION,
    ConnectomeRuntime,
    canonical_connectome,
)
from .p1_n2 import N2RevisionRequest, N2ShadowRevisionRecord
from .p1_n3 import N3ShadowCounterfactualReport, N3ShadowDirective
from .p1_n4 import (
    N4InterventionScoreSet,
    N4PreactionArtifactV2,
    N4PreactionInterventionSet,
)

__all__ = [
    "ComponentClass",
    "CanonicalOrganAdapter",
    "CONSUMER_RECEIPT_SCHEMA_VERSION",
    "ConsumerReceipt",
    "ConsumerVerdictClass",
    "OrganTrace",
    "AuthorityEffect",
    "SYMBIOSIS_TRACE_SCHEMA_VERSION",
    "SymbiosisIdentity",
    "SymbiosisTrace",
    "SymbioticNeuralCoordinator",
    "canonical_adapter_registry",
    "integration_census",
    "validate_active_census",
    "validate_consumer_receipt",
    "CONNECTOME_ACTIVITY_SCHEMA_VERSION",
    "CONNECTOME_SCHEMA_VERSION",
    "ConnectomeRuntime",
    "canonical_connectome",
    "N2RevisionRequest",
    "N2ShadowRevisionRecord",
    "N3ShadowDirective",
    "N3ShadowCounterfactualReport",
    "N4PreactionInterventionSet",
    "N4PreactionArtifactV2",
    "N4InterventionScoreSet",
    "AdversarialAgent",
    "AgentCycleReport",
    "ConnectomicsAgent",
    "CurriculumLearningAgent",
    "DevelopmentLineageAgent",
    "HorizontalCreativityAgent",
    "LatentCommunicationAgent",
    "MetacognitiveEpistemicAgent",
    "MemoryConsolidationAgent",
    "ModelDataImmuneAgent",
    "MetabolicBudgetAgent",
    "PedagogicalTeacherAgent",
    "NeuralOrchestrationAgent",
    "SensorimotorWorldModelAgent",
    "SocialExocortexAgent",
    "SymbiosisSynergyAgent",
    "SpecializedAgentBundle",
]


def __getattr__(name: str):
    """Evita ciclos entre contratos de integración, coordinador y agentes."""

    if name == "SymbioticNeuralCoordinator":
        from .coordinator import SymbioticNeuralCoordinator

        return SymbioticNeuralCoordinator
    if name in {
        "AdversarialAgent",
        "AgentCycleReport",
        "ConnectomicsAgent",
        "CurriculumLearningAgent",
        "DevelopmentLineageAgent",
        "HorizontalCreativityAgent",
        "LatentCommunicationAgent",
        "MetacognitiveEpistemicAgent",
        "MemoryConsolidationAgent",
        "ModelDataImmuneAgent",
        "MetabolicBudgetAgent",
        "PedagogicalTeacherAgent",
        "NeuralOrchestrationAgent",
        "SensorimotorWorldModelAgent",
        "SocialExocortexAgent",
        "SymbiosisSynergyAgent",
        "SpecializedAgentBundle",
    }:
        from runtime.neural import agents

        return getattr(agents, name)
    raise AttributeError(name)
