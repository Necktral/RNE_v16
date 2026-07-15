"""Agentes de coordinación neural no autoritativos."""

from .adversarial import AdversarialAgent
from .creativity import HorizontalCreativityAgent
from .curriculum import CurriculumLearningAgent
from .connectomics import ConnectomicsAgent
from .contracts import (
    NEURAL_AGENT_CYCLE_SCHEMA_VERSION,
    NEURAL_AGENT_REPORT_SCHEMA_VERSION,
    NEURAL_SPECIALIZED_AGENT_BUNDLE_SCHEMA_VERSION,
    CORE_AGENT_ROLES,
    AgentCycleReport,
    AgentFinding,
    AgentReport,
    AgentRole,
    AgentState,
    FindingSeverity,
    GainModulation,
    SpecializedAgentBundle,
)
from .epistemic import MetacognitiveEpistemicAgent
from .development import DevelopmentLineageAgent
from .latent import LatentCommunicationAgent
from .memory import MemoryConsolidationAgent
from .immune import ModelDataImmuneAgent
from .interoception import InteroceptiveHomeostaticAgent
from .metabolic import MetabolicBudgetAgent
from .pedagogy import PedagogicalTeacherAgent
from .orchestration import NeuralOrchestrationAgent
from .sensorimotor import SensorimotorWorldModelAgent
from .social import SocialExocortexAgent
from .symbiosis import SymbiosisSynergyAgent

__all__ = [
    "AdversarialAgent",
    "AgentCycleReport",
    "AgentFinding",
    "AgentReport",
    "AgentRole",
    "AgentState",
    "ConnectomicsAgent",
    "CurriculumLearningAgent",
    "CORE_AGENT_ROLES",
    "FindingSeverity",
    "DevelopmentLineageAgent",
    "GainModulation",
    "LatentCommunicationAgent",
    "HorizontalCreativityAgent",
    "InteroceptiveHomeostaticAgent",
    "MetacognitiveEpistemicAgent",
    "MemoryConsolidationAgent",
    "ModelDataImmuneAgent",
    "MetabolicBudgetAgent",
    "PedagogicalTeacherAgent",
    "NEURAL_AGENT_CYCLE_SCHEMA_VERSION",
    "NEURAL_AGENT_REPORT_SCHEMA_VERSION",
    "NEURAL_SPECIALIZED_AGENT_BUNDLE_SCHEMA_VERSION",
    "NeuralOrchestrationAgent",
    "SensorimotorWorldModelAgent",
    "SocialExocortexAgent",
    "SpecializedAgentBundle",
    "SymbiosisSynergyAgent",
]
