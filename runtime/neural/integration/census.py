"""Censo ejecutable de callers, consumidores y perfiles simbióticos."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ComponentClass(str, Enum):
    LIVE = "LIVE"
    SHADOW_CONSUMED = "SHADOW_CONSUMED"
    REFERENCE_ONLY = "REFERENCE_ONLY"
    STUB = "STUB"
    DEAD = "DEAD"


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    organ: str
    classification: ComponentClass
    callers: tuple[str, ...]
    consumers: tuple[str, ...]
    active_profiles: tuple[str, ...] = ()
    reference_only: bool = False
    stub_detected: bool = False
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "organ": self.organ,
            "classification": self.classification.value,
            "caller_count": len(self.callers),
            "consumer_count": len(self.consumers),
            "active_profile_count": len(self.active_profiles),
            "reference_only": self.reference_only,
            "live": self.classification is ComponentClass.LIVE,
            "shadow_consumed": self.classification is ComponentClass.SHADOW_CONSUMED,
            "stub_detected": self.stub_detected,
            "callers": list(self.callers),
            "consumers": list(self.consumers),
            "evidence": self.evidence,
        }


COMPONENT_SPECS: tuple[ComponentSpec, ...] = (
    ComponentSpec("N0", ComponentClass.LIVE, ("ScenarioEpisodeRunner",), ("N1-N6",), ("symbiotic",), evidence="NeuralRuntime.infer_reference"),
    ComponentSpec("AGENT-ORCHESTRATION", ComponentClass.LIVE, ("SymbioticNeuralCoordinator",), ("connectomics", "adversarial", "latent communication", "symbiosis", "certification metadata"), ("symbiotic",), evidence="NeuralOrchestrationAgent.run_cycle"),
    ComponentSpec("AGENT-CONNECTOMICS", ComponentClass.SHADOW_CONSUMED, ("NeuralOrchestrationAgent",), ("orchestration", "symbiosis"), ("symbiotic",), evidence="typed topology/activity audit; graph_mutated=false"),
    ComponentSpec("AGENT-LATENT", ComponentClass.SHADOW_CONSUMED, ("NeuralOrchestrationAgent",), ("symbiosis", "certification metadata"), ("symbiotic",), evidence="bounded gain proposal; apply_authorized=false"),
    ComponentSpec("AGENT-ADVERSARIAL", ComponentClass.LIVE, ("NeuralOrchestrationAgent",), ("latent quarantine", "symbiosis", "certification metadata"), ("symbiotic",), evidence="identity/hash/authority fail-closed inspection"),
    ComponentSpec("AGENT-SYMBIOSIS", ComponentClass.SHADOW_CONSUMED, ("NeuralOrchestrationAgent",), ("orchestration", "certification metadata"), ("symbiotic",), evidence="receipt-backed connectivity coverage"),
    ComponentSpec("AGENT-EPISTEMIC", ComponentClass.SHADOW_CONSUMED, ("SymbioticNeuralCoordinator.prepare_certification",), ("specialized agent bundle", "certification metadata", "future META/teacher policy"), ("symbiotic",), evidence="PROB/CAU/CTF/sequence measurements; authority none; gain unmeasured pre-outcome"),
    ComponentSpec("AGENT-MEMORY", ComponentClass.SHADOW_CONSUMED, ("SymbioticNeuralCoordinator.prepare_certification",), ("specialized agent bundle", "certification metadata", "future longitudinal learning"), ("symbiotic",), evidence="N3/N5/MFM provenance audit; no direct write or promotion"),
    ComponentSpec("AGENT-IMMUNE", ComponentClass.SHADOW_CONSUMED, ("SymbioticNeuralCoordinator.prepare_certification",), ("specialized agent bundle", "training evidence gate", "artifact quarantine proposal"), ("symbiotic",), evidence="artifact/manifest, receipt and persistence integrity audit; N0 authority preserved"),
    ComponentSpec("AGENT-PEDAGOGY", ComponentClass.SHADOW_CONSUMED, ("SymbioticNeuralCoordinator.finalize_episode",), ("final symbiosis trace", "future curriculum gate"), ("symbiotic",), evidence="lesson-to-bias-to-severity comparison; teacher authority none"),
    ComponentSpec("AGENT-CURRICULUM", ComponentClass.SHADOW_CONSUMED, ("SymbioticNeuralCoordinator.finalize_episode",), ("teacher efficiency report", "future offline training gate"), ("symbiotic",), evidence="paired no-teacher/local-7B/Codex protocol; no training or promotion authority"),
    ComponentSpec("AGENT-SENSORIMOTOR", ComponentClass.SHADOW_CONSUMED, ("SymbioticNeuralCoordinator.prepare_certification",), ("specialized agent bundle", "future post-outcome prediction audit"), ("symbiotic",), evidence="observation/causal attestation/committed N4 binding; no actuation authority"),
    ComponentSpec("AGENT-METABOLIC", ComponentClass.SHADOW_CONSUMED, ("SymbioticNeuralCoordinator.prepare_certification",), ("specialized agent bundle", "N0 budget proposal"), ("symbiotic",), evidence="MSRC and physical pressure observation; no budget mutation"),
    ComponentSpec("AGENT-DEVELOPMENT", ComponentClass.SHADOW_CONSUMED, ("SymbioticNeuralCoordinator.prepare_certification",), ("specialized agent bundle", "N6 quarantine proposal"), ("symbiotic",), evidence="lineage and rollback-token audit; no mutation or rollback authority"),
    ComponentSpec("AGENT-CREATIVITY", ComponentClass.SHADOW_CONSUMED, ("SymbioticNeuralCoordinator.prepare_certification",), ("specialized agent bundle", "future shadow alternative tests"), ("symbiotic",), evidence="reasoning-family breadth; Mamba2 not required; no selection authority"),
    ComponentSpec("AGENT-SOCIAL", ComponentClass.SHADOW_CONSUMED, ("SymbioticNeuralCoordinator.prepare_certification",), ("specialized agent bundle", "external evidence quarantine"), ("symbiotic",), evidence="source/hash provenance boundary; no external write or decision authority"),
    ComponentSpec("N1", ComponentClass.SHADOW_CONSUMED, ("SymbioticNeuralCoordinator",), ("scheduler comparison", "experience evidence"), ("symbiotic",), evidence="proposal compared with validated_sequence"),
    ComponentSpec("N2", ComponentClass.SHADOW_CONSUMED, ("SymbioticNeuralCoordinator",), ("DED", "LOT-F", "NESY", "certification"), ("symbiotic",), evidence="semantic candidate verification"),
    ComponentSpec("N3", ComponentClass.LIVE, ("SymbioticNeuralCoordinator",), ("next episode reasoning", "MFM candidate", "continuity audit"), ("symbiotic",), evidence="state keyed by organism+scenario+lineage"),
    ComponentSpec("N4", ComponentClass.SHADOW_CONSUMED, ("SymbioticNeuralCoordinator",), ("CAU/CTF/C-GWM comparison", "certification", "experience evidence"), ("symbiotic",), evidence="typed proposal never selects action"),
    ComponentSpec("N5", ComponentClass.LIVE, ("SymbioticNeuralCoordinator",), ("SMG", "MFM", "reasoning context"), ("symbiotic",), evidence="deterministic chunks with offsets"),
    ComponentSpec("N6", ComponentClass.SHADOW_CONSUMED, ("SymbioticNeuralCoordinator",), ("sandbox evaluator", "certification", "autoevolution evidence"), ("symbiotic",), evidence="versioned proposal; no apply_fn"),
    ComponentSpec("NESY", ComponentClass.SHADOW_CONSUMED, ("N2 verification", "MetaScheduler deep profile"), ("N2 trace", "PROB"), ("symbiotic-verifier",), evidence="deterministic symbol-number coherence"),
    ComponentSpec("EVO_SEARCH", ComponentClass.REFERENCE_ONLY, ("MetaScheduler deep profile",), ("PROB",), reference_only=True, evidence="deterministic GA; inactive in baseline profile"),
    ComponentSpec("IMAGINATION/A11", ComponentClass.REFERENCE_ONLY, ("MetaScheduler deep profile",), ("A12", "PROB"), reference_only=True, evidence="deterministic advisory world model; inactive baseline"),
    ComponentSpec("A12", ComponentClass.REFERENCE_ONLY, ("MetaScheduler deep profile", "ScenarioEpisodeRunner gated recomputation"), ("intervention override guard",), reference_only=True, evidence="inactive without explicit deep/actuation gates"),
    ComponentSpec("CAU", ComponentClass.LIVE, ("MetaScheduler",), ("CTF", "PROB", "N4 comparator"), ("baseline", "symbiotic"), evidence="core causal inference family"),
    ComponentSpec("CTF", ComponentClass.LIVE, ("MetaScheduler",), ("PROB", "N4 comparator"), ("baseline", "symbiotic"), evidence="counterfactual family"),
    ComponentSpec("C-GWM", ComponentClass.LIVE, ("ScenarioEpisodeRunner",), ("world transition", "N4 comparator"), ("baseline", "symbiotic"), evidence="scenario simulation and factual transition"),
    ComponentSpec("DED", ComponentClass.LIVE, ("MetaScheduler",), ("PROB", "N2 verifier"), ("baseline", "symbiotic"), evidence="LOT-F boolean solver"),
    ComponentSpec("LOT-F", ComponentClass.LIVE, ("ScenarioEpisodeRunner",), ("DED", "N2 verifier", "certification"), ("baseline", "symbiotic"), evidence="parsed and type-checked formula"),
    ComponentSpec("MFM", ComponentClass.LIVE, ("ScenarioEpisodeRunner", "PromotionGate"), ("reasoning", "next episode", "N1", "N3"), ("baseline", "symbiotic"), evidence="retrieval and gated condensation"),
    ComponentSpec("SMG", ComponentClass.LIVE, ("ScenarioEpisodeRunner",), ("artifact", "certification", "MFM"), ("baseline", "symbiotic"), evidence="observations, signs and relations"),
    ComponentSpec("scheduler", ComponentClass.LIVE, ("ScenarioEpisodeRunner",), ("world override guard", "certification", "N1 comparator"), ("baseline", "symbiotic"), evidence="authoritative family sequence"),
    ComponentSpec("certification", ComponentClass.LIVE, ("ScenarioEpisodeRunner",), ("memory promotion", "experience", "autoevolution"), ("baseline", "symbiotic"), evidence="PromotionGate"),
    ComponentSpec("experience", ComponentClass.LIVE, ("ScenarioEpisodeRunner",), ("future intervention bias",), ("experience-enabled",), evidence="ExperienceStore recall and wisdom"),
    ComponentSpec("autoevolution", ComponentClass.LIVE, ("ScenarioEpisodeRunner",), ("lineage", "future knobs"), ("baseline",), evidence="AutoEvolutionController"),
)


def integration_census() -> list[dict[str, Any]]:
    return [spec.to_dict() for spec in COMPONENT_SPECS]


def validate_active_census() -> list[str]:
    failures = []
    for row in integration_census():
        if row["active_profile_count"] > 0 and (
            row["caller_count"] == 0
            or row["consumer_count"] == 0
            or row["stub_detected"]
        ):
            failures.append(str(row["organ"]))
    return failures
