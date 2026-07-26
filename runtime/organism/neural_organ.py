"""Persistent bounded neural organ and causal integration bus."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from runtime.neural.integration.contracts import canonical_sha256
from runtime.neural.integration.n3_scoring import n3_scale_multiplier

SCHEMA_VERSION = "neural-organ-snapshot-v1"
PLASTICITY_VERSION = "bounded-online-plasticity-v1"
MODES = {"off", "observe", "modulate", "bounded_actuation"}


def _bounded(value: float, low: float = -1.0, high: float = 1.0) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("neural_nonfinite")
    return min(high, max(low, value))


def _items(value: Mapping[str, Any] | None) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted((str(k), v) for k, v in (value or {}).items()))


@dataclass(frozen=True, slots=True)
class NeuralPercept:
    unit_id: str
    episode_index: int
    scenario: str
    features: tuple[tuple[str, Any], ...]
    alarm: bool
    allowed_interventions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NeuralBodyState:
    viability_margin: float = 1.0
    continuity_score: float = 1.0
    risk_score: float = 0.0
    ioc_proxy: float = 0.0
    identity_continuity: float = 1.0
    current_mode: str = "normal"
    reversible: bool = True


@dataclass(frozen=True, slots=True)
class NeuralMemoryState:
    retrieved_count: int = 0
    scale_counts: tuple[tuple[str, int], ...] = ()
    wound_count: int = 0
    lesson_count: int = 0


@dataclass(frozen=True, slots=True)
class NeuralReasoningState:
    uncertainty: float = 0.0
    contradiction: float = 0.0
    budget: int = 6
    previous_sequence: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class NeuralResourceState:
    cpu_pressure: float = 0.0
    memory_pressure: float = 0.0
    vram_pressure: float = 0.0
    thermal_pressure: float = 0.0
    gpu_available: bool = False


@dataclass(frozen=True, slots=True)
class NeuralGovernanceState:
    accepted: bool = True
    blockers: tuple[str, ...] = ()
    compute_tier: str = "tier_1"
    autonomy_scope: str = "bounded"


@dataclass(frozen=True, slots=True)
class NeuralIntegratedState:
    unit_id: str
    episode_index: int
    world_features: tuple[tuple[str, Any], ...]
    body_features: tuple[tuple[str, Any], ...]
    memory_features: tuple[tuple[str, Any], ...]
    reasoning_features: tuple[tuple[str, Any], ...]
    resource_features: tuple[tuple[str, Any], ...]
    governance_features: tuple[tuple[str, Any], ...]
    previous_neural_state_hash: str


@dataclass(frozen=True, slots=True)
class NeuralProposal:
    unit_id: str
    memory_scale_weights: tuple[tuple[str, float], ...]
    memory_candidate_modifiers: tuple[tuple[str, float], ...]
    family_priority_modifiers: tuple[tuple[str, float], ...]
    reasoning_budget_modifier: int
    intervention_prior: str | None
    compute_tier_advisory: str
    recovery_urgency: float
    abstention_signal: bool
    prediction: float
    uncertainty: float
    state_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NeuralOutcomeFeedback:
    selected_action: str
    observed_outcome: float
    predicted_outcome: float
    prediction_error: float
    utility: float
    regret: float
    viability_delta: float
    continuity_delta: float
    risk_delta: float
    resource_cost: float
    decision_accepted: bool
    neural_influence_delivered: bool


@dataclass(frozen=True, slots=True)
class NeuralPlasticityUpdate:
    learning_rate: float
    before_hash: str
    after_hash: str
    magnitude: float
    episode_count: int


@dataclass(frozen=True, slots=True)
class NeuralOrganSnapshot:
    neural_schema_version: str
    neural_state: tuple[tuple[str, Any], ...]
    neural_state_sha256: str
    plasticity_version: str
    episode_count: int
    last_prediction: float
    last_prediction_error: float
    modulation_state: tuple[tuple[str, Any], ...]
    backend_identity: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NeuralInfluenceReceipt:
    unit_id: str
    mode: str
    requested: bool
    delivered: bool
    pathways: tuple[str, ...]
    proposal_hash: str
    baseline_action: str | None = None
    selected_action: str | None = None
    governance_accepted: bool = False


@dataclass(slots=True)
class _State:
    episode_count: int = 0
    recurrent: float = 0.0
    threat_sensitivity: float = 0.5
    uncertainty_threshold: float = 0.55
    scale_bias: dict[str, float] = field(
        default_factory=lambda: {"micro": 0.0, "meso": 0.0, "macro": 0.0}
    )
    family_bias: dict[str, float] = field(
        default_factory=lambda: {"cau": 0.0, "ctf": 0.0, "ded": 0.0, "prob": 0.0}
    )
    action_bias: dict[str, float] = field(default_factory=dict)
    last_prediction: float = 0.0
    last_prediction_error: float = 0.0


class NeuralOrgan:
    """N3 recurrent state, pre-action prediction and bounded online plasticity."""

    def __init__(self, *, mode: str = "off", learning_rate: float = 0.08):
        if mode not in MODES:
            raise ValueError("neural_mode_invalid")
        self.mode = mode
        self.learning_rate = _bounded(learning_rate, 0.0, 0.25)
        self._state = _State()

    def _payload(self) -> dict[str, Any]:
        state = self._state
        return {
            "episode_count": state.episode_count,
            "recurrent": state.recurrent,
            "threat_sensitivity": state.threat_sensitivity,
            "uncertainty_threshold": state.uncertainty_threshold,
            "scale_bias": dict(sorted(state.scale_bias.items())),
            "family_bias": dict(sorted(state.family_bias.items())),
            "action_bias": dict(sorted(state.action_bias.items())),
            "last_prediction": state.last_prediction,
            "last_prediction_error": state.last_prediction_error,
        }

    @property
    def state_hash(self) -> str:
        return canonical_sha256(self._payload())

    def integrate(
        self, percept: NeuralPercept, body: NeuralBodyState,
        memory: NeuralMemoryState, reasoning: NeuralReasoningState,
        resources: NeuralResourceState, governance: NeuralGovernanceState,
    ) -> NeuralIntegratedState:
        return NeuralIntegratedState(
            percept.unit_id, percept.episode_index, percept.features,
            _items(asdict(body)), _items(asdict(memory)), _items(asdict(reasoning)),
            _items(asdict(resources)), _items(asdict(governance)), self.state_hash,
        )

    def produce_proposal(
        self, integrated: NeuralIntegratedState, allowed: Sequence[str]
    ) -> NeuralProposal:
        world, body, resource = (
            dict(integrated.world_features), dict(integrated.body_features),
            dict(integrated.resource_features),
        )
        pressure = max(float(resource.get(k, 0.0)) for k in (
            "cpu_pressure", "memory_pressure", "vram_pressure", "thermal_pressure"
        ))
        threat = _bounded(
            float(bool(world.get("alarm"))) +
            (1.0 - float(body.get("viability_margin", 1.0))) * self._state.threat_sensitivity,
            0.0, 1.0,
        )
        novelty = min(1.0, abs(self._state.last_prediction_error))
        signals = {
            "micro": _bounded(self._state.scale_bias["micro"] + novelty),
            "meso": _bounded(self._state.scale_bias["meso"] + self._state.recurrent),
            "macro": _bounded(self._state.scale_bias["macro"] + threat),
        }
        weights = tuple((scale, n3_scale_multiplier(scale, signals))
                        for scale in ("micro", "meso", "macro"))
        uncertainty = _bounded(0.5 * novelty + 0.3 * pressure, 0.0, 1.0)
        prior = max(
            (str(item) for item in allowed),
            key=lambda item: (self._state.action_bias.get(item, 0.0), -list(allowed).index(item)),
        ) if allowed else None
        prediction = _bounded(0.55 * self._state.recurrent +
                              0.45 * self._state.last_prediction)
        budget = -2 if pressure >= .8 else (-1 if pressure >= .6 else
                                            (1 if novelty > .4 else 0))
        digest = canonical_sha256({
            "unit": integrated.unit_id, "weights": weights, "family": self._state.family_bias,
            "budget": budget, "prior": prior, "prediction": prediction,
        })
        return NeuralProposal(
            integrated.unit_id, weights, (), tuple(sorted(self._state.family_bias.items())),
            budget, prior, "tier_1" if pressure >= .7 else "tier_2_specialized",
            max(threat, pressure),
            bool(threat >= .95 and uncertainty >= self._state.uncertainty_threshold),
            prediction, uncertainty, digest,
        )

    def apply_plasticity(self, feedback: NeuralOutcomeFeedback) -> NeuralPlasticityUpdate:
        before, state, lr = self.state_hash, self._state, self.learning_rate
        error = _bounded(feedback.prediction_error)
        state.recurrent = _bounded((1 - lr) * state.recurrent + lr * feedback.observed_outcome)
        state.last_prediction = _bounded(feedback.predicted_outcome)
        state.last_prediction_error = error
        state.threat_sensitivity = _bounded(
            state.threat_sensitivity + lr * max(feedback.risk_delta, 0.0), .1, 1.0)
        state.uncertainty_threshold = _bounded(
            state.uncertainty_threshold + lr * (abs(error) - .25), .2, .9)
        scale = "micro" if abs(error) > .5 else "meso"
        state.scale_bias[scale] = _bounded(state.scale_bias[scale] + lr * abs(error))
        delta = lr * _bounded(feedback.utility - feedback.regret)
        state.action_bias[feedback.selected_action] = _bounded(
            state.action_bias.get(feedback.selected_action, 0.0) + delta)
        family = "ctf" if abs(error) > .35 else "cau"
        state.family_bias[family] = _bounded(state.family_bias[family] + lr * abs(error))
        state.episode_count += 1
        return NeuralPlasticityUpdate(lr, before, self.state_hash,
                                      abs(error) * lr + abs(delta), state.episode_count)

    def snapshot(self) -> NeuralOrganSnapshot:
        payload = self._payload()
        return NeuralOrganSnapshot(
            SCHEMA_VERSION, _items(payload), canonical_sha256(payload),
            PLASTICITY_VERSION, self._state.episode_count, self._state.last_prediction,
            self._state.last_prediction_error,
            _items({"scale_bias": payload["scale_bias"], "family_bias": payload["family_bias"]}),
            "N3-reference+bounded-recurrent-v1",
        )

    def restore(self, snapshot: Mapping[str, Any] | NeuralOrganSnapshot) -> None:
        data = snapshot.to_dict() if isinstance(snapshot, NeuralOrganSnapshot) else dict(snapshot)
        if data.get("neural_schema_version") != SCHEMA_VERSION:
            raise ValueError("neural_snapshot_incompatible")
        raw = data.get("neural_state")
        state = dict(raw) if isinstance(raw, (dict, tuple, list)) else {}
        if canonical_sha256(state) != data.get("neural_state_sha256"):
            raise ValueError("neural_snapshot_hash_mismatch")
        restored = _State(
            episode_count=int(state["episode_count"]),
            recurrent=_bounded(state["recurrent"]),
            threat_sensitivity=_bounded(state["threat_sensitivity"], .1, 1.0),
            uncertainty_threshold=_bounded(state["uncertainty_threshold"], .2, .9),
            scale_bias={k: _bounded(v) for k, v in dict(state["scale_bias"]).items()},
            family_bias={k: _bounded(v) for k, v in dict(state["family_bias"]).items()},
            action_bias={str(k): _bounded(v) for k, v in dict(state["action_bias"]).items()},
            last_prediction=_bounded(state["last_prediction"]),
            last_prediction_error=_bounded(state["last_prediction_error"]),
        )
        self._state = restored


class OrganismNeuralBus:
    """Versioned causal-order transport. It never overrules governance."""

    def __init__(self, organ: NeuralOrgan):
        self.organ, self._phase = organ, "idle"
        self._integrated: NeuralIntegratedState | None = None
        self._proposal: NeuralProposal | None = None

    def collect_inputs(self, **inputs: Any) -> dict[str, Any]:
        if self._phase not in {"idle", "learned"}:
            raise RuntimeError("neural_bus_causal_order_violation")
        self._phase = "collected"
        return inputs

    def integrate(self, **inputs: Any) -> NeuralIntegratedState:
        if self._phase != "collected":
            raise RuntimeError("neural_bus_integrate_order")
        self._integrated = self.organ.integrate(**inputs)
        self._phase = "integrated"
        return self._integrated

    def produce_proposal(self, allowed: Sequence[str]) -> NeuralProposal:
        if self._phase != "integrated" or self._integrated is None:
            raise RuntimeError("neural_bus_proposal_order")
        self._proposal = self.organ.produce_proposal(self._integrated, allowed)
        self._phase = "proposed"
        return self._proposal

    def deliver_modulations(
        self, *, baseline_action: str | None, governance_accepted: bool
    ) -> NeuralInfluenceReceipt:
        if self._phase != "proposed" or self._proposal is None:
            raise RuntimeError("neural_bus_delivery_order")
        active = self.organ.mode in {"modulate", "bounded_actuation"}
        paths = ("memory", "reasoning", "resources") if active else ()
        if self.organ.mode == "bounded_actuation" and governance_accepted:
            paths += ("action",)
        self._phase = "delivered"
        return NeuralInfluenceReceipt(
            self._proposal.unit_id, self.organ.mode, self.organ.mode != "off",
            bool(paths), paths, self._proposal.state_hash, baseline_action,
            governance_accepted=governance_accepted,
        )

    def collect_feedback(self, feedback: NeuralOutcomeFeedback) -> NeuralOutcomeFeedback:
        if self._phase != "delivered":
            raise RuntimeError("neural_bus_feedback_order")
        self._phase = "feedback"
        return feedback

    def apply_plasticity(self, feedback: NeuralOutcomeFeedback) -> NeuralPlasticityUpdate:
        if self._phase != "feedback":
            raise RuntimeError("neural_bus_plasticity_order")
        update = self.organ.apply_plasticity(feedback)
        self._phase = "learned"
        return update

    def snapshot(self) -> NeuralOrganSnapshot:
        return self.organ.snapshot()

    def restore(self, snapshot: Mapping[str, Any] | NeuralOrganSnapshot) -> None:
        self.organ.restore(snapshot)
        self._phase = "idle"
