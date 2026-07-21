"""P1 N4 pre-action scoring and counterfactual evaluation contracts.

The scorer sees only the current observation and evidence available before an
action is committed.  Outcomes enter through the separate evaluator and can
therefore never change the candidate or its hash.
"""

from __future__ import annotations

import math
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import canonical_sha256, canonicalize


PREACTION_SCHEMA_VERSION = "n4-preaction-intervention-set-v1"
SCORE_SET_SCHEMA_VERSION = "n4-intervention-score-set-v1"
EVALUATION_SCHEMA_VERSION = "n4-preaction-evaluation-v1"
ARTIFACT_SCHEMA_VERSION = "n4-preaction-artifact-v2"
MANIFEST_SCHEMA_VERSION = "n4-preaction-manifest-v1"
PREACTION_BACKEND = "rnfe-n4-preaction-linear-v2"
MAX_INTERVENTIONS = 512
_EPSILON = 1e-9
FEATURE_NAMES = (
    "observed_value",
    "prior_delta",
    "prior_confidence",
    "lagged_delta",
    "lagged_confidence",
    "n3_trend",
    "n3_uncertainty",
    "n3_risk",
    "n3_importance",
    "n3_continuity",
)

_PREACTION_FIELDS = frozenset(
    {
        "schema_version",
        "scenario_id",
        "main_variable",
        "optimization_direction",
        "observation",
        "interventions",
        "canonical_intervention",
        "prior_evidence",
        "lagged_evidence",
        "n3_signals",
    }
)
_FORBIDDEN_KEYS = frozenset(
    {
        "causal_attestation",
        "committed_action",
        "committed_intervention",
        "counterfactual",
        "counterfactual_transition",
        "current_transition",
        "factual",
        "factual_transition",
        "ground_truth",
        "outcome",
        "outcomes",
        "relation",
        "relation_kind",
        "selected_intervention",
        "transition",
    }
)
_PRIOR_FIELDS = frozenset(
    {"expected_delta", "signed_delta", "confidence", "evidence_ref"}
)
_LAGGED_FIELDS = frozenset(
    {"mean_delta", "signed_delta", "confidence", "sample_count", "evidence_ref"}
)
_N3_FIELDS = frozenset(
    {"trend", "uncertainty", "risk", "importance", "continuity", "evidence_ref"}
)


def _required_text(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"n4_preaction_{field}_required")
    return text


def _finite(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"n4_preaction_{field}_must_be_finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"n4_preaction_{field}_must_be_finite")
    return number


def _optional_finite(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    return _finite(value, field=field)


def _reject_forbidden_keys(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).strip().lower()
            if key in _FORBIDDEN_KEYS:
                raise ValueError(f"n4_preaction_outcome_leak:{path}.{key}")
            _reject_forbidden_keys(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden_keys(child, path=f"{path}[{index}]")


def _evidence_table(
    raw: Mapping[str, Any] | None,
    *,
    interventions: tuple[str, ...],
    allowed_fields: frozenset[str],
    label: str,
) -> dict[str, dict[str, Any]]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"n4_preaction_{label}_must_be_mapping")
    unknown_actions = set(raw) - set(interventions)
    if unknown_actions:
        raise ValueError(
            f"n4_preaction_{label}_unknown_intervention:{sorted(unknown_actions)[0]}"
        )
    table: dict[str, dict[str, Any]] = {}
    for action in interventions:
        if action not in raw:
            continue
        item = raw[action]
        if not isinstance(item, Mapping):
            raise ValueError(f"n4_preaction_{label}_entry_must_be_mapping:{action}")
        unknown = set(item) - allowed_fields
        if unknown:
            raise ValueError(
                f"n4_preaction_{label}_field_forbidden:{action}:{sorted(unknown)[0]}"
            )
        normalized = canonicalize(dict(item))
        confidence = _optional_finite(normalized.get("confidence"), field="confidence")
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError("n4_preaction_confidence_out_of_range")
        if "sample_count" in normalized:
            sample_count = normalized["sample_count"]
            if isinstance(sample_count, bool) or int(sample_count) != sample_count or sample_count < 0:
                raise ValueError("n4_preaction_sample_count_invalid")
        for delta_field in ("expected_delta", "mean_delta", "signed_delta"):
            if delta_field in normalized:
                normalized[delta_field] = _finite(
                    normalized[delta_field], field=delta_field
                )
        if "evidence_ref" in normalized:
            normalized["evidence_ref"] = _required_text(
                normalized["evidence_ref"], field="evidence_ref"
            )
        table[action] = normalized
    return table


def _n3_signal_table(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("n4_preaction_n3_signals_must_be_mapping")
    unknown = set(raw) - _N3_FIELDS
    if unknown:
        raise ValueError(f"n4_preaction_n3_signal_forbidden:{sorted(unknown)[0]}")
    normalized = canonicalize(dict(raw))
    for field in _N3_FIELDS - {"evidence_ref"}:
        if field not in normalized:
            continue
        normalized[field] = _finite(normalized[field], field=f"n3_{field}")
        if field in {"uncertainty", "risk", "importance", "continuity"} and not (
            0.0 <= normalized[field] <= 1.0
        ):
            raise ValueError(f"n4_preaction_n3_{field}_out_of_range")
    if "evidence_ref" in normalized:
        normalized["evidence_ref"] = _required_text(
            normalized["evidence_ref"], field="n3_evidence_ref"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class N4PreactionInterventionSet:
    scenario_id: str
    main_variable: str
    optimization_direction: str
    observation: Mapping[str, Any]
    interventions: tuple[str, ...]
    canonical_intervention: str
    prior_evidence: Mapping[str, Mapping[str, Any]]
    lagged_evidence: Mapping[str, Mapping[str, Any]]
    n3_signals: Mapping[str, Any]
    schema_version: str = PREACTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PREACTION_SCHEMA_VERSION:
            raise ValueError("n4_preaction_schema_mismatch")
        object.__setattr__(self, "scenario_id", _required_text(self.scenario_id, field="scenario_id"))
        object.__setattr__(
            self, "main_variable", _required_text(self.main_variable, field="main_variable")
        )
        direction = str(self.optimization_direction).strip().lower()
        if direction not in {"minimize", "maximize"}:
            raise ValueError("n4_preaction_optimization_direction_unsupported")
        object.__setattr__(self, "optimization_direction", direction)
        if not isinstance(self.observation, Mapping):
            raise ValueError("n4_preaction_observation_must_be_mapping")
        observation = canonicalize(dict(self.observation))
        _reject_forbidden_keys(observation, path="observation")
        _finite(observation.get(self.main_variable), field="observed_main_value")
        object.__setattr__(self, "observation", observation)

        interventions = tuple(
            _required_text(item, field="intervention") for item in self.interventions
        )
        if not interventions:
            raise ValueError("n4_preaction_interventions_required")
        if len(interventions) > MAX_INTERVENTIONS:
            raise ValueError("n4_preaction_intervention_budget_exceeded")
        if len(set(interventions)) != len(interventions):
            raise ValueError("n4_preaction_interventions_must_be_ordered_unique")
        object.__setattr__(self, "interventions", interventions)
        canonical = _required_text(
            self.canonical_intervention, field="canonical_intervention"
        )
        if canonical not in interventions:
            raise ValueError("n4_preaction_canonical_intervention_not_allowed")
        object.__setattr__(self, "canonical_intervention", canonical)

        prior = _evidence_table(
            self.prior_evidence,
            interventions=interventions,
            allowed_fields=_PRIOR_FIELDS,
            label="prior_evidence",
        )
        lagged = _evidence_table(
            self.lagged_evidence,
            interventions=interventions,
            allowed_fields=_LAGGED_FIELDS,
            label="lagged_evidence",
        )
        _reject_forbidden_keys(prior, path="prior_evidence")
        _reject_forbidden_keys(lagged, path="lagged_evidence")
        object.__setattr__(self, "prior_evidence", prior)
        object.__setattr__(self, "lagged_evidence", lagged)
        n3_signals = _n3_signal_table(self.n3_signals)
        _reject_forbidden_keys(n3_signals, path="n3_signals")
        object.__setattr__(self, "n3_signals", n3_signals)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "N4PreactionInterventionSet":
        if not isinstance(raw, Mapping):
            raise ValueError("n4_preaction_input_must_be_mapping")
        unknown = set(raw) - _PREACTION_FIELDS
        if unknown:
            raise ValueError(f"n4_preaction_field_forbidden:{sorted(unknown)[0]}")
        missing = _PREACTION_FIELDS - set(raw)
        if missing:
            raise ValueError(f"n4_preaction_field_missing:{sorted(missing)[0]}")
        return cls(
            schema_version=raw["schema_version"],
            scenario_id=raw["scenario_id"],
            main_variable=raw["main_variable"],
            optimization_direction=raw["optimization_direction"],
            observation=raw["observation"],
            interventions=tuple(raw["interventions"]),
            canonical_intervention=raw["canonical_intervention"],
            prior_evidence=raw["prior_evidence"],
            lagged_evidence=raw["lagged_evidence"],
            n3_signals=raw["n3_signals"],
        )

    @property
    def input_hash(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "main_variable": self.main_variable,
            "optimization_direction": self.optimization_direction,
            "observation": dict(self.observation),
            "interventions": list(self.interventions),
            "canonical_intervention": self.canonical_intervention,
            "prior_evidence": {key: dict(value) for key, value in self.prior_evidence.items()},
            "lagged_evidence": {key: dict(value) for key, value in self.lagged_evidence.items()},
            "n3_signals": dict(self.n3_signals),
        }


@dataclass(frozen=True, slots=True)
class N4PreactionArtifactV2:
    """Validated, immutable pre-action model; labels never enter this object."""

    model_id: str
    coefficients: Mapping[str, float]
    pair_bias: Mapping[str, float]
    feature_ranges: Mapping[str, Mapping[str, Sequence[float]]]
    calibration_half_width: float
    confidence: float
    training_provenance: Mapping[str, Any]
    artifact_sha256: str
    schema_version: str = ARTIFACT_SCHEMA_VERSION
    backend: str = PREACTION_BACKEND
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def __post_init__(self) -> None:
        if self.schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ValueError("n4_preaction_artifact_schema_mismatch")
        if self.backend != PREACTION_BACKEND:
            raise ValueError("n4_preaction_artifact_backend_mismatch")
        if tuple(self.feature_names) != FEATURE_NAMES:
            raise ValueError("n4_preaction_artifact_feature_schema_mismatch")
        if len(self.artifact_sha256) != 64:
            raise ValueError("n4_preaction_artifact_hash_invalid")
        coefficients = {
            str(name): _finite(value, field=f"coefficient_{name}")
            for name, value in self.coefficients.items()
        }
        if set(coefficients) != set(FEATURE_NAMES):
            raise ValueError("n4_preaction_artifact_coefficients_incomplete")
        pair_bias = {
            _required_text(pair, field="supported_pair"): _finite(
                value, field="pair_bias"
            )
            for pair, value in self.pair_bias.items()
        }
        if not pair_bias:
            raise ValueError("n4_preaction_artifact_supported_pairs_required")
        ranges: dict[str, dict[str, tuple[float, float]]] = {}
        if set(self.feature_ranges) != set(pair_bias):
            raise ValueError("n4_preaction_artifact_ranges_mismatch")
        for pair, raw_ranges in self.feature_ranges.items():
            if set(raw_ranges) != set(FEATURE_NAMES):
                raise ValueError("n4_preaction_artifact_feature_ranges_incomplete")
            resolved: dict[str, tuple[float, float]] = {}
            for name, bounds in raw_ranges.items():
                if not isinstance(bounds, Sequence) or isinstance(bounds, (str, bytes)) or len(bounds) != 2:
                    raise ValueError("n4_preaction_artifact_range_invalid")
                lower = _finite(bounds[0], field=f"range_{name}_lower")
                upper = _finite(bounds[1], field=f"range_{name}_upper")
                if lower > upper:
                    raise ValueError("n4_preaction_artifact_range_reversed")
                resolved[str(name)] = (lower, upper)
            ranges[str(pair)] = resolved
        provenance = canonicalize(dict(self.training_provenance))
        counts = provenance.get("trajectory_counts")
        if counts != {"evaluation": 12, "train": 24, "validation": 6}:
            raise ValueError("n4_preaction_artifact_split_contract_mismatch")
        if provenance.get("split_disjoint") is not True:
            raise ValueError("n4_preaction_artifact_split_overlap")
        width = _finite(self.calibration_half_width, field="calibration_half_width")
        confidence = _finite(self.confidence, field="artifact_confidence")
        if width < 0.0 or not 0.0 <= confidence <= 1.0:
            raise ValueError("n4_preaction_artifact_calibration_invalid")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "pair_bias", pair_bias)
        object.__setattr__(self, "feature_ranges", ranges)
        object.__setattr__(self, "training_provenance", provenance)
        object.__setattr__(self, "calibration_half_width", width)
        object.__setattr__(self, "confidence", confidence)

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, artifact_sha256: str
    ) -> "N4PreactionArtifactV2":
        return cls(
            schema_version=str(payload.get("schema_version") or ""),
            backend=str(payload.get("backend") or ""),
            model_id=str(payload.get("model_id") or ""),
            feature_names=tuple(payload.get("feature_names") or ()),
            coefficients=dict(payload.get("coefficients") or {}),
            pair_bias=dict(payload.get("pair_bias") or {}),
            feature_ranges=dict(payload.get("feature_ranges") or {}),
            calibration_half_width=payload.get("calibration_half_width"),
            confidence=payload.get("confidence"),
            training_provenance=dict(payload.get("training_provenance") or {}),
            artifact_sha256=artifact_sha256,
        )


def load_preaction_artifact_v2(
    manifest_path: str | Path,
    *,
    artifact_root: str | Path | None = None,
) -> N4PreactionArtifactV2:
    """Load manifest+artifact with a fail-closed content-hash binding."""

    manifest_file = Path(manifest_path)
    if not manifest_file.is_absolute():
        if artifact_root is None:
            raise ValueError("n4_preaction_artifact_root_required")
        manifest_file = Path(artifact_root) / manifest_file
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("n4_preaction_manifest_schema_mismatch")
    if manifest.get("backend") != PREACTION_BACKEND:
        raise ValueError("n4_preaction_manifest_backend_mismatch")
    artifact_file = Path(str(manifest.get("artifact_path") or ""))
    if not artifact_file.is_absolute():
        artifact_file = manifest_file.parent / artifact_file
    raw = artifact_file.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != manifest.get("artifact_sha256"):
        raise ValueError("n4_preaction_artifact_hash_mismatch")
    payload = json.loads(raw.decode("utf-8"))
    return N4PreactionArtifactV2.from_payload(payload, artifact_sha256=digest)


@dataclass(frozen=True, slots=True)
class N4InterventionScoreSet:
    scenario_id: str
    main_variable: str
    optimization_direction: str
    input_hash: str
    canonical_intervention: str
    scores: tuple[Mapping[str, Any], ...]
    model_id: str | None = None
    artifact_sha256: str | None = None
    execution_class: str = "abstained"
    schema_version: str = SCORE_SET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCORE_SET_SCHEMA_VERSION:
            raise ValueError("n4_score_set_schema_mismatch")
        if not self.scores:
            raise ValueError("n4_score_set_scores_required")
        normalized = tuple(canonicalize(dict(item)) for item in self.scores)
        actions = [str(item.get("intervention") or "") for item in normalized]
        if any(not action for action in actions) or len(set(actions)) != len(actions):
            raise ValueError("n4_score_set_interventions_invalid")
        if self.canonical_intervention not in actions:
            raise ValueError("n4_score_set_canonical_intervention_not_allowed")
        object.__setattr__(self, "scores", normalized)

    @property
    def candidate_hash(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "main_variable": self.main_variable,
            "optimization_direction": self.optimization_direction,
            "input_hash": self.input_hash,
            "canonical_intervention": self.canonical_intervention,
            "scores": [dict(item) for item in self.scores],
            "model": {
                "model_id": self.model_id,
                "backend": PREACTION_BACKEND if self.model_id else None,
                "artifact_sha256": self.artifact_sha256,
                "execution_class": self.execution_class,
            },
            "authority": {
                "authority_effect": "none",
                "proposal_only": True,
                "may_choose_intervention": False,
                "may_authorize_action": False,
                "may_mutate_graph": False,
            },
            "decision_influence": "none",
        }


def _delta(item: Mapping[str, Any], fields: Sequence[str]) -> float | None:
    for field in fields:
        if field in item:
            return _finite(item[field], field=field)
    return None


def _confidence(item: Mapping[str, Any], *, default: float) -> float:
    raw = item.get("confidence")
    if raw is not None:
        return _finite(raw, field="confidence")
    sample_count = item.get("sample_count")
    if sample_count is not None:
        count = float(sample_count)
        return count / (count + 4.0) if count > 0.0 else 0.0
    return default


def preaction_feature_row(
    request: N4PreactionInterventionSet, intervention: str
) -> dict[str, float]:
    """Materialize the exact feature schema shared by trainer and runtime."""

    observed = _finite(request.observation[request.main_variable], field="observed_main_value")
    prior = request.prior_evidence.get(intervention, {})
    lagged = request.lagged_evidence.get(intervention, {})
    n3 = request.n3_signals
    return {
        "observed_value": observed,
        "prior_delta": _delta(prior, ("expected_delta", "signed_delta")) or 0.0,
        "prior_confidence": _confidence(prior, default=0.0) if prior else 0.0,
        "lagged_delta": _delta(lagged, ("mean_delta", "signed_delta")) or 0.0,
        "lagged_confidence": _confidence(lagged, default=0.0) if lagged else 0.0,
        "n3_trend": _finite(n3.get("trend", 0.0), field="n3_trend"),
        "n3_uncertainty": _finite(n3.get("uncertainty", 1.0), field="n3_uncertainty"),
        "n3_risk": _finite(n3.get("risk", 0.0), field="n3_risk"),
        "n3_importance": _finite(n3.get("importance", 0.0), field="n3_importance"),
        "n3_continuity": _finite(n3.get("continuity", 0.0), field="n3_continuity"),
    }


def _feature_ood(
    features: Mapping[str, float], ranges: Mapping[str, Sequence[float]]
) -> str | None:
    for name in FEATURE_NAMES:
        lower, upper = (float(value) for value in ranges[name])
        span = max(upper - lower, abs(lower), abs(upper), 1e-6)
        margin = 0.10 * span
        if features[name] < lower - margin or features[name] > upper + margin:
            return name
    return None


def score_preaction_interventions(
    request: N4PreactionInterventionSet,
    *,
    artifact: N4PreactionArtifactV2 | None = None,
    artifact_error: str | None = None,
) -> N4InterventionScoreSet:
    """Score all actions with a trained v2 artifact or abstain on every row."""

    observed = _finite(request.observation[request.main_variable], field="observed_main_value")
    rows: list[dict[str, Any]] = []
    for action in request.interventions:
        prior = request.prior_evidence.get(action, {})
        prior_delta = _delta(prior, ("expected_delta", "signed_delta"))
        pair = f"{request.scenario_id}::{action}"
        if artifact is None:
            rows.append(
                {
                    "intervention": action,
                    "status": "abstained",
                    "predicted_delta": None,
                    "predicted_next_value": None,
                    "lower": None,
                    "upper": None,
                    "confidence": 0.0,
                    "uncertainty": 1.0,
                    "prior_delta": prior_delta,
                    "evidence_refs": [],
                    "abstention_reason": artifact_error or "artifact_missing",
                }
            )
            continue
        if pair not in artifact.pair_bias:
            reason = "artifact_incompatible:unsupported_scenario_intervention"
            features = None
        else:
            features = preaction_feature_row(request, action)
            ood_feature = _feature_ood(features, artifact.feature_ranges[pair])
            reason = f"ood:{ood_feature}" if ood_feature else None
        if reason is not None:
            rows.append(
                {
                    "intervention": action,
                    "status": "abstained",
                    "predicted_delta": None,
                    "predicted_next_value": None,
                    "lower": None,
                    "upper": None,
                    "confidence": 0.0,
                    "uncertainty": 1.0,
                    "prior_delta": prior_delta,
                    "evidence_refs": [],
                    "abstention_reason": reason,
                }
            )
            continue
        assert features is not None
        predicted_delta = artifact.pair_bias[pair] + sum(
            artifact.coefficients[name] * features[name] for name in FEATURE_NAMES
        )
        combined_confidence = artifact.confidence
        uncertainty = 1.0 - artifact.confidence
        predicted_next = observed + predicted_delta
        half_width = artifact.calibration_half_width
        rows.append(
            {
                "intervention": action,
                "status": "scored",
                "predicted_delta": predicted_delta,
                "predicted_next_value": predicted_next,
                "lower": predicted_next - half_width,
                "upper": predicted_next + half_width,
                "confidence": combined_confidence,
                "uncertainty": uncertainty,
                "prior_delta": prior_delta,
                "evidence_refs": [
                    f"artifact:{artifact.artifact_sha256}",
                    *(
                        [str(prior["evidence_ref"])]
                        if prior.get("evidence_ref")
                        else []
                    ),
                ],
                "abstention_reason": None,
            }
        )
    return N4InterventionScoreSet(
        scenario_id=request.scenario_id,
        main_variable=request.main_variable,
        optimization_direction=request.optimization_direction,
        input_hash=request.input_hash,
        canonical_intervention=request.canonical_intervention,
        scores=tuple(rows),
        model_id=artifact.model_id if artifact else None,
        artifact_sha256=artifact.artifact_sha256 if artifact else None,
        execution_class="trained_v2" if artifact else "abstained",
    )


def causal_signature_prior_evidence(
    signature: Any,
    *,
    interventions: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Project declared causal priors without reading the current outcome."""

    allowed = set(str(item) for item in interventions)
    evidence: dict[str, dict[str, Any]] = {}
    for effect in getattr(signature, "intervention_effects", ()) or ():
        action = str(getattr(effect, "intervention_name", "") or "")
        if action not in allowed:
            continue
        magnitude = _finite(
            getattr(effect, "expected_magnitude", 0.0), field="expected_magnitude"
        )
        direction = str(getattr(effect, "expected_direction", "") or "").strip()
        signed = -abs(magnitude) if direction in {"-", "decrease", "negative"} else abs(magnitude)
        evidence[action] = {
            "expected_delta": signed,
            "confidence": 0.5,
            "evidence_ref": f"causal_signature:{action}",
        }
    return evidence


def lagged_evidence_from_memory(
    hits: Sequence[Mapping[str, Any]],
    *,
    interventions: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Aggregate only completed earlier episodes, never the current simulation."""

    allowed = set(str(item) for item in interventions)
    values: dict[str, list[float]] = {action: [] for action in allowed}
    refs: dict[str, list[str]] = {action: [] for action in allowed}
    for hit in hits:
        structure = hit.get("structure") if isinstance(hit, Mapping) else None
        if not isinstance(structure, Mapping):
            continue
        context = structure.get("context")
        result = structure.get("result")
        context = context if isinstance(context, Mapping) else {}
        result = result if isinstance(result, Mapping) else {}
        action = str(context.get("intervention") or structure.get("intervention") or "")
        if action not in allowed:
            continue
        raw_delta = result.get("factual_delta", structure.get("factual_delta"))
        try:
            delta = _finite(raw_delta, field="lagged_factual_delta")
        except ValueError:
            continue
        values[action].append(delta)
        refs[action].append(str(hit.get("memory_id") or hit.get("episode_id") or "memory"))
    out: dict[str, dict[str, Any]] = {}
    for action in interventions:
        rows = values.get(str(action), [])
        if not rows:
            continue
        count = len(rows)
        out[str(action)] = {
            "mean_delta": sum(rows) / count,
            "sample_count": count,
            "confidence": count / (count + 4.0),
            "evidence_ref": "lagged_memory:" + canonical_sha256(refs[str(action)]),
        }
    return out


def _outcome_value(raw: Any, main_variable: str) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return _finite(raw, field="outcome_value")
    if not isinstance(raw, Mapping):
        raise ValueError("n4_evaluation_outcome_invalid")
    state = raw.get("state")
    if isinstance(state, Mapping) and main_variable in state:
        return _finite(state[main_variable], field="outcome_value")
    for key in ("next_value", main_variable):
        if key in raw:
            return _finite(raw[key], field="outcome_value")
    return None


def _utility(value: float, direction: str) -> float:
    return value if direction == "maximize" else -value


def _top_set(
    values: Mapping[str, float], order: Sequence[str], *, epsilon: float
) -> tuple[str, ...]:
    if not values:
        return ()
    best = max(values.values())
    return tuple(action for action in order if action in values and best - values[action] <= epsilon)


def evaluate_preaction_scores(
    candidate: N4InterventionScoreSet,
    *,
    outcomes: Mapping[str, Any],
    observed_value: float,
    oracle_snapshot_sha256: str | None = None,
    outcome_set_sha256: str | None = None,
    epsilon: float = _EPSILON,
) -> dict[str, Any]:
    """Evaluate a frozen candidate against hidden sandbox outcomes."""

    if epsilon < 0.0 or not math.isfinite(float(epsilon)):
        raise ValueError("n4_evaluation_epsilon_invalid")
    actions = tuple(str(row["intervention"]) for row in candidate.scores)
    unknown = set(outcomes) - set(actions)
    if unknown:
        raise ValueError(f"n4_evaluation_unknown_intervention:{sorted(unknown)[0]}")
    observed = _finite(observed_value, field="observed_main_value")
    candidate_hash_before = candidate.candidate_hash
    row_by_action = {str(row["intervention"]): row for row in candidate.scores}
    truth = {
        action: value
        for action in actions
        if (value := _outcome_value(outcomes.get(action), candidate.main_variable)) is not None
    }
    truth_utility = {
        action: _utility(value, candidate.optimization_direction)
        for action, value in truth.items()
    }
    predicted_utility = {
        action: _utility(float(row["predicted_next_value"]), candidate.optimization_direction)
        for action, row in row_by_action.items()
        if row.get("status") == "scored" and row.get("predicted_next_value") is not None
    }
    prior_utility = {
        action: _utility(observed + float(row["prior_delta"]), candidate.optimization_direction)
        for action, row in row_by_action.items()
        if row.get("prior_delta") is not None
    }
    truth_top = _top_set(truth_utility, actions, epsilon=epsilon)
    oracle_seal_verified = False
    if (oracle_snapshot_sha256 is None) != (outcome_set_sha256 is None):
        raise ValueError("n4_evaluation_oracle_seal_incomplete")
    if oracle_snapshot_sha256 is not None and outcome_set_sha256 is not None:
        sealed_rows = []
        for action in actions:
            raw = outcomes.get(action)
            state = raw.get("state") if isinstance(raw, Mapping) else None
            if action not in truth or not isinstance(state, Mapping):
                raise ValueError("n4_evaluation_sealed_outcome_incomplete")
            sealed_rows.append(
                {
                    "intervention": action,
                    "value": truth[action],
                    "delta": truth[action] - observed,
                    "state": dict(state),
                }
            )
        recomputed_outcome_set_sha256 = hashlib.sha256(
            json.dumps(
                {
                    "snapshot_sha256": oracle_snapshot_sha256,
                    "outcomes": sealed_rows,
                    "best_actions": list(truth_top),
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        if recomputed_outcome_set_sha256 != outcome_set_sha256:
            raise ValueError("n4_evaluation_outcome_seal_mismatch")
        oracle_seal_verified = True
    predicted_top = _top_set(predicted_utility, actions, epsilon=epsilon)
    prior_top = _top_set(prior_utility, actions, epsilon=epsilon)
    shadow_action = predicted_top[0] if predicted_top else None
    prior_action = prior_top[0] if prior_top else None
    best_truth = max(truth_utility.values()) if truth_utility else None

    def regret(action: str | None) -> float | None:
        if action is None or action not in truth_utility or best_truth is None:
            return None
        return best_truth - truth_utility[action]

    absolute_errors = [
        abs(float(row_by_action[action]["predicted_next_value"]) - truth[action])
        for action in actions
        if action in truth and action in predicted_utility
    ]
    delta_errors = [
        abs(float(row_by_action[action]["predicted_delta"]) - (truth[action] - observed))
        for action in actions
        if action in truth and action in predicted_utility
    ]
    interval_hits = [
        bool(
            float(row_by_action[action]["lower"]) - epsilon
            <= truth[action]
            <= float(row_by_action[action]["upper"]) + epsilon
        )
        for action in actions
        if action in truth and action in predicted_utility
    ]
    pairwise_correct = 0
    pairwise_total = 0
    for left_index, left in enumerate(actions):
        for right in actions[left_index + 1 :]:
            if left not in truth_utility or right not in truth_utility:
                continue
            if left not in predicted_utility or right not in predicted_utility:
                continue
            truth_diff = truth_utility[left] - truth_utility[right]
            if abs(truth_diff) <= epsilon:
                continue
            predicted_diff = predicted_utility[left] - predicted_utility[right]
            pairwise_total += 1
            pairwise_correct += int(truth_diff * predicted_diff > 0.0)

    canonical_regret = regret(candidate.canonical_intervention)
    n4_regret = regret(shadow_action)
    prior_regret = regret(prior_action)
    result = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "scenario_id": candidate.scenario_id,
        "candidate_hash": candidate_hash_before,
        "input_hash": candidate.input_hash,
        "artifact_sha256": candidate.artifact_sha256,
        "oracle_snapshot_sha256": oracle_snapshot_sha256,
        "outcome_set_sha256": outcome_set_sha256,
        "oracle_seal_verified": oracle_seal_verified,
        "candidate_hash_preserved": candidate_hash_before == candidate.candidate_hash,
        "authority_effect": "none",
        "decision_influence": "none",
        "coverage": sum(row.get("status") == "scored" for row in candidate.scores)
        / len(candidate.scores),
        "ground_truth_coverage": len(truth) / len(actions),
        "ground_truth_top_interventions": list(truth_top),
        "predicted_top_interventions": list(predicted_top),
        "prior_top_interventions": list(prior_top),
        "shadow_intervention": shadow_action,
        "top1_correct": (
            None if not truth_top or not predicted_top else bool(set(truth_top) & set(predicted_top))
        ),
        "mae_next_value": (
            sum(absolute_errors) / len(absolute_errors) if absolute_errors else None
        ),
        "mae_delta": sum(delta_errors) / len(delta_errors) if delta_errors else None,
        "interval_coverage": sum(interval_hits) / len(interval_hits) if interval_hits else None,
        "pairwise_ranking_accuracy": (
            pairwise_correct / pairwise_total if pairwise_total else None
        ),
        "pairwise_comparisons": pairwise_total,
        "canonical_regret": canonical_regret,
        "n4_regret": n4_regret,
        "prior_regret": prior_regret,
        "regret_delta_vs_canonical": (
            canonical_regret - n4_regret
            if canonical_regret is not None and n4_regret is not None
            else None
        ),
        "regret_delta_vs_prior": (
            prior_regret - n4_regret
            if prior_regret is not None and n4_regret is not None
            else None
        ),
    }
    if result["candidate_hash_preserved"] is not True:
        raise RuntimeError("n4_evaluation_mutated_candidate")
    result["evaluation_sha256"] = canonical_sha256(result)
    return result


__all__ = [
    "EVALUATION_SCHEMA_VERSION",
    "ARTIFACT_SCHEMA_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "PREACTION_BACKEND",
    "FEATURE_NAMES",
    "N4PreactionArtifactV2",
    "N4InterventionScoreSet",
    "N4PreactionInterventionSet",
    "PREACTION_SCHEMA_VERSION",
    "SCORE_SET_SCHEMA_VERSION",
    "evaluate_preaction_scores",
    "causal_signature_prior_evidence",
    "lagged_evidence_from_memory",
    "load_preaction_artifact_v2",
    "preaction_feature_row",
    "score_preaction_interventions",
]
