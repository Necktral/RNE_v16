"""Entrenamiento PyTorch/CUDA y exportación a inferencia pura."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FEATURE_NAMES = (
    "empirical_gain",
    "holdout_support",
    "coverage",
    "simplicity",
    "stability",
    "invariant_safety",
)

RISK_EPSILON = 1e-6
GAIN_EPSILON = 1e-6
V2_LOSS_WEIGHTS = {"rank": 1.0, "valid": 0.5, "gain": 0.3, "risk": 0.2}
RISK_LABEL_VERSION = "n4-risk-label.multistep.v1"


@dataclass(frozen=True)
class N4CandidateRecord:
    """Una alternativa dentro de un conjunto lógico de ranking."""

    hypothesis_id: str
    candidate_source: str
    features: tuple[float, ...]
    valid_label: bool
    mae_gain_label: float
    invariant_risk_label: float | None
    risk_label_available: bool = True
    risk_label_version: str | None = RISK_LABEL_VERSION
    risk_report_sha256: str | None = None
    safety_contract_version: str | None = None
    rollout_horizon: int | None = None
    risk_components: Mapping[str, float] | None = None
    risk_event_families: tuple[str, ...] = ()
    risk_hazard_first_step: int | None = None

    def __post_init__(self) -> None:
        if not self.hypothesis_id:
            raise ValueError("n4_candidate_hypothesis_id_required")
        if not self.candidate_source:
            raise ValueError("n4_candidate_source_required")
        if len(self.features) != len(FEATURE_NAMES):
            raise ValueError("n4_candidate_feature_shape_invalid")
        values = (
            *self.features,
            self.mae_gain_label,
        )
        if not all(math.isfinite(float(item)) for item in values):
            raise ValueError("n4_candidate_values_must_be_finite")
        if not -1.0 <= self.mae_gain_label <= 1.0:
            raise ValueError("n4_candidate_gain_out_of_range")
        if not self.risk_label_available:
            if any(
                item is not None
                for item in (
                    self.invariant_risk_label,
                    self.risk_label_version,
                    self.risk_report_sha256,
                    self.safety_contract_version,
                    self.rollout_horizon,
                )
            ):
                raise ValueError("n4_unavailable_risk_metadata_must_be_null")
        else:
            if self.invariant_risk_label is None or not math.isfinite(
                float(self.invariant_risk_label)
            ):
                raise ValueError("n4_available_risk_label_must_be_finite")
            if not 0.0 <= float(self.invariant_risk_label) <= 1.0:
                raise ValueError("n4_candidate_risk_out_of_range")
            if self.risk_label_version != RISK_LABEL_VERSION:
                raise ValueError("n4_candidate_risk_version_unknown")
            if self.rollout_horizon is not None and self.rollout_horizon < 1:
                raise ValueError("n4_candidate_rollout_horizon_invalid")


@dataclass(frozen=True)
class CandidateSetSample:
    """Conjunto indivisible para entrenamiento, evaluación y particionado."""

    candidate_set_id: str
    scenario_id: str
    seed: int
    split: str
    candidates: tuple[N4CandidateRecord, ...]

    def __post_init__(self) -> None:
        if not self.candidate_set_id or not self.scenario_id:
            raise ValueError("n4_candidate_set_identity_required")
        if self.split not in {"train", "validation", "holdout"}:
            raise ValueError("n4_candidate_set_split_invalid")
        if not self.candidates:
            raise ValueError("n4_candidate_set_empty")
        identifiers = [item.hypothesis_id for item in self.candidates]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("n4_candidate_set_duplicate_hypothesis")
        if identifiers != sorted(identifiers):
            raise ValueError("n4_candidate_set_order_invalid")
        availability = {item.risk_label_available for item in self.candidates}
        if len(availability) != 1:
            raise ValueError("candidate_set_risk_supervision_incomplete")
        versions = {item.risk_label_version for item in self.candidates}
        if len(versions) != 1:
            raise ValueError("candidate_set_mixed_risk_versions")

    @property
    def logical_hash(self) -> str:
        payload = {
            "candidate_set_id": self.candidate_set_id,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "split": self.split,
            "candidates": [
                {
                    "hypothesis_id": item.hypothesis_id,
                    "candidate_source": item.candidate_source,
                    "features": list(item.features),
                    "valid_label": item.valid_label,
                    "mae_gain_label": item.mae_gain_label,
                    "risk_label": item.invariant_risk_label,
                    "risk_label_available": item.risk_label_available,
                    "risk_label_version": item.risk_label_version,
                    "risk_report_sha256": item.risk_report_sha256,
                    "safety_contract_version": item.safety_contract_version,
                    "rollout_horizon": item.rollout_horizon,
                    "risk_event_families": list(item.risk_event_families),
                    "risk_hazard_first_step": item.risk_hazard_first_step,
                }
                for item in self.candidates
            ],
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CandidateSetBatch:
    features: Any
    valid_labels: Any
    gain_labels: Any
    risk_labels: Any
    risk_label_mask: Any
    candidate_mask: Any
    candidate_set_ids: tuple[str, ...]


@dataclass(frozen=True)
class N4ModelOutput:
    rank_score: Any
    validity_logit: Any
    expected_mae_gain: Any
    risk_logit: Any


def load_candidate_sets(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[CandidateSetSample, ...]:
    """Agrupa filas JSONL v1 en conjuntos deterministas y split-exclusivos."""
    grouped: dict[str, list[N4CandidateRecord]] = {}
    metadata: dict[str, tuple[str, int, str]] = {}
    for row in rows:
        try:
            candidate_set_id = str(row["candidate_set_id"])
            scenario_id = str(row["scenario"])
            seed = int(row["seed"])
            split = str(row["split"])
            feature_map = row["features"]
            risk_available = bool(
                row.get("risk_label_available", False)
            )
            risk_report = row.get("risk_report")
            risk_report_sha256 = row.get("risk_report_sha256")
            if risk_available:
                if risk_report is None or risk_report_sha256 is None:
                    raise ValueError("risk_report_required")
                encoded_report = json.dumps(
                    risk_report,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                if hashlib.sha256(encoded_report).hexdigest() != str(
                    risk_report_sha256
                ):
                    raise ValueError("risk_report_hash_mismatch")
            risk_events = (
                tuple((risk_report or {}).get("oracle_events", ()))
                if risk_available
                else ()
            )
            record = N4CandidateRecord(
                hypothesis_id=str(row["hypothesis_id"]),
                candidate_source=str(row["source"]),
                features=tuple(float(feature_map[name]) for name in FEATURE_NAMES),
                valid_label=bool(row["valid"]),
                mae_gain_label=float(row["mae_gain"]),
                invariant_risk_label=(
                    float(row["risk_label"])
                    if risk_available
                    else None
                ),
                risk_label_available=risk_available,
                risk_label_version=row.get("risk_label_version"),
                risk_report_sha256=risk_report_sha256,
                safety_contract_version=row.get("safety_contract_version"),
                rollout_horizon=row.get("rollout_horizon"),
                risk_components=row.get("risk_components"),
                risk_event_families=tuple(
                    sorted(
                        {
                            str(event["predicate_id"]).split(":", 1)[0]
                            for event in risk_events
                        }
                    )
                ),
                risk_hazard_first_step=(
                    min(int(event["step"]) for event in risk_events)
                    if risk_events
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("n4_candidate_row_invalid") from exc
        identity = (scenario_id, seed, split)
        previous = metadata.setdefault(candidate_set_id, identity)
        if previous != identity:
            if previous[2] != split:
                raise ValueError("n4_candidate_set_crosses_splits")
            raise ValueError("n4_candidate_set_metadata_conflict")
        grouped.setdefault(candidate_set_id, []).append(record)
    if not grouped:
        raise ValueError("n4_dataset_has_no_candidate_sets")
    return tuple(
        CandidateSetSample(
            candidate_set_id=candidate_set_id,
            scenario_id=metadata[candidate_set_id][0],
            seed=metadata[candidate_set_id][1],
            split=metadata[candidate_set_id][2],
            candidates=tuple(
                sorted(
                    grouped[candidate_set_id],
                    key=lambda item: item.hypothesis_id,
                )
            ),
        )
        for candidate_set_id in sorted(grouped)
    )


def collate_candidate_sets(
    samples: Sequence[CandidateSetSample],
    *,
    device: Any = None,
) -> CandidateSetBatch:
    """Rellena conjuntos variables sin convertir padding en candidatos reales."""
    if not samples:
        raise ValueError("n4_candidate_batch_empty")
    import torch

    max_candidates = max(len(item.candidates) for item in samples)
    shape = (len(samples), max_candidates)
    features = torch.zeros((*shape, len(FEATURE_NAMES)), dtype=torch.float32)
    valid = torch.zeros(shape, dtype=torch.float32)
    gain = torch.zeros(shape, dtype=torch.float32)
    risk = torch.zeros(shape, dtype=torch.float32)
    risk_mask = torch.zeros(shape, dtype=torch.bool)
    mask = torch.zeros(shape, dtype=torch.bool)
    for set_index, sample in enumerate(samples):
        for candidate_index, candidate in enumerate(sample.candidates):
            features[set_index, candidate_index] = torch.tensor(candidate.features)
            valid[set_index, candidate_index] = float(candidate.valid_label)
            gain[set_index, candidate_index] = candidate.mae_gain_label
            if candidate.risk_label_available:
                risk[set_index, candidate_index] = float(
                    candidate.invariant_risk_label
                )
                risk_mask[set_index, candidate_index] = True
            mask[set_index, candidate_index] = True
    return CandidateSetBatch(
        features=features.to(device=device),
        valid_labels=valid.to(device=device),
        gain_labels=gain.to(device=device),
        risk_labels=risk.to(device=device),
        risk_label_mask=risk_mask.to(device=device),
        candidate_mask=mask.to(device=device),
        candidate_set_ids=tuple(item.candidate_set_id for item in samples),
    )


def compare_candidates(a: N4CandidateRecord, b: N4CandidateRecord) -> int:
    """Preferencia seguridad > validez > ganancia, sin inventar empates."""
    if a.risk_label_available != b.risk_label_available:
        raise ValueError("n4_pair_mixed_risk_availability")
    if a.risk_label_available:
        if (
            float(a.invariant_risk_label) + RISK_EPSILON
            < float(b.invariant_risk_label)
        ):
            return 1
        if (
            float(b.invariant_risk_label) + RISK_EPSILON
            < float(a.invariant_risk_label)
        ):
            return -1
    if a.valid_label != b.valid_label:
        return 1 if a.valid_label else -1
    if a.valid_label and b.valid_label:
        if a.mae_gain_label > b.mae_gain_label + GAIN_EPSILON:
            return 1
        if b.mae_gain_label > a.mae_gain_label + GAIN_EPSILON:
            return -1
    return 0


def informative_pairs(
    sample: CandidateSetSample,
) -> tuple[tuple[int, int], ...]:
    """Devuelve una sola orientación (preferido, no preferido) por par."""
    pairs: list[tuple[int, int]] = []
    for left in range(len(sample.candidates)):
        for right in range(left + 1, len(sample.candidates)):
            preference = compare_candidates(
                sample.candidates[left], sample.candidates[right]
            )
            if preference > 0:
                pairs.append((left, right))
            elif preference < 0:
                pairs.append((right, left))
    return tuple(pairs)


def create_n4_ranker_v2(torch, *, hidden_dim: int = 16):
    """Construye el modelo exportable sin cargar torch al importar el módulo."""
    if hidden_dim < 1:
        raise ValueError("n4_v2_hidden_dim_invalid")

    class N4RankerV2(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = torch.nn.Sequential(
                torch.nn.Linear(len(FEATURE_NAMES), hidden_dim),
                torch.nn.ReLU(),
            )
            self.rank_head = torch.nn.Linear(hidden_dim, 1)
            self.validity_head = torch.nn.Linear(hidden_dim, 1)
            self.gain_head = torch.nn.Linear(hidden_dim, 1)
            self.risk_head = torch.nn.Linear(hidden_dim, 1)

        def forward(self, features):
            hidden = self.trunk(features)
            return N4ModelOutput(
                rank_score=self.rank_head(hidden).squeeze(-1),
                validity_logit=self.validity_head(hidden).squeeze(-1),
                expected_mae_gain=torch.tanh(
                    self.gain_head(hidden).squeeze(-1)
                ),
                risk_logit=self.risk_head(hidden).squeeze(-1),
            )

    return N4RankerV2()


def n4_multitask_loss(
    torch,
    output: N4ModelOutput,
    batch: CandidateSetBatch,
    samples: Sequence[CandidateSetSample],
    *,
    loss_weights: Mapping[str, float] = V2_LOSS_WEIGHTS,
):
    """Pérdida v2: primero normaliza pares por set, luego pointwise por máscara."""
    if len(samples) != len(batch.candidate_set_ids):
        raise ValueError("n4_candidate_batch_sample_mismatch")
    set_losses = []
    for set_index, sample in enumerate(samples):
        pairs = informative_pairs(sample)
        if pairs:
            differences = torch.stack(
                [
                    output.rank_score[set_index, preferred]
                    - output.rank_score[set_index, other]
                    for preferred, other in pairs
                ]
            )
            set_losses.append(torch.nn.functional.softplus(-differences).mean())
    zero = output.rank_score.sum() * 0.0
    pair_loss = torch.stack(set_losses).mean() if set_losses else zero
    mask = batch.candidate_mask
    valid_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output.validity_logit[mask],
        batch.valid_labels[mask],
    )
    positive_mask = mask & batch.valid_labels.bool()
    gain_loss = (
        torch.nn.functional.smooth_l1_loss(
            output.expected_mae_gain[positive_mask],
            batch.gain_labels[positive_mask],
        )
        if bool(positive_mask.any())
        else zero
    )
    risk_mask = mask & batch.risk_label_mask
    risk_loss = (
        torch.nn.functional.smooth_l1_loss(
            torch.sigmoid(output.risk_logit[risk_mask]),
            batch.risk_labels[risk_mask],
        )
        if bool(risk_mask.any())
        else zero
    )
    components = {
        "rank": pair_loss,
        "valid": valid_loss,
        "gain": gain_loss,
        "risk": risk_loss,
    }
    total = sum(
        float(loss_weights[name]) * component
        for name, component in components.items()
    )
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("n4_multitask_loss_nonfinite")
    return total, components


def train_n4_ranking_v2(
    samples: Sequence[CandidateSetSample],
    *,
    artifact_path: Path,
    seed: int = 42,
    epochs: int = 300,
    patience: int = 30,
    loss_weights: Mapping[str, float] = V2_LOSS_WEIGHTS,
    sampling_policy: str = "natural",
    epoch_history_path: Path | None = None,
    training_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Entrenamiento de desarrollo: train+validation, nunca holdout."""
    import torch

    from .n4_campaign import sample_candidate_sets_for_epoch

    effective_weights = _validated_loss_weights(loss_weights)
    if any(item.split == "holdout" for item in samples):
        raise ValueError("n4_development_trainer_rejects_holdout")
    split_sets = {
        split: tuple(item for item in samples if item.split == split)
        for split in ("train", "validation")
    }
    if any(not rows for rows in split_sets.values()):
        raise ValueError("n4_v2_requires_nonempty_explicit_splits")
    ids = [item.candidate_set_id for item in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("n4_v2_candidate_set_ids_must_be_unique")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_n4_ranker_v2(torch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    validation_batch = collate_candidate_sets(
        split_sets["validation"], device=device
    )
    risk_values = _supervised_risk_values(split_sets["train"])
    supervised_sets = [
        item
        for item in split_sets["train"]
        if item.candidates[0].risk_label_available
    ]
    risk_target_informative = _risk_target_is_informative(supervised_sets)
    history_path = epoch_history_path or artifact_path.with_suffix(
        ".history.jsonl"
    )
    if history_path.exists():
        raise FileExistsError("n4_epoch_history_must_be_new")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    best_state = None
    best_key = None
    best_epoch = None
    stale = 0
    epochs_completed = 0
    history_records = []
    for epoch in range(epochs):
        epochs_completed = epoch + 1
        epoch_samples, sampling_report = sample_candidate_sets_for_epoch(
            split_sets["train"],
            policy=sampling_policy,
            model_seed=seed,
            epoch=epoch + 1,
        )
        train_batch = collate_candidate_sets(epoch_samples, device=device)
        model.train()
        output = model(train_batch.features)
        loss, components = n4_multitask_loss(
            torch,
            output,
            train_batch,
            epoch_samples,
            loss_weights=effective_weights,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        validation_metrics = _candidate_set_metrics_v2(
            torch,
            model,
            validation_batch,
            split_sets["validation"],
            calibration={"a": 1.0, "b": 0.0},
        )
        feasible = (
            validation_metrics["brier"] < 0.10
            and validation_metrics["ece"] < 0.05
            and risk_target_informative
            and all(
                math.isfinite(float(value))
                for value in validation_metrics.values()
                if isinstance(value, (int, float))
            )
        )
        history_record = {
            "epoch": epoch + 1,
            "train_total_loss": float(loss.detach().cpu()),
            "train_pair_loss": float(components["rank"].detach().cpu()),
            "train_validity_loss": float(
                components["valid"].detach().cpu()
            ),
            "train_gain_loss": float(components["gain"].detach().cpu()),
            "train_risk_loss": float(components["risk"].detach().cpu()),
            "validation_recall_at_1": validation_metrics["recall_at_1"],
            "validation_recall_at_2": validation_metrics["recall_at_2"],
            "validation_mrr": validation_metrics["mrr"],
            "validation_ndcg_at_2": validation_metrics["ndcg_at_2"],
            "validation_brier": validation_metrics["brier"],
            "validation_ece": validation_metrics["ece"],
            "validation_risk_at_2": validation_metrics["top2_risk"],
            "validation_risk_mae": validation_metrics["risk_mae"],
            "validation_false_safe_rate": validation_metrics[
                "false_safe_rate"
            ],
            "checkpoint_feasible": feasible,
            "sampling": sampling_report,
        }
        history_records.append(history_record)
        with history_path.open("ab") as handle:
            handle.write(
                json.dumps(
                    history_record,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                + b"\n"
            )
        key = (
            int(feasible),
            validation_metrics["recall_at_2"],
            validation_metrics["mrr"],
            validation_metrics["ndcg_at_2"],
            validation_metrics["recall_at_1"],
            -validation_metrics["top2_risk"],
        )
        if best_key is None or key > best_key:
            best_key = key
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("n4_v2_training_failed_to_select_model")
    model.load_state_dict(best_state)
    before_calibration = _candidate_set_metrics_v2(
        torch,
        model,
        validation_batch,
        split_sets["validation"],
        calibration={"a": 1.0, "b": 0.0},
    )
    calibration = _calibrate_platt_v2(
        torch, model, validation_batch, device
    )
    validation_metrics = _candidate_set_metrics_v2(
        torch,
        model,
        validation_batch,
        split_sets["validation"],
        calibration=calibration,
    )
    state = model.state_dict()
    trunk = state["trunk.0.weight"]
    artifact_quality = all(
        bool(torch.isfinite(value).all()) for value in state.values()
    )
    calibration_gate = (
        validation_metrics["brier"] < 0.10
        and validation_metrics["ece"] < 0.05
    )
    history_sha256 = hashlib.sha256(history_path.read_bytes()).hexdigest()
    gates = {
        "artifact_quality": artifact_quality,
        "calibration": calibration_gate,
        "risk_target_informative": risk_target_informative,
        "scientific_holdout": False,
        "promotable": False,
    }
    artifact = {
        "schema": "n4-ranking-artifact.v2",
        "model_kind": "trained_multihead",
        "ranking_objective": "pairwise_constrained_v1",
        "dtype": "float32",
        "feature_order": list(FEATURE_NAMES),
        "feature_transform": {"kind": "identity"},
        "model": {
            "trunk": {
                "layers": [
                    {
                        "input_dim": len(FEATURE_NAMES),
                        "output_dim": int(trunk.shape[0]),
                        "activation": "relu",
                        "weights": _tensor_list(trunk),
                        "bias": _tensor_list(state["trunk.0.bias"]),
                    }
                ]
            },
            "heads": {
                name: {
                    "weights": _tensor_list(state[f"{name}_head.weight"][0]),
                    "bias": float(state[f"{name}_head.bias"][0]),
                }
                for name in ("rank", "validity", "gain", "risk")
            },
        },
        "calibration": {
            "kind": "platt",
            "target": "validity_logit",
            "a": calibration["a"],
            "b": calibration["b"],
            "status": calibration["status"],
            "fallback_reason": calibration["fallback_reason"],
            "fitted_a": calibration["fitted_a"],
            "fitted_b": calibration["fitted_b"],
            "fit_split": "validation",
            "sample_count": sum(
                len(item.candidates) for item in split_sets["validation"]
            ),
            "metrics_before": {
                "brier": before_calibration["brier"],
                "ece": before_calibration["ece"],
            },
            "metrics_after": {
                "brier": validation_metrics["brier"],
                "ece": validation_metrics["ece"],
            },
        },
        "dataset_lineage": dict(
            (training_metadata or {}).get("dataset_lineage") or {}
        ),
        "training_provenance": {
            "seed": seed,
            "epochs": epochs_completed,
            "selected_epoch": best_epoch,
            "patience": patience,
            "loss_weights": effective_weights,
            "sampling_policy": sampling_policy,
            "epoch_history_sha256": history_sha256,
            "epoch_history_path": history_path.name,
            **{
                key: value
                for key, value in dict(training_metadata or {}).items()
                if key != "dataset_lineage"
            },
            "risk_supervised_sample_count": len(risk_values),
            "risk_supervised_candidate_set_count": len(supervised_sets),
            "risk_label_versions": sorted(
                {
                    candidate.risk_label_version
                    for sample in supervised_sets
                    for candidate in sample.candidates
                }
            ),
            "risk_label_coverage": round(
                len(risk_values)
                / sum(len(item.candidates) for item in split_sets["train"]),
                9,
            ),
        },
        "validation_metrics": validation_metrics,
        "gates": gates,
    }
    encoded = json.dumps(
        artifact,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(encoded + b"\n")
    artifact["artifact_sha256"] = hashlib.sha256(encoded + b"\n").hexdigest()
    return artifact


def _validated_loss_weights(
    weights: Mapping[str, float],
) -> dict[str, float]:
    if set(weights) != {"rank", "valid", "gain", "risk"}:
        raise ValueError("n4_loss_weights_invalid")
    result = {name: float(value) for name, value in weights.items()}
    if any(
        not math.isfinite(value) or value < 0.0
        for value in result.values()
    ):
        raise ValueError("n4_loss_weights_invalid")
    if result["rank"] <= 0.0:
        raise ValueError("n4_rank_loss_weight_must_be_positive")
    return result


def _supervised_risk_values(samples):
    return [
        float(candidate.invariant_risk_label)
        for item in samples
        for candidate in item.candidates
        if candidate.risk_label_available
    ]


def _risk_target_is_informative(samples):
    values = _supervised_risk_values(samples)
    return bool(values) and max(values) > min(values) and any(
        max(float(item.invariant_risk_label) for item in sample.candidates)
        > min(float(item.invariant_risk_label) for item in sample.candidates)
        for sample in samples
    )


def _tensor_list(tensor):
    return tensor.detach().cpu().to(dtype=__import__("torch").float32).tolist()


def _calibrate_platt_v2(torch, model, batch, device):
    model.eval()
    with torch.no_grad():
        logits = model(batch.features).validity_logit[
            batch.candidate_mask
        ].detach()
        labels = batch.valid_labels[batch.candidate_mask]
    log_a = torch.zeros((), device=device, requires_grad=True)
    b = torch.zeros((), device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS((log_a, b), max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits * torch.exp(log_a) + b, labels
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return _validated_platt_calibration(
        float(torch.exp(log_a).detach().cpu()),
        float(b.detach().cpu()),
    )


def _candidate_set_metrics_v2(torch, model, batch, samples, calibration):
    model.eval()
    with torch.inference_mode():
        output = model(batch.features)
        probabilities = torch.sigmoid(
            output.validity_logit * calibration["a"] + calibration["b"]
        )
        predicted_risk = torch.sigmoid(output.risk_logit)
    mask = batch.candidate_mask
    probability_values = probabilities[mask].cpu().tolist()
    label_values = batch.valid_labels[mask].cpu().tolist()
    brier = sum(
        (prediction - label) ** 2
        for prediction, label in zip(probability_values, label_values)
    ) / len(label_values)
    bins = [[] for _ in range(10)]
    for prediction, label in zip(probability_values, label_values):
        bins[min(9, int(prediction * 10))].append((prediction, label))
    ece = sum(
        len(bucket) / len(label_values)
        * abs(
            sum(item[0] for item in bucket) / len(bucket)
            - sum(item[1] for item in bucket) / len(bucket)
        )
        for bucket in bins
        if bucket
    )
    recalls_1, recalls_2, reciprocal, ndcg, top2_risks = [], [], [], [], []
    available = 0
    for index, sample in enumerate(samples):
        safe_valid = {
            position
            for position, candidate in enumerate(sample.candidates)
            if candidate.valid_label
            and (
                not candidate.risk_label_available
                or float(candidate.invariant_risk_label) <= RISK_EPSILON
            )
        }
        order = sorted(
            range(len(sample.candidates)),
            key=lambda position: (
                -float(output.rank_score[index, position]),
                sample.candidates[position].hypothesis_id,
            ),
        )
        observed_top2_risks = [
            float(sample.candidates[position].invariant_risk_label)
            for position in order[:2]
            if sample.candidates[position].risk_label_available
        ]
        top2_risks.append(
            sum(observed_top2_risks) / len(observed_top2_risks)
            if observed_top2_risks
            else 0.0
        )
        if not safe_valid:
            continue
        available += 1
        recalls_1.append(float(bool(safe_valid & set(order[:1]))))
        recalls_2.append(float(bool(safe_valid & set(order[:2]))))
        first = next(
            (rank for rank, position in enumerate(order, 1) if position in safe_valid),
            0,
        )
        reciprocal.append(1.0 / first if first else 0.0)
        gains = [1.0 if position in safe_valid else 0.0 for position in order[:2]]
        dcg = sum(value / math.log2(rank + 1) for rank, value in enumerate(gains, 1))
        ideal_count = min(2, len(safe_valid))
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
        ndcg.append(dcg / ideal)
    risk_mask = mask & batch.risk_label_mask
    risk_predictions = predicted_risk[risk_mask].cpu().tolist()
    risk_targets = batch.risk_labels[risk_mask].cpu().tolist()
    risk_mae = (
        torch.abs(
            predicted_risk[risk_mask] - batch.risk_labels[risk_mask]
        ).mean()
        if bool(risk_mask.any())
        else predicted_risk.sum() * 0.0
    )
    absolute_errors = sorted(
        abs(prediction - target)
        for prediction, target in zip(risk_predictions, risk_targets)
    )
    risk_median_error = (
        absolute_errors[len(absolute_errors) // 2]
        if absolute_errors
        else 0.0
    )
    constant = (
        sum(risk_targets) / len(risk_targets) if risk_targets else 0.0
    )
    constant_mae = (
        sum(abs(target - constant) for target in risk_targets)
        / len(risk_targets)
        if risk_targets
        else 0.0
    )
    false_safe_denominator = sum(target >= 0.1 for target in risk_targets)
    false_safe_rate = (
        sum(
            target >= 0.1 and prediction < 0.1
            for prediction, target in zip(risk_predictions, risk_targets)
        )
        / false_safe_denominator
        if false_safe_denominator
        else 0.0
    )
    family_errors: dict[str, list[float]] = {}
    horizon_errors: dict[int, list[float]] = {}
    for set_index, sample in enumerate(samples):
        for candidate_index, candidate in enumerate(sample.candidates):
            if not candidate.risk_label_available:
                continue
            error = abs(
                float(predicted_risk[set_index, candidate_index])
                - float(candidate.invariant_risk_label)
            )
            for family in candidate.risk_event_families:
                family_errors.setdefault(family, []).append(error)
            if candidate.risk_hazard_first_step is not None:
                horizon_errors.setdefault(
                    candidate.risk_hazard_first_step, []
                ).append(error)
    mean = lambda values: sum(values) / len(values) if values else 0.0
    return {
        "brier": round(brier, 9),
        "ece": round(ece, 9),
        "recall_at_1": round(mean(recalls_1), 9),
        "recall_at_2": round(mean(recalls_2), 9),
        "mrr": round(mean(reciprocal), 9),
        "ndcg_at_2": round(mean(ndcg), 9),
        "top2_risk": round(mean(top2_risks), 9),
        "risk_mae": round(float(risk_mae.cpu()), 9),
        "risk_median_error": round(risk_median_error, 9),
        "risk_constant_predictor_mae": round(constant_mae, 9),
        "risk_spearman": round(
            _spearman(risk_predictions, risk_targets), 9
        ),
        "false_safe_rate": round(false_safe_rate, 9),
        "risk_mae_by_event_family": {
            name: round(sum(values) / len(values), 9)
            for name, values in sorted(family_errors.items())
        },
        "risk_mae_by_hazard_first_step": {
            str(step): round(sum(values) / len(values), 9)
            for step, values in sorted(horizon_errors.items())
        },
        "candidate_availability_rate": round(available / len(samples), 9),
        "candidate_set_count": len(samples),
    }


def _spearman(left, right):
    if len(left) < 2 or len(set(left)) < 2 or len(set(right)) < 2:
        return 0.0
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_ranks, right_ranks)
    )
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left_ranks)
        * sum((b - right_mean) ** 2 for b in right_ranks)
    )
    return numerator / denominator if denominator else 0.0


def _average_ranks(values):
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while (
            end < len(ordered)
            and values[ordered[end]] == values[ordered[start]]
        ):
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in ordered[start:end]:
            result[index] = rank
        start = end
    return result


def score_n4_validity(
    features: Sequence[float], artifact: dict[str, Any]
) -> float:
    """Puntuación pura usada para comprobar paridad de exportación/runtime."""
    if len(features) != len(FEATURE_NAMES):
        raise ValueError("n4_training_feature_shape_invalid")
    weights = tuple(float(item) for item in artifact["ranking_weights"])
    if len(weights) != len(FEATURE_NAMES):
        raise ValueError("n4_ranking_weight_shape_invalid")
    logit = float(artifact.get("bias", 0.0)) + sum(
        weight * float(value) for weight, value in zip(weights, features)
    )
    temperature = max(float(artifact.get("temperature", 1.0)), 1e-6)
    return 1.0 / (1.0 + math.exp(-logit / temperature))


def score_n4_artifact_v2(
    features: Sequence[float], artifact: Mapping[str, Any]
) -> dict[str, float]:
    """Referencia offline independiente para comprobar paridad del runtime."""
    import torch

    if len(features) != len(FEATURE_NAMES):
        raise ValueError("n4_training_feature_shape_invalid")
    layer = artifact["model"]["trunk"]["layers"][0]
    values = torch.tensor(features, dtype=torch.float32)
    weights = torch.tensor(layer["weights"], dtype=torch.float32)
    bias = torch.tensor(layer["bias"], dtype=torch.float32)
    hidden = torch.relu(weights @ values + bias)
    outputs = {}
    for name, head in artifact["model"]["heads"].items():
        outputs[name] = float(
            torch.tensor(head["weights"], dtype=torch.float32) @ hidden
            + float(head["bias"])
        )
    calibration = artifact["calibration"]
    return {
        "rank_score": outputs["rank"],
        "validity_logit": outputs["validity"],
        "validity_probability": _stable_sigmoid(
            float(calibration["a"]) * outputs["validity"]
            + float(calibration["b"])
        ),
        "expected_mae_gain": math.tanh(outputs["gain"]),
        "invariant_risk": _stable_sigmoid(outputs["risk"]),
    }


def _validated_platt_calibration(a: float, b: float) -> dict[str, Any]:
    """Valida Platt y aplica el fallback preinscrito sin usar los targets."""
    a = float(a)
    b = float(b)
    reason = None
    if not math.isfinite(a) or not math.isfinite(b):
        reason = "nonfinite_coefficients"
    elif a < 1e-6:
        reason = "scale_below_minimum"
    elif a > 1e6:
        reason = "scale_above_maximum"
    elif abs(b) > 1e6:
        reason = "intercept_out_of_range"
    if reason is not None:
        return {
            "a": 1.0,
            "b": 0.0,
            "status": "identity_fallback",
            "fallback_reason": reason,
            "fitted_a": a if math.isfinite(a) else None,
            "fitted_b": b if math.isfinite(b) else None,
        }
    return {
        "a": round(a, 12),
        "b": round(b, 12),
        "status": "fitted",
        "fallback_reason": None,
        "fitted_a": round(a, 12),
        "fitted_b": round(b, 12),
    }


def _stable_sigmoid(value: float) -> float:
    """Sigmoide estable para logits finitos de cualquier magnitud."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("n4_sigmoid_input_nonfinite")
    if value >= 0.0:
        tail = math.exp(-value)
        return 1.0 / (1.0 + tail)
    head = math.exp(value)
    return head / (1.0 + head)


@dataclass(frozen=True)
class N4TrainingSample:
    features: tuple[float, ...]
    valid: bool
    mae_gain: float
    invariant_risk: float
    scenario: str
    seed: int
    logical_time: int
    split: str | None = None
    risk_label: float | None = None
    risk_label_available: bool = False
    risk_label_version: str | None = None
    risk_report_sha256: str | None = None
    safety_contract_version: str | None = None
    rollout_horizon: int | None = None
    risk_components: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if len(self.features) != len(FEATURE_NAMES):
            raise ValueError("n4_training_feature_shape_invalid")
        values = (*self.features, self.mae_gain, self.invariant_risk)
        if not all(math.isfinite(float(item)) for item in values):
            raise ValueError("n4_training_values_must_be_finite")
        if not self.risk_label_available:
            if any(
                item is not None
                for item in (
                    self.risk_label,
                    self.risk_label_version,
                    self.risk_report_sha256,
                    self.safety_contract_version,
                    self.rollout_horizon,
                )
            ):
                raise ValueError("n4_unavailable_risk_metadata_must_be_null")
        else:
            if self.risk_label is None or not math.isfinite(self.risk_label):
                raise ValueError("n4_available_risk_label_must_be_finite")
            if not 0.0 <= self.risk_label <= 1.0:
                raise ValueError("n4_training_risk_label_out_of_range")
            if self.risk_label_version != RISK_LABEL_VERSION:
                raise ValueError("n4_training_risk_version_unknown")


def train_n4_ranking(
    samples: Sequence[N4TrainingSample],
    *,
    artifact_path: Path,
    seed: int = 42,
    epochs: int = 300,
    patience: int = 30,
    training_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Entrena heads multitarea; particiona por escenario/seed/tiempo."""
    if len(samples) < 30:
        raise ValueError("n4_training_requires_at_least_30_samples")
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    ordered = sorted(
        samples, key=lambda item: (item.scenario, item.seed, item.logical_time)
    )
    train, validation, holdout = _grouped_split(ordered)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class Ranker(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.validity = torch.nn.Linear(len(FEATURE_NAMES), 1)
            self.gain = torch.nn.Linear(len(FEATURE_NAMES), 1)
            self.risk = torch.nn.Linear(len(FEATURE_NAMES), 1)

        def forward(self, values):
            return (
                self.validity(values).squeeze(-1),
                self.gain(values).squeeze(-1),
                torch.sigmoid(self.risk(values).squeeze(-1)),
            )

    model = Ranker().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    best_state = None
    best_loss = float("inf")
    stale = 0
    epochs_completed = 0
    for epoch in range(epochs):
        epochs_completed = epoch + 1
        model.train()
        x, valid, gain, risk = _tensors(torch, train, device)
        logits, gain_prediction, risk_prediction = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, valid)
        loss = loss + torch.nn.functional.smooth_l1_loss(gain_prediction, gain)
        loss = loss + torch.nn.functional.binary_cross_entropy(risk_prediction, risk)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        validation_loss = _loss(torch, model, validation, device)
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("n4_training_failed_to_select_model")
    model.load_state_dict(best_state)
    calibration = _calibrate_platt(torch, model, validation, device)
    temperature = 1.0
    validation_metrics = _metrics(torch, model, validation, device, temperature)
    holdout_metrics = _metrics(torch, model, holdout, device, temperature)
    validity = model.validity
    artifact = {
        "schema": "n4-ranking-artifact.v1",
        "model_kind": "trained",
        "feature_names": list(FEATURE_NAMES),
        "ranking_weights": [
            round(float(item), 12)
            for item in validity.weight.detach().cpu().flatten()
        ],
        "bias": round(float(validity.bias.detach().cpu()[0]), 12),
        "temperature": temperature,
        "training": {
            "seed": seed,
            "optimizer": "AdamW",
            "epochs_completed": epochs_completed,
            "split": "grouped_scenario_seed_60_20_20",
            "sample_count": len(ordered),
            **dict(training_metadata or {}),
        },
        "validation": validation_metrics,
        "holdout": holdout_metrics,
        "calibration": calibration,
        "promotable": bool(
            holdout_metrics["brier"] < 0.15
            and holdout_metrics["ece"] <= 0.10
            and len(holdout) > 0
        ),
    }
    encoded = json.dumps(
        artifact,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(encoded + b"\n")
    artifact["artifact_sha256"] = hashlib.sha256(encoded + b"\n").hexdigest()
    return artifact


def _grouped_split(samples):
    groups: dict[tuple[str, int], list[Any]] = {}
    for sample in samples:
        groups.setdefault((sample.scenario, sample.seed), []).append(sample)
    keys = sorted(groups)
    explicit = {item.split for item in samples}
    if explicit != {None}:
        if None in explicit or not explicit <= {"train", "validation", "holdout"}:
            raise ValueError("n4_training_split_labels_invalid")
        assignment: dict[tuple[str, int], str] = {}
        for key, rows in groups.items():
            labels = {item.split for item in rows}
            if len(labels) != 1:
                raise ValueError("n4_training_seed_group_crosses_splits")
            assignment[key] = str(next(iter(labels)))
        train = [
            row for key in keys if assignment[key] == "train" for row in groups[key]
        ]
        validation = [
            row
            for key in keys
            if assignment[key] == "validation"
            for row in groups[key]
        ]
        holdout = [
            row for key in keys if assignment[key] == "holdout" for row in groups[key]
        ]
        if not train or not validation or not holdout:
            raise ValueError("n4_training_split_requires_nonempty_groups")
        return train, validation, holdout
    if len(keys) < 3:
        raise ValueError("n4_training_split_requires_at_least_three_seed_groups")
    first = max(1, int(len(keys) * 0.60))
    second = max(first + 1, int(len(keys) * 0.80))
    second = min(second, len(keys) - 1)
    train_keys = set(keys[:first])
    validation_keys = set(keys[first:second])
    holdout_keys = set(keys[second:])
    train = [row for key in keys if key in train_keys for row in groups[key]]
    validation = [
        row for key in keys if key in validation_keys for row in groups[key]
    ]
    holdout = [row for key in keys if key in holdout_keys for row in groups[key]]
    if not validation or not holdout:
        raise ValueError("n4_training_split_requires_multiple_temporal_samples")
    return train, validation, holdout


def _tensors(torch, rows, device):
    return (
        torch.tensor([item.features for item in rows], dtype=torch.float32, device=device),
        torch.tensor([float(item.valid) for item in rows], dtype=torch.float32, device=device),
        torch.tensor([item.mae_gain for item in rows], dtype=torch.float32, device=device),
        torch.tensor([item.invariant_risk for item in rows], dtype=torch.float32, device=device),
    )


def _loss(torch, model, rows, device):
    model.eval()
    with torch.inference_mode():
        x, valid, gain, risk = _tensors(torch, rows, device)
        logits, gain_prediction, risk_prediction = model(x)
        return float(
            (
                torch.nn.functional.binary_cross_entropy_with_logits(logits, valid)
                + torch.nn.functional.smooth_l1_loss(gain_prediction, gain)
                + torch.nn.functional.binary_cross_entropy(risk_prediction, risk)
            ).cpu()
        )


def _calibrate_platt(torch, model, rows, device) -> dict[str, float | str]:
    x, valid, _, _ = _tensors(torch, rows, device)
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(x)
    logits = logits.detach()
    log_scale = torch.zeros((), device=device, requires_grad=True)
    offset = torch.zeros((), device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        (log_scale, offset),
        lr=0.25,
        max_iter=100,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        calibrated = logits * torch.exp(log_scale) + offset
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            calibrated, valid
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    scale = float(torch.exp(log_scale).detach().cpu())
    intercept = float(offset.detach().cpu())
    with torch.no_grad():
        model.validity.weight.mul_(scale)
        model.validity.bias.mul_(scale).add_(intercept)
    return {
        "kind": "platt_positive_scale",
        "scale": round(scale, 12),
        "intercept": round(intercept, 12),
        "fit_split": "validation",
    }


def _metrics(torch, model, rows, device, temperature):
    x, valid, _, _ = _tensors(torch, rows, device)
    model.eval()
    with torch.inference_mode():
        logits, _, _ = model(x)
        probabilities = torch.sigmoid(logits / temperature).cpu().tolist()
        labels = valid.cpu().tolist()
    brier = sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / len(labels)
    bins = [[] for _ in range(10)]
    for probability, label in zip(probabilities, labels):
        bins[min(9, int(probability * 10))].append((probability, label))
    ece = sum(
        len(bucket) / len(labels)
        * abs(
            sum(item[0] for item in bucket) / len(bucket)
            - sum(item[1] for item in bucket) / len(bucket)
        )
        for bucket in bins
        if bucket
    )
    return {"brier": round(brier, 9), "ece": round(ece, 9), "count": len(labels)}
