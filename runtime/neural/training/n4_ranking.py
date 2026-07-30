"""Entrenamiento PyTorch/CUDA y exportación a inferencia pura."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


FEATURE_NAMES = (
    "empirical_gain",
    "holdout_support",
    "coverage",
    "simplicity",
    "stability",
    "invariant_safety",
)


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

    def __post_init__(self) -> None:
        if len(self.features) != len(FEATURE_NAMES):
            raise ValueError("n4_training_feature_shape_invalid")
        values = (*self.features, self.mae_gain, self.invariant_risk)
        if not all(math.isfinite(float(item)) for item in values):
            raise ValueError("n4_training_values_must_be_finite")


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
    for _ in range(epochs):
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
    temperature = _calibrate_temperature(torch, model, validation, device)
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
            "epochs_completed": epochs - max(0, patience - stale),
            "split": "grouped_scenario_seed_60_20_20",
            "sample_count": len(ordered),
            **dict(training_metadata or {}),
        },
        "validation": validation_metrics,
        "holdout": holdout_metrics,
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


def _calibrate_temperature(torch, model, rows, device) -> float:
    x, valid, _, _ = _tensors(torch, rows, device)
    model.eval()
    with torch.inference_mode():
        logits, _, _ = model(x)
        choices = (0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0)
        return min(
            choices,
            key=lambda value: float(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    logits / value, valid
                ).cpu()
            ),
        )


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
