"""Entrenamiento CPU offline de N1 MLP y N4 message-passing."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from runtime.neural.contracts import NeuralModelManifest
from runtime.neural.organs.n1_router import FAMILY_CATALOG_V2
from runtime.neural.organs.n4_causal import (
    ARTIFACT_SCHEMA_VERSION,
    EdgeType,
    NodeType,
)
from runtime.neural.training.n1_dataset import CounterfactualSample, DatasetQualityReport


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ece(probabilities: Sequence[float], labels: Sequence[float], bins: int = 10) -> float:
    if not probabilities:
        return 1.0
    total = len(probabilities)
    value = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [
            position
            for position, probability in enumerate(probabilities)
            if lower <= probability < upper or (index == bins - 1 and probability == 1.0)
        ]
        if members:
            confidence = sum(probabilities[item] for item in members) / len(members)
            accuracy = sum(labels[item] for item in members) / len(members)
            value += len(members) / total * abs(confidence - accuracy)
    return value


def _n1_split_metrics(
    model: Any,
    samples: Sequence[CounterfactualSample],
    *,
    feature_names: Sequence[str],
    family_index: Mapping[str, int],
    torch: Any,
    temperature: float = 1.0,
) -> dict[str, Any]:
    if not samples:
        return {"records": 0, "evaluated": False}
    if any(sample.family not in family_index for sample in samples):
        raise ValueError("n1_evaluation_family_outside_catalog_v2")
    x = torch.tensor(
        [
            [float(sample.features.get(name, 0.0)) for name in feature_names]
            for sample in samples
        ],
        dtype=torch.float32,
    )
    family = torch.tensor(
        [family_index[sample.family] for sample in samples], dtype=torch.long
    )
    utility_target = torch.tensor(
        [sample.utility_delta for sample in samples], dtype=torch.float32
    )
    positive_target = torch.tensor(
        [float(sample.positive_utility) for sample in samples], dtype=torch.float32
    )
    with torch.inference_mode():
        utility, logits = model(x)
        selected_utility = utility.gather(1, family[:, None]).squeeze(1)
        selected_logits = logits.gather(1, family[:, None]).squeeze(1)
        probabilities = torch.sigmoid(selected_logits / max(float(temperature), 1e-6))
        utility_rmse = torch.sqrt(
            torch.nn.functional.mse_loss(selected_utility, utility_target)
        )
        binary_cross_entropy = torch.nn.functional.binary_cross_entropy(
            probabilities, positive_target
        )
        accuracy = ((probabilities >= 0.5) == (positive_target >= 0.5)).float().mean()
    probability_values = probabilities.tolist()
    label_values = positive_target.tolist()
    return {
        "records": len(samples),
        "evaluated": True,
        "utility_rmse": float(utility_rmse),
        "positive_bce": float(binary_cross_entropy),
        "positive_accuracy": float(accuracy),
        "calibration_ece": _ece(probability_values, label_values),
        "temperature": float(temperature),
    }


def _fit_validation_temperature(
    model: Any,
    samples: Sequence[CounterfactualSample],
    *,
    feature_names: Sequence[str],
    family_index: Mapping[str, int],
    torch: Any,
) -> float:
    """Fit one scalar on validation only using a deterministic log grid.

    A grid avoids optimizer-state variance and, unlike fitting on the exposed test
    partition, preserves the validation/test boundary.  The bounded range matches
    the reference backend's numerical protections.
    """
    if not samples:
        return 1.0
    x = torch.tensor(
        [[float(sample.features.get(name, 0.0)) for name in feature_names] for sample in samples],
        dtype=torch.float32,
    )
    family = torch.tensor(
        [family_index[sample.family] for sample in samples], dtype=torch.long
    )
    labels = torch.tensor(
        [float(sample.positive_utility) for sample in samples], dtype=torch.float32
    )
    with torch.inference_mode():
        _, logits = model(x)
        selected = logits.gather(1, family[:, None]).squeeze(1)
        candidates = [
            math.exp(math.log(0.05) + index * (math.log(10.0) - math.log(0.05)) / 400)
            for index in range(401)
        ]
        scored = []
        for temperature in candidates:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                selected / temperature, labels
            )
            scored.append((float(loss), abs(math.log(temperature)), temperature))
    return min(max(float(min(scored)[2]), 0.05), 10.0)


def train_n1_router(
    samples: Sequence[CounterfactualSample],
    report: DatasetQualityReport,
    *,
    artifact_root: str | Path,
    seed: int = 31,
    epochs: int = 80,
    dataset_classification: str = "caller_supplied_counterfactual",
    validation_samples: Sequence[CounterfactualSample] = (),
    test_samples: Sequence[CounterfactualSample] = (),
) -> tuple[NeuralModelManifest, dict[str, Any]]:
    if not report.training_ready():
        raise ValueError("n1_dataset_quality_gate_failed")
    import torch

    torch.manual_seed(seed)
    feature_names = tuple(sorted({name for sample in samples for name in sample.features}))
    if not feature_names:
        raise ValueError("n1_training_features_required")
    family_index = {name: index for index, name in enumerate(FAMILY_CATALOG_V2)}
    rows = [sample for sample in samples if sample.family in family_index]
    if len(rows) != len(samples):
        raise ValueError("n1_training_family_outside_catalog_v2")
    x = torch.tensor(
        [[float(sample.features.get(name, 0.0)) for name in feature_names] for sample in rows],
        dtype=torch.float32,
    )
    family = torch.tensor([family_index[sample.family] for sample in rows], dtype=torch.long)
    utility_target = torch.tensor([sample.utility_delta for sample in rows], dtype=torch.float32)
    positive_target = torch.tensor([float(sample.positive_utility) for sample in rows], dtype=torch.float32)

    class Router(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer1 = torch.nn.Linear(len(feature_names), 16)
            self.layer2 = torch.nn.Linear(16, 12)
            self.utility = torch.nn.Linear(12, len(FAMILY_CATALOG_V2), bias=False)
            self.probability = torch.nn.Linear(12, len(FAMILY_CATALOG_V2), bias=False)

        def forward(self, values: Any) -> tuple[Any, Any]:
            hidden = torch.nn.functional.silu(self.layer1(values))
            hidden = torch.nn.functional.silu(self.layer2(hidden))
            return self.utility(hidden), self.probability(hidden)

    model = Router()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    losses = []
    for _epoch in range(max(1, int(epochs))):
        optimizer.zero_grad(set_to_none=True)
        utility, logits = model(x)
        selected_utility = utility.gather(1, family[:, None]).squeeze(1)
        selected_logits = logits.gather(1, family[:, None]).squeeze(1)
        loss = torch.nn.functional.mse_loss(selected_utility, utility_target)
        loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
            selected_logits, positive_target
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    split_ids = {
        "train": {sample.pair_id for sample in rows},
        "validation": {sample.pair_id for sample in validation_samples},
        "test": {sample.pair_id for sample in test_samples},
    }
    if (
        split_ids["train"].intersection(split_ids["validation"])
        or split_ids["train"].intersection(split_ids["test"])
        or split_ids["validation"].intersection(split_ids["test"])
    ):
        raise ValueError("n1_train_validation_test_overlap")
    uncalibrated_validation = _n1_split_metrics(
        model,
        validation_samples,
        feature_names=feature_names,
        family_index=family_index,
        torch=torch,
        temperature=1.0,
    )
    temperature = _fit_validation_temperature(
        model,
        validation_samples,
        feature_names=feature_names,
        family_index=family_index,
        torch=torch,
    )
    split_metrics = {
        "train": _n1_split_metrics(
            model,
            rows,
            feature_names=feature_names,
            family_index=family_index,
            torch=torch,
            temperature=temperature,
        ),
        "validation": _n1_split_metrics(
            model,
            validation_samples,
            feature_names=feature_names,
            family_index=family_index,
            torch=torch,
            temperature=temperature,
        ),
        "test": _n1_split_metrics(
            model,
            test_samples,
            feature_names=feature_names,
            family_index=family_index,
            torch=torch,
            temperature=temperature,
        ),
    }
    calibration_split = "validation" if validation_samples else "train"
    calibration_ece = float(split_metrics[calibration_split]["calibration_ece"])
    evidence = {
        "classification": "counterfactual_paired",
        "dataset_classification": dataset_classification,
        "seed": seed,
        "epochs": len(losses),
        "valid_pairs": report.valid_pairs,
        "unique_contexts": report.unique_contexts,
        "generators": report.generators,
        "families": list(report.families),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "calibration_ece": calibration_ece,
        "calibration_split": calibration_split,
        "calibration_method": "validation_scalar_temperature_log_grid_v1",
        "temperature": temperature,
        "validation_ece_before_calibration": uncalibrated_validation.get(
            "calibration_ece"
        ),
        "heldout_evaluated": bool(validation_samples and test_samples),
        "split_metrics": split_metrics,
        "promotion_eligible": False,
    }
    artifact = {
        "artifact_schema_version": "n1-compact-mlp-artifact-v1",
        "feature_names": list(feature_names),
        "family_catalog": list(FAMILY_CATALOG_V2),
        "catalog_version": "n1-family-catalog-v2",
        "w1": model.layer1.weight.detach().tolist(),
        "b1": model.layer1.bias.detach().tolist(),
        "w2": model.layer2.weight.detach().tolist(),
        "b2": model.layer2.bias.detach().tolist(),
        "utility_head": model.utility.weight.detach().tolist(),
        "probability_head": model.probability.weight.detach().tolist(),
        "temperature": temperature,
        "calibration_ece": calibration_ece,
        "activation_policy": {
            "min_expected_utility": 0.0,
            "min_probability_positive": 0.5,
            "max_uncertainty": 0.5,
            "max_calibration_ece": 0.10,
        },
        "training_evidence": evidence,
    }
    target = Path(artifact_root) / "n1"
    target.mkdir(parents=True, exist_ok=True)
    artifact_path = target / "router-lab-v1.json"
    _write_json(artifact_path, artifact)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    manifest = NeuralModelManifest(
        organ="N1", capability="family_routing_proposal", model_id="rnfe-n1-router-lab-v1",
        version="1.0.0-lab", backend="rnfe-compact-mlp-router-v1",
        artifact_path="n1/router-lab-v1.json", artifact_sha256=_digest(artifact_path),
        input_schema_version="n1-context-features-v1", output_schema_version="n1-routing-proposal-v2",
        supported_devices=("cpu",), parameter_count=parameter_count, peak_vram_gb=0.0,
        license_id="Unlicense", upstream_url="repo://rnfe/runtime/neural/organs/n1_router.py",
        upstream_commit="n1-router-contract-v2", training_provenance=evidence,
        metrics={"training_loss": losses[-1], "ece": calibration_ece},
    )
    _write_json(target / "manifest.json", manifest.to_dict())
    _write_json(target / "model_card.json", {
        "schema_version": "rnfe-model-card-v1", "model_id": manifest.model_id,
        "intended_use": "N1 shadow family routing proposal", "authority_effect": "none",
        "limitations": [
            "scheduler retains authority",
            "promotion requires held-out positive CI and ECE <= 0.10",
            "missing held-out splits fall back to train calibration and remain non-promotable",
        ],
        "training_evidence": evidence,
    })
    return manifest, evidence


def train_n4_causal_graph(
    cases: Sequence[Mapping[str, Any]],
    *,
    artifact_root: str | Path,
    seed: int = 37,
    epochs: int = 100,
    dataset_classification: str = "caller_supplied_typed_causal",
) -> tuple[NeuralModelManifest, dict[str, Any]]:
    """Ajusta las cuatro cabezas N4 sin aprender ni mutar la topología canónica."""

    valid = [dict(item) for item in cases if len(item.get("source_features", ())) == 4 and len(item.get("target_features", ())) == 4]
    if len(valid) < 10:
        raise ValueError("n4_training_requires_ten_typed_cases")
    import torch

    torch.manual_seed(seed)
    input_weight = torch.nn.Parameter(torch.eye(4))
    message_weight = torch.nn.Parameter(torch.eye(4) * 0.5)
    update_weight = torch.nn.Parameter(torch.eye(4) * 0.5)
    output_weight = torch.nn.Parameter(torch.eye(4) * 0.25)
    parameters = [input_weight, message_weight, update_weight, output_weight]
    optimizer = torch.optim.AdamW(parameters, lr=4e-3, weight_decay=1e-4)
    losses = []
    for _epoch in range(max(1, int(epochs))):
        optimizer.zero_grad(set_to_none=True)
        total = torch.tensor(0.0)
        for case in valid:
            source = torch.tanh(input_weight @ torch.tensor(case["source_features"], dtype=torch.float32))
            target = torch.tanh(input_weight @ torch.tensor(case["target_features"], dtype=torch.float32))
            factor = float(case["signed_strength"]) * float(case.get("edge_confidence", 1.0))
            target = torch.tanh(update_weight @ target + factor * (message_weight @ source))
            source = torch.tanh(update_weight @ source)
            heads = output_weight @ ((source + target) / 2.0)
            magnitude = torch.tanh(torch.tensor(abs(float(case["signed_strength"]))) + 0.25 * torch.abs(torch.tanh(heads[1])))
            confidence = torch.sigmoid(heads[2]) * float(case.get("edge_confidence", 1.0))
            uncertainty = torch.sigmoid(heads[3])
            direction = -1.0 if float(case["signed_strength"]) < 0 else 1.0
            next_mean = torch.tanh(heads[0] + direction * magnitude)
            total = total + (magnitude - float(case["target_magnitude"])) ** 2
            total = total + (confidence - float(case["target_confidence"])) ** 2
            total = total + (uncertainty - float(case["target_uncertainty"])) ** 2
            total = total + (next_mean - float(case["target_next_state"])) ** 2
        loss = total / len(valid)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    evidence = {
        "classification": "typed_causal_laboratory", "seed": seed,
        "dataset_classification": dataset_classification,
        "epochs": len(losses), "cases": len(valid), "initial_loss": losses[0],
        "final_loss": losses[-1], "topology_learned": False, "promotion_eligible": False,
    }
    artifact = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION, "model_kind": "trained",
        "trained": True, "frozen": True, "experimental": True, "training_evidence": evidence,
        "input_size": 4, "hidden_size": 4, "message_passing_steps": 1,
        "max_nodes": 64, "max_edges": 128,
        "input_weight": input_weight.detach().tolist(),
        "message_weight": message_weight.detach().tolist(),
        "update_weight": update_weight.detach().tolist(),
        "output_weight": output_weight.detach().tolist(),
        "supported_node_types": [item.value for item in NodeType],
        "supported_edge_types": [item.value for item in EdgeType],
    }
    target = Path(artifact_root) / "n4"
    target.mkdir(parents=True, exist_ok=True)
    artifact_path = target / "causal-lab-v1.json"
    _write_json(artifact_path, artifact)
    manifest = NeuralModelManifest(
        organ="N4", capability="typed_causal_proposal", model_id="rnfe-n4-causal-lab-v1",
        version="1.0.0-lab", backend="rnfe-trained-causal-graph-v1",
        artifact_path="n4/causal-lab-v1.json", artifact_sha256=_digest(artifact_path),
        input_schema_version="n4-causal-graph-v1", output_schema_version="n4-causal-proposal-v1",
        supported_devices=("cpu",), parameter_count=64, peak_vram_gb=0.0,
        license_id="Unlicense", upstream_url="repo://rnfe/runtime/neural/organs/n4_causal.py",
        upstream_commit="n4-typed-contract-v1", training_provenance=evidence,
        metrics={"training_loss": losses[-1]},
    )
    _write_json(target / "manifest.json", manifest.to_dict())
    _write_json(target / "model_card.json", {
        "schema_version": "rnfe-model-card-v1", "model_id": manifest.model_id,
        "intended_use": "N4 shadow signed-effect proposals", "authority_effect": "none",
        "limitations": ["typed laboratory cases", "canonical graph is not learned or mutated", "CAU/CTF/C-GWM retain authority"],
        "training_evidence": evidence,
    })
    return manifest, evidence
