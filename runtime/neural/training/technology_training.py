"""Entrenamiento offline reproducible para modelos compactos N3/N5.

Escribe únicamente en el artifact plane indicado; nunca se importa desde el
camino nominal ni descarga pesos.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from runtime.neural.contracts import NeuralModelManifest
from runtime.neural.technology_backends import (
    HNET_ARTIFACT_SCHEMA,
    HNET_BACKEND_ID,
    HNET_UPSTREAM_COMMIT,
    MAMBA2_ARTIFACT_SCHEMA,
    MAMBA2_BACKEND_ID,
    MAMBA_UPSTREAM_COMMIT,
    N3_FEATURE_NAMES,
    _compact_hnet,
    _compact_mamba2,
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _require_cuda() -> Any:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("technology_training_requires_cuda")
    return torch


def _assert_physical_budget(
    *,
    projected_vram_gb: float = 0.25,
    max_temperature_c: float = 82.0,
    max_resident_vram_gb: float = 6.0,
    min_free_vram_gb: float = 1.5,
) -> None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        rows = [
            tuple(float(item.strip()) for item in line.split(","))
            for line in result.stdout.splitlines()
            if line.strip()
        ]
    except (OSError, ValueError, subprocess.SubprocessError):
        return
    for temperature, used_mib, total_mib in rows:
        used_gb, total_gb = used_mib / 1024.0, total_mib / 1024.0
        if temperature >= max_temperature_c:
            raise RuntimeError(f"training_thermal_budget_exceeded:{temperature:.1f}C")
        if used_gb + projected_vram_gb > max_resident_vram_gb:
            raise RuntimeError(f"training_resident_vram_budget_exceeded:{used_gb:.3f}GiB")
        if total_gb - used_gb - projected_vram_gb < min_free_vram_gb:
            raise RuntimeError("training_free_vram_reserve_would_be_violated")


def _boundary_labels(text: str) -> list[float]:
    labels = [0.0] * len(text.encode("utf-8"))
    if labels:
        labels[0] = 1.0
    byte_cursor = 0
    previous_was_separator = False
    for character in text:
        if previous_was_separator and byte_cursor < len(labels):
            labels[byte_cursor] = 1.0
        previous_was_separator = character.isspace() or character in ".,;:!?()[]{}"
        byte_cursor += len(character.encode("utf-8"))
    return labels


def train_hnet_boundary_model(
    texts: Sequence[str],
    *,
    artifact_root: str | Path,
    seed: int = 17,
    epochs: int = 12,
) -> tuple[NeuralModelManifest, dict[str, Any]]:
    """Entrena H-Net compacto con fronteras start-of-chunk UTF-8."""

    corpus = [str(item) for item in texts if str(item)]
    if len(corpus) < 6:
        raise ValueError("hnet_training_requires_at_least_six_texts")
    torch = _require_cuda()
    _assert_physical_budget()
    torch.manual_seed(seed)
    config = {"d_model": 32, "layers": 1, "d_state": 8, "d_conv": 4, "expand": 2, "chunk_size": 32}
    model = _compact_hnet(torch, config).to("cuda", dtype=torch.float32)
    # El backward Triton de Mamba2 no compila de forma segura en sm_75. El
    # encoder se usa como extractor congelado y sólo se entrena el router H-Net.
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("router."))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=2e-3, weight_decay=1e-4)
    encoded = [list(text.encode("utf-8")) for text in corpus]
    maximum = max(len(item) for item in encoded)
    ids = torch.zeros((len(corpus), maximum), device="cuda", dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    labels = torch.zeros_like(ids, dtype=torch.float32)
    for index, (byte_ids, text) in enumerate(zip(encoded, corpus)):
        length = len(byte_ids)
        ids[index, :length] = torch.tensor(byte_ids, device="cuda", dtype=torch.long)
        mask[index, :length] = True
        labels[index, :length] = torch.tensor(
            _boundary_labels(text), device="cuda", dtype=torch.float32
        )
    losses: list[float] = []
    model.train()
    for _epoch in range(max(1, int(epochs))):
        _assert_physical_budget()
        optimizer.zero_grad(set_to_none=True)
        probabilities = model(ids, mask).float().clamp(1e-5, 1.0 - 1e-5)
        loss = torch.nn.functional.binary_cross_entropy(probabilities[mask], labels[mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    dataset_payload = {
        "schema_version": "rnfe-hnet-boundary-dataset-v1",
        "seed": seed,
        "records": len(corpus),
        "text_hashes": [hashlib.sha256(item.encode("utf-8")).hexdigest() for item in corpus],
        "label_semantics": "utf8_byte_split_offset_start_of_chunk",
    }
    dataset_hash = hashlib.sha256(_canonical_bytes(dataset_payload)).hexdigest()
    evidence = {
        "classification": "laboratory_supervised",
        "seed": seed,
        "epochs": len(losses),
        "records": len(corpus),
        "dataset_sha256": dataset_hash,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "promotion_eligible": False,
        "trained_components": ["RoutingModule.q_proj_layer", "RoutingModule.k_proj_layer"],
        "frozen_encoder": True,
    }
    root = Path(artifact_root)
    target_dir = root / "n5"
    target_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = target_dir / "hnet-boundary-lab-v1.pt"
    torch.save(
        {
            "artifact_schema_version": HNET_ARTIFACT_SCHEMA,
            "classification": "trained",
            "trained": True,
            "model_config": config,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "training_evidence": evidence,
        },
        artifact_path,
    )
    manifest = NeuralModelManifest(
        organ="N5", capability="deterministic_ingestion",
        model_id="rnfe-hnet-boundary-lab-v1", version="1.0.0-lab",
        backend=HNET_BACKEND_ID, artifact_path="n5/hnet-boundary-lab-v1.pt",
        artifact_sha256=_sha256_file(artifact_path),
        input_schema_version="n5-utf8-boundary-input-v1",
        output_schema_version="n5-hnet-chunks-v1",
        supported_devices=("cuda",), supported_dtypes=("float32",),
        parameter_count=sum(item.numel() for item in model.parameters()), peak_vram_gb=0.25,
        license_id="MIT", upstream_url="https://github.com/goombalab/hnet",
        upstream_commit=HNET_UPSTREAM_COMMIT, training_provenance=evidence,
        metrics={"training_loss": losses[-1]},
    )
    _write_json(target_dir / "manifest.json", manifest.to_dict())
    _write_json(target_dir / "dataset_manifest.json", dataset_payload)
    _write_json(target_dir / "model_card.json", {
        "schema_version": "rnfe-model-card-v1", "model_id": manifest.model_id,
        "intended_use": "N5 shadow boundary proposals", "authority_effect": "none",
        "limitations": ["laboratory corpus; not promotion eligible", "requires CUDA", "SMG/MFM retain authority"],
        "training_evidence": evidence,
    })
    del model
    torch.cuda.empty_cache()
    return manifest, evidence


def train_mamba2_temporal_model(
    sequences: Sequence[Mapping[str, Any]],
    *,
    artifact_root: str | Path,
    seed: int = 23,
    epochs: int = 20,
) -> tuple[NeuralModelManifest, dict[str, Any]]:
    """Entrena Mamba2 compacto con secuencias y targets explícitos."""

    records = []
    for item in sequences:
        features, targets = item.get("features"), item.get("targets")
        if not isinstance(features, Sequence) or not isinstance(targets, Sequence):
            continue
        rows = [list(map(float, row)) for row in features]
        target = list(map(float, targets))
        if len(rows) >= 2 and all(len(row) == len(N3_FEATURE_NAMES) for row in rows) and len(target) == 5:
            records.append((rows, target))
    if len(records) < 6:
        raise ValueError("mamba2_training_requires_at_least_six_sequences")
    import torch

    torch.manual_seed(seed)
    config = {
        "input_size": len(N3_FEATURE_NAMES), "d_model": 32, "d_state": 8,
        "nheads": 2, "headdim": 16, "block_len": 4,
        "history_size": 16, "output_size": 5,
    }
    model = _compact_mamba2(torch, config).to("cpu", dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    losses: list[float] = []
    model.train()
    for _epoch in range(max(1, int(epochs))):
        total = 0.0
        for features, targets in records:
            x = torch.tensor([features], device="cpu", dtype=torch.float32)
            y = torch.tensor([targets], device="cpu", dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
        losses.append(total / len(records))
    dataset_payload = {
        "schema_version": "rnfe-mamba2-temporal-dataset-v1", "seed": seed,
        "records": len(records), "feature_names": list(N3_FEATURE_NAMES),
        "record_hashes": [hashlib.sha256(_canonical_bytes({"features": x, "targets": y})).hexdigest() for x, y in records],
    }
    dataset_hash = hashlib.sha256(_canonical_bytes(dataset_payload)).hexdigest()
    evidence = {
        "classification": "laboratory_supervised", "seed": seed, "epochs": len(losses),
        "records": len(records), "dataset_sha256": dataset_hash,
        "initial_loss": losses[0], "final_loss": losses[-1], "promotion_eligible": False,
    }
    root = Path(artifact_root)
    target_dir = root / "n3"
    target_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = target_dir / "mamba2-temporal-lab-v1.pt"
    torch.save({
        "artifact_schema_version": MAMBA2_ARTIFACT_SCHEMA, "classification": "trained", "trained": True,
        "model_config": config,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "training_evidence": evidence,
    }, artifact_path)
    manifest = NeuralModelManifest(
        organ="N3", capability="temporal_reference_state",
        model_id="rnfe-mamba2-temporal-lab-v1", version="1.0.0-lab",
        backend=MAMBA2_BACKEND_ID, artifact_path="n3/mamba2-temporal-lab-v1.pt",
        artifact_sha256=_sha256_file(artifact_path), input_schema_version="n3-vitals-sequence-v1",
        output_schema_version="n3-temporal-proposal-v1", supported_devices=("cpu",),
        supported_dtypes=("float32",), parameter_count=sum(item.numel() for item in model.parameters()),
        peak_vram_gb=0.0, license_id="Apache-2.0",
        upstream_url="https://github.com/state-spaces/mamba", upstream_commit=MAMBA_UPSTREAM_COMMIT,
        training_provenance=evidence, metrics={"training_loss": losses[-1]},
    )
    _write_json(target_dir / "manifest.json", manifest.to_dict())
    _write_json(target_dir / "dataset_manifest.json", dataset_payload)
    _write_json(target_dir / "model_card.json", {
        "schema_version": "rnfe-model-card-v1", "model_id": manifest.model_id,
        "intended_use": "N3 shadow temporal proposals", "authority_effect": "none",
        "limitations": ["laboratory sequences; not promotion eligible", "SSD minimal CPU", "MFM retains authority"],
        "training_evidence": evidence,
    })
    del model
    return manifest, evidence
