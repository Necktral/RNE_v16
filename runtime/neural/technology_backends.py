"""Backends entrenables H-Net/Mamba2 cargados sólo detrás de N0.

Este módulo no importa torch ni engines al importarse. Las dependencias pesadas se
cargan dentro de ``load`` después de que N0 verificó modo, recursos y SHA-256.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping

from .contracts import BackendOutput, NeuralInferenceRequest, NeuralModelManifest
from .organs.n5_ingest import (
    BoundaryOffsets,
    BoundarySemantics,
    OffsetUnit,
    TextOffsetMap,
    _chunks_from_boundaries,
)


HNET_BACKEND_ID = "rnfe-hnet-boundary-v1"
MAMBA2_BACKEND_ID = "rnfe-mamba2-temporal-v1"
HNET_ARTIFACT_SCHEMA = "rnfe-hnet-boundary-artifact-v1"
MAMBA2_ARTIFACT_SCHEMA = "rnfe-mamba2-temporal-artifact-v1"
HNET_UPSTREAM_COMMIT = "3ae01de79e560234776d06ceb1153ab76a5aad32"
MAMBA_UPSTREAM_COMMIT = "e0761ece1db07e0949dd88b4f4cd440420a19fd9"
N3_FEATURE_NAMES = (
    "value",
    "trend",
    "alarm",
    "reference_uncertainty",
    "cpu_pressure",
    "memory_pressure",
    "thermal_pressure",
    "memory_count",
)


def _load_torch_artifact(
    path: str, device: str, *, require_cuda: bool
) -> tuple[Any, Mapping[str, Any]]:
    import torch

    if require_cuda and (device != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError("trained_technology_backend_requires_cuda")
    if not require_cuda and device != "cpu":
        raise RuntimeError("mamba2_ssd_minimal_backend_requires_cpu")
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, Mapping):
        raise ValueError("technology_artifact_must_be_mapping")
    return torch, raw


def _require_training_evidence(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    if raw.get("classification") != "trained" or raw.get("trained") is not True:
        raise ValueError("technology_artifact_must_be_trained")
    evidence = raw.get("training_evidence")
    if not isinstance(evidence, Mapping) or not evidence:
        raise ValueError("technology_training_evidence_required")
    return evidence


def _compact_hnet(torch: Any, config: Mapping[str, Any]) -> Any:
    from hnet.models.config_hnet import AttnConfig, HNetConfig, SSMConfig
    from hnet.modules.dc import RoutingModule
    from hnet.modules.isotropic import Isotropic

    class CompactHNetBoundaryModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            d_model = int(config["d_model"])
            layers = int(config["layers"])
            hnet_config = HNetConfig(
                arch_layout=[f"m{layers}"],
                d_model=[d_model],
                d_intermediate=[0],
                vocab_size=256,
                ssm_cfg=SSMConfig(
                    d_conv=int(config.get("d_conv", 4)),
                    expand=int(config.get("expand", 2)),
                    d_state=int(config.get("d_state", 16)),
                    chunk_size=int(config.get("chunk_size", 64)),
                    use_mem_eff_path=False,
                ),
                # get_stage_cfg indexa todas las listas aunque la etapa sólo use Mamba.
                attn_cfg=AttnConfig(
                    num_heads=[1], rotary_emb_dim=[0], window_size=[-1]
                ),
            )
            self.embedding = torch.nn.Embedding(256, d_model)
            self.encoder = Isotropic(hnet_config, pos_idx=0, stage_idx=0)
            self.router = RoutingModule(d_model)

        def forward(self, byte_ids: Any, mask: Any) -> Any:
            hidden = self.encoder(self.embedding(byte_ids), mask=mask)
            return self.router(hidden, mask=mask).boundary_prob[..., 1]

    return CompactHNetBoundaryModel()


def _compact_mamba2(torch: Any, config: Mapping[str, Any]) -> Any:
    """Mamba2 SSD minimal puro, entrenable y servible en CPU."""

    from einops import rearrange, repeat

    def segsum(x: Any) -> Any:
        length = x.size(-1)
        expanded = repeat(x, "... d -> ... d e", e=length)
        strict_lower = torch.tril(
            torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=-1
        )
        expanded = expanded.masked_fill(~strict_lower, 0)
        result = torch.cumsum(expanded, dim=-2)
        lower = torch.tril(
            torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=0
        )
        return result.masked_fill(~lower, -torch.inf)

    def ssd_minimal(x: Any, a: Any, b: Any, c: Any, block_len: int) -> Any:
        x, a, b, c = [
            rearrange(item, "b (c l) ... -> b c l ...", l=block_len)
            for item in (x, a, b, c)
        ]
        a = rearrange(a, "b c l h -> b h c l")
        a_cumsum = torch.cumsum(a, dim=-1)
        decay = torch.exp(segsum(a))
        y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", c, b, decay, x)
        decay_states = torch.exp(a_cumsum[:, :, :, -1:] - a_cumsum)
        states = torch.einsum("bclhn,bhcl,bclhp->bchpn", b, decay_states, x)
        states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)
        decay_chunk = torch.exp(
            segsum(torch.nn.functional.pad(a_cumsum[:, :, :, -1], (1, 0)))
        )
        states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)[:, :-1]
        y_off = torch.einsum(
            "bclhn,bchpn,bhcl->bclhp", c, states, torch.exp(a_cumsum)
        )
        return rearrange(y_diag + y_off, "b c l h p -> b (c l) (h p)")

    class CompactMamba2TemporalModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            input_size = int(config["input_size"])
            d_model = int(config["d_model"])
            self.nheads = int(config.get("nheads", 2))
            self.headdim = int(config.get("headdim", 16))
            self.d_state = int(config.get("d_state", 8))
            self.block_len = int(config.get("block_len", 4))
            if self.nheads * self.headdim != d_model:
                raise ValueError("mamba2_ssd_head_shape_mismatch")
            output_size = int(config.get("output_size", 5))
            self.input_projection = torch.nn.Linear(input_size, d_model)
            self.x_projection = torch.nn.Linear(d_model, d_model)
            self.a_projection = torch.nn.Linear(d_model, self.nheads)
            self.b_projection = torch.nn.Linear(d_model, self.nheads * self.d_state)
            self.c_projection = torch.nn.Linear(d_model, self.nheads * self.d_state)
            self.output_head = torch.nn.Linear(d_model, output_size)

        def forward(self, sequence: Any) -> Any:
            hidden = torch.tanh(self.input_projection(sequence))
            original_length = hidden.shape[1]
            padding = (-original_length) % self.block_len
            if padding:
                hidden = torch.nn.functional.pad(hidden, (0, 0, 0, padding))
            batch, length, _ = hidden.shape
            x = self.x_projection(hidden).view(batch, length, self.nheads, self.headdim)
            a = -torch.nn.functional.softplus(self.a_projection(hidden))
            b = self.b_projection(hidden).view(batch, length, self.nheads, self.d_state)
            c = self.c_projection(hidden).view(batch, length, self.nheads, self.d_state)
            temporal = ssd_minimal(x, a, b, c, self.block_len)
            return self.output_head(temporal[:, original_length - 1])

    return CompactMamba2TemporalModel()


class HNetBoundaryTorchBackend:
    """H-Net compacto entrenado; produce chunks candidatos, nunca escrituras."""

    def __init__(self) -> None:
        self.model: Any | None = None
        self.torch: Any | None = None
        self.device = "none"
        self.model_id = ""
        self.artifact_hash = ""
        self.parameter_count = 0

    def load(self, manifest: NeuralModelManifest, artifact_path: str, device: str) -> None:
        if manifest.organ != "N5" or manifest.backend != HNET_BACKEND_ID:
            raise ValueError("hnet_boundary_manifest_mismatch")
        if manifest.license_id != "MIT" or manifest.upstream_commit != HNET_UPSTREAM_COMMIT:
            raise ValueError("hnet_license_or_upstream_mismatch")
        torch, raw = _load_torch_artifact(artifact_path, device, require_cuda=True)
        if raw.get("artifact_schema_version") != HNET_ARTIFACT_SCHEMA:
            raise ValueError("hnet_artifact_schema_mismatch")
        _require_training_evidence(raw)
        config = raw.get("model_config")
        if not isinstance(config, Mapping):
            raise ValueError("hnet_model_config_required")
        model = _compact_hnet(torch, config)
        model.load_state_dict(raw["state_dict"], strict=True)
        model.to(device="cuda", dtype=torch.float32).eval()
        self.model = model
        self.torch = torch
        self.device = device
        self.model_id = manifest.model_id
        self.artifact_hash = manifest.artifact_sha256
        self.parameter_count = sum(item.numel() for item in model.parameters())

    def infer(self, request: NeuralInferenceRequest) -> BackendOutput:
        if self.model is None or self.torch is None:
            raise RuntimeError("backend_not_loaded")
        text = str(request.payload.get("text") or "")
        offsets = TextOffsetMap.from_content(text)
        if offsets.byte_length == 0:
            probabilities: list[float] = []
        else:
            ids = self.torch.tensor(
                [list(offsets.utf8_bytes)], device=self.device, dtype=self.torch.long
            )
            mask = self.torch.ones_like(ids, dtype=self.torch.bool)
            with self.torch.inference_mode():
                probabilities = self.model(ids, mask)[0].float().cpu().tolist()
        threshold = min(max(float(request.payload.get("boundary_threshold", 0.5)), 0.0), 1.0)
        native = BoundaryOffsets(
            values=tuple(index for index, value in enumerate(probabilities) if value >= threshold),
            unit=OffsetUnit.BYTE,
            semantics=BoundarySemantics.SPLIT_OFFSET,
        )
        # La conversión rechaza fronteras dentro de un codepoint UTF-8.
        codepoint = BoundaryOffsets(
            values=native.to_codepoint_splits(offsets),
            unit=OffsetUnit.CODEPOINT,
            semantics=BoundarySemantics.SPLIT_OFFSET,
        )
        chunks = [item.to_dict() for item in _chunks_from_boundaries(text, codepoint)]
        certainty = (
            sum(abs(value - 0.5) * 2.0 for value in probabilities) / len(probabilities)
            if probabilities
            else 0.0
        )
        candidate = {
            "status": "ok",
            "backend": HNET_BACKEND_ID,
            "classification": "trained",
            "trained_model": True,
            "hnet_active": True,
            "model_id": self.model_id,
            "artifact_sha256": self.artifact_hash,
            "boundary_offsets": native.to_dict(),
            "boundary_threshold": threshold,
            "probabilities": probabilities,
            "text_sha256": offsets.text_sha256,
            "byte_length": offsets.byte_length,
            "codepoint_length": offsets.codepoint_length,
            "chunks": chunks,
            "memory_candidates": [
                {
                    "chunk": chunk,
                    "provenance": f"{HNET_BACKEND_ID}:{self.artifact_hash}",
                    "promotion": "requires_existing_mfm_gate",
                }
                for chunk in chunks
            ],
            "authority_effect": "none",
        }
        return BackendOutput(
            candidate_output=candidate,
            confidence=certainty,
            uncertainty=1.0 - certainty,
            cost={
                "bytes": offsets.byte_length,
                "chunks": len(chunks),
                "parameter_count": self.parameter_count,
                "backend_device": self.device,
            },
        )

    def unload(self) -> None:
        self.model = None
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        self.torch = None
        self.device = "none"


class Mamba2TemporalTorchBackend:
    """Mamba2 compacto entrenado sobre secuencias de vitals/episodios."""

    OUTPUT_NAMES = ("retrieval_priority", "importance", "risk", "continuity", "confidence")

    def __init__(self) -> None:
        self.model: Any | None = None
        self.torch: Any | None = None
        self.device = "none"
        self.model_id = ""
        self.artifact_hash = ""
        self.input_size = 0
        self.history_size = 16
        self.parameter_count = 0
        self.histories: dict[tuple[str, str, str], deque[tuple[float, ...]]] = {}

    def load(self, manifest: NeuralModelManifest, artifact_path: str, device: str) -> None:
        if manifest.organ != "N3" or manifest.backend != MAMBA2_BACKEND_ID:
            raise ValueError("mamba2_temporal_manifest_mismatch")
        if manifest.license_id != "Apache-2.0" or manifest.upstream_commit != MAMBA_UPSTREAM_COMMIT:
            raise ValueError("mamba2_license_or_upstream_mismatch")
        torch, raw = _load_torch_artifact(artifact_path, device, require_cuda=False)
        if raw.get("artifact_schema_version") != MAMBA2_ARTIFACT_SCHEMA:
            raise ValueError("mamba2_artifact_schema_mismatch")
        _require_training_evidence(raw)
        config = raw.get("model_config")
        if not isinstance(config, Mapping):
            raise ValueError("mamba2_model_config_required")
        if int(config.get("output_size", 0)) != len(self.OUTPUT_NAMES):
            raise ValueError("mamba2_output_contract_mismatch")
        model = _compact_mamba2(torch, config)
        model.load_state_dict(raw["state_dict"], strict=True)
        model.to(device="cpu", dtype=torch.float32).eval()
        self.model = model
        self.torch = torch
        self.device = device
        self.model_id = manifest.model_id
        self.artifact_hash = manifest.artifact_sha256
        self.input_size = int(config["input_size"])
        self.history_size = min(max(int(config.get("history_size", 16)), 2), 128)
        self.parameter_count = sum(item.numel() for item in model.parameters())

    def infer(self, request: NeuralInferenceRequest) -> BackendOutput:
        if self.model is None or self.torch is None:
            raise RuntimeError("backend_not_loaded")
        key = tuple(
            str(request.payload.get(name) or "")
            for name in ("organism_id", "scenario_id", "lineage_id")
        )
        if not all(key):
            raise ValueError("mamba2_temporal_identity_required")
        values = tuple(float(value) for value in request.payload.get("input_vector", ()))
        if len(values) != self.input_size:
            raise ValueError("mamba2_temporal_input_size_mismatch")
        history = self.histories.setdefault(key, deque(maxlen=self.history_size))
        history.append(values)
        sequence = self.torch.tensor(
            [[list(item) for item in history]], device=self.device, dtype=self.torch.float32
        )
        with self.torch.inference_mode():
            logits = self.model(sequence)[0]
            probabilities = self.torch.sigmoid(logits).float().cpu().tolist()
        outputs = dict(zip(self.OUTPUT_NAMES, probabilities))
        candidate = {
            "status": "ok",
            "backend": MAMBA2_BACKEND_ID,
            "classification": "trained",
            "trained_model": True,
            "mamba2_active": True,
            "model_id": self.model_id,
            "artifact_sha256": self.artifact_hash,
            "state_key": list(key),
            "history_length": len(history),
            **outputs,
            "uncertainty": 1.0 - outputs["confidence"],
            "authority_effect": "none",
            "memory_authority": "MFM",
        }
        return BackendOutput(
            candidate_output=candidate,
            confidence=outputs["confidence"],
            uncertainty=1.0 - outputs["confidence"],
            cost={
                "history_length": len(history),
                "parameter_count": self.parameter_count,
                "backend_device": self.device,
            },
        )

    def unload(self) -> None:
        self.model = None
        self.histories.clear()
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        self.torch = None
        self.device = "none"
