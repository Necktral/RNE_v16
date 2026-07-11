"""N5: frontera real de ingestion con fallback determinista."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ..contracts import AdmissionDecision, BackendOutput, NeuralInferenceRequest, NeuralModelManifest


@dataclass(frozen=True, slots=True)
class TextChunk:
    index: int
    text: str
    start: int
    end: int
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "text": self.text, "start": self.start, "end": self.end, "source": self.source}


class DeterministicChunker:
    def __init__(self, *, max_bytes: int = 1024):
        if max_bytes < 32:
            raise ValueError("max_bytes_must_be_at_least_32")
        self.max_bytes = max_bytes

    def chunk(self, content: str | bytes) -> list[TextChunk]:
        text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else str(content)
        text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
        boundaries = {0, len(text)}
        for match in re.finditer(r"\n\s*\n|(?<=[.!?])\s+|\n(?=```)|(?<=```)\n", text):
            boundaries.add(match.end())
        points = sorted(boundaries)
        raw_segments = [text[points[i] : points[i + 1]] for i in range(len(points) - 1)]
        pieces: list[str] = []
        for segment in raw_segments:
            pieces.extend(_split_utf8(segment, self.max_bytes))
        chunks = []
        cursor = 0
        for piece in pieces:
            if not piece:
                continue
            start = text.find(piece, cursor)
            if start < 0:
                start = cursor
            end = start + len(piece)
            chunks.append(TextChunk(len(chunks), piece, start, end, "deterministic"))
            cursor = end
        return chunks


def _split_utf8(text: str, limit: int) -> list[str]:
    if len(text.encode("utf-8")) <= limit:
        return [text]
    pieces = []
    current = ""
    for character in text:
        candidate = current + character
        if current and len(candidate.encode("utf-8")) > limit:
            pieces.append(current)
            current = character
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


class HNetBoundaryBackend:
    """Puerto inyectable al H-Net certificado; evita asumir una API vendor."""

    def __init__(self, loader: Callable[[str, str], Any], infer_boundaries: Callable[[Any, str], Sequence[float]]):
        self.loader = loader
        self.infer_boundaries = infer_boundaries
        self.model: Any | None = None

    def load(self, manifest: NeuralModelManifest, artifact_path: str, device: str) -> None:
        if manifest.organ != "N5" or manifest.license_id.upper() != "MIT":
            raise ValueError("hnet_requires_certified_n5_mit_manifest")
        if manifest.upstream_commit.lower() in {"", "unknown", "unresolved"}:
            raise ValueError("hnet_upstream_commit_unresolved")
        self.model = self.loader(artifact_path, device)

    def infer(self, request: NeuralInferenceRequest) -> BackendOutput:
        if self.model is None:
            raise RuntimeError("backend_not_loaded")
        text = str(request.payload.get("text", ""))
        probabilities = [min(max(float(value), 0.0), 1.0) for value in self.infer_boundaries(self.model, text)]
        if len(probabilities) != len(text):
            raise ValueError("hnet_boundary_length_mismatch")
        threshold = float(request.payload.get("boundary_threshold", 0.5))
        boundaries = [index for index, value in enumerate(probabilities) if value >= threshold]
        return BackendOutput(
            candidate_output={"boundaries": boundaries, "probabilities": probabilities, "source": "hnet"},
            confidence=max(probabilities, default=0.0),
            uncertainty=1.0 - max(probabilities, default=0.0),
            cost={"characters": len(text)},
        )

    def unload(self) -> None:
        self.model = None


class HNetBoundaryAdmission:
    def __call__(self, candidate: Any, request: NeuralInferenceRequest) -> AdmissionDecision:
        if not isinstance(candidate, Mapping) or candidate.get("source") != "hnet":
            return AdmissionDecision(False, reason="n5_hnet_schema_invalid")
        text = str(request.payload.get("text", ""))
        boundaries = sorted({int(value) for value in candidate.get("boundaries", ())})
        if any(value < 0 or value >= len(text) for value in boundaries):
            return AdmissionDecision(False, reason="n5_boundary_out_of_range")
        return AdmissionDecision(True, output={"boundaries": boundaries, "source": "hnet"}, reason="n5_boundaries_valid")


class UnstructuredIngestionService:
    """Caller vivo: segmenta y entrega signos/candidatos por puertos inyectados."""

    def __init__(
        self,
        *,
        sign_sink: Callable[[Mapping[str, Any]], Any],
        memory_candidate_sink: Callable[[Mapping[str, Any]], Any],
        fallback_chunker: DeterministicChunker | None = None,
    ):
        self.sign_sink = sign_sink
        self.memory_candidate_sink = memory_candidate_sink
        self.fallback_chunker = fallback_chunker or DeterministicChunker()

    def ingest(
        self,
        content: str | bytes,
        *,
        run_id: str,
        source_id: str,
        neural_boundaries: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        chunks = (
            _chunks_from_boundaries(content, neural_boundaries)
            if neural_boundaries is not None
            else self.fallback_chunker.chunk(content)
        )
        signs = []
        memories = []
        for chunk in chunks:
            sign = {
                "run_id": run_id,
                "source_id": source_id,
                "chunk": chunk.to_dict(),
                "status": "candidate",
            }
            signs.append(self.sign_sink(sign))
            memories.append(
                self.memory_candidate_sink(
                    {**sign, "promotion": "requires_existing_mfm_gate"}
                )
            )
        return {"chunks": [item.to_dict() for item in chunks], "signs": signs, "memory_candidates": memories}


def _chunks_from_boundaries(content: str | bytes, boundaries: Sequence[int]) -> list[TextChunk]:
    text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else str(content)
    points = [0, *sorted({int(value) + 1 for value in boundaries if 0 <= int(value) < len(text)}), len(text)]
    points = sorted(set(points))
    return [
        TextChunk(index, text[start:end], start, end, "hnet")
        for index, (start, end) in enumerate(zip(points, points[1:]))
        if end > start
    ]
