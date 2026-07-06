"""Contrato de embeddings + factory gated por entorno."""

from __future__ import annotations

import math
import os
from typing import Any, List, Mapping, Optional, Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Provee un vector denso determinista para un texto."""

    dim: int

    def embed(self, text: str) -> Optional[List[float]]:
        """Devuelve el vector, o None si no pudo computarse (degradación)."""
        ...


def embedding_mode() -> str:
    """Modo de embeddings: 'off' (default), 'hashed' o 'llama'."""
    raw = os.environ.get("RNFE_MEMORY_EMBEDDINGS", "off").strip().lower()
    if raw in {"", "0", "off", "false", "no", "none"}:
        return "off"
    if raw in {"hashed", "hash", "ngram"}:
        return "hashed"
    if raw in {"llama", "llama_cpp", "gpu"}:
        return "llama"
    return "off"


def text_from_mapping(mapping: Mapping[str, Any]) -> str:
    """Serializa un dict a un texto canónico y estable para embeber.

    Ordena las claves para que el mismo contenido produzca el mismo texto
    (determinismo), y aplana listas.
    """
    if not isinstance(mapping, Mapping):
        return str(mapping)
    parts: List[str] = []
    for key in sorted(str(k) for k in mapping.keys()):
        value = mapping[key]
        if isinstance(value, (list, tuple)):
            rendered = " ".join(str(item) for item in value)
        elif isinstance(value, Mapping):
            rendered = text_from_mapping(value)
        else:
            rendered = str(value)
        parts.append(f"{key}={rendered}")
    return " ".join(parts)


def cosine_similarity(a: List[float] | None, b: List[float] | None) -> float:
    """Coseno en [0,1] (clamp de negativos a 0). Vacío/None -> 0."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    cos = dot / (math.sqrt(na) * math.sqrt(nb))
    return max(0.0, min(1.0, cos))


_CACHED_MODE: str | None = None
_CACHED_EMBEDDER: Optional[EmbeddingProvider] = None


def get_embedder() -> Optional[EmbeddingProvider]:
    """Devuelve el embedder del modo activo (cacheado), o None si 'off'."""
    global _CACHED_MODE, _CACHED_EMBEDDER
    mode = embedding_mode()
    if mode == _CACHED_MODE:
        return _CACHED_EMBEDDER
    _CACHED_MODE = mode
    if mode == "hashed":
        from .hashed_ngram import HashedNgramEmbedder

        _CACHED_EMBEDDER = HashedNgramEmbedder()
    elif mode == "llama":
        from .llama_cpp_embedder import LlamaCppEmbedder

        _CACHED_EMBEDDER = LlamaCppEmbedder()
    else:
        _CACHED_EMBEDDER = None
    return _CACHED_EMBEDDER


def reset_embedder() -> None:
    """Limpia el cache (para tests que cambian el modo)."""
    global _CACHED_MODE, _CACHED_EMBEDDER
    _CACHED_MODE = None
    _CACHED_EMBEDDER = None
