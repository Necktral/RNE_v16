"""Embeddings locales para recuperación semántica de memoria (Ola 3).

Gated por ``RNFE_MEMORY_EMBEDDINGS=off|hashed|llama``:
- ``off`` (default): sin embeddings, la recuperación es Jaccard puro (byte-idéntico).
- ``hashed``: ``HashedNgramEmbedder`` stdlib-puro (determinista, CPU, <1 ms).
- ``llama``: ``LlamaCppEmbedder`` — embeddings en la GPU vía el mismo binario
  llama.cpp del razonador externo (``--embedding``). Requiere provisión (Bloque E).
"""

from __future__ import annotations

from .provider import (
    EmbeddingProvider,
    cosine_similarity,
    embedding_mode,
    get_embedder,
    reset_embedder,
    text_from_mapping,
)

__all__ = [
    "EmbeddingProvider",
    "cosine_similarity",
    "embedding_mode",
    "get_embedder",
    "reset_embedder",
    "text_from_mapping",
]
