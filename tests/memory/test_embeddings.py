"""Tests del seam de embeddings en recuperación de memoria (Bloque D)."""

from __future__ import annotations

from runtime.memory.embeddings import (
    cosine_similarity,
    embedding_mode,
    get_embedder,
    reset_embedder,
    text_from_mapping,
)
from runtime.memory.embeddings.hashed_ngram import HashedNgramEmbedder
from runtime.memory.mfm_lite.retrieval import MemoryRetrieval


def _retrieval() -> MemoryRetrieval:
    return MemoryRetrieval(storage=object())


def test_embedding_mode_off_by_default(monkeypatch):
    monkeypatch.delenv("RNFE_MEMORY_EMBEDDINGS", raising=False)
    assert embedding_mode() == "off"
    monkeypatch.setenv("RNFE_MEMORY_EMBEDDINGS", "hashed")
    assert embedding_mode() == "hashed"
    monkeypatch.setenv("RNFE_MEMORY_EMBEDDINGS", "llama")
    assert embedding_mode() == "llama"


def test_get_embedder_none_when_off(monkeypatch):
    reset_embedder()
    monkeypatch.delenv("RNFE_MEMORY_EMBEDDINGS", raising=False)
    assert get_embedder() is None
    reset_embedder()


def test_hashed_embedder_deterministic_and_normalized():
    emb = HashedNgramEmbedder(dim=128)
    a = emb.embed("temperature high alarm activate_cooling")
    b = emb.embed("temperature high alarm activate_cooling")
    assert a == b  # determinista
    assert len(a) == 128
    norm = sum(x * x for x in a) ** 0.5
    assert abs(norm - 1.0) < 1e-6  # L2-normalizado


def test_cosine_similarity_bounds():
    emb = HashedNgramEmbedder(dim=128)
    same = cosine_similarity(emb.embed("alarm cooling"), emb.embed("alarm cooling"))
    assert abs(same - 1.0) < 1e-6
    diff = cosine_similarity(emb.embed("alarm cooling"), emb.embed("xyzzy foobar qux"))
    assert 0.0 <= diff < same


def test_score_byte_identical_when_off(monkeypatch):
    reset_embedder()
    monkeypatch.delenv("RNFE_MEMORY_EMBEDDINGS", raising=False)
    r = _retrieval()
    query = {"proposition": "temp_high", "alarm": True}
    structure = {"proposition": "temp_high", "relation_kind": "support"}
    # _score con embeddings off == Jaccard puro.
    assert r._score(query=query, structure=structure) == r._jaccard(query=query, structure=structure)
    reset_embedder()


def test_score_blends_semantic_when_hashed(monkeypatch):
    reset_embedder()
    monkeypatch.setenv("RNFE_MEMORY_EMBEDDINGS", "hashed")
    monkeypatch.setenv("RNFE_MEMORY_EMBEDDINGS_WEIGHT", "0.5")
    r = _retrieval()
    # Estructuras SIN solape léxico exacto pero semánticamente cercanas por n-gramas.
    query = {"proposition": "temperature_high"}
    structure = {"proposition": "temperature_higher"}
    jac = r._jaccard(query=query, structure=structure)
    blended = r._score(query=query, structure=structure)
    # El coseno de n-gramas rescata similaridad que Jaccard exacto pierde.
    assert blended > jac
    reset_embedder()


def test_text_from_mapping_is_order_stable():
    a = text_from_mapping({"b": 1, "a": 2})
    b = text_from_mapping({"a": 2, "b": 1})
    assert a == b
