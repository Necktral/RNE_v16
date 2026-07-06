"""Embedder por feature-hashing de n-gramas — stdlib puro, determinista, CPU.

Suficiente para similaridad estructural blanda sin dependencias ni GPU. Usa
``hashlib`` (no ``hash()``, que no es estable entre procesos) para mapear tokens
y n-gramas de caracteres a índices de un vector denso, con signo por hash — el
truco clásico de feature hashing. L2-normaliza el resultado.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import List, Optional

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _stable_hash(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


class HashedNgramEmbedder:
    """Vector denso por hashing de tokens + tri-gramas de caracteres."""

    def __init__(self, *, dim: int = 256, char_ngram: int = 3):
        self.dim = int(dim)
        self.char_ngram = int(char_ngram)

    def embed(self, text: str) -> Optional[List[float]]:
        if not text:
            return [0.0] * self.dim
        lowered = text.lower()
        vec = [0.0] * self.dim
        features: List[str] = _TOKEN_RE.findall(lowered)
        # Tri-gramas de caracteres sobre el texto compacto (capta subpalabras).
        compact = "".join(features)
        n = self.char_ngram
        for i in range(max(0, len(compact) - n + 1)):
            features.append("#" + compact[i : i + n])
        for feature in features:
            h = _stable_hash(feature)
            index = h % self.dim
            sign = 1.0 if (h >> 1) % 2 == 0 else -1.0
            vec[index] += sign
        norm = math.sqrt(sum(component * component for component in vec))
        if norm <= 0.0:
            return vec
        return [component / norm for component in vec]
