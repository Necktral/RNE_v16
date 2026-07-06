"""Embedder por llama.cpp (GPU) — subprocess al mismo toolchain del razonador.

Reutiliza el patrón de ``runtime/reasoning/external_models`` (subprocess a un
binario llama.cpp con un GGUF). Aquí el binario corre en modo ``--embedding`` y
descarga capas a la GPU con ``-ngl``. Degrada a ``None`` (sin crashear) cuando el
binario o el GGUF de embeddings no están provistos — el llamador cae a Jaccard.

Provisión: ``scripts/provision_llama_gpu.py`` (Bloque E) deja el binario y el
GGUF de embeddings y escribe las variables de entorno.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import List, Optional


class LlamaCppEmbedder:
    """Embeddings vía subprocess a llama.cpp con ``--embedding`` en la GPU."""

    def __init__(self, *, dim: int = 0, cache_size: int = 512):
        # dim=0 => desconocido hasta la primera medición (llama define el tamaño).
        self.dim = int(dim)
        self._cache: dict[str, List[float]] = {}
        self._cache_size = int(cache_size)
        self._available: Optional[bool] = None

    # -- Resolución de artefactos -----------------------------------------
    def _model_path(self) -> str:
        return os.environ.get("RNFE_EMBEDDINGS_GGUF", "").strip()

    def _cli_path(self) -> str:
        explicit = os.environ.get("RNFE_LLAMA_EMBEDDING_CLI", "").strip()
        if explicit:
            return explicit
        # Fallback: junto al binario del razonador, el binario llama-embedding.
        reasoner_cli = os.environ.get("RNFE_LLAMA_CLI_CUDA", "").strip() or os.environ.get(
            "RNFE_LLAMA_CLI_CPU", ""
        ).strip()
        if reasoner_cli:
            candidate = Path(reasoner_cli).with_name("llama-embedding")
            return str(candidate)
        return ""

    def available(self) -> bool:
        if self._available is not None:
            return self._available
        model = self._model_path()
        cli = self._cli_path()
        self._available = bool(model and cli and Path(model).exists() and Path(cli).exists())
        return self._available

    # -- Embed -------------------------------------------------------------
    def embed(self, text: str) -> Optional[List[float]]:
        if not text:
            return None
        if not self.available():
            return None
        if text in self._cache:
            return self._cache[text]
        vec = self._run(text)
        if vec is None:
            return None
        if self.dim == 0:
            self.dim = len(vec)
        if len(self._cache) < self._cache_size:
            self._cache[text] = vec
        return vec

    def _run(self, text: str) -> Optional[List[float]]:
        cli = self._cli_path()
        model = self._model_path()
        ngl = os.environ.get("RNFE_EXTERNAL_REASONER_NGL", "99").strip() or "99"
        command = [
            cli,
            "-m",
            model,
            "-p",
            text,
            "--embedding",
            "-ngl",
            ngl,
            "--no-warmup",
        ]
        try:
            proc = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=float(os.environ.get("RNFE_EMBEDDINGS_TIMEOUT_S", "60") or "60"),
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if proc.returncode != 0:
            return None
        return self._parse_vector(proc.stdout or "")

    @staticmethod
    def _parse_vector(stdout: str) -> Optional[List[float]]:
        # llama-embedding imprime los floats separados por espacio (una o más líneas).
        floats: List[float] = []
        for token in stdout.replace("\n", " ").split():
            try:
                floats.append(float(token))
            except ValueError:
                continue
        return floats or None
