"""Configuracion para razonadores externos via llama.cpp."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


DEFAULT_RESPONSE_SCHEMA_PATH = (
    Path(__file__).resolve().parent / "schemas" / "ext_open_thinker_response.schema.json"
)


def _as_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _as_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _as_optional_int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None and value.strip() else None
    except (TypeError, ValueError):
        return None


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ExternalReasonerConfig:
    """Parametros runtime para invocar ``llama-cli`` sin acoplar RNFE al modelo."""

    model_path: str
    cuda_cli_path: str | None = None
    cpu_cli_path: str | None = None
    backend: str = "cuda"
    ngl: int = 99
    max_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.9
    timeout_s: float = 120.0
    cuda_root: str | None = None
    ld_library_path: str | None = None
    structured_output_mode: str = "json_schema"
    json_schema_path: str | None = str(DEFAULT_RESPONSE_SCHEMA_PATH)
    grammar_path: str | None = None
    reasoning_budget: int = 0
    prompt_style: str = "standard"
    ctx_size: int | None = None
    batch_size: int | None = None
    ubatch_size: int | None = None
    threads: int | None = None
    threads_batch: int | None = None
    mlock: bool = False

    @classmethod
    def from_env(
        cls,
        *,
        backend: str | None = None,
        max_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> "ExternalReasonerConfig":
        selected_backend = (
            backend
            or os.environ.get("RNFE_EXTERNAL_REASONER_BACKEND")
            or os.environ.get("RNFE_REASONING_BACKEND")
            or "cuda"
        )
        selected_backend = selected_backend.strip().lower()
        return cls(
            model_path=os.environ.get("RNFE_REASONING_GGUF", ""),
            cuda_cli_path=os.environ.get("RNFE_LLAMA_CLI_CUDA"),
            cpu_cli_path=os.environ.get("RNFE_LLAMA_CLI_CPU"),
            backend=selected_backend,
            ngl=_as_int(os.environ.get("RNFE_EXTERNAL_REASONER_NGL"), 99),
            max_tokens=(
                int(max_tokens)
                if max_tokens is not None
                else _as_int(os.environ.get("RNFE_EXTERNAL_REASONER_MAX_TOKENS"), 256)
            ),
            temperature=_as_float(os.environ.get("RNFE_EXTERNAL_REASONER_TEMPERATURE"), 0.1),
            top_p=_as_float(os.environ.get("RNFE_EXTERNAL_REASONER_TOP_P"), 0.9),
            timeout_s=(
                float(timeout_s)
                if timeout_s is not None
                else _as_float(os.environ.get("RNFE_EXTERNAL_REASONER_TIMEOUT_S"), 120.0)
            ),
            cuda_root=os.environ.get("CUDA_ROOT"),
            ld_library_path=os.environ.get("LD_LIBRARY_PATH"),
            structured_output_mode=(
                os.environ.get("RNFE_EXTERNAL_REASONER_STRUCTURED_OUTPUT_MODE")
                or "json_schema"
            ).strip().lower(),
            json_schema_path=(
                os.environ.get("RNFE_EXTERNAL_REASONER_JSON_SCHEMA")
                or str(DEFAULT_RESPONSE_SCHEMA_PATH)
            ),
            grammar_path=os.environ.get("RNFE_EXTERNAL_REASONER_GRAMMAR"),
            reasoning_budget=_as_int(os.environ.get("RNFE_EXTERNAL_REASONER_REASONING_BUDGET"), 0),
            prompt_style=(os.environ.get("RNFE_EXTERNAL_REASONER_PROMPT_STYLE") or "standard").strip().lower(),
            ctx_size=_as_optional_int(os.environ.get("RNFE_EXTERNAL_REASONER_CTX_SIZE")),
            batch_size=_as_optional_int(os.environ.get("RNFE_EXTERNAL_REASONER_BATCH_SIZE")),
            ubatch_size=_as_optional_int(os.environ.get("RNFE_EXTERNAL_REASONER_UBATCH_SIZE")),
            threads=_as_optional_int(os.environ.get("RNFE_EXTERNAL_REASONER_THREADS")),
            threads_batch=_as_optional_int(os.environ.get("RNFE_EXTERNAL_REASONER_THREADS_BATCH")),
            mlock=_as_bool(os.environ.get("RNFE_EXTERNAL_REASONER_MLOCK"), False),
        )

    def cli_path_for_backend(self, backend: str | None = None) -> str:
        selected = (backend or self.backend or "cuda").strip().lower()
        if selected == "cuda":
            return self.cuda_cli_path or ""
        if selected == "cpu":
            return self.cpu_cli_path or ""
        return ""

    def validation_error(self, backend: str | None = None) -> str | None:
        selected = (backend or self.backend or "cuda").strip().lower()
        if selected not in {"cuda", "cpu"}:
            return f"unsupported_backend:{selected}"
        if not self.model_path:
            return "missing_model_path:RNFE_REASONING_GGUF"
        if not self.cli_path_for_backend(selected):
            env_name = "RNFE_LLAMA_CLI_CUDA" if selected == "cuda" else "RNFE_LLAMA_CLI_CPU"
            return f"missing_cli_path:{env_name}"
        if not Path(self.model_path).exists():
            return f"model_not_found:{self.model_path}"
        if not Path(self.cli_path_for_backend(selected)).exists():
            return f"cli_not_found:{self.cli_path_for_backend(selected)}"
        mode = (self.structured_output_mode or "off").strip().lower()
        if mode not in {"off", "json_schema", "grammar"}:
            return f"unsupported_structured_output_mode:{mode}"
        if mode == "json_schema":
            if not self.json_schema_path:
                return "missing_json_schema_path:RNFE_EXTERNAL_REASONER_JSON_SCHEMA"
            if not Path(self.json_schema_path).exists():
                return f"json_schema_not_found:{self.json_schema_path}"
        if mode == "grammar":
            if not self.grammar_path:
                return "missing_grammar_path:RNFE_EXTERNAL_REASONER_GRAMMAR"
            if not Path(self.grammar_path).exists():
                return f"grammar_not_found:{self.grammar_path}"
        return None

    def subprocess_env(self, backend: str | None = None) -> Dict[str, str]:
        env = dict(os.environ)
        selected = (backend or self.backend or "cuda").strip().lower()
        if selected != "cuda":
            return env

        parts: list[str] = []
        if self.cuda_root:
            parts.append(str(Path(self.cuda_root) / "lib"))
        parts.append("/usr/lib/wsl/lib")
        cli_path = self.cli_path_for_backend("cuda")
        if cli_path:
            parts.append(str(Path(cli_path).resolve().parent))
        if self.ld_library_path:
            parts.append(self.ld_library_path)
        env["LD_LIBRARY_PATH"] = ":".join(part for part in parts if part)
        return env
