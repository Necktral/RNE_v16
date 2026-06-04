"""Cliente seguro para ``llama-cli`` usado como razonador externo."""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from .config import ExternalReasonerConfig


_TIMING_RE = re.compile(
    r"\[\s*Prompt:\s*(?P<prompt>[0-9.]+)\s*t/s\s*\|\s*Generation:\s*(?P<generation>[0-9.]+)\s*t/s\s*\]"
)


@dataclass(frozen=True)
class LlamaCppRequest:
    prompt: str
    backend: str
    max_tokens: int
    temperature: float
    top_p: float
    timeout_s: float
    ngl: int


class LlamaCppClient:
    """Wrapper minimo, auditable y no obligatorio sobre llama.cpp."""

    def __init__(self, config: ExternalReasonerConfig | None = None):
        self.config = config or ExternalReasonerConfig.from_env()

    def build_command(self, request: LlamaCppRequest) -> List[str]:
        cli = self.config.cli_path_for_backend(request.backend)
        ngl = request.ngl if request.backend == "cuda" else 0
        command = [
            cli,
            "-m",
            self.config.model_path,
            "-p",
            request.prompt,
            "-n",
            str(request.max_tokens),
            "-ngl",
            str(ngl),
            "--temp",
            str(request.temperature),
            "--top-p",
            str(request.top_p),
            "--no-warmup",
            "--simple-io",
            "--single-turn",
            "--no-display-prompt",
            "--reasoning",
            "off",
            "--reasoning-budget",
            str(int(self.config.reasoning_budget)),
        ]
        mode = (self.config.structured_output_mode or "off").strip().lower()
        if mode == "json_schema" and self.config.json_schema_path:
            command.extend(["--json-schema-file", self.config.json_schema_path])
        elif mode == "grammar" and self.config.grammar_path:
            command.extend(["--grammar-file", self.config.grammar_path])
        return command

    def generate(
        self,
        prompt: str,
        *,
        backend: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        timeout_s: float | None = None,
        allow_cpu_fallback: bool = False,
    ) -> Dict[str, Any]:
        selected = (backend or self.config.backend or "cuda").strip().lower()
        primary = self._generate_once(
            LlamaCppRequest(
                prompt=prompt,
                backend=selected,
                max_tokens=int(max_tokens if max_tokens is not None else self.config.max_tokens),
                temperature=float(temperature if temperature is not None else self.config.temperature),
                top_p=float(top_p if top_p is not None else self.config.top_p),
                timeout_s=float(timeout_s if timeout_s is not None else self.config.timeout_s),
                ngl=int(self.config.ngl),
            )
        )
        if primary.get("ok") or selected == "cpu" or not allow_cpu_fallback:
            return primary

        fallback = self._generate_once(
            LlamaCppRequest(
                prompt=prompt,
                backend="cpu",
                max_tokens=int(max_tokens if max_tokens is not None else self.config.max_tokens),
                temperature=float(temperature if temperature is not None else self.config.temperature),
                top_p=float(top_p if top_p is not None else self.config.top_p),
                timeout_s=float(timeout_s if timeout_s is not None else self.config.timeout_s),
                ngl=0,
            )
        )
        fallback["fallback_attempted"] = True
        fallback["primary_error"] = {
            "error_type": primary.get("error_type"),
            "error_message": primary.get("error_message"),
            "backend": primary.get("backend"),
        }
        return fallback

    def _generate_once(self, request: LlamaCppRequest) -> Dict[str, Any]:
        validation_error = self.config.validation_error(request.backend)
        if validation_error:
            return {
                "ok": False,
                "backend": request.backend,
                "structured_output_mode": self.config.structured_output_mode,
                "grammar_used": False,
                "json_schema_used": False,
                "schema_validated": False,
                "error_type": "configuration_error",
                "error_message": validation_error,
                "stdout": "",
                "stderr": "",
                "latency_s": 0.0,
            }

        command = self.build_command(request)
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                shell=False,
                capture_output=True,
                text=True,
                timeout=request.timeout_s,
                env=self.config.subprocess_env(request.backend),
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "ok": False,
                "backend": request.backend,
                "command": command,
                "structured_output_mode": self.config.structured_output_mode,
                "grammar_used": False,
                "json_schema_used": False,
                "schema_validated": False,
                "error_type": "timeout",
                "error_message": f"llama-cli timeout after {request.timeout_s:.2f}s",
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
                "latency_s": time.perf_counter() - started,
            }
        except OSError as exc:
            return {
                "ok": False,
                "backend": request.backend,
                "command": command,
                "structured_output_mode": self.config.structured_output_mode,
                "grammar_used": False,
                "json_schema_used": False,
                "schema_validated": False,
                "error_type": "os_error",
                "error_message": str(exc),
                "stdout": "",
                "stderr": "",
                "latency_s": time.perf_counter() - started,
            }

        latency = time.perf_counter() - started
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        timing = self._parse_timing(stdout + "\n" + stderr)
        ok = completed.returncode == 0 and bool(stdout.strip())
        error_type = None
        error_message = None
        if not ok:
            combined = f"{stdout}\n{stderr}"
            if "libcudart.so" in combined or "cannot open shared object file" in combined:
                error_type = "missing_cuda_library"
            else:
                error_type = "process_error"
            error_message = combined.strip() or f"llama-cli exited {completed.returncode}"

        return {
            "ok": ok,
            "backend": request.backend,
            "command": command,
            "structured_output_mode": self.config.structured_output_mode,
            "grammar_used": (self.config.structured_output_mode or "").strip().lower() == "grammar",
            "json_schema_used": (self.config.structured_output_mode or "").strip().lower() == "json_schema",
            "schema_validated": False,
            "json_schema_path": self.config.json_schema_path,
            "grammar_path": self.config.grammar_path,
            "exit_code": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "raw_output": stdout,
            "output_text": self._extract_generation_text(stdout, prompt=request.prompt),
            "prompt_tps": timing.get("prompt_tps"),
            "generation_tps": timing.get("generation_tps"),
            "latency_s": latency,
            "error_type": error_type,
            "error_message": error_message,
        }

    @staticmethod
    def _parse_timing(text: str) -> Dict[str, float | None]:
        match = _TIMING_RE.search(text or "")
        if not match:
            return {"prompt_tps": None, "generation_tps": None}
        return {
            "prompt_tps": float(match.group("prompt")),
            "generation_tps": float(match.group("generation")),
        }

    @staticmethod
    def _extract_generation_text(stdout: str, *, prompt: str | None = None) -> str:
        text = stdout or ""
        if prompt:
            prompt_index = text.find(prompt)
            if prompt_index >= 0:
                text = text[prompt_index + len(prompt) :]
            stripped = text.lstrip()
            if stripped.startswith(prompt):
                text = stripped[len(prompt) :]
        if "\n\n>" in text:
            text = text.split("\n\n>")[-1]
        text = re.sub(r"\[\s*Prompt:.*?Generation:.*?\]\s*", "", text, flags=re.S)
        text = text.replace("Exiting...", "")
        return text.strip()
