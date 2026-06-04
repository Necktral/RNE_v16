from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from runtime.reasoning.external_models.config import ExternalReasonerConfig
from runtime.reasoning.external_models.llama_cpp_client import LlamaCppClient


def _config(tmp_path: Path) -> ExternalReasonerConfig:
    model = tmp_path / "model.gguf"
    cuda_cli = tmp_path / "llama-cuda"
    cpu_cli = tmp_path / "llama-cpu"
    model.write_text("model", encoding="utf-8")
    cuda_cli.write_text("cuda", encoding="utf-8")
    cpu_cli.write_text("cpu", encoding="utf-8")
    return ExternalReasonerConfig(
        model_path=str(model),
        cuda_cli_path=str(cuda_cli),
        cpu_cli_path=str(cpu_cli),
        backend="cuda",
        ngl=99,
        max_tokens=64,
        timeout_s=5.0,
        cuda_root=str(tmp_path / "cuda"),
        ld_library_path="/already/set",
    )


def test_cuda_command_uses_list_and_shell_false(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout='{"claim":"ok","recommended_intervention":"activate_cooling"}\n[ Prompt: 12.3 t/s | Generation: 45.6 t/s ]',
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    client = LlamaCppClient(_config(tmp_path))
    result = client.generate("hello", backend="cuda")

    command, kwargs = calls[0]
    assert result["ok"] is True
    assert isinstance(command, list)
    assert kwargs["shell"] is False
    assert "-ngl" in command
    assert command[command.index("-ngl") + 1] == "99"
    assert "--json-schema-file" in command
    assert "--reasoning-budget" in command
    assert kwargs["env"]["LD_LIBRARY_PATH"].startswith(str(tmp_path / "cuda" / "lib"))
    assert result["prompt_tps"] == 12.3
    assert result["generation_tps"] == 45.6
    assert result["structured_output_mode"] == "json_schema"
    assert result["json_schema_used"] is True


def test_config_reads_latency_knobs_from_env(monkeypatch) -> None:
    monkeypatch.setenv("RNFE_REASONING_GGUF", "/tmp/model.gguf")
    monkeypatch.setenv("RNFE_EXTERNAL_REASONER_MAX_TOKENS", "192")
    monkeypatch.setenv("RNFE_EXTERNAL_REASONER_CTX_SIZE", "1024")
    monkeypatch.setenv("RNFE_EXTERNAL_REASONER_BATCH_SIZE", "128")
    monkeypatch.setenv("RNFE_EXTERNAL_REASONER_UBATCH_SIZE", "64")
    monkeypatch.setenv("RNFE_EXTERNAL_REASONER_THREADS", "8")
    monkeypatch.setenv("RNFE_EXTERNAL_REASONER_THREADS_BATCH", "8")
    monkeypatch.setenv("RNFE_EXTERNAL_REASONER_MLOCK", "true")
    monkeypatch.setenv("RNFE_EXTERNAL_REASONER_PROMPT_STYLE", "standard")

    config = ExternalReasonerConfig.from_env()

    assert config.max_tokens == 192
    assert config.ctx_size == 1024
    assert config.batch_size == 128
    assert config.ubatch_size == 64
    assert config.threads == 8
    assert config.threads_batch == 8
    assert config.mlock is True
    assert config.prompt_style == "standard"


def test_latency_knobs_are_passed_to_llama_cli(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout='{"claim":"ok"}\n[ Prompt: 1.0 t/s | Generation: 2.0 t/s ]',
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    config = _config(tmp_path)
    client = LlamaCppClient(config)

    client.generate(
        "hello",
        backend="cuda",
        max_tokens=96,
        ctx_size=1024,
        batch_size=128,
        ubatch_size=64,
        threads=6,
        threads_batch=4,
        mlock=True,
    )

    command = calls[0]
    assert command[command.index("-n") + 1] == "96"
    assert command[command.index("--ctx-size") + 1] == "1024"
    assert command[command.index("--batch-size") + 1] == "128"
    assert command[command.index("--ubatch-size") + 1] == "64"
    assert command[command.index("--threads") + 1] == "6"
    assert command[command.index("--threads-batch") + 1] == "4"
    assert "--mlock" in command


def test_timeout_returns_structured_error(monkeypatch, tmp_path: Path) -> None:
    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = LlamaCppClient(_config(tmp_path)).generate("hello", backend="cuda")
    assert result["ok"] is False
    assert result["error_type"] == "timeout"
    assert result["backend"] == "cuda"


def test_missing_libcudart_is_classified(monkeypatch, tmp_path: Path) -> None:
    def fake_run(command, **kwargs):
        return SimpleNamespace(
            returncode=127,
            stdout="",
            stderr="libcudart.so.13: cannot open shared object file",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = LlamaCppClient(_config(tmp_path)).generate("hello", backend="cuda")
    assert result["ok"] is False
    assert result["error_type"] == "missing_cuda_library"


def test_cpu_fallback_only_when_explicit(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            return SimpleNamespace(returncode=1, stdout="", stderr="cuda failed")
        return SimpleNamespace(
            returncode=0,
            stdout='{"claim":"cpu ok"}\n[ Prompt: 3.0 t/s | Generation: 4.0 t/s ]',
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = LlamaCppClient(_config(tmp_path)).generate(
        "hello",
        backend="cuda",
        allow_cpu_fallback=True,
    )
    assert result["ok"] is True
    assert result["backend"] == "cpu"
    assert result["fallback_attempted"] is True
    assert len(calls) == 2
    assert calls[1][calls[1].index("-ngl") + 1] == "0"


def test_extract_generation_text_strips_echoed_prompt() -> None:
    prompt = '{"task":"return only json"}'
    stdout = f'{prompt}\n\n{{"claim":"generated"}}\n[ Prompt: 1.0 t/s | Generation: 2.0 t/s ]'
    assert LlamaCppClient._extract_generation_text(stdout, prompt=prompt) == '{"claim":"generated"}'
