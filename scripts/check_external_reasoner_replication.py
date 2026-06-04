#!/usr/bin/env python3
"""Check that the RNFE external reasoner environment is reproducible."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPECTED_SHA256 = "0b7344e4bf1c68fc40c4a10b14b9bd51f367423b8453d83544ea5bdbe08e7e5e"
SCHEMA_PATH = REPO_ROOT / "runtime/reasoning/external_models/schemas/ext_open_thinker_response.schema.json"
PROHIBITED_WEIGHT_SUFFIXES = {
    ".gguf",
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".onnx",
    ".ckpt",
}
REQUIRED_ENV = [
    "RNFE_MODELS_ROOT",
    "RNFE_REASONING_GGUF",
    "RNFE_LLAMA_CLI_CUDA",
    "RNFE_LLAMA_CLI_CPU",
    "CUDA_ROOT",
    "LD_LIBRARY_PATH",
]
RECOMMENDED_ENV = [
    "HF_HOME",
    "HF_HUB_CACHE",
    "TORCH_HOME",
    "PIP_CACHE_DIR",
]


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    severity: str = "FAIL"


def _run(args: list[str], *, timeout_s: float = 10.0) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            args,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_ls_files(path: Path | None = None) -> set[str]:
    args = ["git", "ls-files"]
    if path is not None:
        args.append(str(path))
    proc = _run(args, timeout_s=20.0)
    if proc is None or proc.returncode != 0:
        return set()
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def _git_status_untracked() -> set[str]:
    proc = _run(["git", "status", "--short"], timeout_s=20.0)
    if proc is None or proc.returncode != 0:
        return set()
    out: set[str] = set()
    for line in proc.stdout.splitlines():
        if line.startswith("?? "):
            out.add(line[3:].rstrip("/"))
    return out


def _iter_repo_weights() -> list[Path]:
    ignored_dirs = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".cache",
    }
    weights: list[Path] = []
    for root, dirs, files in os.walk(REPO_ROOT):
        rel_root = Path(root).relative_to(REPO_ROOT)
        dirs[:] = [item for item in dirs if item not in ignored_dirs]
        for filename in files:
            path = Path(root) / filename
            if path.suffix.lower() in PROHIBITED_WEIGHT_SUFFIXES:
                weights.append(path.relative_to(REPO_ROOT))
    return sorted(weights)


def check_env() -> list[Check]:
    checks: list[Check] = []
    for name in REQUIRED_ENV:
        value = os.environ.get(name, "")
        checks.append(Check(f"env:{name}", bool(value), "set" if value else "missing"))
    for name in RECOMMENDED_ENV:
        value = os.environ.get(name, "")
        checks.append(
            Check(
                f"env:{name}",
                True,
                "set" if value else "missing recommended cache env",
                severity="WARN",
            )
        )
    return checks


def check_paths() -> list[Check]:
    checks: list[Check] = []
    gguf_raw = os.environ.get("RNFE_REASONING_GGUF", "")
    cuda_cli_raw = os.environ.get("RNFE_LLAMA_CLI_CUDA", "")
    cpu_cli_raw = os.environ.get("RNFE_LLAMA_CLI_CPU", "")
    cuda_root_raw = os.environ.get("CUDA_ROOT", "")
    gguf = Path(gguf_raw) if gguf_raw else None
    cuda_cli = Path(cuda_cli_raw) if cuda_cli_raw else None
    cpu_cli = Path(cpu_cli_raw) if cpu_cli_raw else None
    cuda_root = Path(cuda_root_raw) if cuda_root_raw else None

    checks.append(Check("gguf_exists", bool(gguf and gguf.exists()), str(gguf) if gguf else "missing env"))
    checks.append(
        Check(
            "llama_cli_cuda_exists",
            bool(cuda_cli and cuda_cli.exists()),
            str(cuda_cli) if cuda_cli else "missing env",
        )
    )
    checks.append(
        Check(
            "llama_cli_cpu_exists",
            bool(cpu_cli and cpu_cli.exists()),
            str(cpu_cli) if cpu_cli else "missing env",
        )
    )
    checks.append(
        Check(
            "cuda_root_lib_exists",
            bool(cuda_root and (cuda_root / "lib").exists()),
            str(cuda_root / "lib") if cuda_root else "missing env",
        )
    )

    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    expected_cuda_lib = str(cuda_root / "lib") if cuda_root else ""
    checks.append(
        Check(
            "ld_library_path_contains_cuda_root_lib",
            bool(expected_cuda_lib and expected_cuda_lib in ld_library_path),
            expected_cuda_lib or "missing CUDA_ROOT",
        )
    )
    checks.append(Check("schema_exists", SCHEMA_PATH.exists(), str(SCHEMA_PATH)))
    return checks


def check_hash() -> list[Check]:
    gguf_raw = os.environ.get("RNFE_REASONING_GGUF", "")
    expected = os.environ.get("RNFE_EXPECTED_GGUF_SHA256", DEFAULT_EXPECTED_SHA256).strip().lower()
    if not gguf_raw:
        return [Check("gguf_sha256", False, "RNFE_REASONING_GGUF missing")]
    gguf = Path(gguf_raw)
    if not gguf.exists():
        return [Check("gguf_sha256", False, f"not found: {gguf}")]
    try:
        actual = _sha256(gguf)
    except OSError as exc:
        return [Check("gguf_sha256", False, f"read error: {exc}")]
    return [Check("gguf_sha256", actual == expected, f"actual={actual} expected={expected}")]


def check_commands() -> list[Check]:
    checks: list[Check] = []
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        proc = _run([nvidia_smi], timeout_s=10.0)
        checks.append(
            Check(
                "nvidia_smi",
                proc is not None and proc.returncode == 0,
                "available" if proc is not None and proc.returncode == 0 else "command failed",
            )
        )
    else:
        checks.append(Check("nvidia_smi", False, "not found in PATH"))

    cuda_cli = os.environ.get("RNFE_LLAMA_CLI_CUDA", "")
    if not cuda_cli:
        checks.append(Check("llama_cli_json_schema_flag", False, "RNFE_LLAMA_CLI_CUDA missing"))
        return checks
    proc = _run([cuda_cli, "--help"], timeout_s=20.0)
    help_text = ""
    if proc is not None:
        help_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    checks.append(
        Check(
            "llama_cli_json_schema_flag",
            proc is not None and "--json-schema-file" in help_text,
            "--json-schema-file present" if "--json-schema-file" in help_text else "flag not found",
        )
    )
    return checks


def check_repo_weights() -> list[Check]:
    weights = _iter_repo_weights()
    if not weights:
        return [Check("repo_weight_files", True, "none")]

    tracked = _git_ls_files()
    untracked_roots = _git_status_untracked()
    failing: list[str] = []
    tracked_legacy: list[str] = []
    for path in weights:
        path_str = str(path)
        if path_str in tracked:
            tracked_legacy.append(path_str)
            continue
        if any(path_str == root or path_str.startswith(root + "/") for root in untracked_roots):
            failing.append(path_str)
        else:
            failing.append(path_str)

    checks: list[Check] = []
    if tracked_legacy:
        checks.append(
            Check(
                "repo_weight_files_tracked_legacy",
                True,
                "tracked before checkpoint: " + ", ".join(tracked_legacy),
                severity="WARN",
            )
        )
    checks.append(
        Check(
            "repo_weight_files_new_or_untracked",
            not failing,
            "none" if not failing else ", ".join(failing),
        )
    )
    return checks


def print_report(checks: list[Check]) -> None:
    print("RNFE external reasoner replication check")
    print(f"repo={REPO_ROOT}")
    print("")
    for check in checks:
        if check.ok and check.severity == "FAIL":
            status = "OK"
        elif check.ok:
            status = check.severity
        else:
            status = check.severity
        print(f"[{status}] {check.name}: {check.detail}")


def main() -> int:
    checks: list[Check] = []
    try:
        checks.extend(check_env())
        checks.extend(check_paths())
        checks.extend(check_hash())
        checks.extend(check_commands())
        checks.extend(check_repo_weights())
    except Exception as exc:  # pragma: no cover - defensive CLI guard
        checks.append(Check("unexpected_error", False, str(exc)))

    print_report(checks)
    failed = [check for check in checks if not check.ok and check.severity == "FAIL"]
    if failed:
        print("")
        print(f"ready=false failures={len(failed)}")
        return 1
    print("")
    print("ready=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
