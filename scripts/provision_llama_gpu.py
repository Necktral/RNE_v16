#!/usr/bin/env python3
"""Provisión del workload GPU: llama.cpp-CUDA + GGUFs (razonador + embeddings).

Este es el ÚNICO paso que necesita red/descargas. Todo el núcleo (router real,
sensado GPU, razonador admitido, seam de embeddings) funciona sin él; sin
artefactos, el razonador devuelve skip y los embeddings caen a Jaccard.

Uso (con el usuario, confirmando tamaños):
  # 1) Diagnóstico (sin red): qué GPU hay y qué falta.
  python scripts/provision_llama_gpu.py --check

  # 2) Descarga (necesita red + consentimiento; ~4.5GB el GGUF del 7B).
  python scripts/provision_llama_gpu.py --download --models-root ~/rnfe_models

  # 3) Escribir el .env nativo con las rutas resueltas.
  python scripts/provision_llama_gpu.py --write-env .env.external_reasoner.local --models-root ~/rnfe_models

  # 4) Smoke en la GPU (una llamada real; observá el salto de VRAM en nvidia-smi).
  python scripts/provision_llama_gpu.py --smoke --models-root ~/rnfe_models

Diseño: por defecto NO descarga nada. Requiere --download explícito para tocar la red.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# GGUF del razonador (coincide con .env.external_reasoner.example).
REASONER_REPO = "lmstudio-community/OpenThinker3-7B-GGUF"
REASONER_FILE = "OpenThinker3-7B-Q4_K_M.gguf"
REASONER_SHA256 = "0b7344e4bf1c68fc40c4a10b14b9bd51f367423b8453d83544ea5bdbe08e7e5e"

# GGUF de embeddings (pequeño, ~80MB).
EMBED_REPO = "nomic-ai/nomic-embed-text-v1.5-GGUF"
EMBED_FILE = "nomic-embed-text-v1.5.Q4_K_M.gguf"

# llama.cpp NO publica binario CUDA para Linux; el camino que usa la GPU sin toolkit
# es el backend VULKAN (el RTX 2070 es un dispositivo Vulkan vía el driver NVIDIA).
LLAMA_TAG = "b9874"
LLAMA_VULKAN_ASSET = f"llama-{LLAMA_TAG}-bin-ubuntu-vulkan-x64.tar.gz"
LLAMA_VULKAN_URL = (
    f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/{LLAMA_VULKAN_ASSET}"
)
LLAMA_RELEASE_HINT = (
    f"Bajá {LLAMA_VULKAN_URL} y descomprimí en "
    "<models-root>/tools/llama.cpp/build-vulkan (backend Vulkan, offload a la GPU con -ngl)."
)


def _gpu_info() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (out.stdout or "").strip() or None


def _find_llama_cli(models_root: Path) -> Path | None:
    env = os.environ.get("RNFE_LLAMA_CLI_CUDA", "").strip()
    if env and Path(env).exists():
        return Path(env)
    base = models_root / "tools" / "llama.cpp"
    # Busca el llama-cli del build Vulkan (o cualquier subdir del release).
    for pattern in ("build-vulkan/**/llama-cli", "build-cuda/**/llama-cli", "**/llama-cli"):
        for candidate in base.glob(pattern):
            if candidate.is_file():
                return candidate
    found = shutil.which("llama-cli")
    return Path(found) if found else None


def _download_llama_vulkan(models_root: Path) -> Path | None:
    import tarfile
    import urllib.request

    dest_dir = models_root / "tools" / "llama.cpp" / "build-vulkan"
    dest_dir.mkdir(parents=True, exist_ok=True)
    tar_path = dest_dir / LLAMA_VULKAN_ASSET
    print(f"  bajando {LLAMA_VULKAN_ASSET} ...")
    try:
        urllib.request.urlretrieve(LLAMA_VULKAN_URL, tar_path)
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(dest_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR bajando llama.cpp: {type(exc).__name__}: {exc}")
        return None
    return _find_llama_cli(models_root)


def _paths(models_root: Path) -> dict[str, Path]:
    return {
        "reasoner_gguf": models_root / "gguf" / "OpenThinker3-7B" / REASONER_FILE,
        "embed_gguf": models_root / "gguf" / "embeddings" / EMBED_FILE,
    }


def cmd_check(models_root: Path) -> int:
    gpu = _gpu_info()
    print("== Diagnóstico de provisión GPU ==")
    print(f"GPU: {gpu or 'NO detectada (nvidia-smi ausente o sin GPU)'}")
    cli = _find_llama_cli(models_root)
    print(f"llama.cpp CUDA cli: {cli or 'FALTA -> ' + LLAMA_RELEASE_HINT}")
    paths = _paths(models_root)
    for label, p in paths.items():
        print(f"{label}: {'OK ' + str(p) if p.exists() else 'FALTA ' + str(p)}")
    missing = [k for k, p in paths.items() if not p.exists()] + ([] if cli else ["llama_cli"])
    if missing:
        print(f"\nFaltan: {', '.join(missing)}")
        print("Corré con --download para bajar los GGUF (necesita red y ~4.6GB).")
        return 1
    print("\nTodo presente. Usá --write-env para generar el .env y --smoke para probar en GPU.")
    return 0


def _hf_download(repo: str, filename: str, dest: Path) -> Path:
    from huggingface_hub import hf_hub_download  # type: ignore

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  bajando {repo}/{filename} ...")
    local = hf_hub_download(repo_id=repo, filename=filename)
    shutil.copy(local, dest)
    return dest


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_download(models_root: Path, *, embeddings: bool) -> int:
    print("== Descarga de artefactos (red) ==")
    paths = _paths(models_root)
    try:
        if not paths["reasoner_gguf"].exists():
            _hf_download(REASONER_REPO, REASONER_FILE, paths["reasoner_gguf"])
            digest = _sha256(paths["reasoner_gguf"])
            if REASONER_SHA256 and digest != REASONER_SHA256:
                print(f"  ADVERTENCIA: SHA256 no coincide ({digest} != {REASONER_SHA256})")
        if embeddings and not paths["embed_gguf"].exists():
            _hf_download(EMBED_REPO, EMBED_FILE, paths["embed_gguf"])
    except Exception as exc:  # noqa: BLE001 - reporte claro al usuario
        print(f"  ERROR de descarga: {type(exc).__name__}: {exc}")
        return 1
    cli = _find_llama_cli(models_root)
    if cli is None:
        print("  binario llama.cpp ausente -> bajando build Vulkan ...")
        cli = _download_llama_vulkan(models_root)
    if cli is None:
        print(f"\nNo se pudo obtener el binario. {LLAMA_RELEASE_HINT}")
        return 1
    print(f"Descarga OK. llama-cli: {cli}")
    return 0


def cmd_write_env(models_root: Path, out: Path) -> int:
    cli = _find_llama_cli(models_root)
    paths = _paths(models_root)
    cli_path = str(cli or models_root / "tools/llama.cpp/build-vulkan/llama-cli")
    lines = [
        f'export RNFE_MODELS_ROOT="{models_root}"',
        f'export RNFE_REASONING_GGUF="{paths["reasoner_gguf"]}"',
        f"export RNFE_EXPECTED_GGUF_SHA256={REASONER_SHA256}",
        # El binario es Vulkan; el config lo usa como "cli cuda" (único camino con -ngl>0).
        f'export RNFE_LLAMA_CLI_CUDA="{cli_path}"',
        f'export RNFE_LLAMA_CLI_CPU="{cli_path}"',
        "export RNFE_EXTERNAL_REASONER_BACKEND=cuda",
        "export RNFE_EXTERNAL_REASONER_NGL=99",
        # Aísla el RTX 2070 del iGPU Intel (ajustá el índice si tu layout difiere).
        "export GGML_VK_VISIBLE_DEVICES=1",
        "export RNFE_EXTERNAL_REASONER_RUNTIME=1",
        "export RNFE_CONJUNCTION_ROUTING_ENFORCED=1",
        "export RNFE_HOST_SENSING=1",
        # 'hashed' (CPU): este release de llama.cpp no trae llama-embedding. Para
        # embeddings en GPU: llama-server --embedding (seguimiento aparte).
        "export RNFE_MEMORY_EMBEDDINGS=hashed",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Escrito {out}. Cargalo con:  source {out}")
    return 0


def cmd_smoke(models_root: Path) -> int:
    cli = _find_llama_cli(models_root)
    gguf = _paths(models_root)["reasoner_gguf"]
    if cli is None or not gguf.exists():
        print("No se puede smoke: falta el binario CUDA o el GGUF. Corré --check.")
        return 1
    command = [
        str(cli), "-m", str(gguf), "-p", "Say OK.", "-n", "8", "-ngl", "99",
        "--no-warmup", "--simple-io", "--single-turn", "--no-display-prompt",
    ]
    print("== Smoke en GPU (mirá nvidia-smi en otra terminal) ==")
    try:
        out = subprocess.run(command, check=False, capture_output=True, text=True, timeout=240)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"ERROR: {exc}")
        return 1
    print("stdout:", (out.stdout or "").strip()[:200])
    return 0 if out.returncode == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Provisión GPU llama.cpp + GGUFs.")
    parser.add_argument("--models-root", default=os.environ.get("RNFE_MODELS_ROOT", str(Path.home() / "rnfe_models")))
    parser.add_argument("--check", action="store_true", help="Diagnóstico (default, sin red).")
    parser.add_argument("--download", action="store_true", help="Descargar GGUFs (necesita red).")
    parser.add_argument("--no-embeddings", action="store_true", help="No bajar el GGUF de embeddings.")
    parser.add_argument("--write-env", metavar="PATH", help="Escribir un .env con las rutas.")
    parser.add_argument("--smoke", action="store_true", help="Correr una inferencia de prueba en GPU.")
    args = parser.parse_args(argv)

    models_root = Path(os.path.expanduser(args.models_root)).resolve()

    if args.download:
        return cmd_download(models_root, embeddings=not args.no_embeddings)
    if args.write_env:
        return cmd_write_env(models_root, Path(args.write_env))
    if args.smoke:
        return cmd_smoke(models_root)
    return cmd_check(models_root)


if __name__ == "__main__":
    sys.exit(main())
