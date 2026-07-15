"""Valida y prepara artefactos neurales de laboratorio para ejecución SHADOW.

No activa variables de entorno, no cambia autoridad y rechaza cualquier artefacto
que se declare promocionable. El destino es un artifact plane, nunca el paquete de
código nominal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.neural.contracts import NeuralModelManifest
from runtime.neural.integration.model_bindings import MODEL_MANIFEST_ENV
from runtime.neural.technology_backends import HNET_BACKEND_ID, MAMBA2_BACKEND_ID


EXPECTED_BACKENDS: Mapping[str, str] = {
    "N1": "rnfe-compact-mlp-router-v1",
    "N3": MAMBA2_BACKEND_ID,
    "N4": "rnfe-trained-causal-graph-v1",
    "N5": HNET_BACKEND_ID,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: str, *, label: str) -> PurePosixPath:
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"neural_artifact_path_unsafe:{label}")
    return relative


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.staging-{os.getpid()}")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.staging-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def stage_lab_artifacts(
    *,
    source_root: Path,
    target_root: Path,
    organs: Iterable[str] = ("N1", "N3", "N4", "N5"),
) -> dict[str, Any]:
    source_root = source_root.resolve()
    target_root = target_root.resolve()
    staged: list[dict[str, Any]] = []
    missing: list[str] = []
    for organ in tuple(dict.fromkeys(str(item).upper() for item in organs)):
        if organ not in EXPECTED_BACKENDS:
            raise ValueError(f"neural_artifact_organ_unsupported:{organ}")
        source_dir = source_root / organ.lower()
        manifest_path = source_dir / "manifest.json"
        if not manifest_path.is_file():
            missing.append(organ)
            continue
        manifest = NeuralModelManifest.from_dict(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        if manifest.organ != organ or manifest.backend != EXPECTED_BACKENDS[organ]:
            raise ValueError(f"neural_artifact_manifest_contract_mismatch:{organ}")
        if manifest.training_provenance.get("promotion_eligible") is not False:
            raise ValueError(f"neural_lab_artifact_must_be_non_promotable:{organ}")
        artifact_relative = _safe_relative(
            manifest.artifact_path, label=f"{organ}:artifact"
        )
        source_artifact = (source_root / Path(*artifact_relative.parts)).resolve()
        try:
            source_artifact.relative_to(source_root)
        except ValueError as exc:
            raise ValueError(f"neural_artifact_escapes_source:{organ}") from exc
        if not source_artifact.is_file():
            raise FileNotFoundError(f"neural_artifact_missing:{organ}")
        if _sha256(source_artifact) != manifest.artifact_sha256:
            raise ValueError(f"neural_artifact_sha256_mismatch:{organ}")

        target_artifact = target_root / Path(*artifact_relative.parts)
        target_manifest = target_root / organ.lower() / "manifest.json"
        _atomic_copy(source_artifact, target_artifact)
        _atomic_copy(manifest_path, target_manifest)
        copied_metadata = []
        for name in ("model_card.json", "dataset_manifest.json"):
            source_metadata = source_dir / name
            if source_metadata.is_file():
                _atomic_copy(source_metadata, target_root / organ.lower() / name)
                copied_metadata.append(name)
        staged.append(
            {
                "organ": organ,
                "model_id": manifest.model_id,
                "backend": manifest.backend,
                "manifest": f"{organ.lower()}/manifest.json",
                "artifact": manifest.artifact_path,
                "artifact_sha256": manifest.artifact_sha256,
                "metadata": copied_metadata,
                "environment": {
                    MODEL_MANIFEST_ENV[organ]: f"{organ.lower()}/manifest.json"
                },
                "authority_ceiling": "shadow",
                "training_authorized": False,
                "promotion_authorized": False,
            }
        )
    profile = {
        "schema_version": "rnfe-neural-lab-activation-profile-v1",
        "classification": "laboratory_shadow_only",
        "source_root": str(source_root),
        "target_root": str(target_root),
        "staged": staged,
        "missing": missing,
        "training_authorized": False,
        "promotion_authorized": False,
        "activation_automatic": False,
    }
    _write_json(target_root / "activation_profile.json", profile)
    return profile


def _parse_organs(value: str) -> list[str]:
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--organs", default="N1,N3,N4,N5")
    args = parser.parse_args(argv)
    profile = stage_lab_artifacts(
        source_root=args.source_root,
        target_root=args.target_root,
        organs=_parse_organs(args.organs),
    )
    print(json.dumps(profile, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
