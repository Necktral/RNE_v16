"""Primitives for the integral neural qualification campaign.

The module is deliberately orchestration-only: it never grants model authority,
never downloads weights and never silently changes the selected storage backend.
PostgreSQL is the evidence plane for a campaign; filesystem artifacts are indexed
by digest through :class:`runtime.storage.StorageFacade`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import SplitResult, urlsplit, urlunsplit

from runtime.neural.benchmark import expected_calibration_error
from runtime.neural.organs._math import matvec, sigmoid, silu
from runtime.neural.organs.n1_router import FAMILY_CATALOG_V2
from runtime.neural.training import CounterfactualDatasetBuilder, train_n1_router
from runtime.storage import StorageConfig, StorageFactory


CAMPAIGN_SCHEMA_VERSION = "rnfe-integral-neural-campaign-v1"
CHECKPOINT_SCHEMA_VERSION = "rnfe-integral-neural-checkpoint-v1"
HOLDOUT_SCHEMA_VERSION = "rnfe-n1-sealed-holdout-spec-v1"
VERDICT_SCHEMA_VERSION = "rnfe-integral-neural-verdict-v1"
BLOCK_STATES = frozenset({"pending", "running", "completed", "failed"})
SENSITIVE_KEY = re.compile(r"(?:password|passwd|secret|token|dsn|credential)", re.I)
SAFE_CAMPAIGN_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,79}$")
OFFICIAL_ORGANS = tuple(f"N{index}" for index in range(7))
DEFAULT_N1_SEEDS = (31, 47, 73)


class CampaignError(RuntimeError):
    """Fail-closed campaign contract violation."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.staging-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, default=str)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CampaignError(f"campaign_json_object_required:{path}")
    return value


def redact(value: Any) -> Any:
    """Recursively remove credentials before a value reaches evidence files."""
    if isinstance(value, Mapping):
        return {
            str(key): "<redacted>" if SENSITIVE_KEY.search(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str) and "://" in value and "@" in value:
        parsed = urlsplit(value)
        if parsed.hostname:
            host = parsed.hostname
            if parsed.port:
                host = f"{host}:{parsed.port}"
            return urlunsplit((parsed.scheme, f"<redacted>@{host}", parsed.path, "", ""))
    return value


def load_env_file(path: str | Path, *, override: bool = False) -> tuple[str, ...]:
    """Load a simple dotenv without ever returning values to the caller."""
    env_path = Path(path).expanduser().resolve()
    if not env_path.is_file():
        raise CampaignError(f"campaign_env_file_missing:{env_path}")
    loaded: list[str] = []
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        value = value.strip().strip('"').strip("'")
        if override or key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return tuple(sorted(loaded))


def validate_campaign_id(value: str) -> str:
    campaign_id = value.strip().lower()
    if not SAFE_CAMPAIGN_ID.fullmatch(campaign_id):
        raise CampaignError("campaign_id_must_be_safe_lowercase_identifier")
    return campaign_id


def resolve_writable_artifact_root(
    requested: str | Path, *, native_fallback: str | Path
) -> tuple[Path, bool]:
    """Resolve stale native-Linux mount paths without leaving the ext4 workspace.

    The root `.env` may retain a `/media/<user>/<uuid>/...` path from a direct
    Linux boot. Under WSL the same physical partition is mounted at `/home/wis`.
    We use the requested path whenever its nearest existing ancestor is writable;
    otherwise we require an explicitly supplied, writable native fallback.
    """
    requested_path = Path(requested).expanduser().resolve()
    anchor = requested_path
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    if anchor.exists() and os.access(anchor, os.W_OK):
        return requested_path, False
    fallback = Path(native_fallback).expanduser().resolve()
    fallback_anchor = fallback
    while not fallback_anchor.exists() and fallback_anchor != fallback_anchor.parent:
        fallback_anchor = fallback_anchor.parent
    if not fallback_anchor.exists() or not os.access(fallback_anchor, os.W_OK):
        raise CampaignError("campaign_artifact_root_has_no_writable_native_mount")
    return fallback, True


def campaign_database_name(campaign_id: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", validate_campaign_id(campaign_id)).strip("_")
    digest = hashlib.sha256(campaign_id.encode("utf-8")).hexdigest()[:8]
    prefix = normalized[:40] or "campaign"
    return f"rnfe_campaign_{prefix}_{digest}"[:63]


def replace_dsn_database(dsn: str, database: str) -> str:
    parsed = urlsplit(dsn)
    if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname:
        raise CampaignError("campaign_postgres_dsn_invalid")
    return urlunsplit(
        SplitResult(parsed.scheme, parsed.netloc, f"/{database}", parsed.query, "")
    )


def postgres_endpoint(dsn: str) -> dict[str, Any]:
    parsed = urlsplit(dsn)
    return {
        "scheme": parsed.scheme,
        "host": parsed.hostname,
        "port": parsed.port or 5432,
        "database": parsed.path.lstrip("/"),
        "credentials_present": bool(parsed.username),
    }


class PostgresCampaignDatabase:
    """Own a campaign database without exposing credentials in evidence."""

    def __init__(self, *, base_dsn: str, campaign_id: str, schema_path: str | Path):
        self.base_dsn = base_dsn
        self.campaign_id = validate_campaign_id(campaign_id)
        self.database = campaign_database_name(self.campaign_id)
        self.dsn = replace_dsn_database(base_dsn, self.database)
        self.schema_path = Path(schema_path).resolve()

    @property
    def schema_sha256(self) -> str:
        if not self.schema_path.is_file():
            raise CampaignError(f"campaign_postgres_schema_missing:{self.schema_path}")
        return file_sha256(self.schema_path)

    def probe(self, *, campaign_database: bool = False) -> dict[str, Any]:
        import psycopg

        dsn = self.dsn if campaign_database else self.base_dsn
        with psycopg.connect(dsn, connect_timeout=5) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT current_database(), current_schema(), version()")
                database, schema, version = cursor.fetchone()
        return {
            "reachable": True,
            "database": str(database),
            "schema": str(schema),
            "version": str(version).split(",", 1)[0],
            "endpoint": postgres_endpoint(dsn),
        }

    def ensure(self, *, allow_existing: bool) -> dict[str, Any]:
        import psycopg
        from psycopg import sql

        created = False
        with psycopg.connect(self.base_dsn, autocommit=True, connect_timeout=5) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s", (self.database,)
                )
                exists = cursor.fetchone() is not None
                if exists and not allow_existing:
                    raise CampaignError(
                        f"campaign_postgres_database_exists_without_resume:{self.database}"
                    )
                if not exists:
                    cursor.execute(
                        sql.SQL("CREATE DATABASE {}").format(sql.Identifier(self.database))
                    )
                    created = True
        # Initializing the backend applies the idempotent canonical schema.
        storage = self.storage(artifact_root=self.schema_path.parent / ".probe-artifacts")
        storage.close()
        result = self.probe(campaign_database=True)
        result.update(
            {
                "created": created,
                "schema_sha256": self.schema_sha256,
                "storage_mode": "postgres",
                "fallback_used": False,
            }
        )
        return result

    def storage(self, *, artifact_root: str | Path):
        return StorageFactory.create_facade(
            StorageConfig(
                mode="postgres",
                sqlite_db_path=":unused:",
                postgres_dsn=self.dsn,
                artifact_root=Path(artifact_root),
                prefer_postgres_reads=True,
                strict_dual_write=False,
            )
        )

    def dump(self, target: str | Path) -> dict[str, Any]:
        dump_path = Path(target)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = dump_path.with_name(f".{dump_path.name}.staging-{os.getpid()}")
        pg_dump = shutil.which("pg_dump")
        if pg_dump:
            process = subprocess.run(
                [pg_dump, "--format=custom", "--file", str(temporary), self.dsn],
                check=False,
                capture_output=True,
                text=True,
            )
            error = process.stderr.strip()
        else:
            docker = None
            for candidate in (shutil.which("docker.exe"), shutil.which("docker")):
                if not candidate:
                    continue
                check = subprocess.run(
                    [candidate, "version", "--format", "{{.Server.Version}}"],
                    check=False,
                    capture_output=True,
                )
                if check.returncode == 0:
                    docker = candidate
                    break
            if not docker:
                raise CampaignError("campaign_pg_dump_and_docker_unavailable")
            parsed = urlsplit(self.dsn)
            process = subprocess.run(
                [
                    docker,
                    "exec",
                    "rnfe-postgres",
                    "pg_dump",
                    "--format=custom",
                    "--username",
                    parsed.username or "rnfe",
                    "--dbname",
                    self.database,
                ],
                check=False,
                capture_output=True,
            )
            error = process.stderr.decode("utf-8", errors="replace").strip()
            if process.returncode == 0:
                temporary.write_bytes(process.stdout)
        if process.returncode != 0:
            temporary.unlink(missing_ok=True)
            error = error.replace(self.dsn, "<redacted-postgres-dsn>")
            error = error.replace(self.base_dsn, "<redacted-postgres-dsn>")
            raise CampaignError(f"campaign_pg_dump_failed:{error[-500:]}")
        os.replace(temporary, dump_path)
        return {
            "path": str(dump_path.resolve()),
            "sha256": file_sha256(dump_path),
            "size_bytes": dump_path.stat().st_size,
            "database": self.database,
        }


class CampaignState:
    """Atomic manifest and checkpoint state machine."""

    def __init__(self, root: str | Path, manifest: Mapping[str, Any]):
        self.root = Path(root).resolve()
        self.manifest_path = self.root / "campaign_manifest.json"
        self.checkpoint_path = self.root / "checkpoint.json"
        self.manifest = dict(manifest)
        self._validate()

    @classmethod
    def create(
        cls,
        *,
        root: str | Path,
        campaign_id: str,
        commit: str,
        database: str,
        schema_sha256: str,
        artifact_root: str | Path,
        blocks: Sequence[str],
        configuration: Mapping[str, Any],
    ) -> "CampaignState":
        target = Path(root).resolve()
        target.mkdir(parents=True, exist_ok=True)
        manifest_path = target / "campaign_manifest.json"
        if manifest_path.exists():
            raise CampaignError(f"campaign_manifest_already_exists:{manifest_path}")
        created_at = utc_now()
        manifest = {
            "schema_version": CAMPAIGN_SCHEMA_VERSION,
            "campaign_id": validate_campaign_id(campaign_id),
            "commit": commit,
            "created_at": created_at,
            "updated_at": created_at,
            "storage": {
                "mode": "postgres",
                "database": database,
                "schema_sha256": schema_sha256,
                "artifact_root": str(Path(artifact_root).resolve()),
                "sqlite_official_evidence": False,
            },
            "configuration": redact(dict(configuration)),
            "blocks": {
                name: {
                    "status": "pending",
                    "attempts": 0,
                    "started_at": None,
                    "completed_at": None,
                    "result": None,
                    "error": None,
                }
                for name in blocks
            },
            "manual_overnight_approval_required": True,
            "authority_ceiling": "shadow",
            "training_authorized": False,
            "promotion_authorized": False,
        }
        state = cls(target, manifest)
        state.save()
        return state

    @classmethod
    def load(cls, root: str | Path) -> "CampaignState":
        target = Path(root).resolve()
        manifest_path = target / "campaign_manifest.json"
        if not manifest_path.is_file():
            raise CampaignError(f"campaign_manifest_missing:{manifest_path}")
        return cls(target, read_json(manifest_path))

    def _validate(self) -> None:
        if self.manifest.get("schema_version") != CAMPAIGN_SCHEMA_VERSION:
            raise CampaignError("campaign_manifest_schema_mismatch")
        validate_campaign_id(str(self.manifest.get("campaign_id") or ""))
        if self.manifest.get("storage", {}).get("mode") != "postgres":
            raise CampaignError("campaign_official_storage_must_be_postgres")
        blocks = self.manifest.get("blocks")
        if not isinstance(blocks, Mapping) or not blocks:
            raise CampaignError("campaign_blocks_required")
        for payload in blocks.values():
            if not isinstance(payload, Mapping) or payload.get("status") not in BLOCK_STATES:
                raise CampaignError("campaign_block_state_invalid")

    @property
    def campaign_id(self) -> str:
        return str(self.manifest["campaign_id"])

    def save(self) -> None:
        self.manifest["updated_at"] = utc_now()
        atomic_write_json(self.manifest_path, redact(self.manifest))

    def begin(self, name: str) -> None:
        block = self._block(name)
        if block["status"] == "completed":
            return
        block.update(
            {
                "status": "running",
                "attempts": int(block.get("attempts") or 0) + 1,
                "started_at": utc_now(),
                "completed_at": None,
                "result": None,
                "error": None,
            }
        )
        self.save()

    def complete(self, name: str, result: Mapping[str, Any]) -> None:
        block = self._block(name)
        if block["status"] != "running":
            raise CampaignError(f"campaign_block_not_running:{name}")
        block.update(
            {
                "status": "completed",
                "completed_at": utc_now(),
                "result": redact(dict(result)),
                "error": None,
            }
        )
        self.save()

    def fail(self, name: str, error: BaseException) -> None:
        block = self._block(name)
        block.update(
            {
                "status": "failed",
                "completed_at": utc_now(),
                "result": None,
                "error": {
                    "type": type(error).__name__,
                    "message": str(redact(str(error)))[:1000],
                },
            }
        )
        self.save()

    def reset_incomplete(self) -> tuple[str, ...]:
        reset = []
        for name, block in self.manifest["blocks"].items():
            if block["status"] in {"running", "failed"}:
                block.update(
                    {
                        "status": "pending",
                        "started_at": None,
                        "completed_at": None,
                        "result": None,
                        "error": None,
                    }
                )
                reset.append(name)
        self.save()
        return tuple(reset)

    def checkpoint(self, *, phase: str, next_block: str | None) -> dict[str, Any]:
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "phase": phase,
            "next_block": next_block,
            "manifest_sha256": file_sha256(self.manifest_path),
            "created_at": utc_now(),
        }
        payload["checkpoint_hash"] = canonical_sha256(payload)
        atomic_write_json(self.checkpoint_path, payload)
        return payload

    def verify_checkpoint(self, expected_hash: str) -> dict[str, Any]:
        if not self.checkpoint_path.is_file():
            raise CampaignError("campaign_checkpoint_missing")
        payload = read_json(self.checkpoint_path)
        checkpoint_hash = str(payload.pop("checkpoint_hash", ""))
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise CampaignError("campaign_checkpoint_schema_mismatch")
        if payload.get("campaign_id") != self.campaign_id:
            raise CampaignError("campaign_checkpoint_identity_mismatch")
        if checkpoint_hash != canonical_sha256(payload) or checkpoint_hash != expected_hash:
            raise CampaignError("campaign_checkpoint_hash_mismatch")
        if payload.get("manifest_sha256") != file_sha256(self.manifest_path):
            raise CampaignError("campaign_checkpoint_manifest_drift")
        payload["checkpoint_hash"] = checkpoint_hash
        return payload

    def next_pending(self, names: Sequence[str]) -> str | None:
        return next(
            (name for name in names if self._block(name)["status"] != "completed"),
            None,
        )

    def _block(self, name: str) -> dict[str, Any]:
        try:
            return self.manifest["blocks"][name]
        except KeyError as exc:
            raise CampaignError(f"campaign_block_unknown:{name}") from exc


def n1_recalibrate_candidates(
    *,
    paired_records_path: str | Path,
    output_root: str | Path,
    seeds: Sequence[int] = DEFAULT_N1_SEEDS,
    epochs: int = 80,
) -> dict[str, Any]:
    """Train three development candidates and select without a sovereign holdout."""
    source = Path(paired_records_path)
    records = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    builder = CounterfactualDatasetBuilder()
    samples, quality = builder.build(records)
    splits = builder.split(samples)
    if not quality.training_ready() or not all(splits.values()):
        raise CampaignError("campaign_n1_development_data_not_ready")
    unique_seeds = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if len(unique_seeds) < 3:
        raise CampaignError("campaign_n1_requires_three_independent_seeds")
    root = Path(output_root).resolve()
    candidates: list[dict[str, Any]] = []
    for seed in unique_seeds:
        artifact_root = root / "models" / f"seed-{seed}"
        manifest, evidence = train_n1_router(
            splits["train"],
            quality,
            artifact_root=artifact_root,
            seed=seed,
            epochs=epochs,
            dataset_classification="development_exposed_v2_not_sovereign_holdout",
            validation_samples=splits["validation"],
            test_samples=splits["test"],
        )
        validation = dict(evidence["split_metrics"]["validation"])
        candidates.append(
            {
                "seed": seed,
                "artifact_root": str(artifact_root),
                "manifest": manifest.to_dict(),
                "validation": validation,
                "development_test": evidence["split_metrics"]["test"],
                "temperature": evidence["temperature"],
                "promotion_eligible": False,
            }
        )
    candidates.sort(
        key=lambda item: (
            float(item["validation"]["calibration_ece"]),
            float(item["validation"]["positive_bce"]),
            float(item["validation"]["utility_rmse"]),
            int(item["seed"]),
        )
    )
    selected = candidates[0]
    candidate_root = root / "candidate"
    if candidate_root.exists():
        shutil.rmtree(candidate_root)
    shutil.copytree(Path(selected["artifact_root"]), candidate_root)
    report = {
        "schema_version": "rnfe-n1-recalibration-candidates-v1",
        "classification": "development_only",
        "source": str(source.resolve()),
        "source_sha256": file_sha256(source),
        "quality": asdict(quality),
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "seeds": list(unique_seeds),
        "candidates": candidates,
        "selected_seed": selected["seed"],
        "selected_artifact_root": str(candidate_root),
        "holdout_opened": False,
        "promotion_eligible": False,
        "training_authorized": False,
    }
    atomic_write_json(root / "recalibration_report.json", report)
    return report


def seal_holdout_spec(
    *,
    target: str | Path,
    campaign_id: str,
    seed_base: int,
    contexts_per_generator: int,
    scenarios: Sequence[str],
) -> dict[str, Any]:
    path = Path(target)
    if path.exists():
        existing = read_json(path)
        stored_hash = str(existing.pop("spec_hash", ""))
        if stored_hash != canonical_sha256(existing):
            raise CampaignError("campaign_holdout_spec_hash_mismatch")
        existing["spec_hash"] = stored_hash
        return existing
    spec = {
        "schema_version": HOLDOUT_SCHEMA_VERSION,
        "campaign_id": validate_campaign_id(campaign_id),
        "seed_namespace": "sovereign_final_holdout",
        "seed_base": int(seed_base),
        "contexts_per_generator": int(contexts_per_generator),
        "scenarios": list(scenarios),
        "opened": False,
        "created_at": utc_now(),
    }
    spec["spec_hash"] = canonical_sha256(spec)
    atomic_write_json(path, spec)
    return spec


def evaluate_n1_artifact(
    *, artifact_path: str | Path, samples_path: str | Path
) -> dict[str, Any]:
    """Evaluate a frozen JSON router once without fitting on the supplied rows."""
    artifact = read_json(artifact_path)
    rows = [
        json.loads(line)
        for line in Path(samples_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise CampaignError("campaign_n1_holdout_samples_missing")
    features = tuple(str(item) for item in artifact["feature_names"])
    catalog = tuple(str(item) for item in artifact["family_catalog"])
    if catalog != FAMILY_CATALOG_V2:
        raise CampaignError("campaign_n1_catalog_mismatch")
    index = {family: position for position, family in enumerate(catalog)}
    temperature = max(float(artifact.get("temperature", 1.0)), 1e-6)
    predicted_utility: list[float] = []
    actual_utility: list[float] = []
    probabilities: list[float] = []
    labels: list[bool] = []
    for row in rows:
        family = str(row["family"])
        if family not in index:
            raise CampaignError(f"campaign_n1_holdout_family_unknown:{family}")
        x = [float(dict(row["features"]).get(name, 0.0)) for name in features]
        hidden1 = [silu(value) for value in matvec(artifact["w1"], x, artifact["b1"])]
        hidden2 = [
            silu(value)
            for value in matvec(artifact["w2"], hidden1, artifact["b2"])
        ]
        utilities = matvec(artifact["utility_head"], hidden2)
        logits = matvec(artifact["probability_head"], hidden2)
        position = index[family]
        predicted_utility.append(float(utilities[position]))
        actual_utility.append(float(row["utility_delta"]))
        probabilities.append(sigmoid(float(logits[position]) / temperature))
        labels.append(bool(row["positive_utility"]))
    count = len(rows)
    rmse = math.sqrt(
        sum((predicted - actual) ** 2 for predicted, actual in zip(predicted_utility, actual_utility))
        / count
    )
    brier = sum(
        (probability - float(label)) ** 2
        for probability, label in zip(probabilities, labels)
    ) / count
    accuracy = sum((probability >= 0.5) == label for probability, label in zip(probabilities, labels)) / count
    return {
        "schema_version": "rnfe-n1-frozen-holdout-evaluation-v1",
        "records": count,
        "artifact_sha256": file_sha256(artifact_path),
        "samples_sha256": file_sha256(samples_path),
        "temperature": temperature,
        "utility_rmse": rmse,
        "brier": brier,
        "positive_accuracy": accuracy,
        "calibration_ece": expected_calibration_error(probabilities, labels),
        "fitted_on_holdout": False,
    }


def reconcile_artifact_plane(
    *, storage: Any, run_id: str, artifact_root: str | Path
) -> dict[str, Any]:
    root = Path(artifact_root).resolve()
    # The campaign database intentionally uses distinct run_ids for OFF/SHADOW,
    # organs, teacher variants and PostgreSQL integration tests. Reconciliation is
    # therefore scoped by the dedicated artifact root, not one logical run_id.
    indexed = storage.list_artifacts(run_id=None, limit=10_000)
    missing: list[str] = []
    divergent: list[str] = []
    indexed_paths: set[Path] = set()
    for record in indexed:
        path = Path(record.abs_path).resolve()
        indexed_paths.add(path)
        if not path.is_file():
            missing.append(record.artifact_id)
        elif path.stat().st_size != record.size_bytes or file_sha256(path) != record.sha256:
            divergent.append(record.artifact_id)
    filesystem = {
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and ".staging-" not in path.name
    }
    unindexed = sorted(str(path) for path in filesystem - indexed_paths)
    return {
        "schema_version": "rnfe-campaign-artifact-reconciliation-v1",
        "run_id": run_id,
        "indexed": len(indexed),
        "missing_artifact_ids": sorted(missing),
        "divergent_artifact_ids": sorted(divergent),
        "unindexed_paths": unindexed,
        "passed": not missing and not divergent and not unindexed,
    }


def build_integral_verdict(
    *,
    campaign_id: str,
    regression_passed: bool,
    postgres_passed: bool,
    organ_reports: Mapping[str, Mapping[str, Any]],
    holdout: Mapping[str, Any] | None,
    impact_report: Mapping[str, Any] | None,
    artifact_reconciliation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    missing_organs = sorted(set(OFFICIAL_ORGANS) - set(organ_reports))
    n1_ece = float((holdout or {}).get("calibration_ece", 1.0))
    impact_eligible = bool((impact_report or {}).get("promotion_eligible", False))
    safety_violations = int(
        sum(int(report.get("safety_violations", 0)) for report in organ_reports.values())
    )
    gates = {
        "regression_complete": bool(regression_passed),
        "postgres_integration": bool(postgres_passed),
        "all_organs_reported": not missing_organs,
        "n1_fresh_holdout_opened": bool(holdout),
        "n1_holdout_ece_lte_0_10": bool(holdout) and n1_ece <= 0.10,
        "a_m0_impact_favorable": impact_eligible,
        "zero_safety_violations": safety_violations == 0,
        "artifact_reconciliation": bool(
            artifact_reconciliation and artifact_reconciliation.get("passed")
        ),
    }
    staging_authorized = all(gates.values())
    return {
        "schema_version": VERDICT_SCHEMA_VERSION,
        "campaign_id": validate_campaign_id(campaign_id),
        "classification": "experimental_shadow_only",
        "gates": gates,
        "missing_organs": missing_organs,
        "safety_violations": safety_violations,
        "staging_authorized": staging_authorized,
        "shadow_qualification_passed": staging_authorized,
        "promotion_eligible": False,
        "promotion_authorized": False,
        "training_authorized": False,
        "authority_ceiling": "shadow",
    }


__all__ = [
    "BLOCK_STATES",
    "CAMPAIGN_SCHEMA_VERSION",
    "CHECKPOINT_SCHEMA_VERSION",
    "CampaignError",
    "CampaignState",
    "DEFAULT_N1_SEEDS",
    "HOLDOUT_SCHEMA_VERSION",
    "OFFICIAL_ORGANS",
    "PostgresCampaignDatabase",
    "VERDICT_SCHEMA_VERSION",
    "atomic_write_json",
    "build_integral_verdict",
    "campaign_database_name",
    "canonical_sha256",
    "evaluate_n1_artifact",
    "file_sha256",
    "load_env_file",
    "n1_recalibrate_candidates",
    "postgres_endpoint",
    "reconcile_artifact_plane",
    "redact",
    "resolve_writable_artifact_root",
    "replace_dsn_database",
    "seal_holdout_spec",
    "validate_campaign_id",
]
