#!/usr/bin/env python3
"""Run the resumable RNFE integral neural campaign on native Linux through WSL.

Official evidence is written directly to a dedicated PostgreSQL database. SQLite
is used only by the explicit contingency block and ephemeral paired-world scratch
branches. All neural outputs remain experimental proposals with a SHADOW ceiling.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.life import LifeKernel, LifeKernelConfig
from runtime.neural import ImpactObservation, OrganismImpactVector, build_impact_report
from runtime.neural.campaign import (
    CampaignError,
    CampaignState,
    OFFICIAL_ORGANS,
    PostgresCampaignDatabase,
    atomic_write_json,
    build_integral_verdict,
    evaluate_n1_artifact,
    file_sha256,
    load_env_file,
    n1_recalibrate_candidates,
    reconcile_artifact_plane,
    seal_holdout_spec,
    validate_campaign_id,
)
from runtime.neural.connectome import canonical_connectome
from runtime.neural.contracts import NeuralModelManifest
from runtime.neural.lab.n4_benchmark import run_n4_synthetic_benchmark
from runtime.neural.organs.n4_causal import (
    ARTIFACT_SCHEMA_VERSION as N4_ARTIFACT_SCHEMA_VERSION,
    GRAPH_SCHEMA_VERSION as N4_GRAPH_SCHEMA_VERSION,
    OUTPUT_SCHEMA_VERSION as N4_OUTPUT_SCHEMA_VERSION,
    CausalMessagePassingBackend,
)
from runtime.neural.organs.n5_ingest import DeterministicChunker
from runtime.neural.organs.n6_evolution import (
    StructuralEvolutionGate,
    StructuralMutationProposal,
)
from runtime.storage import StorageConfig, StorageFactory
from runtime.storage.migrations import migrate_sqlite_ledger_to_postgres
from scripts.benchmark_n1_counterfactual import run_n1_counterfactual_campaign
from scripts.benchmark_teacher_advanced import run_campaign as run_teacher_campaign
from scripts.stage_neural_lab_artifacts import stage_lab_artifacts


DEFAULT_ENV_FILE = Path("/home/wis/Desarrollo/RNE_v16/.env")
SCHEMA_PATH = REPO_ROOT / "runtime/storage/backends/postgres/schema.sql"
GGUF_7B = Path(
    "/home/wis/rnfe_models/gguf/OpenThinker3-7B/OpenThinker3-7B-Q4_K_M.gguf"
)
SCENARIOS = (
    "thermal_homeostasis",
    "resource_management",
    "grid_thermal_5x5",
    "deferred_load_trap",
)
PREFLIGHT_BLOCKS = (
    "preflight_environment",
    "preflight_postgres",
    "preflight_connectome",
    "preflight_artifacts",
)
REHEARSAL_BLOCKS = (
    "regression_full",
    "qualify_n0",
    "qualify_n1",
    "qualify_n2",
    "qualify_n3",
    "qualify_n4",
    "qualify_n5",
    "qualify_n6",
    "life_kernel_paired_rehearsal",
    "teacher_7b_rehearsal",
    "sqlite_contingency",
    "reconcile_rehearsal",
    "dump_rehearsal",
)
OVERNIGHT_BLOCKS = (
    "open_fresh_holdout",
    "life_kernel_paired_overnight",
    "teacher_7b_overnight",
    "reconcile_overnight",
    "verdict_overnight",
    "dump_overnight",
)
ALL_BLOCKS = PREFLIGHT_BLOCKS + REHEARSAL_BLOCKS + OVERNIGHT_BLOCKS


@dataclass(slots=True)
class RuntimeContext:
    state: CampaignState
    postgres: PostgresCampaignDatabase
    storage_config: StorageConfig
    artifact_root: Path
    env_file: Path
    storage: Any | None = None

    def ensure_storage(self) -> Any:
        if self.storage is None:
            self.storage = self.postgres.storage(artifact_root=self.artifact_root)
        return self.storage

    def close(self) -> None:
        if self.storage is not None:
            self.storage.close()
            self.storage = None


@contextmanager
def temporary_environ(values: Mapping[str, str | None]):
    previous = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def _load_context(args: argparse.Namespace) -> RuntimeContext:
    env_file = Path(args.env_file).expanduser().resolve()
    loaded_keys = load_env_file(env_file, override=True)
    config = StorageConfig.from_env()
    if config.mode != "postgres" or not config.postgres_dsn:
        raise CampaignError("integral_campaign_requires_postgres_direct_configuration")
    campaign_id = validate_campaign_id(args.campaign_id)
    output_base = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else config.artifact_root.resolve() / "integral_campaigns"
    )
    root = output_base / campaign_id
    postgres = PostgresCampaignDatabase(
        base_dsn=config.postgres_dsn,
        campaign_id=campaign_id,
        schema_path=SCHEMA_PATH,
    )
    manifest_path = root / "campaign_manifest.json"
    if manifest_path.is_file():
        state = CampaignState.load(root)
        if state.manifest["storage"]["database"] != postgres.database:
            raise CampaignError("campaign_database_manifest_mismatch")
        if state.manifest["commit"] != _git("rev-parse", "HEAD"):
            raise CampaignError("campaign_code_commit_drift_requires_new_campaign")
    else:
        state = CampaignState.create(
            root=root,
            campaign_id=campaign_id,
            commit=_git("rev-parse", "HEAD"),
            database=postgres.database,
            schema_sha256=postgres.schema_sha256,
            artifact_root=root / "artifacts",
            blocks=ALL_BLOCKS,
            configuration={
                "env_file": str(env_file),
                "env_keys_loaded": list(loaded_keys),
                "scenarios": list(SCENARIOS),
                "rehearsal_max_minutes": 90,
                "overnight_max_minutes": 480,
                "organ_time_budget_equal": True,
                "n1_training_seeds": [31, 47, 73],
                "teacher_modes": ["post_experience", "tier_3_bounded"],
                "mamba2_role": "experimental_temporal_alternative",
            },
        )
    artifact_root = root / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    return RuntimeContext(
        state=state,
        postgres=postgres,
        storage_config=StorageConfig(
            mode="postgres",
            sqlite_db_path=":unused:",
            postgres_dsn=postgres.dsn,
            artifact_root=artifact_root,
            prefer_postgres_reads=True,
            strict_dual_write=False,
        ),
        artifact_root=artifact_root,
        env_file=env_file,
    )


def _compose_env_file(env_file: Path) -> Path:
    candidate = env_file.parent / "infra/docker/.env"
    if candidate.is_file():
        return candidate
    local = REPO_ROOT / "infra/docker/.env"
    if local.is_file():
        return local
    raise CampaignError("campaign_postgres_compose_env_missing")


def _docker_executable() -> str | None:
    for candidate in (shutil.which("docker.exe"), shutil.which("docker")):
        if not candidate:
            continue
        result = subprocess.run(
            [candidate, "version", "--format", "{{.Server.Version}}"],
            check=False,
            capture_output=True,
        )
        if result.returncode == 0:
            return candidate
    return None


def _docker_path(path: Path, docker: str) -> str:
    if not docker.lower().endswith(".exe"):
        return str(path)
    converted = subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()
    return converted


def _docker_has_image(docker: str, image: str) -> bool:
    result = subprocess.run(
        [docker, "image", "inspect", image],
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def _ensure_postgres(ctx: RuntimeContext, *, allow_existing: bool) -> dict[str, Any]:
    try:
        base_probe = ctx.postgres.probe()
    except Exception as initial_error:
        compose = REPO_ROOT / "infra/docker/docker-compose.yml"
        docker = _docker_executable()
        if not docker:
            raise CampaignError(
                "campaign_postgres_unreachable_and_docker_unavailable_in_wsl"
            ) from initial_error
        compose_files = ["-f", _docker_path(compose, docker)]
        if not _docker_has_image(docker, "postgres:16-alpine") and _docker_has_image(
            docker, "postgres:16"
        ):
            override = ctx.state.root / "postgres/docker-compose.image-override.yml"
            override.parent.mkdir(parents=True, exist_ok=True)
            override.write_text(
                "services:\n  postgres:\n    image: postgres:16\n", encoding="utf-8"
            )
            compose_files.extend(("-f", _docker_path(override, docker)))
        command = [
            docker,
            "compose",
            "--env-file",
            _docker_path(_compose_env_file(ctx.env_file), docker),
            *compose_files,
            "up",
            "-d",
            "postgres",
        ]
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = ((result.stdout or "") + (result.stderr or "")).strip()
            raise CampaignError(
                f"campaign_postgres_start_failed:{type(initial_error).__name__}:"
                f"{detail[-800:]}"
            ) from initial_error
        deadline = time.monotonic() + 120.0
        while True:
            try:
                base_probe = ctx.postgres.probe()
                break
            except Exception as error:
                if time.monotonic() >= deadline:
                    raise CampaignError(
                        f"campaign_postgres_health_timeout:{type(error).__name__}"
                    ) from error
                time.sleep(2.0)
    campaign_probe = ctx.postgres.ensure(allow_existing=allow_existing)
    marker_storage = ctx.postgres.storage(artifact_root=ctx.artifact_root)
    marker_id = f"campaign-marker::{ctx.state.campaign_id}"
    markers = marker_storage.list_events(
        run_id=ctx.state.campaign_id,
        event_types=("neural.campaign.created",),
        limit=20,
    )
    matching = next((event for event in markers if event.event_id == marker_id), None)
    expected_marker = {
        "campaign_id": ctx.state.campaign_id,
        "database": ctx.postgres.database,
        "commit": ctx.state.manifest["commit"],
        "schema_sha256": ctx.postgres.schema_sha256,
        "official_storage": "postgres",
    }
    if matching is None:
        marker_storage.append_event(
            event_id=marker_id,
            event_type="neural.campaign.created",
            run_id=ctx.state.campaign_id,
            source="integral_neural_campaign",
            payload=expected_marker,
        )
    elif dict(matching.payload) != expected_marker:
        marker_storage.close()
        raise CampaignError("campaign_postgres_ownership_marker_mismatch")
    marker_storage.close()
    campaign_probe["ownership_marker"] = marker_id
    return {"base": base_probe, "campaign": campaign_probe, "passed": True}


def _execute_block(
    ctx: RuntimeContext,
    name: str,
    function: Callable[[], Mapping[str, Any]],
) -> dict[str, Any]:
    current = ctx.state.manifest["blocks"][name]
    if current["status"] == "completed":
        return dict(current.get("result") or {})
    ctx.state.begin(name)
    try:
        result = dict(function())
        ctx.state.complete(name, result)
        if ctx.storage is not None:
            ctx.storage.append_event(
                event_type="neural.campaign.block.completed",
                run_id=ctx.state.campaign_id,
                source="integral_neural_campaign",
                payload={
                    "campaign_id": ctx.state.campaign_id,
                    "block": name,
                    "passed": bool(result.get("passed", True)),
                    "manifest_sha256": file_sha256(ctx.state.manifest_path),
                    "authority_effect": "none",
                },
            )
        return result
    except Exception as error:
        ctx.state.fail(name, error)
        if ctx.storage is not None:
            try:
                ctx.storage.append_event(
                    event_type="neural.campaign.block.failed",
                    run_id=ctx.state.campaign_id,
                    source="integral_neural_campaign",
                    payload={
                        "campaign_id": ctx.state.campaign_id,
                        "block": name,
                        "error_type": type(error).__name__,
                        "authority_effect": "none",
                    },
                )
            except Exception:
                pass
        raise


def _environment_report() -> dict[str, Any]:
    status = _git("status", "--porcelain")
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,temperature.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "passed": not status,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_clean": not status,
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "wsl_distro": os.environ.get("WSL_DISTRO_NAME", "unknown"),
        "gpu_available": gpu.returncode == 0,
        "gpu_summary": gpu.stdout.strip().splitlines() if gpu.returncode == 0 else [],
        "pg_dump_available": shutil.which("pg_dump") is not None,
        "docker_available": _docker_executable() is not None,
        "authority_effect": "none",
    }


def _connectome_report() -> dict[str, Any]:
    topology = canonical_connectome()
    forbidden = [
        edge.edge_id
        for edge in topology.edges
        if edge.source in OFFICIAL_ORGANS and edge.authority_ceiling.value == "authoritative"
    ]
    return {
        "passed": len(topology.nodes) == 22 and len(topology.edges) == 38 and not forbidden,
        "nodes": len(topology.nodes),
        "edges": len(topology.edges),
        "topology_hash": topology.topology_hash,
        "forbidden_authoritative_edges": forbidden,
        "matrix": [
            {
                "edge_id": edge.edge_id,
                "source": edge.source,
                "target": edge.target,
                "status": "available",
                "authority_ceiling": edge.authority_ceiling.value,
            }
            for edge in topology.edges
        ],
        "authority_effect": "none",
    }


def _configured_neural_root() -> Path:
    candidates = [
        Path(os.environ.get("RNFE_ARTIFACT_ROOT", "rnfe_artifacts")) / "neural",
        REPO_ROOT / "rnfe_artifacts/neural",
    ]
    return next((path.resolve() for path in candidates if path.is_dir()), candidates[-1].resolve())


def _artifact_report() -> dict[str, Any]:
    root = _configured_neural_root()
    rows = []
    for organ in ("N1", "N3", "N4", "N5"):
        manifest_path = root / organ.lower() / "manifest.json"
        row: dict[str, Any] = {"organ": organ, "manifest_present": manifest_path.is_file()}
        if manifest_path.is_file():
            manifest = NeuralModelManifest.from_dict(json.loads(manifest_path.read_text()))
            artifact = root / manifest.artifact_path
            row.update(
                {
                    "model_id": manifest.model_id,
                    "backend": manifest.backend,
                    "artifact_present": artifact.is_file(),
                    "hash_valid": artifact.is_file()
                    and file_sha256(artifact) == manifest.artifact_sha256,
                    "promotion_eligible": bool(
                        manifest.training_provenance.get("promotion_eligible")
                    ),
                }
            )
        rows.append(row)
    return {
        "passed": all(
            row.get("hash_valid", False) for row in rows if row["organ"] in {"N1", "N3", "N4"}
        ),
        "neural_root": str(root),
        "models": rows,
        "gguf_7b": {
            "path": str(GGUF_7B),
            "present": GGUF_7B.is_file(),
            "size_bytes": GGUF_7B.stat().st_size if GGUF_7B.is_file() else 0,
            "sha256": file_sha256(GGUF_7B) if GGUF_7B.is_file() else None,
        },
        "runtime_downloads": False,
        "authority_effect": "none",
    }


def _run_command(
    *, ctx: RuntimeContext, name: str, command: Sequence[str], env: Mapping[str, str] | None = None
) -> dict[str, Any]:
    started = time.monotonic()
    process = subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        env=dict(env) if env is not None else None,
        check=False,
        capture_output=True,
        text=True,
    )
    output = (process.stdout or "") + (process.stderr or "")
    for secret in (ctx.postgres.base_dsn, ctx.postgres.dsn):
        output = output.replace(secret, "<redacted-postgres-dsn>")
    log_path = ctx.state.root / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output, encoding="utf-8")
    return {
        "command": list(command),
        "returncode": process.returncode,
        "passed": process.returncode == 0,
        "elapsed_s": round(time.monotonic() - started, 3),
        "log_path": str(log_path),
        "log_sha256": file_sha256(log_path),
    }


def _regression(ctx: RuntimeContext, *, skip: bool) -> dict[str, Any]:
    if skip:
        return {"passed": False, "skipped": True, "staging_blocked": True}
    shard_path = ctx.state.root / "regression_shards.json"
    commit = _git("rev-parse", "HEAD")
    shard_state = (
        json.loads(shard_path.read_text())
        if shard_path.is_file()
        else {"schema_version": "rnfe-regression-shards-v1", "commit": commit, "shards": {}}
    )
    if shard_state.get("commit") != commit:
        shard_state = {
            "schema_version": "rnfe-regression-shards-v1",
            "commit": commit,
            "shards": {},
        }
    test_files = sorted(REPO_ROOT.joinpath("tests").rglob("test_*.py"))
    for test_file in test_files:
        relative = str(test_file.relative_to(REPO_ROOT))
        previous = shard_state["shards"].get(relative) or {}
        if previous.get("passed") is True:
            continue
        result = _run_command(
            ctx=ctx,
            name="pytest-" + relative.replace("/", "-").removesuffix(".py"),
            command=[sys.executable, "-m", "pytest", "-q", relative],
        )
        shard_state["shards"][relative] = result
        atomic_write_json(shard_path, shard_state)
        if not result["passed"]:
            break
    base_passed = len(shard_state["shards"]) == len(test_files) and all(
        result.get("passed") for result in shard_state["shards"].values()
    )
    base = {
        "passed": base_passed,
        "test_file_shards": len(test_files),
        "completed_shards": sum(
            bool(result.get("passed")) for result in shard_state["shards"].values()
        ),
        "state_path": str(shard_path),
        "state_sha256": file_sha256(shard_path),
    }
    pg_env = os.environ.copy()
    pg_env.update(
        {
            "RNFE_RUN_PG_TESTS": "1",
            "RNFE_STORAGE_MODE": "postgres",
            "RNFE_POSTGRES_DSN": ctx.postgres.dsn,
            "RNFE_ARTIFACT_ROOT": str(ctx.artifact_root / "pg-tests"),
        }
    )
    postgres = (
        _run_command(
            ctx=ctx,
            name="pytest-postgres",
            command=[sys.executable, "-m", "pytest", "-q", "-m", "requires_postgres"],
            env=pg_env,
        )
        if base_passed
        else {"passed": False, "skipped": True, "reason": "base_shard_failed"}
    )
    _register_report(
        ctx,
        shard_path,
        kind="regression_shard_state",
        run_id=ctx.state.campaign_id,
    )
    return {
        "passed": base["passed"] and postgres["passed"],
        "base": base,
        "postgres": postgres,
        "postgres_tests_expected": 4,
        "staging_blocked": not (base["passed"] and postgres["passed"]),
    }


def _register_report(ctx: RuntimeContext, path: Path, *, kind: str, run_id: str) -> None:
    ctx.ensure_storage().register_artifact(
        kind=kind,
        abs_path=path,
        run_id=run_id,
        metadata={
            "campaign_id": ctx.state.campaign_id,
            "authority_effect": "none",
            "official_storage": "postgres",
        },
    )


def _write_organ_report(ctx: RuntimeContext, organ: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    report = {
        "schema_version": "rnfe-integral-organ-qualification-v1",
        "campaign_id": ctx.state.campaign_id,
        "organ": organ,
        "equal_budget": True,
        "experimental": True,
        "authority_effect": "none",
        "promotion_eligible": False,
        "safety_violations": int(payload.get("safety_violations", 0)),
        **dict(payload),
    }
    path = ctx.state.root / "organs" / f"{organ.lower()}.json"
    atomic_write_json(path, report)
    _register_report(ctx, path, kind="neural_organ_qualification", run_id=ctx.state.campaign_id)
    return report


def _qualify_n0(ctx: RuntimeContext) -> dict[str, Any]:
    storage = ctx.ensure_storage()
    event_id = f"{ctx.state.campaign_id}-n0-storage-probe"
    storage.append_event(
        event_id=event_id,
        event_type="neural.campaign.n0.storage_probe",
        run_id=ctx.state.campaign_id,
        source="integral_neural_campaign",
        payload={"postgres_direct": True, "fallback_used": False},
    )
    durable = any(event.event_id == event_id for event in storage.list_events(run_id=ctx.state.campaign_id))
    return _write_organ_report(
        ctx,
        "N0",
        {
            "passed": durable,
            "postgres_durable": durable,
            "fallback_used": False,
            "msrc_enabled": True,
            "host_sensing_required": True,
            "continuity_checkpointing": True,
        },
    )


def _prepare_evaluation_artifacts(ctx: RuntimeContext) -> Path:
    target_base = ctx.state.root / "evaluation_artifacts"
    target = target_base / "neural"
    source = _configured_neural_root()
    target.mkdir(parents=True, exist_ok=True)
    for organ in ("n3", "n4", "n5"):
        source_dir = source / organ
        if source_dir.is_dir() and not (target / organ).exists():
            shutil.copytree(source_dir, target / organ)
    n1_source = ctx.state.root / "n1_recalibration/candidate/n1"
    if n1_source.is_dir():
        shutil.rmtree(target / "n1", ignore_errors=True)
        shutil.copytree(n1_source, target / "n1")
    return target_base


def _qualify_n1(ctx: RuntimeContext, *, epochs: int) -> dict[str, Any]:
    source = REPO_ROOT / "data/reports/neural/n1-counterfactual-native-v2/paired_records.jsonl"
    root = ctx.state.root / "n1_recalibration"
    report = n1_recalibrate_candidates(
        paired_records_path=source,
        output_root=root,
        seeds=(31, 47, 73),
        epochs=epochs,
    )
    holdout = seal_holdout_spec(
        target=ctx.state.root / "sealed_holdout.json",
        campaign_id=ctx.state.campaign_id,
        seed_base=931_000,
        contexts_per_generator=20,
        scenarios=("thermal_homeostasis", "resource_management", "grid_thermal_5x5"),
    )
    _prepare_evaluation_artifacts(ctx)
    selected = next(
        item for item in report["candidates"] if item["seed"] == report["selected_seed"]
    )
    payload = {
        "passed": True,
        "development_only": True,
        "three_seed_models": len(report["seeds"]) >= 3,
        "selected_seed": report["selected_seed"],
        "validation_ece": selected["validation"]["calibration_ece"],
        "temperature": selected["temperature"],
        "holdout_spec_hash": holdout["spec_hash"],
        "holdout_opened": False,
        "promotion_eligible": False,
        "training_authorized": False,
    }
    return _write_organ_report(ctx, "N1", payload)


def _qualify_n2(ctx: RuntimeContext) -> dict[str, Any]:
    topology = canonical_connectome()
    edges = [edge.edge_id for edge in topology.edges if edge.source == "N2" or edge.target == "N2"]
    return _write_organ_report(
        ctx,
        "N2",
        {
            "passed": bool(edges),
            "deterministic_reference": True,
            "symbolic_verifiers": ["DED", "LOT-F", "NESY"],
            "connectome_edges": edges,
            "learned_model": False,
        },
    )


def _manifest_for(organ: str, root: Path | None = None) -> tuple[NeuralModelManifest, Path]:
    neural_root = root or _configured_neural_root()
    manifest_path = neural_root / organ.lower() / "manifest.json"
    manifest = NeuralModelManifest.from_dict(json.loads(manifest_path.read_text()))
    artifact = neural_root / manifest.artifact_path
    if not artifact.is_file() or file_sha256(artifact) != manifest.artifact_sha256:
        raise CampaignError(f"campaign_model_integrity_failed:{organ}")
    return manifest, artifact


def _qualify_n3(ctx: RuntimeContext) -> dict[str, Any]:
    try:
        manifest, artifact = _manifest_for("N3")
        ready = True
        details = {"model_id": manifest.model_id, "artifact_sha256": file_sha256(artifact)}
    except Exception as error:
        ready = False
        details = {"error": type(error).__name__}
    return _write_organ_report(
        ctx,
        "N3",
        {
            "passed": ready,
            "reference_vs_mamba_required": True,
            "mamba2_role": "experimental_temporal_alternative",
            "communication_fast_path": False,
            **details,
        },
    )


def _qualify_n4(ctx: RuntimeContext) -> dict[str, Any]:
    staged_manifest, staged_artifact = _manifest_for("N4")
    reference_root = ctx.state.root / "n4_contract_reference"
    reference_artifact = reference_root / "n4/reference.json"
    reference_payload = {
        "artifact_schema_version": N4_ARTIFACT_SCHEMA_VERSION,
        "model_kind": "reference",
        "trained": False,
        "frozen": True,
        "experimental": True,
        "input_size": 2,
        "hidden_size": 2,
        "message_passing_steps": 2,
        "max_nodes": 32,
        "max_edges": 64,
        "input_weight": [[1.0, 0.0], [0.0, 1.0]],
        "message_weight": [[0.5, 0.0], [0.0, 0.5]],
        "update_weight": [[0.8, 0.0], [0.0, 0.8]],
        "output_weight": [[0.2, 0.1], [0.1, 0.1], [0.0, 0.0], [0.0, 0.0]],
        "supported_node_types": [
            "world_variable",
            "observation",
            "intervention",
            "sign",
            "evidence",
            "memory",
            "goal",
            "constraint",
        ],
        "supported_edge_types": [
            "causal_positive",
            "causal_negative",
            "temporal",
            "support",
            "contradiction",
            "counterfactual",
            "semantic",
            "morphism",
        ],
    }
    atomic_write_json(reference_artifact, reference_payload)
    manifest = NeuralModelManifest(
        organ="N4",
        capability="causal_prediction",
        model_id="n4-campaign-contract-reference",
        version="1.0.0",
        backend="n4-reference",
        artifact_path="n4/reference.json",
        artifact_sha256=file_sha256(reference_artifact),
        input_schema_version=N4_GRAPH_SCHEMA_VERSION,
        output_schema_version=N4_OUTPUT_SCHEMA_VERSION,
        supported_devices=("cpu",),
        parameter_count=32,
        peak_vram_gb=0.0,
        license_id="Unlicense",
        upstream_url="repo://rnfe/runtime/neural/organs/n4_causal.py",
        upstream_commit="reference-contract-v1",
        training_provenance={"classification": "reference", "trained": False},
    )
    backend = CausalMessagePassingBackend()
    backend.load(manifest, str(reference_artifact), "cpu")
    try:
        benchmark = run_n4_synthetic_benchmark(backend)
    finally:
        backend.unload()
    contract_passed = all(bool(value) for value in benchmark["a_m0"].values())
    return _write_organ_report(
        ctx,
        "N4",
        {
            "passed": contract_passed,
            "contract_benchmark": benchmark,
            "staged_model_id": staged_manifest.model_id,
            "staged_artifact_sha256": file_sha256(staged_artifact),
            "contract_reference_sha256": file_sha256(reference_artifact),
            "causal_generalization": "not_evaluated",
            "committed_action_binding_required": True,
            "promotion_eligible": False,
        },
    )


def _qualify_n5(ctx: RuntimeContext) -> dict[str, Any]:
    text = "Árbol neural.\n\nRazón, memoria y acción segura. 𐍈"
    chunks = DeterministicChunker(max_bytes=32).chunk(text)
    reassembled = "".join(chunk.text for chunk in chunks)
    manifest_present = (_configured_neural_root() / "n5/manifest.json").is_file()
    return _write_organ_report(
        ctx,
        "N5",
        {
            "passed": reassembled == text,
            "utf8_roundtrip": reassembled == text,
            "chunks": len(chunks),
            "hnet_manifest_present": manifest_present,
            "semantic_corpus_ready": False,
            "promotion_eligible": False,
        },
    )


def _qualify_n6(ctx: RuntimeContext) -> dict[str, Any]:
    proposal = StructuralMutationProposal(
        mutation_type="optional_family_budget",
        target="N1.max_optional_families",
        value=2,
        expected_gain=0.1,
        rollback_token="campaign-n6-no-apply",
    )
    result = StructuralEvolutionGate().evaluate_and_apply(
        proposal,
        sandbox=lambda _: {"passed": True},
        certify=lambda _: True,
        apply_fn=None,
        rollback=None,
    )
    return _write_organ_report(
        ctx,
        "N6",
        {
            "passed": result.get("applied") is False,
            "sandbox_result": result,
            "mutation_applied": False,
            "rollback_required_for_authority": True,
        },
    )


def _model_environment(ctx: RuntimeContext) -> dict[str, str | None]:
    base = _prepare_evaluation_artifacts(ctx)
    neural = base / "neural"
    values: dict[str, str | None] = {
        "RNFE_ARTIFACT_ROOT": str(base),
        "RNFE_NEURAL_N1_MANIFEST": "n1/manifest.json" if (neural / "n1/manifest.json").is_file() else None,
        "RNFE_NEURAL_N3_MANIFEST": "n3/manifest.json" if (neural / "n3/manifest.json").is_file() else None,
        "RNFE_NEURAL_N4_MANIFEST": "n4/manifest.json" if (neural / "n4/manifest.json").is_file() else None,
        "RNFE_NEURAL_N5_MANIFEST": "n5/manifest.json" if (neural / "n5/manifest.json").is_file() else None,
        "RNFE_HOST_SENSING": "1",
    }
    return values


def _step_vector(rows: Sequence[Mapping[str, Any]], *, elapsed_s: float) -> tuple[float, OrganismImpactVector]:
    vitals = [dict(row["vital_signs"]) for row in rows]
    episodes = [dict(row.get("episode_result") or {}) for row in rows]
    primary = fmean(float(item.get("cognitive_quality", 0.0)) for item in vitals)
    certified = [bool(item.get("certified", False)) for item in vitals]
    closure = [
        str(episode.get("certification", {}).get("verdict", "")).lower()
        in {"certified", "pass", "passed"}
        for episode in episodes
    ]
    resource_states = [
        dict(episode.get("neural_symbiosis_trace", {}).get("resource_state") or {})
        for episode in episodes
    ]
    def mean_resource(name: str, default: float = 0.0) -> float:
        values = [float(item.get(name, default) or default) for item in resource_states]
        return fmean(values) if values else default

    violations = 0
    for episode in episodes:
        trace = dict(episode.get("neural_symbiosis_trace") or {})
        for organ in trace.get("organs") or ():
            if str(organ.get("authority_effect", "none")) not in {"none", "evidence_only"}:
                violations += 1
            candidate = organ.get("candidate") or {}
            if isinstance(candidate, Mapping) and bool(candidate.get("applied")):
                violations += 1
    return primary, OrganismImpactVector(
        closure_rate=fmean(float(value) for value in closure),
        certification_rate=fmean(float(value) for value in certified),
        continuity=fmean(float(item.get("identity_continuity", 0.0)) for item in vitals),
        viability=fmean(float(item.get("viability_margin", 0.0)) for item in vitals),
        latency_ms=elapsed_s * 1000.0 / max(len(rows), 1),
        cpu_pressure=mean_resource("cpu_pressure"),
        memory_pressure=mean_resource("memory_pressure"),
        vram_gb=mean_resource("vram_used_gb"),
        thermal_pressure=mean_resource("thermal_pressure"),
        safety_violations=violations,
    )


def _run_life_lane(
    ctx: RuntimeContext,
    *,
    phase: str,
    lane: str,
    seed: int,
    steps: int,
    tier3: bool = False,
) -> dict[str, Any]:
    run_id = f"{ctx.state.campaign_id}-{phase}-{lane}-seed-{seed}"
    values = {
        **_model_environment(ctx),
        "RNFE_NEURAL_MODE": "shadow" if lane != "off" else "off",
        "RNFE_STORAGE_MODE": "postgres",
        "RNFE_POSTGRES_DSN": ctx.postgres.dsn,
        "RNFE_ALLOW_EXTERNAL_REASONER": "1" if tier3 else "0",
        "RNFE_MAX_COMPUTE_TIER": "tier_3_external" if tier3 else "tier_2_specialized",
    }
    started = time.monotonic()
    with temporary_environ(values):
        kernel = LifeKernel(
            config=LifeKernelConfig(
                run_id=run_id,
                organism_id=f"organism-{phase}-{lane}-{seed}",
                scenarios=SCENARIOS,
                block_size=max(1, steps // len(SCENARIOS)),
                restore=False,
                checkpoint_interval=1,
                allow_external_reasoner=tier3,
                max_compute_tier="tier_3_external" if tier3 else "tier_2_specialized",
                enable_msrc=True,
            ),
            storage=ctx.ensure_storage(),
        )
        rows = [
            kernel.step(external_input=0.04 + ((seed + index * 7) % 11) / 100.0).to_dict()
            for index in range(steps)
        ]
    elapsed = time.monotonic() - started
    primary, vector = _step_vector(rows, elapsed_s=elapsed)
    path = ctx.state.root / "life_kernel" / phase / f"{lane}-seed-{seed}.json"
    atomic_write_json(
        path,
        {
            "schema_version": "rnfe-integral-life-lane-v1",
            "campaign_id": ctx.state.campaign_id,
            "phase": phase,
            "lane": lane,
            "seed": seed,
            "tier3": tier3,
            "steps": rows,
            "primary_metric": primary,
            "impact_vector": vector.to_dict(),
            "elapsed_s": elapsed,
            "authority_effect": "none",
        },
    )
    _register_report(ctx, path, kind="neural_life_lane", run_id=run_id)
    return {
        "run_id": run_id,
        "path": str(path),
        "sha256": file_sha256(path),
        "primary": primary,
        "vector": vector,
        "rows": rows,
    }


def _organ_runtime_summary(rows: Iterable[Mapping[str, Any]], organ: str) -> dict[str, Any]:
    entries = []
    receipts = []
    agent_reports = []
    for row in rows:
        episode = dict(row.get("episode_result") or {})
        trace = dict(episode.get("neural_symbiosis_trace") or {})
        symbiosis = dict(episode.get("neural_symbiosis") or {})
        entries.extend(item for item in trace.get("organs") or () if item.get("organ") == organ)
        receipts.extend(item for item in trace.get("consumer_receipts") or () if item.get("organ") == organ)
        cycle = symbiosis.get("neural_agents") or {}
        agent_reports.extend(cycle.get("reports") or ())
    return {
        "observations": len(entries),
        "candidates": sum(bool(item.get("candidate_hash")) for item in entries),
        "consumer_receipts": len(receipts),
        "fallbacks": sum(bool(item.get("fallback_used")) for item in entries),
        "agent_reports": len(agent_reports),
        "authority_effect": "none",
        "safety_violations": sum(
            str(item.get("authority_effect", "none")) not in {"none", "evidence_only"}
            for item in entries
        ),
    }


def _agent_runtime_report(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    core_reports = []
    extension_reports = []
    for row in rows:
        episode = dict(row.get("episode_result") or {})
        symbiosis = dict(episode.get("neural_symbiosis") or {})
        trace = dict(episode.get("neural_symbiosis_trace") or {})
        core_reports.extend((symbiosis.get("neural_agents") or {}).get("reports") or ())
        extension_reports.extend(
            (symbiosis.get("neural_agent_extensions") or {}).get("reports") or ()
        )
        extension_reports.extend(
            (trace.get("neural_agent_extensions") or {}).get("reports") or ()
        )
    core_roles = sorted({str(report.get("role")) for report in core_reports})
    extension_roles = sorted({str(report.get("role")) for report in extension_reports})
    authority_violations = [
        str(report.get("role"))
        for report in (*core_reports, *extension_reports)
        if str(report.get("authority_effect", "none"))
        not in {"none", "evidence_only"}
    ]
    return {
        "schema_version": "rnfe-integral-agent-runtime-report-v1",
        "core_roles": core_roles,
        "extension_roles": extension_roles,
        "core_role_count": len(core_roles),
        "extension_role_count": len(extension_roles),
        "reports_observed": len(core_reports) + len(extension_reports),
        "authority_violations": authority_violations,
        "passed": len(core_roles) == 5
        and len(extension_roles) == 11
        and not authority_violations,
        "authority_effect": "none",
    }


def _paired_life(ctx: RuntimeContext, *, phase: str, steps: int) -> dict[str, Any]:
    observations = []
    lanes = []
    shadow_rows: list[Mapping[str, Any]] = []
    for seed in (811001, 811101, 811201):
        baseline = _run_life_lane(ctx, phase=phase, lane="off", seed=seed, steps=steps)
        candidate = _run_life_lane(ctx, phase=phase, lane="shadow", seed=seed, steps=steps)
        lanes.extend((baseline, candidate))
        shadow_rows.extend(candidate["rows"])
        observations.append(
            ImpactObservation(
                seed=seed,
                baseline_primary=baseline["primary"],
                candidate_primary=candidate["primary"],
                baseline=baseline["vector"],
                candidate=candidate["vector"],
            )
        )
    n1_report = json.loads((ctx.state.root / "organs/n1.json").read_text())
    holdout_path = ctx.state.root / "fresh_holdout/holdout_evaluation.json"
    if phase == "overnight" and holdout_path.is_file():
        ece = float(json.loads(holdout_path.read_text()).get("calibration_ece", 1.0))
    else:
        ece = float(n1_report.get("validation_ece", 1.0))
    impact = build_impact_report(
        organ="N1",
        model_id="rnfe-n1-router-campaign-candidate",
        observations=observations,
        ece=ece,
        bootstrap_seed=20260715,
    )
    impact_path = ctx.state.root / "impact" / f"{phase}.json"
    atomic_write_json(impact_path, impact.to_dict())
    _register_report(ctx, impact_path, kind="organism_impact_report", run_id=ctx.state.campaign_id)
    runtime_organs = {
        organ: _organ_runtime_summary(shadow_rows, organ)
        for organ in ("N1", "N2", "N3", "N4", "N5", "N6")
    }
    runtime_path = ctx.state.root / "organs" / f"runtime-{phase}.json"
    atomic_write_json(runtime_path, {"organs": runtime_organs})
    _register_report(ctx, runtime_path, kind="neural_runtime_qualification", run_id=ctx.state.campaign_id)
    agent_report = _agent_runtime_report(shadow_rows)
    agent_path = ctx.state.root / "agents" / f"{phase}.json"
    atomic_write_json(agent_path, agent_report)
    _register_report(ctx, agent_path, kind="neural_agent_qualification", run_id=ctx.state.campaign_id)
    return {
        "passed": agent_report["passed"]
        and all(summary["safety_violations"] == 0 for summary in runtime_organs.values()),
        "phase": phase,
        "steps_per_lane": steps,
        "pairs": len(observations),
        "lane_reports": [
            {key: value for key, value in lane.items() if key not in {"rows", "vector"}}
            for lane in lanes
        ],
        "impact_report": impact.to_dict(),
        "runtime_organs": runtime_organs,
        "agents": agent_report,
    }


def _teacher(ctx: RuntimeContext, *, phase: str, timeout_s: float) -> dict[str, Any]:
    output_root = ctx.state.root / "teacher_7b"
    campaign_id = f"{ctx.state.campaign_id}-{phase}"
    target = output_root / campaign_id
    shutil.rmtree(target, ignore_errors=True)
    seeds = (42, 101, 202) if phase == "rehearsal" else (307, 401, 503, 601, 701)
    horizon = 3 if phase == "rehearsal" else 6
    try:
        result_dir = run_teacher_campaign(
            campaign_id=campaign_id,
            output_root=output_root,
            scenarios=(
                "thermal_homeostasis",
                "resource_management",
                "deferred_load_trap",
            ),
            seeds=seeds,
            max_tokens=160,
            timeout_s=timeout_s,
            temperature=0.25,
            horizon=horizon,
            profile="pilot" if phase == "rehearsal" else "heldout_v1",
            storage_config=ctx.storage_config,
        )
        verdict = json.loads((result_dir / "verdict.json").read_text())
        teacher_result = {
            "completed": True,
            "path": str(result_dir),
            "verdict": verdict,
        }
        for name in ("manifest.json", "summary.json", "verdict.json", "REPORT.md"):
            _register_report(
                ctx,
                result_dir / name,
                kind="teacher_7b_evidence",
                run_id=ctx.state.campaign_id,
            )
    except Exception as error:
        teacher_result = {
            "completed": False,
            "error_type": type(error).__name__,
            "quarantined": True,
        }
    tier3 = _run_life_lane(
        ctx,
        phase=phase,
        lane="tier3-bounded",
        seed=919,
        steps=1 if phase == "rehearsal" else 3,
        tier3=True,
    )
    return {
        "passed": bool(teacher_result.get("completed")),
        "teacher": teacher_result,
        "tier3": {key: value for key, value in tier3.items() if key not in {"rows", "vector"}},
        "training_authorized": False,
        "promotion_authorized": False,
        "role": "supervised_student_and_bounded_proposer",
    }


def _sqlite_contingency(ctx: RuntimeContext) -> dict[str, Any]:
    root = ctx.state.root / "sqlite_contingency"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    sqlite_path = root / "contingency.db"
    storage = StorageFactory.create_facade(
        StorageConfig(
            mode="sqlite",
            sqlite_db_path=str(sqlite_path),
            postgres_dsn=None,
            artifact_root=root / "artifacts",
            prefer_postgres_reads=False,
            strict_dual_write=False,
        )
    )
    run_id = f"{ctx.state.campaign_id}-sqlite-contingency"
    storage.append_event(
        event_id=f"{run_id}-event",
        event_type="neural.campaign.sqlite.contingency",
        run_id=run_id,
        source="integral_neural_campaign",
        payload={"official_evidence": False, "fallback_test": True},
    )
    storage.close()
    first = migrate_sqlite_ledger_to_postgres(
        sqlite_db_path=str(sqlite_path), postgres_dsn=ctx.postgres.dsn
    )
    second = migrate_sqlite_ledger_to_postgres(
        sqlite_db_path=str(sqlite_path), postgres_dsn=ctx.postgres.dsn
    )
    migrated = ctx.ensure_storage().list_events(run_id=run_id, limit=20)
    report = {
        "schema_version": "rnfe-sqlite-contingency-report-v1",
        "passed": len(migrated) == 1 and first == second == 1,
        "official_evidence": False,
        "sqlite_rows": 1,
        "first_migration_processed": first,
        "second_migration_processed": second,
        "postgres_unique_rows": len(migrated),
        "idempotent": len(migrated) == 1,
    }
    path = root / "report.json"
    atomic_write_json(path, report)
    _register_report(ctx, path, kind="sqlite_contingency", run_id=ctx.state.campaign_id)
    return report


def _reconcile(ctx: RuntimeContext, *, phase: str) -> dict[str, Any]:
    report = reconcile_artifact_plane(
        storage=ctx.ensure_storage(),
        run_id=ctx.state.campaign_id,
        artifact_root=ctx.artifact_root,
    )
    path = ctx.state.root / "reconciliation" / f"{phase}.json"
    atomic_write_json(path, report)
    # The reconciliation report itself is outside the indexed artifact root, so
    # registering it cannot make the just-computed filesystem set stale.
    _register_report(ctx, path, kind="artifact_reconciliation", run_id=ctx.state.campaign_id)
    return report


def _dump(ctx: RuntimeContext, *, phase: str) -> dict[str, Any]:
    report = ctx.postgres.dump(ctx.state.root / "postgres" / f"{phase}.dump")
    path = ctx.state.root / "postgres" / f"{phase}.dump.json"
    atomic_write_json(path, report)
    _register_report(ctx, path, kind="postgres_dump_manifest", run_id=ctx.state.campaign_id)
    return {"passed": True, **report}


def _open_holdout(ctx: RuntimeContext, *, contexts: int) -> dict[str, Any]:
    spec_path = ctx.state.root / "sealed_holdout.json"
    spec = json.loads(spec_path.read_text())
    access_path = ctx.state.root / "holdout_access.json"
    if access_path.exists():
        raise CampaignError("campaign_holdout_already_opened")
    original_hash = spec["spec_hash"]
    atomic_write_json(
        access_path,
        {
            "schema_version": "rnfe-n1-holdout-single-access-v1",
            "campaign_id": ctx.state.campaign_id,
            "spec_hash": original_hash,
            "opened_at": time.time(),
            "single_access": True,
        },
    )
    ctx.ensure_storage().append_event(
        event_type="neural.campaign.holdout.opened",
        run_id=ctx.state.campaign_id,
        source="integral_neural_campaign",
        payload={
            "campaign_id": ctx.state.campaign_id,
            "spec_hash": original_hash,
            "single_access": True,
            "authority_effect": "none",
        },
    )
    output = ctx.state.root / "fresh_holdout"
    shutil.rmtree(output, ignore_errors=True)
    generated = run_n1_counterfactual_campaign(
        output_dir=output,
        contexts_per_generator=contexts,
        seed_base=int(spec["seed_base"]),
        train=False,
    )
    artifact = ctx.state.root / "n1_recalibration/candidate/n1/router-lab-v1.json"
    evaluation = evaluate_n1_artifact(
        artifact_path=artifact,
        samples_path=output / "counterfactual_samples.jsonl",
    )
    report = {
        "schema_version": "rnfe-n1-sovereign-holdout-result-v1",
        "passed": evaluation["calibration_ece"] <= 0.10,
        "spec_hash": original_hash,
        "single_access": True,
        "generator_result": generated,
        **evaluation,
    }
    path = output / "holdout_evaluation.json"
    atomic_write_json(path, report)
    _register_report(ctx, path, kind="n1_fresh_holdout", run_id=ctx.state.campaign_id)
    return report


def _load_organ_reports(ctx: RuntimeContext) -> dict[str, dict[str, Any]]:
    reports = {}
    for organ in OFFICIAL_ORGANS:
        path = ctx.state.root / "organs" / f"{organ.lower()}.json"
        if path.is_file():
            reports[organ] = json.loads(path.read_text())
    runtime_path = ctx.state.root / "organs/runtime-overnight.json"
    if runtime_path.is_file():
        runtime = json.loads(runtime_path.read_text()).get("organs", {})
        for organ, values in runtime.items():
            reports.setdefault(organ, {}).update({"runtime": values})
            reports[organ]["safety_violations"] = int(values.get("safety_violations", 0))
    return reports


def _build_report(ctx: RuntimeContext) -> dict[str, Any]:
    regression = ctx.state.manifest["blocks"]["regression_full"].get("result") or {}
    holdout_path = ctx.state.root / "fresh_holdout/holdout_evaluation.json"
    impact_path = ctx.state.root / "impact/overnight.json"
    reconciliation_path = ctx.state.root / "reconciliation/overnight.json"
    holdout = json.loads(holdout_path.read_text()) if holdout_path.is_file() else None
    impact = json.loads(impact_path.read_text()) if impact_path.is_file() else None
    reconciliation = (
        json.loads(reconciliation_path.read_text()) if reconciliation_path.is_file() else None
    )
    verdict = build_integral_verdict(
        campaign_id=ctx.state.campaign_id,
        regression_passed=bool(regression.get("base", {}).get("passed")),
        postgres_passed=bool(regression.get("postgres", {}).get("passed")),
        organ_reports=_load_organ_reports(ctx),
        holdout=holdout,
        impact_report=impact,
        artifact_reconciliation=reconciliation,
    )
    environment = ctx.state.manifest["blocks"]["preflight_environment"].get("result") or {}
    verdict["gates"]["clean_commit"] = bool(environment.get("git_clean"))
    verdict["staging_authorized"] = all(verdict["gates"].values())
    verdict["shadow_qualification_passed"] = verdict["staging_authorized"]
    verdict_path = ctx.state.root / "verdict.json"
    atomic_write_json(verdict_path, verdict)
    _register_report(ctx, verdict_path, kind="integral_neural_verdict", run_id=ctx.state.campaign_id)
    lines = [
        f"# Campaña neural integral {ctx.state.campaign_id}",
        "",
        f"- Storage oficial: PostgreSQL (`{ctx.postgres.database}`)",
        f"- Staging SHADOW autorizado: `{verdict['staging_authorized']}`",
        "- Promoción operativa autorizada: `False`",
        "- Entrenamiento 7B autorizado: `False`",
        "",
        "## Gates",
        "",
    ]
    lines.extend(f"- {name}: `{passed}`" for name, passed in verdict["gates"].items())
    report_path = ctx.state.root / "REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _register_report(ctx, report_path, kind="integral_neural_report", run_id=ctx.state.campaign_id)
    return verdict


def _run_preflight(ctx: RuntimeContext) -> None:
    _execute_block(ctx, "preflight_environment", _environment_report)
    postgres_block = ctx.state.manifest["blocks"]["preflight_postgres"]
    if postgres_block["status"] == "completed":
        _ensure_postgres(ctx, allow_existing=True)
    else:
        allow_existing = postgres_block["attempts"] > 0
        _execute_block(
            ctx,
            "preflight_postgres",
            lambda: _ensure_postgres(ctx, allow_existing=allow_existing),
        )
    ctx.ensure_storage()
    _execute_block(ctx, "preflight_connectome", _connectome_report)
    _execute_block(ctx, "preflight_artifacts", _artifact_report)


def _run_sequence(
    ctx: RuntimeContext,
    *,
    phase: str,
    names: Sequence[str],
    functions: Mapping[str, Callable[[], Mapping[str, Any]]],
    max_minutes: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + max_minutes * 60.0
    for name in names:
        try:
            _execute_block(ctx, name, functions[name])
        except Exception as error:
            checkpoint = ctx.state.checkpoint(phase=phase, next_block=name)
            raise CampaignError(
                f"campaign_block_failed:{name}:{type(error).__name__}:"
                f"resume_checkpoint={checkpoint['checkpoint_hash']}"
            ) from error
        if time.monotonic() >= deadline:
            next_block = ctx.state.next_pending(names)
            return ctx.state.checkpoint(phase=phase, next_block=next_block)
    return ctx.state.checkpoint(phase=phase, next_block=None)


def command_preflight(args: argparse.Namespace) -> int:
    ctx = _load_context(args)
    try:
        _run_preflight(ctx)
        checkpoint = ctx.state.checkpoint(phase="preflight", next_block="regression_full")
        print(json.dumps(checkpoint, indent=2, sort_keys=True))
        return 0
    finally:
        ctx.close()


def command_run(args: argparse.Namespace) -> int:
    ctx = _load_context(args)
    try:
        _run_preflight(ctx)
        if args.phase == "rehearsal":
            functions = {
                "regression_full": lambda: _regression(ctx, skip=args.skip_regression),
                "qualify_n0": lambda: _qualify_n0(ctx),
                "qualify_n1": lambda: _qualify_n1(ctx, epochs=args.n1_epochs),
                "qualify_n2": lambda: _qualify_n2(ctx),
                "qualify_n3": lambda: _qualify_n3(ctx),
                "qualify_n4": lambda: _qualify_n4(ctx),
                "qualify_n5": lambda: _qualify_n5(ctx),
                "qualify_n6": lambda: _qualify_n6(ctx),
                "life_kernel_paired_rehearsal": lambda: _paired_life(
                    ctx, phase="rehearsal", steps=args.life_steps
                ),
                "teacher_7b_rehearsal": lambda: _teacher(
                    ctx, phase="rehearsal", timeout_s=args.teacher_timeout
                ),
                "sqlite_contingency": lambda: _sqlite_contingency(ctx),
                "reconcile_rehearsal": lambda: _reconcile(ctx, phase="rehearsal"),
                "dump_rehearsal": lambda: _dump(ctx, phase="rehearsal"),
            }
            checkpoint = _run_sequence(
                ctx,
                phase="rehearsal",
                names=REHEARSAL_BLOCKS,
                functions=functions,
                max_minutes=args.max_minutes,
            )
        else:
            if not args.checkpoint:
                raise CampaignError("overnight_requires_manually_approved_checkpoint_hash")
            approved = ctx.state.verify_checkpoint(args.checkpoint)
            if approved["phase"] != "rehearsal" or approved["next_block"] is not None:
                raise CampaignError("overnight_requires_completed_rehearsal_checkpoint")
            functions = {
                "open_fresh_holdout": lambda: _open_holdout(
                    ctx, contexts=args.holdout_contexts
                ),
                "life_kernel_paired_overnight": lambda: _paired_life(
                    ctx, phase="overnight", steps=args.life_steps
                ),
                "teacher_7b_overnight": lambda: _teacher(
                    ctx, phase="overnight", timeout_s=args.teacher_timeout
                ),
                "reconcile_overnight": lambda: _reconcile(ctx, phase="overnight"),
                "verdict_overnight": lambda: _build_report(ctx),
                "dump_overnight": lambda: _dump(ctx, phase="overnight"),
            }
            checkpoint = _run_sequence(
                ctx,
                phase="overnight",
                names=OVERNIGHT_BLOCKS,
                functions=functions,
                max_minutes=args.max_minutes,
            )
        print(json.dumps(checkpoint, indent=2, sort_keys=True))
        return 0
    finally:
        ctx.close()


def command_resume(args: argparse.Namespace) -> int:
    ctx = _load_context(args)
    try:
        checkpoint = ctx.state.verify_checkpoint(args.checkpoint)
        ctx.state.reset_incomplete()
        args.phase = checkpoint["phase"] if checkpoint["phase"] in {"rehearsal", "overnight"} else "rehearsal"
        refreshed = ctx.state.checkpoint(
            phase=args.phase,
            next_block=ctx.state.next_pending(
                REHEARSAL_BLOCKS if args.phase == "rehearsal" else OVERNIGHT_BLOCKS
            ),
        )
        args.checkpoint = refreshed["checkpoint_hash"]
    finally:
        ctx.close()
    return command_run(args)


def command_report(args: argparse.Namespace) -> int:
    ctx = _load_context(args)
    try:
        _ensure_postgres(ctx, allow_existing=True)
        ctx.ensure_storage()
        verdict = _build_report(ctx)
        print(json.dumps(verdict, indent=2, sort_keys=True))
        return 0
    finally:
        ctx.close()


def command_stage(args: argparse.Namespace) -> int:
    ctx = _load_context(args)
    try:
        verdict_path = ctx.state.root / "verdict.json"
        if not verdict_path.is_file():
            raise CampaignError("campaign_integral_verdict_missing")
        verdict = json.loads(verdict_path.read_text())
        if not verdict.get("staging_authorized"):
            raise CampaignError("campaign_shadow_staging_gates_failed")
        if _git("status", "--porcelain"):
            raise CampaignError("campaign_shadow_staging_requires_clean_worktree")
        checkpoint = json.loads(ctx.state.checkpoint_path.read_text())
        if checkpoint.get("phase") != "overnight" or checkpoint.get("next_block") is not None:
            raise CampaignError("campaign_overnight_not_complete")
        target = (
            Path(args.target_root).expanduser().resolve()
            if args.target_root
            else Path(os.environ["RNFE_ARTIFACT_ROOT"]).resolve()
            / "neural-qualified"
            / ctx.state.campaign_id
        )
        profile = stage_lab_artifacts(
            source_root=ctx.state.root / "n1_recalibration/candidate",
            target_root=target,
            organs=("N1",),
            qualification={
                "campaign_id": ctx.state.campaign_id,
                "verdict_sha256": file_sha256(verdict_path),
                "checkpoint_hash": checkpoint["checkpoint_hash"],
                "staging_authorized": True,
                "shadow_qualification_passed": True,
            },
        )
        print(json.dumps(profile, indent=2, sort_keys=True))
        return 0
    finally:
        ctx.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--output-root", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight")
    run = commands.add_parser("run")
    run.add_argument("--phase", choices=("rehearsal", "overnight"), required=True)
    run.add_argument("--checkpoint", default=None)
    run.add_argument("--max-minutes", type=float, default=None)
    run.add_argument("--life-steps", type=int, default=None)
    run.add_argument("--n1-epochs", type=int, default=80)
    run.add_argument("--holdout-contexts", type=int, default=20)
    run.add_argument("--teacher-timeout", type=float, default=120.0)
    run.add_argument("--skip-regression", action="store_true")
    resume = commands.add_parser("resume")
    resume.add_argument("--checkpoint", required=True)
    resume.add_argument("--max-minutes", type=float, default=90.0)
    resume.add_argument("--life-steps", type=int, default=3)
    resume.add_argument("--n1-epochs", type=int, default=80)
    resume.add_argument("--holdout-contexts", type=int, default=20)
    resume.add_argument("--teacher-timeout", type=float, default=120.0)
    resume.add_argument("--skip-regression", action="store_true")
    commands.add_parser("report")
    stage = commands.add_parser("stage")
    stage.add_argument("--target-root", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.max_minutes is None:
            args.max_minutes = 90.0 if args.phase == "rehearsal" else 480.0
        if args.life_steps is None:
            args.life_steps = 3 if args.phase == "rehearsal" else 24
    try:
        if args.command == "preflight":
            return command_preflight(args)
        if args.command == "run":
            return command_run(args)
        if args.command == "resume":
            return command_resume(args)
        if args.command == "report":
            return command_report(args)
        if args.command == "stage":
            return command_stage(args)
    except CampaignError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
