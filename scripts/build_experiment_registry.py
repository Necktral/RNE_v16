#!/usr/bin/env python3
"""Build a deterministic, privacy-preserving registry of local RNFE evidence.

The registry contains metadata and checksums, never artifact bodies.  Directory
checksums cover the normalized inventory (relative path, size, and selected
evidence hashes); individual analysis files use a full content checksum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "experiment-registry.v1"
STALE_AFTER_SECONDS = 6 * 60 * 60
IGNORED_PARTS = {
    ".cache",
    ".git",
    ".grimp_cache",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "venv",
}
EVIDENCE_NAMES = {
    "campaign_manifest.json",
    "checkpoint.json",
    "evidence_manifest.json",
    "manifest.json",
    "QUARANTINE.json",
    "report.json",
    "SUPERVISOR.json",
    "verdict.json",
}
ALLOWED_SCALAR_KEYS = {
    "authority",
    "campaign_id",
    "commit",
    "completed_at",
    "created_at",
    "finished_at",
    "git_commit",
    "heartbeat_at",
    "id",
    "last_heartbeat",
    "phase",
    "promotion_authorized",
    "promotion_eligible",
    "reason",
    "run_id",
    "schema_version",
    "shadow_qualification_passed",
    "stage",
    "staging_authorized",
    "started_at",
    "state",
    "status",
    "timestamp",
    "training_authorized",
    "updated_at",
    "verdict",
    "version",
}
SENSITIVE_FRAGMENTS = {
    "api_key",
    "authorization",
    "context",
    "credential",
    "dsn",
    "password",
    "payload",
    "prompt",
    "secret",
    "token",
}
GITHUB_BY_BRANCH = {
    "feat/reasoning-family-quality-deep": "https://github.com/Necktral/RNE_v16/pull/5",
    "repair/P12": "https://github.com/Necktral/RNE_v16/pull/6",
    "feat/neural-substrate": "https://github.com/Necktral/RNE_v16/pull/7",
    "codex/neural-n0-a-m0": "https://github.com/Necktral/RNE_v16/pull/8",
    "codex/shadow-causal-observability-v1": "https://github.com/Necktral/RNE_v16/pull/9",
    "codex/ascg-v1-1-phase1-ledger": "https://github.com/Necktral/RNE_v16/pull/10",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def logical_uri(path: Path, development_root: Path) -> str:
    relative = path.resolve().relative_to(development_root.resolve())
    return "development://" + relative.as_posix()


def _is_ignored(path: Path) -> bool:
    return any(part in IGNORED_PARTS for part in path.parts) or path.name.endswith(".pyc")


def _extract_allowed(value: Any) -> Any:
    """Return only non-sensitive public execution metadata."""
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for raw_key in sorted(value):
            key = str(raw_key)
            lowered = key.lower()
            if any(fragment in lowered for fragment in SENSITIVE_FRAGMENTS):
                continue
            item = value[raw_key]
            if key in ALLOWED_SCALAR_KEYS and isinstance(item, (str, int, float, bool, type(None))):
                clean[key] = item
            elif "gate" in lowered and isinstance(item, (dict, list, str, int, float, bool, type(None))):
                clean[key] = _extract_gate_data(item)
        return clean
    return None


def _extract_gate_data(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _extract_gate_data(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if not any(fragment in str(key).lower() for fragment in SENSITIVE_FRAGMENTS)
        }
    if isinstance(value, list):
        return [_extract_gate_data(item) for item in value]
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return str(value)


def _read_evidence(path: Path, root: Path) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "kind": path.name,
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        evidence["parse_status"] = "invalid"
        return evidence
    evidence["parse_status"] = "valid"
    metadata = _extract_allowed(decoded)
    if metadata:
        evidence["metadata"] = metadata
    return evidence


def _iter_files(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return
    for current, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(name for name in dirs if name not in IGNORED_PARTS)
        base = Path(current)
        for name in sorted(files):
            path = base / name
            if not _is_ignored(path.relative_to(root)) and not path.is_symlink():
                yield path


def aggregate_directory(root: Path) -> dict[str, Any]:
    file_count = 0
    total_bytes = 0
    types: Counter[str] = Counter()
    evidence: list[dict[str, Any]] = []
    inventory = hashlib.sha256()
    for path in _iter_files(root):
        try:
            size = path.stat().st_size
        except OSError:
            continue
        relative = path.relative_to(root).as_posix()
        suffix = path.suffix.lower() or "[no_extension]"
        file_count += 1
        total_bytes += size
        types[suffix] += 1
        line = f"{relative}\0{size}\n".encode("utf-8", errors="surrogateescape")
        inventory.update(line)
        if path.name in EVIDENCE_NAMES:
            item = _read_evidence(path, root)
            evidence.append(item)
            inventory.update(item["sha256"].encode("ascii"))
    return {
        "file_count": file_count,
        "size_bytes": total_bytes,
        "types": dict(sorted(types.items())),
        "sha256": inventory.hexdigest(),
        "hash_scope": "normalized_inventory_and_selected_evidence",
        "evidence": sorted(evidence, key=lambda item: item["path"]),
    }


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def classify_status(evidence: list[dict[str, Any]], snapshot_date: str) -> str:
    if any(item["kind"] == "QUARANTINE.json" for item in evidence):
        return "failed_closed"
    if any(item["parse_status"] == "invalid" for item in evidence):
        return "invalid"
    metadata = [item.get("metadata", {}) for item in evidence]
    states = {
        str(data.get(key, "")).lower()
        for data in metadata
        for key in ("status", "state", "verdict")
        if data.get(key) is not None
    }
    if states & {"running", "in_progress"}:
        heartbeats = [
            parsed
            for data in metadata
            for key in ("heartbeat_at", "last_heartbeat", "updated_at", "timestamp")
            if (parsed := _parse_timestamp(data.get(key))) is not None
        ]
        cutoff = datetime.combine(
            datetime.strptime(snapshot_date, "%Y-%m-%d").date(), time.max, tzinfo=timezone.utc
        ).timestamp() - STALE_AFTER_SECONDS
        if not heartbeats or max(heartbeats).timestamp() < cutoff:
            return "stale_running"
        return "running"
    if states & {"failed", "fail", "blocked", "quarantined", "rejected"}:
        return "failed"
    if states & {"completed", "complete", "passed", "pass", "accepted", "success"}:
        return "completed"
    if any(item["kind"] == "verdict.json" for item in evidence):
        return "completed"
    if evidence:
        return "incomplete"
    return "discovered"


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9._-]+", "-", value.lower()).strip("-")
    return normalized or "unnamed"


def make_entry(family: str, path: Path, development_root: Path) -> dict[str, Any]:
    aggregate = aggregate_directory(path)
    uri = logical_uri(path, development_root)
    return {
        "id": f"{family}:{_slug(path.name)}",
        "family": family,
        "logical_path": uri,
        "branch": None,
        "commit": None,
        "status": classify_status(aggregate["evidence"], BUILD_SNAPSHOT_DATE),
        **aggregate,
        "github": None,
    }


def _git(path: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def worktree_entry(path: Path, development_root: Path) -> dict[str, Any]:
    uri = logical_uri(path, development_root)
    branch = _git(path, "branch", "--show-current") or None
    commit = _git(path, "rev-parse", "HEAD")
    porcelain = _git(path, "status", "--short")
    tracked = _git(path, "ls-files", "-z")
    files = [name for name in (tracked or "").split("\0") if name]
    total_bytes = 0
    types: Counter[str] = Counter()
    inventory = hashlib.sha256()
    for name in sorted(files):
        candidate = path / name
        try:
            size = candidate.stat().st_size
        except OSError:
            continue
        total_bytes += size
        types[candidate.suffix.lower() or "[no_extension]"] += 1
        inventory.update(f"{name}\0{size}\n".encode())
    return {
        "id": f"worktree:{_slug(path.name)}",
        "family": "worktree",
        "logical_path": uri,
        "branch": branch,
        "commit": commit,
        "status": "dirty" if porcelain else "clean",
        "file_count": len(files),
        "size_bytes": total_bytes,
        "types": dict(sorted(types.items())),
        "sha256": inventory.hexdigest(),
        "hash_scope": "tracked_file_inventory",
        "evidence": [],
        "github": GITHUB_BY_BRANCH.get(branch or ""),
    }


def analysis_entry(path: Path, root: Path, development_root: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    suffix = path.suffix.lower() or "[no_extension]"
    return {
        "id": f"analysis:{_slug(relative)}",
        "family": "analysis",
        "logical_path": logical_uri(path, development_root),
        "branch": None,
        "commit": None,
        "status": "archived",
        "file_count": 1,
        "size_bytes": path.stat().st_size,
        "types": {suffix: 1},
        "sha256": sha256_file(path),
        "hash_scope": "file_content",
        "evidence": [],
        "github": None,
    }


def _directories(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted((path for path in root.iterdir() if path.is_dir() and not path.is_symlink()), key=lambda p: p.name)


BUILD_SNAPSHOT_DATE = "1970-01-01"


def build_registry(development_root: Path, snapshot_date: str) -> dict[str, Any]:
    global BUILD_SNAPSHOT_DATE
    datetime.strptime(snapshot_date, "%Y-%m-%d")
    BUILD_SNAPSHOT_DATE = snapshot_date
    development_root = development_root.resolve()
    repo = development_root / "RNE_v16"
    worktrees_root = development_root / "RNE_v16_worktrees"
    entries: list[dict[str, Any]] = []

    for path in _directories(worktrees_root):
        if path.name == "experiment-registry-sync-20260807":
            continue
        entries.append(worktree_entry(path, development_root))

    campaign_root = repo / "rnfe_artifacts" / "integral_campaigns"
    for path in _directories(campaign_root):
        entries.append(make_entry("integral_campaign", path, development_root))

    for path in _directories(repo / "rnfe_artifacts"):
        entries.append(make_entry("rnfe_artifact_root", path, development_root))

    for path in _directories(repo / "data" / "artifacts"):
        entries.append(make_entry("data_artifact_run", path, development_root))

    for path in _directories(repo / "work"):
        entries.append(make_entry("local_work_family", path, development_root))

    analysis_root = development_root / "RNE_v16_analysis"
    if analysis_root.is_dir():
        for path in sorted(analysis_root.rglob("*")):
            if path.is_file() and not path.is_symlink() and not _is_ignored(path.relative_to(analysis_root)):
                entries.append(analysis_entry(path, analysis_root, development_root))

    entries.sort(key=lambda item: (item["family"], item["logical_path"], item["id"]))
    ids = [item["id"] for item in entries]
    if len(ids) != len(set(ids)):
        duplicates = sorted(name for name, count in Counter(ids).items() if count > 1)
        raise ValueError(f"duplicate registry ids: {duplicates}")

    family_counts = Counter(item["family"] for item in entries)
    status_counts = Counter(item["status"] for item in entries)
    return {
        "schema_version": SCHEMA_VERSION,
        "snapshot_date": snapshot_date,
        "development_root": "development://",
        "policy": {
            "artifact_bodies_published": False,
            "directory_hash_scope": "normalized inventory plus selected evidence hashes",
            "external_teacher": "Codex/frontier",
            "local_7b_role": "supervised student/proposer without authority",
            "sensitive_fields_excluded": sorted(SENSITIVE_FRAGMENTS),
        },
        "summary": {
            "entries": len(entries),
            "by_family": dict(sorted(family_counts.items())),
            "by_status": dict(sorted(status_counts.items())),
        },
        "entries": entries,
    }


def canonical_json(registry: dict[str, Any]) -> str:
    return json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--snapshot-date", required=True)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rendered = canonical_json(build_registry(args.development_root, args.snapshot_date))
    if args.check:
        try:
            current = args.output.read_text(encoding="utf-8")
        except OSError:
            print(f"registry missing: {args.output}", file=sys.stderr)
            return 1
        if current != rendered:
            print(f"registry is stale: {args.output}", file=sys.stderr)
            return 1
        print(f"registry is current: {args.output}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"wrote {len(json.loads(rendered)['entries'])} entries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
