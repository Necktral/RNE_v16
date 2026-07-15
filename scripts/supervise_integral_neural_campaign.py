#!/usr/bin/env python3
"""Supervise a complete RNFE neural campaign without a human night operator.

The supervisor is deliberately deterministic.  It starts or resumes the integral
campaign, monitors disk/GPU/wall-clock limits, retries interrupted atomic blocks,
records the user's standing authorization for unattended overnight diagnostics,
and closes PostgreSQL evidence.  It may stage a fully qualified SHADOW artifact,
but it can never train, promote, activate, or grant operational authority.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.neural.campaign import (
    CampaignError,
    CampaignState,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    load_env_file,
    resolve_writable_artifact_root,
    validate_campaign_id,
)
from runtime.storage import StorageConfig
from scripts.run_integral_neural_campaign import (
    DEFAULT_ENV_FILE,
    OVERNIGHT_BLOCKS,
    REHEARSAL_BLOCKS,
)


SUPERVISOR_SCHEMA_VERSION = "rnfe-neural-night-supervisor-v1"
APPROVAL_SCHEMA_VERSION = "rnfe-neural-unattended-approval-v1"
QUARANTINE_SCHEMA_VERSION = "rnfe-neural-quarantine-v1"
RUNNER = REPO_ROOT / "scripts/run_integral_neural_campaign.py"


class SupervisorError(RuntimeError):
    """Fail-closed unattended campaign error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def nightly_campaign_id(*, commit: str, now: datetime | None = None) -> str:
    """Return one resumable campaign identity per local night and commit."""
    local_now = now or datetime.now().astimezone()
    return validate_campaign_id(
        f"neural-nightly-{local_now:%Y%m%d}-{commit[:8].lower()}"
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SupervisorError(f"supervisor_json_object_required:{path}")
    return value


def _phase_complete(root: Path, names: Sequence[str]) -> bool:
    manifest_path = root / "campaign_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = _read_json(manifest_path)
    blocks = manifest.get("blocks", {})
    return all(blocks.get(name, {}).get("status") == "completed" for name in names)


def _checkpoint(root: Path) -> dict[str, Any] | None:
    path = root / "checkpoint.json"
    return _read_json(path) if path.is_file() else None


@dataclass(frozen=True, slots=True)
class SupervisorPolicy:
    max_attempts_per_phase: int = 4
    retry_delays_s: tuple[float, ...] = (30.0, 120.0, 300.0)
    monitor_interval_s: float = 15.0
    max_wall_minutes: float = 600.0
    rehearsal_max_minutes: float = 90.0
    overnight_max_minutes: float = 480.0
    minimum_free_gb: float = 20.0
    maximum_gpu_temperature_c: float = 88.0
    thermal_grace_samples: int = 3
    rehearsal_life_steps: int = 3
    overnight_life_steps: int = 24
    n1_epochs: int = 80
    holdout_contexts: int = 20
    teacher_timeout_s: float = 120.0
    auto_stage_qualified_shadow: bool = False

    def validate(self) -> None:
        if self.max_attempts_per_phase < 1:
            raise SupervisorError("supervisor_attempts_must_be_positive")
        if self.monitor_interval_s <= 0 or self.max_wall_minutes <= 0:
            raise SupervisorError("supervisor_time_budget_must_be_positive")
        if self.minimum_free_gb < 0:
            raise SupervisorError("supervisor_minimum_free_gb_invalid")
        if self.maximum_gpu_temperature_c <= 0:
            raise SupervisorError("supervisor_gpu_temperature_invalid")
        if self.thermal_grace_samples < 1:
            raise SupervisorError("supervisor_thermal_grace_invalid")


class SupervisorJournal:
    """Small atomic heartbeat and event journal outside the official blob plane."""

    def __init__(
        self,
        *,
        root: Path,
        campaign_id: str,
        commit: str,
        policy: SupervisorPolicy,
    ) -> None:
        self.path = root / "SUPERVISOR.json"
        if self.path.is_file():
            self.payload = _read_json(self.path)
        else:
            self.payload = {
                "schema_version": SUPERVISOR_SCHEMA_VERSION,
                "campaign_id": campaign_id,
                "commit": commit,
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "status": "initializing",
                "authority_ceiling": "shadow",
                "training_authorized": False,
                "promotion_authorized": False,
                "activation_automatic": False,
                "policy": asdict(policy),
                "invocations": 0,
                "events": [],
            }
        self.payload["invocations"] = int(self.payload.get("invocations") or 0) + 1
        self.write()

    def write(self) -> None:
        self.payload["updated_at"] = utc_now()
        atomic_write_json(self.path, self.payload)

    def event(self, event: str, *, status: str | None = None, **details: Any) -> None:
        if status is not None:
            self.payload["status"] = status
        events = list(self.payload.get("events") or [])
        events.append({"at": utc_now(), "event": event, **details})
        self.payload["events"] = events[-500:]
        self.write()

    def heartbeat(self, *, process_id: int, health: Mapping[str, Any]) -> None:
        self.payload["heartbeat"] = {
            "at": utc_now(),
            "process_id": process_id,
            "health": dict(health),
        }
        self.write()


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    """Prevent Task Scheduler, recovery, and a human invocation from overlapping."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SupervisorError("nightly_supervisor_already_running") from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} started_at={utc_now()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class NightlySupervisor:
    def __init__(
        self,
        *,
        campaign_id: str,
        env_file: Path,
        output_root: Path,
        policy: SupervisorPolicy,
    ) -> None:
        self.campaign_id = validate_campaign_id(campaign_id)
        self.env_file = env_file.resolve()
        self.output_root = output_root.resolve()
        self.root = self.output_root / self.campaign_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.policy = policy
        self.commit = _git("rev-parse", "HEAD")
        self.journal = SupervisorJournal(
            root=self.root,
            campaign_id=self.campaign_id,
            commit=self.commit,
            policy=policy,
        )
        self.started_monotonic = time.monotonic()
        self._thermal_samples = 0

    def _remaining_wall_s(self) -> float:
        elapsed = time.monotonic() - self.started_monotonic
        return max(0.0, self.policy.max_wall_minutes * 60.0 - elapsed)

    def _health(self) -> dict[str, Any]:
        usage = shutil.disk_usage(self.output_root)
        health: dict[str, Any] = {
            "disk_free_gb": round(usage.free / (1024**3), 3),
            "disk_total_gb": round(usage.total / (1024**3), 3),
            "gpu_available": False,
            "gpu_temperature_c": None,
            "gpu_memory_used_mib": None,
        }
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if gpu.returncode == 0 and gpu.stdout.strip():
            first = gpu.stdout.strip().splitlines()[0].split(",")
            try:
                health.update(
                    {
                        "gpu_available": True,
                        "gpu_temperature_c": float(first[0].strip()),
                        "gpu_memory_used_mib": float(first[1].strip()),
                    }
                )
            except (IndexError, ValueError):
                health["gpu_parse_error"] = True
        return health

    def _unsafe_reason(self, health: Mapping[str, Any]) -> str | None:
        if float(health["disk_free_gb"]) < self.policy.minimum_free_gb:
            return "minimum_free_disk_guard"
        temperature = health.get("gpu_temperature_c")
        if temperature is not None and float(temperature) > self.policy.maximum_gpu_temperature_c:
            self._thermal_samples += 1
        else:
            self._thermal_samples = 0
        if self._thermal_samples >= self.policy.thermal_grace_samples:
            return "gpu_thermal_guard"
        if self._remaining_wall_s() <= 0:
            return "supervisor_wall_clock_guard"
        return None

    @staticmethod
    def _terminate(process: subprocess.Popen[Any]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)

    def _run_process(self, *, label: str, command: Sequence[str]) -> int:
        health = self._health()
        unsafe = self._unsafe_reason(health)
        if unsafe:
            raise SupervisorError(unsafe)
        log_dir = self.root / "supervisor_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        sequence = len(list(log_dir.glob("*.log"))) + 1
        log_path = log_dir / f"{sequence:03d}-{label}.log"
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(REPO_ROOT)
        self.journal.event(
            "command_started",
            status="running",
            label=label,
            command=list(command),
            log_path=str(log_path),
        )
        with log_path.open("ab") as output:
            process = subprocess.Popen(
                list(command),
                cwd=REPO_ROOT,
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                while process.poll() is None:
                    health = self._health()
                    self.journal.heartbeat(process_id=process.pid, health=health)
                    unsafe = self._unsafe_reason(health)
                    if unsafe:
                        self.journal.event(
                            "command_guard_triggered",
                            status="recovering",
                            label=label,
                            guard=unsafe,
                            health=health,
                        )
                        self._terminate(process)
                        raise SupervisorError(unsafe)
                    time.sleep(
                        min(self.policy.monitor_interval_s, self._remaining_wall_s())
                    )
            except BaseException:
                self._terminate(process)
                raise
        returncode = int(process.returncode or 0)
        self.journal.event(
            "command_completed",
            status="recovering" if returncode else "running",
            label=label,
            returncode=returncode,
            log_path=str(log_path),
            log_sha256=file_sha256(log_path),
        )
        return returncode

    def _runner_base(self) -> list[str]:
        return [
            sys.executable,
            str(RUNNER),
            "--campaign-id",
            self.campaign_id,
            "--env-file",
            str(self.env_file),
            "--output-root",
            str(self.output_root),
        ]

    def _phase_arguments(self, phase: str) -> list[str]:
        if phase == "rehearsal":
            minutes = self.policy.rehearsal_max_minutes
            steps = self.policy.rehearsal_life_steps
        else:
            minutes = self.policy.overnight_max_minutes
            steps = self.policy.overnight_life_steps
        return [
            "--max-minutes",
            str(minutes),
            "--life-steps",
            str(steps),
            "--n1-epochs",
            str(self.policy.n1_epochs),
            "--holdout-contexts",
            str(self.policy.holdout_contexts),
            "--teacher-timeout",
            str(self.policy.teacher_timeout_s),
        ]

    def _resume_command(self, *, phase: str, checkpoint_hash: str) -> list[str]:
        return self._runner_base() + [
            "resume",
            "--checkpoint",
            checkpoint_hash,
            *self._phase_arguments(phase),
        ]

    def _fresh_phase_command(
        self, *, phase: str, checkpoint_hash: str | None = None
    ) -> list[str]:
        command = self._runner_base() + ["run", "--phase", phase]
        if checkpoint_hash:
            command.extend(["--checkpoint", checkpoint_hash])
        command.extend(self._phase_arguments(phase))
        return command

    def _sleep_backoff(self, attempt: int) -> None:
        index = min(attempt - 1, len(self.policy.retry_delays_s) - 1)
        delay = self.policy.retry_delays_s[index]
        end = time.monotonic() + min(delay, self._remaining_wall_s())
        self.journal.event(
            "retry_backoff", status="recovering", attempt=attempt, delay_s=delay
        )
        while time.monotonic() < end:
            health = self._health()
            unsafe = self._unsafe_reason(health)
            if unsafe:
                raise SupervisorError(unsafe)
            self.journal.payload["heartbeat"] = {"at": utc_now(), "health": health}
            self.journal.write()
            time.sleep(min(self.policy.monitor_interval_s, end - time.monotonic()))

    def _run_phase(self, phase: str) -> None:
        names = REHEARSAL_BLOCKS if phase == "rehearsal" else OVERNIGHT_BLOCKS
        if _phase_complete(self.root, names):
            self.journal.event("phase_already_complete", phase=phase)
            return
        for attempt in range(1, self.policy.max_attempts_per_phase + 1):
            checkpoint = _checkpoint(self.root)
            if checkpoint and checkpoint.get("phase") == phase:
                state = CampaignState.load(self.root)
                try:
                    state.verify_checkpoint(str(checkpoint["checkpoint_hash"]))
                except CampaignError:
                    # A power loss can land in the narrow interval between an
                    # atomic manifest replacement and its companion checkpoint.
                    # Rebuild only from the current campaign manifest/commit; the
                    # runner will still reset every running/failed block to zero.
                    checkpoint = state.checkpoint(
                        phase=phase,
                        next_block=state.next_pending(names),
                    )
                    self.journal.event(
                        "recovery_checkpoint_rebuilt",
                        status="recovering",
                        phase=phase,
                        checkpoint_hash=checkpoint["checkpoint_hash"],
                    )
                command = self._resume_command(
                    phase=phase,
                    checkpoint_hash=str(checkpoint["checkpoint_hash"]),
                )
                label = f"{phase}-resume-{attempt}"
            elif phase == "overnight":
                if not checkpoint or checkpoint.get("phase") != "rehearsal":
                    raise SupervisorError("overnight_rehearsal_checkpoint_missing")
                command = self._fresh_phase_command(
                    phase=phase,
                    checkpoint_hash=str(checkpoint["checkpoint_hash"]),
                )
                label = f"{phase}-start-{attempt}"
            else:
                command = self._fresh_phase_command(phase=phase)
                label = f"{phase}-start-{attempt}"
            try:
                returncode = self._run_process(label=label, command=command)
            except SupervisorError as error:
                returncode = 124
                self.journal.event(
                    "phase_attempt_interrupted",
                    status="recovering",
                    phase=phase,
                    attempt=attempt,
                    reason=str(error),
                )
            if _phase_complete(self.root, names):
                self.journal.event("phase_completed", phase=phase, attempt=attempt)
                return
            if attempt == self.policy.max_attempts_per_phase:
                raise SupervisorError(
                    f"phase_attempts_exhausted:{phase}:last_returncode={returncode}"
                )
            self._sleep_backoff(attempt)
        raise AssertionError("unreachable")

    def _authorize_overnight(self) -> str:
        checkpoint = _checkpoint(self.root)
        if (
            not checkpoint
            or checkpoint.get("phase") != "rehearsal"
            or checkpoint.get("next_block") is not None
        ):
            raise SupervisorError("completed_rehearsal_checkpoint_required")
        approval = {
            "schema_version": APPROVAL_SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "commit": self.commit,
            "authorized_at": utc_now(),
            "authorization_source": "user_standing_unattended_nightly_policy",
            "rehearsal_checkpoint_hash": checkpoint["checkpoint_hash"],
            "scope": "overnight_diagnostic_and_gate_bound_shadow_staging",
            "authority_ceiling": "shadow",
            "training_authorized": False,
            "promotion_authorized": False,
            "activation_automatic": False,
        }
        approval["approval_hash"] = canonical_sha256(approval)
        approval_path = self.root / "unattended_overnight_approval.json"
        atomic_write_json(approval_path, approval)

        state = CampaignState.load(self.root)
        state.manifest["manual_overnight_approval_required"] = False
        state.manifest["overnight_approval"] = {
            "mode": "unattended_user_authorized_policy",
            "path": str(approval_path),
            "sha256": file_sha256(approval_path),
            "approval_hash": approval["approval_hash"],
            "authority_ceiling": "shadow",
        }
        state.save()
        execution_checkpoint = state.checkpoint(phase="rehearsal", next_block=None)
        self.journal.event(
            "overnight_authorized",
            status="running",
            approval_path=str(approval_path),
            approval_hash=approval["approval_hash"],
            checkpoint_hash=execution_checkpoint["checkpoint_hash"],
        )
        return str(execution_checkpoint["checkpoint_hash"])

    def _write_quarantine(self, *, reason: str, verdict: Mapping[str, Any] | None) -> Path:
        path = self.root / "QUARANTINE.json"
        payload = {
            "schema_version": QUARANTINE_SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "created_at": utc_now(),
            "reason": reason,
            "gates": dict((verdict or {}).get("gates") or {}),
            "classification": (verdict or {}).get("classification", "failed_closed"),
            "authority_effect": "none",
            "staging_performed": False,
            "training_authorized": False,
            "promotion_authorized": False,
            "activation_automatic": False,
        }
        atomic_write_json(path, payload)
        return path

    def _verify_final_dump(self) -> dict[str, Any]:
        manifest = _read_json(self.root / "campaign_manifest.json")
        result = manifest["blocks"]["dump_overnight"].get("result") or {}
        dump_path = Path(str(result.get("path") or ""))
        if not dump_path.is_file():
            raise SupervisorError("overnight_dump_missing")
        actual_hash = file_sha256(dump_path)
        actual_size = dump_path.stat().st_size
        if actual_hash != result.get("sha256") or actual_size != result.get("size_bytes"):
            raise SupervisorError("overnight_dump_integrity_mismatch")
        return {
            "path": str(dump_path),
            "sha256": actual_hash,
            "size_bytes": actual_size,
        }

    def run(self) -> int:
        self.policy.validate()
        previous_status = str(self.journal.payload.get("status") or "")
        if (
            _phase_complete(self.root, OVERNIGHT_BLOCKS)
            and previous_status
            in {"quarantined", "qualified_shadow_staged", "qualified_shadow_awaiting_stage"}
        ):
            self.journal.event(
                "terminal_campaign_already_managed",
                status=previous_status,
            )
            return 0
        self.journal.event("supervisor_started", status="running")
        if _git("status", "--porcelain"):
            quarantine = self._write_quarantine(
                reason="dirty_worktree_blocks_unattended_campaign", verdict=None
            )
            self.journal.event(
                "supervisor_quarantined",
                status="quarantined",
                reason="dirty_worktree",
                quarantine_path=str(quarantine),
            )
            return 0
        initial_health = self._health()
        unsafe = self._unsafe_reason(initial_health)
        if unsafe:
            raise SupervisorError(unsafe)

        if not _phase_complete(self.root, OVERNIGHT_BLOCKS):
            self._run_phase("rehearsal")
            current = _checkpoint(self.root)
            if not current or current.get("phase") != "overnight":
                self._authorize_overnight()
            self._run_phase("overnight")

        report_code = self._run_process(
            label="final-report", command=self._runner_base() + ["report"]
        )
        if report_code != 0:
            raise SupervisorError(f"final_report_failed:returncode={report_code}")
        verdict_path = self.root / "verdict.json"
        if not verdict_path.is_file():
            raise SupervisorError("final_verdict_missing")
        verdict = _read_json(verdict_path)
        dump = self._verify_final_dump()
        staging_performed = False
        outcome = "quarantined"
        if verdict.get("staging_authorized"):
            if self.policy.auto_stage_qualified_shadow:
                stage_code = self._run_process(
                    label="qualified-shadow-stage",
                    command=self._runner_base() + ["stage"],
                )
                if stage_code != 0:
                    raise SupervisorError(
                        f"qualified_shadow_stage_failed:returncode={stage_code}"
                    )
                staging_performed = True
                outcome = "qualified_shadow_staged"
            else:
                outcome = "qualified_shadow_awaiting_stage"
        else:
            quarantine = self._write_quarantine(
                reason="integral_gates_failed", verdict=verdict
            )
            self.journal.payload["quarantine_path"] = str(quarantine)

        self.journal.payload.update(
            {
                "status": outcome,
                "completed_at": utc_now(),
                "checkpoint": _checkpoint(self.root),
                "verdict_path": str(verdict_path),
                "verdict_sha256": file_sha256(verdict_path),
                "dump": dump,
                "staging_performed": staging_performed,
                "training_authorized": False,
                "promotion_authorized": False,
                "activation_automatic": False,
            }
        )
        self.journal.event("supervisor_completed", status=outcome)
        return 0

    def manage(self) -> int:
        """Run fail-closed while leaving a durable operational incident marker."""
        try:
            return self.run()
        except SupervisorError as error:
            quarantine = self._write_quarantine(
                reason=f"supervisor_operational_failure:{error}", verdict=None
            )
            self.journal.event(
                "supervisor_failed",
                status="failed",
                reason=str(error),
                quarantine_path=str(quarantine),
            )
            raise


def _resolve_output_root(env_file: Path, output_root: Path | None) -> Path:
    load_env_file(env_file, override=True)
    config = StorageConfig.from_env()
    if config.mode != "postgres" or not config.postgres_dsn:
        raise SupervisorError("nightly_supervisor_requires_postgres_direct")
    requested = config.artifact_root.resolve()
    effective, _ = resolve_writable_artifact_root(
        requested, native_fallback=env_file.parent / "rnfe_artifacts"
    )
    target = output_root.resolve() if output_root else effective / "integral_campaigns"
    target.mkdir(parents=True, exist_ok=True)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--max-wall-minutes", type=float, default=600.0)
    parser.add_argument("--minimum-free-gb", type=float, default=20.0)
    parser.add_argument("--maximum-gpu-temperature-c", type=float, default=88.0)
    parser.add_argument("--monitor-interval-s", type=float, default=15.0)
    parser.add_argument("--auto-stage-qualified-shadow", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        env_file = args.env_file.expanduser().resolve()
        output_root = _resolve_output_root(env_file, args.output_root)
        commit = _git("rev-parse", "HEAD")
        campaign_id = args.campaign_id or nightly_campaign_id(commit=commit)
        policy = SupervisorPolicy(
            max_attempts_per_phase=args.max_attempts,
            monitor_interval_s=args.monitor_interval_s,
            max_wall_minutes=args.max_wall_minutes,
            minimum_free_gb=args.minimum_free_gb,
            maximum_gpu_temperature_c=args.maximum_gpu_temperature_c,
            auto_stage_qualified_shadow=args.auto_stage_qualified_shadow,
        )
        lock_path = output_root / ".nightly-supervisor.lock"
        with exclusive_lock(lock_path):
            supervisor = NightlySupervisor(
                campaign_id=campaign_id,
                env_file=env_file,
                output_root=output_root,
                policy=policy,
            )
            return supervisor.manage()
    except SupervisorError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
