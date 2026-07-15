from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from runtime.neural.campaign import CampaignError, CampaignState, campaign_database_name
from scripts.run_integral_neural_campaign import (
    _overnight_checkpoint_allowed,
    _run_sequence,
)
from scripts.supervise_integral_neural_campaign import (
    NightlySupervisor,
    SupervisorPolicy,
    SupervisorError,
    exclusive_lock,
    nightly_campaign_id,
)


def _state(tmp_path: Path) -> CampaignState:
    return CampaignState.create(
        root=tmp_path / "campaign",
        campaign_id="supervisor-test",
        commit="c" * 40,
        database=campaign_database_name("supervisor-test"),
        schema_sha256="d" * 64,
        artifact_root=tmp_path / "artifacts",
        blocks=("first", "second"),
        configuration={},
    )


def test_nightly_campaign_id_is_stable_per_local_night_and_commit() -> None:
    now = datetime(2026, 7, 15, 22, 0, tzinfo=timezone.utc)

    assert nightly_campaign_id(commit="ABCDEF012345", now=now) == (
        "neural-nightly-20260715-abcdef01"
    )


def test_supervisor_lock_rejects_overlapping_invocation(tmp_path: Path) -> None:
    lock = tmp_path / "nightly.lock"

    with exclusive_lock(lock):
        with pytest.raises(SupervisorError, match="already_running"):
            with exclusive_lock(lock):
                pass


def test_atomic_sequence_keeps_checkpoint_valid_after_block_failure(tmp_path: Path) -> None:
    state = _state(tmp_path)
    ctx = SimpleNamespace(state=state, storage=None)

    with pytest.raises(CampaignError, match="campaign_block_failed:first"):
        _run_sequence(
            ctx,
            phase="overnight",
            names=("first", "second"),
            functions={
                "first": lambda: (_ for _ in ()).throw(RuntimeError("interrupted")),
                "second": lambda: {"passed": True},
            },
            max_minutes=1.0,
        )

    checkpoint = json.loads(state.checkpoint_path.read_text(encoding="utf-8"))
    verified = state.verify_checkpoint(checkpoint["checkpoint_hash"])
    assert verified["phase"] == "overnight"
    assert verified["next_block"] == "first"
    assert state.manifest["blocks"]["first"]["status"] == "failed"


def test_atomic_sequence_checkpoints_next_pending_block(tmp_path: Path) -> None:
    state = _state(tmp_path)
    ctx = SimpleNamespace(state=state, storage=None)

    checkpoint = _run_sequence(
        ctx,
        phase="rehearsal",
        names=("first", "second"),
        functions={
            "first": lambda: {"passed": True},
            "second": lambda: {"passed": True},
        },
        max_minutes=1.0,
    )

    assert checkpoint["next_block"] is None
    assert state.verify_checkpoint(checkpoint["checkpoint_hash"])["phase"] == "rehearsal"


def test_overnight_resume_accepts_only_explicit_resume_mode() -> None:
    rehearsal = {"phase": "rehearsal", "next_block": None}
    running = {"phase": "overnight", "next_block": "teacher_7b_overnight"}

    assert _overnight_checkpoint_allowed(rehearsal, resume_mode=False) is True
    assert _overnight_checkpoint_allowed(running, resume_mode=False) is False
    assert _overnight_checkpoint_allowed(running, resume_mode=True) is True


def test_terminal_supervised_campaign_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "terminal-campaign"
    state = CampaignState.create(
        root=root,
        campaign_id="terminal-campaign",
        commit="c" * 40,
        database=campaign_database_name("terminal-campaign"),
        schema_sha256="d" * 64,
        artifact_root=root / "artifacts",
        blocks=(
            "open_fresh_holdout",
            "life_kernel_paired_overnight",
            "teacher_7b_overnight",
            "reconcile_overnight",
            "verdict_overnight",
            "dump_overnight",
        ),
        configuration={},
    )
    for name in state.manifest["blocks"]:
        state.begin(name)
        state.complete(name, {"passed": True})

    supervisor = NightlySupervisor(
        campaign_id="terminal-campaign",
        env_file=tmp_path / ".env",
        output_root=tmp_path,
        policy=SupervisorPolicy(minimum_free_gb=0),
    )
    supervisor.journal.payload["status"] = "quarantined"
    supervisor.journal.write()

    assert supervisor.run() == 0
    assert supervisor.journal.payload["status"] == "quarantined"
