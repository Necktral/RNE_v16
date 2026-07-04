from __future__ import annotations

from runtime.organism.constitution import OrganismConstitution
from runtime.organism.snapshot import OrganismSnapshot
from runtime.organism.state import IdentityState, OrganismBeliefState, OrganismState, PolicyState, ViabilityState
from runtime.organism.trajectory import OrganismTrajectory
from runtime.organism.constitution_flow import ConstitutionalFlowEngine


def _snapshot(idx: int, *, drift: float = 0.0, identity_hash: str = "const") -> OrganismSnapshot:
    state = OrganismState(
        state_id=f"s-{idx}",
        timestamp=f"2026-01-01T00:00:0{idx}Z",
        active_regime="thermal_homeostasis",
        episode_count=idx,
        belief=OrganismBeliefState(
            alarm_probability=min(1.0, 0.1 + 0.05 * idx),
            intervention_efficacy=max(0.0, 0.8 - 0.03 * idx),
            causal_support_confidence=max(0.0, 0.9 - 0.04 * idx),
            memory_purity_estimate=max(0.0, 0.95 - 0.03 * idx),
            trace_integrity_confidence=max(0.0, 0.90 - 0.02 * idx),
            regime_uncertainty=min(1.0, 0.2 + 0.03 * idx),
        ),
        policy=PolicyState(
            control_class="reactive",
            sensitivity=0.6,
            perturbation_tolerance=0.4,
            recovery_capacity=0.8,
            accumulated_drift=drift,
        ),
        identity=IdentityState(
            lineage_id="L1",
            constitution_hash=identity_hash,
            active_invariants=frozenset({"triadic_closure", "min_memory_purity"}),
        ),
        viability=ViabilityState(
            viability_margin=max(0.0, 0.9 - 0.05 * idx),
            accumulated_degradation=min(1.0, 0.1 + 0.04 * idx),
            recovery_debt=min(1.0, 0.1 + 0.02 * idx),
        ),
    )
    return OrganismSnapshot.from_state(state)


def test_trajectory_digest_tracks_drift_hysteresis_recovery() -> None:
    traj = OrganismTrajectory(trajectory_id="traj-1")
    traj.append(snapshot=_snapshot(0, drift=0.0), regime="thermal_homeostasis")
    traj.append(snapshot=_snapshot(1, drift=0.1), regime="thermal_homeostasis")
    traj.append(snapshot=_snapshot(2, drift=0.2), regime="thermal_homeostasis")
    digest = traj.digest(window_size=8)

    assert digest.trajectory_id == "traj-1"
    assert digest.window_end_episode == 2
    assert digest.drift_score > 0.0
    assert 0.0 <= digest.hysteresis_score <= 1.0
    assert 0.0 <= digest.recovery_score <= 1.0


def test_flow_can_fail_even_if_snapshots_are_pointwise_valid() -> None:
    constitution = OrganismConstitution()
    traj = OrganismTrajectory(trajectory_id="traj-2")

    s0 = _snapshot(0, drift=0.05, identity_hash="h0")
    s1 = _snapshot(1, drift=0.45, identity_hash="h1")
    s2 = _snapshot(2, drift=0.85, identity_hash="h2")

    # Snapshot-level checks can pass with soft drifts.
    assert constitution.validate(s0.to_state()).is_valid
    assert constitution.validate(s1.to_state()).is_valid
    assert constitution.validate(s2.to_state()).is_valid

    traj.append(snapshot=s0, regime="thermal_homeostasis")
    traj.append(snapshot=s1, regime="thermal_homeostasis")
    traj.append(snapshot=s2, regime="thermal_homeostasis")

    engine = ConstitutionalFlowEngine(
        identity_curvature_bound=0.05,
        policy_phase_jump_bound=0.10,
        constitutional_erosion_rate_max=0.20,
    )
    result = engine.evaluate(traj, window_size=8)

    assert result.flow_validity is False
    assert result.erosion > 0.0
    assert len(result.violations) >= 1
