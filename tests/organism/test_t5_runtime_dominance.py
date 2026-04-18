"""T5 Sovereignty Tests: Runtime Dominance.

These tests verify that T5 trajectory ontology is OPERATIONALLY DOMINANT,
not just architecturally present.

CRITICAL: These tests must FAIL if T5 loses sovereignty.
"""

import pytest
from runtime.organism.state import OrganismState, IdentityState, transition_organism_state
from runtime.organism.trajectory import OrganismTrajectory, TrajectoryWindow
from runtime.organism.constitution import OrganismConstitution
from runtime.organism.viability import ViabilityKernel
from runtime.storage.records import utc_now_iso


class TestT5RuntimeDominance:
    """Test suite proving T5 trajectory is sovereign runtime unit."""

    def test_trajectory_is_not_mere_list_of_snapshots(self):
        """GATE 1: Trajectory must have evolutionary semantics, not just snapshot collection."""
        traj = OrganismTrajectory(organism_id="test-org")
        constitution = OrganismConstitution()

        # Create first state
        state1 = OrganismState(
            state_id="state-1",
            timestamp=utc_now_iso(),
            active_regime="baseline",
        )

        validation1 = constitution.validate(state1)
        traj.append_point(
            state=state1,
            regime="baseline",
            episode_id="ep-1",
            timestamp=utc_now_iso(),
            constitutional_validation=validation1,
            viability_margin=1.0,
        )

        # Trajectory must have constitutional flow score
        assert hasattr(traj, "constitutional_flow_score")
        assert traj.constitutional_flow_score > 0.0

        # Trajectory must track regime history
        assert hasattr(traj, "regime_history")
        assert traj.current_regime == "baseline"

        # Trajectory is NOT just len(points)
        assert traj.length == 1
        assert traj.total_episodes == 1
        assert traj.constitutional_flow_score == 1.0  # Initially perfect

    def test_trajectory_has_evolutionary_dynamics(self):
        """GATE 1: Trajectory must have flow dynamics, not static snapshots."""
        traj = OrganismTrajectory(organism_id="test-org")
        constitution = OrganismConstitution()

        # Add multiple states with degrading constitution
        for i in range(5):
            state = OrganismState(
                state_id=f"state-{i}",
                timestamp=utc_now_iso(),
                active_regime="baseline",
                episode_count=i,
            )
            validation = constitution.validate(state)
            traj.append_point(
                state=state,
                regime="baseline",
                episode_id=f"ep-{i}",
                timestamp=utc_now_iso(),
                constitutional_validation=validation,
                viability_margin=0.9 - (i * 0.1),  # Degrading
            )

        # Constitutional flow score must evolve (decay with violations or time)
        assert traj.constitutional_flow_score < 1.0  # Decayed
        assert traj.length == 5

        # Window must preserve evolutionary information
        window = traj.get_window(window_size=3)
        assert window.length == 3
        assert len(window.margin_trajectory) == 3
        # Margin trajectory must show degradation
        assert window.margin_trajectory[0] > window.margin_trajectory[-1]

    def test_trajectory_tracks_regime_transitions(self):
        """GATE 1: Trajectory must track regime changes as first-class events."""
        traj = OrganismTrajectory(organism_id="test-org")
        constitution = OrganismConstitution()

        regimes = ["baseline", "baseline", "adaptive", "adaptive", "reactive"]
        for i, regime in enumerate(regimes):
            state = OrganismState(
                state_id=f"state-{i}",
                timestamp=utc_now_iso(),
                active_regime=regime,
            )
            validation = constitution.validate(state)
            traj.append_point(
                state=state,
                regime=regime,
                episode_id=f"ep-{i}",
                timestamp=utc_now_iso(),
                constitutional_validation=validation,
            )

        # Regime history must capture transitions
        assert len(traj.regime_history) > 0
        # First transition: baseline → adaptive at step 2
        # Second transition: adaptive → reactive at step 4
        assert any(r == "adaptive" for r, _ in traj.regime_history)
        assert any(r == "reactive" for r, _ in traj.regime_history)
        assert traj.current_regime == "reactive"

    def test_trajectory_window_operational(self):
        """GATE 1: TrajectoryWindow must be operational unit for runtime."""
        traj = OrganismTrajectory(organism_id="test-org")
        constitution = OrganismConstitution()

        # Add 20 points
        for i in range(20):
            state = OrganismState(
                state_id=f"state-{i}",
                timestamp=utc_now_iso(),
                active_regime="baseline",
            )
            validation = constitution.validate(state)
            traj.append_point(
                state=state,
                regime="baseline",
                episode_id=f"ep-{i}",
                timestamp=utc_now_iso(),
                constitutional_validation=validation,
                viability_margin=0.95,
            )

        # Window of last 10
        window = traj.get_window(window_size=10)

        # Window must be operational
        assert window.length == 10
        assert window.window_start_index == 10
        assert window.current_state is not None
        assert window.current_state.state_id == "state-19"

        # Window must provide trajectorial views
        assert len(window.margin_trajectory) == 10
        assert len(window.regime_sequence) == 10
        assert len(window.constitutional_validity_sequence) == 10

    def test_state_transition_creates_evolutionary_link(self):
        """GATE 1: State transitions must maintain prev_state_id chain."""
        traj = OrganismTrajectory(organism_id="test-org")
        constitution = OrganismConstitution()

        state1 = OrganismState(
            state_id="state-1",
            timestamp=utc_now_iso(),
            active_regime="baseline",
        )
        traj.append_point(
            state=state1,
            regime="baseline",
            episode_id="ep-1",
            timestamp=utc_now_iso(),
            constitutional_validation=constitution.validate(state1),
        )

        state2 = OrganismState(
            state_id="state-2",
            timestamp=utc_now_iso(),
            active_regime="baseline",
        )
        traj.append_point(
            state=state2,
            regime="baseline",
            episode_id="ep-2",
            timestamp=utc_now_iso(),
            constitutional_validation=constitution.validate(state2),
        )

        # Second point must have prev_state_id
        assert traj.points[0].prev_state_id is None  # First has no previous
        assert traj.points[1].prev_state_id == "state-1"

    def test_viability_kernel_operates_on_trajectory(self):
        """GATE 1: ViabilityKernel must assess trajectory dynamics, not just snapshots."""
        constitution = OrganismConstitution()
        kernel = ViabilityKernel(constitution=constitution)

        state1 = OrganismState(
            state_id="state-1",
            timestamp=utc_now_iso(),
            active_regime="baseline",
        )

        state2 = OrganismState(
            state_id="state-2",
            timestamp=utc_now_iso(),
            active_regime="baseline",
        )

        # Assess with trajectory context (previous state)
        assessment = kernel.assess(state=state2, previous_state=state1)

        # Assessment must include margin_delta (trajectory property)
        assert hasattr(assessment, "margin_delta")
        assert hasattr(assessment, "degradation_rate")

        # margin_delta and degradation_rate are ZERO for identical states
        # but the mechanism exists for trajectory-based assessment
        assert isinstance(assessment.margin_delta, float)
        assert isinstance(assessment.degradation_rate, float)

    def test_trajectory_serialization_preserves_evolutionary_metadata(self):
        """GATE 1: Trajectory serialization must preserve flow, not just snapshots."""
        traj = OrganismTrajectory(organism_id="test-org", start_timestamp=utc_now_iso())
        constitution = OrganismConstitution()

        for i in range(3):
            state = OrganismState(
                state_id=f"state-{i}",
                timestamp=utc_now_iso(),
                active_regime="baseline" if i < 2 else "adaptive",
            )
            traj.append_point(
                state=state,
                regime=state.active_regime,
                episode_id=f"ep-{i}",
                timestamp=utc_now_iso(),
                constitutional_validation=constitution.validate(state),
            )

        # Serialize
        traj_dict = traj.to_dict()

        # Must preserve evolutionary metadata
        assert "constitutional_flow_score" in traj_dict
        assert "regime_history" in traj_dict
        assert "total_episodes" in traj_dict
        assert "current_regime" in traj_dict
        assert traj_dict["current_regime"] == "adaptive"
        assert traj_dict["total_episodes"] == 3

        # Points must include evolutionary links
        assert all("prev_state_id" in p for p in traj_dict["points"])


class TestT5TrajectoryDominatesSnapshot:
    """Tests proving trajectory dominates snapshot-centric patterns."""

    def test_transition_organism_state_maintains_trajectory_semantics(self):
        """State transitions must be trajectory-aware."""
        state = OrganismState(
            state_id="state-0",
            timestamp=utc_now_iso(),
            active_regime="unknown",
            episode_count=0,
        )

        episode_result = {
            "episode": {
                "result": {"relation_kind": "support"},
                "context": {"observation": {"alarm": False}},
            },
            "belief_state": {
                "posterior": {
                    "policy_confidence": 0.8,
                    "causal_support_confidence": 0.9,
                }
            },
            "certification": {"verdict": "certified"},
        }

        new_state = transition_organism_state(
            current=state,
            episode_result=episode_result,
            regime="baseline",
            new_state_id="state-1",
            timestamp=utc_now_iso(),
        )

        # New state must reflect transition
        assert new_state.state_id == "state-1"
        assert new_state.episode_count == 1
        assert new_state.active_regime == "baseline"

        # Belief must be updated from episode
        assert new_state.belief.alarm_probability < 0.5  # No alarm
        assert new_state.belief.intervention_efficacy == 0.8

    def test_constitutional_flow_penalizes_violations(self):
        """Constitutional flow score must degrade with violations."""
        traj = OrganismTrajectory(organism_id="test-org")
        constitution = OrganismConstitution()

        # Add state with hard violations
        bad_state = OrganismState(
            state_id="bad-state",
            timestamp=utc_now_iso(),
            active_regime="baseline",
            belief=OrganismState().belief._replace(
                memory_purity_estimate=0.1,  # Below threshold
                trace_integrity_confidence=0.1,  # Below threshold
            ),
        )

        validation = constitution.validate(bad_state)
        initial_flow = traj.constitutional_flow_score

        traj.append_point(
            state=bad_state,
            regime="baseline",
            episode_id="ep-bad",
            timestamp=utc_now_iso(),
            constitutional_validation=validation,
        )

        # Flow score MUST degrade
        assert traj.constitutional_flow_score < initial_flow
        assert validation.hard_violation_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
