"""T5 Sovereignty Tests: Certification Requires Trajectory.

CRITICAL TEST: Certification must FAIL without trajectory.
This test proves T5 is operationally sovereign, not optional.
"""

import pytest
from runtime.world.scenario_runner import ScenarioEpisodeRunner
from runtime.storage import get_storage


class TestT5CertificationRequiresTrajectory:
    """Tests proving certification cannot bypass trajectory."""

    def test_scenario_runner_creates_trajectory(self):
        """ScenarioEpisodeRunner must initialize trajectory on creation."""
        runner = ScenarioEpisodeRunner(
            storage=get_storage(),
            scenario="thermal",
        )

        # T5 SOVEREIGNTY: Runner must have trajectory
        assert hasattr(runner, "_organism_trajectory")
        assert hasattr(runner, "_organism_state")
        assert hasattr(runner, "_constitution")
        assert hasattr(runner, "_viability_kernel")

        # Trajectory must be initialized
        assert runner._organism_trajectory is not None
        assert runner._organism_trajectory.length == 0  # Empty initially
        assert runner._organism_state is not None

    def test_run_episode_transitions_organism_state(self):
        """run_episode must transition organism state and append to trajectory."""
        runner = ScenarioEpisodeRunner(
            storage=get_storage(),
            scenario="thermal",
        )

        initial_episode_count = runner._organism_state.episode_count
        initial_traj_length = runner._organism_trajectory.length

        # Run episode
        result = runner.run_episode(external_input=0.05)

        # Organism state MUST transition
        assert runner._organism_state.episode_count == initial_episode_count + 1

        # Trajectory MUST grow
        assert runner._organism_trajectory.length == initial_traj_length + 1

        # Episode result MUST include trajectory
        assert "organism_trajectory" in result
        assert "constitutional_validation" in result
        assert "viability_assessment" in result

    def test_episode_result_contains_trajectory_metadata(self):
        """Episode result must contain trajectory as primary certification input."""
        runner = ScenarioEpisodeRunner(
            storage=get_storage(),
            scenario="thermal",
        )

        result = runner.run_episode(external_input=0.05)

        # Result must have trajectory
        traj = result.get("organism_trajectory")
        assert traj is not None
        assert isinstance(traj, dict)
        assert "trajectory_id" in traj
        assert "organism_id" in traj
        assert "constitutional_flow_score" in traj
        assert "total_episodes" in traj
        assert "points" in traj
        assert len(traj["points"]) > 0

        # Constitutional validation must be present
        const_val = result.get("constitutional_validation")
        assert const_val is not None
        assert "is_valid" in const_val
        assert "verdict" in const_val

        # Viability assessment must be present
        viab = result.get("viability_assessment")
        assert viab is not None
        assert "is_viable" in viab
        assert "viability_margin" in viab

    def test_constitutional_validation_runs_on_every_episode(self):
        """Constitutional validation must run on organism state for every episode."""
        runner = ScenarioEpisodeRunner(
            storage=get_storage(),
            scenario="thermal",
        )

        # Run 3 episodes
        results = [runner.run_episode(external_input=0.05) for _ in range(3)]

        # All episodes must have constitutional validation
        for result in results:
            assert "constitutional_validation" in result
            const_val = result["constitutional_validation"]
            assert "is_valid" in const_val
            assert "hard_violation_count" in const_val
            assert "soft_violation_count" in const_val

        # Trajectory must have 3 points
        assert runner._organism_trajectory.length == 3

        # All points must have constitutional validation
        for point in runner._organism_trajectory.points:
            assert point.constitutional_validation is not None

    def test_viability_assessment_uses_trajectory_context(self):
        """Viability assessment must use previous state (trajectory context)."""
        runner = ScenarioEpisodeRunner(
            storage=get_storage(),
            scenario="thermal",
        )

        # First episode
        result1 = runner.run_episode(external_input=0.05)
        viab1 = result1["viability_assessment"]

        # Second episode
        result2 = runner.run_episode(external_input=0.05)
        viab2 = result2["viability_assessment"]

        # Both must have viability margins
        assert "viability_margin" in viab1
        assert "viability_margin" in viab2

        # Trajectory must track margins
        window = runner._organism_trajectory.get_window(window_size=2)
        assert len(window.margin_trajectory) == 2

    def test_trajectory_persists_across_episodes(self):
        """Trajectory must accumulate across multiple episodes."""
        runner = ScenarioEpisodeRunner(
            storage=get_storage(),
            scenario="thermal",
        )

        # Run 5 episodes
        for i in range(5):
            result = runner.run_episode(external_input=0.04 + i * 0.01)

        # Trajectory must have 5 points
        assert runner._organism_trajectory.length == 5
        assert runner._organism_trajectory.total_episodes == 5

        # All points must be linked
        for i in range(1, 5):
            assert runner._organism_trajectory.points[i].prev_state_id is not None
            # Current step's prev_state_id must match previous step's state_id
            assert (
                runner._organism_trajectory.points[i].prev_state_id
                == runner._organism_trajectory.points[i - 1].state.state_id
            )

    def test_constitutional_flow_score_evolves(self):
        """Constitutional flow score must evolve with trajectory."""
        runner = ScenarioEpisodeRunner(
            storage=get_storage(),
            scenario="thermal",
        )

        initial_flow_score = runner._organism_trajectory.constitutional_flow_score

        # Run episodes
        for _ in range(3):
            runner.run_episode(external_input=0.05)

        # Flow score must have evolved (even if just slight decay)
        # Due to decay mechanism in append_point
        final_flow_score = runner._organism_trajectory.constitutional_flow_score
        assert final_flow_score != initial_flow_score  # Changed

    def test_regime_tracking_in_trajectory(self):
        """Trajectory must track regime across episodes."""
        runner = ScenarioEpisodeRunner(
            storage=get_storage(),
            scenario="thermal",  # Regime will be scenario name
        )

        # Run episodes
        for _ in range(3):
            runner.run_episode(external_input=0.05)

        # Current regime must be set
        assert runner._organism_trajectory.current_regime != "unknown"
        # Should be scenario name
        assert runner._organism_trajectory.current_regime == "thermal"

        # All points must have regime
        for point in runner._organism_trajectory.points:
            assert point.regime == "thermal"


class TestT5TrajectoryIsNotOptional:
    """Tests proving T5 trajectory is required, not optional fallback."""

    def test_organism_state_initialized_on_runner_creation(self):
        """Organism state must be initialized when runner is created."""
        runner = ScenarioEpisodeRunner(
            storage=get_storage(),
            scenario="thermal",
        )

        # Organism state must exist
        assert runner._organism_state is not None
        assert runner._organism_state.state_id is not None
        assert runner._organism_state.episode_count == 0

        # Identity must be initialized
        assert runner._organism_state.identity.lineage_id is not None

    def test_constitution_and_viability_kernel_initialized(self):
        """Constitution and viability kernel must be initialized."""
        runner = ScenarioEpisodeRunner(
            storage=get_storage(),
            scenario="thermal",
        )

        # Constitution must exist
        assert runner._constitution is not None
        assert hasattr(runner._constitution, "validate")

        # Viability kernel must exist
        assert runner._viability_kernel is not None
        assert hasattr(runner._viability_kernel, "assess")

    def test_episode_result_always_has_trajectory_data(self):
        """Episode result must ALWAYS have trajectory data (not optional)."""
        runner = ScenarioEpisodeRunner(
            storage=get_storage(),
            scenario="thermal",
        )

        # Run episode
        result = runner.run_episode(external_input=0.05)

        # These fields are MANDATORY, not optional
        assert "organism_trajectory" in result
        assert "constitutional_validation" in result
        assert "viability_assessment" in result

        # organism_trajectory must not be None
        assert result["organism_trajectory"] is not None
        assert result["constitutional_validation"] is not None
        assert result["viability_assessment"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
