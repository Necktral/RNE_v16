"""Tests para el benchmark matricial de transición NxN."""

from runtime.reality.transition_matrix import (
    GATE_PROFILES,
    _evaluate_matrix_gate,
    run_transition_matrix_benchmark,
)


class TestTransitionMatrixBenchmark:
    def test_generates_2x2_matrix(self):
        """Genera matriz 2x2 con dos escenarios explícitos (N²=4 celdas).

        Antes dependía del tamaño global del registro (eran 2 escenarios); al
        añadirse un 3.º generaba 3x3=9. El test ahora fija sus propios escenarios.
        """
        result = run_transition_matrix_benchmark(
            scenarios=["thermal_homeostasis", "resource_management"],
            warmup_episodes=1,
            probe_episodes=1,
            closure_profile="adaptive_min",
        )
        assert "closure_matrix" in result
        assert "continuity_tensor" in result
        assert "memory_purity_matrix" in result
        assert "transfer_verdict_matrix" in result
        assert "cell_reports" in result
        assert len(result["cell_reports"]) == 4  # 2x2

    def test_produces_specific_artifact(self):
        """Produce artifact de tipo transition_matrix_report."""
        result = run_transition_matrix_benchmark(
            warmup_episodes=1,
            probe_episodes=1,
            closure_profile="adaptive_min",
        )
        assert result["artifact"]["kind"] == "transition_matrix_report"

    def test_each_cell_has_complete_metrics(self):
        """Cada celda trae métricas completas."""
        result = run_transition_matrix_benchmark(
            warmup_episodes=1,
            probe_episodes=1,
            closure_profile="adaptive_min",
        )
        for cell in result["cell_reports"]:
            assert "source_scenario" in cell
            assert "target_scenario" in cell
            assert "compatibility_class" in cell
            assert "closure_rate" in cell
            assert "mean_continuity_composite" in cell
            assert "mean_memory_purity" in cell
            assert "transfer_verdict" in cell
            assert "collapse_count" in cell

    def test_gate_evaluates_on_cells(self):
        """Gate transition_matrix_ci se evalúa sobre celdas."""
        gate = GATE_PROFILES["transition_matrix_ci"]
        # 4 cells that pass
        cells = [
            {
                "closure_rate": 0.95,
                "mean_continuity_composite": 0.60,
                "transfer_verdict": "certified_transfer_safe",
                "collapse_count": 0,
            }
            for _ in range(4)
        ]
        assert _evaluate_matrix_gate(gate_config=gate, cell_reports=cells) is True

    def test_gate_fails_insufficient_cells(self):
        gate = GATE_PROFILES["transition_matrix_ci"]
        cells = [
            {
                "closure_rate": 0.95,
                "mean_continuity_composite": 0.60,
                "transfer_verdict": "certified_transfer_safe",
                "collapse_count": 0,
            }
        ]
        assert _evaluate_matrix_gate(gate_config=gate, cell_reports=cells) is False

    def test_gate_fails_low_closure(self):
        gate = GATE_PROFILES["transition_matrix_ci"]
        cells = [
            {
                "closure_rate": 0.50,
                "mean_continuity_composite": 0.60,
                "transfer_verdict": "certified_transfer_safe",
                "collapse_count": 0,
            }
            for _ in range(4)
        ]
        assert _evaluate_matrix_gate(gate_config=gate, cell_reports=cells) is False

    def test_gate_fails_too_many_collapses(self):
        gate = GATE_PROFILES["transition_matrix_ci"]
        cells = [
            {
                "closure_rate": 0.95,
                "mean_continuity_composite": 0.60,
                "transfer_verdict": "certified_transfer_safe",
                "collapse_count": 1,
            }
            for _ in range(4)
        ]
        assert _evaluate_matrix_gate(gate_config=gate, cell_reports=cells) is False

    def test_diagonal_cells_equivalent(self):
        """Las celdas diagonales (A→A) deben ser equivalent."""
        result = run_transition_matrix_benchmark(
            warmup_episodes=1,
            probe_episodes=1,
            closure_profile="adaptive_min",
        )
        for cell in result["cell_reports"]:
            if cell["source_scenario"] == cell["target_scenario"]:
                assert cell["compatibility_class"] == "equivalent"

    def test_specific_scenarios_subset(self):
        """Se puede ejecutar con un subconjunto de escenarios."""
        result = run_transition_matrix_benchmark(
            scenarios=["thermal_homeostasis"],
            warmup_episodes=1,
            probe_episodes=1,
            closure_profile="adaptive_min",
        )
        assert len(result["cell_reports"]) == 1  # 1x1
        assert result["cell_reports"][0]["source_scenario"] == "thermal_homeostasis"
