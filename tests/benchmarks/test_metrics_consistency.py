"""Test: Metrics Consistency

Verifica que el análisis y métricas no dependan de campos removidos.
"""

import pytest
from pathlib import Path
import re


class TestMetricsConsistency:
    """Valida coherencia arquitectónica entre benchmark, métricas y análisis."""

    def test_no_removed_metrics_in_analysis_report(self):
        """analysis_report.py NO debe referenciar métricas removidas."""

        # Métricas removidas que no deben aparecer
        removed_metrics = [
            'cierre_rate',  # Reemplazado por success_rate
            'continuity_score',  # Reemplazado por viability_margin
            'memory_pressure_mb',  # No instrumentada
            'factual_cf_divergence',  # Requiere trace multi-step
            'world_level_transitions',  # Requiere trace multi-step
            'spatial_coherence_index',  # Requiere evolución temporal
            'counterfactual_overhead_ratio',  # No hay timing separado
            'scheduler_cpu_time_ms',  # No hay timing separado
        ]

        analysis_file = Path(__file__).parent / "analysis_report.py"
        content = analysis_file.read_text()

        # Remover comentarios y docstrings para evitar falsos positivos
        # Solo buscar en código activo
        lines = content.split('\n')
        code_lines = []
        in_docstring = False

        for line in lines:
            stripped = line.strip()

            # Detectar docstrings
            if '"""' in stripped or "'''" in stripped:
                in_docstring = not in_docstring
                continue

            # Skip comentarios y docstrings
            if in_docstring or stripped.startswith('#'):
                continue

            code_lines.append(line)

        code_content = '\n'.join(code_lines)

        # Buscar métricas removidas en código activo
        violations = []
        for metric in removed_metrics:
            # Buscar como string literal (entre comillas)
            pattern = f"['\"]({metric})['\"]"
            matches = re.findall(pattern, code_content)

            if matches:
                violations.append(f"Found removed metric '{metric}' in analysis_report.py")

        assert not violations, (
            f"analysis_report.py references removed metrics:\n" +
            "\n".join(f"  - {v}" for v in violations) +
            "\n\nUse proxy metrics instead (success_rate, viability_margin)"
        )

    def test_analysis_uses_correct_proxy_metrics(self):
        """analysis_report.py debe usar métricas proxy correctas."""

        analysis_file = Path(__file__).parent / "analysis_report.py"
        content = analysis_file.read_text()

        # Métricas proxy que DEBEN estar presentes
        required_proxies = [
            'success_rate',  # Proxy de cierre_rate
            'viability_margin',  # Proxy de continuity_score
        ]

        missing = []
        for proxy in required_proxies:
            pattern = f"['\"]({proxy})['\"]"
            if not re.search(pattern, content):
                missing.append(proxy)

        assert not missing, (
            f"analysis_report.py is missing required proxy metrics:\n" +
            "\n".join(f"  - {m}" for m in missing)
        )

    def test_benchmark_runner_provides_proxy_metrics(self):
        """benchmark_runner.py debe proveer success_rate en to_dict()."""

        runner_file = Path(__file__).parent / "benchmark_runner.py"
        content = runner_file.read_text()

        # Verificar que to_dict() incluye success_rate
        assert "'success_rate':" in content or '"success_rate":' in content, (
            "benchmark_runner.py EpisodeResult.to_dict() must include 'success_rate' proxy"
        )

    def test_ivc_r_receives_proxy_inputs(self):
        """benchmark_runner.py debe inyectar proxies a IVC-R."""

        runner_file = Path(__file__).parent / "benchmark_runner.py"
        content = runner_file.read_text()

        # Verificar que se inyectan los proxies antes de calcular IVC-R
        # Buscar el bloque donde se prepara ivc_r_input
        assert 'ivc_r_input' in content, (
            "benchmark_runner.py must prepare ivc_r_input dict"
        )

        # Verificar que se documentan como proxies
        assert 'cierre_rate' in content or 'continuity_score' in content, (
            "benchmark_runner.py must document proxy mappings for IVC-R:\n"
            "  - cierre_rate = 1.0 if success else 0.0\n"
            "  - continuity_score = viability_margin"
        )

    def test_removed_metrics_documented(self):
        """PHASE1_COMPLETE.md debe documentar métricas removidas."""

        doc_file = Path(__file__).parent.parent.parent / "PHASE1_COMPLETE.md"
        if not doc_file.exists():
            pytest.skip("PHASE1_COMPLETE.md not found")

        content = doc_file.read_text()

        # Verificar que se documenta qué fue removido
        removed_keywords = [
            'REMOVIDA',
            'removed',
            'proxy',
        ]

        found_any = any(kw in content for kw in removed_keywords)

        assert found_any, (
            "PHASE1_COMPLETE.md should document removed metrics and proxies"
        )
