"""Analysis and Reporting: Statistical comparison and report generation.

Genera análisis estadístico comparativo entre 1x1 y 5x5:
- Tests de hipótesis (Mann-Whitney U)
- Bootstrap para intervalos de confianza
- Tamaños de efecto (Cohen's d)
- Tablas comparativas
- Cálculo de ganancia cognitiva neta
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
import math
import random
from collections import defaultdict


class BenchmarkAnalyzer:
    """Analiza y compara resultados de benchmarks."""

    def __init__(self):
        self.data_1x1 = []
        self.data_5x5 = []

    def load_results(self, result_dir_1x1: Path, result_dir_5x5: Path):
        """Carga resultados de ambos escenarios.

        Args:
            result_dir_1x1: Directorio con episodes.jsonl del 1x1.
            result_dir_5x5: Directorio con episodes.jsonl del 5x5.
        """
        self.data_1x1 = self._load_jsonl(result_dir_1x1 / "episodes.jsonl")
        self.data_5x5 = self._load_jsonl(result_dir_5x5 / "episodes.jsonl")

        print(f"Loaded {len(self.data_1x1)} episodes from 1x1")
        print(f"Loaded {len(self.data_5x5)} episodes from 5x5")

    def _load_jsonl(self, filepath: Path) -> List[Dict[str, Any]]:
        """Carga archivo JSONL."""
        data = []
        if not filepath.exists():
            return data

        with open(filepath) as f:
            for line in f:
                data.append(json.loads(line))

        return data

    def compute_statistical_comparison(self) -> Dict[str, Any]:
        """Ejecuta comparación estadística completa.

        CORREGIDO: Usa solo métricas reales del contrato.

        Returns:
            Diccionario con resultados de comparación.
        """
        comparison = {}

        # Métricas a comparar (solo las que existen en el contrato real)
        metrics = [
            'success_rate',  # Proxy de cierre_rate (desde outcome=='success')
            'viability_margin',  # Proxy de continuity_score
            'intervention_precision',
            'proposition_diversity',
            'spatial_information_usage',
            'wall_time_ms',
            'artifact_size_bytes',
            'reasoning_trace_length',
            'ivc_r',
            'family_mix_entropy',
            'family_optional_count',
            'family_optional_used_flag',
            'backbone_floor_satisfied_flag',
            'sequence_validation_fail_flag',
            'fallback_to_safe_sequence_flag',
            'optional_displacement_flag',
            'closure_break_flag',
        ]

        optional_metrics = [
            'scale_selection_accuracy',
            'upgrade_regret',
            'downgrade_regret',
            'missed_upgrade_regret',
            'keep_scale_rate',
            'upgrade_rate',
            'downgrade_rate',
            'probe_rate',
            'probe_commit_rate',
            'oscillation_rate',
            'probe_value_rate',
            'mean_resolution_cost',
            'resolution_efficiency',
            'cross_scale_memory_contamination_rate',
            'vram_headroom_mean',
            'vram_peak_ratio',
            'vram_recompute_avoided',
            'vram_enabled_probe_success_rate',
            'vram_efficiency_after_intelligence',
            'regime_probe_rate',
        ]

        for metric in optional_metrics:
            if self._metric_available_in_both(metric):
                metrics.append(metric)

        for metric in metrics:
            comparison[metric] = self._compare_metric(metric)

        if self._metric_available_in_both('family_optional_used_flag'):
            comparison['optional_family_usage_rate'] = self._compare_metric('family_optional_used_flag')

        comparison['family_specific_activation_counts'] = self._compare_family_activation_counts()

        family_dict_metrics = [
            'family_contribution_proxy',
            'family_delta_ivc_r',
            'family_delta_intervention_precision',
            'family_delta_viability_margin',
            'family_delta_reasoning_trace_length',
            'family_delta_success_rate',
            'family_delta_spatial_information_usage',
        ]
        for metric in family_dict_metrics:
            if self._dict_metric_available_in_both(metric):
                comparison[metric] = self._compare_family_dict_metric(metric)

        # Ganancia cognitiva neta
        comparison['net_cognitive_gain'] = self._compute_net_cognitive_gain()

        # Failure analysis
        comparison['failure_analysis'] = self._compare_failures()

        return comparison

    def _metric_available_in_both(self, metric: str) -> bool:
        has_1 = any(metric in ep and isinstance(ep.get(metric), (int, float)) for ep in self.data_1x1)
        has_5 = any(metric in ep and isinstance(ep.get(metric), (int, float)) for ep in self.data_5x5)
        return has_1 and has_5

    def _dict_metric_available_in_both(self, metric: str) -> bool:
        has_1 = any(metric in ep and isinstance(ep.get(metric), dict) and ep.get(metric) for ep in self.data_1x1)
        has_5 = any(metric in ep and isinstance(ep.get(metric), dict) and ep.get(metric) for ep in self.data_5x5)
        return has_1 and has_5

    def _compare_family_dict_metric(self, metric: str) -> Dict[str, Any]:
        families = set()
        for row in self.data_1x1:
            payload = row.get(metric)
            if isinstance(payload, dict):
                families.update(str(k) for k in payload.keys())
        for row in self.data_5x5:
            payload = row.get(metric)
            if isinstance(payload, dict):
                families.update(str(k) for k in payload.keys())

        result: Dict[str, Any] = {}
        for family in sorted(families):
            values_1x1 = []
            for row in self.data_1x1:
                payload = row.get(metric)
                if isinstance(payload, dict):
                    values_1x1.append(float(payload.get(family, 0.0) or 0.0))
            values_5x5 = []
            for row in self.data_5x5:
                payload = row.get(metric)
                if isinstance(payload, dict):
                    values_5x5.append(float(payload.get(family, 0.0) or 0.0))
            if not values_1x1 or not values_5x5:
                continue
            mean_1x1 = sum(values_1x1) / len(values_1x1)
            mean_5x5 = sum(values_5x5) / len(values_5x5)
            result[family] = {
                "mean_1x1": mean_1x1,
                "mean_5x5": mean_5x5,
                "delta": mean_5x5 - mean_1x1,
            }
        return result

    def _compare_family_activation_counts(self) -> Dict[str, Any]:
        def _aggregate(data: List[Dict[str, Any]]) -> Dict[str, int]:
            counts: Dict[str, int] = defaultdict(int)
            for row in data:
                payload = row.get("family_activation_counts")
                if not isinstance(payload, dict):
                    continue
                for family, value in payload.items():
                    try:
                        counts[str(family)] += int(float(value))
                    except (TypeError, ValueError):
                        continue
            return dict(sorted(counts.items()))

        counts_1x1 = _aggregate(self.data_1x1)
        counts_5x5 = _aggregate(self.data_5x5)
        families = set(counts_1x1.keys()) | set(counts_5x5.keys())
        delta = {
            family: int(counts_5x5.get(family, 0)) - int(counts_1x1.get(family, 0))
            for family in sorted(families)
        }
        return {
            "counts_1x1": counts_1x1,
            "counts_5x5": counts_5x5,
            "delta": delta,
        }

    def _compare_metric(self, metric: str) -> Dict[str, Any]:
        """Compara una métrica específica entre 1x1 y 5x5.

        Args:
            metric: Nombre de la métrica.

        Returns:
            Diccionario con estadísticas comparativas.
        """
        values_1x1 = [float(ep.get(metric, 0.0)) for ep in self.data_1x1 if isinstance(ep.get(metric), (int, float))]
        values_5x5 = [float(ep.get(metric, 0.0)) for ep in self.data_5x5 if isinstance(ep.get(metric), (int, float))]

        if not values_1x1 or not values_5x5:
            return {'error': 'Insufficient data'}

        # Estadísticas descriptivas
        mean_1x1 = sum(values_1x1) / len(values_1x1)
        mean_5x5 = sum(values_5x5) / len(values_5x5)

        std_1x1 = self._std(values_1x1)
        std_5x5 = self._std(values_5x5)

        # Delta
        delta = mean_5x5 - mean_1x1
        delta_pct = (delta / mean_1x1 * 100) if mean_1x1 != 0 else 0.0

        # Cohen's d
        cohens_d = self._cohens_d(values_1x1, values_5x5)

        # Mann-Whitney U test (simplificado)
        p_value = self._mann_whitney_u(values_1x1, values_5x5)

        # Bootstrap CI para delta
        ci_lower, ci_upper = self._bootstrap_ci_delta(values_1x1, values_5x5)

        return {
            'metric': metric,
            'mean_1x1': mean_1x1,
            'mean_5x5': mean_5x5,
            'std_1x1': std_1x1,
            'std_5x5': std_5x5,
            'delta': delta,
            'delta_pct': delta_pct,
            'cohens_d': cohens_d,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
        }

    def _std(self, values: List[float]) -> float:
        """Calcula desviación estándar."""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)

    def _cohens_d(self, values_1: List[float], values_2: List[float]) -> float:
        """Calcula Cohen's d (tamaño del efecto).

        Formula: d = (mean2 - mean1) / pooled_std
        """
        mean_1 = sum(values_1) / len(values_1)
        mean_2 = sum(values_2) / len(values_2)

        std_1 = self._std(values_1)
        std_2 = self._std(values_2)

        n1 = len(values_1)
        n2 = len(values_2)
        dof = n1 + n2 - 2

        # Pooled standard deviation
        if dof <= 0:
            return 0.0

        pooled_var = ((n1 - 1) * std_1**2 + (n2 - 1) * std_2**2) / dof
        pooled_std = math.sqrt(pooled_var)

        if pooled_std == 0:
            return 0.0

        d = (mean_2 - mean_1) / pooled_std
        return d

    def _mann_whitney_u(self, values_1: List[float], values_2: List[float]) -> float:
        """Aproximación simplificada de Mann-Whitney U test.

        Retorna p-value aproximado basado en overlapping de distribuciones.
        """
        # Implementación simplificada: usar overlapping como proxy
        # Para implementación completa, usar scipy.stats.mannwhitneyu

        # Concatenar y ordenar
        combined = [(v, 1) for v in values_1] + [(v, 2) for v in values_2]
        combined.sort()

        # Contar inversiones
        n1 = len(values_1)
        n2 = len(values_2)

        rank_sum_1 = 0
        for i, (v, group) in enumerate(combined):
            if group == 1:
                rank_sum_1 += (i + 1)

        U1 = rank_sum_1 - n1 * (n1 + 1) / 2

        # Z-score aproximado
        mu = n1 * n2 / 2
        sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

        if sigma == 0:
            return 1.0

        z = abs((U1 - mu) / sigma)

        # P-value aproximado (two-tailed)
        # Para z>3, p muy pequeño
        if z > 3:
            p_value = 0.001
        elif z > 2:
            p_value = 0.05
        elif z > 1:
            p_value = 0.2
        else:
            p_value = 0.5

        return p_value

    def _bootstrap_ci_delta(
        self,
        values_1: List[float],
        values_2: List[float],
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """Calcula IC bootstrap para diferencia de medias.

        Args:
            values_1: Valores del grupo 1.
            values_2: Valores del grupo 2.
            n_bootstrap: Número de resamples.
            confidence: Nivel de confianza.

        Returns:
            (ci_lower, ci_upper)
        """
        deltas = []

        for _ in range(n_bootstrap):
            sample_1 = random.choices(values_1, k=len(values_1))
            sample_2 = random.choices(values_2, k=len(values_2))

            mean_1 = sum(sample_1) / len(sample_1)
            mean_2 = sum(sample_2) / len(sample_2)

            deltas.append(mean_2 - mean_1)

        deltas.sort()

        alpha = 1.0 - confidence
        lower_idx = int(n_bootstrap * (alpha / 2))
        upper_idx = int(n_bootstrap * (1 - alpha / 2))

        return deltas[lower_idx], deltas[upper_idx]

    def _compute_net_cognitive_gain(self) -> Dict[str, Any]:
        """Calcula ganancia cognitiva neta según fórmula de especificación.

        CORREGIDO: No usa memory_pressure_mb (no instrumentada).

        Formula:
        Ganancia_Neta = (IVC-R_5x5 - IVC-R_1x1) - Penalización_Costo

        Returns:
            Diccionario con ganancia neta y descomposición.
        """
        # Promedios
        ivc_r_1x1 = self._mean_metric(self.data_1x1, 'ivc_r')
        ivc_r_5x5 = self._mean_metric(self.data_5x5, 'ivc_r')

        wall_time_1x1 = self._mean_metric(self.data_1x1, 'wall_time_ms')
        wall_time_5x5 = self._mean_metric(self.data_5x5, 'wall_time_ms')

        artifact_1x1 = self._mean_metric(self.data_1x1, 'artifact_size_bytes')
        artifact_5x5 = self._mean_metric(self.data_5x5, 'artifact_size_bytes')

        # Ratios
        wall_time_ratio = wall_time_5x5 / wall_time_1x1 if wall_time_1x1 > 0 else 1.0
        artifact_ratio = artifact_5x5 / artifact_1x1 if artifact_1x1 > 0 else 1.0

        # Penalización de costo (sin memory_ratio - no instrumentada)
        def normalize(x, max_val):
            return min(x, max_val) / max_val

        penalizacion_costo = (
            0.15 * normalize(wall_time_ratio - 1.0, 1.0) +  # 0.10 -> 0.15 (redistribuir peso)
            0.05 * normalize(artifact_ratio - 1.0, 2.0)
        )

        # Ganancia bruta
        ganancia_bruta = ivc_r_5x5 - ivc_r_1x1

        # Ganancia neta
        ganancia_neta = ganancia_bruta - penalizacion_costo

        return {
            'ganancia_bruta': ganancia_bruta,
            'penalizacion_costo': penalizacion_costo,
            'ganancia_neta': ganancia_neta,
            'ivc_r_1x1': ivc_r_1x1,
            'ivc_r_5x5': ivc_r_5x5,
            'wall_time_ratio': wall_time_ratio,
            'artifact_ratio': artifact_ratio,
        }

    def _mean_metric(self, data: List[Dict[str, Any]], metric: str) -> float:
        """Calcula media de una métrica."""
        values = [ep.get(metric, 0.0) for ep in data if metric in ep]
        return sum(values) / len(values) if values else 0.0

    def _compare_failures(self) -> Dict[str, Any]:
        """Compara distribución de fallos entre 1x1 y 5x5."""
        failures_1x1 = [ep for ep in self.data_1x1 if ep.get('failure_primary') is not None]
        failures_5x5 = [ep for ep in self.data_5x5 if ep.get('failure_primary') is not None]

        failure_rate_1x1 = len(failures_1x1) / len(self.data_1x1) if self.data_1x1 else 0.0
        failure_rate_5x5 = len(failures_5x5) / len(self.data_5x5) if self.data_5x5 else 0.0

        # Distribución por tipo
        dist_1x1 = defaultdict(int)
        dist_5x5 = defaultdict(int)

        for ep in failures_1x1:
            dist_1x1[ep['failure_primary']] += 1

        for ep in failures_5x5:
            dist_5x5[ep['failure_primary']] += 1

        return {
            'failure_rate_1x1': failure_rate_1x1,
            'failure_rate_5x5': failure_rate_5x5,
            'total_failures_1x1': len(failures_1x1),
            'total_failures_5x5': len(failures_5x5),
            'distribution_1x1': dict(dist_1x1),
            'distribution_5x5': dict(dist_5x5),
        }

    def generate_comparison_table_markdown(self, comparison: Dict[str, Any]) -> str:
        """Genera tabla comparativa en formato Markdown.

        Args:
            comparison: Resultado de compute_statistical_comparison.

        Returns:
            String con tabla Markdown.
        """
        lines = []
        lines.append("## Comparación Agregada 1x1 vs 5x5\n")
        lines.append("| Métrica | 1x1 | 5x5 | Δ | Δ% | Cohen's d | p-value |")
        lines.append("|---------|-----|-----|---|----|-----------|---------|\n")

        metrics_display = {
            'success_rate': 'Success Rate (proxy: cierre_rate)',
            'viability_margin': 'Viability Margin (proxy: continuity_score)',
            'intervention_precision': 'Intervention Precision',
            'proposition_diversity': 'Proposition Diversity',
            'spatial_information_usage': 'Spatial Information Usage',
            'wall_time_ms': 'Wall Time (ms)',
            'artifact_size_bytes': 'Artifact Size (bytes)',
            'reasoning_trace_length': 'Reasoning Trace Length',
            'family_mix_entropy': 'Family Mix Entropy',
            'family_optional_used_flag': 'Optional Family Usage Rate',
            'ivc_r': '**IVC-R**',
        }

        for metric_key, display_name in metrics_display.items():
            if metric_key not in comparison:
                continue

            stats = comparison[metric_key]

            if 'error' in stats:
                continue

            mean_1x1 = stats['mean_1x1']
            mean_5x5 = stats['mean_5x5']
            delta = stats['delta']
            delta_pct = stats['delta_pct']
            cohens_d = stats['cohens_d']
            p_value = stats['p_value']

            # Formatear según tipo de métrica
            if 'bytes' in metric_key:
                mean_1x1_str = f"{mean_1x1:.0f}"
                mean_5x5_str = f"{mean_5x5:.0f}"
                delta_str = f"{delta:+.0f}"
            elif 'ms' in metric_key:
                mean_1x1_str = f"{mean_1x1:.1f}"
                mean_5x5_str = f"{mean_5x5:.1f}"
                delta_str = f"{delta:+.1f}"
            else:
                mean_1x1_str = f"{mean_1x1:.3f}"
                mean_5x5_str = f"{mean_5x5:.3f}"
                delta_str = f"{delta:+.3f}"

            delta_pct_str = f"{delta_pct:+.1f}%"
            cohens_d_str = f"{cohens_d:.2f}"
            p_value_str = f"{p_value:.3f}" if p_value >= 0.001 else "<0.001"

            line = f"| {display_name} | {mean_1x1_str} | {mean_5x5_str} | {delta_str} | {delta_pct_str} | {cohens_d_str} | {p_value_str} |"
            lines.append(line)

        # Ganancia neta
        if 'net_cognitive_gain' in comparison:
            ncg = comparison['net_cognitive_gain']
            lines.append(f"\n### Ganancia Cognitiva Neta\n")
            lines.append(f"- **Ganancia Bruta (Δ IVC-R)**: {ncg['ganancia_bruta']:+.3f}")
            lines.append(f"- **Penalización de Costo**: {ncg['penalizacion_costo']:.3f}")
            lines.append(f"- **Ganancia Neta**: {ncg['ganancia_neta']:+.3f}\n")

        return "\n".join(lines)

    def generate_final_report(
        self,
        comparison: Dict[str, Any],
        output_file: Path,
    ):
        """Genera reporte final completo en Markdown.

        Args:
            comparison: Resultado de análisis estadístico.
            output_file: Archivo de salida.
        """
        lines = []

        lines.append("# Phase 1 Benchmark Report: 1x1 vs 5x5\n")
        lines.append(f"**Fecha**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append("---\n")

        # Tabla comparativa
        lines.append(self.generate_comparison_table_markdown(comparison))

        # Análisis de fallos
        if 'failure_analysis' in comparison:
            fa = comparison['failure_analysis']
            lines.append("\n## Distribución de Fallos\n")
            lines.append(f"- **1x1**: {fa['total_failures_1x1']}/{len(self.data_1x1)} ({fa['failure_rate_1x1']:.1%})")
            lines.append(f"- **5x5**: {fa['total_failures_5x5']}/{len(self.data_5x5)} ({fa['failure_rate_5x5']:.1%})\n")

        # Dictamen
        lines.append("\n## Dictamen Preliminar\n")
        lines.append(self._generate_verdict(comparison))

        # Escribir archivo
        with open(output_file, 'w') as f:
            f.write("\n".join(lines))

        print(f"\n📄 Reporte generado en: {output_file}")

    def _generate_verdict(self, comparison: Dict[str, Any]) -> str:
        """Genera dictamen basado en criterios de especificación.

        CORREGIDO: Usa success_rate en lugar de cierre_rate.
        """
        ncg = comparison.get('net_cognitive_gain', {})
        ganancia_neta = ncg.get('ganancia_neta', 0.0)

        precision_stats = comparison.get('intervention_precision', {})
        precision_delta = precision_stats.get('delta', 0.0)

        success_stats = comparison.get('success_rate', {})
        success_delta = success_stats.get('delta', 0.0)

        wall_time_ratio = ncg.get('wall_time_ratio', 1.0)
        artifact_ratio = ncg.get('artifact_ratio', 1.0)

        lines = []

        if ganancia_neta > 0.08:
            lines.append("✅ **VALOR ARQUITECTÓNICO DEMOSTRADO**")
            lines.append(f"- Ganancia neta: {ganancia_neta:+.3f} (>0.08 requerido)")
        elif ganancia_neta > 0.05:
            lines.append("⚠️ **AVANCE PARCIAL**")
            lines.append(f"- Ganancia neta: {ganancia_neta:+.3f} (marginal)")
        elif ganancia_neta > 0.0:
            lines.append("⚠️ **VALOR MARGINAL BAJO**")
            lines.append(f"- Ganancia neta: {ganancia_neta:+.3f} (riesgo de autoengaño)")
        else:
            lines.append("❌ **CONGELAR RAMA**")
            lines.append(f"- Ganancia neta: {ganancia_neta:+.3f} (costo excede beneficio)")

        lines.append(f"\n### Detalles:")
        lines.append(f"- Precision improvement: {precision_delta:+.1%}")
        lines.append(f"- Success rate delta: {success_delta:+.1%}")
        lines.append(f"- Wall time ratio: {wall_time_ratio:.2f}x")
        lines.append(f"- Artifact size ratio: {artifact_ratio:.2f}x")

        return "\n".join(lines)


# Importar datetime para el reporte
from datetime import datetime
