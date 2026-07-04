# Phase 1 Implementation Guide

**Branch**: `claude/open-clean-branch-1x1-to-5x5`
**Status**: Instrumentación completa implementada
**Próximo paso**: Ejecutar benchmarks experimentales

---

## Estado Actual

### ✅ Completado

1. **Phase 0: Gate Topológico**
   - Test de no-isomorfismo ejecutado y aprobado
   - spatial_information_usage = 1.000 (10/10 pares con diferencias)
   - Rediseño espacial exitoso

2. **Metrics Infrastructure (Grupos 2-5)**
   - `metrics_cognitive_quality.py`: 6 métricas de calidad cognitiva
   - `metrics_operational_cost.py`: 6 métricas de costo operativo
   - `metrics_ivc_r.py`: IVC-R con log-space y bootstrap CI
   - `failure_taxonomy.py`: Clasificación primaria/secundaria de fallos

3. **Benchmark Infrastructure**
   - `benchmark_runner.py`: Orquestador de episodios con captura automática de métricas
   - `benchmark_config.yaml`: Configuración experimental completa
   - `test_benchmark_1x1_vs_5x5.py`: Suite de tests pytest
   - `analysis_report.py`: Análisis estadístico y generación de reportes

### 🔄 Pendiente

1. Ejecutar benchmark baseline (1x1 vs 5x5 uniform)
2. Ejecutar benchmark heterogéneo (5x5 con topologías diversas)
3. Ejecutar barridos por nivel
4. Generar reporte estadístico completo
5. Calcular ganancia cognitiva neta
6. Emitir dictamen arquitectónico final

---

## Cómo Ejecutar los Benchmarks

### Requisitos Previos

Los benchmarks están marcados con `@pytest.mark.requires_extended_bench` y NO se ejecutan automáticamente en test suites normales. Esto es intencional porque cada benchmark puede tomar 5-15 minutos.

### Ejecutar Benchmark Completo

```bash
# Ejecutar todos los benchmarks de Fase 1
pytest tests/benchmarks/test_benchmark_1x1_vs_5x5.py -m requires_extended_bench -v -s

# O ejecutar benchmarks individuales:

# 1. Baseline 1x1 (100 episodios)
pytest tests/benchmarks/test_benchmark_1x1_vs_5x5.py::TestBenchmark1x1vs5x5::test_baseline_comparison_1x1 -v -s

# 2. Baseline 5x5 uniform (100 episodios)
pytest tests/benchmarks/test_benchmark_1x1_vs_5x5.py::TestBenchmark1x1vs5x5::test_baseline_comparison_5x5_uniform -v -s

# 3. Heterogeneous 5x5 (100 episodios con topologías variadas)
pytest tests/benchmarks/test_benchmark_1x1_vs_5x5.py::TestBenchmark1x1vs5x5::test_heterogeneous_5x5 -v -s

# 4. Level sweep 1x1 (100 episodios, 25 por nivel)
pytest tests/benchmarks/test_benchmark_1x1_vs_5x5.py::TestBenchmark1x1vs5x5::test_level_sweep_1x1 -v -s

# 5. Level sweep 5x5 (100 episodios, 25 por nivel)
pytest tests/benchmarks/test_benchmark_1x1_vs_5x5.py::TestBenchmark1x1vs5x5::test_level_sweep_5x5 -v -s
```

### Resultados

Los resultados se guardan en:
```
benchmark_results/
├── grid_thermal_1x1/
│   └── <run_timestamp>/
│       ├── episodes.jsonl       # Un episodio por línea
│       └── summary.json          # Resumen estadístico
├── grid_thermal_5x5_uniform/
│   └── <run_timestamp>/
│       ├── episodes.jsonl
│       └── summary.json
└── grid_thermal_5x5_hetero/
    └── <run_timestamp>/
        ├── uniform/
        ├── hotspot_center/
        ├── gradient_ns/
        ├── gradient_ew/
        └── checkerboard/
```

---

## Análisis Post-Benchmark

Una vez ejecutados los benchmarks, generar el reporte comparativo:

```python
from pathlib import Path
from tests.benchmarks.analysis_report import BenchmarkAnalyzer

# Inicializar analyzer
analyzer = BenchmarkAnalyzer()

# Cargar resultados de ambos escenarios
result_dir_1x1 = Path("benchmark_results/grid_thermal_1x1/<timestamp>")
result_dir_5x5 = Path("benchmark_results/grid_thermal_5x5_uniform/<timestamp>")

analyzer.load_results(result_dir_1x1, result_dir_5x5)

# Ejecutar comparación estadística
comparison = analyzer.compute_statistical_comparison()

# Generar reporte final
output_file = Path("PHASE1_BENCHMARK_REPORT.md")
analyzer.generate_final_report(comparison, output_file)

print(f"Reporte generado en: {output_file}")
```

---

## Estructura de Métricas

### Grupo 2: Calidad Cognitiva

| Métrica | Rango | Interpretación |
|---------|-------|----------------|
| `factual_cf_divergence` | [-∞, 1.0] | 1.0 = factual óptimo, <0 = subóptimo |
| `intervention_precision` | [0.0, 1.0] | Proporción de intervenciones beneficiosas |
| `proposition_diversity` | [0.0, ∞] | Entropía de Shannon de proposiciones |
| `world_level_transitions` | JSON | Matriz de transiciones entre niveles |
| `spatial_coherence_index` | [-1.0, 1.0] | Correlación espacial (solo 5x5) |
| `spatial_information_usage` | [0.0, 1.0] | Proporción de decisiones espaciales |

### Grupo 3: Costo Operativo

| Métrica | Unidad | Descripción |
|---------|--------|-------------|
| `wall_time_ms` | ms | Tiempo real de ejecución |
| `cf_overhead_ratio` | ratio | Tiempo CF / Tiempo factual |
| `memory_pressure_mb` | MB | Memoria incremental usada |
| `scheduler_cpu_time_ms` | ms | CPU dedicada al scheduler |
| `artifact_size_bytes` | bytes | Tamaño del artifact serializado |
| `trace_length` | steps | Número de pasos |

### Grupo 4: IVC-R

```python
IVC-R = exp(
    0.35 * log(cierre_rate + ε) +
    0.25 * log(continuity_score + ε) +
    0.20 * log(intervention_precision + ε) +
    0.10 * log(proposition_diversity_norm + ε) -
    0.10 * log(wall_time_norm + ε)
)
```

### Grupo 5: Fallos

**Primarias**: timeout, error, counterfactual_failed, both_failed, alarm_persistent, oscillation

**Secundarias**: high_initial_temp, weak_cooling, tight_threshold, scheduler_overhead, spatial_complexity

---

## Criterios de Decisión Final

### ✅ Valor Arquitectónico Demostrado

**Criterios obligatorios (AND)**:
- [x] Gate topológico aprobado (spatial_information_usage > 0.2) — **CUMPLIDO**
- [ ] Ganancia_Neta > 0.08
- [ ] intervention_precision mejora ≥8% con p<0.05 y d>0.4
- [ ] cierre_rate no empeora >3%
- [ ] wall_time_ratio < 2.0
- [ ] artifact_size_ratio < 3.0
- [ ] Mejora en ≥3 de 4 niveles del mundo
- [ ] No introduce nuevos modos de fallo críticos

**Acción**: Aprobar como arquitectura válida, candidato a baseline futuro

### ⚠️ Avance Parcial

**Indicadores**:
- Gate topológico aprobado ✅
- Ganancia_Neta positiva pero <0.08
- O: Mejora significativa en 1-2 métricas pero costo marginal >50%
- O: Mejora solo en regímenes específicos

**Acción**: Mantener rama activa, documentar nicho de valor, no promover a baseline

### ❌ Congelar

**Indicadores**:
- Ganancia_Neta ≤ 0.0
- O: cierre_rate empeora >5%
- O: failure_rate >2x del 1x1
- O: wall_time_ratio > 3.0 sin mejora >15%

**Acción**: Congelar rama, archivar como experimento

---

## Fórmula de Ganancia Cognitiva Neta

```python
Ganancia_Neta = (IVC-R_5x5 - IVC-R_1x1) - Penalización_Costo

Penalización_Costo =
  0.10 * normalize(wall_time_ratio - 1.0, max=1.0) +
  0.05 * normalize(artifact_size_ratio - 1.0, max=2.0) +
  0.05 * normalize(memory_ratio - 1.0, max=1.0)
```

**Interpretación**:
- **Ganancia_Neta > 0.10**: Valor neto positivo claro ✅
- **0.05 < Ganancia_Neta < 0.10**: Valor marginal ⚠️
- **0.0 < Ganancia_Neta < 0.05**: Valor marginal bajo, riesgo de autoengaño ⚠️
- **Ganancia_Neta ≤ 0.0**: Costo excede beneficio ❌

---

## Checklist Anti-Autoengaño

Antes de declarar victoria, verificar:

- [ ] Al menos 2 métricas de calidad mejoran (no solo 1)
- [ ] Tamaño de efecto >0.3 en métricas clave (Cohen's d)
- [ ] Mejora consistente en ≥60% de regímenes
- [ ] Penalización de costo incluida en ganancia neta
- [ ] Validación en múltiples niveles del mundo (SAFE, ELEVATED, WARNING, CRITICAL)
- [ ] Distribuciones completas reportadas, no solo medias
- [ ] Intervalos de confianza incluidos
- [ ] Casos de fallo analizados, no solo éxitos

---

## Restricciones Vigentes

### ❌ NO HACER

- Rediseñar 5x5 topológico (ya completado)
- Tocar baseline histórico (1x1 intacto)
- Cambiar DEFAULT_SCENARIO del runtime
- Abrir 10x10 o 20x20
- Modificar defaults globales
- Declarar victoria prematura
- Promocionar 5x5 a baseline sin aprobación

### ✅ SÍ HACER

- Ejecutar benchmarks rigurosamente
- Reportar todos los resultados honestamente
- Incluir análisis de costo completo
- Aplicar checklist anti-autoengaño
- Documentar límites y trade-offs
- Generar dictamen técnico objetivo

---

## Próximos Pasos Recomendados

1. **Ejecutar baseline mínimo** (más rápido):
   ```bash
   # Solo baseline 1x1 y 5x5 uniform (2x100 episodios = ~10-20 min)
   pytest tests/benchmarks/test_benchmark_1x1_vs_5x5.py::TestBenchmark1x1vs5x5::test_baseline_comparison_1x1 -v -s
   pytest tests/benchmarks/test_benchmark_1x1_vs_5x5.py::TestBenchmark1x1vs5x5::test_baseline_comparison_5x5_uniform -v -s
   ```

2. **Analizar resultados baseline**:
   - Generar reporte comparativo
   - Calcular ganancia cognitiva neta
   - Verificar si pasa criterios mínimos

3. **Si baseline pasa**: Ejecutar heterogeneous y level sweeps

4. **Si baseline falla**: Diagnosticar causas y decidir próximos pasos

---

## Comandos Útiles

```bash
# Verificar estructura de archivos
ls -la tests/benchmarks/

# Ver configuración
cat tests/benchmarks/benchmark_config.yaml

# Listar tests disponibles
pytest tests/benchmarks/test_benchmark_1x1_vs_5x5.py --collect-only

# Ejecutar con output detallado
pytest tests/benchmarks/test_benchmark_1x1_vs_5x5.py -m requires_extended_bench -v -s --tb=short

# Verificar resultados
find benchmark_results -name "*.jsonl" -o -name "*.json"

# Analizar un resultado específico
python3 -c "
import json
with open('benchmark_results/grid_thermal_1x1/<timestamp>/summary.json') as f:
    summary = json.load(f)
    print(json.dumps(summary, indent=2))
"
```

---

## Contacto y Documentación

- **Especificación completa**: Ver mensaje de especificación técnica en conversación
- **Phase 0 resultados**: `PHASE0_GATE_RESULTS.md`
- **Código fuente**:
  - Métricas: `tests/benchmarks/metrics_*.py`
  - Runner: `tests/benchmarks/benchmark_runner.py`
  - Análisis: `tests/benchmarks/analysis_report.py`
  - Tests: `tests/benchmarks/test_benchmark_1x1_vs_5x5.py`

---

**Última actualización**: 2026-04-19
**Autor**: Claude Sonnet 4.5 (Experimental Maintainer)
