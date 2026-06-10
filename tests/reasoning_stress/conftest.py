"""Registro de `xfail` para los experimentos de caracterización de `reasoning_stress`.

Estos tests **no verifican el organismo**: caracterizan propiedades geométricas/
estadísticas (criticalidad auto-organizada, escala-invariancia, auto-similaridad
temporal, histéresis, dimensión fractal de fronteras/atractores) buscando dinámica
de "borde del caos" en el meta-scheduler.

Los **bugs genuinos** del andamiaje ya se arreglaron (2026-06-10):
- `fractal_utils._analyze_convergence`: la rugosidad `log(var)/log(ratio)` divergía a
  ~7.7 con var→0 (clamp 1e-10) → fronteras estables mal clasificadas "pathological".
- `fractal_utils._estimate_dimension_from_box_counts` y
  `fractal_geometries.estimate_fractal_dimension_boxcount`: faltaba normalización al
  cubo unitario y restricción a la región de escala (las cajas saturadas aplanaban la
  pendiente) → FD≈0 y falsos "poor_fit".
- `runtime/reasoning/scheduler_meta/budgeting.compute_budget`: `KeyError` ante feature
  dicts parciales (ahora robusto con `.get(..., 0.0)`).

Lo que queda son **hipótesis de investigación que no aplican a un scheduler
DETERMINISTA basado en umbrales** (no es un sistema crítico ni auto-similar), más
brittleness de muestreo finito / pasos discretos / umbrales de activación asumidos que
no coinciden con la política real. Se marcan `xfail` (no estricto) con su razón, en vez
de forzar el código o relajar umbrales (decisión del usuario: "cazar bugs, luego xfail").
"""

import pytest

_R_FRACTAL_DIM = (
    "Caracterización: la dimensión fractal por box-counting subestima con muestreo "
    "finito (estimador ya corregido: normalización + región de escala). Alcanzar los "
    "valores de libro exigiría muestras mucho mayores, no un fix de código."
)
_R_CRITICALITY = (
    "Caracterización: hipótesis de criticalidad auto-organizada / avalanchas no "
    "triviales que un scheduler determinista por umbrales no exhibe (rígido por diseño)."
)
_R_SCALE_INV = (
    "Caracterización: auto-similaridad / escala-invariancia temporal que el scheduler "
    "determinista no presenta (no es un sistema crítico)."
)
_R_DISCRETE_HYST = (
    "Caracterización: histéresis ~0.02 (un paso discreto del sweep) frente a una "
    "expectativa de ~0; brittleness de discretización, no un bug del scheduler."
)
_R_THRESHOLD = (
    "Caracterización: el test asume un umbral de activación que no coincide con la "
    "política real (la política es la fuente de verdad)."
)
_R_REGION = (
    "Caracterización: cobertura de regiones esperada en el hipercubo de features que el "
    "scheduler no produce para los puntos muestreados."
)

# Clave: "<archivo>::<test>" (sufijo del nodeid). Valor: razón documentada.
_XFAIL = {
    # — Dimensión fractal (box-counting, sesgo de muestreo finito) —
    "test_geometry_catalog.py::test_tree_branching_factor_sweep": _R_FRACTAL_DIM,
    "test_geometry_catalog.py::test_lorenz_attractor_generation": _R_FRACTAL_DIM,
    "test_geometry_catalog.py::test_rossler_attractor_generation": _R_FRACTAL_DIM,
    "test_geometry_catalog.py::test_menger_sponge_generation": _R_FRACTAL_DIM,
    "test_geometry_catalog.py::test_sierpinski_tetrahedron_generation": _R_FRACTAL_DIM,
    "test_geometry_catalog.py::test_mandelbrot_set_generation": _R_FRACTAL_DIM,
    "test_geometry_catalog.py::test_rule30_ca_pattern": _R_FRACTAL_DIM,
    "test_geometry_catalog.py::test_scale_free_graph_generation": _R_FRACTAL_DIM,
    "test_geometry_catalog.py::test_small_world_graph_generation": _R_FRACTAL_DIM,
    "test_atlas_comprehensive.py::test_atlas_quality_thresholds": _R_FRACTAL_DIM,
    # — Criticalidad / avalanchas (scheduler determinista = rígido) —
    "test_activation_avalanche.py::test_no_dominant_avalanche_size": _R_CRITICALITY,
    "test_activation_avalanche.py::test_not_rigid_system": _R_CRITICALITY,
    "test_activation_avalanche.py::test_comprehensive_avalanche_analysis": _R_CRITICALITY,
    # — Escala-invariancia / auto-similaridad temporal —
    "test_temporal_cascade.py::test_contradiction_increasing_temporal_cascade": _R_SCALE_INV,
    "test_temporal_cascade.py::test_edge_pressure_increasing_cascade": _R_SCALE_INV,
    "test_temporal_cascade.py::test_contradiction_decreasing_cascade": _R_SCALE_INV,
    "test_temporal_cascade.py::test_temporal_self_similarity_across_features": _R_SCALE_INV,
    "test_temporal_cascade.py::test_scale_invariance_errors_bounded": _R_SCALE_INV,
    "test_temporal_cascade.py::test_activation_persistence_patterns": _R_SCALE_INV,
    "test_temporal_cascade.py::test_no_pathological_cascades[edge_pressure-increasing]": _R_SCALE_INV,
    # — Histéresis discreta (~0.02 = un paso) vs expectativa ~0 —
    "test_temporal_hysteresis.py::test_edge_pressure_heur_hysteresis": _R_DISCRETE_HYST,
    "test_temporal_hysteresis.py::test_contradiction_fal_guard_hysteresis": _R_DISCRETE_HYST,
    "test_temporal_hysteresis.py::test_contradiction_both_guards_symmetric": _R_DISCRETE_HYST,
    "test_temporal_hysteresis.py::test_gradual_decrease_stability": _R_DISCRETE_HYST,
    "test_temporal_hysteresis.py::test_no_undesired_memory_effects": _R_DISCRETE_HYST,
    "test_temporal_hysteresis.py::test_all_thresholds_minimal_hysteresis": _R_DISCRETE_HYST,
    # — Umbrales de activación asumidos ≠ política real —
    "test_boundary_sweep.py::test_edge_pressure_heur_activation_boundary": _R_THRESHOLD,
    "test_boundary_sweep.py::test_contradiction_fal_guard_activation_boundary": _R_THRESHOLD,
    "test_boundary_sweep.py::test_contradiction_both_guards_activate_together": _R_THRESHOLD,
    "test_family_contribution.py::test_dia_adv_fal_guard_contribution": _R_THRESHOLD,
    "test_adversarial_thresholds.py::test_adversarial_multiple_just_above_thresholds": _R_THRESHOLD,
    "test_multiscale_boundary.py::test_contradiction_guards_multiscale_consistency": _R_THRESHOLD,
    # — Cobertura de regiones del hipercubo —
    "test_hypercube_sampling.py::test_hypercube_basic_mapping": _R_REGION,
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        suffix = item.nodeid.split("/")[-1]  # "<archivo>.py::<test>[param]"
        reason = _XFAIL.get(suffix)
        if reason is not None:
            # strict=False: si un fix posterior hace pasar el test, será XPASS (no falla).
            item.add_marker(pytest.mark.xfail(reason=reason, strict=False))
