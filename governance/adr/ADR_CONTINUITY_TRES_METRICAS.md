# ADR — Tres métricas de continuidad coexistentes (documentación, no fusión)

- **Fecha:** 2026-07-01
- **Estado:** aceptado (documenta el estado actual; la unificación queda explícitamente diferida)
- **Contexto:** reorg estructural del repo. La auditoría (`docs/analysis/`) señaló "≥3 implementaciones
  de continuidad" como posible duplicación a unificar. Al examinarlas, son **tres métricas vivas con
  semánticas y consumidores distintos**, no copias accidentales.

## Las tres implementaciones

| # | Ubicación | Qué mide | Consumidores |
|---|---|---|---|
| 1 | `runtime/reality/continuity.py::continuity_score` | Score escalar de continuidad entre episodios en la capa *reality*: solapamiento Jaccard de signos SMG (0.4) + consistencia causal factual/contrafactual sobre `main_variable` (0.3) + estabilidad de secuencia de razonamiento (0.2) + integridad de traza (0.1) | Evaluador de reality; alimenta `continuity_score` persistido en `runtime/storage/records.py` |
| 2 | `runtime/reality/transition_analysis.py::TransitionContinuityVector` / `build_continuity_tensor` | Representación **vectorial/tensorial** (no escalar) de la continuidad de transiciones; base estructural para análisis | `runtime/reality/transition_matrix.py`, `runtime/reality/analogical_lab.py` (vía wrappers `continuity_vector`/`continuity_tensor` de `continuity.py`) |
| 3 | `runtime/certification/continuity_guard.py::ContinuityGuard.score` | Guardia de continuidad **identitaria** en certificación: secuencia (0.6) + estabilidad de la variable principal (0.4), con `fallback_continuity` y umbral de alerta 0.35 | `runtime/certification/promotion_gate.py` (gate de promoción) |

Notas:
- Ninguna hardcodea ya `temperature` como variable: ambas capas resuelven `main_variable` desde
  `scenario_metadata` (con `temperature` solo como default de compatibilidad térmica).
- (1) y (3) difieren a propósito: (1) mide continuidad *epistémica* del episodio (signos SMG +
  contrafactuales); (3) mide continuidad *identitaria* contra el certificado previo, sin acceso a SMG.
- (2) no es comparable con (1)/(3): es una representación estructural, no un score.

## Decisión

**No fusionar en esta reorg.** Los tres viven en el pipeline de certificación/reality con pesos y
entradas distintas; cualquier fusión cambia métricas de certificación (promociones, alertas) y exige
una campaña de benchmarks propia para validar equivalencia o mejora.

## Consecuencias / trabajo futuro

- Si se quiere unificar, el camino razonable es: (a) extraer los kernels compartidos
  (`_sequence_stability`/`_sequence_score` son casi idénticos — único solapamiento textual real);
  (b) definir en el canon qué "continuidad" gobierna promoción vs evaluación; (c) correr los
  benchmarks de gate (`tests/certification/`, campañas de `scripts/`) antes/después.
- Referencias: `docs/analysis/17_SYNTHESIS.md` (hallazgo original), `docs/analysis/LEGACY_QUARANTINE.md`.
