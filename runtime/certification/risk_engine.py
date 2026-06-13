"""Motor de riesgo del certificado ampliado 𝔠ₜ⁺ (canon RNFE f2.1/f2.4).

Aporta las tres señales de riesgo que el certificado episódico no tenía:

- **CVaR_α[−ΔIoC]**: riesgo de cola de la serie de cierre del run (pérdida
  esperada en el peor (1−α) de los episodios). Es el componente de riesgo del
  certificado formal 𝔠ₜ⁺ y el freno matemático de la regla de herencia S-I-E.
- **B_safe**: barrera logarítmica suave φ_bar(x;δ) = −log(δ + (1−x)) sobre
  telemetría física normalizada (presión VRAM, temperatura). Opt-in vía
  ``RNFE_BSAFE_TELEMETRY=1`` y con caché TTL para no lanzar ``nvidia-smi`` por
  episodio; por defecto queda apagada y el resultado es determinista.
- **Regla S-I-E 2.0**: ACEPTAR / BUFFER / RECHAZAR. RECHAZAR solo si el estado
  sale de la región segura (violación hard constitucional o barrera violada);
  ACEPTAR exige Pr(ΔIoC ≥ 0) ≥ umbral (cota inferior Agresti-Coull) y
  CVaR_α[−ΔIoC] ≤ τ; el resto va a BUFFER (evidencia insuficiente).

Diseño para hardware modesto: Python puro (sin numpy), historial acotado
(≤ ``max_history`` puntos por run) y aritmética O(n) por episodio. Las señales
se registran en ``metadata["risk_plus"]`` del certificado en **modo sombra**:
no alteran el veredicto existente (reproducibilidad intacta); el veredicto
S-I-E queda disponible como gate para la auto-modificación (ρₜ).

Umbrales por defecto calibrados el 2026-06-10 contra el historial real en
PostgreSQL (480 runs × 8 episodios, campaña adaptive_v2): el run sano típico
muestra CVaR₀.₉₅[−ΔIoC] = 0.09 y Pr(ΔIoC≥0)_LCB ≈ 0.4665, de modo que
τ = 0.10 y prob = 0.45 lo aceptan con margen pequeño y dejan en BUFFER
cualquier cola peor (ver ``scripts/calibrate_risk_cvar.py``). Todos
sobre-escribibles por entorno: ``RNFE_RISK_CVAR_ALPHA``,
``RNFE_RISK_CVAR_THRESHOLD``, ``RNFE_RISK_PROB_THRESHOLD``,
``RNFE_RISK_MIN_HISTORY``.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence


# ───────────────────────────── CVaR (Python puro) ────────────────────────────

def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Percentil con interpolación lineal (misma semántica que numpy 'linear')."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    k = (len(sorted_values) - 1) * (q / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_values[int(k)])
    return float(sorted_values[f] * (c - k) + sorted_values[c] * (k - f))


def compute_cvar(losses: Sequence[float], alpha: float = 0.95) -> float:
    """CVaR_α[L] = E[L | L ≥ VaR_α] sobre una muestra empírica de pérdidas."""
    values = [float(v) for v in losses]
    if not values:
        return 0.0
    values.sort()
    var_threshold = _percentile(values, alpha * 100.0)
    tail = [v for v in values if v >= var_threshold]
    if not tail:
        return var_threshold
    return sum(tail) / len(tail)


# ─────────────────────── Pr(ΔIoC ≥ 0) con cota inferior ──────────────────────

def agresti_coull_lcb(successes: int, n: int, z: float = 1.96) -> float:
    """Cota inferior de confianza para una proporción (Agresti-Coull).

    Mismo estimador que usa PROB en core_inference: conservador con n pequeño.
    """
    if n <= 0:
        return 0.0
    n_t = n + z * z
    p_t = (successes + z * z / 2.0) / n_t
    interval = z * math.sqrt(max(0.0, p_t * (1.0 - p_t) / n_t))
    return min(1.0, max(0.0, p_t - interval))


# ─────────────────────────── Barrera B_safe (f2.1) ───────────────────────────

def phi_bar(x: float, delta: float = 1e-3) -> Optional[float]:
    """Barrera suave φ_bar(x;δ) = −log(δ + (1−x)) si x < 1; None si x ≥ 1.

    Devuelve None (en vez de +inf) para que el resultado sea serializable a
    JSON/JSONB; el llamador marca la violación con un flag explícito.
    """
    if x >= 1.0:
        return None
    return -math.log(delta + (1.0 - x))


def compute_b_safe(
    signals: Mapping[str, float],
    *,
    weights: Mapping[str, float] | None = None,
    delta: float = 1e-3,
) -> Dict[str, Any]:
    """B_safe = Σ αᵢ·φ_bar(uᵢ/uᵢ_max; δ) sobre señales ya normalizadas a [0,1].

    ``signals`` mapea nombre → ratio uᵢ/uᵢ_max. Una señal ≥ 1.0 viola la
    barrera (B_safe formal = +inf): se reporta ``violated=True`` y ``value``
    queda en None para mantener el bloque JSON-safe.
    """
    weights = dict(weights or {})
    total = 0.0
    violated: List[str] = []
    per_signal: Dict[str, Any] = {}
    for name, ratio in signals.items():
        r = max(0.0, float(ratio))
        bar = phi_bar(r, delta)
        per_signal[name] = {"ratio": round(r, 6), "barrier": None if bar is None else round(bar, 6)}
        if bar is None:
            violated.append(name)
        else:
            total += float(weights.get(name, 1.0)) * bar
    return {
        "value": None if violated else round(total, 6),
        "violated": bool(violated),
        "violated_signals": violated,
        "signals": per_signal,
        "delta": delta,
    }


# ──────────────── Telemetría física opt-in (cacheada, barata) ────────────────

_BSAFE_CACHE: Dict[str, Any] = {"ts": 0.0, "block": None}
_BSAFE_SAMPLER = None


def _bsafe_enabled() -> bool:
    return os.environ.get("RNFE_BSAFE_TELEMETRY", "").strip().lower() in {"1", "true", "yes", "on"}


def sample_b_safe_telemetry(*, ttl_s: float = 5.0) -> Optional[Dict[str, Any]]:
    """Muestra B_safe desde nvidia-smi, con caché TTL.

    Apagada por defecto (None ⇒ certificados deterministas). Con
    ``RNFE_BSAFE_TELEMETRY=1`` muestrea presión VRAM (límite 1.0 = llena) y
    temperatura normalizada a ``RNFE_BSAFE_TEMP_MAX`` (default 90 °C),
    reutilizando la muestra hasta ``ttl_s`` segundos para no pagar un
    subprocess por episodio en hardware modesto.
    """
    if not _bsafe_enabled():
        return None
    now = time.monotonic()
    if _BSAFE_CACHE["block"] is not None and (now - _BSAFE_CACHE["ts"]) < ttl_s:
        return _BSAFE_CACHE["block"]

    global _BSAFE_SAMPLER
    try:
        if _BSAFE_SAMPLER is None:
            from runtime.control.msrc.vram_sampler import NvidiaVRAMSampler

            _BSAFE_SAMPLER = NvidiaVRAMSampler()
        sample = _BSAFE_SAMPLER.sample()
    except Exception:
        sample = {"available": False}

    if not sample.get("available"):
        block: Optional[Dict[str, Any]] = None
    else:
        try:
            temp_max = float(os.environ.get("RNFE_BSAFE_TEMP_MAX", "90"))
        except (TypeError, ValueError):
            temp_max = 90.0
        block = compute_b_safe(
            {
                "vram_pressure": float(sample.get("vram_pressure", 0.0)),
                "temperature": float(sample.get("temperature_c", 0.0)) / max(temp_max, 1.0),
            },
            weights={"vram_pressure": 1.0, "temperature": 1.0},
        )
        block["source"] = "nvidia-smi"
        block["sample_ts"] = sample.get("sample_ts")
    _BSAFE_CACHE["ts"] = now
    _BSAFE_CACHE["block"] = block
    return block


# ───────────────────────────── Regla S-I-E 2.0 ───────────────────────────────

SIE_ACCEPT = "ACEPTAR"
SIE_BUFFER = "BUFFER"
SIE_REJECT = "RECHAZAR"


def sie_rule(
    *,
    deltas: Sequence[float],
    cvar_alpha: float,
    cvar_threshold: float,
    prob_threshold: float,
    min_history: int,
    hard_violation_count: int = 0,
    b_safe_violated: bool = False,
) -> Dict[str, Any]:
    """Regla de herencia S-I-E 2.0 del canon f2.1.

    - estado ∉ safe (violación hard o barrera) → RECHAZAR
    - Pr(ΔIoC ≥ 0)_LCB ≥ prob_threshold ∧ CVaR_α[−ΔIoC] ≤ τ → ACEPTAR
    - en otro caso → BUFFER
    """
    if hard_violation_count > 0 or b_safe_violated:
        return {
            "verdict": SIE_REJECT,
            "reason": "estado fuera de región segura"
            + (" (violación constitucional hard)" if hard_violation_count > 0 else " (barrera B_safe violada)"),
            "p_delta_nonneg_lcb": None,
            "cvar_neg_delta_ioc": None,
        }
    n = len(deltas)
    if n < min_history:
        return {
            "verdict": SIE_BUFFER,
            "reason": f"historial insuficiente ({n} < {min_history})",
            "p_delta_nonneg_lcb": None,
            "cvar_neg_delta_ioc": None,
        }
    nonneg = sum(1 for d in deltas if d >= 0.0)
    p_lcb = agresti_coull_lcb(nonneg, n)
    cvar = compute_cvar([-d for d in deltas], cvar_alpha)
    if p_lcb >= prob_threshold and cvar <= cvar_threshold:
        verdict, reason = SIE_ACCEPT, "mejora probable con riesgo de cola acotado"
    else:
        parts = []
        if p_lcb < prob_threshold:
            parts.append(f"Pr(ΔIoC≥0)_LCB={p_lcb:.3f} < {prob_threshold}")
        if cvar > cvar_threshold:
            parts.append(f"CVaR={cvar:.4f} > {cvar_threshold}")
        verdict, reason = SIE_BUFFER, "; ".join(parts)
    return {
        "verdict": verdict,
        "reason": reason,
        "p_delta_nonneg_lcb": round(p_lcb, 4),
        "cvar_neg_delta_ioc": round(cvar, 6),
    }


# ───────────────────── Tracker por run (serie de IoC) ────────────────────────

def _env_float(name: str, default: float) -> float:
    try:
        raw = os.environ.get(name)
        return float(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        raw = os.environ.get(name)
        return int(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


class EpisodeRiskTracker:
    """Mantiene la serie de IoC por run y emite el bloque ``risk_plus`` de 𝔠ₜ⁺.

    La serie se siembra desde storage (certificados previos del run) la primera
    vez que se ve un run_id, de modo que el CVaR sobrevive a reinstanciaciones
    del gate. Historial acotado a ``max_history`` (coste por episodio O(n)).
    """

    def __init__(
        self,
        *,
        storage=None,
        cvar_alpha: float | None = None,
        cvar_threshold: float | None = None,
        prob_threshold: float | None = None,
        min_history: int | None = None,
        max_history: int = 64,
    ):
        self.storage = storage
        self.cvar_alpha = cvar_alpha if cvar_alpha is not None else _env_float("RNFE_RISK_CVAR_ALPHA", 0.95)
        self.cvar_threshold = (
            cvar_threshold if cvar_threshold is not None else _env_float("RNFE_RISK_CVAR_THRESHOLD", 0.10)
        )
        self.prob_threshold = (
            prob_threshold if prob_threshold is not None else _env_float("RNFE_RISK_PROB_THRESHOLD", 0.45)
        )
        self.min_history = min_history if min_history is not None else _env_int("RNFE_RISK_MIN_HISTORY", 4)
        self.max_history = max_history
        self._series: Dict[str, List[float]] = {}

    def _seed(self, run_id: str) -> List[float]:
        series: List[float] = []
        if self.storage is not None:
            try:
                previous = self.storage.list_episode_certificates(
                    run_id=run_id, limit=self.max_history
                )
                # storage devuelve más-reciente-primero; la serie va cronológica.
                series = [float(c.ioc_proxy) for c in reversed(previous)]
            except Exception:
                series = []
        self._series[run_id] = series
        return series

    def assess(
        self,
        *,
        run_id: str,
        ioc_value: float,
        hard_violation_count: int = 0,
        b_safe: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        series = self._series.get(run_id)
        if series is None:
            series = self._seed(run_id)

        delta = (float(ioc_value) - series[-1]) if series else None
        series.append(float(ioc_value))
        if len(series) > self.max_history:
            del series[: len(series) - self.max_history]

        deltas = [series[i] - series[i - 1] for i in range(1, len(series))]
        sie = sie_rule(
            deltas=deltas,
            cvar_alpha=self.cvar_alpha,
            cvar_threshold=self.cvar_threshold,
            prob_threshold=self.prob_threshold,
            min_history=self.min_history,
            hard_violation_count=hard_violation_count,
            b_safe_violated=bool(b_safe and b_safe.get("violated")),
        )
        return {
            "schema": "risk_plus.v1",
            "ioc": round(float(ioc_value), 6),
            "delta_ioc": None if delta is None else round(delta, 6),
            "n_history": len(deltas),
            "cvar_alpha": self.cvar_alpha,
            "cvar_neg_delta_ioc": sie["cvar_neg_delta_ioc"],
            "p_delta_nonneg_lcb": sie["p_delta_nonneg_lcb"],
            "thresholds": {
                "cvar": self.cvar_threshold,
                "prob": self.prob_threshold,
                "min_history": self.min_history,
            },
            "b_safe": b_safe,
            "hard_violation_count": int(hard_violation_count),
            "sie_verdict": sie["verdict"],
            "sie_reason": sie["reason"],
        }
