"""Extraccion de signos vitales del organismo autonomo."""

from __future__ import annotations

from typing import Any, Dict

from runtime.organism.lineage import LineageState
from runtime.organism.state import OrganismState

from .contracts import VitalMode, VitalSignsSnapshot


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return min(max(float(value), lo), hi)


def mode_for_vitals(vitals: VitalSignsSnapshot) -> VitalMode:
    if not vitals.reversible or vitals.viability_margin <= 0.05:
        return "rollback"
    if vitals.viability_margin < 0.15 or vitals.identity_continuity < 0.45:
        return "quarantine"
    if (
        vitals.viability_margin < 0.35
        or vitals.risk_score >= 0.80
        or vitals.recovery_debt >= 0.65
    ):
        return "recovery"
    if (
        vitals.resource_pressure >= 0.80
        or vitals.risk_score >= 0.60
        or vitals.continuity_score < 0.60
    ):
        return "conservative"
    return "normal"


class VitalSignsService:
    """Construye snapshots de salud viva a partir del runtime existente."""

    def __init__(self, *, storage=None):
        self.storage = storage

    def from_state(
        self,
        *,
        run_id: str,
        organism_state: OrganismState,
        lineage: LineageState,
        mode: VitalMode = "normal",
        episode_result: Dict[str, Any] | None = None,
        resource_snapshot: Dict[str, Any] | None = None,
    ) -> VitalSignsSnapshot:
        cert = self._latest_certificate(run_id=run_id, episode_result=episode_result)
        cert_meta = getattr(cert, "metadata", None) or {}
        transfer = cert_meta.get("transfer_assessment") or {}
        risk_plus = cert_meta.get("risk_plus") or {}
        omega = cert_meta.get("omega") or {}

        continuity = _as_float(getattr(cert, "continuity_score", None), 1.0)
        ioc = _as_float(getattr(cert, "ioc_proxy", None), 0.0)
        risk = _as_float(getattr(cert, "risk_score", None), 1.0 - ioc)
        certified = getattr(cert, "verdict", None) == "certified"

        viability = episode_result.get("viability_assessment") if episode_result else None
        viability_margin = _as_float(
            (viability or {}).get("viability_margin"),
            organism_state.viability.viability_margin,
        )
        recovery_debt = _as_float(organism_state.viability.recovery_debt, 0.0)
        accumulated_drift = _as_float(organism_state.policy.accumulated_drift, 0.0)
        memory_purity = _as_float(transfer.get("memory_purity_score"), 1.0)
        resource_pressure = _as_float(
            (risk_plus.get("b_safe") or {}).get("pressure"),
            0.0,
        )
        if resource_pressure <= 0.0:
            resource_pressure = _as_float((episode_result or {}).get("resource_pressure"), 0.0)
        # Sensado real de host/GPU (opt-in): domina cuando está presente.
        if resource_snapshot:
            sensed = _as_float(resource_snapshot.get("hardware_pressure"), 0.0)
            if sensed > 0.0:
                resource_pressure = max(resource_pressure, sensed)

        reward = episode_result.get("reasoning_reward") if episode_result else None
        reward_scalar = _as_float((reward or {}).get("reward"), 0.0)
        prob_lcb = self._prob_lcb(episode_result or {})
        cognitive_quality = _clamp(
            0.45 * ioc
            + 0.25 * continuity
            + 0.20 * prob_lcb
            + 0.10 * _clamp((reward_scalar + 1.0) / 2.0)
        )
        identity_continuity = _clamp(0.50 * continuity + 0.50 * lineage.consistency_score())
        reversible = bool((viability or {}).get("rollback_required") is not True)
        if cert is not None:
            reversible = reversible and bool(getattr(cert, "rollback_ready", True))

        draft = VitalSignsSnapshot(
            run_id=run_id,
            episode_count=int(organism_state.episode_count),
            mode=mode,
            viability_margin=round(_clamp(viability_margin), 4),
            continuity_score=round(_clamp(continuity), 4),
            ioc_proxy=round(_clamp(ioc), 4),
            risk_score=round(_clamp(risk), 4),
            memory_purity=round(_clamp(memory_purity), 4),
            cognitive_quality=round(_clamp(cognitive_quality), 4),
            resource_pressure=round(_clamp(resource_pressure), 4),
            recovery_debt=round(_clamp(recovery_debt), 4),
            accumulated_drift=round(_clamp(accumulated_drift), 4),
            reversible=reversible,
            identity_continuity=round(_clamp(identity_continuity), 4),
            certified=bool(certified),
            metadata={
                "certificate_id": getattr(cert, "certificate_id", None),
                "episode_id": getattr(cert, "episode_id", None),
                "sie_verdict": risk_plus.get("sie_verdict"),
                "delta_ioc": risk_plus.get("delta_ioc"),
                "omega_delta_ioc_star": omega.get("delta_ioc_star"),
            },
        )
        return VitalSignsSnapshot(
            **{
                **draft.to_dict(),
                "mode": mode_for_vitals(draft),
            }
        )

    def bootstrap(
        self,
        *,
        run_id: str,
        organism_state: OrganismState,
        lineage: LineageState,
        resource_snapshot: Dict[str, Any] | None = None,
    ) -> VitalSignsSnapshot:
        return self.from_state(
            run_id=run_id,
            organism_state=organism_state,
            lineage=lineage,
            mode="normal",
            episode_result=None,
            resource_snapshot=resource_snapshot,
        )

    def _latest_certificate(self, *, run_id: str, episode_result: Dict[str, Any] | None):
        cert_id = ((episode_result or {}).get("certification") or {}).get("certificate_id")
        if self.storage is None:
            return None
        try:
            if cert_id:
                cert = self.storage.get_episode_certificate(certificate_id=cert_id)
                if cert is not None:
                    return cert
            certs = self.storage.list_episode_certificates(run_id=run_id, limit=1)
            return certs[0] if certs else None
        except Exception:
            return None

    @staticmethod
    def _prob_lcb(episode_result: Dict[str, Any]) -> float:
        state = ((episode_result.get("reasoning") or {}).get("state") or {})
        posterior = state.get("prob_posterior") or {}
        return _clamp(_as_float(posterior.get("lower_confidence_bound"), 0.5))
