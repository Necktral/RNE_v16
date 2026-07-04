"""Contracts T4 para snapshot del organismo.

OrganismSnapshot es la unidad persistente subordinada de una trayectoria.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict

from .state import (
    IdentityState,
    ModificationState,
    OrganismBeliefState,
    OrganismState,
    PolicyState,
    ViabilityState,
)


@dataclass(frozen=True)
class OrganismSnapshot:
    """Snapshot puntual del organismo dentro de una trayectoria."""

    snapshot_id: str = ""
    timestamp: str = ""
    active_regime: str = "unknown"
    episode_count: int = 0
    belief: OrganismBeliefState = field(default_factory=OrganismBeliefState)
    policy: PolicyState = field(default_factory=PolicyState)
    identity: IdentityState = field(default_factory=IdentityState)
    viability: ViabilityState = field(default_factory=ViabilityState)
    modification: ModificationState = field(default_factory=ModificationState)

    @property
    def is_viable(self) -> bool:
        return self.viability.is_viable

    @property
    def composite_health(self) -> float:
        return min(
            1.0,
            max(
                0.0,
                (
                    0.25 * self.belief.composite_confidence
                    + 0.20 * self.policy.stability_score
                    + 0.25 * self.viability.distance_to_edge
                    + 0.15 * (1.0 - self.identity.identity_distance(IdentityState()))
                    + 0.15 * (1.0 if not self.modification.lineage_delta_pending else 0.5)
                ),
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if "identity" in payload and "active_invariants" in payload["identity"]:
            payload["identity"]["active_invariants"] = sorted(payload["identity"]["active_invariants"])
        return payload

    @classmethod
    def from_state(cls, state: OrganismState) -> "OrganismSnapshot":
        return cls(
            snapshot_id=state.state_id,
            timestamp=state.timestamp,
            active_regime=state.active_regime,
            episode_count=state.episode_count,
            belief=state.belief,
            policy=state.policy,
            identity=state.identity,
            viability=state.viability,
            modification=state.modification,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrganismSnapshot":
        return cls(
            snapshot_id=data.get("snapshot_id", "") or data.get("state_id", ""),
            timestamp=data.get("timestamp", ""),
            active_regime=data.get("active_regime", "unknown"),
            episode_count=data.get("episode_count", 0),
            belief=OrganismBeliefState(**data.get("belief", {})),
            policy=PolicyState(**data.get("policy", {})),
            identity=IdentityState(
                active_invariants=frozenset(data.get("identity", {}).get("active_invariants", [])),
                lineage_id=data.get("identity", {}).get("lineage_id", "genesis"),
                constitution_hash=data.get("identity", {}).get("constitution_hash", ""),
                baseline_anchor=data.get("identity", {}).get("baseline_anchor", "baseline_fixed"),
                inheritable_memory_scope=data.get("identity", {}).get("inheritable_memory_scope", "local"),
                min_continuity_threshold=data.get("identity", {}).get("min_continuity_threshold", 0.60),
            ),
            viability=ViabilityState(**data.get("viability", {})),
            modification=ModificationState(
                active_proposals=tuple(),
                lineage_delta_pending=data.get("modification", {}).get("lineage_delta_pending", False),
            ),
        )

    def to_state(self) -> OrganismState:
        return OrganismState(
            state_id=self.snapshot_id,
            timestamp=self.timestamp,
            active_regime=self.active_regime,
            episode_count=self.episode_count,
            belief=self.belief,
            policy=self.policy,
            identity=self.identity,
            viability=self.viability,
            modification=self.modification,
        )
