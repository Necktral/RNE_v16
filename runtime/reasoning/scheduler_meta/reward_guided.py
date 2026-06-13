"""Selector de overlays guiado por la recompensa semi-Markov (canon f2.4 §8).

R3a computó la recompensa r = ΔIoC* − λE·ΔE − λB·B_safe por episodio; hasta
ahora era informativa. Este selector la convierte en GOBIERNO de la ecología
opcional: por run, acumula evidencia de qué recompensa media se observa con y
sin cada familia opcional activa, y emite directivas:

- ``on``  si Δr̄ = r̄_con − r̄_sin > ε con evidencia suficiente (la familia paga),
- ``off`` si Δr̄ < −ε (la familia resta),
- exploración determinista round-robin cuando falta evidencia (una familia
  sub-observada por episodio, acotada por ``max_active``).

Sin engaños, por construcción:
- la recompensa ya internaliza el coste (λE) y la deriva semántica (ΔIoC* vía Ω);
- el selector SOLO toca familias opcionales — núcleo, floors y validación de
  cierre quedan fuera de su alcance (la secuencia ejecutada sigue siendo la
  validada);
- cada decisión queda auditada en ``summary()`` (deltas, n, fase).

Activación: ``RNFE_REWARD_GUIDED_SELECTION=1`` en el runner (apagado por
defecto: disciplina sombra). Python puro, determinista, sin dependencias.
"""

from __future__ import annotations

import os
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from runtime.reasoning.scheduler_meta.family_profiles import (
    CORE_SEQUENCE,
    DELIBERATIVE_FAMILIES,
    OPTIONAL_FAMILIES,
)


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


def is_reward_guided_enabled() -> bool:
    return os.environ.get("RNFE_REWARD_GUIDED_SELECTION", "0").strip() == "1"


_CORE = frozenset(CORE_SEQUENCE)
DEFAULT_CANDIDATES: Tuple[str, ...] = tuple(
    family
    for family in list(OPTIONAL_FAMILIES) + list(DELIBERATIVE_FAMILIES)
    if family != "eml_sr"  # experimental: su gate propio manda
)


class RewardGuidedOverlaySelector:
    """Evidencia por run de Δr̄ por familia opcional + directivas on/off."""

    def __init__(
        self,
        storage: Any = None,
        *,
        candidates: Sequence[str] | None = None,
        epsilon: float | None = None,
        min_obs: int | None = None,
        max_active: int | None = None,
    ) -> None:
        self.storage = storage
        self.candidates: List[str] = [
            family.strip().lower()
            for family in (candidates or DEFAULT_CANDIDATES)
            if family and family.strip().lower() not in _CORE
        ]
        self.epsilon = epsilon if epsilon is not None else _env_float(
            "RNFE_REWARD_GUIDED_EPSILON", 0.005
        )
        self.min_obs = min_obs if min_obs is not None else _env_int(
            "RNFE_REWARD_GUIDED_MIN_OBS", 2
        )
        self.max_active = max_active if max_active is not None else _env_int(
            "RNFE_REWARD_GUIDED_MAX_ACTIVE", 2
        )
        self._observations: Dict[str, List[Tuple[float, FrozenSet[str]]]] = {}

    # ------------------------------------------------------------------ seed

    def _seed(self, run_id: str) -> List[Tuple[float, FrozenSet[str]]]:
        observations: List[Tuple[float, FrozenSet[str]]] = []
        if self.storage is not None:
            try:
                events = self.storage.list_events(run_id=run_id, limit=256)
                for item in reversed(list(events)):  # cronológico
                    if getattr(item, "event_type", None) != "reasoning.reward":
                        continue
                    payload = getattr(item, "payload", None) or {}
                    reward = payload.get("reward")
                    overlays = payload.get("optional_overlays_active")
                    if isinstance(reward, (int, float)) and isinstance(overlays, list):
                        observations.append(
                            (
                                float(reward),
                                frozenset(str(f).strip().lower() for f in overlays),
                            )
                        )
            except Exception:
                observations = []
        self._observations[run_id] = observations
        return observations

    def _obs(self, run_id: str) -> List[Tuple[float, FrozenSet[str]]]:
        if run_id not in self._observations:
            return self._seed(run_id)
        return self._observations[run_id]

    # --------------------------------------------------------------- observe

    @staticmethod
    def overlays_from_sequence(executed_sequence: Sequence[str]) -> List[str]:
        return [
            str(family).strip().lower()
            for family in executed_sequence or ()
            if str(family).strip().lower() not in _CORE
        ]

    def observe(
        self,
        *,
        run_id: str,
        reward_block: Mapping[str, Any],
        executed_sequence: Sequence[str],
    ) -> None:
        reward = (reward_block or {}).get("reward")
        if not isinstance(reward, (int, float)):
            return
        overlays = frozenset(self.overlays_from_sequence(executed_sequence))
        self._obs(run_id).append((float(reward), overlays))

    # ------------------------------------------------------------- evidencia

    def _family_evidence(self, run_id: str, family: str) -> Dict[str, Any]:
        with_values: List[float] = []
        without_values: List[float] = []
        for reward, overlays in self._obs(run_id):
            (with_values if family in overlays else without_values).append(reward)
        n_with, n_without = len(with_values), len(without_values)
        delta: Optional[float] = None
        if n_with >= self.min_obs and n_without >= self.min_obs:
            delta = (sum(with_values) / n_with) - (sum(without_values) / n_without)
        return {"family": family, "n_with": n_with, "n_without": n_without, "delta": delta}

    # ------------------------------------------------------------ directivas

    def directives(self, run_id: str) -> Dict[str, str]:
        """Directivas por familia para el PRÓXIMO episodio del run."""
        evidence = {family: self._family_evidence(run_id, family) for family in self.candidates}
        decided_on: List[Tuple[float, str]] = []
        decided_off: List[str] = []
        undecided: List[str] = []
        for family, info in evidence.items():
            delta = info["delta"]
            if delta is None:
                undecided.append(family)
            elif delta > self.epsilon:
                decided_on.append((delta, family))
            elif delta < -self.epsilon:
                decided_off.append(family)
            else:
                decided_off.append(family)  # zona muerta: no paga el coste de ir

        directives: Dict[str, str] = {family: "off" for family in decided_off}
        decided_on.sort(reverse=True)  # mayor Δr̄ primero
        active: List[str] = []
        for _, family in decided_on:
            if len(active) < self.max_active:
                active.append(family)
            else:
                directives[family] = "off"

        # Exploración determinista: una familia sub-observada por episodio,
        # solo si queda hueco bajo max_active.
        if undecided and len(active) < self.max_active:
            episode_index = len(self._obs(run_id))
            explore = sorted(undecided)[episode_index % len(undecided)]
            active.append(explore)
            for family in undecided:
                if family != explore:
                    directives[family] = "off"
        else:
            for family in undecided:
                directives[family] = "off"

        for family in active:
            directives[family] = "on"
        return directives

    def summary(self, run_id: str) -> Dict[str, Any]:
        """Bloque auditable JSON-safe del estado de evidencia."""
        evidence = []
        for family in self.candidates:
            info = self._family_evidence(run_id, family)
            evidence.append(
                {
                    "family": family,
                    "n_with": info["n_with"],
                    "n_without": info["n_without"],
                    "delta_reward": None if info["delta"] is None else round(info["delta"], 6),
                }
            )
        return {
            "schema": "reward_guided.v1",
            "epsilon": self.epsilon,
            "min_obs": self.min_obs,
            "max_active": self.max_active,
            "n_observations": len(self._obs(run_id)),
            "evidence": evidence,
        }
