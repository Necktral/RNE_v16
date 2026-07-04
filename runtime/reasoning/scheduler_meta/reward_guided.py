"""Selector de overlays guiado por la recompensa semi-Markov (canon f2.4 §8).

R3a computó la recompensa r = ΔIoC* − λE·ΔE − λB·B_safe por episodio; hasta
ahora era informativa. Este selector la convierte en GOBIERNO de la ecología
opcional: por run (y por régimen), acumula evidencia de qué recompensa media se
observa con y sin cada familia opcional activa, y emite directivas:

- ``on``  si Δr̄ = r̄_con − r̄_sin > ε con evidencia suficiente (la familia paga),
- ``off`` si Δr̄ < −ε o cae en la zona muerta (no paga el coste de ir),
- exploración determinista round-robin cuando falta evidencia (una familia
  sub-observada por episodio, acotada por ``max_active``).

La evidencia se estratifica POR RÉGIMEN: una familia que paga en ``viability_edge``
puede restar en ``homogeneous_safe``. ``directives(regime=...)`` usa la evidencia
específica del régimen (fallback a la combinada si hay pocos datos).

Multi-organismo: ``export_evidence``/``merge_from`` permiten que una ecología
herede (padre→hijo) o comparta (par↔par) la política aprendida, con el morfismo
causal y las reglas de herencia como guardas del cruce de contexto.

Sin engaños, por construcción:
- la recompensa ya internaliza el coste (λE) y la deriva semántica (ΔIoC* vía Ω);
- el selector SOLO toca familias opcionales — núcleo, floors y validación de
  cierre quedan fuera de su alcance (la secuencia ejecutada sigue siendo la
  validada);
- cada decisión queda auditada en ``summary()`` (deltas, n, fase, por régimen).

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

# Observación: (recompensa, overlays_activos, régimen). El régimen puede ser "".
Observation = Tuple[float, FrozenSet[str], str]


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
    """Evidencia por run (estratificada por régimen) de Δr̄ por familia opcional."""

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
        self._observations: Dict[str, List[Observation]] = {}

    # ------------------------------------------------------------------ seed

    def _seed(self, run_id: str) -> List[Observation]:
        observations: List[Observation] = []
        if self.storage is not None:
            try:
                events = self.storage.list_events(run_id=run_id, limit=256)
                for item in reversed(list(events)):  # cronológico
                    if getattr(item, "event_type", None) != "reasoning.reward":
                        continue
                    payload = getattr(item, "payload", None) or {}
                    reward = payload.get("reward")
                    overlays = payload.get("optional_overlays_active")
                    regime = payload.get("regime_label") or payload.get("regime") or ""
                    if isinstance(reward, (int, float)) and isinstance(overlays, list):
                        observations.append(
                            (
                                float(reward),
                                frozenset(str(f).strip().lower() for f in overlays),
                                str(regime),
                            )
                        )
            except Exception:
                observations = []
        self._observations[run_id] = observations
        return observations

    def _obs(self, run_id: str) -> List[Observation]:
        if run_id not in self._observations:
            return self._seed(run_id)
        return self._observations[run_id]

    def _obs_for_regime(self, run_id: str, regime: Optional[str]) -> List[Observation]:
        """Observaciones del régimen pedido; si hay <min_obs, cae a las combinadas."""
        rows = self._obs(run_id)
        if not regime:
            return rows
        scoped = [obs for obs in rows if obs[2] == regime]
        # Necesitamos min_obs con y sin overlay para decidir; si el corte por
        # régimen deja muy pocos, usar el pool combinado (no inventar señal).
        if len(scoped) >= max(2 * self.min_obs, self.min_obs + 1):
            return scoped
        return rows

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
        regime: Optional[str] = None,
    ) -> None:
        reward = (reward_block or {}).get("reward")
        if not isinstance(reward, (int, float)):
            return
        overlays = frozenset(self.overlays_from_sequence(executed_sequence))
        self._obs(run_id).append((float(reward), overlays, str(regime or "")))

    # ------------------------------------------------------------- evidencia

    def _family_evidence(
        self, run_id: str, family: str, *, regime: Optional[str] = None
    ) -> Dict[str, Any]:
        rows = self._obs_for_regime(run_id, regime)
        with_values: List[float] = []
        without_values: List[float] = []
        for reward, overlays, _regime in rows:
            (with_values if family in overlays else without_values).append(reward)
        n_with, n_without = len(with_values), len(without_values)
        delta: Optional[float] = None
        if n_with >= self.min_obs and n_without >= self.min_obs:
            delta = (sum(with_values) / n_with) - (sum(without_values) / n_without)
        return {"family": family, "n_with": n_with, "n_without": n_without, "delta": delta}

    # ------------------------------------------------------------ directivas

    def directives(self, run_id: str, regime: Optional[str] = None) -> Dict[str, str]:
        """Directivas por familia para el PRÓXIMO episodio del run (en ``regime``)."""
        evidence = {
            family: self._family_evidence(run_id, family, regime=regime)
            for family in self.candidates
        }
        decided_on: List[Tuple[float, str]] = []
        decided_off: List[str] = []
        undecided: List[str] = []
        for family, info in evidence.items():
            delta = info["delta"]
            if delta is None:
                undecided.append(family)
            elif delta > self.epsilon:
                decided_on.append((delta, family))
            else:
                decided_off.append(family)  # negativo o zona muerta: no paga ir

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
            episode_index = len(self._obs_for_regime(run_id, regime))
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

    # ------------------------------------------------- herencia / compartir

    def export_evidence(self, run_id: str) -> List[Observation]:
        """Serializa la evidencia del run (para herencia padre→hijo)."""
        return list(self._obs(run_id))

    def merge_from(
        self,
        run_id: str,
        source_observations: Sequence[Observation],
        *,
        morphism: Any = None,
        eligible: bool = True,
    ) -> int:
        """Fusiona evidencia de otro selector/organismo en ``run_id``.

        Guardas del cruce de contexto:
        - ``eligible``: resultado de ``LineageState.check_inheritance_eligibility``
          (herencia) — si es False, no se hereda nada.
        - ``morphism``: si se da, la clase adversarial/incompatible BLOQUEA el
          transporte (la deriva de contexto haría engañosa la evidencia). Si es
          compatible, se transporta tal cual (las familias de razonamiento son
          escenario-independientes; el morfismo solo decide si el contexto es
          comparable).

        Devuelve el número de observaciones fusionadas.
        """
        if not eligible:
            return 0
        if morphism is not None:
            morphism_class = getattr(morphism, "morphism_class", None)
            if morphism_class in {"adversarial", "incompatible"}:
                return 0
        target = self._obs(run_id)
        merged = 0
        for obs in source_observations:
            if (
                isinstance(obs, tuple)
                and len(obs) == 3
                and isinstance(obs[0], (int, float))
            ):
                target.append((float(obs[0]), frozenset(obs[1]), str(obs[2])))
                merged += 1
        return merged

    # ------------------------------------------------------------- resumen

    def summary(self, run_id: str, regime: Optional[str] = None) -> Dict[str, Any]:
        """Bloque auditable JSON-safe del estado de evidencia."""
        evidence = []
        for family in self.candidates:
            info = self._family_evidence(run_id, family, regime=regime)
            evidence.append(
                {
                    "family": family,
                    "n_with": info["n_with"],
                    "n_without": info["n_without"],
                    "delta_reward": None if info["delta"] is None else round(info["delta"], 6),
                }
            )
        regimes_seen = sorted({obs[2] for obs in self._obs(run_id) if obs[2]})
        return {
            "schema": "reward_guided.v2",
            "epsilon": self.epsilon,
            "min_obs": self.min_obs,
            "max_active": self.max_active,
            "n_observations": len(self._obs(run_id)),
            "regime_scope": regime or "pooled",
            "regimes_seen": regimes_seen,
            "evidence": evidence,
        }
