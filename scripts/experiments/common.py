"""Infraestructura determinista compartida por los experimentos MCI."""

from __future__ import annotations

import json
import os
import random
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from runtime.storage import StorageConfig, StorageFactory
from runtime.symbolic.mci import MCIPlanningConfig
from runtime.world import ScenarioEpisodeRunner

SEEDS = (42, 123, 777, 2025, 9999, 111, 222, 333, 444, 555)
EXPERIMENT_VERSION = "mci-capability-v1"


def experiment_work_root(result_root: str | Path) -> Path:
    configured = os.environ.get("RNFE_EXPERIMENT_WORK_ROOT")
    return Path(configured) if configured else Path(result_root) / "work"


def generate_external_inputs(seed: int, n: int) -> list[float]:
    rng = random.Random(int(seed))
    return [round(rng.uniform(0.0, 0.08), 12) for _ in range(int(n))]


def bootstrap_paired_diff(
    sample1: Sequence[float],
    sample2: Sequence[float],
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 1729,
) -> tuple[float, float]:
    if len(sample1) != len(sample2) or not sample1:
        raise ValueError("Las muestras deben ser no vacías y estar emparejadas")
    if n_resamples < 1 or not 0.0 < alpha < 1.0:
        raise ValueError("Parámetros de bootstrap inválidos")
    differences = [float(a) - float(b) for a, b in zip(sample1, sample2)]
    rng = random.Random(seed)
    size = len(differences)
    means = sorted(
        sum(differences[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(n_resamples)
    )
    lower_index = max(0, int((alpha / 2.0) * n_resamples))
    upper_index = min(
        n_resamples - 1,
        int((1.0 - alpha / 2.0) * n_resamples),
    )
    return round(means[lower_index], 12), round(means[upper_index], 12)


def run_paired_campaign(
    runner_factory: Callable[..., Any],
    *,
    seeds: Iterable[int],
    n_episodes_per_seed: int,
    external_input_gen: Callable[[int, int], Sequence[float]],
    metric: Callable[[dict[str, Any]], float],
    profiles: tuple[str, str] = ("mci_integrated_v1", "core_plus_opt"),
    replay_prefix: str = "paired",
) -> dict[str, Any]:
    """Ejecuta brazos persistentes y conserva el emparejamiento por seed/episodio."""
    arms: dict[str, list[dict[str, Any]]] = {profile: [] for profile in profiles}
    for seed in tuple(seeds):
        inputs = tuple(external_input_gen(int(seed), int(n_episodes_per_seed)))
        if len(inputs) != n_episodes_per_seed:
            raise ValueError("external_input_gen devolvió una longitud incorrecta")
        for profile in profiles:
            runner = runner_factory(profile=profile, seed=int(seed))
            values: list[float] = []
            for episode, external_input in enumerate(inputs, 1):
                result = runner.run_episode(
                    external_input=float(external_input),
                    replay_unit_id=(
                        f"{replay_prefix}/{profile}/seed-{seed}/ep-{episode}"
                    ),
                )
                values.append(float(metric(result)))
            arms[profile].append(
                {
                    "seed": int(seed),
                    "episode_values": values,
                    "mean": sum(values) / len(values),
                }
            )
    return {
        "version": EXPERIMENT_VERSION,
        "profiles": profiles,
        "episodes_per_seed": int(n_episodes_per_seed),
        "arms": arms,
    }


def write_canonical_json(path: str | Path, payload: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def build_isolated_runner(
    *,
    work_root: str | Path,
    experiment: str,
    profile: str,
    seed: int,
    scenario: str,
    scenario_kwargs: dict[str, Any] | None = None,
    mci_planning_config: MCIPlanningConfig | None = None,
    n4_artifact_path: Path | None = None,
) -> ScenarioEpisodeRunner:
    root = Path(work_root) / experiment / profile / f"seed-{seed}"
    root.mkdir(parents=True, exist_ok=True)
    storage = StorageFactory.create_facade(
        StorageConfig(
            mode="sqlite",
            sqlite_db_path=str(root / "events.db"),
            postgres_dsn=None,
            artifact_root=root / "artifacts",
            prefer_postgres_reads=True,
            strict_dual_write=False,
        )
    )
    return ScenarioEpisodeRunner(
        storage=storage,
        run_id=f"{experiment}-{profile}-seed-{seed}",
        scenario=scenario,
        scenario_kwargs=dict(scenario_kwargs or {}),
        family_profile=profile,
        mci_planning_config=mci_planning_config,
        n4_artifact_path=n4_artifact_path,
    )


def observed_outcome(result: dict[str, Any]) -> dict[str, Any]:
    return dict(
        ((result.get("acting_trace") or {}).get("outcome_link") or {}).get(
            "outcome_observed"
        )
        or {}
    )
