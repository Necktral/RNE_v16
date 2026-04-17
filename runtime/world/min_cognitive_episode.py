"""Episodio cognitivo mínimo: observación -> signo -> LOTF -> intervención -> actualización."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict
from uuid import uuid4

from runtime.lotf import LOTFMin
from runtime.reasoning.scheduler_meta.meta_scheduler import MetaScheduler
from runtime.smg import SMGMin
from runtime.storage import get_storage
from runtime.storage.records import utc_now_iso
from runtime.world.cgwm_min import CGWMMin


class MinimalCognitiveEpisodeRunner:
    def __init__(self, *, storage=None, run_id: str | None = None):
        self.storage = storage or get_storage()
        self.run_id = run_id or f"run-{uuid4()}"
        self.smg = SMGMin(storage=self.storage, run_id=self.run_id)
        self.lotf = LOTFMin()
        self.world = CGWMMin()
        self.scheduler = MetaScheduler(trace_store=self.storage)

    def run_episode(self, *, external_heat: float = 0.04) -> Dict[str, Any]:
        episode_id = f"episode-{uuid4()}"
        observation = self.world.observe()
        observation_ref = self.smg.add_observation(observation)

        temp_high = observation["temperature"] >= self.world.alarm_threshold
        proposition = "TEMP_HIGH" if temp_high else "TEMP_NORMAL"
        sign_main = self.smg.create_sign(
            proposition=proposition,
            observation_id=observation_ref.observation_id,
            metadata={"temperature": observation["temperature"]},
        )

        formula = "TEMP_HIGH -> ACTIVATE_COOLING"
        ast = self.lotf.parse(formula)
        self.lotf.check(ast, {"TEMP_HIGH": "bool", "ACTIVATE_COOLING": "bool"})

        intervention = "activate_cooling" if temp_high else "deactivate_cooling"
        counterfactual = self.world.simulate_counterfactual(
            intervention="deactivate_cooling", external_heat=external_heat
        )
        factual = self.world.factual_transition(
            intervention=intervention, external_heat=external_heat
        )

        sign_intervention = self.smg.create_sign(
            proposition="ACTIVATE_COOLING" if intervention == "activate_cooling" else "KEEP_IDLE",
            observation_id=observation_ref.observation_id,
            metadata={"intervention": intervention},
        )
        relation_kind = (
            "support"
            if factual["temperature"] <= counterfactual["temperature"]
            else "contradiction"
        )
        relation = self.smg.link_signs(
            source_sign_id=sign_main.sign_id,
            target_sign_id=sign_intervention.sign_id,
            kind=relation_kind,
            metadata={
                "factual_temperature": factual["temperature"],
                "counterfactual_temperature": counterfactual["temperature"],
            },
        )

        reasoning = self.scheduler.run(
            {
                "episode_id": episode_id,
                "run_id": self.run_id,
                "observation": observation,
                "intervention": intervention,
            }
        )
        episode_payload = {
            "episode_id": episode_id,
            "timestamp": utc_now_iso(),
            "context": {
                "observation": observation,
                "formula": formula,
                "intervention": intervention,
                "counterfactual": counterfactual,
            },
            "result": {
                "updated_world": factual,
                "relation_kind": relation_kind,
                "reasoning_sequence": reasoning["sequence"],
            },
            "trace": reasoning["trace"],
        }
        self.storage.append_event(
            event_type="episode.closed",
            payload=episode_payload,
            run_id=self.run_id,
            source="min_cognitive_episode",
        )

        artifact_blob = json.dumps(
            {
                "episode": episode_payload,
                "smg_snapshot": self.smg.snapshot(),
                "relation": asdict(relation),
            },
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
        )
        artifact = self.storage.materialize_artifact(
            run_id=self.run_id,
            kind="episode_report",
            content=artifact_blob,
            filename=f"{episode_id}.json",
            metadata={"episode_id": episode_id},
        )
        return {
            "episode": episode_payload,
            "smg_snapshot": self.smg.snapshot(),
            "reasoning": reasoning,
            "artifact": asdict(artifact),
            "run_id": self.run_id,
        }
