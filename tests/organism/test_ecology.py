"""Tests de la ecología multi-organismo (selección por fitness certificado + transferencia)."""

from __future__ import annotations

from pathlib import Path

import pytest

from runtime.organism.ecology import (
    EcologyMember,
    OrganismEcology,
    TransferMode,
    build_member,
)
from runtime.organism.lineage import InheritanceRule, LineageState
from runtime.storage import StorageConfig, StorageFactory


def _storage(tmp_path: Path):
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "eco.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=False,
        strict_dual_write=False,
    )
    return StorageFactory.create_facade(config)


VEDGE = {
    "grid_size": 5,
    "topology": "hotspot_center",
    "initial_temperature": 0.95,
    "alarm_threshold": 0.90,
    "cooling_effect": 0.04,
}


def _ecology(storage, n=4, mode=TransferMode.REASONING_POLICY, seed=7):
    members = [
        build_member(
            member_id=f"m{i}",
            scenario="grid_thermal_5x5",
            scenario_kwargs=VEDGE,
            storage=storage,
        )
        for i in range(n)
    ]
    return OrganismEcology(members, storage=storage, transfer_mode=mode, seed=seed)


class TestGuardedFitness:
    def test_fitness_minus_inf_without_certified_episodes(self, tmp_path):
        storage = _storage(tmp_path)
        eco = _ecology(storage)
        member = eco.members[0]
        # Sin correr nada: no hay eventos ni certificados ⇒ fitness −inf.
        assert eco._fitness(member) == float("-inf")
        assert member.certified_episodes == 0

    def test_fitness_is_mean_reward_over_certified(self, tmp_path):
        storage = _storage(tmp_path)
        eco = _ecology(storage, n=2)
        eco.run_generation(episodes_per_member=4)
        for member in eco.members:
            # Tras una generación los supervivientes tienen episodios certificados.
            if member.certified_episodes > 0:
                assert member.fitness != float("-inf")
                assert -1.0 <= member.fitness <= 1.0


class TestSelectionAndReproduction:
    def test_population_size_preserved_after_generation(self, tmp_path):
        storage = _storage(tmp_path)
        eco = _ecology(storage, n=4)
        eco.run_generation(episodes_per_member=3)
        assert len(eco.members) == 4

    def test_reproduction_records_divergence_and_inherits(self, tmp_path):
        storage = _storage(tmp_path)
        eco = _ecology(storage, n=4)
        summary = eco.run_generation(episodes_per_member=3)
        offspring = summary["reproduction"]["offspring"]
        assert offspring, "debe haber descendencia tras culling"
        # Los hijos están en la población y su linaje registró divergencia.
        child_ids = {o["child_id"] for o in offspring}
        children = [m for m in eco.members if m.member_id in child_ids]
        assert children
        for child in children:
            assert child.lineage.has_diverged
            assert child.generation_born == 1

    def test_deterministic_with_same_seed(self, tmp_path):
        (tmp_path / "a").mkdir(exist_ok=True)
        (tmp_path / "b").mkdir(exist_ok=True)
        s1 = _storage(tmp_path / "a")
        s2 = _storage(tmp_path / "b")
        eco1 = _ecology(s1, n=4, seed=42)
        eco2 = _ecology(s2, n=4, seed=42)
        r1 = eco1.run_generation(episodes_per_member=3)
        r2 = eco2.run_generation(episodes_per_member=3)
        assert [o["mutated_knob"] for o in r1["reproduction"]["offspring"]] == [
            o["mutated_knob"] for o in r2["reproduction"]["offspring"]
        ]


class TestInheritanceGate:
    def test_check_inheritance_eligibility_actually_filters(self):
        # Activación REAL del código antes muerto: una regla fallida ⇒ no elegible.
        lineage = LineageState(
            lineage_id="L",
            inheritance_rules=(InheritanceRule("certified_safe", "certified_safe", ""),),
        )
        ok, failed = lineage.check_inheritance_eligibility(
            is_certified_safe=False,
            is_constitution_consistent=True,
            is_baseline_preserved=True,
            is_contamination_free=True,
        )
        assert ok is False and "certified_safe" in failed

    def test_ineligible_parent_emits_block_event(self, tmp_path):
        storage = _storage(tmp_path)
        eco = _ecology(storage, n=2)
        # Forzar padre no elegible: rollback_ancestry no vacío ⇒ contamination.
        parent = eco.members[0]
        parent.lineage.rollback_ancestry.append("rb-1")
        parent.certified_episodes = 0  # no certified_safe
        child = eco._reproduce(parent)
        assert child.lineage.has_diverged
        events = [e.event_type for e in storage.list_events(run_id="ecology", limit=200)]
        assert "ecology.inheritance_blocked" in events


class TestKnowledgeTransfer:
    def test_share_emits_event(self, tmp_path):
        storage = _storage(tmp_path)
        eco = _ecology(storage, n=3, mode=TransferMode.REASONING_POLICY)
        eco.run_generation(episodes_per_member=3)
        events = [e.event_type for e in storage.list_events(run_id="ecology", limit=400)]
        assert "ecology.knowledge_shared" in events

    def test_isolated_mode_does_not_share(self, tmp_path):
        storage = _storage(tmp_path)
        eco = _ecology(storage, n=3, mode=TransferMode.ISOLATED)
        eco.run_generation(episodes_per_member=3)
        events = [e.event_type for e in storage.list_events(run_id="ecology", limit=400)]
        assert "ecology.knowledge_shared" not in events

    def test_adversarial_morphism_blocks_cross_scenario_transfer(self, tmp_path):
        # thermal (minimize) → resource (maximize): morfismo adversarial ⇒ merge 0.
        storage = _storage(tmp_path)
        thermal = build_member(
            member_id="t", scenario="thermal_homeostasis", storage=storage
        )
        resource = build_member(
            member_id="r", scenario="resource_management", storage=storage
        )
        eco = OrganismEcology(
            [thermal, resource], storage=storage, transfer_mode=TransferMode.REASONING_POLICY
        )
        # Sembrar evidencia en el donante térmico.
        for _ in range(3):
            thermal.selector.observe(
                run_id=thermal.run_id, reward_block={"reward": 0.1},
                executed_sequence=["ABD", "PLAN"], regime="thermal_homeostasis",
            )
        before = resource.selector.summary(resource.run_id)["n_observations"]
        eco._share_knowledge()
        after = resource.selector.summary(resource.run_id)["n_observations"]
        # El morfismo thermal→resource es adversarial ⇒ no se transfiere nada.
        assert after == before


class TestLiveEcology:
    def test_two_generations_produce_ranking_and_events(self, tmp_path):
        storage = _storage(tmp_path)
        eco = _ecology(storage, n=3)
        eco.run_generation(episodes_per_member=4)
        summary = eco.run_generation(episodes_per_member=4)
        assert summary["generation"] == 2
        assert summary["ranking"]
        pop = eco.population_summary()
        assert pop["schema"] == "ecology.v1"
        assert pop["population_size"] == 3
        # Cierre intacto: la ecología no descertifica episodios sanos.
        events = [e.event_type for e in storage.list_events(run_id="ecology", limit=400)]
        assert "ecology.generation" in events
