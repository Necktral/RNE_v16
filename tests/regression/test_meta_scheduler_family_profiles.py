"""Regresiones para perfiles de familias en scheduler META."""

from __future__ import annotations

from runtime.reasoning.scheduler_meta.family_profiles import default_profile_for_mode
from runtime.reasoning.scheduler_meta.meta_scheduler import MetaScheduler


def test_core_only_profile_preserva_secuencia_core() -> None:
    scheduler = MetaScheduler(mode="fixed", family_profile="core_only")
    result = scheduler.run(
        {
            "run_id": "run-core-only",
            "contradiction_signal": 0.9,
            "edge_pressure": 0.9,
            "uncertainty": 0.9,
            "regime_hint": "viability_edge",
        }
    )
    assert result["family_profile"] == "core_only"
    assert result["sequence"] == ["ABD", "ANA", "CAU", "CTF", "DED", "PROB"]


def test_adaptive_ecology_activa_guardas_en_viability_edge() -> None:
    scheduler = MetaScheduler(mode="adaptive", family_profile="adaptive_family_ecology", max_steps=10)
    result = scheduler.run(
        {
            "run_id": "run-adaptive-viability",
            "regime_hint": "viability_edge",
            "observation": {"alarm": True, "world_level": 0.94},
            "contradiction_signal": 0.8,
            "edge_pressure": 0.85,
            "uncertainty": 0.7,
        }
    )
    seq = set(result["sequence"])
    assert "DIA_ADV" in seq or "FAL_GUARD" in seq
    assert result["mode"] == "adaptive"


def test_adaptive_mode_resuelve_v2_por_defecto() -> None:
    assert default_profile_for_mode("adaptive") == "adaptive_family_ecology_v2"

    scheduler = MetaScheduler(mode="adaptive", max_steps=10)
    result = scheduler.run(
        {
            "run_id": "run-adaptive-v2-default",
            "regime_hint": "heterogeneous_warning",
            "observation": {"alarm": True, "world_level": 0.90},
            "contradiction_signal": 0.82,
            "uncertainty": 0.75,
            "edge_pressure": 0.76,
            "continuity_recent": 0.55,
        }
    )

    assert result["family_profile"] == "adaptive_family_ecology_v2"
    assert result["sequence_validation"]["validated_passed"] is True
    assert result["mandatory_family_floor"] == ["ABD", "ANA", "CAU", "CTF", "DED"]
    assert {"ABD", "ANA", "CAU", "CTF", "DED", "PROB"}.issubset(set(result["sequence"]))


def test_adaptive_v2_inserta_overlays_sin_desplazar_backbone() -> None:
    scheduler = MetaScheduler(mode="adaptive", family_profile="adaptive_family_ecology_v2", max_steps=10)
    result = scheduler.run(
        {
            "run_id": "run-adaptive-v2-viability",
            "regime_hint": "viability_edge",
            "observation": {"alarm": True, "world_level": 0.95},
            "contradiction_signal": 0.85,
            "uncertainty": 0.78,
            "edge_pressure": 0.86,
            "continuity_recent": 0.50,
        }
    )

    sequence = result["sequence"]
    assert result["sequence_validation"]["validated_passed"] is True
    assert result["sequence_validation"]["optional_displacement_detected"] is False
    assert {"ABD", "ANA", "CAU", "CTF", "DED", "PROB"}.issubset(set(sequence))
    assert "DIA_ADV" in sequence
    assert "FAL_GUARD" in sequence
    assert sequence.index("DIA_ADV") > sequence.index("ANA")
    assert sequence.index("DIA_ADV") < sequence.index("CAU")
    assert sequence.index("FAL_GUARD") > sequence.index("DED")
    assert sequence.index("FAL_GUARD") < sequence.index("PROB")


def test_full_family_exploration_habilita_ind_si_hay_patron_sin_estructura() -> None:
    scheduler = MetaScheduler(mode="adaptive", family_profile="full_family_exploration", max_steps=10)
    result = scheduler.run(
        {
            "run_id": "run-full-explore",
            "regime_hint": "vram_favorable",
            "observation": {
                "temp_std": 0.20,
                "gradient_strength": 0.25,
                "hotspot_count": 2,
                "propositions": ["TEMP_HIGH"],  # baja estructura explícita
            },
            "uncertainty": 0.4,
            "edge_pressure": 0.3,
            "vram_opportunity_score": 0.8,
            "vram_headroom": 0.7,
        }
    )
    # IND queda ADMITIDA por el patrón sin estructura (entra en la propuesta);
    # bajo presupuesto 10 no caben las 12 familias y el recorte validado protege
    # el núcleo (sombra/deliberativas se descartan primero, DED nunca).
    assert "IND" in set(result["proposed_sequence"])
    executed = set(result["sequence"])
    assert {"ABD", "ANA", "CAU", "CTF", "DED", "PROB"} <= executed
