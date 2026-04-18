from __future__ import annotations

from runtime.organism.failure_atlas import detect_failure_atlas
from runtime.organism.regime_model import RESOURCE_REGIME, THERMAL_REGIME
from runtime.organism.regime_renormalization import RegimeRenormalizationEngine
from runtime.organism.risk_process import ConstitutionalRiskProcess
from runtime.organism.snapshot import OrganismSnapshot
from runtime.organism.state import OrganismState


def test_renormalization_is_asymmetric_and_transforms_constraints() -> None:
    engine = RegimeRenormalizationEngine()
    snap = OrganismSnapshot.from_state(OrganismState())

    ab = engine.renormalize(
        source_regime=THERMAL_REGIME,
        target_regime=RESOURCE_REGIME,
        snapshot=snap,
        constraints={"min_memory_purity": 0.4, "max_policy_drift": 0.5},
    )
    ba = engine.renormalize(
        source_regime=RESOURCE_REGIME,
        target_regime=THERMAL_REGIME,
        snapshot=snap,
        constraints={"min_memory_purity": 0.4, "max_policy_drift": 0.5},
    )

    assert ab.renormalization_map.asymmetry_factor != ba.renormalization_map.asymmetry_factor
    assert ab.constraint_transform.transformed_constraints != ba.constraint_transform.transformed_constraints
    assert 0.0 <= ab.regime_residual.residual_error <= 1.0


def test_risk_process_accumulates_and_produces_profiles() -> None:
    proc = ConstitutionalRiskProcess()

    u1 = proc.update(
        scope_type="organism",
        scope_key="run-1",
        drift_identity=0.20,
        drift_policy=0.10,
        delta_viability=-0.10,
        delta_purity=0.05,
        delta_modification=0.00,
        erosion=0.20,
        renorm_residual=0.10,
    )
    u2 = proc.update(
        scope_type="organism",
        scope_key="run-1",
        drift_identity=0.45,
        drift_policy=0.35,
        delta_viability=-0.35,
        delta_purity=0.30,
        delta_modification=0.20,
        erosion=0.60,
        renorm_residual=0.50,
    )

    assert u2.updated_risk >= u1.updated_risk

    proc.update(
        scope_type="edge",
        scope_key="homeostatic_cooling->inventory_maximization",
        drift_identity=0.30,
        drift_policy=0.40,
        delta_viability=-0.20,
        delta_purity=0.15,
        delta_modification=0.10,
        erosion=0.40,
        renorm_residual=0.55,
    )

    edges, mods, inheritance = proc.to_profiles()
    assert len(edges) >= 1
    assert mods == []
    assert inheritance == []


def test_failure_atlas_detects_required_failure_classes() -> None:
    atlas = detect_failure_atlas(
        drift_identity=0.8,
        drift_policy=0.9,
        delta_viability=0.2,
        memory_purity=0.2,
        modification_impact=0.9,
        erosion=0.9,
        renorm_residual=0.9,
    )
    names = {event.name for event in atlas.events}

    assert "constitutional_erosion" in names
    assert "policy_bifurcation" in names
    assert "latent_contamination" in names
    assert "adversarial_transfer_illusion" in names
    assert "identity_fracture" in names
    assert "inheritance_corruption" in names
