import json
from pathlib import Path
from scripts.build_p2_v2_bounded_closure import OBSERVABILITY_FIELDS, observability_contract, dump

def test_future_contract_has_all_fields_and_no_authority():
    contract=observability_contract()
    assert set(OBSERVABILITY_FIELDS)==set(contract["required_per_arm_unit_fields"])
    assert all(contract["invariants"].values())
    assert all(value is False for value in contract["authority"].values())
    assert contract["authorization_effect"]=="none"

def test_closure_serialization_is_byte_deterministic(tmp_path: Path):
    value={"z":1,"exact_treatment_geometry":{"status":"NOT_RECONSTRUCTIBLE_FROM_FROZEN_EVIDENCE"}}
    one,two=tmp_path/"one.json",tmp_path/"two.json"; dump(one,value); dump(two,value)
    assert one.read_bytes()==two.read_bytes()

def test_builder_module_has_no_backend_or_campaign_dependencies():
    import scripts.build_p2_v2_bounded_closure as module
    names=set(module.__dict__)
    assert "N3Adapter" not in names and "Mamba2TemporalTorchBackend" not in names
    assert "MemoryRetrieval" not in names and "derive_n3_shadow_directive" not in names
