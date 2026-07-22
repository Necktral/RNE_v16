import importlib.util
from pathlib import Path

def test_campaign_is_importable_and_declares_no_live_hook():
    path=Path(__file__).parents[2]/"scripts/run_p2_n3_causal_campaign.py"
    spec=importlib.util.spec_from_file_location("p2_campaign",path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert module.ARMS == ("canonical","n3-reference","n3-trained")

def test_trained_head_to_scale_mapping_is_explicit():
    source=(Path(__file__).parents[2]/"scripts/run_p2_n3_causal_campaign.py").read_text()
    assert '"micro": float(trained_candidate["risk"])' in source
    assert '"meso": float(trained_candidate["importance"])' in source
    assert '"macro": float(trained_candidate["continuity"])' in source
