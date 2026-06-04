from __future__ import annotations

import os

import pytest

from runtime.reasoning.families import ext_open_thinker


@pytest.mark.requires_cuda
def test_open_thinker_gguf_smoke_if_configured() -> None:
    required = ["RNFE_REASONING_GGUF", "RNFE_LLAMA_CLI_CUDA"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        pytest.skip(f"external reasoner env missing: {','.join(missing)}")

    result = ext_open_thinker.execute(
        {
            "observation": {
                "global_temp_mean": 0.88,
                "global_temp_max": 0.88,
                "alarm": True,
                "propositions": ["TEMP_HIGH"],
            },
            "updated_world": {"global_temp_mean": 0.8816, "alarm": True},
            "counterfactual": {"global_temp_mean": 0.8116, "alarm": False},
            "intervention": "deactivate_cooling",
            "formula": "TEMP_HIGH -> ACTIVATE_COOLING",
            "regime_hint": "causal_counterfactual_conflict",
            "scenario_metadata": {"interventions": ["activate_cooling", "deactivate_cooling"]},
            "external_reasoner_backend": "cuda",
        }
    )
    assert result["ok"] is True
    assert result["state_delta"]["external_reasoner_ok"] is True
    assert result["state_delta"]["external_reasoner_schema_validated"] is True
    assert result["state_delta"]["external_reasoner_generation_tps"] >= 0.0
