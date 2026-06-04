from __future__ import annotations

import json

from runtime.reasoning.families import ext_open_thinker


class FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.prompt = None
        self.kwargs = None

    def generate(self, prompt, **kwargs):
        self.prompt = prompt
        self.kwargs = kwargs
        return self.payload


def _state(client: FakeClient) -> dict:
    return {
        "observation": {"world_level": 0.88, "alarm": True, "propositions": ["TEMP_HIGH"]},
        "updated_world": {"world_level": 0.90, "alarm": True},
        "counterfactual": {"world_level": 0.81, "alarm": False},
        "intervention": "deactivate_cooling",
        "formula": "TEMP_HIGH -> ACTIVATE_COOLING",
        "regime_hint": "causal_counterfactual_conflict",
        "scenario_metadata": {"interventions": ["activate_cooling", "deactivate_cooling"]},
        "_external_reasoner_client": client,
    }


def test_adapter_accepts_valid_json_output() -> None:
    payload = {
        "claim": "activation lowers the thermal variable",
        "reasoning_summary": "counterfactual transition is cooler",
        "candidate_hypotheses": ["cooling reduces global_temp_mean"],
        "causal_assumptions": ["lower temperature is better"],
        "counterfactual_checks": ["activate_cooling has lower world_level"],
        "confidence_proxy": 0.72,
        "recommended_intervention": "activate_cooling",
    }
    result = ext_open_thinker.execute(
        _state(
            FakeClient(
                {
                    "ok": True,
                    "backend": "cuda",
                    "output_text": json.dumps(payload),
                    "latency_s": 0.4,
                    "generation_tps": 51.2,
                }
            )
        )
    )
    delta = result["state_delta"]
    assert result["status"] == "ok"
    assert delta["external_reasoner_ok"] is True
    assert delta["external_reasoner_schema_validated"] is True
    assert delta["external_reasoner_recommended_intervention"] == "activate_cooling"
    assert delta["external_reasoner_generation_tps"] == 51.2


def test_parser_accepts_short_valid_json() -> None:
    payload = {
        "claim": "cooler",
        "reasoning_summary": "alt is cooler",
        "candidate_hypotheses": ["cooling works"],
        "causal_assumptions": ["temp should fall"],
        "counterfactual_checks": ["alt lower"],
        "confidence_proxy": 0.8,
        "recommended_intervention": "activate_cooling",
    }
    parsed = ext_open_thinker.parse_external_reasoner_payload(
        json.dumps(payload),
        allowed_interventions=["activate_cooling", "deactivate_cooling"],
    )
    assert parsed["claim"] == "cooler"
    assert parsed["recommended_intervention"] == "activate_cooling"


def test_compact_prompt_is_explicit_and_shorter() -> None:
    payload = {
        "claim": "activation is cooler",
        "reasoning_summary": "counterfactual transition is cooler",
        "candidate_hypotheses": ["cooling reduces thermal load"],
        "causal_assumptions": ["lower temperature is better"],
        "counterfactual_checks": ["activation is cooler"],
        "confidence_proxy": 0.9,
        "recommended_intervention": "activate_cooling",
    }
    client = FakeClient(
        {
            "ok": True,
            "backend": "cuda",
            "output_text": json.dumps(payload),
            "latency_s": 0.2,
            "prompt_tps": 90.0,
            "generation_tps": 50.0,
        }
    )
    compact_state = _state(client)
    compact_state["external_reasoner_prompt_style"] = "compact"

    result = ext_open_thinker.execute(compact_state)
    standard_prompt = ext_open_thinker._build_prompt(_state(FakeClient({})))

    assert result["status"] == "ok"
    assert client.prompt is not None
    assert '"task":"choose_intervention_json"' in client.prompt
    assert len(client.prompt) < len(standard_prompt)
    assert result["state_delta"]["external_reasoner_prompt_style"] == "compact"
    assert result["state_delta"]["external_reasoner_prompt_tps"] == 90.0


def test_adapter_rejects_text_without_json() -> None:
    result = ext_open_thinker.execute(
        _state(
            FakeClient(
                {
                    "ok": True,
                    "backend": "cuda",
                    "output_text": "not json",
                    "latency_s": 0.1,
                    "generation_tps": 2.0,
                }
            )
        )
    )
    assert result["status"] == "error"
    assert result["failure_mode"] == "no_json_object_found"
    assert result["state_delta"]["external_reasoner_ok"] is False
    assert result["state_delta"]["external_reasoner_schema_validated"] is False


def test_adapter_rejects_unallowed_intervention() -> None:
    payload = {
        "claim": "do something else",
        "reasoning_summary": "unsupported intervention is not allowed",
        "candidate_hypotheses": ["unsupported action"],
        "causal_assumptions": ["unsupported action changes state"],
        "counterfactual_checks": ["unsupported action is not in intervention set"],
        "confidence_proxy": 0.8,
        "recommended_intervention": "open_window",
    }
    result = ext_open_thinker.execute(
        _state(
            FakeClient(
                {
                    "ok": True,
                    "backend": "cuda",
                    "output_text": json.dumps(payload),
                    "latency_s": 0.1,
                    "generation_tps": 2.0,
                }
            )
        )
    )
    assert result["status"] == "error"
    assert result["failure_mode"] == "invalid_intervention"
    assert result["state_delta"]["external_reasoner_error_type"] == "invalid_intervention"


def test_adapter_rejects_empty_required_field() -> None:
    payload = {
        "claim": "",
        "reasoning_summary": "has summary",
        "candidate_hypotheses": ["h"],
        "causal_assumptions": ["a"],
        "counterfactual_checks": ["c"],
        "confidence_proxy": 0.8,
        "recommended_intervention": "activate_cooling",
    }
    result = ext_open_thinker.execute(
        _state(
            FakeClient(
                {
                    "ok": True,
                    "backend": "cuda",
                    "output_text": json.dumps(payload),
                    "latency_s": 0.1,
                    "generation_tps": 2.0,
                }
            )
        )
    )
    assert result["status"] == "error"
    assert result["failure_mode"] == "empty_required_field"


def test_adapter_rejects_confidence_out_of_range() -> None:
    payload = {
        "claim": "activation is cooler",
        "reasoning_summary": "has summary",
        "candidate_hypotheses": ["h"],
        "causal_assumptions": ["a"],
        "counterfactual_checks": ["c"],
        "confidence_proxy": 1.4,
        "recommended_intervention": "activate_cooling",
    }
    result = ext_open_thinker.execute(
        _state(
            FakeClient(
                {
                    "ok": True,
                    "backend": "cuda",
                    "output_text": json.dumps(payload),
                    "latency_s": 0.1,
                    "generation_tps": 2.0,
                }
            )
        )
    )
    assert result["status"] == "error"
    assert result["failure_mode"] == "confidence_out_of_range"


def test_adapter_rejects_unknown_fields() -> None:
    payload = {
        "claim": "activation is cooler",
        "reasoning_summary": "has summary",
        "candidate_hypotheses": ["h"],
        "causal_assumptions": ["a"],
        "counterfactual_checks": ["c"],
        "confidence_proxy": 0.8,
        "recommended_intervention": "activate_cooling",
        "free_text": "not allowed",
    }
    result = ext_open_thinker.execute(
        _state(
            FakeClient(
                {
                    "ok": True,
                    "backend": "cuda",
                    "output_text": json.dumps(payload),
                    "latency_s": 0.1,
                    "generation_tps": 2.0,
                }
            )
        )
    )
    assert result["status"] == "error"
    assert result["failure_mode"] == "schema_validation_error"


def test_adapter_ignores_echoed_input_json_before_response() -> None:
    echoed_prompt = json.dumps(
        {
            "task": "Act as an auxiliary verifier",
            "required_schema": {"claim": "short claim"},
            "allowed_interventions": ["activate_cooling", "deactivate_cooling"],
        }
    )
    response = json.dumps(
        {
            "claim": "activation is cooler",
            "reasoning_summary": "activation gives lower temperature",
            "candidate_hypotheses": ["activate cooling"],
            "causal_assumptions": ["cooling lowers thermal state"],
            "counterfactual_checks": ["alternative is cooler"],
            "confidence_proxy": 0.8,
            "recommended_intervention": "activate_cooling",
        }
    )
    result = ext_open_thinker.execute(
        _state(
            FakeClient(
                {
                    "ok": True,
                    "backend": "cuda",
                    "output_text": echoed_prompt + "\n<think>...\n" + response,
                    "latency_s": 0.1,
                    "generation_tps": 2.0,
                }
            )
        )
    )
    assert result["status"] == "ok"
    assert result["state_delta"]["external_reasoner_recommended_intervention"] == "activate_cooling"
    assert "<think>" not in result["state_delta"]["external_reasoner_raw_output_excerpt"]
