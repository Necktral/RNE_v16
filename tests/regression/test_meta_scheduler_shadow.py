from copy import deepcopy

from runtime.reasoning.scheduler_meta.meta_scheduler import MetaScheduler
from runtime.reasoning.families import core_inference


class _NoWriteTraceStore:
    def append_reasoning_trace(self, **_kwargs):  # pragma: no cover - must never run
        raise AssertionError("shadow scheduler persisted a reasoning trace")


def test_run_shadow_isolated_and_blocks_external_reasoner() -> None:
    context = {
        "run_id": "canonical-run",
        "observation": {"alarm": True},
        "scenario_metadata": {"interventions": ["a", "b"]},
        "nested": {"items": [1, 2]},
    }
    original = deepcopy(context)
    scheduler = MetaScheduler(
        sequence=["ext_open_thinker"],
        trace_store=_NoWriteTraceStore(),
        mode="fixed",
    )

    result = scheduler.run_shadow(context)

    assert context == original
    assert result["sequence"] == ["EXT_OPEN_THINKER"]
    assert result["state"]["external_reasoner_used"] is False
    assert result["state"]["shadow_skip_reason"] == "external_family_forbidden"
    assert result["shadow_execution"] == {
        "schema_version": "rnfe-shadow-reasoning-v1",
        "trace_persisted": False,
        "external_reasoner_allowed": False,
        "authority_effect": "none",
        "writes_performed": False,
    }


def test_run_shadow_can_select_bounded_revision_profile() -> None:
    scheduler = MetaScheduler(mode="fixed")
    result = scheduler.run_shadow(
        {
            "observation": {"alarm": False},
            "intervention": "activate_cooling",
            "formula": "temperature > 0.8",
            "scenario_metadata": {
                "interventions": ["activate_cooling", "deactivate_cooling"]
            },
        },
        family_profile="core_plus_ind",
    )

    assert "IND" in result["sequence"]
    assert result["state"]["_shadow_execution"] is True
    assert result["shadow_execution"]["writes_performed"] is False


def test_run_shadow_blocks_indirect_core_llm_augmentation(monkeypatch) -> None:
    class _ExplodingClient:
        def generate(self, *_args, **_kwargs):  # pragma: no cover - must never run
            raise AssertionError("shadow core family called external reasoner")

    monkeypatch.setenv("RNFE_CORE_FAMILIES_LLM", "1")
    monkeypatch.setattr(core_inference, "_detect_conflict", lambda _state: True)
    scheduler = MetaScheduler(sequence=["cau", "ctf", "prob"], mode="fixed")

    result = scheduler.run_shadow(
        {
            "observation": {"temperature": 0.9},
            "intervention": "activate_cooling",
            "formula": "temperature > 0.8",
            "updated_world": {"temperature": 0.7},
            "scenario_metadata": {"interventions": ["activate_cooling"]},
            "_external_reasoner_client": _ExplodingClient(),
        }
    )

    assert result["state"].get("_core_llm_augmentation") is None
