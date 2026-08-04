import pytest

from runtime.neural import observability
from runtime.neural.observability import (
    ShadowSpanCollector,
    latency_attribution,
    observed_span,
    shadow_observation_scope,
)


def test_nested_exclusive_spans_reconcile_without_negative_duration(monkeypatch) -> None:
    ticks = iter((0, 10, 90, 100))
    monkeypatch.setenv("RNFE_SHADOW_CAUSAL_OBSERVABILITY", "1")
    monkeypatch.setattr(observability.time, "monotonic_ns", lambda: next(ticks))
    collector = ShadowSpanCollector(trace_parent="trace-parent")
    with shadow_observation_scope(collector):
        root = collector.begin("life_step_total")
        child = collector.begin("neural_coordination")
        collector.end(child)
        collector.end(root)
    spans = collector.spans()
    assert all(span["duration_ns"] >= 0 for span in spans)
    assert all(span["exclusive_duration_ns"] >= 0 for span in spans)
    report = latency_attribution(spans)
    assert report["life_step_total_ns"] == 100
    assert report["attributed_ns"] == 80
    assert report["uninstrumented_remainder_ns"] == 20
    assert report["coverage"] == 0.8
    assert report["reconciled"] is True


def test_observability_is_off_by_default(monkeypatch) -> None:
    monkeypatch.delenv("RNFE_SHADOW_CAUSAL_OBSERVABILITY", raising=False)
    collector = ShadowSpanCollector(trace_parent="trace-parent")
    with shadow_observation_scope(collector):
        @observed_span("life_step_total")
        def unchanged() -> str:
            return "canonical"

        assert unchanged() == "canonical"
    assert collector.spans() == []


def test_observability_preserves_exceptions(monkeypatch) -> None:
    monkeypatch.setenv("RNFE_SHADOW_CAUSAL_OBSERVABILITY", "1")
    collector = ShadowSpanCollector(trace_parent="trace-parent")
    with pytest.raises(RuntimeError, match="functional_failure"):
        with shadow_observation_scope(collector):
            @observed_span("postgres_event_persistence")
            def failing() -> None:
                raise RuntimeError("functional_failure")

            failing()
    [span] = collector.spans()
    assert span["status"] == "error"
    assert span["error_code"] == "RuntimeError"
