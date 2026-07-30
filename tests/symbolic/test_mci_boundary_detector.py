from __future__ import annotations

import random

from runtime.symbolic.mci.boundary_detector import detect_transition_boundaries
from runtime.symbolic.mci.causal_learning import TransitionEvidence


def _row(
    value: float,
    success: bool,
    *,
    action: str = "act",
    target: str = "outcome",
    logical_time: int = 0,
) -> TransitionEvidence:
    predicted = {target: 1.0}
    return TransitionEvidence(
        state={"signal": value, target: 0.0},
        action=action,
        external_input=0.0,
        predicted=predicted,
        observed={target: 1.0 if success else 0.0},
        logical_time=logical_time,
    )


def _clean(operator: str = ">") -> tuple[TransitionEvidence, ...]:
    values = (0.1, 0.2, 0.25, 0.29, 0.31, 0.35, 0.4, 0.5)
    return tuple(
        _row(value, value > 0.3 if operator == ">" else value < 0.3, logical_time=i)
        for i, value in enumerate(values)
    )


def test_detects_clean_boundary_at_point_three():
    candidates = detect_transition_boundaries(_clean(), variable="signal")
    assert any(item.expression == "signal > 0.3" for item in candidates)


def test_detection_is_invariant_to_input_order():
    rows = list(_clean())
    expected = detect_transition_boundaries(rows, variable="signal")
    random.Random(31).shuffle(rows)
    assert detect_transition_boundaries(rows, variable="signal") == expected


def test_actions_are_not_mixed():
    rows = _clean() + tuple(
        _row(value, value < 0.3, action="other") for value in (0.1, 0.2, 0.4, 0.5)
    )
    candidates = detect_transition_boundaries(rows, variable="signal", action="act")
    assert candidates
    assert {item.action for item in candidates} == {"act"}
    assert any(item.operator == ">" for item in candidates)


def test_effect_signatures_are_not_mixed():
    rows = _clean() + tuple(
        _row(value, value < 0.3, target="alternate")
        for value in (0.1, 0.2, 0.4, 0.5)
    )
    candidates = detect_transition_boundaries(rows, variable="signal")
    assert {item.effect_key for item in candidates} == {"outcome", "alternate"}


def test_repeated_values_do_not_create_spurious_boundaries():
    rows = _clean() + (_row(0.2, False), _row(0.2, False), _row(0.4, True))
    expressions = {
        item.expression for item in detect_transition_boundaries(rows, variable="signal")
    }
    assert "signal > 0.3" in expressions
    assert len(expressions) == 1


def test_isolated_noise_is_rejected_by_purity_and_separation():
    rows = list(_clean())
    rows.append(_row(0.28, True))
    candidates = detect_transition_boundaries(rows, variable="signal", min_support=2)
    assert len(candidates) <= 2
    assert candidates[0].operator == ">"


def test_insufficient_support_returns_no_boundary():
    rows = (_row(0.29, False), _row(0.31, True))
    assert not detect_transition_boundaries(
        rows, variable="signal", min_support=2
    )


def test_operator_is_inverted_when_success_is_below():
    candidates = detect_transition_boundaries(_clean("<"), variable="signal")
    assert any(item.expression == "signal < 0.3" for item in candidates)


def test_output_is_deterministic():
    rows = _clean()
    assert detect_transition_boundaries(rows, variable="signal") == (
        detect_transition_boundaries(rows, variable="signal")
    )


def test_effect_filter_selects_only_requested_signature():
    rows = _clean() + tuple(
        _row(value, value < 0.3, target="alternate")
        for value in (0.1, 0.2, 0.4, 0.5)
    )
    candidates = detect_transition_boundaries(
        rows, variable="signal", effect_key="outcome"
    )
    assert candidates
    assert {item.effect_key for item in candidates} == {"outcome"}
