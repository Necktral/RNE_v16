from runtime.control.msrc import CrossScaleMemoryGuard


def test_memory_guard_blocks_prohibited_dense_fields():
    guard = CrossScaleMemoryGuard()
    report = guard.sanitize_for_cross_scale(
        source_scale_id="5x5",
        target_scale_id="1x1",
        payload={
            "world_level": 0.8,
            "viability_margin": 0.2,
            "cell_states": [{"row": 0, "col": 0, "temperature": 0.8}],
        },
    )
    assert report.contamination_detected is True
    assert report.blocked_fields_count >= 1
    assert "cell_states" in report.blocked_fields


def test_memory_guard_keeps_allowed_invariants_only():
    guard = CrossScaleMemoryGuard()
    report = guard.sanitize_for_cross_scale(
        source_scale_id="1x1",
        target_scale_id="5x5",
        payload={
            "world_level": 0.75,
            "viability_margin": 0.1,
            "episode_signature": "ep-1",
            "raw_grid": [[0.1]],
        },
    )
    sanitized = report.sanitized_payload
    assert sanitized["world_level"] == 0.75
    assert sanitized["viability_margin"] == 0.1
    assert sanitized["episode_signature"] == "ep-1"
    assert "raw_grid" not in sanitized


def test_memory_guard_contamination_rate_aggregation():
    guard = CrossScaleMemoryGuard()
    reports = [
        guard.sanitize_for_cross_scale(
            source_scale_id="1x1",
            target_scale_id="5x5",
            payload={"world_level": 0.8},
        ),
        guard.sanitize_for_cross_scale(
            source_scale_id="5x5",
            target_scale_id="1x1",
            payload={"world_level": 0.7, "cell_states": []},
        ),
    ]
    rate = guard.compute_contamination_rate(reports)
    assert 0.0 < rate <= 1.0
