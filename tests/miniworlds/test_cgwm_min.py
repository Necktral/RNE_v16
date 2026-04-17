from runtime.world.cgwm_min import CGWMMin


def test_cgwm_counterfactual_does_not_mutate_state():
    world = CGWMMin(initial_temperature=0.84, alarm_threshold=0.85)
    before = world.observe()
    cf = world.simulate_counterfactual(intervention="activate_cooling", external_heat=0.02)
    after = world.observe()

    assert before == after
    assert cf["cooling_active"] is True


def test_cgwm_factual_transition_updates_state():
    world = CGWMMin(initial_temperature=0.9, alarm_threshold=0.85)
    next_state = world.factual_transition(intervention="activate_cooling", external_heat=0.02)
    assert next_state["cooling_active"] is True
    assert next_state["temperature"] < 0.9
