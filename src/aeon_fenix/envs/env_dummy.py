class DummyEnv:
    """Stub mínimo para compatibilidad histórica."""

    def reset(self):
        return {}

    def step(self, action):
        return {}, 0.0, True, {"action": action}

