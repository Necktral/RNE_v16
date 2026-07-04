# metacognition_tracker.py — AEON ∆ v1.0
# Registro de desafíos cognitivos superados y activación de autoconciencia estructural

import time
from typing import List, Dict

class MetacognitionTracker:
    def __init__(self):
        self.history: List[Dict] = []
        self.success_count: int = 0
        self.failure_count: int = 0

    def log_challenge(self, challenge_type: str, cycle: int, outcome: str):
        """
        Registra un desafío cognitivo y su resultado.
        outcome: 'success' | 'failure' | 'neutral'
        """
        self.history.append({
            "type": challenge_type,
            "cycle": cycle,
            "outcome": outcome,
            "timestamp": time.time()
        })
        if outcome == "success":
            self.success_count += 1
        elif outcome == "failure":
            self.failure_count += 1

    def summary(self):
        total = len(self.history)
        success_rate = (self.success_count / total) if total else 0.0
        return {
            "total_challenges": total,
            "success_rate": round(success_rate, 3),
            "successes": self.success_count,
            "failures": self.failure_count
        }

    def requires_intervention(self, threshold=0.3):
        """
        Devuelve True si la tasa de éxito es demasiado baja.
        """
        stats = self.summary()
        return stats['success_rate'] < threshold

if __name__ == "__main__":
    tracker = MetacognitionTracker()
    tracker.log_challenge("epistemic_contradiction", 100, "success")
    tracker.log_challenge("semantic_noise", 105, "failure")
    tracker.log_challenge("memory_conflict", 110, "success")

    print(tracker.summary())
    if tracker.requires_intervention():
        print("[META] ⚠️ Nivel de éxito bajo — activar asistencia cognitiva.")
