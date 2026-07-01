# cognitive_self_challenge.py v3.0 — AEON ∆
# Desafíos cognitivos/metacognitivos con integración física y cuántica

import random
import numpy as np
import time
import os
import sys

# Permite ejecutar el script directamente desde la raíz del proyecto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from runtime.core.event_bus import event_bus  # Integración EventBus centralizado
from runtime.core.epistemic_drift_predictor import EpistemicDriftPredictor  # Habilita predictor de deriva
from runtime.reasoning.metacognition_tracker import MetacognitionTracker

class CognitiveSelfChallengeAGI:
    def __init__(self, modules, monitor, quantum_states, challenge_interval=5000, drift_predictor=None, metacognition_tracker=None):
        """
        modules: dict de módulos internos AEON
        monitor: PhysicsAwareMonitor (monitorea VRAM, temp, entropía)
        quantum_states: dict de QuantumState por módulo
        challenge_interval: cada cuántos ciclos activar el desafío
        """
        self.modules = modules
        self.monitor = monitor
        self.quantum_states = quantum_states
        self.challenge_interval = challenge_interval
        self.last_cycle = 0
        self.drift_predictor = drift_predictor  # Puede ser None o instancia
        self.metacognition_tracker = metacognition_tracker or MetacognitionTracker()

    def generate_challenge(self, current_cycle):
        # Bloquea desafíos si hay sobrecarga física
        if hasattr(self.monitor, 'should_quarantine') and self.monitor.should_quarantine():
            event_bus.emit('cognitive_challenge_blocked', {
                'reason': 'physical_overload',
                'cycle': current_cycle,
                'timestamp': time.time()
            })
            print("[AEON] Desafío bloqueado por sobrecarga física.")
            return None
        if (current_cycle - self.last_cycle) < self.challenge_interval:
            return None
        self.last_cycle = current_cycle
        # Selecciona módulo con menor coherencia cuántica
        chosen_module_name = min(self.quantum_states, key=lambda k: getattr(self.quantum_states[k], 'coherence', 1.0))
        chosen_module = self.modules[chosen_module_name]
        qstate = self.quantum_states[chosen_module_name]
        # Decide tipo de desafío según estado cuántico y físico
        challenge_type = None
        outcome = None
        result = None
        if getattr(qstate, 'coherence', 1.0) < 0.2:
            event_bus.emit('cognitive_chaotic_resurrection', {
                'module': chosen_module_name,
                'cycle': current_cycle,
                'timestamp': time.time()
            })
            print(f"[AEON] Resurrección caótica en {chosen_module_name}")
            challenge_type = 'cognitive_chaotic_resurrection'
            outcome = 'success'  # Ejemplo: resurrección exitosa
            result = self._chaotic_resurrection(chosen_module, qstate)
        elif hasattr(self.monitor, 'resource_state') and self.monitor.resource_state.get('entropy', 0) > 0.8:
            event_bus.emit('cognitive_adversarial_perturbation', {
                'module': chosen_module_name,
                'cycle': current_cycle,
                'timestamp': time.time()
            })
            print(f"[AEON] Perturbación adversaria en {chosen_module_name}")
            challenge_type = 'cognitive_adversarial_perturbation'
            outcome = 'failure'  # Ejemplo: perturbación adversaria
            result = self._adversarial_perturbation(chosen_module)
        else:
            event_bus.emit('cognitive_semantic_noise', {
                'module': chosen_module_name,
                'cycle': current_cycle,
                'timestamp': time.time()
            })
            print(f"[AEON] Desafío cognitivo estándar en {chosen_module_name}")
            challenge_type = 'cognitive_semantic_noise'
            outcome = 'neutral'  # Ejemplo: ruido semántico
            result = self._semantic_noise(chosen_module)
        # Log metacognitivo robusto
        self.metacognition_tracker.log_challenge(
            challenge_type=challenge_type,
            cycle=current_cycle,
            outcome=outcome
        )

        # --- Integración de force_mutation si hay predictor de deriva ---
        if self.drift_predictor is not None:
            alert, reason = self.drift_predictor.check_drift(current_cycle)
            if alert:
                print("[AEON] Deriva epistémica detectada. Forzando mutación...")
                self.drift_predictor.force_mutation(reason)
                event_bus.emit('epistemic_drift_mutation', {
                    'reason': reason,
                    'cycle': current_cycle,
                    'timestamp': time.time()
                })
        return result

    def _chaotic_resurrection(self, module, qstate):
        if hasattr(module, 'weights'):
            module.weights = np.random.normal(0, 1, module.weights.shape)
            if hasattr(qstate, 'coherence'):
                qstate.coherence = 1.0
            print("  ↪︎ Resurrección caótica aplicada.")
            return module.weights
        return None

    def _adversarial_perturbation(self, module):
        if hasattr(module, 'weights'):
            perturb = np.random.uniform(-1, 1, module.weights.shape)
            module.weights += perturb * 0.2
            print("  ↪︎ Perturbación adversaria aplicada.")
            return module.weights
        return None

    def _semantic_noise(self, module):
        if hasattr(module, 'weights'):
            noise = np.random.normal(0, 0.1, size=module.weights.shape)
            module.weights += noise
            print("  ↪︎ Ruido semántico inyectado.")
            return module.weights
        return None

    def _metacognitive_question(self):
        question = "¿Qué tan confiable fue mi inferencia más reciente?"
        print(f"  ↪︎ Pregunta meta-cognitiva generada: {question}")
        return question

# Ejemplo de integración avanzada
if __name__ == '__main__':
    USE_REAL_ENV = True  # Cambia a False para usar los dummies

    if USE_REAL_ENV:
        print("\n[PRUEBA] Usando entorno REAL de AEON FENIX-Δ\n" + "-"*60)
        try:
            from runtime.evolution.meta_optimizer import PhysicsAwareMonitor, QuantumState, QuantumExponentialOptimizer, QuantumExponentialConfig

            # Instancia el optimizador cuántico real con configuración por defecto
            optimizer = QuantumExponentialOptimizer(QuantumExponentialConfig())
            # Fuerza la creación de un módulo antes de poblar los diccionarios
            optimizer._spawn_module("test_module")
            modules = optimizer.state['modules']
            quantum_states = {uid: optimizer.state['quantum_state'] for uid in modules}

            monitor = optimizer.physics_monitor  # Usa el monitor real del optimizador
            monitor.resource_state = {'vram': 0.5, 'thermal': 0.5, 'entropy': 0.5}  # Inicialización robusta

            challenger = CognitiveSelfChallengeAGI(modules, monitor, quantum_states, challenge_interval=1000)
            resultados = []
            for ciclo in range(1, 5001, 500):
                print(f"\n[CICLO {ciclo}]")
                resultado = challenger.generate_challenge(ciclo)
                if resultado is not None:
                    print(f"[AEON] Resultado del desafío en ciclo {ciclo}: {resultado}")
                    resultados.append((ciclo, resultado))
            print("\nResumen de resultados:")
            for ciclo, res in resultados:
                print(f"  Ciclo {ciclo}: {res}")
        except Exception as e:
            print(f"[ERROR] No se pudo ejecutar la prueba real: {e}")
            print("Cambia USE_REAL_ENV a False para usar los dummies.")
    else:
        # EventBus dummy para pruebas (si no hay uno real)
        class DummyEventBus:
            def __init__(self):
                self.events = []
            def emit(self, event_type, payload):
                print(f"[EventBus] {event_type}: {payload}")
                self.events.append((event_type, payload))
            def on(self, event_type, callback):
                pass  # No-op para pruebas
        import sys
        if 'event_bus' not in globals() or event_bus is None:
            event_bus = DummyEventBus()
            import types
            sys.modules['runtime.core.event_bus'] = types.ModuleType('event_bus_mod')
            sys.modules['runtime.core.event_bus'].event_bus = event_bus

        class DummyModule:
            def __init__(self):
                self.weights = np.random.randn(2)
            def __call__(self):
                return np.sum(self.weights)
        class DummyQuantumState:
            def __init__(self, coherence=1.0):
                self.coherence = coherence
        class DummyMonitor:
            def __init__(self):
                self.resource_state = {'vram': 0.5, 'thermal': 0.5, 'entropy': 0.5}
            def should_quarantine(self):
                return self.resource_state['vram'] > 0.95 or self.resource_state['thermal'] > 0.95

        modules_example = {'module_a': DummyModule(), 'module_b': DummyModule()}
        quantum_states = {'module_a': DummyQuantumState(0.15), 'module_b': DummyQuantumState(0.8)}
        monitor = DummyMonitor()
        challenger = CognitiveSelfChallengeAGI(modules_example, monitor, quantum_states, challenge_interval=1000)

        print("\n[PRUEBA] Ejecución de desafíos cognitivos/metacognitivos AEON FENIX-Δ\n" + "-"*60)
        resultados = []
        for ciclo in range(1, 5001, 500):  # Prueba cada 500 ciclos
            print(f"\n[CICLO {ciclo}]")
            resultado = challenger.generate_challenge(ciclo)
            if resultado is not None:
                print(f"[AEON] Resultado del desafío en ciclo {ciclo}: {resultado}")
                resultados.append((ciclo, resultado))
        print("\nResumen de resultados:")
        for ciclo, res in resultados:
            print(f"  Ciclo {ciclo}: {res}")
        if hasattr(event_bus, 'events'):
            print(f"\nEventos emitidos durante la prueba: {len(event_bus.events)}")
            for evt, payload in event_bus.events:
                print(f"  - {evt}: {payload}")
