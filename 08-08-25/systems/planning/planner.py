# aeon/systems/planning/planner.py
import logging
from ...state import AEONState
from ...protocols import PlannerProto, SystemStatus

log = logging.getLogger("AEON.Planner")

class AEONPlanner(PlannerProto):
    """
    Un planificador determinista y basado en reglas que implementa la
    Directiva Primaria de AEON:
    1. Supervivencia por encima de todo.
    2. Crecimiento si las condiciones son seguras.
    """
    def __init__(self, config=None):
        # El config podría usarse en el futuro para umbrales dinámicos.
        self.cfg = config or {}
        log.info("Planificador basado en reglas inicializado.")

    def decide(self, state: AEONState) -> str:
        """
        Aplica una jerarquía de reglas estricta para determinar la próxima acción.
        
        Args:
            state: El estado de conciencia completo de AEON.

        Returns:
            La acción recomendada como un string.
        """
        # --- REGLA 1: SUPERVIVENCIA INMEDIATA ---
        # Si el controlador homeostático reporta un estado crítico, la única
        # prioridad es reducir la carga para evitar daños.
        if state.metrics.system_status == SystemStatus.CRITICAL:
            log.warning(
                f"DECISIÓN: 'rest'. Estado del sistema CRÍTICO. "
                f"Temp: {state.metrics.temp_c}°C. Priorizando supervivencia."
            )
            return "rest"

        # --- REGLA 2: PRECAUCIÓN Y ESTABILIDAD ---
        # Si el estado es de advertencia, tomamos una medida menos drástica
        # para evitar llegar a un estado crítico.
        if state.metrics.system_status == SystemStatus.WARNING:
            log.info(
                f"DECISIÓN: 'pause'. Estado del sistema en ADVERTENCIA. "
                f"Temp: {state.metrics.temp_c}°C. Reduciendo estrés."
            )
            return "pause"
            
        # --- Lógica futura para estancamiento cognitivo iría aquí ---
        # ej: if self._is_stagnated(state.loss_history):
        #         return "trigger_evolution"

        # --- REGLA 3: ACCIÓN POR DEFECTO (CRECIMIENTO) ---
        # Si no hay amenazas a la integridad del sistema, la directiva es
        # continuar aprendiendo y mejorando.
        log.debug(f"DECISIÓN: 'train'. Estado NOMINAL. Procediendo con el entrenamiento.")
        return "train"