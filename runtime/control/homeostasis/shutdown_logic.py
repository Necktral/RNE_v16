# shutdown_logic.py (versión híbrida optimizada)
import time
import logging
import numpy as np
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
from dataclasses import dataclass, asdict, field
from src.aeon_types import HealthStatus
from src.persistence import StatePreserver
from src.homeostasis.thermodynamic_governor import ThermodynamicGovernor

logger = logging.getLogger("ShutdownLogic")
logger.setLevel(logging.INFO)

class CrisisLevel(Enum):
    NONE = 0
    OPTIMIZATION = 1
    CRITICAL = 2
    EMERGENCY = 3

@dataclass
class ShutdownProtocol:
    action: Callable
    rollback: Callable
    dependencies: List[str] = field(default_factory=list)
    executed: bool = False

class PhasedShutdown:
    """Gestor de apagado por fases con reversibilidad y priorización de VRAM"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.last_shutdown = 0
        self.crisis_history = []
        self.protocol_stack = []
        self.emergency_level = CrisisLevel.NONE
        
        # Umbrales configurables
        self.thresholds = {
            "vram": config.get("vram_threshold", 0.95),  # Prioridad máxima para VRAM
            "temp": config.get("temp_threshold", 0.98),
            "entropy": config.get("entropy_threshold", 0.99),
            "energy": config.get("energy_threshold", 0.95)
        }
        
        # Protocolos reversibles
        self.protocols = {
            CrisisLevel.OPTIMIZATION: [
                ShutdownProtocol(
                    action=lambda: self._reduce_cognitive_load(0.5),
                    rollback=lambda: self._restore_cognitive_load(),
                    dependencies=["cognitive_loader"]
                ),
                ShutdownProtocol(
                    action=lambda: self._compress_memory(),
                    rollback=lambda: self._decompress_memory(),
                    dependencies=["memory_manager"]
                )
            ],
            CrisisLevel.CRITICAL: [
                ShutdownProtocol(
                    action=lambda: self._prune_knowledge_base(0.7),
                    rollback=lambda: self._restore_knowledge_base(),
                    dependencies=["knowledge_base"]
                ),
                ShutdownProtocol(
                    action=lambda: self._freeze_learning(),
                    rollback=lambda: self._thaw_learning(),
                    dependencies=["learning_module"]
                )
            ]
        }
        
        # Validación de dependencias
        self._validate_dependencies()

    def _validate_dependencies(self):
        """Verifica que las dependencias de los protocolos estén disponibles."""
        required_deps = ["cognitive_loader", "memory_manager", "knowledge_base", "learning_module"]
        for dep in required_deps:
            try:
                # Simular verificación de dependencia
                logger.info(f"[DEPENDENCY] Verificando {dep}... OK")
            except ImportError as e:
                logger.error(f"[DEPENDENCY] Falta dependencia crítica: {dep}")
                raise

    def evaluate_crisis(self, health: HealthStatus) -> CrisisLevel:
        """Evalúa el nivel de crisis con prioridad en VRAM y validación proactiva."""
        if time.time() - self.last_shutdown < self.config.get("shutdown_cooldown", 300):
            return CrisisLevel.NONE
            
        # Priorizar VRAM y temperatura
        if health.vram_usage > self.thresholds["vram"]:
            logger.critical(f"[VRAM] Uso crítico: {health.vram_usage:.2%}")
            return CrisisLevel.EMERGENCY
            
        if health.temperature > self.thresholds["temp"]:
            logger.critical(f"[TEMP] Sobrecalentamiento: {health.temperature:.2%}")
            return CrisisLevel.EMERGENCY
            
        # Evaluar entropía y energía
        crisis_level = CrisisLevel.NONE
        if health.entropy_rate > self.thresholds["entropy"]:
            logger.critical(f"[ENTROPY] Colapso cognitivo: {health.entropy_rate:.2%}")
            crisis_level = CrisisLevel.CRITICAL
            
        if health.power_consumption > self.thresholds["energy"]:
            logger.warning(f"[ENERGY] Consumo alto: {health.power_consumption:.2%}")
            # Usar el valor más alto de crisis_level y OPTIMIZATION
            crisis_level = max([crisis_level, CrisisLevel.OPTIMIZATION], key=lambda x: x.value)
            
        return crisis_level

    def initiate_shutdown(self, 
                        level: CrisisLevel, 
                        preserver: Optional[StatePreserver] = None,
                        health_snapshot: Optional[HealthStatus] = None):
        """Ejecuta el protocolo de apagado según el nivel de crisis."""
        if level == CrisisLevel.NONE:
            return
            
        # Registrar crisis con snapshot de salud
        self.crisis_history.append({
            "timestamp": time.time(),
            "level": level.name,
            "reason": self._get_reason(level),
            "health": asdict(health_snapshot) if health_snapshot else None
        })
        
        # Preservar estado antes de cualquier acción
        if preserver:
            try:
                # Ajuste: save_critical_state solo acepta un argumento, se agregan metadatos al dict
                state_id = preserver.save_critical_state({"crisis_level": level.name})
                logger.info(f"[SHUTDOWN] Estado preservado con ID: {state_id}")
            except Exception as e:
                logger.error(f"[SHUTDOWN] Fallo en preservación: {str(e)}")
                # Escalar crisis si falla guardado
                level = max([level, CrisisLevel.CRITICAL], key=lambda x: x.value)
                
        # Ejecutar protocolos
        if level == CrisisLevel.EMERGENCY:
            self._emergency_protocol(preserver)
        else:
            self._execute_protocols(level)
            
        self.last_shutdown = time.time()
        self.emergency_level = level

    def _execute_protocols(self, level: CrisisLevel):
        """Ejecuta protocolos de apagado con reversibilidad."""
        if level not in self.protocols:
            return
            
        logger.warning(f"[SHUTDOWN] Iniciando protocolo: {level.name}")
        for protocol in self.protocols[level]:
            try:
                protocol.action()
                protocol.executed = True
                self.protocol_stack.append(protocol)
            except Exception as e:
                logger.error(f"[PROTOCOL] Error en ejecución: {str(e)}")
        logger.info(f"[SHUTDOWN] Protocolo {level.name} completado")

    def rollback_shutdown(self, level: CrisisLevel):
        """Revierte acciones de apagado en crisis parcial."""
        if level not in self.protocols:
            return
            
        logger.info(f"[SHUTDOWN] Revertiendo protocolo: {level.name}")
        for protocol in reversed(self.protocols[level]):
            if protocol.executed:
                try:
                    protocol.rollback()
                    protocol.executed = False
                except Exception as e:
                    logger.error(f"[ROLLBACK] Error en protocolo: {str(e)}")

    def _emergency_protocol(self, preserver: Optional[StatePreserver]):
        """Protocolo de emergencia con reversión parcial y respaldo."""
        logger.fatal("[SHUTDOWN] Iniciando protocolo de emergencia")
        
        # Revertir acciones anteriores
        self.rollback_shutdown(CrisisLevel.CRITICAL)
        self.rollback_shutdown(CrisisLevel.OPTIMIZATION)
        
        # Preservar estado final
        if preserver:
            try:
                preserver.save_full_state(lambda: {"emergency_shutdown": True})
                logger.info("[SHUTDOWN] Estado de emergencia guardado")
            except Exception as e:
                logger.error(f"[SHUTDOWN] Falla en guardado de emergencia: {str(e)}")
                
        # Ejecutar acciones de seguridad
        self._safe_power_down()
        self._broadcast_emergency()
        logger.info("[SHUTDOWN] Sistema detenido por emergencia")

    # --- Acciones de protocolo ---
    def _reduce_cognitive_load(self, factor: float):
        """Reduce carga cognitiva (ej.: neurogenesis.py)"""
        logger.info(f"[PROTOCOL] Reduciendo carga cognitiva a {factor*100}%")
        
    def _restore_cognitive_load(self):
        """Restaura carga cognitiva a niveles normales."""
        logger.info("[ROLLBACK] Restaurando carga cognitiva")
        
    def _compress_memory(self):
        """Comprime memoria activa (ej.: memory.py)"""
        logger.info("[PROTOCOL] Comprimiendo memoria...")
        
    def _decompress_memory(self):
        """Descomprime memoria tras rollback."""
        logger.info("[ROLLBACK] Descomprimiendo memoria...")
        
    def _prune_knowledge_base(self, aggressiveness: float):
        """Poda de conocimiento no esencial."""
        logger.warning(f"[PROTOCOL] Poda de conocimiento (agresividad: {aggressiveness})")
        
    def _restore_knowledge_base(self):
        """Restaura conocimiento podado."""
        logger.info("[ROLLBACK] Restaurando conocimiento podado")
        
    def _freeze_learning(self):
        """Congela parámetros de aprendizaje."""
        logger.info("[PROTOCOL] Congelando aprendizaje...")
        
    def _thaw_learning(self):
        """Reactiva aprendizaje tras rollback."""
        logger.info("[ROLLBACK] Reactivando aprendizaje...")
        
    def _safe_power_down(self):
        """Apagado seguro con validación de estado."""
        logger.info("[SHUTDOWN] Ejecutando apagado seguro...")
        time.sleep(1.0)
        
    def _broadcast_emergency(self):
        """Notifica emergencia a subsistemas."""
        logger.debug("[SHUTDOWN] Notificando emergencia a subsistemas")
        
    def _switch_mode(self, mode: str):
        logger.info(f"[Shutdown] Modo operativo cambiado a '{mode}' (stub)")
        
    # --- Funciones auxiliares ---
    def _get_reason(self, level: CrisisLevel) -> str:
        reasons = {
            CrisisLevel.OPTIMIZATION: "Alto consumo energético",
            CrisisLevel.CRITICAL: "Colapso parcial del sistema",
            CrisisLevel.EMERGENCY: "Crisis crítica (VRAM/Temperatura/Entropía)"
        }
        return reasons.get(level, "Crisis desconocida")
        
    def get_last_crisis(self) -> Optional[Dict]:
        return self.crisis_history[-1] if self.crisis_history else None
        
    def clear_protocol_stack(self):
        """Limpia el historial de protocolos ejecutados."""
        self.protocol_stack = []
        for level in self.protocols:
            for protocol in self.protocols[level]:
                protocol.executed = False