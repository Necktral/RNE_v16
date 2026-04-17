# main.py
# AEON FENIX-Δ - Punto de Entrada Unificado y Profesional
# Este script es el único responsable de configurar, ensamblar e iniciar el sistema AEON.

import argparse
import asyncio
import logging
import os
import torch
import json
from pathlib import Path
from datetime import datetime

# --- Importaciones del Ecosistema AEON ---

from orchestrator import Orchestrator
from state import AEONState
from aeon.core.infrastructure import EventBus

# Protocolos y Tipos
from protocols import PlannerProto, HomeoProto
from aeon.aeon_types import *
from typing import Dict

# Sistemas de Alto Nivel
from aeon.systems.homeostasis.controller import HomeoController
from aeon.systems.planning.planner import AEONPlanner
from aeon.systems.episteme.episteme_meter import EpistemeMeter
from aeon.systems.homeostasis.thermodynamic_governor import ThermodynamicGovernor
from aeon.systems.homeostasis.shutdown_logic import PhasedShutdown
from aeon.systems.persistence.persistence import StatePreserver
from aeon.systems.homeostasis.life_monitor import LifeMonitor

# Componentes de Bajo Nivel
from aeon.vitals.sensors import NVIDIASensor, MockSensor
from aeon.utils.influx_logger import InfluxLogger

# Configuración de Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("AEON.Main")


def parse_args():
    """Configura y parsea los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description="AEON FENIX-Δ: Plataforma de Lanzamiento")
    
    parser.add_argument('--config', type=str, default='configs/default_config.json', help='Ruta al archivo de configuración JSON.')
    parser.add_argument('--run-name', type=str, default=f'aeon_run_{datetime.now().strftime("%Y%m%d-%H%M%S")}', help='Nombre para esta ejecución.')
    parser.add_argument('--log-dir', type=str, default='runs', help='Directorio para logs y salidas.')
    parser.add_argument('--cycles', type=int, default=None, help='Sobrescribe el número de ciclos de entrenamiento del config.')
    parser.add_argument('--no-gpu', action='store_true', help='Forzar ejecución en CPU, usando sensores simulados.')
    parser.add_argument('--test-mode', action='store_true', help='Modo de prueba para CI/CD, ejecuta un número limitado de ciclos y sale.')

    return parser.parse_args()

def load_config(config_path: str) -> Dict:
    """Carga la configuración desde un archivo JSON."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"El archivo de configuración no se encontró en: {config_path}")
    with open(path, 'r') as f:
        return json.load(f)

async def assemble_aeon(cfg: Dict, args: argparse.Namespace):
    """
    Ensambla todas las partes de AEON usando inyección de dependencias.
    Esta función es el corazón de la inicialización.
    """
    log.info("Iniciando fase de ensamblaje de AEON...")

    # --- 1. Componentes Fundacionales ---
    state = AEONState()
    bus = EventBus()
    run_path = Path(args.log_dir) / args.run_name
    run_path.mkdir(parents=True, exist_ok=True)

    # --- 2. Sistemas de Bajo Nivel (Percepción y Persistencia) ---
    use_gpu = torch.cuda.is_available() and not args.no_gpu
    sensor = NVIDIASensor() if use_gpu else MockSensor()
    
    influx_logger = None
    if cfg.get('influxdb', {}).get('enabled', False):
        try:
            influx_logger = InfluxLogger(**cfg['influxdb'])
            log.info("Logger de InfluxDB conectado.")
        except Exception as e:
            log.warning(f"No se pudo conectar a InfluxDB: {e}")

    preserver = StatePreserver(cfg.get('persistence', {}))

    # --- 3. Ecosistema de Homeostasis ---
    governor = ThermodynamicGovernor(cfg.get('governor', {}))
    shutdown_system = PhasedShutdown(cfg.get('shutdown', {}))
    episteme_meter = EpistemeMeter(cfg.get('episteme', {}))
    
    # El LifeMonitor es el supervisor autónomo que integra todo lo anterior
    life_monitor = LifeMonitor(
        shutdown_system=shutdown_system,
        episteme_meter=episteme_meter,
        governor=governor,
        preserver=preserver,
        config=cfg.get('life_monitor', {})
    )
    
    # El HomeoController es la interfaz que el Orchestrator consulta
    homeo_controller = HomeoController(
        sensor=sensor,
        config=cfg.get('homeostasis', {})
    )

    # --- 4. Sistemas Cognitivos Superiores ---
    planner = AEONPlanner(cfg.get('planner', {}))

    # --- 5. El Orquestador Central ---
    # Sobrescribimos la configuración de ciclos si se pasa por argumento
    if args.cycles:
        cfg['orchestrator']['training_steps'] = args.cycles
    if args.test_mode:
        cfg['orchestrator']['training_steps'] = 10 # Ciclos cortos para prueba
        cfg['orchestrator']['log_interval'] = 2
        cfg['orchestrator']['val_interval'] = 5

    orchestrator = Orchestrator(
        cfg=cfg['orchestrator'],
        state=state,
        bus=bus,
        planner=planner,
        homeo=homeo_controller,
    )
    
    log.info("Ensamblaje de AEON completado. Todos los sistemas están interconectados.")
    return orchestrator, life_monitor, sensor

async def main():
    """Punto de entrada asíncrono principal."""
    args = parse_args()
    cfg = load_config(args.config)
    
    # --- Diagnóstico Inicial ---
    log.info(f"Iniciando ejecución: {args.run_name}")
    log.info(f"Configuración cargada desde: {args.config}")
    if torch.cuda.is_available() and not args.no_gpu:
        log.info(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    else:
        log.info("Operando en modo CPU.")

    orchestrator, life_monitor, sensor = await assemble_aeon(cfg, args)
    
    # --- Bucle de Vida Principal ---
    monitor_task = None
    try:
        # Iniciar el monitor de vida en segundo plano si es coroutine
        # monitor_task = asyncio.create_task(life_monitor.start())  # Descomentar si start() es async

        # Iniciar el orquestador principal
        # await orchestrator.run_life_cycle()  # Método no implementado

    except KeyboardInterrupt:
        log.warning("Interrupción por teclado detectada. Iniciando apagado ordenado...")
        if monitor_task and not monitor_task.done():
            monitor_task.cancel()
            try:
                await monitor_task
            except Exception:
                pass
    except asyncio.CancelledError:
        log.info("Tareas canceladas. Procediendo al apagado.")
    except Exception as e:
        log.critical(f"Error crítico no manejado en el bucle principal: {e}", exc_info=True)
    finally:
        log.info("Iniciando secuencia de apagado final...")
        if life_monitor:
            life_monitor.stop()
        if sensor:
            sensor.shutdown()
        log.info("AEON FENIX-Δ ha completado su ciclo de vida.")

if __name__ == "__main__":
    import sys
    import warnings
    if sys.version_info < (3, 11):
        warnings.warn("Se recomienda Python 3.11+ para AEON FENIX-Δ.")
    import asyncio
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Ejecución interrumpida por el usuario. Apagando AEON FENIX-Δ...")