# aeon_main_loop.py

import asyncio
import logging
import signal
import torch
import psutil
import platform
from runtime.legacy.module_orchestrator import Orchestrator
from torch.utils.tensorboard import SummaryWriter  # <-- Añadido para TensorBoard

# Configurar logging básico
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """
    Punto de entrada principal para iniciar el sistema AEON FENIX-Δ.
    Crea e inicia el Orquestador, que gestiona todo el ciclo de vida de la IA.
    """
    # Diagnóstico de GPU
    cuda_disp = torch.cuda.is_available()
    print("CUDA disponible:", cuda_disp)
    logger.info(f"CUDA disponible: {cuda_disp}")
    if cuda_disp:
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        vram_used = round(torch.cuda.memory_allocated(0) / 1e9, 3)
        vram_reserved = round(torch.cuda.memory_reserved(0)  / 1e9, 3)
        print(f"Dispositivo   : {gpu_name}")
        print(f"VRAM total GB : {vram_total}")
        print(f"VRAM usada GB : {vram_used}")
        print(f"VRAM reserv GB: {vram_reserved}")
        logger.info(f"Dispositivo   : {gpu_name}")
        logger.info(f"VRAM total GB : {vram_total}")
        logger.info(f"VRAM usada GB : {vram_used}")
        logger.info(f"VRAM reserv GB: {vram_reserved}")
    else:
        print("No se detectó GPU CUDA. Usando CPU.")
        logger.info("No se detectó GPU CUDA. Usando CPU.")

    logger.info("🚀 Iniciando AEON FENIX-Δ...")
    
    # En el futuro, aquí se podría cargar una configuración externa.
    # Por ahora, usamos la configuración por defecto del Orquestador.
    latent_dim = 32 
    writer = SummaryWriter("runs/aeon_run0")
    orchestrator = Orchestrator(latent_dim=latent_dim, tensorboard_writer=writer)

    # Log de estado de salud inicial
    health = orchestrator.health_status()
    logger.info(f"[HOMEOSTASIS] Estado inicial: temp={health.temp}, vram={health.vram}, entropy={health.entropy}")

    # Manejar el apagado correctamente
    # loop = asyncio.get_running_loop()
    # def request_shutdown():
    #     logger.info("🔌 Solicitud de apagado recibida. Iniciando cierre ordenado...")
    #     orchestrator._shutdown.set()

    # NOTA: add_signal_handler no está implementado en Windows para asyncio.
    # El apagado se maneja solo con KeyboardInterrupt (Ctrl+C).

    try:
        logger.info("[AEON] Iniciando ciclo principal del orquestador...")
        try:
            await orchestrator.run_forever()
        except Exception as e:
            logger.error(f"Error en el ciclo principal: {e}", exc_info=True)
        finally:
            # Log de estado de salud final
            health = orchestrator.health_status()
            logger.info(f"[HOMEOSTASIS] Estado final: temp={health.temp}, vram={health.vram}, entropy={health.entropy}")
            logger.info("🧩 Sistema AEON FENIX-Δ finalizado.")
    except asyncio.CancelledError:
        logger.info("Tarea principal cancelada.")
    finally:
        logger.info("🧩 Sistema AEON FENIX-Δ finalizado.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Cierre forzado por el usuario.")
