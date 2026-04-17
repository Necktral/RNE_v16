# run_aeon.py

import asyncio
import logging
from aeon.orchestrator import Orchestrator

def main():
    """
    Punto de entrada principal para iniciar el sistema AEON.
    """
    logging.info("Iniciando el sistema AEON...")
    
    # Instanciar el cerebro de AEON
    aeon_orchestrator = Orchestrator()
    
    try:
        # Iniciar el ciclo de vida asíncrono
        asyncio.run(aeon_orchestrator.run_forever())
    except KeyboardInterrupt:
        logging.info("Apagado iniciado por el usuario.")
    finally:
        logging.info("Cerrando sistema AEON.")

if __name__ == "__main__":
    main()