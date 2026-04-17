# aeon/orchestrator/run.py
# ----------------------------------------------------------------------
import asyncio, logging, argparse, importlib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s · %(name)s · %(levelname)s · %(message)s",
)

parser = argparse.ArgumentParser(description="Arranca AEON Orchestrator.")
parser.add_argument(
    "--config",
    default="configs.hnet_760m_config",
    help="Ruta de módulo de configuración (módulo import-style).",
)
args = parser.parse_args()

cfg = importlib.import_module(args.config)
from . import Orchestrator  # import tardío para que logging esté listo

asyncio.run(Orchestrator(cfg).__init__(cfg=cfg).run())
