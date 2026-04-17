# utils.py

import os
import yaml
from omegaconf import OmegaConf

def load_config(path: str = "config/config.yaml"):
    """
    Carga el archivo de configuración YAML del sistema usando OmegaConf.

    Args:
        path (str): Ruta al archivo YAML.

    Returns:
        DictConfig: Configuración cargada como objeto OmegaConf.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"El archivo de configuración no existe: {path}")

    try:
        config = OmegaConf.load(path)
        return config
    except Exception as e:
        raise RuntimeError(f"Error al cargar la configuración: {e}")
