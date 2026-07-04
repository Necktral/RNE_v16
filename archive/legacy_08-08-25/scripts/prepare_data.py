# scripts/prepare_data.py
import os
import requests
from configs import hnet_760m_config as aeon_config

def prepare_dataset():
    """Descarga y guarda el dataset si no existe."""
    os.makedirs(os.path.dirname(aeon_config.DATA_CACHE_PATH), exist_ok=True)
    
    if not os.path.exists(aeon_config.DATA_CACHE_PATH):
        print(f"Dataset no encontrado. Descargando desde {aeon_config.DATASET_URL}...")
        try:
            response = requests.get(aeon_config.DATASET_URL)
            response.raise_for_status()
            with open(aeon_config.DATA_CACHE_PATH, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"Dataset guardado en {aeon_config.DATA_CACHE_PATH}")
        except requests.exceptions.RequestException as e:
            print(f"Error al descargar el dataset: {e}")
            return False
    else:
        print("Dataset ya existe localmente.")
    
    return True

if __name__ == "__main__":
    prepare_dataset()