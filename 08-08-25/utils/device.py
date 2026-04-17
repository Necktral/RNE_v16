# aeon/utils/device.py

import torch

# --- Configuración Global de Precisión y Dispositivo ---
# Nuestra estrategia es forzar float16 para ser compatibles con la arquitectura Turing.

# Selecciona el tipo de dato para el cómputo.
# Opciones: torch.float16, torch.bfloat16 (no soportado), torch.float32
DTYPE = torch.float16

# Selecciona el dispositivo de cómputo.
# Prioriza la GPU si está disponible.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_device():
    """
    Retorna el dispositivo de cómputo global.

    Returns:
        str: "cuda" o "cpu".
    """
    return DEVICE


def get_dtype():
    """
    Retorna el tipo de dato (dtype) global para los tensores.

    Returns:
        torch.dtype: El tipo de dato para el cómputo (e.g., torch.float16).
    """
    return DTYPE


def print_device_info():
    """
    Imprime información relevante sobre el dispositivo de cómputo y la memoria.
    """
    print("=" * 40)
    print(f"AEON Engine: Initialized")
    print(f"Computation Device: {get_device().upper()}")
    print(f"Computation DType: {str(get_dtype())}")

    if torch.cuda.is_available():
        gpu_index = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_index)
        total_mem = torch.cuda.get_device_properties(gpu_index).total_memory / (1024**3)
        allocated_mem = torch.cuda.memory_allocated(gpu_index) / (1024**3)
        cached_mem = torch.cuda.memory_reserved(gpu_index) / (1024**3)

        print(f"GPU Model: {gpu_name}")
        print(f"Total VRAM: {total_mem:.2f} GB")
        print(f"Allocated VRAM: {allocated_mem:.2f} GB")
        print(f"Cached VRAM: {cached_mem:.2f} GB")

    print("=" * 40)


# Ejecutar una vez al importar para mostrar la configuración
print_device_info()