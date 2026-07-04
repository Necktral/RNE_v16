# configs/aeon_config.py

# -- Parámetros del Modelo --
D_MODEL = 256       # Dimensión del modelo
N_LAYERS = 4        # Número de capas Mamba en la red principal
D_STATE = 16        # Dimensión del estado SSM
D_CONV = 4          # Dimensión de la convolución
EXPAND_FACTOR = 2   # Factor de expansión en el bloque Mamba

# -- Parámetros de Datos y Entrenamiento --
DATASET_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_CACHE_PATH = "data/tinyshakespeare.txt"
SEQ_LEN = 256       # Longitud de la secuencia para el entrenamiento
BATCH_SIZE = 8      # Tamaño del lote (batch size)
LEARNING_RATE = 1e-3
TRAINING_STEPS = 200 # Número de pasos de entrenamiento para la prueba
LOG_INTERVAL = 20    # Cada cuántos pasos mostrar información
SAVE_PATH = "checkpoints/aeon_model.pth" # Dónde guardar el modelo entrenado
CHUNK_RATIO = 4.0    # Ratio de compresión objetivo para Dynamic Chunking