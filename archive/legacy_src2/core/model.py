# src/core/model.py

import torch.nn as nn

class CombinedModel(nn.Module):
    """
    Unifica el modelo generativo y el posterior en un único módulo de PyTorch.
    
    Esta es la representación principal del modelo del sistema, que se pasa al 
    entrenador y a los módulos de adaptación.
    """
    def __init__(self, generative_model: nn.Module, posterior_model: nn.Module):
        super().__init__()
        self.generative_model = generative_model
        self.posterior = posterior_model

# --- Mock avanzado para validación extrema (no afecta el modelo real) ---
try:
    BaseModel
except NameError:
    class BaseModel:
        def observe(self, x):
            pass
        def tick(self):
            pass
        def random_observation(self, scale=1.0):
            import random
            return random.uniform(-scale, scale)
        def allocate_dummy_tensor(self):
            return 0
        def stress_step(self):
            pass
        def structure_hash(self):
            import hashlib, time
            return hashlib.md5(str(time.time()).encode()).hexdigest()
