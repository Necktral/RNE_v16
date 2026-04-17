# scripts/train.py

import torch
import torch.nn as nn
import os
import numpy as np

from configs import hnet_760m_config as aeon_config
from aeon.models.hnet import HNet
from aeon.utils.device import get_device
from aeon.data.data_normalizer import DataNormalizer
from aeon.data.val_loader_config import create_train_val_loaders
from aeon.core.trainer import HNetTrainer # Asumiendo que HNetTrainer está en aeon/core/

def main():
    # Cargar datos de bytes preprocesados
    data_raw = np.load(aeon_config.DATA_CACHE_PATH)
    data_tensor_float = torch.from_numpy(data_raw).float().unsqueeze(1)
    data_tensor_long = torch.from_numpy(data_raw).long()

    # 1. Cargar estadísticas de normalización
    normalizer = DataNormalizer()
    normalizer.load_stats(aeon_config.DATA_STATS_PATH)
    
    # 2. Crear DataLoaders
    # Nota: Pasamos el tensor de floats para la normalización (será la entrada X)
    # y el tensor de longs para los objetivos (Y)
    train_loader, val_loader = create_train_val_loaders(
        data_tensor_float,
        data_tensor_long, # Pasamos Y por separado
        batch_size=aeon_config.BATCH_SIZE,
        val_split=0.1,
        normalizer=normalizer
    )

    device = get_device()
    
    # Inicializar modelo y componentes
    embedding = nn.Embedding(aeon_config.VOCAB_SIZE, aeon_config.D_MODEL).to(device)
    model = HNet(d_model=aeon_config.D_MODEL, n_layers=aeon_config.N_LAYERS).to(device)
    output_head = nn.Linear(aeon_config.D_MODEL, aeon_config.VOCAB_SIZE).to(device)
    
    all_params = list(embedding.parameters()) + list(model.parameters()) + list(output_head.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=aeon_config.LEARNING_RATE)
    
    # Configuración para el Trainer
    trainer_config = {
        'device': device,
        'dtype': torch.float16 if device.type == 'cuda' else torch.float32,
        'vocab_size': aeon_config.VOCAB_SIZE,
        'chunk_ratio': aeon_config.CHUNK_RATIO,
        'ratio_loss_weight': 0.03
    }

    # 3. Instanciar el HNetTrainer
    trainer = HNetTrainer(model, embedding, output_head, optimizer, trainer_config)

    print(f"Iniciando entrenamiento en {device.type.upper()}...")
    for step, (x_norm, y) in enumerate(train_loader):
        if step > aeon_config.TRAINING_STEPS:
            break
        
        x_norm = x_norm.to(device)
        y = y.to(device)
        
        # 4. Ejecutar paso de entrenamiento usando el Trainer
        total_loss, main_loss, ratio_loss = trainer.train_step(x_norm, y)
        
        if step % aeon_config.LOG_INTERVAL == 0:
            print(f"Paso {step}/{aeon_config.TRAINING_STEPS}: Pérdida Total={total_loss:.4f}, Principal={main_loss:.4f}, Ratio={ratio_loss:.4f}")

    print("Entrenamiento finalizado. Guardando modelo...")
    # Lógica de guardado...
    
if __name__ == "__main__":
    # Asegurarse de que los datos y las estadísticas estén preparados
    if not os.path.exists(aeon_config.DATA_CACHE_PATH) or not os.path.exists(aeon_config.DATA_STATS_PATH):
        from scripts.prepare_data import prepare_dataset_and_stats
        print("Preparando datos y estadísticas por primera vez...")
        prepare_dataset_and_stats()
    main()