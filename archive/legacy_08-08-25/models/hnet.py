# aeon/models/hnet.py

import torch
import torch.nn as nn
from typing import Dict, Tuple

from aeon.components.ssm_block import MambaBlock
from aeon.components.dynamic_chunking import DynamicChunkingLayer, DechunkingLayer

class HNet(nn.Module):
    """
    Implementa la arquitectura de Red Jerárquica (H-Net) con un núcleo Mamba.
    La arquitectura sigue un patrón de Encoder -> Chunking -> MainNetwork -> Dechunking -> Decoder.
    """
    def __init__(self, d_model: int, n_layers: int, vocab_size: int = 256):
        super().__init__()
        
        # H-Net tiene múltiples etapas. Para esta implementación, usaremos una sola etapa jerárquica.
        # Las capas se dividen para simular la estructura U-Net.
        encoder_layers = n_layers // 2
        main_network_layers = n_layers - encoder_layers

        # 1. Encoder: Procesa la secuencia de entrada en la resolución original.
        self.encoder = nn.Sequential(*[MambaBlock(d_model) for _ in range(encoder_layers)])
        
        # 2. Dynamic Chunking: Reduce la dimensionalidad de la secuencia.
        self.chunking_layer = DynamicChunkingLayer(d_model)

        # 3. Main Network: Procesa la secuencia comprimida (semánticamente más densa).
        self.main_network = nn.Sequential(*[MambaBlock(d_model) for _ in range(main_network_layers)])

        # 4. Dechunking & Decoder: Reconstruye la secuencia a su resolución original.
        self.dechunking_layer = DechunkingLayer(d_model)
        
        # Capa de proyección para la conexión residual (skip connection)
        self.skip_projection = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, target_ratio: float) -> Dict[str, torch.Tensor | Tuple]:
        """
        Realiza la pasada hacia adelante a través de la arquitectura jerárquica.

        Args:
            x (torch.Tensor): Tensor de entrada con embeddings (batch, seq_len, d_model).
            target_ratio (float): El ratio de compresión objetivo para el chunking.

        Returns:
            Dict[str, torch.Tensor | Tuple]: Un diccionario que contiene:
                - 'final_logits': Salida final del modelo.
                - 'ratio_loss_components': Tupla (boundary_probs, boundary_decisions).
                - 'latent_features': Características de la red principal para InfoNCE.
        """
        # 1. Encoder
        encoded_output = self.encoder(x)
        
        # 2. Dynamic Chunking
        # Esta capa devuelve la secuencia segmentada y los datos para la loss de regularización.
        chunked_sequence, boundary_probs, boundary_decisions = self.chunking_layer(
            encoded_output, target_ratio
        )
        
        # 3. Main Network (Bottleneck)
        # La salida de esta red son las características latentes de alto nivel.
        latent_features = self.main_network(chunked_sequence)
        
        # 4. Dechunking & Decoder
        # La capa de dechunking expande la secuencia y la combina con la skip connection.
        decoded_output = self.dechunking_layer(
            latent_features, 
            boundary_decisions,
            skip_connection_input=self.skip_projection(encoded_output)
        )
        
        # Devolvemos un diccionario estructurado para la UnifiedSeedLoss
        model_outputs = {
            "final_logits": decoded_output,
            "ratio_loss_components": (boundary_probs, boundary_decisions),
            "latent_features": latent_features
        }
        
        return model_outputs