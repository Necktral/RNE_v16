# aeon/components/dynamic_chunking.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from aeon.utils.device import get_device, get_dtype

class DynamicChunkingLayer(nn.Module):
    """
    Implementa la capa de Chunking que aprende a segmentar una secuencia
    de forma dinámica para reducir su longitud. (Versión Actualizada)
    """
    def __init__(self, d_model):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)

    def forward(self, x, target_ratio=6.0):
        """
        Pase hacia adelante actualizado para la nueva interfaz de HNet.

        Returns:
            torch.Tensor: La secuencia de salida comprimida (chunked).
            torch.Tensor: Las probabilidades de límite (boundary_probs).
            torch.Tensor: Las decisiones de límite discretas (boundary_decisions).
        """
        batch_size, seq_len, d_model = x.shape

        # 1. Módulo de Enrutamiento (Calcula probabilidades de límite)
        q = self.q_proj(x)
        k = self.k_proj(x)
        similarity = F.cosine_similarity(k[:, :-1], q[:, 1:], dim=-1)
        boundary_probs = 0.5 * (1 - similarity)
        boundary_probs = torch.cat(
            [torch.ones(batch_size, 1, device=get_device(), dtype=get_dtype()), boundary_probs],
            dim=1
        )
        # --- CAMBIO: Decisiones como float para facilitar el cálculo de confianza ---
        boundary_decisions_float = (boundary_probs >= 0.5).float()
        # --- NUEVO: Cálculo del coeficiente de confianza para cada decisión ---
        # La confianza es alta si la decisión y la probabilidad coinciden, baja si no.
        confidence_scores = boundary_decisions_float * boundary_probs + (1 - boundary_decisions_float) * (1 - boundary_probs)
        # 2. Downsampler (Comprime la secuencia)
        mask = boundary_decisions_float.bool()
        chunks_per_item = mask.sum(dim=1)
        max_chunks = chunks_per_item.max()
        chunked_sequence = torch.zeros(
            batch_size, max_chunks, d_model,
            device=get_device(), dtype=get_dtype()
        )

        for i in range(batch_size):
            selected_vectors = x[i, mask[i]]
            chunked_sequence[i, :chunks_per_item[i]] = selected_vectors

        # --- CAMBIO: Ahora devolvemos la confianza junto con las otras salidas ---
        return chunked_sequence, boundary_probs, boundary_decisions_float, confidence_scores

# --- NUEVO COMPONENTE ---
class DechunkingLayer(nn.Module):
    """
    Implementa la capa de Dechunking que expande una secuencia comprimida
    a su resolución original y la combina con una conexión residual.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, latent_features, boundary_decisions, skip_connection_input, confidence_scores):
        """
        Pase hacia adelante de la capa de Dechunking.

        Args:
            latent_features (torch.Tensor): Salida de la red principal (B, L_chunked, D).
            boundary_decisions (torch.Tensor): Decisiones de límite (B, L_original).
            skip_connection_input (torch.Tensor): Salida del codificador (B, L_original, D).

        Returns:
            torch.Tensor: Tensor expandido combinado con la conexión residual.
        """
        # 1. Upsampling (Expansión de la secuencia)
        # Generamos un índice para cada chunk basado en las decisiones de límite.
        chunk_indices = torch.cumsum(boundary_decisions, dim=1).long() - 1
        
        # Aseguramos que los índices no se salgan de los límites de la secuencia comprimida.
        chunk_indices = torch.clamp(chunk_indices, max=latent_features.shape[1] - 1)
        
        # Expandimos los índices para que coincidan con la dimensión del modelo.
        expanded_indices = chunk_indices.unsqueeze(-1).expand(-1, -1, self.d_model)
        
        # Usamos `gather` para seleccionar el vector de chunk correspondiente para cada posición original.
        upsampled_output = torch.gather(latent_features, 1, expanded_indices)

        # --- NUEVO: Trust-Gated Upsampling usando el coeficiente de confianza ---
        # Ponderamos la salida re-expandida por la confianza de cada decisión de segmentación.
        # Usamos STE (Straight-Through Estimator) para el gradiente
        ste_confidence = confidence_scores + (1.0 - confidence_scores).detach()
        gated_upsampled_output = upsampled_output * ste_confidence.unsqueeze(-1)

        # 2. Combinación con Conexión Residual (Skip Connection)
        # Sumamos la salida expandida con la salida del codificador (ya proyectada).
        decoder_input = gated_upsampled_output + skip_connection_input
        
        return decoder_input