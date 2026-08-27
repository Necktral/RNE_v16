"""Corpus y evaluador para fine-tunear el chunker de H-Net sobre la experiencia de RNFE.

Este paquete NO entrena. Provee las dos piezas que hacen *falsable* cualquier
afirmación sobre el chunker:

- `ground_truth`: derivación de fronteras VERDADERAS en offsets de BYTES.
- `metrics`: precisión/recall/F1 con tolerancia, curva sobre el umbral, y los
  controles negativos obligatorios (corte de tamaño fijo, corte aleatorio).
- `model`: el camino de fronteras de H-Net (embeddings -> encoder -> RoutingModule).
- `corpus`: extractor deduplicado, con split por `run_id` y procedencia por shard.

Todo lo que no se puede derivar de forma defendible se marca como NO EVALUABLE.
No se fabrica verdad de campo en ningún punto.
"""

from lab.hnet_chunker.ground_truth import (
    GroundTruth,
    code_python_ground_truth,
    json_ground_truth,
    prose_ground_truth,
)
from lab.hnet_chunker.metrics import (
    SegmentationScore,
    fixed_size_cuts,
    random_cuts,
    score_boundaries,
    threshold_curve,
)

__all__ = [
    "GroundTruth",
    "SegmentationScore",
    "code_python_ground_truth",
    "fixed_size_cuts",
    "json_ground_truth",
    "prose_ground_truth",
    "random_cuts",
    "score_boundaries",
    "threshold_curve",
]
