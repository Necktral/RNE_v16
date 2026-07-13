from dataclasses import dataclass, field
from typing import List, Union


@dataclass
class AttnConfig:

    num_heads: List = field(default_factory=list)
    rotary_emb_dim: List = field(default_factory=list)
    window_size: List = field(default_factory=list)


@dataclass
class SSMConfig:

    d_conv: int = 4
    expand: int = 2
    d_state: int = 128
    chunk_size: int = 256
    # `Mamba2(use_mem_eff_path=True)` (el default de upstream, que se preserva acá) toma el
    # camino fusionado, que llama a `causal_conv1d_fn` — una extensión CUDA que exige nvcc.
    # En una máquina sin nvcc ese camino revienta con `TypeError: 'NoneType' object is not
    # callable`.  El camino lento (`False`) cae a `F.conv1d` de torch puro y da lo mismo.
    # `get_stage_cfg` hace `asdict(cfg)`, así que sin este campo la opción era INEXPRESABLE
    # desde un `HNetConfig`/JSON.  El default NO cambia el comportamiento de nadie.
    use_mem_eff_path: bool = True


@dataclass
class HNetConfig:
    arch_layout: List[Union[str, List]] = field(default_factory=list)
    d_model: List[int] = field(default_factory=list)
    # intermediate dimension for the FFNs (0 indicates no FFN)
    d_intermediate: List[int] = field(default_factory=list)
    vocab_size: int = 256
    ssm_cfg: SSMConfig = field(default_factory=SSMConfig)
    attn_cfg: AttnConfig = field(default_factory=AttnConfig)
    tie_embeddings: bool = False
