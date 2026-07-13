"""El camino de fronteras de H-Net: bytes -> Embedding -> encoder -> RoutingModule.

Sólo se instancian los 28.76 M de parámetros que participan de la decisión de
frontera (4.2 % de los 679 M del modelo).  El `main_network` (T22, 1536-dim), el
`decoder`, el `dechunk` y el `lm_head` NO se cargan: no influyen en
`boundary_prob` y meterlos costaría 2.5 GB de VRAM en fp32 sobre una GPU que
tiene 7.6 GB.  Es exactamente el subconjunto que el paquete propone fine-tunear.

⚠ DTYPE: fp32 obligatorio.  MEDIDO en esta máquina (RTX 2070, sm_75):

    "a"*24, cortes con p>=0.5, excluyendo el índice 0 (que se fuerza a 1.0):
        fp32  ->  5 / 23
        bf16  -> 23 / 23   ← el chunker se DEGRADA a "cortar en todos lados"

`boundary_prob = (1 - cos_sim)/2` con cos_sim ~ 0.99 en texto homogéneo: los 8
bits de mantisa de bf16 no resuelven `1 - 0.99`.  El error relativo de la resta
explota.  En bf16 el chunker no está peor calibrado: está ROTO.  Cualquier
medición del chunker en bf16 es basura, y cualquier fine-tuning en bf16 estaría
optimizando contra una señal destruida.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

__all__ = ["BoundaryProbe", "DEFAULT_CONFIG", "DEFAULT_WEIGHTS", "build_config"]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "engines" / "hnet" / "configs" / "hnet_1stage_L.json"
DEFAULT_WEIGHTS = Path("/home/wis/rnfe_models/hnet/hnet_1stage_L.pt")

# sha256 publicado por upstream para hnet_1stage_L.pt (commit cdfb6a08).
EXPECTED_SHA256 = "b92e051cc1dee9bfcfe5a0910661307f5dde670fc938bed191331e71138e4614"


def build_config(path: Path | str = DEFAULT_CONFIG):
    """Construye el `HNetConfig` a mano.

    `HNetConfig(**raw)` NO convierte `ssm_cfg`/`attn_cfg`: los deja como `dict`
    crudos (dataclass no valida tipos).  Verificado: sin esto, `get_stage_cfg`
    recibe un dict y `asdict()` revienta.

    `use_mem_eff_path=False` es OBLIGATORIO acá: el default (True) toma el camino
    fusionado de Mamba-2, que llama a `causal_conv1d_fn`, una extensión CUDA que
    exige nvcc — y nvcc no está en esta máquina.  El campo NO está en el JSON de
    upstream; hay que inyectarlo.
    """
    from hnet.models.config_hnet import AttnConfig, HNetConfig, SSMConfig

    raw = json.loads(Path(path).read_text())
    ssm = dict(raw.pop("ssm_cfg"))
    ssm.setdefault("use_mem_eff_path", False)
    attn = dict(raw.pop("attn_cfg"))
    return HNetConfig(**raw, ssm_cfg=SSMConfig(**ssm), attn_cfg=AttnConfig(**attn))


@dataclass
class ProbeInfo:
    n_params: int
    n_encoder_tensors: int
    device: str
    dtype: str
    window: int
    warmup: int


class BoundaryProbe(nn.Module):
    """Devuelve P(frontera) por byte. Una probabilidad por byte, nada más.

    Ventaneo: los documentos de RNFE llegan a decenas de KB.  Se procesan en
    ventanas de `window` bytes con `warmup` bytes de contexto que NO se puntúan.
    Por qué: el `RoutingModule` fuerza `boundary_prob[0] = 1.0` en cada ventana
    (`PAD_PROB`), y las primeras posiciones de una ventana tienen el estado de
    Mamba recién reseteado.  Puntuar esas posiciones sería medir el artefacto del
    ventaneo, no el modelo.  Con `warmup` bytes de arranque descartados, cada
    posición puntuada tiene al menos `warmup` bytes de contexto real.

    La posición 0 del DOCUMENTO conserva su 1.0 forzado; el scorer la excluye.
    """

    def __init__(
        self,
        config,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        window: int = 4096,
        warmup: int = 256,
    ) -> None:
        super().__init__()
        from hnet.modules.dc import RoutingModule
        from hnet.modules.isotropic import Isotropic

        if dtype is not torch.float32:
            raise ValueError(
                "BoundaryProbe exige fp32. En bf16 el chunker se degrada a cortar en "
                "todas las posiciones (medido: 23/23 en 'a'*24 vs 5/23 en fp32). "
                "Si querés medir esa degradación a propósito, usá `allow_lossy_dtype`."
            )
        self._build(config, device, dtype, window, warmup)

    def _build(self, config, device, dtype, window, warmup) -> None:
        from hnet.modules.dc import RoutingModule
        from hnet.modules.isotropic import Isotropic

        d_model = config.d_model[0]
        self.embeddings = nn.Embedding(config.vocab_size, d_model, device=device, dtype=dtype)
        self.encoder = Isotropic(config, pos_idx=0, stage_idx=0, device=device, dtype=dtype)
        self.routing_module = RoutingModule(d_model, device=device, dtype=dtype)
        self.window = window
        self.warmup = warmup
        self._device = device
        self._dtype = dtype
        self.eval()

    @classmethod
    def allow_lossy_dtype(
        cls, config, *, device: str = "cuda", dtype: torch.dtype, window: int = 4096, warmup: int = 256
    ) -> "BoundaryProbe":
        """Escotilla explícita para MEDIR la degradación por dtype, no para usarla."""
        probe = cls.__new__(cls)
        nn.Module.__init__(probe)
        probe._build(config, device, dtype, window, warmup)
        return probe

    def load_pretrained(self, weights: Path | str = DEFAULT_WEIGHTS) -> ProbeInfo:
        """Carga los pesos del camino de fronteras. `strict=True`: si falta uno, revienta."""
        sd = torch.load(str(weights), map_location="cpu", mmap=True, weights_only=True)

        enc_sd = {k[len("backbone.encoder.") :]: v for k, v in sd.items() if k.startswith("backbone.encoder.")}
        rm_sd = {
            k[len("backbone.routing_module.") :]: v
            for k, v in sd.items()
            if k.startswith("backbone.routing_module.")
        }
        if "embeddings.weight" not in sd:
            raise KeyError("el checkpoint no tiene `embeddings.weight`")
        if not enc_sd or not rm_sd:
            raise KeyError("el checkpoint no tiene el camino de fronteras (encoder/routing_module)")

        with torch.no_grad():
            self.embeddings.weight.copy_(sd["embeddings.weight"])
        self.encoder.load_state_dict(enc_sd, strict=True)
        self.routing_module.load_state_dict(rm_sd, strict=True)

        n_params = sum(p.numel() for p in self.parameters())
        del sd
        return ProbeInfo(
            n_params=n_params,
            n_encoder_tensors=len(enc_sd),
            device=str(self._device),
            dtype=str(self._dtype),
            window=self.window,
            warmup=self.warmup,
        )

    @torch.no_grad()
    def _window_prob(self, ids: torch.Tensor) -> torch.Tensor:
        # `mask` es obligatorio: sin él, `Isotropic.forward` asume modo empacado y
        # rompe con un assert. Es `torch.ones(1, L, bool)` porque no hay padding.
        mask = torch.ones(1, ids.shape[1], dtype=torch.bool, device=ids.device)
        h = self.embeddings(ids)
        h = self.encoder(h, mask=mask)
        out = self.routing_module(h, mask=mask)
        return out.boundary_prob[..., 1][0].float()

    @torch.no_grad()
    def boundary_prob(self, data: bytes) -> np.ndarray:
        """P(frontera) por byte, shape (len(data),). `p[0]` siempre es 1.0 (PAD_PROB)."""
        n = len(data)
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        arr = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
        out = np.zeros(n, dtype=np.float32)

        stride = self.window - self.warmup
        if stride <= 0:
            raise ValueError("window debe ser > warmup")

        start = 0
        while start < n:
            end = min(start + self.window, n)
            ids = torch.from_numpy(arr[start:end]).unsqueeze(0).to(self._device)
            p = self._window_prob(ids).cpu().numpy()
            keep_from = 0 if start == 0 else self.warmup
            out[start + keep_from : end] = p[keep_from:]
            if end == n:
                break
            start += stride
        out[0] = 1.0
        return out
