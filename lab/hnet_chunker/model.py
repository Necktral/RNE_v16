"""El camino de fronteras de H-Net: bytes -> Embedding -> encoder -> RoutingModule.

Sólo se instancian los 28.76 M de parámetros que participan de la decisión de
frontera (4.2 % de los 679 M del modelo).  El `main_network` (T22, 1536-dim), el
`decoder`, el `dechunk` y el `lm_head` NO se cargan: no influyen en
`boundary_prob`.  Es exactamente el subconjunto que el paquete propone fine-tunear.

═══════════════════════════════════════════════════════════════════════════════
⚠⚠ DTYPE: **fp32 ESTÁ ROTO EN ESTE REPO.  HAY QUE EVALUAR EN fp16.** ⚠⚠
═══════════════════════════════════════════════════════════════════════════════

Esto CONTRADICE la premisa con la que llegó el paquete ("para evaluar usá fp32").
La premisa es falsa, y su causa está localizada en UNA línea.

  El kernel REAL (`engines/mamba_vendor/mamba_ssm/ops/triton/layer_norm.py:414`)
  devuelve, de `_layer_norm_fwd`:

      residual_out if residual_out is not None else x        # ← el fallback

  El shim (`flash_attn/ops/triton/layer_norm.py:119-142`) replicó la CONDICIÓN DE
  ASIGNACIÓN de `residual_out` (que en fp32 con `residual=None` y
  `residual_in_fp32=True` da None, porque `residual_dtype == x_dtype`) pero NO
  replicó ese fallback: devuelve `None`.

  `Block.forward` (`engines/hnet/modules/block.py:115-137`) toma ese `None` como
  el residual y se lo pasa al bloque siguiente.  Resultado: en fp32 **el stream
  residual del encoder nunca se acumula — las 4 capas Mamba pierden TODAS sus
  conexiones residuales.**  En fp16/bf16 el bug es invisible, porque ahí
  `residual_dtype (fp32) != x_dtype` y `residual_out` sí se materializa.

MEDIDO (RTX 2070, sm_75), mismo texto inglés, cortes con p>=0.5:

    fp32 (como está)     : 48 cortes, 2.27 B/chunk
        |The |q|ui|c|k |brown| fox| |j|umps |o|ve|r| |th|e| laz|y| dog|. |H|ie|ra...
    fp16                 : 24 cortes, 4.54 B/chunk
        |The| quick| brown| fo|x| ju|mps| over| the la|zy| dog|.| Hi|erarchical|...
    fp32 + residual restaurado (parche en memoria, NO en disco):
                           24 cortes, 4.54 B/chunk — IDÉNTICO a fp16, byte a byte.

Ese último renglón es la prueba: la matemática CORRECTA en precisión COMPLETA da
la respuesta de fp16.  fp16 no acierta de casualidad — computa la función buena en
menos bits.  fp32, tal como está el repo, computa OTRA función.

NO se arregla `flash_attn/` acá (está fuera del scope del paquete).  Lo que se
hace es DETECTARLO en runtime (`_residual_stream_is_alive`) y NEGARSE a correr en
fp32 mientras el bug exista.  Si alguien lo arregla, el guard deja pasar fp32 solo.
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


def residual_stream_is_alive(dtype: torch.dtype, device: str = "cpu") -> bool:
    """¿El stream residual del encoder sobrevive con este dtype?

    Comprueba el contrato que `Block.forward` necesita: `RMSNorm(..., prenorm=True)`
    tiene que devolver un residual USABLE aunque no se le haya pasado uno.  El
    kernel real lo garantiza (devuelve `x` como fallback).  El shim de este repo
    devuelve `None` en fp32 ⇒ el residual muere.

    Es un chequeo, no un parche: si alguien arregla el shim, esto empieza a dar
    True y fp32 pasa a estar permitido, sin tocar nada más.
    """
    from flash_attn.ops.triton.layer_norm import RMSNorm

    norm = RMSNorm(8, eps=1e-5, device=device, dtype=dtype)
    x = torch.randn(1, 2, 8, device=device, dtype=dtype)
    _, residual_out = norm(x, residual=None, prenorm=True, residual_in_fp32=True)
    return residual_out is not None


@dataclass
class ProbeInfo:
    n_params: int
    n_encoder_tensors: int
    device: str
    dtype: str
    window: int
    warmup: int
    residual_stream_alive: bool = True


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
        dtype: torch.dtype = torch.float16,
        window: int = 4096,
        warmup: int = 256,
    ) -> None:
        super().__init__()
        if not residual_stream_is_alive(dtype, device):
            raise ValueError(
                f"Con dtype={dtype} el stream residual del encoder MUERE en este repo: "
                "`flash_attn/ops/triton/layer_norm.py` devuelve residual_out=None donde el "
                "kernel real (engines/mamba_vendor/mamba_ssm/ops/triton/layer_norm.py:414) "
                "devuelve `x`. Las 4 capas Mamba del encoder pierden sus conexiones "
                "residuales y el chunker mide OTRA función. Usá fp16 (verificado idéntico a "
                "fp32-con-el-residual-restaurado). Si de verdad querés reproducir el bug, "
                "usá `allow_lossy_dtype`."
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
        """Escotilla explícita para MEDIR el bug del dtype, no para usarlo.

        Con esto se reproduce el fp32 roto (residual muerto) y se lo compara contra
        fp16.  Existe para que el bug sea FALSABLE, no para que sea usable.
        """
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
            residual_stream_alive=residual_stream_is_alive(self._dtype, self._device),
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
