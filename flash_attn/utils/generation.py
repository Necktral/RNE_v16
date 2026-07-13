"""``flash_attn.utils.generation.GenerationMixin`` — **andamiaje, y lo decimos**.

Consumidor: ``engines/hnet/models/mixer_seq.py:7`` ⇒
``class HNetForCausalLM(nn.Module, GenerationMixin)``.

Qué necesita H-Net de esto: **nada, salvo existir**.  ``HNetForCausalLM`` define por su
cuenta ``allocate_inference_cache``, ``forward`` y ``step``; el mixin sólo aparece en las
bases de la clase.  Nadie en el repo llama a ``.generate()``
(``exocortex/tools/generate_hnet.py`` implementa su propio bucle de decodificación).

Por eso acá hay una **clase base mínima**.  Los métodos que FlashAttention implementa de
verdad (``generate``, y ``allocate_inference_cache`` como hook abstracto) **tiran
``NotImplementedError``** si alguien los llama: la generación de FlashAttention hace
sampling, CUDA graphs y gestión de caché que NO están replicados acá.  Un ``pass`` o un
tensor vacío serían exactamente la mentira que este paquete existe para no contar.

Y lo que RNFE necesita de H-Net **no pasa por acá**: N5 pide una probabilidad de frontera por
byte (``RoutingModule``), no texto generado.
"""

from __future__ import annotations

__all__ = ["GenerationMixin"]

_NOT_IMPLEMENTED = (
    "`GenerationMixin.{method}` NO está implementado en la capa de compatibilidad torch de "
    "RNFE (`flash_attn/` en la raíz del repo). Este mixin existe SÓLO para que "
    "`HNetForCausalLM(nn.Module, GenerationMixin)` sea importable sin FlashAttention. "
    "La generación real de FlashAttention (sampling, CUDA graphs, gestión de KV-cache) no "
    "está replicada. Si necesitás generar texto con H-Net, usá el bucle de decodificación de "
    "`exocortex/tools/generate_hnet.py`, o implementá esto con equivalencia real."
)


class GenerationMixin:
    """Base mínima. NO genera nada; falla ruidosamente si se la usa para generar."""

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        raise NotImplementedError(
            _NOT_IMPLEMENTED.format(method="allocate_inference_cache")
            + " (HNetForCausalLM lo sobreescribe: si ves este error, la que falla es otra "
            "clase que hereda el mixin sin implementarlo.)"
        )

    def generate(self, *args, **kwargs):
        raise NotImplementedError(_NOT_IMPLEMENTED.format(method="generate"))
