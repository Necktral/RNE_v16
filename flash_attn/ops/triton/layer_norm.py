"""``flash_attn.ops.triton.layer_norm`` en torch puro. NO hay Triton acá.

``RMSNorm`` es **el único símbolo de FlashAttention que el encoder de H-Net necesita de
verdad** (``engines/hnet/modules/isotropic.py:12`` y ``engines/hnet/modules/block.py:8``).

TRAMPA que este archivo tiene que sortear
=========================================
``engines/hnet/modules/block.py:106`` hace::

    assert isinstance(self.norm1, RMSNorm), "Only RMSNorm is supported"

donde ``RMSNorm`` es **el que se importó de acá**.  O sea: la clase que se instancia
(``norm_cls = partial(RMSNorm, eps=..., **factory_kwargs)``, ``block.py:74``) tiene que ser
literalmente esta clase.  Un alias, un wrapper o un ``functools.partial`` que devuelva otra
cosa rompe el assert.

Por qué NO reusamos el ``RMSNorm`` del vendor de Mamba
======================================================
``engines/mamba_vendor/mamba_ssm/ops/triton/layer_norm.py:953`` tiene un ``RMSNorm`` que es
copia literal del de FlashAttention y **funciona**… pero es un kernel **Triton**: exige GPU.
``RMSNorm`` y ``swiglu`` no tienen ninguna razón para exigir GPU, y los tests de este shim
corren en CPU.  Así que acá va la versión torch, y el test
``test_rmsnorm_coincide_con_el_kernel_triton_del_vendor`` la contrasta contra ese kernel
Triton en GPU: el vendor es el **oráculo**, no la referencia escrita por mí.

Semántica replicada AL PIE DE LA LETRA (incluido un comportamiento contraintuitivo)
===================================================================================
Del kernel upstream (``_layer_norm_fwd`` + ``_layer_norm_fwd_1pass_kernel``):

* Todo el cómputo se hace en **fp32**; ``y`` se devuelve en ``x.dtype``.
* ``rstd = 1 / sqrt(mean(x²) + eps)``;  ``y = (x · rstd) · weight``.
* Con ``residual``:  ``x ← x + residual`` (en fp32) **antes** de normalizar.
* ``residual_dtype = residual.dtype`` si hay residual; si no, ``fp32`` cuando
  ``residual_in_fp32``, si no ``None``.
* ⚠ **``residual_out`` sólo se materializa si**
  ``residual is not None`` **o** ``residual_dtype != x.dtype``.
  Consecuencia real: con ``x`` en **fp32**, ``residual=None`` y ``residual_in_fp32=True``
  (que es exactamente el primer bloque de H-Net), upstream devuelve ``(y, None)``.
  El stream residual queda en ``None`` y **no se acumula**.  Es un comportamiento del
  upstream, no un bug de este shim: H-Net está pensado para correr en bf16/fp16, donde
  ``residual_dtype (fp32) != x.dtype`` y el residual sí se materializa.
  Lo reproducimos tal cual — "arreglarlo" nos alejaría del modelo real y de sus pesos.
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = ["RMSNorm", "layer_norm_fn", "rms_norm_fn"]

_UNSUPPORTED = (
    "`layer_norm_fn(..., {feature})` no está implementado en la capa de compatibilidad "
    "torch de RNFE (`flash_attn/` en la raíz del repo). {why} No lo fabrico."
)


def layer_norm_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    residual: Optional[torch.Tensor] = None,
    x1: Optional[torch.Tensor] = None,
    weight1: Optional[torch.Tensor] = None,
    bias1: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    dropout_p: float = 0.0,
    rowscale: Optional[torch.Tensor] = None,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    is_rms_norm: bool = False,
    return_dropout_mask: bool = False,
):
    if x1 is not None or weight1 is not None or bias1 is not None:
        raise NotImplementedError(
            _UNSUPPORTED.format(
                feature="x1/weight1/bias1",
                why="El LayerNorm paralelo (dos ramas) no lo usa nadie en `engines/`.",
            )
        )
    if rowscale is not None:
        raise NotImplementedError(
            _UNSUPPORTED.format(
                feature="rowscale",
                why="No lo usa nadie en `engines/`.",
            )
        )
    if dropout_p:
        raise NotImplementedError(
            _UNSUPPORTED.format(
                feature=f"dropout_p={dropout_p}",
                why=(
                    "El dropout fusionado de FlashAttention usa su RNG por fila; la máscara "
                    "no es reproducible con el RNG de torch."
                ),
            )
        )
    if return_dropout_mask:
        raise NotImplementedError(
            _UNSUPPORTED.format(
                feature="return_dropout_mask=True",
                why="Sin dropout fusionado no hay máscara que devolver.",
            )
        )

    x_dtype = x.dtype
    residual_dtype = (
        residual.dtype
        if residual is not None
        else (torch.float32 if residual_in_fp32 else None)
    )

    acc = x.float()
    if residual is not None:
        acc = acc + residual.float()

    # Espejo EXACTO de la condición de `_layer_norm_fwd` que decide si `residual_out` existe.
    store_residual_out = residual is not None or (
        residual_dtype is not None and residual_dtype != x_dtype
    )
    residual_out = (
        acc.to(residual_dtype if residual_dtype is not None else x_dtype)
        if store_residual_out
        else None
    )

    if is_rms_norm:
        rstd = torch.rsqrt(acc.square().mean(dim=-1, keepdim=True) + eps)
        x_hat = acc * rstd
    else:
        mean = acc.mean(dim=-1, keepdim=True)
        centered = acc - mean
        rstd = torch.rsqrt(centered.square().mean(dim=-1, keepdim=True) + eps)
        x_hat = centered * rstd

    y = x_hat * weight.float()
    if bias is not None:
        y = y + bias.float()
    y = y.to(x_dtype)

    return y if not prenorm else (y, residual_out)


def rms_norm_fn(
    x,
    weight,
    bias,
    residual=None,
    x1=None,
    weight1=None,
    bias1=None,
    eps: float = 1e-6,
    dropout_p: float = 0.0,
    rowscale=None,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    return_dropout_mask: bool = False,
):
    return layer_norm_fn(
        x,
        weight,
        bias,
        residual=residual,
        x1=x1,
        weight1=weight1,
        bias1=bias1,
        eps=eps,
        dropout_p=dropout_p,
        rowscale=rowscale,
        prenorm=prenorm,
        residual_in_fp32=residual_in_fp32,
        is_rms_norm=True,
        return_dropout_mask=return_dropout_mask,
    )


class RMSNorm(torch.nn.Module):
    """Misma firma de constructor y de ``forward`` que el ``RMSNorm`` de FlashAttention.

    Constructor: ``RMSNorm(hidden_size, eps=1e-5, dropout_p=0.0, device=None, dtype=None)``
    — es la que usan ``isotropic.py:98`` (``RMSNorm(d_model, eps=1e-5, **factory_kwargs)``) y
    ``block.py:74`` (``partial(RMSNorm, eps=norm_epsilon, **factory_kwargs)``), donde
    ``factory_kwargs = {"device": ..., "dtype": ...}``.

    ``forward(x, residual=None, prenorm=False, residual_in_fp32=False)``:
      * ``prenorm=False`` → devuelve ``y``.
      * ``prenorm=True``  → devuelve ``(y, residual_out)``.  Leé la advertencia del docstring
        del módulo sobre cuándo ``residual_out`` es ``None``.
    """

    def __init__(self, hidden_size, eps=1e-5, dropout_p=0.0, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        if dropout_p > 0.0:
            raise NotImplementedError(
                _UNSUPPORTED.format(
                    feature=f"dropout_p={dropout_p}",
                    why="El dropout fusionado no es reproducible con el RNG de torch.",
                )
            )
        self.drop = None
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        return rms_norm_fn(
            x,
            self.weight,
            self.bias,
            residual=residual,
            eps=self.eps,
            dropout_p=0.0,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
        )
