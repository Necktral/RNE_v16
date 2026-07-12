"""``flash_attn.ops.triton`` — capa de compatibilidad torch (NO es FlashAttention).

⚠ El nombre dice "triton" porque **así se llama el módulo upstream** y los imports de
``engines/hnet`` lo piden así (``from flash_attn.ops.triton.layer_norm import RMSNorm``).
**Acá adentro NO hay Triton**: es torch puro.  El nombre es una obligación de compatibilidad,
no una descripción.
"""
