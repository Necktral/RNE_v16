# aeon/components/ssm_block.py (Versión Correcta)

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from aeon.utils.device import get_device, get_dtype

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        d_inner = int(self.expand * self.d_model)

        self.in_proj = nn.Linear(self.d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(d_inner, self.d_conv + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_conv, d_inner, bias=True)

        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(d_inner, 1)))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, self.d_model, bias=False)

    def forward(self, x):
        (b, l, d) = x.shape
        x_z = self.in_proj(x)
        x, z = x_z.chunk(2, dim=-1)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :l]
        x = x.transpose(1, 2)
        x = F.silu(x)
        y = self.ssm(x)
        z = F.silu(z)
        output = y * z
        output = self.out_proj(output)
        return output

    def ssm(self, x):
        input_dtype = x.dtype
        x_dbl = self.x_proj(x)
        delta, B, C = torch.split(x_dbl, [self.d_conv, self.d_state, self.d_state], dim=-1)
        delta = self.dt_proj(delta)

        x = x.to(torch.float32)
        delta = F.softplus(delta.to(torch.float32))
        B = B.to(torch.float32)
        C = C.to(torch.float32)
        
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        # --- CORRECCIÓN DEFINITIVA: Permutar tensores para el formato del kernel ---
        x_transposed = rearrange(x, 'b l d -> b d l')
        delta_transposed = rearrange(delta, 'b l d -> b d l')
        B_transposed = rearrange(B, 'b l n -> b n l')
        C_transposed = rearrange(C, 'b l n -> b n l')
        # --- FIN DE LA CORRECCIÓN ---

        y = selective_scan_fn(
            x_transposed,
            delta_transposed,
            A.contiguous(),
            B_transposed.contiguous(),
            C_transposed.contiguous(),
            D.contiguous(),
            z=None,
            delta_bias=None,
            delta_softplus=True
        )

        # Revertir la permutación para la salida
        y = rearrange(y, 'b d l -> b l d')

        return y.to(dtype=input_dtype)