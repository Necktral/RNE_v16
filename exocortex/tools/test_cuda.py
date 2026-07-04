"""Smoke check opcional para CUDA.

Uso CLI:
    python exocortex/tools/test_cuda.py

Uso pytest (opt-in):
    pytest -q exocortex/tools/test_cuda.py -m "requires_torch and requires_cuda"
"""

from __future__ import annotations

import pytest


@pytest.mark.requires_torch
@pytest.mark.requires_cuda
def test_cuda_runtime_available() -> None:
    import torch

    assert torch.cuda.is_available(), "CUDA no disponible"
    _ = torch.cuda.get_device_name(0)


def main() -> None:
    import torch

    available = torch.cuda.is_available()
    print("GPU Disponible:", available)
    if available:
        print("Nombre:", torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()
