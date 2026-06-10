import sys
import os
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Aislamiento de storage en tests: por defecto SQLite (hermético), salvo que se
# soliciten explícitamente las pruebas PostgreSQL (RNFE_RUN_PG_TESTS=1). Evita que un
# `.env` con RNFE_STORAGE_MODE=postgres haga que la suite escriba en la base PG real.
if os.environ.get("RNFE_RUN_PG_TESTS") != "1":
    os.environ["RNFE_STORAGE_MODE"] = "sqlite"
    os.environ.pop("RNFE_POSTGRES_DSN", None)


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "requires_torch: test requiere torch")
    config.addinivalue_line(
        "markers",
        "requires_postgres: test requiere PostgreSQL y RNFE_POSTGRES_DSN",
    )
    config.addinivalue_line("markers", "requires_cuda: test requiere CUDA")
    config.addinivalue_line(
        "markers",
        "requires_extended_bench: test requiere RNFE_RUN_EXTENDED_BENCH=1",
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("requires_torch") and not _module_available("torch"):
        pytest.skip("Saltado: torch no disponible en entorno actual.")

    if item.get_closest_marker("requires_postgres"):
        if os.environ.get("RNFE_RUN_PG_TESTS") != "1":
            pytest.skip("Saltado: RNFE_RUN_PG_TESTS=1 no está activo.")
        if not os.environ.get("RNFE_POSTGRES_DSN"):
            pytest.skip("Saltado: RNFE_POSTGRES_DSN no está definido.")

    if item.get_closest_marker("requires_cuda"):
        if not _module_available("torch"):
            pytest.skip("Saltado: torch no disponible para validar CUDA.")
        import torch

        if not torch.cuda.is_available():
            pytest.skip("Saltado: CUDA no está disponible.")

    if item.get_closest_marker("requires_extended_bench"):
        if os.environ.get("RNFE_RUN_EXTENDED_BENCH") != "1":
            pytest.skip("Saltado: RNFE_RUN_EXTENDED_BENCH=1 no está activo.")
