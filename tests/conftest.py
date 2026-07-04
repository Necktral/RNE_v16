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


class _ProgressReporter:
    """Barra de progreso para las corridas largas (opt-in: ``RNFE_PROGRESS=1``).

    En terminal usa ``rich`` (barra viva con conteo, %, transcurrido y ETA);
    cuando la salida va a archivo (p. ej. corridas en background) emite líneas
    de progreso escalonadas legibles con ``tail -f``. Cero dependencias nuevas
    e inerte salvo que se active, de modo que las corridas normales y la
    reproducibilidad de baselines no cambian.
    """

    def __init__(self) -> None:
        import time

        self._time = time.time
        self.total = 0
        self.done = 0
        self.failed = 0
        self.start = self._time()
        self._tty = sys.stderr.isatty()
        self._rich = None
        self._task = None
        self._step = 1
        self._last_line = -1

    def pytest_collection_finish(self, session) -> None:
        self.total = len(session.items)
        self.start = self._time()
        self._step = max(1, self.total // 40)  # ~40 líneas en modo archivo
        if self._tty:
            try:
                from rich.console import Console
                from rich.progress import (
                    BarColumn,
                    MofNCompleteColumn,
                    Progress,
                    TextColumn,
                    TimeElapsedColumn,
                    TimeRemainingColumn,
                )

                self._rich = Progress(
                    TextColumn("[bold]tests"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TextColumn("{task.percentage:>3.0f}%"),
                    TextColumn("·"),
                    TimeElapsedColumn(),
                    TextColumn("ETA"),
                    TimeRemainingColumn(),
                    console=Console(file=sys.stderr),
                    transient=False,
                )
                self._rich.start()
                self._task = self._rich.add_task("run", total=self.total)
            except Exception:
                self._rich = None

    def pytest_runtest_logreport(self, report) -> None:
        if report.failed and report.when != "teardown":
            self.failed += 1
        if report.when != "teardown":
            return
        self.done += 1
        if self._rich is not None:
            self._rich.update(
                self._task,
                completed=self.done,
                description=f"run ({self.failed} fail)" if self.failed else "run",
            )
        elif self.done % self._step == 0 or self.done == self.total:
            if self.done == self._last_line:
                return
            self._last_line = self.done
            elapsed = self._time() - self.start
            rate = self.done / elapsed if elapsed > 0 else 0.0
            eta = (self.total - self.done) / rate if rate > 0 else 0.0
            pct = 100.0 * self.done / self.total if self.total else 0.0
            filled = int(30 * self.done / self.total) if self.total else 0
            bar = "#" * filled + "-" * (30 - filled)
            fail = f"  {self.failed} fail" if self.failed else ""
            print(
                f"[progreso] [{bar}] {self.done}/{self.total} {pct:3.0f}%  "
                f"transcurrido {elapsed:5.0f}s  ETA {eta:5.0f}s{fail}",
                file=sys.stderr,
                flush=True,
            )

    def pytest_sessionfinish(self, session, exitstatus) -> None:
        if self._rich is not None:
            self._rich.stop()


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
    if os.environ.get("RNFE_PROGRESS", "").strip().lower() in {"1", "true", "yes", "on"}:
        config.pluginmanager.register(_ProgressReporter(), "rnfe_progress")


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
