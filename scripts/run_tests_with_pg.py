#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path


CANDIDATE_PORTS = [55432, 5433, 15432]
HEALTH_TIMEOUT_SECONDS = 120
DSN_TIMEOUT_SECONDS = 30
POSTGRES_USER = "rnfe"
POSTGRES_PASSWORD = "rnfe_local_dev_only"
POSTGRES_DB = "rnfe"
POSTGRES_CONTAINER = "rnfe-postgres"


def log(message: str) -> None:
    print(f"[runner] {message}", flush=True)


def run_cmd(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[bytes]:
    rendered = " ".join(cmd)
    log(f"$ {rendered}")
    return subprocess.run(cmd, env=env, cwd=cwd, check=False)


def check_prerequisites() -> None:
    missing: list[str] = []
    for required_bin in ("docker", "pytest"):
        if shutil.which(required_bin) is None:
            missing.append(required_bin)

    if not Path(sys.executable).exists():
        missing.append("python")

    try:
        import psycopg  # noqa: F401
    except ImportError:
        missing.append("psycopg (modulo de Python)")

    if missing:
        raise RuntimeError(
            "Faltan prerequisitos: "
            + ", ".join(missing)
            + ". Instala dependencias y vuelve a ejecutar."
        )


def ensure_local_env(env_path: Path, env_example_path: Path) -> bool:
    if env_path.exists():
        return False
    if not env_example_path.exists():
        raise RuntimeError(
            f"No existe plantilla de entorno: {env_example_path}"
        )
    shutil.copy2(env_example_path, env_path)
    log(f"Creado archivo local de entorno: {env_path}")
    return True


def port_is_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def choose_port(forced_port: int | None) -> int:
    if forced_port is not None:
        if not port_is_available(forced_port):
            raise RuntimeError(
                f"El puerto solicitado por --port ({forced_port}) no esta disponible."
            )
        return forced_port

    for port in CANDIDATE_PORTS:
        if port_is_available(port):
            return port

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_container_health(container_name: str, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_state = "unknown"

    while time.monotonic() < deadline:
        inspect = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Health.Status}}", container_name],
            check=False,
            capture_output=True,
            text=True,
        )
        if inspect.returncode == 0:
            status = inspect.stdout.strip()
            last_state = status or "unknown"
            if status == "healthy":
                return
        else:
            stderr = inspect.stderr.strip()
            last_state = stderr or "container-no-disponible"
        time.sleep(2)

    raise RuntimeError(
        "Timeout esperando PostgreSQL healthy. "
        f"Ultimo estado: {last_state}"
    )


def validate_dsn(dsn: str, timeout_seconds: int) -> None:
    import psycopg

    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            with psycopg.connect(dsn, connect_timeout=3) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            return
        except Exception as exc:  # pragma: no cover - control operacional
            last_error = exc
            time.sleep(2)

    raise RuntimeError(f"No fue posible validar DSN: {last_error}")


def mask_dsn_password(dsn: str) -> str:
    marker = "://"
    if marker not in dsn:
        return dsn
    prefix, rest = dsn.split(marker, 1)
    if "@" not in rest or ":" not in rest.split("@", 1)[0]:
        return dsn
    credentials, tail = rest.split("@", 1)
    user = credentials.split(":", 1)[0]
    return f"{prefix}://{user}:***@{tail}"


def build_compose_env(port: int) -> dict[str, str]:
    env = os.environ.copy()
    env["POSTGRES_PORT"] = str(port)
    return env


def compose_cmd(compose_file: Path, *args: str) -> list[str]:
    return ["docker", "compose", "-f", str(compose_file), *args]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Runner de pruebas RNFE con PostgreSQL dedicado "
            "(auto-lift + auto-clean)."
        )
    )
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Omite la corrida de pytest base (pytest -q).",
    )
    parser.add_argument(
        "--skip-pg",
        action="store_true",
        help="Omite la corrida de pruebas marcadas requires_postgres.",
    )
    parser.add_argument(
        "--keep-services",
        action="store_true",
        help="No apaga servicios docker al finalizar.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Puerto host para PostgreSQL dedicado (opcional).",
    )

    args = parser.parse_args()

    if args.skip_base and args.skip_pg:
        log("ERROR: --skip-base y --skip-pg no pueden usarse juntos.")
        return 2

    repo_root = Path(__file__).resolve().parents[1]
    compose_file = repo_root / "infra" / "docker" / "docker-compose.yml"
    env_path = repo_root / "infra" / "docker" / ".env"
    env_example = repo_root / "infra" / "docker" / ".env.example"

    if not compose_file.exists():
        log(f"ERROR: No existe compose file: {compose_file}")
        return 2

    results: list[tuple[str, str]] = []
    overall_code = 0
    selected_port: int | None = None
    dsn: str | None = None
    pg_started = False

    try:
        log("Fase setup: verificando prerequisitos.")
        check_prerequisites()
        results.append(("setup/prerequisitos", "ok"))

        log("Fase setup: garantizando infra/docker/.env.")
        copied = ensure_local_env(env_path, env_example)
        results.append(("setup/env", "copiado" if copied else "ok"))

        if not args.skip_pg:
            selected_port = choose_port(args.port)
            compose_env = build_compose_env(selected_port)
            log(f"Fase setup: puerto PostgreSQL seleccionado: {selected_port}")

            up = run_cmd(
                compose_cmd(compose_file, "up", "-d", "--force-recreate", "postgres"),
                env=compose_env,
                cwd=repo_root,
            )
            if up.returncode != 0:
                raise RuntimeError("docker compose up fallo.")
            pg_started = True
            results.append(("setup/docker-up", "ok"))

            wait_for_container_health(POSTGRES_CONTAINER, HEALTH_TIMEOUT_SECONDS)
            results.append(("setup/healthcheck", "ok"))

            dsn = (
                "postgresql://"
                f"{POSTGRES_USER}:{POSTGRES_PASSWORD}@localhost:{selected_port}/{POSTGRES_DB}"
            )
            log(f"Fase setup: validando DSN {mask_dsn_password(dsn)}")
            validate_dsn(dsn, DSN_TIMEOUT_SECONDS)
            results.append(("setup/dsn", "ok"))
        else:
            results.append(("setup/postgres", "omitido (--skip-pg)"))

        if not args.skip_base:
            log("Fase pruebas base: ejecutando pytest -q")
            base = run_cmd(["pytest", "-q"], cwd=repo_root)
            if base.returncode == 0:
                results.append(("tests/base", "ok"))
            else:
                results.append(("tests/base", f"fallo ({base.returncode})"))
                overall_code = base.returncode
        else:
            results.append(("tests/base", "omitido (--skip-base)"))

        if not args.skip_pg:
            assert dsn is not None  # Garantizado por setup sin skip-pg.
            log("Fase pruebas postgres: ejecutando pytest -q -m requires_postgres")
            pg_env = os.environ.copy()
            pg_env["RNFE_RUN_PG_TESTS"] = "1"
            pg_env["RNFE_POSTGRES_DSN"] = dsn
            pg = run_cmd(["pytest", "-q", "-m", "requires_postgres"], env=pg_env, cwd=repo_root)
            if pg.returncode == 0:
                results.append(("tests/requires_postgres", "ok"))
            else:
                results.append(("tests/requires_postgres", f"fallo ({pg.returncode})"))
                if overall_code == 0:
                    overall_code = pg.returncode
        else:
            results.append(("tests/requires_postgres", "omitido (--skip-pg)"))

    except Exception as exc:
        log(f"ERROR: {exc}")
        if overall_code == 0:
            overall_code = 1
    finally:
        if pg_started:
            if args.keep_services:
                results.append(("cleanup", "omitido (--keep-services)"))
                log("Fase cleanup: omitida por --keep-services.")
            else:
                assert selected_port is not None
                log("Fase cleanup: apagando servicios docker del runner.")
                down = run_cmd(
                    compose_cmd(compose_file, "down", "-v"),
                    env=build_compose_env(selected_port),
                    cwd=repo_root,
                )
                if down.returncode == 0:
                    results.append(("cleanup", "ok"))
                else:
                    results.append(("cleanup", f"fallo ({down.returncode})"))
                    if overall_code == 0:
                        overall_code = down.returncode
        else:
            results.append(("cleanup", "sin servicios levantados"))

    log("Resumen de fases:")
    for phase, status in results:
        log(f"- {phase}: {status}")

    if overall_code == 0:
        log("Resultado final: OK")
    else:
        log(f"Resultado final: ERROR (exit={overall_code})")
    return overall_code


if __name__ == "__main__":
    raise SystemExit(main())
