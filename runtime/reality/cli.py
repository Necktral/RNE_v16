"""CLI para ejecutar validación de realidad del organismo."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Iterable

from runtime.storage import get_storage
from runtime.reality.service import RealityValidationService


def run_reality_validation(
    *,
    gate_profile: str = "ci",
    run_id: str | None = None,
    external_heat_values: Iterable[float] | None = None,
    output_json: bool = False,
) -> dict:
    """Ejecuta benchmark de validación de realidad y retorna resultado.

    Args:
        gate_profile: Perfil del gate ('ci' o 'extended').
        run_id: ID de corrida opcional.
        external_heat_values: Pack de valores de calor externo para episodios.
        output_json: Si True, imprime resultado como JSON.

    Returns:
        Diccionario con resultado del benchmark incluyendo passed, summary, artifacts.
    """
    storage = get_storage()
    service = RealityValidationService(storage=storage)

    result = service.run_benchmark(
        run_id=run_id,
        gate_profile=gate_profile,
        external_heat_values=external_heat_values,
    )

    if output_json:
        print(json.dumps(result, ensure_ascii=True, indent=2))

    return result


def main(args: list[str] | None = None) -> int:
    """Entrypoint CLI principal para validación de realidad."""
    parser = argparse.ArgumentParser(
        prog="reality-validate",
        description="Ejecuta benchmark de validación de realidad del organismo cognitivo.",
    )
    parser.add_argument(
        "--profile",
        choices=["ci", "extended"],
        default="ci",
        help="Perfil del gate de validación (default: ci).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="ID de corrida explícito.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Imprime resultado completo como JSON.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Solo imprime passed/failed y código de salida.",
    )

    parsed = parser.parse_args(args)

    try:
        result = run_reality_validation(
            gate_profile=parsed.profile,
            run_id=parsed.run_id,
            output_json=parsed.json and not parsed.quiet,
        )
    except Exception as exc:
        if not parsed.quiet:
            print(f"Error: {exc}", file=sys.stderr)
        return 2

    passed = result.get("passed", False)
    bench_run = result.get("bench_run", {})

    if parsed.quiet:
        print("PASSED" if passed else "FAILED")
    elif not parsed.json:
        print(f"Reality Validation: {'PASSED' if passed else 'FAILED'}")
        print(f"  Profile: {bench_run.get('gate_profile', 'unknown')}")
        print(f"  Episodes: {bench_run.get('total_episodes', 0)}")
        print(f"  Closure Rate: {bench_run.get('closure_rate', 0.0):.2%}")
        print(f"  Continuity Mean: {bench_run.get('continuity_mean', 0.0):.2%}")
        print(f"  Collapse Count: {bench_run.get('collapse_count', 0)}")
        if result.get("artifact"):
            print(f"  Artifact: {result['artifact'].get('abs_path', 'N/A')}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
