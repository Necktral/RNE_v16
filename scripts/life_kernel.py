"""CLI del Life Kernel soberano RNFE.

Uso:
    PYTHONPATH=. python scripts/life_kernel.py --max-steps 40
    PYTHONPATH=. python scripts/life_kernel.py --run-id life-demo --no-restore
"""

from __future__ import annotations

import argparse
import os

from runtime.life import LifeKernel, LifeKernelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Life Kernel autonomo RNFE")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--scenarios",
        default="thermal_homeostasis,resource_management",
        help="mundos/regimenes separados por coma",
    )
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--interval", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=0, help="0 = sin limite")
    parser.add_argument("--no-restore", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--memory-mode", default="strict_same_scenario")
    parser.add_argument("--closure-profile", default="baseline_fixed")
    parser.add_argument("--allow-external-reasoner", action="store_true")
    parser.add_argument("--disable-msrc", action="store_true")
    parser.add_argument("--storage-mode", choices=["sqlite", "postgres", "hybrid"], default=None)
    parser.add_argument("--sqlite-db", default=None)
    parser.add_argument("--postgres-dsn", default=None)
    parser.add_argument("--artifact-root", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.storage_mode:
        os.environ["RNFE_STORAGE_MODE"] = args.storage_mode
    if args.sqlite_db:
        os.environ["AEON_EVENT_DB"] = args.sqlite_db
    if args.postgres_dsn:
        os.environ["RNFE_POSTGRES_DSN"] = args.postgres_dsn
    if args.artifact_root:
        os.environ["RNFE_ARTIFACT_ROOT"] = args.artifact_root
    scenarios = tuple(item.strip() for item in args.scenarios.split(",") if item.strip())
    kernel = LifeKernel(
        config=LifeKernelConfig(
            run_id=args.run_id,
            scenarios=scenarios or ("thermal_homeostasis",),
            block_size=args.block_size,
            interval_s=args.interval,
            max_steps=args.max_steps,
            restore=not args.no_restore,
            checkpoint_interval=args.checkpoint_interval,
            memory_filter_mode=args.memory_mode,
            closure_profile=args.closure_profile,
            allow_external_reasoner=args.allow_external_reasoner,
            enable_msrc=not args.disable_msrc,
        )
    )
    print(f"[life-kernel] run_id={kernel.run_id} steps={kernel.total_steps}")
    results = kernel.run_until_stopped()
    for result in results:
        print(
            "[life-kernel] "
            f"step={result.step_index} action={result.decision.action} "
            f"mode={result.vital_signs.mode} "
            f"viability={result.vital_signs.viability_margin:.3f} "
            f"ioc={result.vital_signs.ioc_proxy:.3f} "
            f"checkpoint={result.checkpoint_artifact_id or '-'}"
        )
    print(
        "[life-kernel] done "
        f"run_id={kernel.run_id} total_steps={kernel.total_steps} "
        f"episode_count={kernel.organism_state.episode_count if kernel.organism_state else 0}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
