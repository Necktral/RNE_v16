"""CLI del Life Kernel soberano RNFE.

Uso:
    PYTHONPATH=. python scripts/life_kernel.py --max-steps 40
    PYTHONPATH=. python scripts/life_kernel.py --run-id life-demo --no-restore
"""

from __future__ import annotations

import argparse
import os
import signal

from runtime.life import LifeKernel, LifeKernelConfig


_TRUE = {"1", "true", "yes", "on"}
_COMPUTE_TIERS = (
    "tier_0_deterministic",
    "tier_1_local_light",
    "tier_2_specialized",
    "tier_3_external",
)


class _Reposo(Exception):
    """Señal interna de reposo gracioso (SIGINT/SIGTERM)."""


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE


def _default_max_compute_tier() -> str:
    raw = (
        os.environ.get("RNFE_MAX_COMPUTE_TIER")
        or os.environ.get("RNFE_LIFE_MAX_COMPUTE_TIER")
        or "tier_2_specialized"
    )
    value = raw.strip().lower()
    return value if value in _COMPUTE_TIERS else "tier_2_specialized"


def _install_signal_handlers() -> None:
    def _handler(signum, _frame):
        raise _Reposo()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


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
    parser.add_argument(
        "--allow-external-reasoner",
        action="store_true",
        default=(
            _env_bool("RNFE_ALLOW_EXTERNAL_REASONER")
            or _env_bool("RNFE_EXTERNAL_REASONER_RUNTIME")
        ),
        help="habilita consult_external y ruteo tier_3; también se puede activar con RNFE_ALLOW_EXTERNAL_REASONER=1",
    )
    parser.add_argument(
        "--max-compute-tier",
        choices=_COMPUTE_TIERS,
        default=_default_max_compute_tier(),
        help="techo del router de cómputo; RNFE_MAX_COMPUTE_TIER=tier_3_external habilita el 7B cuando la política lo permite",
    )
    parser.add_argument(
        "--revive",
        action="store_true",
        help="si despierta dañado (cuarentena/rollback), rueda atrás a su último yo sano antes de vivir",
    )
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
            max_compute_tier=args.max_compute_tier,
            enable_msrc=not args.disable_msrc,
        )
    )
    _install_signal_handlers()
    # Revivir: si despierta dañado, rueda atrás a su último checkpoint sano.
    if args.revive and kernel.last_vitals is not None and kernel.last_vitals.mode in {"quarantine", "rollback"}:
        prev_steps = kernel.total_steps
        if kernel._restore_latest_healthy_checkpoint():
            print(
                f"[life-kernel] ⟲ rollback: despertó dañado (modo={kernel.last_vitals.mode}); "
                f"rodó atrás a su último yo sano (paso {prev_steps} → {kernel.total_steps}).",
                flush=True,
            )
            kernel.last_vitals = kernel.vitals_service.bootstrap(
                run_id=kernel.run_id,
                organism_state=kernel.organism_state,
                lineage=kernel.lineage,
            )
    born = kernel.total_steps == 0
    print(
        f"[life-kernel] {'⟡ nace' if born else '⟡ despierta'} "
        f"run_id={kernel.run_id} desde_paso={kernel.total_steps} — viviendo...",
        flush=True,
    )
    import time as _time

    limit = int(args.max_steps)
    count = 0
    try:
        while True:
            if limit and count >= limit:
                break
            result = kernel.step()
            count += 1
            print(
                "[life-kernel] "
                f"latido={result.step_index} acción={result.decision.action} "
                f"modo={result.vital_signs.mode} "
                f"viab={result.vital_signs.viability_margin:.3f} "
                f"ioc={result.vital_signs.ioc_proxy:.3f} "
                f"riesgo={result.vital_signs.risk_score:.3f} "
                f"presión={result.vital_signs.resource_pressure:.3f} "
                f"cont={result.vital_signs.identity_continuity:.3f}",
                flush=True,
            )
            if result.decision.action == "shutdown":
                print("[life-kernel] ■ el organismo decidió apagarse (shutdown seguro).", flush=True)
                break
            if args.interval > 0:
                _time.sleep(args.interval)
    except _Reposo:
        print(
            "\n[life-kernel] ☾ reposo: el organismo se detiene; su identidad quedó "
            f"preservada en checkpoint (run_id={kernel.run_id}, paso {kernel.total_steps}). "
            "Relanzá con el mismo --run-id para resucitarlo.",
            flush=True,
        )
    print(
        "[life-kernel] estado: "
        f"run_id={kernel.run_id} total_steps={kernel.total_steps} "
        f"episode_count={kernel.organism_state.episode_count if kernel.organism_state else 0}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
