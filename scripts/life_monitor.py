#!/usr/bin/env python3
"""Monitor de signos vitales en vivo del organismo (LifeKernel).

Tail de solo-lectura sobre el ledger de eventos en postgres. Muestra el latido:
vitales, decisión, tier de cómputo, invocaciones del razonador 7B y VRAM del GPU.
No toca el organismo — solo observa.

Uso:
  source .env.life
  PYTHONPATH=. python scripts/life_monitor.py --run-id aeon-01
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List

# Asegura runtime importable si se corre desde otro cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.storage.config import StorageConfig  # noqa: E402


def _connect(dsn: str):
    import psycopg  # psycopg v3

    return psycopg.connect(dsn, autocommit=True)


def _rows(conn, run_id: str, event_types: List[str], limit: int) -> List[Dict[str, Any]]:
    placeholders = ",".join(["%s"] * len(event_types))
    sql = (
        "SELECT event_type, payload_jsonb, event_ts FROM ledger_events "
        f"WHERE run_id = %s AND event_type IN ({placeholders}) "
        "ORDER BY event_ts DESC LIMIT %s"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (run_id, *event_types, limit))
        out = []
        for event_type, payload, ts in cur.fetchall():
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    payload = {}
            out.append({"event_type": event_type, "payload": payload or {}, "ts": ts})
        return out


def _count(conn, run_id: str, like: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM ledger_events WHERE run_id = %s AND event_type LIKE %s",
            (run_id, like),
        )
        return int(cur.fetchone()[0])


def _external_calls(conn, run_id: str) -> int:
    """Cuenta señales de que el 7B pensó: tier_3 elegido o eventos externos."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM ledger_events WHERE run_id = %s AND ("
            "event_type LIKE '%%external%%' OR event_type LIKE '%%ext_open%%' OR "
            "payload_jsonb::text LIKE '%%tier_3_external%%')",
            (run_id,),
        )
        return int(cur.fetchone()[0])


def _vram() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        vals = [v.strip() for v in (out.stdout or "").splitlines() if v.strip()]
        return f"{vals[-1]} MiB" if vals else "?"
    except Exception:
        return "n/a"


def _f(payload: Dict[str, Any], *path, default="?"):
    cur: Any = payload
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


CLR = "\033[2J\033[H"
DIM = "\033[2m"; RST = "\033[0m"; B = "\033[1m"
GREEN = "\033[32m"; YELLOW = "\033[33m"; RED = "\033[31m"; CYAN = "\033[36m"


def _mode_color(mode: str) -> str:
    return {
        "normal": GREEN, "conservative": YELLOW, "recovery": YELLOW,
        "quarantine": RED, "rollback": RED, "shutdown_safe": RED,
    }.get(str(mode), RST)


def render(run_id: str, steps: List[Dict[str, Any]], genesis: List[Dict[str, Any]],
           ext_calls: int, checkpoints: int) -> str:
    lines = [f"{CLR}{B}{CYAN}⟡ {run_id} — signos vitales{RST}   {DIM}{time.strftime('%H:%M:%S')}{RST}",
             "─" * 66]
    if not steps and not genesis:
        lines.append(f"{DIM}esperando el primer latido...{RST}")
        lines.append(f"VRAM RTX 2070: {_vram()}")
        return "\n".join(lines)

    if steps:
        top = steps[0]["payload"]
        vs = top.get("vital_signs", {})
        dec = top.get("decision", {})
        mode = _f(vs, "mode")
        mc = _mode_color(mode)
        step_idx = _f(top, "step_index")
        lines += [
            f"latido {B}#{step_idx}{RST}   acción: {B}{_f(dec,'action')}{RST}   "
            f"modo: {mc}{mode}{RST}   escenario: {_f(top,'scenario')}",
            "",
            f"  viabilidad  {_bar(_f(vs,'viability_margin'))}   "
            f"IoC {_bar(_f(vs,'ioc_proxy'))}",
            f"  riesgo      {_bar(_f(vs,'risk_score'), invert=True)}   "
            f"presión rec {_bar(_f(vs,'resource_pressure'), invert=True)}",
            f"  continuidad {_bar(_f(vs,'identity_continuity'))}   "
            f"calidad cog {_bar(_f(vs,'cognitive_quality'))}",
            f"  pureza mem  {_bar(_f(vs,'memory_purity'))}   "
            f"certificado {_f(vs,'certified')}",
        ]
        # tira de últimos latidos (acciones)
        actions = [str(_f(s['payload'].get('decision', {}), 'action', default='·')) for s in reversed(steps)]
        short = {"act": "▮", "explore": "◇", "sleep": "z", "self_modify": "✎",
                 "rollback": "⟲", "quarantine": "▨", "shutdown": "■"}
        strip = "".join(short.get(a, "·") for a in actions)
        lines += ["", f"  latidos: {DIM}{strip}{RST}"]

    lines += [
        "─" * 66,
        f"  episodios: {B}{steps[0]['payload'].get('step_index','?') if steps else 0}{RST}   "
        f"pensó-fuerte(7B): {B}{ext_calls}{RST}   checkpoints: {checkpoints}   "
        f"VRAM: {B}{_vram()}{RST}",
    ]
    return "\n".join(lines)


def _bar(value: Any, width: int = 14, invert: bool = False) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return f"{DIM}{'?':>{width+6}}{RST}"
    v = max(0.0, min(1.0, v))
    filled = int(round(v * width))
    good = (v < 0.5) if invert else (v >= 0.5)
    color = GREEN if good else (YELLOW if (0.3 <= v <= 0.7) else RED)
    return f"{color}{'█'*filled}{DIM}{'░'*(width-filled)}{RST} {v:.2f}"


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Monitor de signos vitales del organismo.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--window", type=int, default=40, help="latidos en la tira")
    ap.add_argument("--once", action="store_true", help="una sola lectura y salir")
    args = ap.parse_args(argv)

    cfg = StorageConfig.from_env()
    dsn = cfg.postgres_dsn
    if not dsn:
        print("No hay RNFE_POSTGRES_DSN / modo postgres. ¿Cargaste .env.life?", file=sys.stderr)
        return 2
    conn = _connect(dsn)
    try:
        while True:
            steps = _rows(conn, args.run_id, ["life.step.completed"], args.window)
            genesis = _rows(conn, args.run_id, ["life.genesis", "life.identity.restored"], 1)
            ext_calls = _external_calls(conn, args.run_id)
            checkpoints = _count(conn, args.run_id, "life.checkpoint.%")
            sys.stdout.write(render(args.run_id, steps, genesis, ext_calls, checkpoints) + "\n")
            sys.stdout.flush()
            if args.once:
                return 0
            time.sleep(max(0.2, args.interval))
    except KeyboardInterrupt:
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
