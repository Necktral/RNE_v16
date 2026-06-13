"""Calibración de umbrales del motor de riesgo 𝔠ₜ⁺ con el historial real.

Lee (SOLO LECTURA) los certificados episódicos del storage PostgreSQL, agrupa
las series de IoC por run, computa por run: ΔIoC, fracción de mejoras,
Pr(ΔIoC≥0)_LCB (Agresti-Coull) y CVaR_α[−ΔIoC]; y reporta la distribución para
elegir los umbrales por defecto de ``runtime/certification/risk_engine.py``
(τ de CVaR y umbral de probabilidad de la regla S-I-E).

Uso:
    PYTHONPATH=. python scripts/calibrate_risk_cvar.py [--dsn <dsn>] [--alpha 0.95]

El DSN se toma de --dsn, RNFE_POSTGRES_DSN o el .env del repo (vía
StorageConfig.from_env). Salida: reporte por stdout + JSON en
data/diagnostics/risk_calibration.json (directorio no versionado).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from runtime.certification.risk_engine import agresti_coull_lcb, compute_cvar


def _percentiles(values, qs):
    if not values:
        return {q: None for q in qs}
    vs = sorted(values)
    out = {}
    for q in qs:
        k = (len(vs) - 1) * (q / 100.0)
        f, c = int(k), min(int(k) + 1, len(vs) - 1)
        out[q] = round(vs[f] + (vs[c] - vs[f]) * (k - f), 6)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=None)
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--min-episodes", type=int, default=4, help="mínimo de episodios por run")
    args = ap.parse_args()

    dsn = args.dsn or os.environ.get("RNFE_POSTGRES_DSN")
    if not dsn:
        from runtime.storage.config import StorageConfig

        dsn = StorageConfig.from_env().postgres_dsn
    if not dsn:
        raise SystemExit("Sin DSN de PostgreSQL (usa --dsn o RNFE_POSTGRES_DSN)")

    import psycopg

    with psycopg.connect(dsn) as conn:
        rows = conn.execute(
            "SELECT run_id, ioc_proxy, created_at FROM episode_certificates "
            "ORDER BY run_id, created_at"
        ).fetchall()

    series = defaultdict(list)
    for run_id, ioc, _ts in rows:
        series[run_id].append(float(ioc))

    per_run = []
    for run_id, iocs in series.items():
        if len(iocs) < args.min_episodes + 1:
            continue
        deltas = [iocs[i] - iocs[i - 1] for i in range(1, len(iocs))]
        nonneg = sum(1 for d in deltas if d >= 0.0)
        per_run.append(
            {
                "run_id": run_id,
                "episodes": len(iocs),
                "frac_nonneg": round(nonneg / len(deltas), 4),
                "p_lcb": round(agresti_coull_lcb(nonneg, len(deltas)), 4),
                "cvar": round(compute_cvar([-d for d in deltas], args.alpha), 6),
                "mean_delta": round(sum(deltas) / len(deltas), 6),
                "worst_delta": round(min(deltas), 6),
            }
        )

    cvars = [r["cvar"] for r in per_run]
    lcbs = [r["p_lcb"] for r in per_run]
    fracs = [r["frac_nonneg"] for r in per_run]

    report = {
        "alpha": args.alpha,
        "total_certificates": len(rows),
        "total_runs": len(series),
        "runs_analizados": len(per_run),
        "cvar_percentiles": _percentiles(cvars, [25, 50, 75, 90, 95, 100]),
        "p_lcb_percentiles": _percentiles(lcbs, [5, 10, 25, 50, 75, 90]),
        "frac_nonneg_percentiles": _percentiles(fracs, [5, 25, 50, 75, 95]),
        "sugerencia": {},
    }
    # Umbral τ de CVaR: que un run "sano" típico pase — p75 observado + margen
    # de 0.01. Umbral de probabilidad: p25 de las LCB observadas redondeado
    # HACIA ABAJO a múltiplo de 0.05 (redondear hacia arriba rechazaría a los
    # mismos runs que calibran el umbral).
    if cvars:
        import math

        tau = math.ceil(report["cvar_percentiles"][75] * 100) / 100 + 0.01
        prob = max(0.0, math.floor(report["p_lcb_percentiles"][25] * 20) / 20)
        report["sugerencia"] = {
            "RNFE_RISK_CVAR_THRESHOLD": round(tau, 2),
            "RNFE_RISK_PROB_THRESHOLD": prob,
            "criterio": "CVaR τ = p75 + 0.01; prob = p25 de LCBs (floor a 0.05)",
        }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("\n--- peores 5 runs por CVaR ---")
    for r in sorted(per_run, key=lambda r: r["cvar"], reverse=True)[:5]:
        print(f"  {r['run_id'][:40]:42s} ep={r['episodes']:4d} cvar={r['cvar']:.4f} "
              f"lcb={r['p_lcb']:.3f} worstΔ={r['worst_delta']:.4f}")

    out_path = Path("data/diagnostics/risk_calibration.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"report": report, "per_run": per_run}, indent=2), encoding="utf-8")
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
