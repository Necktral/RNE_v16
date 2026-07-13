"""El reporte: el barrido ENTERO, las curvas, y la tabla final contra la baseline.

Tres cosas, y las tres tienen que poder decir que no:

  `barrido`  — TODOS los puntos, incluidos los que divergieron y los que
               empeoraron. Un barrido del que sólo se muestra el ganador no
               permite distinguir una señal del máximo de un ruido.

  `curvas`   — LM y ratio (B/chunk) paso a paso. Si divergió, se ve.

  `final`    — las 8 familias, H-Net contra el CORTE FIJO y contra el AZAR, con
               las 4 lecturas (2 convenciones × 2 máscaras). El veredicto se
               calcula, no se redacta.

Uso:
    python -m lab.hnet_chunker.report barrido /home/wis/rnfe_models/hnet/finetuned/sweep_*.jsonl
    python -m lab.hnet_chunker.report curvas  /home/wis/rnfe_models/hnet/finetuned/<name>.trainlog.json
    python -m lab.hnet_chunker.report final   <baseline.json> <finetuned.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

JSON_FAMILIES = (
    "json_rnfe/events", "json_rnfe/memory_records",
    "json_rnfe/organism_snapshots", "json_rnfe/reasoning_traces",
)
TEXT_FAMILIES = ("prose_en", "prose_es", "code_py", "code_py_ident")
ALL_FAMILIES = JSON_FAMILIES + TEXT_FAMILIES
READINGS = ("token_start/skeleton", "token_start/strict", "token_end/skeleton", "token_end/strict")

BLOCKS = " ▁▂▃▄▅▆▇█"


def spark(vals: list[float], width: int = 56) -> str:
    """Curva en una línea. No reemplaza al JSON: lo hace legible de un vistazo."""
    if not vals:
        return ""
    if len(vals) > width:
        step = len(vals) / width
        vals = [vals[int(i * step)] for i in range(width)]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return BLOCKS[4] * len(vals)
    return "".join(BLOCKS[min(8, int((v - lo) / (hi - lo) * 8))] for v in vals)


# ── el barrido completo ────────────────────────────────────────────────────────


def print_sweep(paths: list[Path]) -> None:
    rows = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                rows.append((Path(p).stem, json.loads(line)))

    print("=" * 132)
    print("EL BARRIDO COMPLETO — TODOS los puntos. Los que fallaron también, y sobre todo.")
    print("=" * 132)
    hdr = (f"{'fase':10s} {'mix':>4s} {'N':>4s} {'lr':>7s} {'λ':>5s} {'pasos':>5s} | "
           f"{'B/ch json':>9s} {'F1 json':>7s} {'Δbase':>6s} {'margen':>6s} {'g1':>3s} {'r4':>3s} | "
           f"{'F1 texto':>8s} {'caída':>6s} {'g2':>3s} | {'PASA':>5s}")
    print(hdr)
    print("-" * 132)
    for phase, r in rows:
        c = r["config"]
        if r.get("summary") is None:
            print(f"{phase[:10]:10s} {c['mix_structure']:4.2f} {c['target_n']:4.1f} {c['lr']:7.1e} "
                  f"{c['lam']:5.2f} {c['steps']:5d} | *** DIVERGIO: pérdida no finita en el paso "
                  f"{r['meta'].get('nan_at')} — sin checkpoint ***")
            continue
        s = r["summary"]
        f = s["families"]
        jf = [x for x in JSON_FAMILIES if x in f]
        tf = [x for x in TEXT_FAMILIES if x in f]
        bch = sum(f[x]["bytes_per_chunk"] for x in jf) / max(1, len(jf))
        tf1 = sum(f[x]["hnet_f1"] for x in tf) / max(1, len(tf))
        print(f"{phase[:10]:10s} {c['mix_structure']:4.2f} {c['target_n']:4.1f} {c['lr']:7.1e} "
              f"{c['lam']:5.2f} {c['steps']:5d} | {bch:9.2f} {s['json_mean_f1']:7.3f} "
              f"{s.get('json_mean_delta_vs_baseline', float('nan')):+6.3f} "
              f"{s['json_mean_margin_vs_fixed']:+6.3f} "
              f"{'SI' if s['gate1_json_beats_fixed'] else 'no':>3s} "
              f"{'SI' if s['gate1_json_robust_4'] else 'no':>3s} | {tf1:8.3f} "
              f"{s['worst_text_drop']:+6.3f} {'SI' if s['gate2'] else 'no':>3s} | "
              f"{'** SI **' if s['gate_both'] else 'no':>5s}")
    print("-" * 132)
    print("B/ch json = bytes por chunk en JSON (la baseline corta a 2.8–3.3; la verdad está a 5.2–7.1)")
    print("Δbase     = F1 json contra el MISMO modelo preentrenado en la MISMA muestra reducida")
    print("margen    = F1 H-Net − F1 corte fijo (rate-matched). NEGATIVO = el corte fijo gana.")
    print("g1 = json le gana al fijo en las 4 familias | r4 = y en las 4 lecturas | g2 = no se olvidó de leer")


# ── las curvas ─────────────────────────────────────────────────────────────────


def print_curves(paths: list[Path]) -> None:
    for p in paths:
        d = json.loads(Path(p).read_text())
        meta, log = d["meta"], d["log"]
        c = meta["config"]
        steps = [r for r in log if "lm" in r]
        vals = [r for r in log if r.get("event") == "val"]
        print("\n" + "=" * 100)
        print(f"{c['name']}  |  mix={c['mix_structure']} N={c['target_n']} lr={c['lr']:.1e} "
              f"λ={c['lam']} pasos={meta['steps_done']}/{c['steps']}")
        print(f"perdida finita: {meta['loss_finite']}"
              + ("" if meta["loss_finite"] else f"  *** DIVERGIO en el paso {meta['nan_at']} ***"))
        print("=" * 100)
        if not steps:
            print("  (sin pasos registrados)")
            continue
        lm = [r["lm"] for r in steps]
        lb = [r["lb"] for r in steps]
        bpc = [r["bytes_per_chunk"] for r in steps]
        gn = [r["grad_norm"] for r in steps]
        print(f"  LM      {spark(lm)}   {lm[0]:.3f} -> {lm[-1]:.3f}")
        print(f"  ratio   {spark(lb)}   lb {lb[0]:.3f} -> {lb[-1]:.3f}  (mínimo teórico 1.000 en ratio=1/N)")
        print(f"  B/chunk {spark(bpc)}   {bpc[0]:.2f} -> {bpc[-1]:.2f}  (objetivo N={c['target_n']})")
        print(f"  |grad|  {spark(gn)}   {gn[0]:.3f} -> {gn[-1]:.3f}")
        if vals:
            print(f"\n  {'paso':>6s} {'val LM':>8s} {'LM json':>8s} {'LM texto':>9s} "
                  f"{'B/ch json':>10s} {'B/ch texto':>11s}")
            for v in vals:
                print(f"  {v['step']:6d} {v['val_lm']:8.4f} {v['val_lm_structure']:8.4f} "
                      f"{v['val_lm_text']:9.4f} {v['val_bpc_structure']:10.2f} {v['val_bpc_text']:11.2f}")
        bad = [r for r in log if r.get("event") == "grad_norm_no_finita"]
        if bad:
            print(f"\n  ⚠ {len(bad)} pasos con grad_norm NO FINITA (saltados): {[r['step'] for r in bad][:10]}")


# ── la tabla final ─────────────────────────────────────────────────────────────


def _row(rep: dict, fam: str) -> dict | None:
    r = rep["families"].get(fam)
    if not r or not r.get("n_docs_evaluated"):
        return None
    v = r["verdict"]["skeleton"]
    best = max(r["curves"]["token_start"], key=lambda x: x["hnet_skeleton"]["f1"])
    h = best["hnet_skeleton"]
    return {
        "hnet": v["hnet_f1"], "fixed": v["fixed_f1"], "random": v["random_f1"],
        "beats": v["beats_fixed"], "thr": v["best_threshold"],
        "p": h["precision"], "r": h["recall"], "bpc": h["compression"],
        "unk": h["unknown_rate"],
        "rob4": r["robustness"]["beats_fixed_in_all_4_readings"],
        "readings": r["robustness"]["readings"],
    }


def print_final(base_path: Path, ft_path: Path) -> None:
    base = json.loads(Path(base_path).read_text())
    ft = json.loads(Path(ft_path).read_text())

    print("=" * 122)
    print("TABLA FINAL — H-Net FINE-TUNEADO contra la BASELINE y contra el CORTE FIJO (rate-matched)")
    print(f"  baseline   : {Path(base_path).name}  (dtype {base.get('dtype')})")
    print(f"  fine-tune  : {Path(ft_path).name}  (dtype {ft.get('dtype')})")
    print("=" * 122)
    print(f"{'familia':28s} | {'F1 base':>7s} {'F1 FT':>7s} {'Δ':>6s} | {'F1 fijo':>7s} {'F1 azar':>7s} "
          f"| {'gana?':>5s} {'rob4':>4s} | {'B/ch':>5s} {'P':>5s} {'R':>5s} {'unk%':>5s}")
    print("-" * 122)
    g1, g2 = True, True
    worst_drop = -9.0
    for fam in ALL_FAMILIES:
        b, f = _row(base, fam), _row(ft, fam)
        if not b or not f:
            print(f"{fam:28s} | (no evaluada)")
            continue
        d = f["hnet"] - b["hnet"]
        if fam in JSON_FAMILIES:
            g1 &= bool(f["beats"] and f["rob4"])
        else:
            worst_drop = max(worst_drop, b["hnet"] - f["hnet"])
        print(f"{fam:28s} | {b['hnet']:7.3f} {f['hnet']:7.3f} {d:+6.3f} | {f['fixed']:7.3f} "
              f"{f['random']:7.3f} | {'SI' if f['beats'] else 'NO':>5s} {'SI' if f['rob4'] else 'NO':>4s} "
              f"| {f['bpc']:5.2f} {f['p']:5.3f} {f['r']:5.3f} {f['unk']*100:5.1f}")
    g2 = worst_drop <= 0.02
    print("-" * 122)

    print("\nLAS 4 LECTURAS (F1 H-Net / F1 corte fijo). Un veredicto que se da vuelta según una")
    print("elección arbitraria de 1 byte no es un veredicto.")
    print(f"{'familia':28s} " + " ".join(f"{k.replace('token_','')[:13]:>15s}" for k in READINGS) + "  ROB4")
    for fam in ALL_FAMILIES:
        f = _row(ft, fam)
        if not f:
            continue
        cells = []
        for k in READINGS:
            c = f["readings"][k]
            mark = "" if c["beats_fixed"] else "*"
            cells.append(f"{c['hnet_f1']:.3f}/{c['fixed_f1']:.3f}{mark}".rjust(15))
        print(f"{fam:28s} " + " ".join(cells) + f"  {'SI' if f['rob4'] else 'NO'}")
    print("  (* = el corte fijo gana esa lectura)")

    print("\n" + "=" * 122)
    print(f"COMPUERTA 1 — JSON le gana al corte fijo en las 4 familias Y en las 4 lecturas : "
          f"{'SI' if g1 else 'NO'}")
    print(f"COMPUERTA 2 — prosa+código no cae más de 0.020 (caída máxima medida: {worst_drop:+.3f}): "
          f"{'SI' if g2 else 'NO'}")
    print("-" * 122)
    print(f"VEREDICTO: ¿H-Net fine-tuneado le gana al corte fijo en los datos del organismo?  "
          f"{'SI' if (g1 and g2) else 'NO'}")
    print("=" * 122)


def main() -> None:
    ap = argparse.ArgumentParser(description="Reporte del fine-tune del chunker")
    ap.add_argument("modo", choices=("barrido", "curvas", "final"))
    ap.add_argument("paths", nargs="+", type=Path)
    a = ap.parse_args()
    if a.modo == "barrido":
        print_sweep(a.paths)
    elif a.modo == "curvas":
        print_curves(a.paths)
    else:
        if len(a.paths) != 2:
            ap.error("final necesita exactamente 2 rutas: <baseline.json> <finetuned.json>")
        print_final(a.paths[0], a.paths[1])


if __name__ == "__main__":
    main()
