"""EL BARRIDO. Se reportan TODOS los puntos, no el ganador.

Un barrido del que sólo se muestra el mejor es una mentira por omisión: sin los
puntos que salieron mal no se puede saber si el ganador es una señal o el máximo
de un ruido.  Este módulo escribe un JSONL con UNA LÍNEA POR PUNTO, incluidos los
que divergieron, los que se olvidaron de leer y los que no movieron nada.

Ejes barridos:
    --mix-structure   fracción de JSON en la mezcla   (el corpus está 90.3:1)
    --target-n        N de la load_balancing_loss     (compresión objetivo)
    --lr              learning rate
    --lam             PESO DEL TERMINO DE RATIO — el paquete no lo pidió, y es de
                      primer orden: con λ=0.03 (el valor del paper) el ratio se
                      mueve PARA EL LADO EQUIVOCADO, porque la pérdida LM PREMIA
                      cortar de más (más chunks ⇒ más tokens por el main_network
                      ⇒ menor perplejidad).  Sin λ suficiente, no hay fine-tune.

La evaluación de cada punto usa el MISMO evaluador y el MISMO split `val` que la
baseline, con menos documentos por familia (`--eval-per-family`) para que el
barrido sea pagable.  El finalista se re-evalúa con el tamaño COMPLETO (50), que
es el que reprodujo la baseline bit a bit.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from lab.hnet_chunker.corpus import REPO_ROOT
from lab.hnet_chunker.evaluate import collect_eval_docs, evaluate_family
from lab.hnet_chunker.finetune import CKPT_DIR, BytePools, FinetuneConfig, finetune
from lab.hnet_chunker.ground_truth import Convention
from lab.hnet_chunker.model import BoundaryProbe, build_config

JSON_FAMILIES = (
    "json_rnfe/events",
    "json_rnfe/memory_records",
    "json_rnfe/organism_snapshots",
    "json_rnfe/reasoning_traces",
)
TEXT_FAMILIES = ("prose_en", "prose_es", "code_py", "code_py_ident")

# La baseline, medida en fp16 con --per-family 50 y REPRODUCIDA BIT A BIT en HEAD 45a520d.
BASELINE_HNET = {
    "prose_en": 0.8276, "prose_es": 0.7681, "code_py": 0.6721, "code_py_ident": 0.7152,
    "json_rnfe/events": 0.7781, "json_rnfe/memory_records": 0.7552,
    "json_rnfe/organism_snapshots": 0.7430, "json_rnfe/reasoning_traces": 0.7614,
}
BASELINE_FIXED = {
    "prose_en": 0.5987, "prose_es": 0.6225, "code_py": 0.4295, "code_py_ident": 0.4497,
    "json_rnfe/events": 0.8301, "json_rnfe/memory_records": 0.8202,
    "json_rnfe/organism_snapshots": 0.7831, "json_rnfe/reasoning_traces": 0.8283,
}
FORGETTING_TOL = 0.02  # la compuerta 2: el F1 de prosa/código no baja más que esto


def evaluate_checkpoint(
    weights: Path | None,
    *,
    per_family: int,
    corpus_dir: Path,
    dtype: torch.dtype = torch.float16,
    max_bytes: int = 65536,
) -> dict:
    """Evalúa un checkpoint con EL MISMO evaluador que la baseline.

    fp16 y no bf16: el camino de fronteras NO atraviesa el `main_network` (que es
    donde fp16 desborda), y está verificado 0 % NaN / 99.97–100 % de acuerdo de
    cortes contra bf16.  Es el dtype en el que se midió la baseline, así que el
    "antes" y el "después" pasan por el mismo código y el mismo dtype.
    """
    probe = BoundaryProbe(build_config(), device="cuda", dtype=dtype)
    from lab.hnet_chunker.model import DEFAULT_WEIGHTS

    probe.load_pretrained(weights or DEFAULT_WEIGHTS)
    fams = collect_eval_docs(corpus_dir, per_family=per_family)
    conventions = (Convention.TOKEN_START, Convention.TOKEN_END)
    out = {}
    for family, docs in sorted(fams.items()):
        out[family] = evaluate_family(
            probe, docs, family, conventions=conventions, max_bytes=max_bytes, verbose=False
        )
    del probe
    torch.cuda.empty_cache()
    return out


def summarize(report: dict, baseline: dict) -> dict:
    """Los números que deciden la compuerta, y nada más.

    ⚠ `baseline` DEBE haberse medido con el MISMO `per_family` que `report`.  Con
    `per_family` distinto el conjunto de documentos es otro (es un prefijo del
    mismo shuffle), el micro-promedio cambia, y restar F1 de muestras distintas
    fabrica una caída (o una mejora) que no existe.  Por eso el barrido mide su
    PROPIA baseline reducida en vez de usar la constante de 50 docs.
    """
    s: dict = {"families": {}}
    for fam, r in report.items():
        if not r.get("n_docs_evaluated"):
            continue
        v = r["verdict"]["skeleton"]
        rows = r["curves"]["token_start"]
        best = max(rows, key=lambda x: x["hnet_skeleton"]["f1"])
        h = best["hnet_skeleton"]
        s["families"][fam] = {
            "hnet_f1": v["hnet_f1"], "fixed_f1": v["fixed_f1"], "random_f1": v["random_f1"],
            "beats_fixed": v["beats_fixed"], "thr": v["best_threshold"],
            "precision": h["precision"], "recall": h["recall"],
            "bytes_per_chunk": h["compression"], "unknown_rate": h["unknown_rate"],
            "robust_4": r["robustness"]["beats_fixed_in_all_4_readings"],
        }
    f = s["families"]
    # COMPUERTA 1 — JSON: le gana al corte fijo en las 4 familias Y en las 4 lecturas.
    s["gate1_json_beats_fixed"] = all(
        f.get(x, {}).get("beats_fixed", False) for x in JSON_FAMILIES
    )
    s["gate1_json_robust_4"] = all(f.get(x, {}).get("robust_4", False) for x in JSON_FAMILIES)
    s["gate1"] = s["gate1_json_beats_fixed"] and s["gate1_json_robust_4"]
    # COMPUERTA 2 — el organismo TIENE QUE SEGUIR SABIENDO LEER: prosa/código no cae > 0.02.
    drops = {
        x: round(baseline[x]["hnet_f1"] - f[x]["hnet_f1"], 4)
        for x in TEXT_FAMILIES if x in f and x in baseline
    }
    s["text_drops_vs_baseline"] = drops
    s["worst_text_drop"] = max(drops.values()) if drops else None
    s["gate2"] = bool(drops) and max(drops.values()) <= FORGETTING_TOL
    s["gate_both"] = bool(s["gate1"] and s["gate2"])
    jf = [x for x in JSON_FAMILIES if x in f]
    s["json_mean_f1"] = round(sum(f[x]["hnet_f1"] for x in jf) / max(1, len(jf)), 4)
    s["json_mean_margin_vs_fixed"] = round(
        sum(f[x]["hnet_f1"] - f[x]["fixed_f1"] for x in jf) / max(1, len(jf)), 4
    )
    s["json_mean_delta_vs_baseline"] = round(
        sum(f[x]["hnet_f1"] - baseline[x]["hnet_f1"] for x in jf if x in baseline) / max(1, len(jf)), 4
    )
    return s


def baseline_summary(report: dict) -> dict:
    """F1 del modelo PREENTRENADO en el mismo conjunto reducido. Es el 'antes' honesto."""
    out = {}
    for fam, r in report.items():
        if not r.get("n_docs_evaluated"):
            continue
        v = r["verdict"]["skeleton"]
        out[fam] = {"hnet_f1": v["hnet_f1"], "fixed_f1": v["fixed_f1"],
                    "beats_fixed": v["beats_fixed"],
                    "robust_4": r["robustness"]["beats_fixed_in_all_4_readings"]}
    return out


def print_point(tag: str, meta: dict, s: dict, baseline: dict) -> None:
    f = s["families"]
    print(f"\n  ── {tag} ──  s/paso={meta['s_per_step']:.3f}  perdida finita={meta['loss_finite']}")
    print(f"     {'familia':28s} {'F1':>6s} {'base':>6s} {'Δbase':>7s} {'fijo':>6s} {'gana':>5s} {'rob4':>5s} {'B/ch':>6s} {'P':>5s} {'R':>5s} {'unk%':>5s}")
    for fam in JSON_FAMILIES + TEXT_FAMILIES:
        if fam not in f:
            continue
        d = f[fam]
        b = baseline.get(fam, {}).get("hnet_f1", float("nan"))
        print(f"     {fam:28s} {d['hnet_f1']:6.3f} {b:6.3f} {d['hnet_f1']-b:+7.3f} "
              f"{d['fixed_f1']:6.3f} {'SI' if d['beats_fixed'] else 'no':>5s} "
              f"{'SI' if d['robust_4'] else 'no':>5s} {d['bytes_per_chunk']:6.2f} "
              f"{d['precision']:5.3f} {d['recall']:5.3f} {d['unknown_rate']*100:5.1f}")
    print(f"     COMPUERTA 1 (json>fijo en 4 fams): {'SI' if s['gate1_json_beats_fixed'] else 'NO'}"
          f" | robusta a las 4 lecturas: {'SI' if s['gate1_json_robust_4'] else 'NO'}"
          f" | COMPUERTA 2 (no olvida leer; caída max {s['worst_text_drop']}): {'SI' if s['gate2'] else 'NO'}")


def run_point(cfg: FinetuneConfig, pools, val_pools, corpus_dir: Path,
              eval_per_family: int, baseline: dict) -> dict:
    t0 = time.time()
    meta = finetune(cfg, corpus_dir=corpus_dir, pools=pools, val_pools=val_pools, verbose=True)
    if not meta["loss_finite"]:
        print(f"\n  ── {cfg.name} ── *** DIVERGIO (pérdida no finita en el paso {meta['nan_at']}). "
              f"El punto se reporta igual: un barrido sin sus fracasos no es un barrido. ***")
        return {"config": cfg.as_dict(), "meta": meta, "summary": None,
                "note": "DIVERGIO — pérdida no finita."}
    rep = evaluate_checkpoint(
        Path(meta["checkpoint"]), per_family=eval_per_family, corpus_dir=corpus_dir
    )
    s = summarize(rep, baseline)
    out = {"config": cfg.as_dict(), "meta": meta, "summary": s,
           "wall_s": round(time.time() - t0, 1)}
    print_point(cfg.name, meta, s, baseline)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Barrido mezcla × N × lr × λ del chunker")
    ap.add_argument("--phase", required=True, help="etiqueta de la fase (va al nombre del jsonl)")
    ap.add_argument("--grid", required=True,
                    help="JSON: lista de dicts con mix_structure/target_n/lr/lam/steps")
    ap.add_argument("--eval-per-family", type=int, default=12)
    ap.add_argument("--budget-mb", type=float, default=24.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--corpus", default=str(REPO_ROOT / "data" / "hnet_corpus"))
    args = ap.parse_args()

    corpus_dir = Path(args.corpus)
    grid = json.loads(args.grid)
    out_path = Path(args.out or CKPT_DIR / f"sweep_{args.phase}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[barrido {args.phase}] {len(grid)} puntos, eval per-family={args.eval_per_family}", flush=True)
    t0 = time.time()
    pools = BytePools(corpus_dir, budget_mb=args.budget_mb, seed=0, split="train")
    val_pools = BytePools(corpus_dir, budget_mb=2.0, seed=991, split="val")
    print(f"  pools listos en {time.time()-t0:.0f}s: "
          f"{ {k: f'{v/1e6:.1f}MB' for k, v in pools.sizes().items()} }", flush=True)

    # EL 'ANTES', medido en EL MISMO conjunto reducido. Sin esto, los Δ del barrido
    # comparan muestras distintas y no significan nada.
    bl_path = CKPT_DIR / f"baseline_pf{args.eval_per_family}.json"
    if bl_path.exists():
        baseline = json.loads(bl_path.read_text())
        print(f"  baseline reducida (per-family={args.eval_per_family}) desde {bl_path.name}", flush=True)
    else:
        print(f"  midiendo baseline PREENTRENADA con per-family={args.eval_per_family}...", flush=True)
        baseline = baseline_summary(
            evaluate_checkpoint(None, per_family=args.eval_per_family, corpus_dir=corpus_dir)
        )
        bl_path.write_text(json.dumps(baseline, indent=1), encoding="utf-8")
    for fam in JSON_FAMILIES + TEXT_FAMILIES:
        b = baseline.get(fam)
        if b:
            print(f"    base {fam:28s} hnet={b['hnet_f1']:.3f} fijo={b['fixed_f1']:.3f} "
                  f"gana={'SI' if b['beats_fixed'] else 'no'} rob4={'SI' if b['robust_4'] else 'no'}",
                  flush=True)

    with out_path.open("a", encoding="utf-8") as fh:
        for i, pt in enumerate(grid):
            name = pt.pop("name", None) or (
                f"{args.phase}_mix{pt['mix_structure']}_n{pt['target_n']}"
                f"_lr{pt['lr']:.0e}_lam{pt['lam']}"
            ).replace(".", "p")
            cfg = FinetuneConfig(name=name, budget_mb=args.budget_mb, **pt)
            print(f"\n{'='*100}\n[{i+1}/{len(grid)}] {name}: mix={cfg.mix_structure} N={cfg.target_n} "
                  f"lr={cfg.lr} lam={cfg.lam} steps={cfg.steps}\n{'='*100}", flush=True)
            rec = run_point(cfg, pools, val_pools, corpus_dir, args.eval_per_family, baseline)
            rec["baseline_per_family"] = args.eval_per_family
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
    print(f"\n[barrido {args.phase}] LISTO en {(time.time()-t0)/60:.1f} min -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
