"""EL EVALUADOR. Sin esto, "el chunker mejoró" es una opinión.

Corre el camino de fronteras de H-Net sobre documentos con verdad de campo
derivable y reporta, para cada familia de datos y para cada umbral:

    - F1 del modelo (lecturas `skeleton` y `strict` — ver ground_truth.py)
    - F1 del CORTE DE TAMAÑO FIJO a la MISMA tasa   (control obligatorio)
    - F1 del CORTE ALEATORIO a la MISMA tasa        (piso de azar)

Si el modelo no le gana al corte fijo, no está segmentando: está cortando.  El
reporte lo dice explícitamente en `verdict.beats_fixed`.

Uso:
    python -m lab.hnet_chunker.evaluate --out data/hnet_corpus/baseline_pretrained.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from lab.hnet_chunker.corpus import REPO_ROOT, Document, load_shard
from lab.hnet_chunker.ground_truth import (
    CodeLexError,
    Convention,
    GroundTruth,
    JsonLexError,
    code_python_ground_truth,
    json_ground_truth,
    prose_ground_truth,
)
from lab.hnet_chunker.metrics import (
    TOLERANCE_BYTES,
    aggregate,
    fixed_size_cuts,
    random_cuts,
    score_boundaries,
)

THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@dataclass
class EvalDoc:
    name: str
    family: str  # "json_rnfe" | "prose_es" | "prose_en" | "code_py"
    data: bytes
    gt: dict[str, GroundTruth]  # convención -> verdad


def _gt_for(doc: Document, conventions: Iterable[Convention]) -> dict[str, GroundTruth] | None:
    """Deriva la verdad de campo, o devuelve None si NO se puede derivar.

    NO se fabrica una segmentación plausible para lo que no parsea: se descarta y
    se cuenta en `n_skipped`.  Un documento sin verdad derivable no se evalúa."""
    out: dict[str, GroundTruth] = {}
    for conv in conventions:
        try:
            if doc.content_kind == "json":
                out[conv.value] = json_ground_truth(doc.text.encode("utf-8"), conv)
            elif doc.content_kind == "code_py":
                out[conv.value] = code_python_ground_truth(doc.text, conv)
            elif doc.content_kind == "code_py_ident":
                # CONVENCIÓN, no verdad: agrega cortes en los quiebres snake_case y
                # CamelCase DENTRO de los identificadores. Nada en la gramática de
                # Python dice que `boundary_prob` sean dos unidades. Va como familia
                # SEPARADA, nunca mezclada con la verdad léxica del lexer.
                out[conv.value] = code_python_ground_truth(
                    doc.text, conv, subsplit_identifiers=True
                )
            elif doc.content_kind == "prose":
                out[conv.value] = prose_ground_truth(doc.text, conv)
            else:
                return None
        except (JsonLexError, CodeLexError, ValueError):
            return None
    return out


SKIP_TOO_BIG = "too_big"
SKIP_TOO_SMALL = "too_small"
SKIP_NO_TRUTH = "no_truth"


def prepare(doc: Document, max_bytes: int) -> tuple[str | None, str]:
    """Prepara el documento para evaluar, o lo descarta con motivo.

    ⚠ TRUNCAR DESTRUYE LA GRAMÁTICA, y eso invalida la verdad de campo:

      - un JSON cortado a 8192 bytes NO parsea → no hay fronteras sintácticas;
      - un .py cortado deja un docstring abierto → `tokenize` levanta TokenError.

    Una primera versión de este evaluador truncaba a ciegas y se comía el 37 % de
    `reasoning_traces` en silencio, quedándose SÓLO con los documentos cortos.  Eso
    es exactamente el sesgo de selección que el evaluador existe para no cometer.

    Política:
      - json / code_py: documento ENTERO o nada.  Los que exceden `max_bytes` se
        SALTAN y se CUENTAN (`n_skipped_too_big`), y la cobertura en bytes se
        reporta.  Sesgo declarado: los documentos evaluados son los más chicos.
      - prose: se trunca en un límite de palabra.  Un prefijo de prosa sigue
        siendo prosa y su segmentación por palabras sigue siendo válida.
    """
    raw = doc.text.encode("utf-8")
    if len(raw) < 32:
        return None, SKIP_TOO_SMALL

    if doc.content_kind in ("json", "code_py", "code_py_ident"):
        if len(raw) > max_bytes:
            return None, SKIP_TOO_BIG
        return doc.text, ""

    if len(raw) <= max_bytes:
        return doc.text, ""
    cut = raw[:max_bytes]
    while cut and (cut[-1] & 0xC0) == 0x80:  # no partir un codepoint
        cut = cut[:-1]
    if cut and cut[-1] >= 0xC0:
        cut = cut[:-1]
    text = cut.decode("utf-8")
    sp = max(text.rfind(" "), text.rfind("\n"))
    if sp > 32:
        text = text[:sp]
    return text, ""


def evaluate_family(
    probe,
    docs: list[Document],
    family: str,
    *,
    conventions: tuple[Convention, ...],
    thresholds: tuple[float, ...] = THRESHOLDS,
    max_bytes: int = 65536,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Evalúa una familia de documentos. Micro-promedio sobre bytes reales."""
    per_conv: dict[str, dict[float, dict[str, list]]] = {}
    skipped: dict[str, int] = {SKIP_TOO_BIG: 0, SKIP_TOO_SMALL: 0, SKIP_NO_TRUTH: 0}
    n_eval = 0
    total_bytes = 0
    offered_bytes = sum(d.n_bytes for d in docs)
    t0 = time.time()

    for i, doc in enumerate(docs):
        text, motivo = prepare(doc, max_bytes)
        if text is None:
            skipped[motivo] += 1
            continue
        stub = Document(**{**doc.as_record(), "text": text})
        gts = _gt_for(stub, conventions)
        if gts is None:
            # Sin verdad derivable NO se evalúa. No se inventa una segmentación
            # plausible: ese es el hallazgo central de la campaña.
            skipped[SKIP_NO_TRUTH] += 1
            continue

        data = text.encode("utf-8")
        prob = probe.boundary_prob(data)
        n_eval += 1
        total_bytes += len(data)

        for conv_name, gt in gts.items():
            slot = per_conv.setdefault(conv_name, {})
            for thr in thresholds:
                pred = np.flatnonzero(prob >= thr)
                pred = pred[pred > 0]
                n_cuts = int(pred.size)
                bucket = slot.setdefault(thr, {})
                for masked in (True, False):
                    tag = "skeleton" if masked else "strict"
                    bucket.setdefault(f"hnet_{tag}", []).append(
                        score_boundaries(pred, gt, masked=masked, label="hnet")
                    )
                    bucket.setdefault(f"fixed_{tag}", []).append(
                        score_boundaries(
                            fixed_size_cuts(gt.n_bytes, n_cuts), gt, masked=masked, label="fixed"
                        )
                    )
                    bucket.setdefault(f"random_{tag}", []).append(
                        score_boundaries(
                            random_cuts(gt.n_bytes, n_cuts, seed + i), gt, masked=masked, label="random"
                        )
                    )
        if verbose and (i + 1) % 25 == 0:
            print(f"    [{family}] {i+1}/{len(docs)} docs  ({time.time()-t0:.0f}s)", flush=True)

    if not n_eval:
        return {
            "family": family,
            "n_docs_evaluated": 0,
            "n_docs_skipped": skipped,
            "note": "ningún documento con verdad de campo derivable — NO se evalúa",
        }

    curves: dict[str, list[dict]] = {}
    for conv_name, slot in per_conv.items():
        rows = []
        for thr in thresholds:
            bucket = slot[thr]
            row: dict = {"threshold": thr}
            for key, scores in bucket.items():
                row[key] = aggregate(scores).as_dict()
            rows.append(row)
        curves[conv_name] = rows

    primary = Convention.TOKEN_START.value
    best = _best_row(curves[primary], "hnet_skeleton")

    # ROBUSTEZ: ¿el veredicto sobrevive a las 4 combinaciones (convención × máscara)?
    # No es un adorno. En el JSON de RNFE el veredicto SE DA VUELTA según la convención:
    # las dos verdades tienen las MISMAS fronteras corridas 1 byte, pero eso cambia el
    # *clustering* (pares a 2 bytes vs pares adyacentes) y el cortador periódico es mucho
    # más sensible a eso que H-Net. Un veredicto que depende de una elección arbitraria a
    # nivel de 1 byte NO es un veredicto: es una preferencia. Hay que decirlo.
    combos = {}
    for conv_name, rows in curves.items():
        for tag in ("skeleton", "strict"):
            b = max(rows, key=lambda r: r[f"hnet_{tag}"]["f1"])
            combos[f"{conv_name}/{tag}"] = {
                "hnet_f1": b[f"hnet_{tag}"]["f1"],
                "fixed_f1": b[f"fixed_{tag}"]["f1"],
                "random_f1": b[f"random_{tag}"]["f1"],
                "beats_fixed": bool(b[f"hnet_{tag}"]["f1"] > b[f"fixed_{tag}"]["f1"]),
            }
    robust = all(c["beats_fixed"] for c in combos.values())

    return {
        "family": family,
        "n_docs_offered": len(docs),
        "n_docs_evaluated": n_eval,
        "n_docs_skipped": skipped,
        "total_bytes": total_bytes,
        # SESGO DECLARADO: los documentos que exceden max_bytes se saltan (truncarlos
        # rompería la gramática). La cobertura dice cuánto del material real se miró.
        "byte_coverage_of_sample": round(total_bytes / max(1, offered_bytes), 4),
        "max_bytes_per_doc": max_bytes,
        "tolerance_bytes": TOLERANCE_BYTES,
        # densidad de la verdad = fronteras por byte. Es el piso de azar del problema.
        "truth_density_per_byte": round(_density(curves, primary), 5),
        "curves": curves,
        "verdict": _verdict(curves[primary], best),
        "robustness": {
            "beats_fixed_in_all_4_readings": robust,
            "readings": combos,
            "note": (
                "4 lecturas = 2 convenciones (token_start / token_end) × 2 máscaras "
                "(skeleton / strict). Si `beats_fixed_in_all_4_readings` es False, la "
                "ventaja de H-Net DEPENDE de una elección arbitraria y NO es una ventaja."
            ),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }


def _density(curves: dict, conv: str) -> float:
    row = curves[conv][0]
    d = row["hnet_skeleton"]
    return d["n_true"] / max(1, d["n_bytes"])


def _best_row(rows: list[dict], key: str) -> dict:
    return max(rows, key=lambda r: r[key]["f1"])


def _verdict(rows: list[dict], best: dict) -> dict:
    """¿Le gana H-Net al corte fijo? La pregunta que no se puede esquivar."""
    thr = best["threshold"]
    out = {}
    for tag in ("skeleton", "strict"):
        h = best[f"hnet_{tag}"]["f1"]
        f = best[f"fixed_{tag}"]["f1"]
        r = best[f"random_{tag}"]["f1"]
        out[tag] = {
            "best_threshold": thr,
            "hnet_f1": h,
            "fixed_f1": f,
            "random_f1": r,
            "beats_fixed": bool(h > f),
            "margin_vs_fixed": round(h - f, 4),
            "beats_random": bool(h > r),
            "margin_vs_random": round(h - r, 4),
        }
    # ¿le gana el corte fijo en ALGÚN umbral? (más honesto que mirar sólo el mejor de H-Net)
    out["hnet_beats_fixed_at_any_threshold"] = bool(
        any(r["hnet_skeleton"]["f1"] > r["fixed_skeleton"]["f1"] for r in rows)
    )
    out["fixed_best_f1_over_curve"] = max(r["fixed_skeleton"]["f1"] for r in rows)
    out["hnet_best_f1_over_curve"] = max(r["hnet_skeleton"]["f1"] for r in rows)
    return out


# ── selección de documentos ────────────────────────────────────────────────────

_ES_HINT = ("que", "para", "los", " la ", " el ", "con", "porque", "más")


def _looks_spanish(text: str) -> bool:
    low = text.lower()
    hits = sum(low.count(w) for w in _ES_HINT)
    return hits >= max(5, len(text) // 400)


def collect_eval_docs(corpus_dir: Path, *, per_family: int, seed: int = 7) -> dict[str, list[Document]]:
    """Toma documentos del split VAL. Nunca de train: si no, el "antes" y el
    "después" no son comparables una vez que alguien entrene."""
    rng = random.Random(seed)
    fams: dict[str, list[Document]] = {}

    def take(shard: Path, family: str, filt: Callable[[Document], bool] | None = None) -> None:
        if not shard.exists():
            return
        docs = [d for d in load_shard(shard) if (filt is None or filt(d))]
        rng.shuffle(docs)
        fams.setdefault(family, []).extend(docs[:per_family])

    for table in ("events", "reasoning_traces", "memory_records", "organism_snapshots"):
        take(corpus_dir / f"db_{table}.val.jsonl", f"json_rnfe/{table}")

    take(corpus_dir / "repo_code_py.val.jsonl", "code_py")
    # Misma muestra, otra definición de verdad: el subsplit de identificadores que pidió
    # el paquete. Se evalúa aparte porque es CONVENCIÓN y no verdad léxica.
    shard = corpus_dir / "repo_code_py.val.jsonl"
    if shard.exists():
        docs = list(load_shard(shard))
        random.Random(seed).shuffle(docs)
        fams["code_py_ident"] = [
            Document(**{**d.as_record(), "content_kind": "code_py_ident"}) for d in docs[:per_family]
        ]
    take(corpus_dir / "repo_prose.val.jsonl", "prose_es", lambda d: _looks_spanish(d.text))
    take(corpus_dir / "repo_prose.val.jsonl", "prose_en", lambda d: not _looks_spanish(d.text))
    return fams


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluador de fronteras de H-Net (baseline preentrenado)")
    ap.add_argument("--corpus", default=str(REPO_ROOT / "data" / "hnet_corpus"))
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "hnet_corpus" / "baseline_pretrained.json"))
    ap.add_argument("--per-family", type=int, default=40)
    ap.add_argument("--max-bytes", type=int, default=65536)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--weights", default=None)
    ap.add_argument(
        "--dtype",
        default="float16",
        choices=("float16", "bfloat16", "float32"),
        help=(
            "fp16 es el correcto. fp32 REPRODUCE EL BUG del residual (ver model.py) y sólo "
            "existe para poder medirlo; el reporte lo marca como no-fidedigno."
        ),
    )
    args = ap.parse_args()

    import torch

    from lab.hnet_chunker.model import (
        DEFAULT_WEIGHTS,
        BoundaryProbe,
        build_config,
        residual_stream_is_alive,
    )

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    cfg = build_config()
    alive = residual_stream_is_alive(dtype, args.device)
    if alive:
        probe = BoundaryProbe(cfg, device=args.device, dtype=dtype)
    else:
        print(
            f"\n⚠⚠ ADVERTENCIA: con dtype={args.dtype} el STREAM RESIDUAL del encoder está MUERTO\n"
            "   (flash_attn/ops/triton/layer_norm.py devuelve residual_out=None donde el kernel\n"
            "    real devuelve `x`). Lo que sigue NO es H-Net: es H-Net sin conexiones residuales.\n"
            "   Se corre igual, a propósito, para dejar la evidencia del bug.\n",
            flush=True,
        )
        probe = BoundaryProbe.allow_lossy_dtype(cfg, device=args.device, dtype=dtype)
    info = probe.load_pretrained(Path(args.weights) if args.weights else DEFAULT_WEIGHTS)
    print(f"probe: {info}", flush=True)

    fams = collect_eval_docs(Path(args.corpus), per_family=args.per_family)
    conventions = (Convention.TOKEN_START, Convention.TOKEN_END)

    report = {
        "kind": "baseline_pretrained",
        "model": "hnet_1stage_L",
        "dtype": args.dtype,
        "residual_stream_alive": alive,
        "faithful": alive,  # si es False, estos números NO son de H-Net
        "probe": info.__dict__,
        "tolerance_bytes": TOLERANCE_BYTES,
        "thresholds": list(THRESHOLDS),
        "families": {},
    }
    for family, docs in sorted(fams.items()):
        print(f"  evaluando {family}: {len(docs)} docs", flush=True)
        report["families"][family] = evaluate_family(
            probe, docs, family, conventions=conventions, max_bytes=args.max_bytes
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreporte -> {out}")
    print_summary(report)


def print_summary(report: dict) -> None:
    print("\n" + "=" * 96)
    fiel = report.get("faithful", True)
    marca = "" if fiel else "  ⚠ RESIDUAL MUERTO — ESTO NO ES H-NET ⚠"
    print(
        f"BASELINE — H-Net preentrenado, {report.get('dtype')}, tolerancia ±1 byte, "
        f"convención token_start{marca}"
    )
    print("=" * 96)
    hdr = (
        f"{'familia':26s} {'n':>3s} {'cob':>5s} {'thr*':>5s} {'F1 hnet':>8s} {'F1 fijo':>8s} "
        f"{'F1 azar':>8s} {'gana?':>6s} {'B/chunk':>8s} {'dens':>6s} {'?%':>6s}"
    )
    print(hdr)
    print("-" * 96)
    for fam, r in sorted(report["families"].items()):
        if not r.get("n_docs_evaluated"):
            print(f"{fam:26s}  (sin verdad derivable — NO evaluado)")
            continue
        v = r["verdict"]["skeleton"]
        rows = r["curves"]["token_start"]
        best = max(rows, key=lambda x: x["hnet_skeleton"]["f1"])
        comp = best["hnet_skeleton"]["compression"]
        unk = best["hnet_skeleton"]["unknown_rate"] * 100
        print(
            f"{fam:26s} {r['n_docs_evaluated']:3d} {r['byte_coverage_of_sample']*100:4.0f}% "
            f"{v['best_threshold']:5.1f} {v['hnet_f1']:8.3f} {v['fixed_f1']:8.3f} "
            f"{v['random_f1']:8.3f} {'SI' if v['beats_fixed'] else 'NO':>6s} {comp:8.2f} "
            f"{r['truth_density_per_byte']:6.3f} {unk:5.1f}%"
        )
    print("-" * 96)
    print("cob     = bytes evaluados / bytes ofrecidos (los > max_bytes se saltan: truncar rompe la gramática)")
    print("B/chunk = bytes por chunk al mejor umbral (ratio de compresión del modelo)")
    print("dens    = densidad de la verdad (fronteras/byte). Alta densidad ⇒ piso de azar alto.")
    print("?%      = cortes en terreno SIN verdad (interior de strings), descartados en la lectura `skeleton`")
    print("\nROBUSTEZ — ¿le gana al corte fijo en las 4 lecturas (2 convenciones × 2 máscaras)?")
    print("Un veredicto que se da vuelta según una elección arbitraria de 1 byte NO es un veredicto.")
    keys = ["token_start/skeleton", "token_start/strict", "token_end/skeleton", "token_end/strict"]
    print(f"  {'familia':26s} " + " ".join(f"{k.replace('token_','')[:12]:>14s}" for k in keys) + "   ROBUSTO")
    for fam, r in sorted(report["families"].items()):
        if not r.get("n_docs_evaluated"):
            continue
        rb = r["robustness"]
        cells = []
        for k in keys:
            c = rb["readings"][k]
            cells.append(f"{c['hnet_f1']:.3f}/{c['fixed_f1']:.3f}".rjust(14))
        marca = "SI" if rb["beats_fixed_in_all_4_readings"] else "NO"
        print(f"  {fam:26s} " + " ".join(cells) + f"   {marca}")
    print("  (cada celda: F1 H-Net / F1 corte fijo)")


if __name__ == "__main__":
    main()
