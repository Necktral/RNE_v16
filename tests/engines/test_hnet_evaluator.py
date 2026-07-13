"""Tests del evaluador (sin GPU): la política de preparación y el veredicto.

`test_truncar_json_destruye_la_verdad_por_eso_no_se_trunca` pinea un bug REAL que
tuvo la primera versión de este evaluador: truncaba los documentos a 8192 bytes y
se comía en silencio el 37 % de `reasoning_traces`, evaluando sólo los cortos.  Si
alguien vuelve a poner un truncado a ciegas, este test se pone en rojo.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from lab.hnet_chunker.corpus import Document
from lab.hnet_chunker.evaluate import (
    SKIP_NO_TRUTH,
    SKIP_TOO_BIG,
    SKIP_TOO_SMALL,
    _gt_for,
    _verdict,
    prepare,
)
from lab.hnet_chunker.ground_truth import Convention, json_ground_truth
from lab.hnet_chunker.metrics import score_boundaries


def _doc(text: str, kind: str) -> Document:
    return Document(
        doc_id="d",
        source="test",
        content_kind=kind,
        run_id="run-0",
        split="val",
        sha256="0" * 64,
        n_bytes=len(text.encode()),
        text=text,
    )


def test_truncar_json_destruye_la_verdad_por_eso_no_se_trunca():
    big = json.dumps({"k": ["v" * 40 for _ in range(200)]})
    assert len(big.encode()) > 4096

    # Truncar a ciegas: el JSON deja de parsear y NO hay verdad de campo.
    truncado = _doc(big[:4096], "json")
    assert _gt_for(truncado, (Convention.TOKEN_START,)) is None

    # La política real: se SALTA y se CUENTA, no se trunca ni se inventa.
    text, motivo = prepare(_doc(big, "json"), max_bytes=4096)
    assert text is None and motivo == SKIP_TOO_BIG

    # Y por debajo del tope, el documento va ENTERO.
    text, motivo = prepare(_doc(big, "json"), max_bytes=1_000_000)
    assert text == big and motivo == ""


def test_truncar_codigo_rompe_el_lexer_por_eso_no_se_trunca():
    src = '"""docstring larguísimo\n' + ("x" * 100 + "\n") * 60 + '"""\ny = 1\n'
    cortado = _doc(src[:1000], "code_py")
    assert _gt_for(cortado, (Convention.TOKEN_START,)) is None, "un .py cortado deja el docstring abierto"

    text, motivo = prepare(_doc(src, "code_py"), max_bytes=1000)
    assert text is None and motivo == SKIP_TOO_BIG


def test_la_prosa_si_se_puede_truncar_y_sigue_teniendo_verdad():
    text = "palabra ácida " * 500
    out, motivo = prepare(_doc(text, "prose"), max_bytes=1024)
    assert motivo == "" and out is not None
    raw = out.encode()
    assert len(raw) <= 1024
    assert not out.endswith(" "), "el truncado corta en límite de palabra"
    assert _gt_for(_doc(out, "prose"), (Convention.TOKEN_START,)) is not None


def test_documentos_minusculos_se_descartan():
    text, motivo = prepare(_doc("{}", "json"), max_bytes=4096)
    assert text is None and motivo == SKIP_TOO_SMALL


def test_todo_lo_que_pasa_prepare_tiene_verdad_derivable():
    """Invariante del evaluador: si `prepare` lo deja pasar, `_gt_for` NO puede
    devolver None por un problema que `prepare` haya causado."""
    docs = [
        _doc(json.dumps({"a": [1, 2, {"b": "ñandú"}]}), "json"),
        _doc("def f(x):\n    # suma uno\n    return x + 1\n", "code_py"),
        _doc("Una frase con acentos: canción, órgano. " * 20, "prose"),
    ]
    for d in docs:
        text, motivo = prepare(d, max_bytes=65536)
        assert text is not None, motivo
        assert _gt_for(_doc(text, d.content_kind), (Convention.TOKEN_START, Convention.TOKEN_END)) is not None


def test_un_documento_sin_verdad_derivable_se_cuenta_no_se_inventa():
    roto = _doc('{"a": ' + "x" * 100, "json")  # JSON inválido
    text, motivo = prepare(roto, max_bytes=65536)
    assert text is not None and motivo == ""  # prepare no juzga gramática
    assert _gt_for(roto, (Convention.TOKEN_START,)) is None  # el evaluador sí
    assert SKIP_NO_TRUTH == "no_truth"


def test_el_veredicto_dice_si_le_gana_al_corte_fijo():
    data = json.dumps({"a": 1, "b": [2, 3], "c": "x"}).encode()
    gt = json_ground_truth(data)

    # un "modelo" perfecto
    perfecto = np.zeros(len(data), dtype=np.float32)
    perfecto[gt.boundaries] = 0.9
    rows = _rows(perfecto, gt)
    v = _verdict(rows, max(rows, key=lambda r: r["hnet_skeleton"]["f1"]))
    assert v["skeleton"]["beats_fixed"] is True

    # un "modelo" que corta en todos lados (lo que hace el preentrenado en bf16)
    todo = np.full(len(data), 0.95, dtype=np.float32)
    rows = _rows(todo, gt)
    v = _verdict(rows, max(rows, key=lambda r: r["hnet_skeleton"]["f1"]))
    assert v["skeleton"]["hnet_f1"] < 1.0
    assert "beats_fixed" in v["skeleton"] and "beats_random" in v["skeleton"]


def _rows(prob, gt):
    from lab.hnet_chunker.evaluate import THRESHOLDS
    from lab.hnet_chunker.metrics import fixed_size_cuts, random_cuts

    out = []
    for thr in THRESHOLDS:
        pred = np.flatnonzero(prob >= thr)
        pred = pred[pred > 0]
        row = {"threshold": thr, "n_cuts": int(pred.size)}
        for masked in (True, False):
            tag = "skeleton" if masked else "strict"
            row[f"hnet_{tag}"] = score_boundaries(pred, gt, masked=masked).as_dict()
            row[f"fixed_{tag}"] = score_boundaries(
                fixed_size_cuts(gt.n_bytes, pred.size), gt, masked=masked
            ).as_dict()
            row[f"random_{tag}"] = score_boundaries(
                random_cuts(gt.n_bytes, pred.size, 0), gt, masked=masked
            ).as_dict()
        out.append(row)
    return out
