"""Tests de la verdad de campo y de las métricas del chunker de H-Net.

No hay torch acá a propósito: la verdad de campo y el scorer son puros y tienen
que poder falsificarse sin GPU.  Si estos tests pasan con un evaluador roto, el
evaluador no sirve — así que varios están escritos como FALSIFICADORES (rompé la
implementación y se ponen en rojo), no como confirmaciones.
"""

from __future__ import annotations

import itertools
import json
import random

import numpy as np
import pytest

from lab.hnet_chunker.ground_truth import (
    CodeLexError,
    Convention,
    JsonLexError,
    code_python_ground_truth,
    json_ground_truth,
    json_tokens,
    prose_ground_truth,
    verify_json_tokens,
)
from lab.hnet_chunker.metrics import (
    aggregate,
    fixed_size_cuts,
    match_boundaries,
    random_cuts,
    score_boundaries,
    threshold_curve,
)


# ══════════════════════════════════════════════════════════════════════════════
# JSON — la verdad dura
# ══════════════════════════════════════════════════════════════════════════════


def test_json_tokens_offsets_exactos():
    data = b'{"a": 1}'
    toks = json_tokens(data)
    assert [(t.kind, t.start, t.end) for t in toks] == [
        ("{", 0, 1),
        ("string", 1, 4),  # "a"
        (":", 4, 5),
        ("number", 6, 7),
        ("}", 7, 8),
    ]


def test_json_ground_truth_token_start_excluye_el_cero():
    gt = json_ground_truth(b'{"a": 1}')
    # inicios de token: 0 (excluido), 1, 4, 6, 7
    assert gt.boundaries.tolist() == [1, 4, 6, 7]


def test_json_ground_truth_token_end_es_la_convencion_complementaria():
    gt = json_ground_truth(b'{"a": 1}', Convention.TOKEN_END)
    # fines de token: 1, 4, 5, 7, 8 (el 8 == n_bytes cae fuera del documento)
    assert gt.boundaries.tolist() == [1, 4, 5, 7]


def test_las_dos_convenciones_difieren_solo_por_el_blanco():
    """Con `json.dumps` (separadores ', ' y ': ') el desfase es de 1 byte —
    exactamente lo que la tolerancia ±1 absorbe. Esto lo MIDE, no lo supone."""
    data = json.dumps({"a": 1, "b": [2, 3]}).encode()
    start = set(json_ground_truth(data, Convention.TOKEN_START).boundaries.tolist())
    end = set(json_ground_truth(data, Convention.TOKEN_END).boundaries.tolist())
    solo_start = sorted(start - end)
    for p in solo_start:
        assert (p - 1) in end, f"la frontera token_start {p} no está a ±1 de ninguna token_end"


def test_json_utf8_las_fronteras_estan_en_bytes_no_en_codepoints():
    """Una ñ ocupa 2 bytes. Si el tokenizador trabajara en codepoints, los offsets
    de todo lo que viene después estarían corridos."""
    data = json.dumps({"señal": 1}, ensure_ascii=False).encode("utf-8")
    assert len(data) == len('{"señal": 1}') + 1  # 1 byte extra por la ñ
    toks = json_tokens(data)
    kinds = [t.kind for t in toks]
    assert kinds == ["{", "string", ":", "number", "}"]
    string_tok = toks[1]
    assert data[string_tok.start : string_tok.end].decode("utf-8") == '"señal"'
    # el número arranca DESPUÉS del byte extra
    num = toks[3]
    assert data[num.start : num.end] == b"1"


def test_json_strings_escapados_no_rompen_el_lexer():
    data = rb'{"k": "a\"b\\", "j": "x"}'
    toks = json_tokens(data)
    verify_json_tokens(data, toks)
    strings = [data[t.start : t.end] for t in toks if t.kind == "string"]
    assert strings == [b'"k"', rb'"a\"b\\"', b'"j"', b'"x"']


def test_json_roundtrip_falsifica_un_lexer_roto():
    data = b'{"a": [1, 2, {"b": null}], "c": true}'
    toks = json_tokens(data)
    verify_json_tokens(data, toks)  # no levanta

    roto = [t for t in toks if t.kind != ","]  # se come las comas
    with pytest.raises(JsonLexError):
        verify_json_tokens(data, roto)


def test_json_invalido_no_produce_verdad_inventada():
    with pytest.raises(JsonLexError):
        json_ground_truth(b'{"a": }')
    with pytest.raises(JsonLexError):
        json_tokens(b'{"a": "sin cerrar}')


def test_json_interior_de_string_es_no_evaluable():
    data = b'{"clave": "un texto largo"}'
    gt = json_ground_truth(data)
    i = data.index(b'"un texto')
    # el inicio del string SÍ es evaluable (es frontera verdadera)
    assert gt.evaluable[i]
    # el interior NO
    assert not gt.evaluable[i + 5]
    assert not gt.evaluable[i + 10]
    # ninguna frontera verdadera cae en terreno no evaluable
    assert gt.evaluable[gt.boundaries].all()


# ══════════════════════════════════════════════════════════════════════════════
# PROSA — una referencia, y el test lo dice
# ══════════════════════════════════════════════════════════════════════════════


def test_prosa_fronteras_de_palabra_en_bytes_con_acentos():
    text = "el órgano cortó"
    gt = prose_ground_truth(text)
    raw = text.encode("utf-8")
    assert gt.n_bytes == len(raw) == 17  # ó y ó suman 2 bytes extra
    # "el"=0, "órgano"=3, "cortó"=11  (en BYTES, no en chars)
    assert gt.boundaries.tolist() == [3, 11]
    for p in gt.boundaries:
        assert (raw[p] & 0xC0) != 0x80, "una frontera cayó en medio de un codepoint UTF-8"


def test_prosa_la_puntuacion_es_su_propio_token():
    gt = prose_ground_truth("hola, mundo")
    assert gt.boundaries.tolist() == [4, 6]  # la coma y la 'm'


def test_prosa_el_espacio_queda_a_un_byte_de_la_frontera():
    """La justificación de la tolerancia ±1, hecha test."""
    text = "hola mundo"
    gt = prose_ground_truth(text)
    assert gt.boundaries.tolist() == [5]
    corte_antes_del_espacio = np.array([4])
    s = score_boundaries(corte_antes_del_espacio, gt, tolerance=1)
    assert s.tp == 1, "con ±1, cortar antes del espacio cuenta como acierto"
    s0 = score_boundaries(corte_antes_del_espacio, gt, tolerance=0)
    assert s0.tp == 0, "con ±0, la misma decisión es un error — se está midiendo la convención"


# ══════════════════════════════════════════════════════════════════════════════
# CÓDIGO
# ══════════════════════════════════════════════════════════════════════════════


def test_codigo_python_fronteras_del_lexer():
    src = "x = 1\n"
    gt = code_python_ground_truth(src)
    # tokens: NAME(0) OP(2) NUMBER(4) NEWLINE(5)
    assert gt.boundaries.tolist() == [2, 4, 5]


def test_codigo_python_comentarios_y_strings_son_no_evaluables():
    src = 'a = "texto largo"  # un comentario\n'
    gt = code_python_ground_truth(src)
    i = src.index('"texto')
    assert gt.evaluable[i]  # el string arranca acá: es frontera
    assert not gt.evaluable[i + 4]  # su interior, no
    j = src.index("# un")
    assert gt.evaluable[j]
    assert not gt.evaluable[j + 3]


def test_codigo_subsplit_de_identificadores_es_convencion_y_esta_separado():
    src = "boundary_prob = RoutingModule\n"
    base = code_python_ground_truth(src)
    sub = code_python_ground_truth(src, subsplit_identifiers=True)
    assert base.kind == "code_py" and sub.kind == "code_py+ident"
    assert set(base.boundaries) < set(sub.boundaries)
    assert src.encode().index(b"_prob") in set(sub.boundaries.tolist())
    assert src.encode().index(b"Module") in set(sub.boundaries.tolist())


def test_codigo_no_parseable_no_produce_verdad():
    with pytest.raises(CodeLexError):
        code_python_ground_truth("def (:\n  ???\n")


def test_codigo_offsets_en_bytes_con_acentos_en_comentarios():
    src = "x = 1  # está roto\ny = 2\n"
    gt = code_python_ground_truth(src)
    raw = src.encode("utf-8")
    for p in gt.boundaries:
        assert (raw[p] & 0xC0) != 0x80
    # 'y' arranca después del \n, en BYTES (la á suma uno)
    assert raw.index(b"y = 2") in set(gt.boundaries.tolist())


# ══════════════════════════════════════════════════════════════════════════════
# MATCHING Y MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════════


def _brute_force_max_matching(pred, true, tol):
    """Matching máximo por fuerza bruta. Sólo para tests: valida que el greedy
    del scorer no esté sobre- ni sub-contando."""
    best = 0
    pred, true = list(pred), list(true)
    for r in range(min(len(pred), len(true)), -1, -1):
        for ps in itertools.combinations(range(len(pred)), r):
            for ts in itertools.permutations(range(len(true)), r):
                if all(abs(pred[p] - true[t]) <= tol for p, t in zip(ps, ts)):
                    return r
        if best:
            break
    return 0


def test_el_matching_greedy_es_optimo():
    """El greedy izquierda-a-derecha no es una aproximación: sobre secuencias
    ordenadas es el matching MÁXIMO. Se verifica contra fuerza bruta."""
    rng = random.Random(1234)
    for _ in range(300):
        n = rng.randint(0, 6)
        pred = sorted(rng.sample(range(1, 25), n))
        true = sorted(rng.sample(range(1, 25), rng.randint(0, 6)))
        mp, _ = match_boundaries(np.array(pred), np.array(true), tolerance=1)
        assert mp.size == _brute_force_max_matching(pred, true, 1), (pred, true)


def test_un_corte_no_puede_cubrir_dos_fronteras():
    gt = prose_ground_truth("ab cd ef")  # fronteras en 3 y 6
    s = score_boundaries(np.array([3]), gt, tolerance=1)
    assert (s.tp, s.fp, s.fn) == (1, 0, 1), "un corte no puede contar por dos aciertos"


def test_el_indice_cero_no_regala_aciertos():
    gt = prose_ground_truth("hola mundo")
    assert 0 not in gt.boundaries
    s = score_boundaries(np.array([0, 5]), gt)
    assert s.n_pred == 1 and s.tp == 1, "el 0 (PAD_PROB=1.0 del RoutingModule) se descarta"


def test_lectura_skeleton_vs_strict_solo_difiere_en_precision():
    data = b'{"k": "hola mundo"}'
    gt = json_ground_truth(data)
    dentro = data.index(b"mundo")  # corte dentro del cuerpo del string
    pred = np.array(sorted(set(gt.boundaries.tolist()) | {dentro}))

    sk = score_boundaries(pred, gt, masked=True)
    st = score_boundaries(pred, gt, masked=False)

    assert sk.recall == st.recall == 1.0, "el recall no puede cambiar con la máscara"
    assert sk.n_pred_unknown == 1 and sk.fp == 0
    assert st.fp == 1 and st.n_pred_unknown == 0
    assert st.precision < sk.precision, "strict es la COTA INFERIOR de precisión"


def test_score_perfecto_cuando_se_predice_la_verdad():
    gt = json_ground_truth(json.dumps({"a": 1, "b": "c"}).encode())
    s = score_boundaries(gt.boundaries, gt)
    assert (s.precision, s.recall, s.f1) == (1.0, 1.0, 1.0)


# ── controles negativos ───────────────────────────────────────────────────────


def test_corte_fijo_iguala_la_tasa_del_modelo():
    cuts = fixed_size_cuts(1000, 9)
    assert cuts.size == 9
    assert np.allclose(np.diff(cuts), 100, atol=1)
    assert cuts.min() > 0 and cuts.max() < 1000


def test_corte_aleatorio_iguala_la_tasa_y_es_determinista():
    a = random_cuts(1000, 20, seed=3)
    b = random_cuts(1000, 20, seed=3)
    c = random_cuts(1000, 20, seed=4)
    assert a.size == 20 and np.array_equal(a, b) and not np.array_equal(a, c)


def test_el_piso_de_azar_no_es_cero_y_por_eso_hay_que_reportarlo():
    """Con verdad densa y tolerancia ±1, cortar al azar YA saca un F1 alto.
    Cualquier baseline que no reporte este piso está inflando su resultado."""
    text = " ".join(["abc"] * 400)  # frontera cada 4 bytes
    gt = prose_ground_truth(text)
    n = gt.boundaries.size
    s = score_boundaries(random_cuts(gt.n_bytes, n, seed=0), gt)
    assert s.f1 > 0.4, f"el azar saca F1={s.f1:.2f}: el piso NO es 0 y hay que decirlo"


def test_threshold_curve_incluye_los_dos_controles_en_cada_umbral():
    data = json.dumps({"a": 1, "b": [2, 3], "c": "x"}).encode()
    gt = json_ground_truth(data)
    prob = np.zeros(len(data), dtype=np.float32)
    prob[gt.boundaries] = 0.95
    rows = threshold_curve(prob, gt)
    assert len(rows) == 9
    for r in rows:
        for k in ("hnet_skeleton", "fixed_skeleton", "random_skeleton", "hnet_strict"):
            assert k in r
        assert r["fixed_skeleton"]["n_pred"] <= r["n_cuts"]
    mejor = max(rows, key=lambda r: r["hnet_skeleton"]["f1"])
    assert mejor["hnet_skeleton"]["f1"] == 1.0


def test_aggregate_es_micro_no_macro():
    gt_chico = prose_ground_truth("a b")
    gt_grande = prose_ground_truth(" ".join("x" * 1 for _ in range(200)))
    s1 = score_boundaries(gt_chico.boundaries, gt_chico)  # perfecto, doc chico
    s2 = score_boundaries(np.array([]), gt_grande)  # nulo, doc grande
    agg = aggregate([s1, s2])
    assert agg.tp == s1.tp and agg.fn == s1.fn + s2.fn
    assert agg.recall < 0.5, "el micro-promedio no deja que un doc chico y perfecto tape uno grande y nulo"
