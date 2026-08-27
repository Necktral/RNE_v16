"""Tests del fine-tune y del barrido. Falsificadores, no decorativos.

Los tres que más pesan:

  * `test_la_compuerta_NO_se_pasa_con_3_de_4_familias_json` — la compuerta es
    conjuntiva por definición.  Si alguien la relaja a "la mayoría de las
    familias" para que su corrida pase, esto se pone en rojo.

  * `test_el_checkpoint_es_DROP_IN_para_el_evaluador` — el "antes" y el "después"
    tienen que pasar por EXACTAMENTE el mismo código de evaluación.  Si el
    checkpoint del fine-tune necesitara un cargador propio, la comparación
    dejaría de ser una comparación.

  * `test_los_deltas_exigen_baseline_del_MISMO_tamano_de_muestra` — restar F1
    medidos sobre conjuntos de documentos distintos fabrica caídas y mejoras que
    no existen.  Es el error que este barrido cometió y corrigió.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab.hnet_chunker.finetune import (
    STRUCTURE_SOURCES,
    TEXT_SOURCES,
    BytePools,
    FinetuneConfig,
    _lr_at,
    source_weights,
)
from lab.hnet_chunker.sweep import (
    FORGETTING_TOL,
    JSON_FAMILIES,
    TEXT_FAMILIES,
    summarize,
)

CORPUS = Path(__file__).resolve().parents[2] / "data" / "hnet_corpus"
requires_corpus = pytest.mark.skipif(
    not (CORPUS / "db_events.train.jsonl").exists(),
    reason="hace falta el corpus: `python -m lab.hnet_chunker.corpus`",
)


# ── la mezcla ──────────────────────────────────────────────────────────────────


def test_la_mezcla_no_tiene_default_y_reparte_uniforme_dentro_de_cada_grupo():
    w = source_weights(0.7)
    assert sum(w.values()) == pytest.approx(1.0)
    assert sum(w[s] for s in STRUCTURE_SOURCES) == pytest.approx(0.7)
    assert sum(w[s] for s in TEXT_SOURCES) == pytest.approx(0.3)
    # uniforme DENTRO de la estructura: `events` tiene 246 MB y `snapshots` 3.4 MB.
    # Pesar por bytes sería entrenar casi sólo con `events`, y las 4 son familias
    # de la compuerta.
    assert len({round(w[s], 9) for s in STRUCTURE_SOURCES}) == 1
    assert len({round(w[s], 9) for s in TEXT_SOURCES}) == 1


def test_mezcla_extrema_100_estructura_deja_el_texto_en_cero():
    w = source_weights(1.0)
    assert all(w[s] == 0.0 for s in TEXT_SOURCES)
    assert sum(w[s] for s in STRUCTURE_SOURCES) == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [-0.1, 1.1, 2.0])
def test_mezcla_fuera_de_rango_revienta(bad):
    with pytest.raises(ValueError):
        source_weights(bad)


# ── el scheduler ───────────────────────────────────────────────────────────────


def test_el_lr_calienta_y_despues_decae():
    cfg = FinetuneConfig(name="t", mix_structure=0.7, target_n=6, lr=1e-4, steps=1000, warmup=100)
    assert _lr_at(0, cfg) == pytest.approx(1e-4 / 100)
    assert _lr_at(99, cfg) == pytest.approx(1e-4)
    mid, end = _lr_at(550, cfg), _lr_at(999, cfg)
    assert _lr_at(99, cfg) > mid > end
    assert end == pytest.approx(1e-4 * 0.1, rel=0.02)  # piso del coseno


# ── los pools ──────────────────────────────────────────────────────────────────


@requires_corpus
def test_los_pools_entregan_ventanas_de_exactamente_L_bytes():
    import random

    pools = BytePools(CORPUS, budget_mb=0.5, seed=0, split="val")
    rng = random.Random(0)
    for src in STRUCTURE_SOURCES + TEXT_SOURCES:
        assert len(pools.window(src, rng, 1024)) == 1024


@requires_corpus
def test_los_pools_son_deterministas_con_la_misma_semilla():
    import random

    a = BytePools(CORPUS, budget_mb=0.5, seed=7, split="val")
    b = BytePools(CORPUS, budget_mb=0.5, seed=7, split="val")
    assert a.sizes() == b.sizes()
    assert a.window("db_events", random.Random(1), 512) == b.window("db_events", random.Random(1), 512)


@requires_corpus
def test_los_pools_salen_del_split_pedido_y_no_lo_mezclan():
    """Entrenar con `val` y evaluar con `val` sería medirse a sí mismo."""
    tr = BytePools(CORPUS, budget_mb=0.3, seed=0, split="train")
    va = BytePools(CORPUS, budget_mb=0.3, seed=0, split="val")
    assert tr.pools["db_events"] != va.pools["db_events"]


# ── LA COMPUERTA ───────────────────────────────────────────────────────────────


def _fam(hnet, fixed, *, robust=True):
    return {
        "verdict": {"skeleton": {"hnet_f1": hnet, "fixed_f1": fixed, "random_f1": 0.5,
                                 "beats_fixed": hnet > fixed, "best_threshold": 0.5}},
        "curves": {"token_start": [{"threshold": 0.5, "hnet_skeleton": {
            "f1": hnet, "precision": 0.8, "recall": 0.8, "compression": 5.0, "unknown_rate": 0.1}}]},
        "robustness": {"beats_fixed_in_all_4_readings": robust},
        "n_docs_evaluated": 10,
    }


def _report(json_hnet, json_fixed, text_hnet, *, robust=True):
    r = {f: _fam(json_hnet, json_fixed, robust=robust) for f in JSON_FAMILIES}
    r.update({f: _fam(text_hnet, 0.5, robust=True) for f in TEXT_FAMILIES})
    return r


BASE = {f: {"hnet_f1": 0.80, "fixed_f1": 0.83, "beats_fixed": False, "robust_4": False}
        for f in JSON_FAMILIES}
BASE.update({f: {"hnet_f1": 0.80, "fixed_f1": 0.50, "beats_fixed": True, "robust_4": True}
             for f in TEXT_FAMILIES})


def test_la_compuerta_pasa_solo_si_json_gana_Y_no_se_olvida_de_leer():
    s = summarize(_report(0.90, 0.83, 0.80), BASE)
    assert s["gate1"] and s["gate2"] and s["gate_both"]


def test_la_compuerta_NO_se_pasa_con_3_de_4_familias_json():
    """La compuerta es CONJUNTIVA. Tres de cuatro no es 'casi'."""
    rep = _report(0.90, 0.83, 0.80)
    rep["json_rnfe/events"] = _fam(0.80, 0.83)  # una sola familia pierde
    s = summarize(rep, BASE)
    assert not s["gate1_json_beats_fixed"]
    assert not s["gate1"]
    assert not s["gate_both"]


def test_ganar_en_una_sola_lectura_NO_es_ganar():
    """El baseline YA le gana al corte fijo en 3 de las 4 lecturas del JSON. Lo que
    hay que dar vuelta es la 4ta. Si `robust_4` es False, no hay ventaja."""
    s = summarize(_report(0.90, 0.83, 0.80, robust=False), BASE)
    assert s["gate1_json_beats_fixed"]      # gana en la lectura primaria...
    assert not s["gate1_json_robust_4"]     # ...pero no en las 4
    assert not s["gate1"]                   # ⇒ NO pasa


def test_olvidarse_de_leer_hunde_la_compuerta_aunque_el_json_sea_perfecto():
    s = summarize(_report(0.99, 0.83, 0.80 - FORGETTING_TOL - 0.001), BASE)
    assert s["gate1"]
    assert not s["gate2"]
    assert not s["gate_both"]


def test_la_tolerancia_de_olvido_es_exactamente_002_y_no_se_mueve():
    assert FORGETTING_TOL == 0.02
    justo = summarize(_report(0.99, 0.83, 0.80 - 0.02), BASE)
    pasado = summarize(_report(0.99, 0.83, 0.80 - 0.0201), BASE)
    assert justo["gate2"] and not pasado["gate2"]


def test_los_deltas_exigen_baseline_del_MISMO_tamano_de_muestra():
    """Con `per_family` distinto el conjunto de documentos es OTRO. Restar F1 de
    muestras distintas fabrica una caída que no existe: acá, un modelo IDENTICO a
    su baseline 'pierde' 0.05 sólo por comparar contra la muestra equivocada."""
    rep = _report(0.90, 0.83, 0.80)
    otra_muestra = {**BASE, **{f: {"hnet_f1": 0.85, "fixed_f1": 0.5, "beats_fixed": True,
                                   "robust_4": True} for f in TEXT_FAMILIES}}
    s_mal = summarize(rep, otra_muestra)
    s_bien = summarize(rep, BASE)
    assert s_mal["worst_text_drop"] == pytest.approx(0.05)   # caída FABRICADA
    assert not s_mal["gate2"]
    assert s_bien["worst_text_drop"] == pytest.approx(0.0)   # la real
    assert s_bien["gate2"]


# ── el contrato con el evaluador ───────────────────────────────────────────────


def test_el_checkpoint_es_DROP_IN_para_el_evaluador():
    """`save_chunker_checkpoint` tiene que emitir EXACTAMENTE las claves que
    `BoundaryProbe.load_pretrained` busca. Si no, el "antes" y el "después" no
    pasan por el mismo código y la comparación deja de serlo."""
    import inspect

    from lab.hnet_chunker import finetune as ft
    from lab.hnet_chunker import model as md

    guardadas = inspect.getsource(ft.save_chunker_checkpoint)
    leidas = inspect.getsource(md.BoundaryProbe.load_pretrained)
    for clave in ("embeddings.weight", "backbone.encoder.", "backbone.routing_module."):
        assert clave in guardadas, f"el checkpoint no guarda {clave}"
        assert clave in leidas, f"el probe no lee {clave}"


def test_la_perdida_es_la_de_los_autores_y_no_una_inventada():
    """Si alguien reemplaza `load_balancing_loss` por una pérdida propia, el
    resultado deja de ser comparable con el paper y hay que decirlo."""
    import inspect

    from lab.hnet_chunker import finetune as ft

    src = inspect.getsource(ft.finetune)
    assert "from hnet.utils.train import load_balancing_loss" in src
    assert "load_balancing_loss(ro, cfg.target_n)" in src
    assert "cross_entropy" in src


def test_el_fine_tune_NO_usa_la_verdad_de_campo_del_evaluador():
    """Entrenar contra el lexer que después mide sería medir con la regla con la
    que se enseñó. El fine-tune es NO SUPERVISADO: LM + ratio, nada más."""
    import inspect

    from lab.hnet_chunker import finetune as ft

    src = inspect.getsource(ft)
    for prohibido in ("ground_truth", "json_ground_truth", "prose_ground_truth",
                      "code_python_ground_truth", "metrics"):
        assert prohibido not in src, f"el fine-tune importa {prohibido}: hay fuga de la verdad de campo"


def test_solo_se_entrena_el_camino_de_fronteras():
    import inspect

    from lab.hnet_chunker import finetune as ft

    src = inspect.getsource(ft.build_model)
    assert "p.requires_grad = False" in src          # todo congelado primero
    assert '".encoder." in n or ".routing_module." in n' in src


def test_el_default_de_dtype_es_bf16_porque_fp16_desborda_en_prosa():
    """fp16 NO da NaN en JSON (53k < 65504) pero SI en prosa (132k > 65504). Como la
    mezcla incluye prosa por diseño, entrenar en fp16 es entrenar con NaN."""
    cfg = FinetuneConfig(name="t", mix_structure=0.7, target_n=6, lr=1e-5)
    assert cfg.dtype == "bfloat16"


@requires_corpus
def test_config_serializa_entera_para_que_el_barrido_sea_auditable():
    cfg = FinetuneConfig(name="x", mix_structure=0.5, target_n=4.0, lr=2e-5, lam=1.5)
    d = cfg.as_dict()
    for k in ("mix_structure", "target_n", "lr", "lam", "steps", "seq_len", "seed", "dtype"):
        assert k in d
    assert json.loads(json.dumps(d)) == d
