"""Fine-tune del CHUNKER de H-Net sobre la experiencia del organismo.

Se entrena SÓLO el camino de fronteras — `encoder` + `routing_module`, 28.50 M de
679.4 M (4.20 %) — con la pérdida ORIGINAL DE LOS AUTORES:

    loss = CE(bytes)  +  λ · load_balancing_loss(bpred, N)

`load_balancing_loss` está en `hnet/utils/train.py`, la trajeron los autores y su
docstring dice "this is the loss we used for all experiments".  No se inventa una
pérdida nueva.

═══════════════════════════════════════════════════════════════════════════════
NO SE ENTRENA CONTRA LA VERDAD DE CAMPO DEL EVALUADOR — y es a propósito
═══════════════════════════════════════════════════════════════════════════════
El evaluador deriva las fronteras verdaderas con un lexer de JSON/Python.  Sería
trivial usar ese lexer como supervisión y entrenar el `routing_module` con una
BCE contra sus fronteras.  Eso subiría el F1 y NO PROBARÍA NADA: estaríamos
midiendo con la misma regla con la que enseñamos.  El fine-tune es NO
SUPERVISADO (LM + ratio); el lexer sólo aparece en `evaluate.py`, sobre el split
`val`, y nunca toca un gradiente.

═══════════════════════════════════════════════════════════════════════════════
DTYPE: bf16 OBLIGATORIO. Y la razón NO es la que dice el folclore.
═══════════════════════════════════════════════════════════════════════════════
MEDIDO en esta rama (RTX 2070, sm_75, HEAD 45a520d), con hooks sobre los 22
`mlp.fc2` del `main_network`:

    familia        |fc2|max (medido en bf16)     techo de fp16 = 65 504
    ---------------------------------------------------------------------
    JSON events              53 248              cabe   → fp16 da 0 % de NaN
    código Python            79 360              DESBORDA → 62–95 % de NaN
    PROSA                   132 096              DESBORDA → 100 % de NaN

El NaN de fp16 es **OVERFLOW en `main_network.main_network.layers.0.mlp.fc2`**, y
es **DEPENDIENTE DE LOS DATOS**: fp16 corre perfecto sobre JSON y explota sobre
prosa.  Como la mezcla de entrenamiento TIENE que incluir prosa (si no, el
organismo se olvida de leer), **fp16 es inservible para entrenar**.  La conclusión
"bf16 obligatorio" es correcta; la explicación habitual ("NaN en el 100 % de los
logits", "activaciones ~597") no lo es: son 132 096, y sólo en prosa/código.

Costo de bf16 en Turing: sm_75 no tiene tensor cores bf16 ⇒ ~0.66 s/paso contra
~0.16 s/paso en fp16.  Se paga, porque la alternativa es entrenar con NaN.

El CAMINO DE FRONTERAS (`BoundaryProbe`: embeddings + encoder + routing_module)
NO atraviesa el `main_network` y por eso **sí es fp16-seguro** — verificado: 0 %
de NaN y 99.97–100 % de acuerdo de cortes contra bf16.  Por eso la evaluación
puede seguir en fp16 y la baseline es comparable.

═══════════════════════════════════════════════════════════════════════════════
EL LEARNING RATE DE UN FINE-TUNE NORMAL **NO MUEVE NADA ACÁ**. Y hay una razón.
═══════════════════════════════════════════════════════════════════════════════
El consejo habitual —"es un fine-tune, no un entrenamiento: empezá en 1e-5 … 5e-5"—
es FALSO para este módulo, y no por poco.  MEDIDO: 400 pasos a lr=3e-5, con λ
barrido en 0.03 / 0.3 / 1.0 / 3.0 (dos órdenes de magnitud):

    λ         B/chunk en JSON (val)      ‖Δw‖/‖w‖ en q_proj
    0.03          2.45 → 2.44                 0.0033
    0.3           2.45 → 2.44                 0.0060
    1.0           2.45 → 2.44                 0.0068
    3.0           2.45 → 2.46                 0.0073

CERO.  El ratio no se movió con NINGÚN λ.  Los pesos se movieron un 0.3–0.7 %.

La causa está en la forma del `RoutingModule` (`hnet/modules/dc.py:86-96`):

    cos_sim       = cos( q_proj(h[t]) , k_proj(h[t+1]) )
    boundary_prob = clamp( (1 - cos_sim) / 2 , 0, 1 )

**No hay logit, no hay bias, no hay umbral que correr.**  La probabilidad de
frontera ES la disimilitud coseno entre estados consecutivos del encoder, y
`q_proj`/`k_proj` arrancan en la **identidad**.  Bajar la tasa de corte exige
REORIENTAR LA GEOMETRÍA de los estados ocultos —hacerlos más parecidos entre
vecinos—, no empujar un sesgo.  Un 0.5 % de cambio en una matriz identidad no
reorienta nada.

MEDIDO, 150 pasos, λ=1.0, N=6:

    lr        B/chunk en JSON (val)     LM (val)
    3e-4          2.62 → 2.73           1.022 → 0.994     (sigue sin moverse)
    1e-3          2.62 → 5.49           1.022 → 1.184     ← se mueve
    3e-3          2.63 → 5.56           1.022 → 0.939     ← se mueve

El lr útil está en **1e-3 … 3e-3: entre 20× y 100× por encima de lo que pide la
intuición de fine-tune**.  Con la pérdida finita, verificada, en los dos casos.

═══════════════════════════════════════════════════════════════════════════════
LA MEZCLA Y λ SON PARÁMETROS, NO CONSTANTES
═══════════════════════════════════════════════════════════════════════════════
El corpus de train está 90.3:1 a favor de la estructura (398.2 MB de JSON contra
4.4 MB de prosa+código).  Entrenar con esa proporción es enseñarle al organismo a
segmentar JSON y a olvidarse de leer.  `--mix-structure` NO tiene default sano:
se barre y se mide.

λ tampoco, aunque resultó ser de SEGUNDO orden frente al lr: la pérdida LM
**premia cortar de más** (más chunks ⇒ más tokens por el `main_network` de 1536
dims ⇒ menor perplejidad por byte), así que el término de ratio es lo único que
frena el sobre-corte — pero sólo puede frenarlo si el lr le alcanza para mover la
geometría.

═══════════════════════════════════════════════════════════════════════════════
EL RESULTADO — y el punto donde MÁS ENTRENAMIENTO EMPEORA
═══════════════════════════════════════════════════════════════════════════════
Ganador de un barrido de 16 puntos: **mix=0.7, N=8, lr=1e-3, λ=1.0, 800 pasos**
(`chunker_rnfe_v1.pt`).  Evaluado con el MISMO evaluador, el MISMO split y el
MISMO dtype que la baseline (per-family 50, fp16):

    familia          F1 base   F1 FT    F1 corte fijo   ¿gana?   4 lecturas
    events            0.778    0.857        0.596         SI         SI
    memory_records    0.755    0.858        0.584         SI         SI
    organism_snaps    0.743    0.874        0.601         SI         SI
    reasoning_traces  0.761    0.862        0.590         SI         SI
    prose_en          0.828    0.838        0.577         SI         SI
    prose_es          0.768    0.809        0.585         SI         SI
    code_py           0.672    0.749        0.501         SI         SI
    code_py_ident     0.715    0.782        0.530         SI         SI

EL MECANISMO ES EL QUE SE BUSCABA.  En `events`: la precisión pasa de 0.656 a
0.954 y el sobre-corte se parte al medio (3.01 → 5.12 B/chunk, contra una verdad
de 6.22).  El recall baja de 0.955 a 0.778 — **no se conservó entero, y hay que
decirlo**: el objetivo declarado era "bajar el sobre-corte SIN perder recall", y
se perdió 0.18.  La compuerta igual se pasa con holgura porque la precisión sube
0.30.  Replicado en 3 semillas (margen contra el corte fijo: +0.270 / +0.239 /
+0.167) y confirmado en una muestra de val DISJUNTA (solapamiento 0–6 % en las 4
familias JSON).

⚠ **3000 PASOS FALLA LA COMPUERTA** (margen −0.034, contra +0.270 a los 800).
No es under-training a los 800: es que el objetivo LM y el objetivo "frontera
sintáctica" DIVERGEN.  A los 3000 pasos el LM de val en JSON llega a 0.26 (contra
0.51 a los 800) — el modelo predice bytes MUCHO mejor — y al mismo tiempo vuelve a
poner las fronteras donde le conviene al LM y no donde está la gramática.
Mejor pérdida, peor chunker.  El criterio de parada NO puede ser la pérdida.

Uso:
    python -m lab.hnet_chunker.finetune --name chunker_rnfe_v1 \
        --mix-structure 0.7 --target-n 8 --lr 1e-3 --lam 1.0 --steps 800
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lab.hnet_chunker.corpus import REPO_ROOT
from lab.hnet_chunker.model import DEFAULT_WEIGHTS, build_config

__all__ = ["FinetuneConfig", "BytePools", "finetune", "save_chunker_checkpoint"]

CKPT_DIR = Path("/home/wis/rnfe_models/hnet/finetuned")

# Las 4 tablas del organismo (estructura) y las 2 fuentes de texto del repo.
STRUCTURE_SOURCES = ("db_events", "db_reasoning_traces", "db_memory_records", "db_organism_snapshots")
TEXT_SOURCES = ("repo_prose", "repo_code_py")


# ══════════════════════════════════════════════════════════════════════════════
# DATOS
# ══════════════════════════════════════════════════════════════════════════════


class BytePools:
    """Un blob de bytes por fuente, empacado, con muestreo de ventanas contiguas.

    EMPAQUETADO: los documentos de una fuente se concatenan con `\\n`.  Es el
    packing estándar de LM.  LÍMITE DECLARADO: una ventana puede cruzar el borde
    entre dos documentos (sobre todo en `organism_snapshots`, cuyo documento medio
    mide 944 bytes < L=1024), y ahí el modelo ve una frontera que no es del
    dominio.  La alternativa —quedarse sólo con documentos ≥ L— eliminaría
    `organism_snapshots` casi entero, que es UNA DE LAS FAMILIAS DE LA COMPUERTA.
    Se elige el packing y se dice.

    PRESUPUESTO: un run de 20 000 pasos a L=1024 consume 20 MB de bytes.  No hace
    falta cargar los 398 MB: se toma una muestra ALEATORIA de documentos (no el
    prefijo del shard, que está ordenado por id y sería un corte temporal) hasta
    llenar `budget_mb` por fuente.
    """

    def __init__(self, corpus_dir: Path, *, budget_mb: float = 24.0, seed: int = 0, split: str = "train"):
        self.pools: dict[str, bytes] = {}
        self.corpus_dir = Path(corpus_dir)
        budget = int(budget_mb * 1e6)
        for src in STRUCTURE_SOURCES + TEXT_SOURCES:
            shard = self.corpus_dir / f"{src}.{split}.jsonl"
            if not shard.exists():
                raise FileNotFoundError(f"falta el shard {shard} — corré `python -m lab.hnet_chunker.corpus`")
            with shard.open("rb") as fh:
                lines = fh.readlines()
            rng = random.Random(hash((seed, src)) & 0xFFFF)
            rng.shuffle(lines)
            parts, total = [], 0
            for ln in lines:
                if not ln.strip():
                    continue
                t = json.loads(ln)["text"].encode("utf-8")
                parts.append(t)
                total += len(t) + 1
                if total >= budget:
                    break
            self.pools[src] = b"\n".join(parts)

    def sizes(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.pools.items()}

    def window(self, src: str, rng: random.Random, L: int) -> bytes:
        blob = self.pools[src]
        if len(blob) <= L:
            return blob.ljust(L, b" ")
        i = rng.randrange(0, len(blob) - L)
        return blob[i : i + L]


def source_weights(mix_structure: float) -> dict[str, float]:
    """Pesos por fuente a partir de UN escalar: la fracción de ESTRUCTURA.

    Dentro de la estructura: UNIFORME sobre las 4 tablas.  NO proporcional a los
    bytes reales (events tiene 246 MB y snapshots 3.4 MB: proporcional sería
    entrenar casi sólo con `events`).  Las 4 tablas son 4 familias de la
    compuerta y pesan igual.  Es una decisión, y queda escrita.

    Dentro del texto: UNIFORME sobre prosa y código, por la misma razón (la
    compuerta mira prose_en, prose_es, code_py y code_py_ident).
    """
    if not 0.0 <= mix_structure <= 1.0:
        raise ValueError(f"mix_structure debe estar en [0,1], no {mix_structure}")
    w = {s: mix_structure / len(STRUCTURE_SOURCES) for s in STRUCTURE_SOURCES}
    w.update({s: (1.0 - mix_structure) / len(TEXT_SOURCES) for s in TEXT_SOURCES})
    return w


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class FinetuneConfig:
    name: str
    mix_structure: float          # fracción de JSON en la mezcla (el resto: prosa+código)
    target_n: float               # N de la load_balancing_loss = factor de compresión objetivo
    lr: float
    lam: float = 1.0              # peso del término de ratio
    steps: int = 1500
    seq_len: int = 1024
    batch_size: int = 1           # medido: no hay ganancia de throughput con B>1 (compute-bound)
    warmup: int = 50
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 0
    budget_mb: float = 24.0
    log_every: int = 25
    val_every: int = 250
    dtype: str = "bfloat16"

    def as_dict(self) -> dict:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT — compatible con `BoundaryProbe.load_pretrained` SIN tocar el evaluador
# ══════════════════════════════════════════════════════════════════════════════


def save_chunker_checkpoint(model, path: Path, meta: dict) -> Path:
    """Guarda SÓLO el camino de fronteras, con las MISMAS claves que el checkpoint base.

    `embeddings.weight` + `backbone.encoder.*` + `backbone.routing_module.*`.  Con
    eso, `evaluate.py --weights <este_archivo>` funciona sin cambiar UNA LÍNEA del
    evaluador: el "antes" y el "después" pasan por exactamente el mismo código.
    Se guarda en fp32 (el probe castea al dtype que le pidan).
    """
    sd = {}
    for k, v in model.state_dict().items():
        if k == "embeddings.weight" or k.startswith("backbone.encoder.") or k.startswith(
            "backbone.routing_module."
        ):
            sd[k] = v.detach().to(torch.float32).cpu()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, path)
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════


def build_model(dtype: torch.dtype, weights: Path = DEFAULT_WEIGHTS):
    from hnet.models.mixer_seq import HNetForCausalLM

    model = HNetForCausalLM(build_config(), device="cuda", dtype=dtype)
    sd = torch.load(str(weights), map_location="cpu", mmap=True, weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"carga de pesos sucia: {len(missing)} faltantes, {len(unexpected)} inesperados")
    del sd

    for p in model.parameters():
        p.requires_grad = False
    trainable = []
    for n, p in model.named_parameters():
        if ".encoder." in n or ".routing_module." in n:
            p.requires_grad = True
            trainable.append(p)
    n_tr = sum(p.numel() for p in trainable)
    n_all = sum(p.numel() for p in model.parameters())
    return model, trainable, n_tr, n_all


def _lr_at(step: int, cfg: FinetuneConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    prog = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))


def finetune(
    cfg: FinetuneConfig,
    *,
    corpus_dir: Path | None = None,
    verbose: bool = True,
    pools: "BytePools | None" = None,
    val_pools: "BytePools | None" = None,
) -> dict:
    """`pools`/`val_pools` se pueden pasar ya construidos: en un barrido de N
    configuraciones, releer los shards N veces es tiempo de GPU tirado. Los pools
    NO dependen de la configuración (mix, N, lr, λ) — el muestreo por fuente sí, y
    ese se hace acá."""
    from hnet.utils.train import load_balancing_loss

    corpus_dir = Path(corpus_dir or REPO_ROOT / "data" / "hnet_corpus")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[cfg.dtype]

    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    t0 = time.time()
    if pools is None:
        pools = BytePools(corpus_dir, budget_mb=cfg.budget_mb, seed=cfg.seed, split="train")
    if val_pools is None:
        val_pools = BytePools(corpus_dir, budget_mb=2.0, seed=cfg.seed + 991, split="val")
    if verbose:
        print(f"  pools train: { {k: f'{v/1e6:.1f}MB' for k, v in pools.sizes().items()} }", flush=True)

    model, trainable, n_tr, n_all = build_model(dtype)
    if verbose:
        print(f"  entrenables: {n_tr/1e6:.2f}M / {n_all/1e6:.1f}M ({100*n_tr/n_all:.2f}%) "
              f"| carga+pools {time.time()-t0:.0f}s", flush=True)

    opt = torch.optim.AdamW(trainable, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)

    w = source_weights(cfg.mix_structure)
    srcs = [s for s, v in w.items() if v > 0]
    probs = [w[s] for s in srcs]

    # Batch de validación FIJO (mismo para todos los pasos y todas las corridas del
    # barrido): si cambia, las curvas de val no son comparables entre configuraciones.
    vrng = random.Random(4242)
    val_batches = [
        (s, val_pools.window(s, vrng, cfg.seq_len))
        for s in (STRUCTURE_SOURCES + TEXT_SOURCES)
        for _ in range(3)
    ]

    log: list[dict] = []
    mask = torch.ones(cfg.batch_size, cfg.seq_len, dtype=torch.bool, device="cuda")
    nan_at: int | None = None
    t_train = time.time()

    for step in range(cfg.steps):
        lr = _lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        src = rng.choices(srcs, weights=probs, k=1)[0]
        buf = b"".join(pools.window(src, rng, cfg.seq_len) for _ in range(cfg.batch_size))
        ids = torch.frombuffer(bytearray(buf), dtype=torch.uint8).long().view(
            cfg.batch_size, cfg.seq_len).cuda()

        out = model(ids, mask=mask)
        logits = out.logits
        lm = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(), ids[:, 1:].reshape(-1)
        )
        ro = out.bpred_output[0]
        lb = load_balancing_loss(ro, cfg.target_n)
        loss = lm + cfg.lam * lb

        # LA PERDIDA SE VERIFICA FINITA ANTES DE CUALQUIER OTRA COSA. Reportar una
        # velocidad sobre una pérdida NaN es reportar nada.
        if not torch.isfinite(loss):
            nan_at = step
            print(f"  *** PERDIDA NO FINITA en el paso {step}: lm={lm.item()} lb={lb.item()} — SE CORTA ***",
                  flush=True)
            break

        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
        if not torch.isfinite(gnorm):
            opt.zero_grad(set_to_none=True)
            log.append({"step": step, "event": "grad_norm_no_finita", "grad_norm": float(gnorm)})
            continue
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            ratio = ro.boundary_mask.float().mean().item()
            rec = {
                "step": step, "src": src, "lr": lr,
                "lm": round(lm.item(), 5), "lb": round(lb.item(), 5),
                "loss": round(loss.item(), 5),
                "ratio": round(ratio, 5),
                "bytes_per_chunk": round(1.0 / max(ratio, 1e-9), 3),
                "grad_norm": round(float(gnorm), 4),
            }
            log.append(rec)
            if verbose and (step % (cfg.log_every * 4) == 0 or step == cfg.steps - 1):
                print(f"    paso {step:5d} lm={rec['lm']:.4f} lb={rec['lb']:.4f} "
                      f"B/chunk={rec['bytes_per_chunk']:6.2f} (obj {cfg.target_n}) gn={rec['grad_norm']:.3f} "
                      f"[{src}]", flush=True)

        if cfg.val_every and (step % cfg.val_every == 0 or step == cfg.steps - 1):
            log.append(_validate(model, val_batches, cfg, step))
            if verbose:
                v = log[-1]
                print(f"    ---- val@{step}: LM={v['val_lm']:.4f} "
                      f"(json {v['val_lm_structure']:.4f} / texto {v['val_lm_text']:.4f})  "
                      f"B/chunk json={v['val_bpc_structure']:.2f} texto={v['val_bpc_text']:.2f}", flush=True)

    train_s = time.time() - t_train
    done = cfg.steps if nan_at is None else nan_at

    ckpt = CKPT_DIR / f"{cfg.name}.pt"
    meta = {
        "config": cfg.as_dict(),
        "n_trainable": n_tr, "n_total": n_all,
        "steps_done": done, "nan_at": nan_at,
        "s_per_step": round(train_s / max(1, done), 4),
        "train_s": round(train_s, 1),
        "source_weights": w,
        "pool_sizes_mb": {k: round(v / 1e6, 2) for k, v in pools.sizes().items()},
        "loss_finite": nan_at is None,
        "note": "loss = CE(bytes) + lam * load_balancing_loss(bpred, N) — la de los autores",
    }
    if nan_at is None:
        save_chunker_checkpoint(model, ckpt, meta)
    meta["checkpoint"] = str(ckpt) if nan_at is None else None

    log_path = CKPT_DIR / f"{cfg.name}.trainlog.json"
    log_path.write_text(json.dumps({"meta": meta, "log": log}, indent=1), encoding="utf-8")
    meta["trainlog"] = str(log_path)
    if verbose:
        print(f"  => {done} pasos en {train_s/60:.1f} min ({meta['s_per_step']:.3f} s/paso) "
              f"| perdida finita: {nan_at is None} | ckpt {ckpt.name}", flush=True)

    del model, opt, trainable
    torch.cuda.empty_cache()
    return meta


@torch.no_grad()
def _validate(model, val_batches, cfg: FinetuneConfig, step: int) -> dict:
    from hnet.utils.train import load_balancing_loss

    model.eval()
    acc: dict[str, list] = {}
    mask = torch.ones(1, cfg.seq_len, dtype=torch.bool, device="cuda")
    for src, buf in val_batches:
        ids = torch.frombuffer(bytearray(buf[: cfg.seq_len].ljust(cfg.seq_len, b" ")),
                               dtype=torch.uint8).long().view(1, cfg.seq_len).cuda()
        out = model(ids, mask=mask)
        lg = out.logits
        lm = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]).float(), ids[:, 1:].reshape(-1))
        ratio = out.bpred_output[0].boundary_mask.float().mean().item()
        grp = "structure" if src in STRUCTURE_SOURCES else "text"
        acc.setdefault(f"lm_{grp}", []).append(lm.item())
        acc.setdefault(f"ratio_{grp}", []).append(ratio)
    model.train()

    def mean(k):
        return float(np.mean(acc[k])) if acc.get(k) else float("nan")

    lm_s, lm_t = mean("lm_structure"), mean("lm_text")
    r_s, r_t = mean("ratio_structure"), mean("ratio_text")
    return {
        "step": step, "event": "val",
        "val_lm": round((lm_s + lm_t) / 2, 5),
        "val_lm_structure": round(lm_s, 5), "val_lm_text": round(lm_t, 5),
        "val_bpc_structure": round(1 / max(r_s, 1e-9), 3), "val_bpc_text": round(1 / max(r_t, 1e-9), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune del chunker de H-Net (encoder + routing_module)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--mix-structure", type=float, required=True,
                    help="fracción de JSON en la mezcla. NO tiene default: se barre y se mide.")
    ap.add_argument("--target-n", type=float, required=True, help="N de la load_balancing_loss")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--lam", type=float, default=1.0, help="peso del término de ratio")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--budget-mb", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    args = ap.parse_args()

    cfg = FinetuneConfig(
        name=args.name, mix_structure=args.mix_structure, target_n=args.target_n,
        lr=args.lr, lam=args.lam, steps=args.steps, seq_len=args.seq_len,
        warmup=args.warmup, budget_mb=args.budget_mb, seed=args.seed, dtype=args.dtype,
    )
    print(f"[{cfg.name}] mix={cfg.mix_structure} N={cfg.target_n} lr={cfg.lr} lam={cfg.lam} "
          f"steps={cfg.steps} dtype={cfg.dtype}", flush=True)
    meta = finetune(cfg)
    print(json.dumps({k: meta[k] for k in ("steps_done", "loss_finite", "s_per_step", "checkpoint")}, indent=1))


if __name__ == "__main__":
    main()
