"""Métricas de segmentación: P/R/F1 con tolerancia, curva de umbral y controles negativos.

═══════════════════════════════════════════════════════════════════════════════
POR QUÉ LA TOLERANCIA ES ±1 BYTE — justificación, no conveniencia
═══════════════════════════════════════════════════════════════════════════════

La tolerancia NO está para inflar el número.  Está porque la convención de a
quién pertenece el blanco separador es genuinamente indecidible:

    "hola mundo"          → ¿el corte va antes del espacio (byte 4) o antes de
                             la 'm' (byte 5)?
    '{"a": 1, "b": 2}'    → `json.dumps` mete `", "` y `": "`.  El fin del token
                             `,` es el byte p; el inicio del token siguiente es
                             p+1.  Un byte de diferencia, y ninguna de las dos
                             respuestas es más correcta que la otra.

±1 byte absorbe EXACTAMENTE ese blanco de un byte y nada más.  Con ±2 ya se
estarían perdonando errores reales.  Con ±0 se estaría midiendo qué convención
eligió el modelo, no si segmenta bien.

CONTRAPARTIDA, y hay que mirarla de frente: la tolerancia SUBE EL PISO DE AZAR.
Si la verdad tiene una frontera cada 5 bytes, entonces 3 de cada 5 posiciones
están a ≤1 byte de una frontera y un chunker que corte AL AZAR acierta ~60% de
sus cortes.  Por eso `random_cuts` no es un adorno: sin el piso de azar, un F1
de 0.55 puede ser indistinguible del ruido y parecer un logro.

MATCHING UNO-A-UNO.  Un corte predicho no puede "cubrir" tres fronteras
verdaderas: se emparejan greedy de izquierda a derecha, cada predicho con a lo
sumo una verdadera y viceversa.  Sobre secuencias ordenadas y ventanas de
tolerancia simétricas, el greedy más-a-la-izquierda es óptimo (es el matching
máximo del bigrafo de intervalos) — y eso está TESTEADO contra fuerza bruta en
tests/engines/test_hnet_ground_truth.py, no supuesto.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from lab.hnet_chunker.ground_truth import GroundTruth

__all__ = [
    "SegmentationScore",
    "fixed_size_cuts",
    "match_boundaries",
    "random_cuts",
    "score_boundaries",
    "threshold_curve",
]

TOLERANCE_BYTES = 1


@dataclass
class SegmentationScore:
    """Resultado de comparar cortes predichos contra fronteras verdaderas."""

    n_true: int
    n_pred: int
    tp: int
    fp: int
    fn: int
    n_pred_unknown: int  # cortes en terreno SIN verdad (solo en modo `skeleton`)
    tolerance: int
    masked: bool
    n_bytes: int
    kind: str = ""
    convention: str = ""
    label: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def compression(self) -> float:
        """Bytes por chunk producido por el chunker (L / n_cortes)."""
        return self.n_bytes / self.n_pred if self.n_pred else float("inf")

    @property
    def unknown_rate(self) -> float:
        """Fracción de los cortes predichos que cayeron en terreno sin verdad.

        NÚMERO CLAVE: si es alto, el F1 `skeleton` está mirando una minoría de
        lo que el modelo hace y no debe leerse como "el modelo es bueno"."""
        return self.n_pred_unknown / self.n_pred if self.n_pred else 0.0

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(
            precision=round(self.precision, 4),
            recall=round(self.recall, 4),
            f1=round(self.f1, 4),
            compression=round(self.compression, 3),
            unknown_rate=round(self.unknown_rate, 4),
        )
        return d


def match_boundaries(
    pred: np.ndarray, true: np.ndarray, tolerance: int = TOLERANCE_BYTES
) -> tuple[np.ndarray, np.ndarray]:
    """Empareja predichos con verdaderos, uno a uno, dentro de ±tolerance.

    Devuelve (índices de `pred` emparejados, índices de `true` emparejados).
    Ambos arrays ordenados; greedy de izquierda a derecha.
    """
    pred = np.asarray(pred, dtype=np.int64)
    true = np.asarray(true, dtype=np.int64)
    matched_p: list[int] = []
    matched_t: list[int] = []
    j = 0  # primer verdadero todavía libre
    for i, p in enumerate(pred):
        while j < true.size and true[j] < p - tolerance:
            j += 1
        if j < true.size and abs(int(true[j]) - int(p)) <= tolerance:
            matched_p.append(i)
            matched_t.append(j)
            j += 1
    return np.asarray(matched_p, dtype=np.int64), np.asarray(matched_t, dtype=np.int64)


def score_boundaries(
    pred: np.ndarray,
    gt: GroundTruth,
    *,
    tolerance: int = TOLERANCE_BYTES,
    masked: bool = True,
    label: str = "",
) -> SegmentationScore:
    """Puntúa un conjunto de cortes contra la verdad de campo.

    `masked=True`  → lectura `skeleton`: los cortes no emparejados que caen en
                     terreno NO EVALUABLE se descartan (`n_pred_unknown`).
    `masked=False` → lectura `strict`: esos cortes son falsos positivos.

    El índice 0 se excluye SIEMPRE, de predichos y de verdaderos: el
    `RoutingModule` lo fuerza a 1.0 y cualquier verdad sintáctica lo contiene,
    así que puntuarlo sería un acierto regalado para todo el mundo (incluido el
    chunker aleatorio).
    """
    pred = np.asarray(sorted(set(int(x) for x in pred if 0 < int(x) < gt.n_bytes)), dtype=np.int64)
    true = gt.boundaries

    mp, mt = match_boundaries(pred, true, tolerance=tolerance)
    tp = int(mp.size)

    unmatched_pred = np.setdiff1d(np.arange(pred.size, dtype=np.int64), mp, assume_unique=True)
    unmatched_pos = pred[unmatched_pred]

    if masked and unmatched_pos.size:
        known = gt.evaluable[unmatched_pos]
        fp = int(np.count_nonzero(known))
        unknown = int(np.count_nonzero(~known))
    else:
        fp = int(unmatched_pos.size)
        unknown = 0

    fn = int(true.size - mt.size)

    return SegmentationScore(
        n_true=int(true.size),
        n_pred=int(pred.size),
        tp=tp,
        fp=fp,
        fn=fn,
        n_pred_unknown=unknown,
        tolerance=tolerance,
        masked=masked,
        n_bytes=gt.n_bytes,
        kind=gt.kind,
        convention=gt.convention.value,
        label=label,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CONTROLES NEGATIVOS — innegociables
# ═══════════════════════════════════════════════════════════════════════════════


def fixed_size_cuts(n_bytes: int, n_cuts: int) -> np.ndarray:
    """Chunker de TAMAÑO FIJO: cortar cada N bytes.

    Se le da EXACTAMENTE la misma cantidad de cortes que produjo el modelo, de
    modo que N = n_bytes / n_cuts iguala su ratio de compresión.  Sin igualar la
    tasa, la comparación no significa nada: cortar más seguido sube el recall
    gratis.

    Si H-Net no le gana a esto, H-Net no está segmentando: está cortando.
    """
    if n_cuts <= 0 or n_bytes <= 1:
        return np.zeros(0, dtype=np.int64)
    stride = n_bytes / (n_cuts + 1)
    cuts = np.round(np.arange(1, n_cuts + 1) * stride).astype(np.int64)
    cuts = cuts[(cuts > 0) & (cuts < n_bytes)]
    return np.unique(cuts)


def random_cuts(n_bytes: int, n_cuts: int, seed: int) -> np.ndarray:
    """Chunker ALEATORIO, con la misma tasa. Es el PISO DE AZAR.

    No es un baseline "extra": con tolerancia ±1 y verdad densa (una frontera
    cada ~5 bytes en prosa), el azar acierta una fracción grande de sus cortes.
    Cualquier F1 que no supere claramente este piso es ruido con buena prensa.
    """
    if n_cuts <= 0 or n_bytes <= 2:
        return np.zeros(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    k = min(n_cuts, n_bytes - 1)
    return np.sort(rng.choice(np.arange(1, n_bytes, dtype=np.int64), size=k, replace=False))


def threshold_curve(
    boundary_prob: np.ndarray,
    gt: GroundTruth,
    thresholds: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    *,
    tolerance: int = TOLERANCE_BYTES,
    seed: int = 0,
) -> list[dict]:
    """Curva sobre el umbral, con los dos controles negativos rate-matched en cada punto.

    El F1 a un solo umbral es engañoso: el umbral fija la tasa de corte, y la
    tasa mueve precisión y recall en direcciones opuestas.  Lo que importa es la
    curva entera y, sobre todo, si en ALGÚN umbral el modelo le gana al corte
    fijo de la MISMA tasa.
    """
    rows = []
    for thr in thresholds:
        pred = np.flatnonzero(boundary_prob >= thr)
        pred = pred[pred > 0]
        n_cuts = int(pred.size)
        row = {"threshold": thr, "n_cuts": n_cuts}
        for masked in (True, False):
            tag = "skeleton" if masked else "strict"
            m = score_boundaries(pred, gt, tolerance=tolerance, masked=masked, label=f"hnet@{thr}")
            row[f"hnet_{tag}"] = m.as_dict()
            fx = score_boundaries(
                fixed_size_cuts(gt.n_bytes, n_cuts), gt, tolerance=tolerance, masked=masked,
                label=f"fixed@{thr}",
            )
            row[f"fixed_{tag}"] = fx.as_dict()
            rd = score_boundaries(
                random_cuts(gt.n_bytes, n_cuts, seed), gt, tolerance=tolerance, masked=masked,
                label=f"random@{thr}",
            )
            row[f"random_{tag}"] = rd.as_dict()
        rows.append(row)
    return rows


def aggregate(scores: list[SegmentationScore]) -> SegmentationScore:
    """Micro-promedio: suma TP/FP/FN sobre documentos.

    Micro y no macro a propósito: el macro-promedio de F1 sobre documentos le da
    el mismo peso a un payload de 200 bytes que a uno de 40 KB, y ahí es trivial
    esconder un modelo que funciona sólo en documentos cortos.  El micro dice qué
    pasa con los BYTES reales del corpus.
    """
    if not scores:
        raise ValueError("no hay scores para agregar")
    head = scores[0]
    return SegmentationScore(
        n_true=sum(s.n_true for s in scores),
        n_pred=sum(s.n_pred for s in scores),
        tp=sum(s.tp for s in scores),
        fp=sum(s.fp for s in scores),
        fn=sum(s.fn for s in scores),
        n_pred_unknown=sum(s.n_pred_unknown for s in scores),
        tolerance=head.tolerance,
        masked=head.masked,
        n_bytes=sum(s.n_bytes for s in scores),
        kind=head.kind,
        convention=head.convention,
        label=head.label,
        extra={"n_docs": len(scores)},
    )
