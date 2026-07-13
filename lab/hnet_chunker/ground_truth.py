"""Verdad de campo: fronteras VERDADERAS en offsets de BYTES.

═══════════════════════════════════════════════════════════════════════════════
CONVENCIÓN DE FRONTERA — la única, y hay que fijarla antes de medir nada
═══════════════════════════════════════════════════════════════════════════════

El `RoutingModule` de H-Net (engines/hnet/modules/dc.py:86-96) calcula

    boundary_prob[t] = (1 - cos_sim(q(h[t-1]), k(h[t]))) / 2

o sea: la probabilidad en el índice `t` es la de que **un chunk EMPIECE en la
posición t**, es decir un CORTE ENTRE el byte t-1 y el byte t.  La posición 0 se
fuerza a 1.0 (`PAD_PROB`) porque todo chunk empieza en algún lado.

Por lo tanto la verdad de campo se expresa igual: un conjunto de offsets de byte
`p` tales que **una unidad sintáctica empieza en el byte p**.  "Frontera en p" ⇔
"corte antes del byte p".

El enunciado del paquete pedía "fin de `{`, fin de la clave, fin de `:`…" — que
es la convención COMPLEMENTARIA (offset después del último byte del token).  Las
dos coinciden exactamente cuando no hay blancos entre tokens, y difieren en
`len(blancos)` cuando los hay.  Se implementan LAS DOS (`Convention.TOKEN_START`
y `Convention.TOKEN_END`) y el evaluador reporta ambas, porque elegir una sola y
callarse es precisamente el tipo de default benigno que esta campaña persigue.

═══════════════════════════════════════════════════════════════════════════════
POSICIONES NO EVALUABLES — dónde NO tenemos verdad
═══════════════════════════════════════════════════════════════════════════════

Dentro del cuerpo de un string JSON (o de un string/comentario de Python) NO
EXISTE verdad sintáctica: el contenido suele ser prosa, código, un UUID o un
timestamp, y ahí una frontera podría ser perfectamente legítima sin que ninguna
gramática lo diga.  Esas posiciones se marcan `evaluable=False`.

Se reportan DOS lecturas, y hay que mirar las dos:

  - `skeleton` (enmascarada): un corte predicho DENTRO del cuerpo de un string,
    que no casó con ninguna frontera verdadera, no cuenta ni como acierto ni
    como error — se descarta y se reporta aparte (`n_pred_unknown`).  Es la
    lectura DEFENDIBLE, y también la GENEROSA: le esconde al modelo todos sus
    cortes en terreno sin verdad.
  - `strict` (sin máscara): esos cortes cuentan como falsos positivos.  Su
    precisión es una COTA INFERIOR real, porque castiga cortes que podrían ser
    correctos.

El F1 verdadero está entre las dos.  Cualquier reporte que dé un solo número
sin decir cuál de las dos lecturas es, está mintiendo por omisión.

El *recall* es idéntico en ambas lecturas: toda frontera verdadera cae, por
construcción, en terreno evaluable.  El bracket es de PRECISIÓN.
"""

from __future__ import annotations

import io
import json
import re
import tokenize
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Literal

import numpy as np

__all__ = [
    "Convention",
    "GroundTruth",
    "Token",
    "code_python_ground_truth",
    "json_ground_truth",
    "json_tokens",
    "prose_ground_truth",
]


class Convention(str, Enum):
    """Dónde cae la frontera respecto del token."""

    TOKEN_START = "token_start"  # corte ANTES del primer byte del token (semántica de H-Net)
    TOKEN_END = "token_end"  # corte DESPUÉS del último byte del token


@dataclass(frozen=True)
class Token:
    """Un token léxico, en offsets de BYTES sobre el documento original.

    `start` inclusivo, `end` exclusivo.  `opaque=True` marca los tokens cuyo
    interior NO tiene verdad de campo (cuerpos de string, comentarios).
    """

    kind: str
    start: int
    end: int
    opaque: bool = False


@dataclass(frozen=True)
class GroundTruth:
    """Fronteras verdaderas de un documento, en bytes.

    - `boundaries`: offsets ordenados, sin repetir. **NO incluye el 0.**
      El 0 se excluye a propósito: el `RoutingModule` lo fuerza a 1.0 y toda
      derivación sintáctica lo produce trivialmente, así que puntuarlo sería
      regalarle un acierto gratis a cualquier chunker, incluido el aleatorio.
    - `evaluable`: máscara booleana de largo `n_bytes`.  `False` = "acá no
      tenemos verdad" (ver el docstring del módulo).
    - `kind` / `convention`: procedencia de la derivación, para que ningún
      número quede huérfano de su definición.
    """

    boundaries: np.ndarray  # int64, ordenado, estrictamente > 0
    evaluable: np.ndarray  # bool, shape (n_bytes,)
    n_bytes: int
    kind: str
    convention: Convention

    def __post_init__(self) -> None:
        if self.evaluable.shape != (self.n_bytes,):
            raise ValueError(
                f"máscara evaluable de shape {self.evaluable.shape}, se esperaba ({self.n_bytes},)"
            )
        if self.boundaries.size:
            if self.boundaries.min() <= 0:
                raise ValueError("las fronteras deben ser > 0 (el 0 se excluye del scoring)")
            if self.boundaries.max() >= self.n_bytes:
                raise ValueError("frontera fuera del documento")
            if not np.all(np.diff(self.boundaries) > 0):
                raise ValueError("las fronteras deben venir ordenadas y sin repetir")

    @property
    def density(self) -> float:
        """Fronteras por byte. Es el *piso de azar* del problema: cuanto más densa
        la verdad, más fácil es acertar por casualidad, y más obligatorio es el
        control negativo."""
        return float(self.boundaries.size) / max(1, self.n_bytes - 1)


def _tokens_to_gt(
    tokens: list[Token],
    n_bytes: int,
    kind: str,
    convention: Convention,
) -> GroundTruth:
    if convention is Convention.TOKEN_START:
        raw = [t.start for t in tokens]
    else:
        raw = [t.end for t in tokens]

    # 0 fuera (acierto gratis) y n_bytes fuera (no es una posición del documento:
    # no hay byte n_bytes sobre el cual el modelo pueda emitir probabilidad).
    bounds = sorted({p for p in raw if 0 < p < n_bytes})

    evaluable = np.ones(n_bytes, dtype=bool)
    for t in tokens:
        if t.opaque:
            # Interior estricto del token: posiciones t.start+1 .. t.end-1.
            # (t.start es frontera verdadera; t.end lo es del token siguiente.)
            lo = min(t.start + 1, n_bytes)
            hi = min(t.end, n_bytes)
            if hi > lo:
                evaluable[lo:hi] = False
    # Una frontera verdadera nunca puede caer en terreno no evaluable.
    if bounds:
        evaluable[np.asarray(bounds, dtype=np.int64)] = True

    return GroundTruth(
        boundaries=np.asarray(bounds, dtype=np.int64),
        evaluable=evaluable,
        n_bytes=n_bytes,
        kind=kind,
        convention=convention,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# JSON — la verdad DURA.  Un parser sabe exactamente dónde están las fronteras.
# ═══════════════════════════════════════════════════════════════════════════════

_WS = b" \t\n\r"
_STRUCTURAL = b"{}[]:,"
_NUMBER_BYTES = b"-+0123456789.eE"
_LITERALS = (b"true", b"false", b"null")


class JsonLexError(ValueError):
    """El documento no es JSON válido. NO se inventa una segmentación plausible."""


def json_tokens(data: bytes) -> list[Token]:
    """Tokeniza JSON **sobre bytes** y devuelve offsets de byte.

    Por qué sobre bytes y no sobre codepoints: las fronteras que el modelo
    predice son POR BYTE (H-Net come bytes crudos, vocab_size=256).  Tokenizar
    sobre `str` y después mapear a bytes es un rodeo con una oportunidad de bug;
    tokenizar bytes directo no la tiene.

    Es seguro escanear bytes buscando `"` (0x22) y `\\` (0x5C) sin decodificar:
    en UTF-8 ningún byte de una secuencia multibyte cae en ASCII (los de
    continuación son 0x80-0xBF, los líderes 0xC2-0xF4).  Un `"` en el flujo de
    bytes ES un `"` del documento, nunca la cola de una ñ.

    Levanta `JsonLexError` si el documento no es JSON válido — deliberadamente:
    sin gramática no hay verdad, y una verdad inventada es peor que ninguna.
    """
    tokens: list[Token] = []
    i, n = 0, len(data)

    while i < n:
        b = data[i]
        if b in _WS:
            i += 1
            continue

        if b in _STRUCTURAL:
            tokens.append(Token(kind=chr(b), start=i, end=i + 1))
            i += 1
            continue

        if b == 0x22:  # '"'
            j = i + 1
            while j < n:
                c = data[j]
                if c == 0x5C:  # '\'
                    j += 2
                    continue
                if c == 0x22:
                    j += 1
                    break
                j += 1
            else:
                raise JsonLexError(f"string sin cerrar en byte {i}")
            if j > n:
                raise JsonLexError(f"escape colgando al final del string en byte {i}")
            tokens.append(Token(kind="string", start=i, end=j, opaque=True))
            i = j
            continue

        matched = False
        for lit in _LITERALS:
            if data.startswith(lit, i):
                tokens.append(Token(kind=lit.decode(), start=i, end=i + len(lit)))
                i += len(lit)
                matched = True
                break
        if matched:
            continue

        if b in _NUMBER_BYTES:
            j = i
            while j < n and data[j] in _NUMBER_BYTES:
                j += 1
            tokens.append(Token(kind="number", start=i, end=j))
            i = j
            continue

        raise JsonLexError(f"byte inesperado {b!r} en offset {i}")

    if not tokens:
        raise JsonLexError("documento vacío")
    return tokens


def verify_json_tokens(data: bytes, tokens: list[Token]) -> None:
    """Falsificador del tokenizador, no decoración.

    Invariante: quitar TODO el blanco entre tokens preserva la semántica de un
    documento JSON válido.  Entonces, si concatenar los textos de los tokens
    produce un JSON que parsea al MISMO objeto que el original, el tokenizador
    encontró todos los tokens y ninguno de más, y sus offsets son correctos.

    Esto es un test real: un tokenizador que se coma un byte, que parta un
    string escapado o que se salte una coma, falla acá.

    Es también LA validación gramatical: `json_tokens` es un LEXER, no un parser
    (`{"a": }` lexea bien y NO es JSON válido).  El round-trip delega la gramática
    al `json` de la stdlib, que sí la conoce.  Sin gramática no hay verdad.
    """
    joined = b"".join(data[t.start : t.end] for t in tokens)
    try:
        rebuilt = json.loads(joined.decode("utf-8"))
        original = json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise JsonLexError(f"no es JSON válido: {exc}") from exc
    if rebuilt != original:
        raise JsonLexError("el round-trip del tokenizador NO reconstruye el documento")


def json_ground_truth(
    data: bytes,
    convention: Convention = Convention.TOKEN_START,
    *,
    verify: bool = True,
) -> GroundTruth:
    """Fronteras sintácticas de un documento JSON, en bytes.

    Los cuerpos de string quedan NO EVALUABLES: ver el docstring del módulo.
    """
    tokens = json_tokens(data)
    if verify:
        verify_json_tokens(data, tokens)
    return _tokens_to_gt(tokens, len(data), kind="json", convention=convention)


# ═══════════════════════════════════════════════════════════════════════════════
# PROSA — una REFERENCIA, no una verdad.  Hay que decirlo cada vez.
# ═══════════════════════════════════════════════════════════════════════════════

_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _char_to_byte(text: str) -> np.ndarray:
    """Tabla de offsets: char index -> byte offset (len(text)+1 entradas)."""
    lens = np.fromiter((len(ch.encode("utf-8")) for ch in text), dtype=np.int64, count=len(text))
    out = np.zeros(len(text) + 1, dtype=np.int64)
    if len(text):
        np.cumsum(lens, out=out[1:])
    return out


def prose_ground_truth(
    text: str,
    convention: Convention = Convention.TOKEN_START,
) -> GroundTruth:
    """Fronteras de palabra, en bytes.

    ⚠ ESTO NO ES VERDAD DE CAMPO EN EL MISMO SENTIDO QUE EL JSON.  Es una
    REFERENCIA: un segmentador de palabras (`\\w+` unicode, más cada signo de
    puntuación como token propio).  El F1 contra ella mide ACUERDO CON UN
    SEGMENTADOR DE PALABRAS, no corrección.  Un chunker que cortara en morfemas
    ("re|escrib|ir") o en unidades subléxicas sería penalizado sin estar mal.  El
    paper de H-Net usa esta misma referencia; heredamos su límite, no lo
    disimulamos.

    Los blancos NO son token.  La frontera de "hola mundo" cae en el byte de la
    'm' (5), no en el del espacio (4).  El corte "antes del espacio" queda a 1
    byte — y por eso la tolerancia ±1 del scorer no es un maquillaje, es
    exactamente la ambigüedad "¿el espacio cierra la palabra anterior o abre la
    siguiente?", que ninguna convención resuelve mejor que la otra.
    """
    data = text.encode("utf-8")
    c2b = _char_to_byte(text)
    tokens = [
        Token(kind="word", start=int(c2b[m.start()]), end=int(c2b[m.end()]))
        for m in _WORD_RE.finditer(text)
    ]
    return _tokens_to_gt(tokens, len(data), kind="prose", convention=convention)


# ═══════════════════════════════════════════════════════════════════════════════
# CÓDIGO (Python) — verdad léxica del propio lexer de la stdlib
# ═══════════════════════════════════════════════════════════════════════════════

_SKIP_TOKENS = frozenset(
    {tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER, tokenize.ENCODING}
)
_OPAQUE_TOKENS = frozenset({tokenize.STRING, tokenize.COMMENT, getattr(tokenize, "FSTRING_MIDDLE", -1)})
_IDENT_SPLIT_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+|_+")

CodeConvention = Literal["lexer", "lexer+ident"]


class CodeLexError(ValueError):
    """El archivo no es Python parseable. No se evalúa: sin lexer no hay verdad."""


def _line_byte_starts(text: str) -> list[int]:
    starts, acc = [0], 0
    for line in text.splitlines(keepends=True):
        acc += len(line.encode("utf-8"))
        starts.append(acc)
    return starts


def code_python_ground_truth(
    text: str,
    convention: Convention = Convention.TOKEN_START,
    *,
    subsplit_identifiers: bool = False,
) -> GroundTruth:
    """Fronteras léxicas de un fuente Python, en bytes.

    La base (`subsplit_identifiers=False`) es VERDAD, no convención: sale del
    lexer de la stdlib (`tokenize`), el mismo que usa CPython.  Un NAME, un OP,
    un NUMBER, un STRING, un COMMENT, un NEWLINE — cada uno arranca donde el
    lexer dice que arranca.  No hay opinión.

    `subsplit_identifiers=True` agrega cortes DENTRO de los identificadores en
    los quiebres snake_case y CamelCase.  ESO SÍ ES CONVENCIÓN, no verdad: nada
    en la gramática de Python dice que `boundary_prob` sean dos unidades.  Se
    reporta por separado, nunca mezclado con la base.

    Interiores de STRING y COMMENT quedan NO EVALUABLES (son prosa embebida).
    """
    try:
        raw = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        raise CodeLexError(f"no tokeniza como Python: {exc}") from exc

    line_starts = _line_byte_starts(text)
    lines = text.splitlines(keepends=True)

    def to_byte(row: int, col: int) -> int:
        # row es 1-based; col es un índice de CARACTER dentro de la línea.
        if row - 1 >= len(lines):
            return line_starts[-1]
        return line_starts[row - 1] + len(lines[row - 1][:col].encode("utf-8"))

    tokens: list[Token] = []
    for tok in raw:
        if tok.type in _SKIP_TOKENS:
            continue
        if not tok.string:
            continue
        start = to_byte(*tok.start)
        end = to_byte(*tok.end)
        if end <= start:
            continue
        if tok.type in _OPAQUE_TOKENS:
            tokens.append(Token(kind=tokenize.tok_name[tok.type], start=start, end=end, opaque=True))
            continue
        if subsplit_identifiers and tok.type == tokenize.NAME:
            name = tok.string
            # offsets de byte de cada sub-pieza dentro del identificador
            for m in _IDENT_SPLIT_RE.finditer(name):
                s = start + len(name[: m.start()].encode("utf-8"))
                e = start + len(name[: m.end()].encode("utf-8"))
                tokens.append(Token(kind="ident_part", start=s, end=e))
            continue
        tokens.append(Token(kind=tokenize.tok_name[tok.type], start=start, end=end))

    if not tokens:
        raise CodeLexError("archivo sin tokens")

    tokens.sort(key=lambda t: (t.start, t.end))
    kind = "code_py+ident" if subsplit_identifiers else "code_py"
    return _tokens_to_gt(tokens, len(text.encode("utf-8")), kind=kind, convention=convention)


def iter_conventions() -> Iterator[Convention]:
    yield Convention.TOKEN_START
    yield Convention.TOKEN_END
