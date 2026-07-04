"""Traduccion de un subconjunto booleano de LOT-F hacia Z3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set

from z3 import And, Bool, BoolRef, Implies, Not, Or

from runtime.lotf import AndNode, FormulaNode, ImplicationNode, NotNode, OrNode, SymbolNode


class UnsupportedLOTFNodeError(ValueError):
    """Error cuando el AST contiene un nodo fuera del subconjunto soportado."""


@dataclass(frozen=True)
class TranslationResult:
    expression: BoolRef
    symbol_table: Dict[str, BoolRef]


def collect_symbols(node: FormulaNode) -> List[str]:
    """Retorna los simbolos del AST en orden reproducible."""

    found: Set[str] = set()

    def _visit(current: FormulaNode) -> None:
        if isinstance(current, SymbolNode):
            found.add(current.name)
            return
        if isinstance(current, NotNode):
            _visit(current.operand)
            return
        if isinstance(current, (AndNode, OrNode)):
            _visit(current.left)
            _visit(current.right)
            return
        if isinstance(current, ImplicationNode):
            _visit(current.antecedent)
            _visit(current.consequent)
            return
        raise UnsupportedLOTFNodeError(
            f"Nodo LOT-F no soportado por DED v1: {type(current).__name__}"
        )

    _visit(node)
    return sorted(found)


def build_type_env(symbols: Iterable[str]) -> Dict[str, str]:
    """Construye el contexto de tipos booleano usado por LOTFMin.check()."""

    return {symbol: "bool" for symbol in symbols}


def translate_lotf(node: FormulaNode) -> TranslationResult:
    """Traduce un AST LOT-F booleano a una expresion Z3."""

    symbol_names = collect_symbols(node)
    symbol_table = {name: Bool(name) for name in symbol_names}

    def _translate(current: FormulaNode) -> BoolRef:
        if isinstance(current, SymbolNode):
            return symbol_table[current.name]
        if isinstance(current, NotNode):
            return Not(_translate(current.operand))
        if isinstance(current, AndNode):
            return And(_translate(current.left), _translate(current.right))
        if isinstance(current, OrNode):
            return Or(_translate(current.left), _translate(current.right))
        if isinstance(current, ImplicationNode):
            return Implies(_translate(current.antecedent), _translate(current.consequent))
        raise UnsupportedLOTFNodeError(
            f"Nodo LOT-F no soportado por DED v1: {type(current).__name__}"
        )

    return TranslationResult(expression=_translate(node), symbol_table=symbol_table)
