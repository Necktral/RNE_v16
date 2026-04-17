"""Language of Thought mínimo para RNFE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


class FormulaNode:
    pass


@dataclass(frozen=True)
class SymbolNode(FormulaNode):
    name: str


@dataclass(frozen=True)
class NotNode(FormulaNode):
    operand: FormulaNode


@dataclass(frozen=True)
class AndNode(FormulaNode):
    left: FormulaNode
    right: FormulaNode


@dataclass(frozen=True)
class OrNode(FormulaNode):
    left: FormulaNode
    right: FormulaNode


@dataclass(frozen=True)
class ImplicationNode(FormulaNode):
    antecedent: FormulaNode
    consequent: FormulaNode


class LOTFMin:
    """Parser/checker booleano con operadores NOT/AND/OR/->."""

    def tokenize(self, expression: str) -> List[str]:
        normalized = (
            expression.replace("(", " ( ")
            .replace(")", " ) ")
            .replace("->", " -> ")
            .strip()
        )
        tokens = [token for token in normalized.split() if token]
        if not tokens:
            raise ValueError("Expresión vacía")
        return tokens

    def parse(self, expression: str) -> FormulaNode:
        tokens = self.tokenize(expression)
        node, idx = self._parse_implication(tokens, 0)
        if idx != len(tokens):
            raise ValueError(f"Tokens restantes sin parsear: {tokens[idx:]}")
        return node

    def check(self, node: FormulaNode, type_env: Dict[str, str]) -> None:
        node_type = self._infer_type(node, type_env)
        if node_type != "bool":
            raise TypeError(f"Tipo raíz inválido: {node_type}")

    def _infer_type(self, node: FormulaNode, type_env: Dict[str, str]) -> str:
        if isinstance(node, SymbolNode):
            if node.name not in type_env:
                raise TypeError(f"Símbolo no tipado: {node.name}")
            return type_env[node.name]
        if isinstance(node, NotNode):
            typ = self._infer_type(node.operand, type_env)
            if typ != "bool":
                raise TypeError("NOT solo admite bool")
            return "bool"
        if isinstance(node, AndNode) or isinstance(node, OrNode):
            left = self._infer_type(node.left, type_env)
            right = self._infer_type(node.right, type_env)
            if left != "bool" or right != "bool":
                raise TypeError("AND/OR solo admiten bool")
            return "bool"
        if isinstance(node, ImplicationNode):
            ant = self._infer_type(node.antecedent, type_env)
            cons = self._infer_type(node.consequent, type_env)
            if ant != "bool" or cons != "bool":
                raise TypeError("-> solo admite bool")
            return "bool"
        raise TypeError(f"Nodo desconocido: {type(node)}")

    def _parse_implication(self, tokens: List[str], idx: int):
        left, idx = self._parse_or(tokens, idx)
        if idx < len(tokens) and tokens[idx] == "->":
            right, idx = self._parse_implication(tokens, idx + 1)
            return ImplicationNode(left, right), idx
        return left, idx

    def _parse_or(self, tokens: List[str], idx: int):
        node, idx = self._parse_and(tokens, idx)
        while idx < len(tokens) and tokens[idx] == "OR":
            right, idx = self._parse_and(tokens, idx + 1)
            node = OrNode(node, right)
        return node, idx

    def _parse_and(self, tokens: List[str], idx: int):
        node, idx = self._parse_not(tokens, idx)
        while idx < len(tokens) and tokens[idx] == "AND":
            right, idx = self._parse_not(tokens, idx + 1)
            node = AndNode(node, right)
        return node, idx

    def _parse_not(self, tokens: List[str], idx: int):
        if idx < len(tokens) and tokens[idx] == "NOT":
            operand, idx = self._parse_not(tokens, idx + 1)
            return NotNode(operand), idx
        return self._parse_atom(tokens, idx)

    def _parse_atom(self, tokens: List[str], idx: int):
        if idx >= len(tokens):
            raise ValueError("Fin inesperado de expresión")
        token = tokens[idx]
        if token == "(":
            node, new_idx = self._parse_implication(tokens, idx + 1)
            if new_idx >= len(tokens) or tokens[new_idx] != ")":
                raise ValueError("Paréntesis no balanceados")
            return node, new_idx + 1
        if token in {"AND", "OR", "NOT", "->", ")"}:
            raise ValueError(f"Token inesperado: {token}")
        return SymbolNode(token), idx + 1
