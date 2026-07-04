"""Árbol y serialización de expresiones EML."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal


Op = Literal[
    "const",
    "var",
    "add",
    "sub",
    "mul",
    "div",
    "pow2",
    "log1p",
    "exp",
]


@dataclass(frozen=True, slots=True)
class ExprNode:
    op: Op
    value: float | None = None
    name: str | None = None
    left: "ExprNode | None" = None
    right: "ExprNode | None" = None

    def depth(self) -> int:
        if self.op in {"const", "var"}:
            return 1
        left_d = self.left.depth() if self.left else 0
        right_d = self.right.depth() if self.right else 0
        return 1 + max(left_d, right_d)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"op": self.op}
        if self.value is not None:
            payload["value"] = self.value
        if self.name is not None:
            payload["name"] = self.name
        if self.left is not None:
            payload["left"] = self.left.to_dict()
        if self.right is not None:
            payload["right"] = self.right.to_dict()
        return payload

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "ExprNode":
        left = ExprNode.from_dict(payload["left"]) if "left" in payload else None
        right = ExprNode.from_dict(payload["right"]) if "right" in payload else None
        return ExprNode(
            op=payload["op"],
            value=payload.get("value"),
            name=payload.get("name"),
            left=left,
            right=right,
        )

