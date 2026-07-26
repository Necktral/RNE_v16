"""Traducción tipada de overlays fuente a hipótesis revisables destino."""

from __future__ import annotations

import ast
from typing import Any, cast

from runtime.symbolic.schemas import sealed_sha256

from .contracts import CausalOverlay
from .structural_morphism import StructuralMorphism
from .transfer_package import TransferPackage, TransferredHypothesis


class _NameTranslator(ast.NodeTransformer):
    def __init__(
        self,
        names: dict[str, str],
        transforms: dict[str, Any] | None = None,
    ) -> None:
        self.names = names
        self.transforms = transforms or {}

    def visit_Name(self, node: ast.Name) -> ast.Name:
        return ast.copy_location(
            ast.Name(id=self.names.get(node.id, node.id), ctx=node.ctx), node
        )

    def visit_Compare(self, node: ast.Compare) -> ast.Compare:
        original_name = node.left.id if isinstance(node.left, ast.Name) else None
        translated = cast(ast.Compare, self.generic_visit(node))
        transform = self.transforms.get(original_name or "")
        if (
            transform is not None
            and len(translated.comparators) == 1
            and isinstance(translated.comparators[0], ast.Constant)
            and isinstance(translated.comparators[0].value, (int, float))
        ):
            translated.comparators[0] = ast.copy_location(
                ast.Constant(transform.apply(float(translated.comparators[0].value))),
                translated.comparators[0],
            )
        return translated


def translate_condition(
    expression: str,
    variable_map: dict[str, str],
    transforms: dict[str, Any] | None = None,
) -> str:
    """Renombra identificadores mediante AST, nunca por sustitución textual."""

    tree = ast.parse(expression, mode="eval")
    translated = _NameTranslator(variable_map, transforms).visit(tree)
    ast.fix_missing_locations(translated)
    return ast.unparse(translated.body)


class OverlayTranslator:
    def build_package(
        self,
        overlay: CausalOverlay,
        morphism: StructuralMorphism,
        *,
        evidence_refs: tuple[str, ...] = (),
        confidence: float = 0.75,
    ) -> TransferPackage:
        metrics = {
            "train_mae_before": overlay.train_mae_before,
            "train_mae_after": overlay.train_mae_after,
            "holdout_mae_before": overlay.holdout_mae_before,
            "holdout_mae_after": overlay.holdout_mae_after,
        }
        claims: dict[str, Any] = {
            "parameters": dict(overlay.parameter_updates),
            "preconditions": dict(overlay.precondition_updates),
        }
        return TransferPackage(
            package_id="",
            source_spec_sha256=morphism.source_spec_sha256,
            source_overlay_id=overlay.overlay_id,
            source_overlay_sha256=sealed_sha256(overlay.to_dict()),
            morphism_id=morphism.morphism_id,
            normalized_claims=claims,
            evidence_refs=evidence_refs,
            source_metrics=metrics,
            confidence=confidence,
        )

    def translate_overlay(
        self,
        overlay: CausalOverlay,
        morphism: StructuralMorphism,
        package: TransferPackage,
        *,
        logical_time: int,
        transfer_penalty: float = 0.10,
    ) -> list[TransferredHypothesis]:
        if package.source_overlay_id != overlay.overlay_id:
            raise ValueError("transfer_package_overlay_mismatch")
        if package.morphism_id != morphism.morphism_id:
            raise ValueError("transfer_package_morphism_mismatch")
        confidence = package.confidence
        output: list[TransferredHypothesis] = []
        for source_id, value in sorted(overlay.parameter_updates.items()):
            target_id = morphism.parameter_map.get(source_id)
            if target_id is None:
                continue
            transform = morphism.scale_transforms.get(source_id)
            translated = transform.apply(float(value)) if transform else float(value)
            output.append(
                TransferredHypothesis(
                    hypothesis_id="",
                    package_id=package.package_id,
                    morphism_id=morphism.morphism_id,
                    kind="parameter",
                    target_id=target_id,
                    expression=None,
                    proposed_value=translated,
                    transfer_confidence=confidence,
                    transfer_penalty=transfer_penalty,
                    source_evidence_refs=package.evidence_refs,
                    logical_time=logical_time,
                )
            )
        for source_id, expression in sorted(overlay.precondition_updates.items()):
            target_id = morphism.effect_map.get(source_id)
            if target_id is None or expression is None:
                continue
            output.append(
                TransferredHypothesis(
                    hypothesis_id="",
                    package_id=package.package_id,
                    morphism_id=morphism.morphism_id,
                    kind="precondition",
                    target_id=target_id,
                    expression=translate_condition(
                        expression,
                        dict(morphism.variable_map),
                        dict(morphism.scale_transforms),
                    ),
                    proposed_value=None,
                    transfer_confidence=confidence,
                    transfer_penalty=transfer_penalty,
                    source_evidence_refs=package.evidence_refs,
                    logical_time=logical_time,
                )
            )
        return sorted(output, key=lambda item: item.hypothesis_id)
