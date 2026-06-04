"""Catálogo formal de escalas para MSRC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .contracts import ScaleSpec


@dataclass(frozen=True)
class ScaleCatalog:
    """Catálogo extensible de escalas representacionales."""

    specs: Dict[str, ScaleSpec]

    @classmethod
    def default(cls) -> "ScaleCatalog":
        specs = {
            "1x1": ScaleSpec(
                scale_id="1x1",
                grid_shape=(1, 1),
                resolution_rank=1,
                scenario_name="grid_thermal_1x1",
                is_executable=True,
                supports_local_structure=False,
                supports_local_intervention=False,
                supports_spatial_memory=False,
                expected_time_cost=1.0,
                expected_artifact_cost=1.0,
                expected_information_gain_prior=0.35,
                memory_compatibility_profile={
                    "allow_cross_scale_summary": True,
                    "allow_dense_cell_transfer": False,
                    "preferred_memory_scale": "macro",
                },
            ),
            "2x2": ScaleSpec(
                scale_id="2x2",
                grid_shape=(2, 2),
                resolution_rank=2,
                scenario_name="grid_thermal_2x2",
                is_executable=False,
                supports_local_structure=True,
                supports_local_intervention=True,
                supports_spatial_memory=True,
                expected_time_cost=1.2,
                expected_artifact_cost=1.3,
                expected_information_gain_prior=0.45,
                memory_compatibility_profile={
                    "allow_cross_scale_summary": True,
                    "allow_dense_cell_transfer": False,
                    "preferred_memory_scale": "meso",
                },
            ),
            "3x3": ScaleSpec(
                scale_id="3x3",
                grid_shape=(3, 3),
                resolution_rank=3,
                scenario_name="grid_thermal_3x3",
                is_executable=False,
                supports_local_structure=True,
                supports_local_intervention=True,
                supports_spatial_memory=True,
                expected_time_cost=1.5,
                expected_artifact_cost=1.7,
                expected_information_gain_prior=0.58,
                memory_compatibility_profile={
                    "allow_cross_scale_summary": True,
                    "allow_dense_cell_transfer": False,
                    "preferred_memory_scale": "meso",
                },
            ),
            "5x5": ScaleSpec(
                scale_id="5x5",
                grid_shape=(5, 5),
                resolution_rank=5,
                scenario_name="grid_thermal_5x5_uniform",
                is_executable=True,
                supports_local_structure=True,
                supports_local_intervention=True,
                supports_spatial_memory=True,
                expected_time_cost=2.2,
                expected_artifact_cost=2.2,
                expected_information_gain_prior=0.72,
                memory_compatibility_profile={
                    "allow_cross_scale_summary": True,
                    "allow_dense_cell_transfer": False,
                    "preferred_memory_scale": "meso",
                },
            ),
            "10x10": ScaleSpec(
                scale_id="10x10",
                grid_shape=(10, 10),
                resolution_rank=10,
                scenario_name="grid_thermal_10x10",
                is_executable=False,
                supports_local_structure=True,
                supports_local_intervention=True,
                supports_spatial_memory=True,
                expected_time_cost=4.5,
                expected_artifact_cost=5.5,
                expected_information_gain_prior=0.83,
                memory_compatibility_profile={
                    "allow_cross_scale_summary": True,
                    "allow_dense_cell_transfer": False,
                    "preferred_memory_scale": "micro",
                },
            ),
            "30x30": ScaleSpec(
                scale_id="30x30",
                grid_shape=(30, 30),
                resolution_rank=30,
                scenario_name="grid_thermal_30x30",
                is_executable=False,
                supports_local_structure=True,
                supports_local_intervention=True,
                supports_spatial_memory=True,
                expected_time_cost=12.0,
                expected_artifact_cost=16.0,
                expected_information_gain_prior=0.92,
                memory_compatibility_profile={
                    "allow_cross_scale_summary": True,
                    "allow_dense_cell_transfer": False,
                    "preferred_memory_scale": "micro",
                },
            ),
        }
        return cls(specs=specs)

    def list_all(self) -> List[ScaleSpec]:
        return sorted(self.specs.values(), key=lambda item: item.resolution_rank)

    def list_ids(self) -> List[str]:
        return [item.scale_id for item in self.list_all()]

    def executable_scales(self) -> List[ScaleSpec]:
        return [item for item in self.list_all() if item.is_executable]

    def get(self, scale_id: str) -> ScaleSpec:
        if scale_id not in self.specs:
            available = ", ".join(self.list_ids())
            raise KeyError(f"Scale '{scale_id}' no existe. Disponibles: {available}")
        return self.specs[scale_id]

    def nearest_executable(self, scale_id: str) -> ScaleSpec:
        requested = self.get(scale_id)
        executable = self.executable_scales()
        return min(
            executable,
            key=lambda item: abs(item.resolution_rank - requested.resolution_rank),
        )

    def candidates_at_or_above(self, minimum_rank: int, *, executable_only: bool = False) -> List[ScaleSpec]:
        items = [item for item in self.list_all() if item.resolution_rank >= minimum_rank]
        if executable_only:
            items = [item for item in items if item.is_executable]
        return items

    def scenario_params_for(
        self,
        scale_id: str,
        *,
        topology: str = "uniform",
        base_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Construye parámetros de escenario por escala.

        Nota: v1 ejecuta solo 1x1 y 5x5 con GridThermalScenario.
        """
        spec = self.get(scale_id)
        params = dict(base_params or {})
        rows, cols = spec.grid_shape
        params["grid_size"] = max(rows, cols)
        if params["grid_size"] > 1:
            params.setdefault("topology", topology)
        else:
            params.pop("topology", None)
        return params

    def has_scale(self, scale_id: str) -> bool:
        return scale_id in self.specs

    def is_executable(self, scale_id: str) -> bool:
        return self.get(scale_id).is_executable

    def as_dict(self) -> Dict[str, Dict[str, Any]]:
        return {scale_id: spec.to_dict() for scale_id, spec in self.specs.items()}
