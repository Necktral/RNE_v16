"""Servicios de validación de realidad operativa del organismo."""

__all__ = ["RealityValidationService"]


def __getattr__(name: str):
    if name == "RealityValidationService":
        from .service import RealityValidationService

        return RealityValidationService
    raise AttributeError(name)
