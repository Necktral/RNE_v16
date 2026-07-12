"""Integración simbiótica N0-N6 con el organismo vivo."""

from .census import ComponentClass, integration_census, validate_active_census
from .adapters import CanonicalOrganAdapter, canonical_adapter_registry
from .contracts import (
    SYMBIOSIS_TRACE_SCHEMA_VERSION,
    OrganTrace,
    SymbiosisIdentity,
    SymbiosisTrace,
)
from .coordinator import SymbioticNeuralCoordinator

__all__ = [
    "ComponentClass",
    "CanonicalOrganAdapter",
    "OrganTrace",
    "SYMBIOSIS_TRACE_SCHEMA_VERSION",
    "SymbiosisIdentity",
    "SymbiosisTrace",
    "SymbioticNeuralCoordinator",
    "canonical_adapter_registry",
    "integration_census",
    "validate_active_census",
]
