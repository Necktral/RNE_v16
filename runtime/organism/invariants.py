"""Invariantes constitucionales — definiciones de referencia.

Re-exporta los invariantes y utilidades de constitution.py para
acceso directo cuando solo se necesitan las definiciones.
"""

from .constitution import (
    HARD_INVARIANTS,
    SOFT_INVARIANTS,
    MUTABLE_COMPONENTS,
    IMMUTABLE_COMPONENTS,
    ConstitutionalInvariant,
    InvariantViolation,
    ConstitutionalValidation,
    InvariantSeverity,
)

__all__ = [
    "HARD_INVARIANTS",
    "SOFT_INVARIANTS",
    "MUTABLE_COMPONENTS",
    "IMMUTABLE_COMPONENTS",
    "ConstitutionalInvariant",
    "InvariantViolation",
    "ConstitutionalValidation",
    "InvariantSeverity",
]
