"""Strict, local canonical JSON utilities for ASCG v1.1.
This module depends only on the standard library and has no import-time I/O.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any


def canonicalize(value: Any) -> Any:
    """Return a detached value in the explicit canonical JSON domain."""

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical_float_must_be_finite")
        return 0.0 if value == 0.0 else value
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("canonical_mapping_keys_must_be_strings")
        return {key: canonicalize(item) for key, item in value.items()}
    raise ValueError(f"canonical_type_unsupported:{type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a canonical JSON value deterministically as UTF-8 bytes."""

    normalized = canonicalize(value)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
