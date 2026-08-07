from __future__ import annotations
import hashlib
import json
import math

import pytest

from runtime.experiments.cognitive_gain.canonical import (
    canonical_json,
    canonical_json_bytes,
    canonical_sha256,
    canonicalize,
)


def test_canonical_domain_and_reference_parity():
    value = {"z": [None, True, 4, 1.25], "a": {"text": "niño"}}
    expected = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    assert canonical_json(value) == expected
    assert canonical_json_bytes(value) == expected.encode("utf-8")


def test_mapping_order_does_not_change_json_or_hash():
    left = {"b": 2, "a": {"y": 1, "x": 0}}
    right = {"a": {"x": 0, "y": 1}, "b": 2}
    assert canonical_json(left) == canonical_json(right)
    assert canonical_sha256(left) == canonical_sha256(right)


def test_semantic_change_changes_hash():
    assert canonical_sha256({"value": 1}) != canonical_sha256({"value": 2})


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_numbers_are_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        canonicalize(value)


def test_non_string_mapping_key_is_rejected():
    with pytest.raises(ValueError, match="keys"):
        canonicalize({1: "value"})


def test_unknown_type_is_rejected():
    with pytest.raises(ValueError, match="unsupported"):
        canonicalize({"bad": object()})


def test_utf8_hash_is_deterministic_and_not_ascii_escaped():
    value = {"señal": "causal 🧠"}
    encoded = canonical_json_bytes(value)
    assert b"\\u" not in encoded
    assert canonical_sha256(value) == hashlib.sha256(encoded).hexdigest()


def test_canonicalize_detaches_mutable_inputs():
    source = {"items": [{"value": 1}]}
    detached = canonicalize(source)
    source["items"][0]["value"] = 9
    assert detached == {"items": [{"value": 1}]}
