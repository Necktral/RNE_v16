"""Central authority policy for the experimental organism.

RNFE is an experimental system: cognition loops are active unless an ablation
explicitly disables them.  Unknown values fail closed.  This policy does not
bypass the existing action, governance, safety, or closure gates.
"""

from __future__ import annotations

import os

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def experimental_capability_enabled(name: str, *, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return bool(default)
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    return False
