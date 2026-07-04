"""Bridge exocortex <-> runtime exclusivamente por contratos."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


CONTRACTS_DIR = Path(__file__).resolve().parents[2] / "contracts"


def load_contract(contract_name: str) -> Dict[str, Any]:
    path = CONTRACTS_DIR / f"{contract_name}.schema.json"
    return json.loads(path.read_text(encoding="utf-8"))

