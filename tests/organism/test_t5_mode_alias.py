from __future__ import annotations

from runtime.organism.t5_mode import get_t5_mode


def test_t5_mode_reads_t5_first(monkeypatch) -> None:
    monkeypatch.setenv("RNFE_T4_MODE", "off")
    monkeypatch.setenv("RNFE_T5_MODE", "on")
    assert get_t5_mode() == "on"


def test_t5_mode_fallbacks_to_t4(monkeypatch) -> None:
    monkeypatch.delenv("RNFE_T5_MODE", raising=False)
    monkeypatch.setenv("RNFE_T4_MODE", "shadow")
    assert get_t5_mode() == "shadow"

