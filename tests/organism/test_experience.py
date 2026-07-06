"""Tests del sistema de Experiencia: recordar golpes, sabiduría ∝ daño, no repetir."""

from __future__ import annotations

import os
import tempfile

import pytest

from runtime.organism.experience import (
    ExperienceRecord,
    ExperienceStore,
    build_experience,
    compute_severity,
    experience_enabled,
    situation_key,
)


def _store():
    os.environ["RNFE_STORAGE_MODE"] = "sqlite"
    os.environ["AEON_EVENT_DB"] = os.path.join(tempfile.mkdtemp(), "exp.db")
    os.environ["RNFE_ARTIFACT_ROOT"] = tempfile.mkdtemp()
    from runtime.storage import get_storage, reset_storage

    reset_storage()
    return ExperienceStore(storage=get_storage())


def test_flag_off_by_default(monkeypatch):
    monkeypatch.delenv("RNFE_EXPERIENCE", raising=False)
    assert experience_enabled() is False
    monkeypatch.setenv("RNFE_EXPERIENCE", "1")
    assert experience_enabled() is True


def test_severity_proportional_to_damage():
    # Cuanto más profundo el golpe, mayor la severidad (la sabiduría será ∝ a esto).
    deep = compute_severity(viability_margin=0.12, ioc=0.15, risk=0.72, action="quarantine", certified=False)
    mild = compute_severity(viability_margin=0.6, ioc=0.5, risk=0.4, action="act", certified=True)
    healthy = compute_severity(viability_margin=0.9, ioc=0.88, risk=0.14, action="act", certified=True)
    assert deep > mild > healthy
    assert deep >= 0.85  # cuarentena = herida grave
    assert healthy < 0.30  # buen episodio


def test_situation_key_stable_and_discriminates():
    a = situation_key(scenario="thermal", regime="alarm", main_variable="temp")
    b = situation_key(scenario="thermal", regime="alarm", main_variable="temp")
    c = situation_key(scenario="thermal", regime="calm", main_variable="temp")
    assert a == b
    assert a != c


def test_record_and_recall_wound():
    store = _store()
    sig = situation_key(scenario="thermal", regime="alarm", main_variable="temp")
    exp = build_experience(
        organism_id="aeon-test", run_id="life-1", episode_id="e1",
        scenario="thermal", regime="alarm", main_variable="temp", causal_status="",
        intervention="deactivate_cooling", viability_margin=0.12, ioc=0.15, risk=0.72,
        reward=-0.5, action="act", certified=False, closure_passed=False, viability_delta=-0.4,
    )
    assert exp.wound is True
    assert exp.severity >= 0.5
    store.record(exp)
    recalled = store.recall(organism_id="aeon-test", situation=sig)
    assert len(recalled) == 1
    assert recalled[0]["intervention"] == "deactivate_cooling"


def test_wisdom_avoids_what_hurt_prefers_what_relieved():
    store = _store()
    sig = situation_key(scenario="thermal", regime="alarm", main_variable="temp")
    # Un golpe profundo con 'deactivate_cooling'
    store.record(build_experience(
        organism_id="aeon-w", run_id="l1", episode_id="e1", scenario="thermal", regime="alarm",
        main_variable="temp", causal_status="", intervention="deactivate_cooling",
        viability_margin=0.12, ioc=0.15, risk=0.72, reward=-0.5, action="act",
        certified=False, closure_passed=False, viability_delta=-0.4))
    # Un buen episodio con 'activate_cooling'
    store.record(build_experience(
        organism_id="aeon-w", run_id="l1", episode_id="e2", scenario="thermal", regime="alarm",
        main_variable="temp", causal_status="", intervention="activate_cooling",
        viability_margin=0.9, ioc=0.88, risk=0.14, reward=0.5, action="act",
        certified=True, closure_passed=True, viability_delta=0.05))
    w = store.wisdom(organism_id="aeon-w", situation=sig)
    assert w.avoid == "deactivate_cooling"
    assert w.prefer == "activate_cooling"
    assert w.scar["deactivate_cooling"] > w.scar.get("activate_cooling", 0.0)
    assert w.max_severity >= 0.5


def test_cross_life_recall_same_organism_id():
    """Vidas distintas (run_id distinto) comparten experiencia bajo el mismo organism_id."""
    store = _store()
    sig = situation_key(scenario="thermal", regime="alarm", main_variable="temp")
    store.record(build_experience(
        organism_id="aeon-x", run_id="vida-1", episode_id="e1", scenario="thermal", regime="alarm",
        main_variable="temp", causal_status="", intervention="deactivate_cooling",
        viability_margin=0.12, ioc=0.15, risk=0.72, reward=-0.5, action="act",
        certified=False, closure_passed=False, viability_delta=-0.4))
    # "Nueva vida": otro run_id, mismo organism_id -> debe recordar el golpe de la vida anterior
    recalled = store.recall(organism_id="aeon-x", situation=sig)
    assert len(recalled) == 1
    assert recalled[0]["intervention"] == "deactivate_cooling"
