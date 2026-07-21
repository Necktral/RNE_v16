"""Tests del sensado de recursos de host (Bloque A del router GPU)."""

from __future__ import annotations

from runtime.control.msrc.host_sampler import (
    HostResourceSampler,
    build_resource_snapshot,
    host_sensing_enabled,
)
from runtime.control.msrc.vram_sampler import FixedVRAMSampler


def test_host_sensing_disabled_by_default(monkeypatch):
    monkeypatch.delenv("RNFE_HOST_SENSING", raising=False)
    assert host_sensing_enabled() is False
    for value in ("1", "true", "yes", "on"):
        monkeypatch.setenv("RNFE_HOST_SENSING", value)
        assert host_sensing_enabled() is True


def test_sampler_fields_clamped_and_present():
    snap = HostResourceSampler(ttl_seconds=0.0).sample()
    assert snap["available"] is True
    for key in ("cpu_pressure", "memory_pressure", "swap_pressure", "thermal_pressure"):
        assert 0.0 <= snap[key] <= 1.0
    # Campos de GPU neutros en el sampler de host.
    assert snap["gpu_available"] is False


def test_sampler_falls_back_without_psutil(monkeypatch):
    sampler = HostResourceSampler(ttl_seconds=0.0)
    sampler._psutil = None  # fuerza el camino stdlib /proc
    snap = sampler.sample()
    assert snap["source"] == "proc"
    assert 0.0 <= snap["memory_pressure"] <= 1.0


def test_ttl_cache_returns_same_snapshot():
    sampler = HostResourceSampler(ttl_seconds=60.0)
    a = sampler.sample()
    b = sampler.sample()
    assert a["sample_ts"] == b["sample_ts"]


class _FakeVram:
    def __init__(
        self,
        *,
        available,
        vram_pressure=0.0,
        vram_headroom=1.0,
        opp=1.0,
        fragmentation=0.1,
        used_gb=1.0,
        total_gb=8.0,
        temperature_c=65.0,
    ):
        self._d = {
            "available": available,
            "source": "fixture-gpu",
            "sample_ts": 123.0,
            "used_gb": used_gb,
            "total_gb": total_gb,
            "temperature_c": temperature_c,
            "vram_pressure": vram_pressure,
            "vram_headroom": vram_headroom,
            "vram_fragmentation_risk": fragmentation,
            "vram_opportunity_score": opp,
        }

    def sample(self):
        return dict(self._d)


def test_build_snapshot_merges_gpu_when_available():
    snap = build_resource_snapshot(
        host_sampler=HostResourceSampler(ttl_seconds=0.0),
        vram_sampler=_FakeVram(available=True, vram_pressure=0.1, vram_headroom=0.9, opp=0.9),
    )
    assert snap["gpu_available"] is True
    assert snap["vram_pressure"] == 0.1
    assert snap["gpu_acceleration"] > 0.0
    assert "hardware_pressure" in snap


def test_build_snapshot_preserves_full_gpu_state_for_neural_and_msrc():
    snap = build_resource_snapshot(
        host_sampler=HostResourceSampler(ttl_seconds=0.0),
        vram_sampler=_FakeVram(
            available=True,
            vram_pressure=0.25,
            vram_headroom=0.75,
            opp=0.82,
            fragmentation=0.13,
            used_gb=2.0,
            total_gb=8.0,
            temperature_c=66.0,
        ),
    )

    assert snap["used_gb"] == snap["vram_used_gb"] == 2.0
    assert snap["total_gb"] == snap["vram_total_gb"] == 8.0
    assert snap["temperature_c"] == snap["gpu_temperature_c"] == 66.0
    assert snap["vram_fragmentation_risk"] == 0.13
    assert snap["vram_opportunity_score"] == snap["gpu_opportunity_score"] == 0.82
    assert snap["gpu_source"] == "fixture-gpu"
    assert snap["gpu_sample_ts"] == 123.0

    fixed = FixedVRAMSampler(snap)
    assert fixed.sample() == fixed.sample()
    assert fixed.sample() == {
        "available": True,
        "source": "fixture-gpu",
        "used_gb": 2.0,
        "total_gb": 8.0,
        "temperature_c": 66.0,
        "vram_headroom": 0.75,
        "vram_pressure": 0.25,
        "vram_fragmentation_risk": 0.13,
        "vram_opportunity_score": 0.82,
        "sample_ts": 123.0,
    }


def test_fixed_vram_sampler_does_not_confuse_host_with_gpu_availability():
    fixed = FixedVRAMSampler(
        {
            "available": True,
            "source": "psutil",
            "gpu_available": False,
            "sample_ts": 456.0,
            "vram_pressure": 0.0,
        }
    )

    assert fixed.sample()["available"] is False


def test_build_snapshot_ignores_gpu_when_unavailable():
    snap = build_resource_snapshot(
        host_sampler=HostResourceSampler(ttl_seconds=0.0),
        vram_sampler=_FakeVram(available=False),
    )
    assert snap["gpu_available"] is False
    assert snap["gpu_acceleration"] == 0.0


def test_high_vram_pressure_caps_gpu_acceleration():
    snap = build_resource_snapshot(
        host_sampler=HostResourceSampler(ttl_seconds=0.0),
        vram_sampler=_FakeVram(available=True, vram_pressure=0.95, vram_headroom=0.05, opp=0.9),
    )
    assert snap["gpu_acceleration"] <= 0.25


def test_hardware_pressure_is_max_of_pressures():
    fake = _FakeVram(available=True, vram_pressure=0.99, vram_headroom=0.01, opp=0.0)
    snap = build_resource_snapshot(
        host_sampler=HostResourceSampler(ttl_seconds=0.0),
        vram_sampler=fake,
    )
    assert snap["hardware_pressure"] >= 0.99
