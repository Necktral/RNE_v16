from __future__ import annotations

from pathlib import Path

from runtime.control.msrc.vram_sampler import FixedVRAMSampler, NullVRAMSampler
from runtime.life import LifeKernel, LifeKernelConfig
from runtime.storage import StorageConfig, StorageFactory


def _storage(tmp_path: Path, name: str):
    return StorageFactory.create_facade(
        StorageConfig(
            mode="sqlite",
            sqlite_db_path=str(tmp_path / f"{name}.db"),
            postgres_dsn=None,
            artifact_root=tmp_path / f"{name}-artifacts",
            prefer_postgres_reads=False,
            strict_dual_write=False,
        )
    )


def test_override_feeds_the_same_sealed_gpu_state_to_neural_and_msrc(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("RNFE_NEURAL_MODE", "off")
    override = {
        "available": True,
        "source": "psutil",
        "sample_ts": 100.0,
        "cpu_pressure": 0.12,
        "memory_pressure": 0.23,
        "swap_pressure": 0.0,
        "thermal_pressure": 0.34,
        "gpu_available": True,
        "gpu_source": "nvidia-smi",
        "gpu_sample_ts": 101.0,
        "used_gb": 2.5,
        "total_gb": 8.0,
        "temperature_c": 67.0,
        "vram_used_gb": 2.5,
        "vram_total_gb": 8.0,
        "gpu_temperature_c": 67.0,
        "vram_pressure": 0.3125,
        "vram_headroom": 0.6875,
        "vram_fragmentation_risk": 0.17,
        "vram_opportunity_score": 0.81,
        "gpu_opportunity_score": 0.81,
        "gpu_load": 0.3125,
        "hardware_pressure": 0.34,
        "gpu_acceleration": 0.81,
    }
    storage = _storage(tmp_path, "fixed")
    kernel = LifeKernel(
        config=LifeKernelConfig(
            run_id="fixed-resource-state",
            scenarios=("thermal_homeostasis",),
            restore=False,
            enable_msrc=True,
            enable_operational_conjunction=False,
            resource_snapshot_override=override,
        ),
        storage=storage,
    )

    result = kernel.step(external_input=0.05)

    neural = result.episode_result["neural_symbiosis_trace"]["resource_state"]
    msrc = result.msrc["vram_snapshot"]
    assert isinstance(kernel._msrc_controller.vram_sampler, FixedVRAMSampler)
    assert msrc == {
        "available": neural["gpu_available"],
        "source": "nvidia-smi",
        "used_gb": neural["vram_used_gb"],
        "total_gb": neural["vram_total_gb"],
        "temperature_c": neural["gpu_temperature_c"],
        "vram_headroom": neural["vram_headroom"],
        "vram_pressure": neural["vram_pressure"],
        "vram_fragmentation_risk": neural["vram_fragmentation_risk"],
        "vram_opportunity_score": neural["vram_opportunity_score"],
        "sample_ts": 101.0,
    }
    # El sampler controlado queda sellado: MSRC no re-muestrea ni cambia tiempo.
    assert kernel._msrc_controller.vram_sampler.sample() == msrc
    storage.close()


def test_without_override_keeps_the_nominal_null_sampler_and_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("RNFE_NEURAL_MODE", "off")
    monkeypatch.delenv("RNFE_HOST_SENSING", raising=False)
    storage = _storage(tmp_path, "nominal")
    kernel = LifeKernel(
        config=LifeKernelConfig(
            run_id="nominal-resource-state",
            scenarios=("thermal_homeostasis",),
            restore=False,
            enable_msrc=True,
            enable_operational_conjunction=False,
        ),
        storage=storage,
    )

    result = kernel.step(external_input=0.05)

    assert isinstance(kernel._msrc_controller.vram_sampler, NullVRAMSampler)
    assert "vram_snapshot" not in result.msrc
    storage.close()
