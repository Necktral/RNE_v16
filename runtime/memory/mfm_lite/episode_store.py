"""Persistencia multiescala de episodios certificados."""

from __future__ import annotations

from typing import Any, Dict

from runtime.storage import StorageFacade


class EpisodeMemoryStore:
    def __init__(self, *, storage: StorageFacade):
        self.storage = storage

    def write_micro(
        self,
        *,
        run_id: str,
        episode_id: str,
        certificate_id: str,
        ioc_proxy: float,
        structure: Dict[str, Any],
    ):
        return self.storage.write_memory_record(
            run_id=run_id,
            episode_id=episode_id,
            scale="micro",
            structure_json=structure,
            certificate_id=certificate_id,
            ioc_proxy=ioc_proxy,
            ttl_seconds=7 * 24 * 3600,
            no_interference=True,
            support_count=1,
            metadata={"origin": "episode_store.micro"},
        )

    def write_meso(
        self,
        *,
        run_id: str,
        episode_id: str,
        certificate_id: str,
        ioc_proxy: float,
        structure: Dict[str, Any],
    ):
        return self.storage.write_memory_record(
            run_id=run_id,
            episode_id=episode_id,
            scale="meso",
            structure_json=structure,
            certificate_id=certificate_id,
            ioc_proxy=ioc_proxy,
            ttl_seconds=30 * 24 * 3600,
            no_interference=True,
            support_count=1,
            metadata={"origin": "episode_store.meso"},
        )

    def write_macro(
        self,
        *,
        run_id: str,
        episode_id: str,
        certificate_id: str,
        ioc_proxy: float,
        support_count: int,
        structure: Dict[str, Any],
    ):
        return self.storage.write_memory_record(
            run_id=run_id,
            episode_id=episode_id,
            scale="macro",
            structure_json=structure,
            certificate_id=certificate_id,
            ioc_proxy=ioc_proxy,
            ttl_seconds=None,
            no_interference=True,
            support_count=support_count,
            metadata={"origin": "episode_store.macro"},
        )
