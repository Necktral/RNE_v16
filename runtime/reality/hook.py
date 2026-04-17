"""Hook de validación de realidad para integración con RuntimeRunner."""

from __future__ import annotations

from typing import Any, Dict

from runtime.reality.service import RealityValidationService


class RealityValidationHook:
    """Hook que ejecuta validación de realidad al cierre del runtime."""

    def __init__(
        self,
        *,
        storage=None,
        gate_profile: str = "ci",
        enabled: bool = True,
        run_on_shutdown: bool = False,
    ):
        """Inicializa el hook de validación.

        Args:
            storage: Storage facade opcional.
            gate_profile: Perfil del gate ('ci' o 'extended').
            enabled: Si el hook está activo.
            run_on_shutdown: Si debe ejecutar validación al cierre del runner.
        """
        self.storage = storage
        self.gate_profile = gate_profile
        self.enabled = enabled
        self.run_on_shutdown = run_on_shutdown
        self._last_result: Dict[str, Any] | None = None

    def run_validation(self, *, run_id: str | None = None) -> Dict[str, Any]:
        """Ejecuta la validación de realidad y almacena resultado.

        Args:
            run_id: ID de corrida opcional.

        Returns:
            Resultado del benchmark con passed, summary, artifacts.
        """
        if not self.enabled:
            return {"passed": True, "skipped": True, "reason": "hook_disabled"}

        service = RealityValidationService(storage=self.storage)
        result = service.run_benchmark(
            run_id=run_id,
            gate_profile=self.gate_profile,
        )
        self._last_result = result
        return result

    @property
    def last_result(self) -> Dict[str, Any] | None:
        """Retorna el último resultado de validación."""
        return self._last_result

    def on_shutdown(self, *, run_id: str | None = None) -> Dict[str, Any] | None:
        """Callback para ejecutar al cierre del runner.

        Args:
            run_id: ID de corrida del runtime.

        Returns:
            Resultado de validación si run_on_shutdown está activo, None si no.
        """
        if not self.run_on_shutdown:
            return None
        return self.run_validation(run_id=run_id)

    def passed(self) -> bool:
        """Indica si la última validación pasó."""
        if self._last_result is None:
            return True  # No se ha ejecutado aún
        return bool(self._last_result.get("passed", False))
