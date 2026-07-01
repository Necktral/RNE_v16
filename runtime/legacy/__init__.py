"""Cuarentena física del stack legacy AEON FENIX-Δ — ver README.md.

No importar desde código vivo. Sin re-exports eager: cada módulo se importa
explícitamente (p. ej. `from runtime.legacy.module_orchestrator import
Orchestrator`) para no arrastrar torch/pydantic al importar el paquete.
"""
