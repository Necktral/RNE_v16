"""Clientes para modelos externos de razonamiento experimental."""

from .config import ExternalReasonerConfig
from .llama_cpp_client import LlamaCppClient

__all__ = ["ExternalReasonerConfig", "LlamaCppClient"]
