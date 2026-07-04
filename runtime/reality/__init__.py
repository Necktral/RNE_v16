"""Servicios de validación de realidad operativa del organismo."""

__all__ = [
    "RealityValidationService",
    "RealityValidationHook",
    "run_reality_validation",
    "MSRCPolicyBenchmarkRunner",
]


def __getattr__(name: str):
    if name == "RealityValidationService":
        from .service import RealityValidationService

        return RealityValidationService
    if name == "RealityValidationHook":
        from .hook import RealityValidationHook

        return RealityValidationHook
    if name == "run_reality_validation":
        from .cli import run_reality_validation

        return run_reality_validation
    if name == "MSRCPolicyBenchmarkRunner":
        from .msrc_policy_benchmark import MSRCPolicyBenchmarkRunner

        return MSRCPolicyBenchmarkRunner
    raise AttributeError(name)
