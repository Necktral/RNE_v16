"""Certificación episódica y gates de promoción para RNFE."""

from .certificate_builder import CertificateBuilder
from .continuity_guard import ContinuityGuard
from .ioc_proxy import IoCProxy
from .promotion_gate import PromotionGate

__all__ = [
    "CertificateBuilder",
    "ContinuityGuard",
    "IoCProxy",
    "PromotionGate",
    "TransferAssessment",
    "TransferVerdict",
    "assess_transfer",
]


def __getattr__(name: str):
    if name in ("TransferAssessment", "TransferVerdict", "assess_transfer"):
        from .transfer_assessment import TransferAssessment, TransferVerdict, assess_transfer
        _map = {
            "TransferAssessment": TransferAssessment,
            "TransferVerdict": TransferVerdict,
            "assess_transfer": assess_transfer,
        }
        return _map[name]
    raise AttributeError(name)
