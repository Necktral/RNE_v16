"""Entrenamiento offline reproducible de N4."""

from .candidate_labeler import CandidateLabel, label_candidate_counterfactually
from .evidence_collector import EvidenceJSONLCollector
from .n4_ranking import (
    CandidateSetSample,
    N4CandidateRecord,
    N4TrainingSample,
    load_candidate_sets,
    score_n4_validity,
    train_n4_ranking,
    train_n4_ranking_v2,
)
from .n4_campaign import (
    LoadedCompositeDataset,
    load_composite_manifest,
    sample_candidate_sets_for_epoch,
    select_configuration_and_median_run,
)

__all__ = [
    "CandidateLabel",
    "EvidenceJSONLCollector",
    "CandidateSetSample",
    "N4CandidateRecord",
    "N4TrainingSample",
    "LoadedCompositeDataset",
    "label_candidate_counterfactually",
    "load_candidate_sets",
    "load_composite_manifest",
    "sample_candidate_sets_for_epoch",
    "select_configuration_and_median_run",
    "score_n4_validity",
    "train_n4_ranking",
    "train_n4_ranking_v2",
]
