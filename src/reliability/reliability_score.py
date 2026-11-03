"""
MODULE IV - Evidence-Aware Prediction & Reliability Assessment.

R(x) = w1*C(x) + w2*A(x) + w3*E(x),  w1 + w2 + w3 == 1

  C(x): classifier confidence (max softmax probability)
  A(x): prototype agreement (src/prototypes/similarity.py)
  E(x): perturbation consistency / PCS (src/stability/pcs.py)

A high C(x) alone is NOT treated as sufficient for reliability -- that is
the entire point of this module. R(x) is compared against a frozen
threshold tau (selected on validation data) to produce a binary decision.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class ReliabilityWeights:
    w1_confidence: float
    w2_prototype_agreement: float
    w3_perturbation_consistency: float

    def validate(self):
        total = self.w1_confidence + self.w2_prototype_agreement + self.w3_perturbation_consistency
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Reliability weights must sum to 1.0 (got {total}). "
                "Fix configs/reliability.yaml."
            )


def reliability_score(confidence: float, prototype_agreement: float, perturbation_consistency: float,
                       weights: ReliabilityWeights) -> float:
    weights.validate()
    return (
        weights.w1_confidence * confidence
        + weights.w2_prototype_agreement * prototype_agreement
        + weights.w3_perturbation_consistency * perturbation_consistency
    )


def reliability_decision(r_score: float, tau: float) -> str:
    return "RELIABLE" if r_score >= tau else "UNCERTAIN_EXPERT_REVIEW"


def select_tau_on_validation(r_scores: np.ndarray, correct: np.ndarray, candidate_taus=None) -> dict:
    """Selects tau on VALIDATION data by maximizing F1 of the binary
    'is this reliability decision aligned with actual correctness' task,
    i.e. we want RELIABLE predictions to actually be correct, and
    UNCERTAIN predictions to actually be the model's mistakes.

    r_scores: array of R(x) for validation samples
    correct: boolean array, whether the classifier's top-1 prediction was
        actually correct for that validation sample
    Returns {"best_tau": float, "best_f1": float, "curve": [...]}
    """
    if candidate_taus is None:
        candidate_taus = np.linspace(0.05, 0.95, 19)

    best_tau, best_f1 = candidate_taus[0], -1.0
    curve = []
    for tau in candidate_taus:
        predicted_reliable = r_scores >= tau
        tp = np.sum(predicted_reliable & correct)
        fp = np.sum(predicted_reliable & ~correct)
        fn = np.sum(~predicted_reliable & correct)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        curve.append({"tau": float(tau), "precision": float(precision), "recall": float(recall), "f1": float(f1)})
        if f1 > best_f1:
            best_tau, best_f1 = float(tau), float(f1)

    return {"best_tau": best_tau, "best_f1": best_f1, "curve": curve}


TRANSITION_TYPES = {
    (True, True): "Reliable -> Reliable",
    (False, False): "Unreliable -> Unreliable",
    (True, False): "Reliable -> Unreliable",   # potential overconfidence signal
    (False, True): "Unreliable -> Reliable",
}


def reliability_transition_analysis(before_reliable: list, after_reliable: list) -> dict:
    """before_reliable / after_reliable: parallel boolean lists (e.g. FP32
    student vs INT8 student, or teacher vs student), True == RELIABLE.

    Returns counts per transition type plus the list of sample indices that
    moved Reliable -> Unreliable (flagged as the highest-priority group to
    inspect, since it can indicate overconfidence introduced by edge
    optimization).
    """
    assert len(before_reliable) == len(after_reliable)
    counts = {label: 0 for label in set(TRANSITION_TYPES.values())}
    reliable_to_unreliable_indices = []

    for i, (b, a) in enumerate(zip(before_reliable, after_reliable)):
        label = TRANSITION_TYPES[(bool(b), bool(a))]
        counts[label] += 1
        if label == "Reliable -> Unreliable":
            reliable_to_unreliable_indices.append(i)

    return {
        "counts": counts,
        "reliable_to_unreliable_indices": reliable_to_unreliable_indices,
        "total": len(before_reliable),
    }
