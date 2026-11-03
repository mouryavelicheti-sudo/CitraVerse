import numpy as np
import pytest

from src.reliability.reliability_score import (
    ReliabilityWeights,
    reliability_decision,
    reliability_score,
    reliability_transition_analysis,
    select_tau_on_validation,
)


def test_weights_must_sum_to_one():
    bad_weights = ReliabilityWeights(0.5, 0.5, 0.5)
    with pytest.raises(ValueError):
        bad_weights.validate()


def test_reliability_score_is_weighted_average():
    weights = ReliabilityWeights(0.4, 0.35, 0.25)
    r = reliability_score(confidence=1.0, prototype_agreement=1.0, perturbation_consistency=1.0, weights=weights)
    assert abs(r - 1.0) < 1e-9

    r_zero = reliability_score(confidence=0.0, prototype_agreement=0.0, perturbation_consistency=0.0, weights=weights)
    assert abs(r_zero - 0.0) < 1e-9


def test_reliability_decision_threshold():
    assert reliability_decision(0.7, tau=0.6) == "RELIABLE"
    assert reliability_decision(0.5, tau=0.6) == "UNCERTAIN_EXPERT_REVIEW"


def test_select_tau_prefers_threshold_separating_correct_from_incorrect():
    # Correct predictions cluster at high R(x), incorrect ones at low R(x).
    r_scores = np.array([0.9, 0.85, 0.8, 0.2, 0.15, 0.1])
    correct = np.array([True, True, True, False, False, False])
    result = select_tau_on_validation(r_scores, correct)
    assert 0.15 < result["best_tau"] < 0.85
    assert result["best_f1"] > 0.9


def test_reliability_transition_analysis_counts_and_flags_overconfidence():
    before = [True, True, False, False]
    after = [True, False, True, False]
    result = reliability_transition_analysis(before, after)

    assert result["counts"]["Reliable -> Reliable"] == 1
    assert result["counts"]["Reliable -> Unreliable"] == 1
    assert result["counts"]["Unreliable -> Reliable"] == 1
    assert result["counts"]["Unreliable -> Unreliable"] == 1
    assert result["reliable_to_unreliable_indices"] == [1]
