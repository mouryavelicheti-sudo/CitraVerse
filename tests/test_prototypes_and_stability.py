import numpy as np

from src.prototypes.prototype_builder import build_prototypes
from src.prototypes.similarity import compute_dess, cosine_similarity, prototype_agreement
from src.stability.pcs import evidence_stability_loss, perturbation_consistency_score


def test_build_prototypes_groups_by_class_and_domain():
    z_vectors = [np.ones(4), np.ones(4) * 2, np.zeros(4)]
    classes = ["healthy", "healthy", "canker"]
    domains = ["d1", "d2", "d1"]

    prototypes = build_prototypes(z_vectors, classes, domains)

    assert ("healthy", "d1") in prototypes
    assert ("healthy", "d2") in prototypes
    assert ("canker", "d1") in prototypes
    assert np.allclose(prototypes[("healthy", "d1")]["prototype"], np.ones(4))
    assert prototypes[("healthy", "d1")]["n_samples"] == 1


def test_dess_undefined_for_single_domain_class():
    z_vectors = [np.ones(4), np.ones(4) * 0.9, np.zeros(4)]
    classes = ["healthy", "healthy", "canker"]
    domains = ["d1", "d2", "d1"]  # "canker" only ever appears in d1
    prototypes = build_prototypes(z_vectors, classes, domains)

    dess = compute_dess(prototypes)
    assert dess["healthy"] is not None
    assert dess["canker"] is None  # cannot measure cross-dataset stability from one domain


def test_dess_high_for_identical_prototypes_across_domains():
    z_vectors = [np.array([1.0, 0.0]), np.array([1.0, 0.0])]
    classes = ["healthy", "healthy"]
    domains = ["d1", "d2"]
    prototypes = build_prototypes(z_vectors, classes, domains)
    dess = compute_dess(prototypes, metric="cosine")
    assert abs(dess["healthy"] - 1.0) < 1e-6


def test_prototype_agreement_higher_for_matching_evidence():
    z_vectors = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    classes = ["healthy", "canker"]
    domains = ["d1", "d1"]
    prototypes = build_prototypes(z_vectors, classes, domains)

    agreement_match = prototype_agreement(np.array([1.0, 0.0]), "healthy", prototypes)
    agreement_mismatch = prototype_agreement(np.array([1.0, 0.0]), "canker", prototypes)
    assert agreement_match > agreement_mismatch


def test_evidence_stability_loss_zero_for_identical_signatures():
    z = np.array([1.0, 2.0, 3.0])
    assert evidence_stability_loss(z, z) == 0.0


def test_pcs_decreases_as_perturbation_drift_increases():
    z_orig = np.array([1.0, 0.0, 0.0])
    small_drift = {"p1": np.array([1.01, 0.0, 0.0])}
    large_drift = {"p1": np.array([5.0, 5.0, 5.0])}

    result_small = perturbation_consistency_score(z_orig, small_drift)
    result_large = perturbation_consistency_score(z_orig, large_drift)

    assert result_small["PCS"] > result_large["PCS"]
    assert 0.0 < result_small["PCS"] <= 1.0
