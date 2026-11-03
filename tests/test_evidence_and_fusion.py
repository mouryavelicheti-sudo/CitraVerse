import numpy as np

from src.evidence.color_evidence import color_evidence_dim, extract_color_evidence
from src.evidence.fusion import EvidenceComponents, fit_fusion_stats, fuse
from src.evidence.shape_evidence import extract_shape_evidence, shape_evidence_dim
from src.evidence.texture_evidence import extract_texture_evidence, texture_evidence_dim


def test_color_evidence_matches_declared_dim(synthetic_image_bgr):
    vec = extract_color_evidence(synthetic_image_bgr, hist_bins=32)
    assert vec.shape[0] == color_evidence_dim(hist_bins=32)
    assert np.all(np.isfinite(vec))


def test_texture_evidence_matches_declared_dim(synthetic_image_bgr):
    vec = extract_texture_evidence(synthetic_image_bgr, lbp_points=24, lbp_radius=3,
                                    glcm_distances=[1, 2], glcm_angles_deg=[0, 45, 90, 135])
    assert vec.shape[0] == texture_evidence_dim(lbp_points=24)
    assert np.all(np.isfinite(vec))


def test_shape_evidence_matches_declared_dim(synthetic_image_bgr):
    vec = extract_shape_evidence(synthetic_image_bgr)
    assert vec.shape[0] == shape_evidence_dim()
    assert np.all(np.isfinite(vec))


def _fake_components(n, color_dim=8, texture_dim=6, shape_dim=4, deep_dim=16, seed=0):
    rng = np.random.RandomState(seed)
    return [
        EvidenceComponents(
            color=rng.rand(color_dim).astype(np.float32),
            texture=rng.rand(texture_dim).astype(np.float32),
            shape=rng.rand(shape_dim).astype(np.float32),
            deep=rng.rand(deep_dim).astype(np.float32),
        )
        for _ in range(n)
    ]


def test_fusion_stats_fit_on_train_only_and_fuse_produces_expected_dim():
    train_components = _fake_components(20, seed=1)
    stats = fit_fusion_stats(train_components)

    total_dim = 8 + 6 + 4 + 16
    test_component = _fake_components(1, seed=99)[0]
    z = fuse(test_component, stats, normalize="l2")

    assert z.shape[0] == total_dim
    assert abs(np.linalg.norm(z) - 1.0) < 1e-4  # l2-normalized


def test_fuse_zscore_normalize_has_zero_mean():
    train_components = _fake_components(20, seed=2)
    stats = fit_fusion_stats(train_components)
    test_component = _fake_components(1, seed=7)[0]
    z = fuse(test_component, stats, normalize="zscore")
    assert abs(float(z.mean())) < 1e-4
