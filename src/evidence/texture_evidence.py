"""MODULE II - Texture Evidence Et(x).

Real descriptors via scikit-image:
  - Local Binary Patterns (LBP) histogram (captures lesion surface texture:
    smooth vs. rough/necrotic patches)
  - Gray-Level Co-occurrence Matrix (GLCM): contrast, homogeneity, energy,
    correlation, entropy -- computed at multiple distances/angles and
    averaged, which is standard practice for rotation-robust texture
    description.
"""
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


def extract_texture_evidence(
    image_bgr: np.ndarray,
    lbp_points: int = 24,
    lbp_radius: int = 3,
    glcm_distances=(1, 2),
    glcm_angles_deg=(0, 45, 90, 135),
) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # --- Local Binary Patterns ---
    lbp = local_binary_pattern(gray, P=lbp_points, R=lbp_radius, method="uniform")
    n_bins = lbp_points + 2
    lbp_hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)

    # --- GLCM (needs quantized gray levels for tractable computation) ---
    gray_q = (gray / 32).astype(np.uint8)  # 8 gray levels
    angles = [np.deg2rad(a) for a in glcm_angles_deg]
    glcm = graycomatrix(gray_q, distances=list(glcm_distances), angles=angles, levels=8, symmetric=True, normed=True)

    contrast = graycoprops(glcm, "contrast").mean()
    homogeneity = graycoprops(glcm, "homogeneity").mean()
    energy = graycoprops(glcm, "energy").mean()
    correlation = graycoprops(glcm, "correlation").mean()
    dissimilarity = graycoprops(glcm, "dissimilarity").mean()
    entropy = _glcm_entropy(glcm)

    glcm_features = np.array([contrast, homogeneity, energy, correlation, dissimilarity, entropy], dtype=np.float32)

    feature = np.concatenate([lbp_hist.astype(np.float32), glcm_features])
    return feature


def _glcm_entropy(glcm: np.ndarray) -> float:
    p = glcm.astype(np.float64)
    p = p / (p.sum() + 1e-12)
    return float(-np.sum(p * np.log2(p + 1e-12)))


def texture_evidence_dim(lbp_points: int = 24) -> int:
    return (lbp_points + 2) + 6
