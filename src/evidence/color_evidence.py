"""MODULE II - Color Evidence Ec(x).

Real, computed color descriptors (not fabricated placeholders):
  - RGB channel mean/std
  - HSV channel mean/std
  - Hue and Saturation histograms (captures lesion vs. healthy-leaf color
    distribution, which is a strong cue for citrus disease symptoms such as
    canker lesions or HLB blotchy mottling)
"""
import cv2
import numpy as np


def extract_color_evidence(image_bgr: np.ndarray, hist_bins: int = 32) -> np.ndarray:
    """image_bgr: HxWx3 uint8 array (as read by cv2.imread).
    Returns a 1D float32 feature vector.
    """
    img = image_bgr.astype(np.float32) / 255.0
    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0

    rgb_mean = img.reshape(-1, 3).mean(axis=0)
    rgb_std = img.reshape(-1, 3).std(axis=0)
    hsv_mean = hsv.reshape(-1, 3).mean(axis=0)
    hsv_std = hsv.reshape(-1, 3).std(axis=0)

    hue_hist = cv2.calcHist([hsv], [0], None, [hist_bins], [0, 1]).flatten()
    sat_hist = cv2.calcHist([hsv], [1], None, [hist_bins], [0, 1]).flatten()
    hue_hist = hue_hist / (hue_hist.sum() + 1e-8)
    sat_hist = sat_hist / (sat_hist.sum() + 1e-8)

    feature = np.concatenate([rgb_mean, rgb_std, hsv_mean, hsv_std, hue_hist, sat_hist]).astype(np.float32)
    return feature


def color_evidence_dim(hist_bins: int = 32) -> int:
    return 3 + 3 + 3 + 3 + hist_bins + hist_bins
