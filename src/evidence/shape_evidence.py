"""MODULE II - Shape / Lesion Evidence Es(x).

Real lesion-region ESTIMATION (not ground-truth segmentation, which the
datasets do not provide): we approximate lesion/diseased regions using
color-based thresholding (dark/brown/yellow spot detection against green
leaf background) and then compute standard shape descriptors from the
resulting contours. This is a documented, defensible proxy -- not claimed to
be a state-of-the-art segmentation model. If pixel-level lesion masks become
available for your dataset, replace `_estimate_lesion_mask` with a proper
segmentation model and the rest of this module is unaffected.
"""
import cv2
import numpy as np


def _estimate_lesion_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Healthy citrus leaf: green hue band, reasonably high saturation.
    healthy_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    # Everything reasonably foreground (non-background) but NOT healthy-green
    # is treated as a candidate lesion / discoloration region.
    non_background = cv2.inRange(hsv, (0, 20, 20), (179, 255, 255))
    lesion_candidate = cv2.bitwise_and(non_background, cv2.bitwise_not(healthy_mask))

    # Morphological cleanup to remove speckle noise.
    kernel = np.ones((3, 3), np.uint8)
    lesion_candidate = cv2.morphologyEx(lesion_candidate, cv2.MORPH_OPEN, kernel, iterations=1)
    lesion_candidate = cv2.morphologyEx(lesion_candidate, cv2.MORPH_CLOSE, kernel, iterations=1)
    return lesion_candidate


def extract_shape_evidence(image_bgr: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    image_area = float(h * w)
    mask = _estimate_lesion_mask(image_bgr)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 4]  # drop noise specks

    lesion_area_ratio = float(mask.sum() / 255) / image_area if image_area else 0.0
    num_lesion_regions = len(contours)

    if not contours:
        # No lesion-like region detected -> likely healthy or lesion below
        # detection threshold. Return a well-defined zero-shape vector.
        return np.array([lesion_area_ratio, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    areas = [cv2.contourArea(c) for c in contours]
    perimeters = [cv2.arcLength(c, True) for c in contours]
    circularities = [
        (4 * np.pi * a) / (p ** 2) if p > 0 else 0.0 for a, p in zip(areas, perimeters)
    ]

    largest_idx = int(np.argmax(areas))
    largest_area_frac = areas[largest_idx] / image_area
    mean_circularity = float(np.mean(circularities))
    mean_perimeter_norm = float(np.mean(perimeters)) / (h + w)

    feature = np.array([
        lesion_area_ratio,
        float(num_lesion_regions),
        largest_area_frac,
        mean_circularity,
        mean_perimeter_norm,
        float(np.std(areas)) / (image_area + 1e-8),
    ], dtype=np.float32)
    return feature


def shape_evidence_dim() -> int:
    return 6
