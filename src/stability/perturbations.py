"""
MODULE III - Perturbation generator T_e(x).

Realistic, configurable visual perturbations used to test whether the fused
evidence signature Z(x) remains stable under conditions that a real edge
camera would encounter (lighting shifts, shadows, sensor noise, blur, and
contrast changes).
"""
import cv2
import numpy as np


def brightness(image_bgr: np.ndarray, delta: float) -> np.ndarray:
    """delta in [-1, 1], fraction of max intensity to add/subtract."""
    img = image_bgr.astype(np.float32) + delta * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def contrast(image_bgr: np.ndarray, factor: float) -> np.ndarray:
    mean = image_bgr.astype(np.float32).mean()
    img = (image_bgr.astype(np.float32) - mean) * factor + mean
    return np.clip(img, 0, 255).astype(np.uint8)


def gaussian_noise(image_bgr: np.ndarray, std: float, seed=None) -> np.ndarray:
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, std * 255.0, image_bgr.shape)
    img = image_bgr.astype(np.float32) + noise
    return np.clip(img, 0, 255).astype(np.uint8)


def gaussian_blur(image_bgr: np.ndarray, kernel: int) -> np.ndarray:
    k = kernel if kernel % 2 == 1 else kernel + 1
    return cv2.GaussianBlur(image_bgr, (k, k), 0)


def synthetic_shadow(image_bgr: np.ndarray, strength: float, seed=None) -> np.ndarray:
    """Overlays a soft elliptical shadow patch -- approximates leaf
    self-shadowing or occlusion under natural/field lighting."""
    rng = np.random.RandomState(seed)
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    cx, cy = rng.randint(0, w), rng.randint(0, h)
    axes = (rng.randint(w // 4, w // 2 + 1), rng.randint(h // 4, h // 2 + 1))
    cv2.ellipse(mask, (cx, cy), axes, angle=rng.randint(0, 180), startAngle=0, endAngle=360, color=1.0, thickness=-1)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    img = image_bgr.astype(np.float32) * (1 - strength * mask[..., None])
    return np.clip(img, 0, 255).astype(np.uint8)


PERTURBATION_REGISTRY = {
    "brightness_up": lambda img, cfg, seed: brightness(img, cfg["brightness_delta"][1]),
    "brightness_down": lambda img, cfg, seed: brightness(img, cfg["brightness_delta"][0]),
    "contrast_low": lambda img, cfg, seed: contrast(img, cfg["contrast_factor"][0]),
    "contrast_high": lambda img, cfg, seed: contrast(img, cfg["contrast_factor"][1]),
    "gaussian_noise": lambda img, cfg, seed: gaussian_noise(img, cfg["gaussian_noise_std"], seed),
    "gaussian_blur": lambda img, cfg, seed: gaussian_blur(img, cfg["gaussian_blur_kernel"]),
    "shadow": lambda img, cfg, seed: synthetic_shadow(img, cfg["shadow_strength"], seed),
}


def apply_all_perturbations(image_bgr: np.ndarray, cfg: dict, seed: int = 42) -> dict:
    """Returns {perturbation_name: perturbed_image_bgr} for every
    perturbation registered above, using settings from configs/edge.yaml."""
    return {name: fn(image_bgr, cfg, seed) for name, fn in PERTURBATION_REGISTRY.items()}
