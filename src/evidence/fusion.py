"""MODULE II - Fusion of the four evidence components into Z(x).

Z(x) = normalize(concat(Ec(x), Et(x), Es(x), Ed(x)))

Each sub-evidence vector is z-scored independently (using train-set
statistics only, to avoid test-set leakage) before concatenation, so that no
single evidence type dominates the fused signature purely due to differing
numeric scales (e.g. deep CNN embeddings vs. a handful of shape scalars).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class EvidenceComponents:
    color: np.ndarray
    texture: np.ndarray
    shape: np.ndarray
    deep: np.ndarray


@dataclass
class FusionStats:
    """Per-component mean/std computed on TRAIN split only; reused at
    inference time so normalization never depends on val/test statistics."""
    color_mean: np.ndarray
    color_std: np.ndarray
    texture_mean: np.ndarray
    texture_std: np.ndarray
    shape_mean: np.ndarray
    shape_std: np.ndarray
    deep_mean: np.ndarray
    deep_std: np.ndarray

    def to_dict(self):
        return {k: v.tolist() for k, v in self.__dict__.items()}

    @staticmethod
    def from_dict(d):
        return FusionStats(**{k: np.array(v, dtype=np.float32) for k, v in d.items()})


def fit_fusion_stats(components_list: list) -> FusionStats:
    """components_list: list[EvidenceComponents] from TRAIN split only."""
    def stack(attr):
        return np.stack([getattr(c, attr) for c in components_list], axis=0)

    def ms(arr):
        return arr.mean(axis=0), np.clip(arr.std(axis=0), 1e-6, None)

    c_mean, c_std = ms(stack("color"))
    t_mean, t_std = ms(stack("texture"))
    s_mean, s_std = ms(stack("shape"))
    d_mean, d_std = ms(stack("deep"))

    return FusionStats(
        color_mean=c_mean, color_std=c_std,
        texture_mean=t_mean, texture_std=t_std,
        shape_mean=s_mean, shape_std=s_std,
        deep_mean=d_mean, deep_std=d_std,
    )


def fuse(components: EvidenceComponents, stats: FusionStats, normalize: str = "l2") -> np.ndarray:
    c = (components.color - stats.color_mean) / stats.color_std
    t = (components.texture - stats.texture_mean) / stats.texture_std
    s = (components.shape - stats.shape_mean) / stats.shape_std
    d = (components.deep - stats.deep_mean) / stats.deep_std

    z = np.concatenate([c, t, s, d]).astype(np.float32)

    if normalize == "l2":
        norm = np.linalg.norm(z) + 1e-8
        z = z / norm
    elif normalize == "zscore":
        z = (z - z.mean()) / (z.std() + 1e-8)
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"Unknown fusion normalize mode: {normalize}")

    return z
