"""
MODULE III - Disease Evidence Prototypes P(c,d).

P(c,d) is the mean fused evidence signature Z(x) for canonical disease class
c within dataset/domain d, computed EXCLUSIVELY from the training split.
Using test (or even validation) data here would leak information into the
similarity/DESS computation and invalidate the reliability assessment.
"""
from collections import defaultdict

import numpy as np


def build_prototypes(z_vectors: list, canonical_classes: list, domains: list) -> dict:
    """
    z_vectors: list[np.ndarray], TRAIN-split fused evidence signatures Z(x)
    canonical_classes: list[str], same length as z_vectors
    domains: list[str], same length as z_vectors (dataset/domain id per sample)

    Returns: {(canonical_class, domain): {"prototype": np.ndarray, "n_samples": int}}
    """
    assert len(z_vectors) == len(canonical_classes) == len(domains), (
        "z_vectors, canonical_classes, and domains must be the same length"
    )
    groups = defaultdict(list)
    for z, c, d in zip(z_vectors, canonical_classes, domains):
        groups[(c, d)].append(z)

    prototypes = {}
    for key, vectors in groups.items():
        stacked = np.stack(vectors, axis=0)
        prototypes[key] = {
            "prototype": stacked.mean(axis=0),
            "std": stacked.std(axis=0),
            "n_samples": len(vectors),
        }
    return prototypes


def save_prototypes(prototypes: dict, path: str) -> None:
    import pickle
    with open(path, "wb") as f:
        pickle.dump(prototypes, f)


def load_prototypes(path: str) -> dict:
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)
