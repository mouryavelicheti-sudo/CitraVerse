"""
MODULE III - Prototype similarity + DESS.

DESS (Disease Evidence Stability Score) for a given canonical class c
measures how CONSISTENTLY the fused evidence signature for that disease is
supported ACROSS DATASETS/DOMAINS. It is a project-proposed metric, not an
established literature metric -- documented explicitly here and in the
architecture README.

Definition used in this implementation:
  For class c, let {P(c,d1), P(c,d2), ..., P(c,dk)} be its prototypes across
  the k domains that contain class c. DESS(c) is the mean pairwise
  similarity between all P(c,di), P(c,dj) pairs (i != j). A DESS close to 1
  (cosine) means the disease's evidence signature is stable/domain-invariant;
  a low DESS flags a class whose visual evidence differs substantially
  across datasets (e.g. due to acquisition conditions, cultivar, or label
  noise), which is itself an actionable research finding.
"""
from itertools import combinations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def euclidean_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Converted to a similarity in (0, 1] via 1 / (1 + distance)."""
    dist = float(np.linalg.norm(a - b))
    return 1.0 / (1.0 + dist)


def prototype_similarity(z: np.ndarray, prototype: np.ndarray, metric: str = "cosine") -> float:
    if metric == "cosine":
        return cosine_similarity(z, prototype)
    if metric == "euclidean":
        return euclidean_similarity(z, prototype)
    raise ValueError(f"Unknown similarity metric: {metric}")


def compute_dess(prototypes: dict, metric: str = "cosine", aggregate: str = "mean") -> dict:
    """prototypes: output of prototype_builder.build_prototypes(), keyed by
    (canonical_class, domain) -> {"prototype": np.ndarray, ...}

    Returns {canonical_class: dess_score}. Classes present in only one
    domain get DESS = None (undefined -- cannot measure cross-dataset
    stability with a single domain; this is reported, not silently 1.0).
    """
    by_class = {}
    for (c, d), info in prototypes.items():
        by_class.setdefault(c, {})[d] = info["prototype"]

    dess = {}
    for c, domain_protos in by_class.items():
        domains = list(domain_protos.keys())
        if len(domains) < 2:
            dess[c] = None
            continue
        sims = []
        for d1, d2 in combinations(domains, 2):
            sim_fn = cosine_similarity if metric == "cosine" else euclidean_similarity
            sims.append(sim_fn(domain_protos[d1], domain_protos[d2]))
        dess[c] = float(np.mean(sims)) if aggregate == "mean" else float(np.min(sims))
    return dess


def prototype_agreement(z: np.ndarray, canonical_class: str, prototypes: dict, metric: str = "cosine") -> float:
    """A(x): agreement of a single evidence signature z with the (possibly
    multiple, cross-domain) prototypes of its predicted class, aggregated by
    mean similarity across domains that have that class."""
    sims = [
        prototype_similarity(z, info["prototype"], metric)
        for (c, _d), info in prototypes.items()
        if c == canonical_class
    ]
    if not sims:
        return 0.0
    return float(np.mean(sims))
