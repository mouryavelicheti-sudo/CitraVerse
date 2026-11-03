"""
MODULE III - Perturbation Consistency Score (PCS) and Evidence Stability
Loss L_ES.

    L_ES = || Z(x) - Z(T_e(x)) ||^2

PCS(x) converts the (average, over registered perturbations) evidence drift
into a bounded [0, 1] consistency score via PCS = 1 / (1 + mean_L_ES), so
that E(x) in the reliability score (Module IV) is directly usable as a
"higher is better" signal.

This is a project-defined metric, not an established benchmark metric --
documented here and in configs/reliability.yaml.
"""
import numpy as np


def evidence_stability_loss(z_original: np.ndarray, z_perturbed: np.ndarray) -> float:
    """L_ES = ||Z(x) - Z(T_e(x))||^2 for a single perturbation."""
    diff = z_original.astype(np.float64) - z_perturbed.astype(np.float64)
    return float(np.dot(diff, diff))


def perturbation_consistency_score(z_original: np.ndarray, z_perturbed_dict: dict, aggregate: str = "mean") -> dict:
    """z_perturbed_dict: {perturbation_name: Z(T_e(x))}

    Returns {
        "per_perturbation_L_ES": {name: loss},
        "aggregate_L_ES": float,
        "PCS": float in (0, 1],
    }
    """
    losses = {name: evidence_stability_loss(z_original, z_p) for name, z_p in z_perturbed_dict.items()}
    values = list(losses.values())
    agg_loss = float(np.mean(values)) if aggregate == "mean" else float(np.max(values))
    pcs = 1.0 / (1.0 + agg_loss)
    return {"per_perturbation_L_ES": losses, "aggregate_L_ES": agg_loss, "PCS": pcs}
