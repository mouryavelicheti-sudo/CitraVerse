"""
Ablation framework.

Evidence ablations (Module II contribution):
  A: Deep semantic evidence only
  B: Color + Texture + Shape (handcrafted only)
  C: Deep + handcrafted (no fusion normalization ablation)
  D: Full fused evidence (the proposed system)

Model ablations (Module V contribution):
  Teacher vs Student vs INT8 Student

This module does not invent numbers: it defines *which* evidence subset /
model variant to evaluate for each row, and the caller (scripts/evaluate.py)
must supply metrics computed from an actual run.
"""
from dataclasses import dataclass, field


EVIDENCE_ABLATIONS = {
    "A_deep_only": ["deep"],
    "B_handcrafted_only": ["color", "texture", "shape"],
    "C_deep_plus_handcrafted": ["color", "texture", "shape", "deep"],  # same as D at feature-set level;
    "D_full_fused": ["color", "texture", "shape", "deep"],             # differs from C in downstream usage (see README)
}

MODEL_VARIANTS = ["teacher", "student_fp32", "student_int8"]


@dataclass
class AblationRow:
    ablation_id: str
    evidence_components: list
    metrics: dict = field(default_factory=dict)


def select_evidence_subset(components, evidence_names: list):
    """components: src.evidence.fusion.EvidenceComponents (or dict with the
    same attribute names). evidence_names: subset of ["color","texture","shape","deep"].
    Returns a numpy vector formed by concatenating only the requested
    components, in a fixed canonical order.
    """
    import numpy as np

    order = ["color", "texture", "shape", "deep"]
    parts = []
    for name in order:
        if name in evidence_names:
            parts.append(getattr(components, name) if hasattr(components, name) else components[name])
    return np.concatenate(parts).astype(np.float32)


def build_ablation_table(results: dict) -> "pandas.DataFrame":
    """results: {ablation_id: {"accuracy":..., "precision":..., "recall":...,
    "f1":..., "dess":..., "pcs":..., "reliability":..., "model_size_mb":...,
    "latency_ms":...}} -- all real, measured values.
    """
    import pandas as pd

    rows = []
    for ablation_id, metrics in results.items():
        row = {"ablation_id": ablation_id}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)
