"""
MODULE I - Semantic Class Harmonization.

Loads configs/class_mapping.yaml and applies it to raw (dataset, original_class)
pairs. Anything not explicitly mapped is flagged for manual review rather than
silently included -- this is a hard requirement, not a convenience default.
"""
from dataclasses import dataclass, field
from difflib import SequenceMatcher


@dataclass
class HarmonizationResult:
    mapped: list = field(default_factory=list)          # rows successfully mapped
    unmapped: list = field(default_factory=list)         # (dataset, original_class, num_images) needing review
    excluded: list = field(default_factory=list)         # explicitly excluded classes


def harmonize(records_by_dataset_class: dict, mapping_config: dict) -> HarmonizationResult:
    """records_by_dataset_class: {(dataset, original_class): count}
    mapping_config: parsed configs/class_mapping.yaml
    """
    mapping = mapping_config.get("mapping", {})
    excluded_pairs = set(mapping_config.get("excluded_classes", []) or [])
    canonical_classes = set(mapping_config.get("canonical_classes", []))

    result = HarmonizationResult()

    for (dataset, original_class), count in records_by_dataset_class.items():
        pair_key = f"{dataset}:{original_class}"
        if pair_key in excluded_pairs:
            result.excluded.append({"dataset": dataset, "original_class": original_class, "count": count})
            continue

        dataset_map = mapping.get(dataset, {}) or {}
        canonical = dataset_map.get(original_class, None)

        if canonical is None:
            result.unmapped.append({
                "dataset": dataset,
                "original_class": original_class,
                "count": count,
                "reason": "no mapping entry found (or explicitly null)",
            })
            continue

        if canonical not in canonical_classes:
            result.unmapped.append({
                "dataset": dataset,
                "original_class": original_class,
                "count": count,
                "reason": f"mapped target '{canonical}' is not in canonical_classes list",
            })
            continue

        result.mapped.append({
            "dataset": dataset,
            "original_class": original_class,
            "canonical_class": canonical,
            "count": count,
        })

    return result


def suggest_mapping(records_by_dataset_class: dict, canonical_classes: list, min_ratio: float = 0.45) -> dict:
    """Fuzzy-match original class folder names against the canonical taxonomy
    to produce a SUGGESTED mapping file. This is explicitly a suggestion --
    the harmonize() function above never trusts it automatically; a human
    must copy accepted suggestions into configs/class_mapping.yaml.
    """
    suggestions = {}
    for (dataset, original_class), _count in records_by_dataset_class.items():
        best_match, best_score = None, 0.0
        norm_original = _normalize(original_class)
        for canon in canonical_classes:
            score = SequenceMatcher(None, norm_original, _normalize(canon)).ratio()
            if score > best_score:
                best_match, best_score = canon, score
        suggestions.setdefault(dataset, {})[original_class] = (
            best_match if best_score >= min_ratio else None
        )
    return suggestions


def _normalize(s: str) -> str:
    return s.lower().replace("_", " ").replace("-", " ").strip()
