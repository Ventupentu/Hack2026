"""Validation metrics for multi-product retrieval."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Set


def evaluate_bundle_retrieval(
    bundle_ids: Sequence[str],
    predictions: Sequence[Sequence[str]],
    ground_truth: Dict[str, Set[str]],
    ks: Iterable[int],
) -> Dict[str, float]:
    """Compute mean Hit@K and Recall@K across bundles."""
    ks_sorted = sorted(set(int(k) for k in ks if int(k) > 0))
    if not ks_sorted:
        raise ValueError("Provide at least one positive K.")
    if len(bundle_ids) != len(predictions):
        raise ValueError("bundle_ids and predictions must have same length.")

    hit_sums = defaultdict(float)
    recall_sums = defaultdict(float)
    evaluated = 0

    for bundle_id, pred_ids in zip(bundle_ids, predictions):
        gt = ground_truth.get(bundle_id, set())
        if not gt:
            continue
        pred_list: List[str] = list(pred_ids)
        evaluated += 1
        for k in ks_sorted:
            topk = pred_list[:k]
            matches = sum(1 for pid in topk if pid in gt)
            hit_sums[k] += 1.0 if matches > 0 else 0.0
            recall_sums[k] += matches / max(len(gt), 1)

    if evaluated == 0:
        return {"num_bundles_evaluated": 0.0}

    out: Dict[str, float] = {"num_bundles_evaluated": float(evaluated)}
    for k in ks_sorted:
        out[f"hit@{k}"] = hit_sums[k] / evaluated
        out[f"recall@{k}"] = recall_sums[k] / evaluated
    return out
