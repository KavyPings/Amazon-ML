"""Exact re-implementation of the leaderboard metric + blocking diagnostics."""
from __future__ import annotations

import numpy as np


def f05(pred: set, truth: set) -> float:
    """F0.5 for one S1 entity. Singleton: 1.0 for an empty prediction, else 0.0."""
    if not truth:
        return 1.0 if not pred else 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred: dict, truth: dict) -> float:
    """Macro-average F0.5 over every S1 entity in ``truth`` (the leaderboard metric)."""
    if not truth:
        return float("nan")
    return float(np.mean([f05(pred.get(s1, set()), t) for s1, t in truth.items()]))


def breakdown(pred: dict, truth: dict) -> dict:
    """Score split into singletons vs entities that have matches, plus P/R."""
    sing = [f05(pred.get(s, set()), t) for s, t in truth.items() if not t]
    mat = [f05(pred.get(s, set()), t) for s, t in truth.items() if t]
    tp = sum(len(pred.get(s, set()) & t) for s, t in truth.items())
    npred = sum(len(pred.get(s, set())) for s in truth)
    ntrue = sum(len(t) for t in truth.values())
    return {
        "macro_f05": macro_f05(pred, truth),
        "singleton_score": float(np.mean(sing)) if sing else None,
        "matched_entity_score": float(np.mean(mat)) if mat else None,
        "pair_precision": tp / npred if npred else None,
        "pair_recall": tp / ntrue if ntrue else None,
        "n_entities": len(truth),
        "n_singletons": len(sing),
    }


def blocking_report(cands: dict, truth: dict, n_pool: int) -> dict:
    """Candidate-set quality: recall ceiling, size per S1, reduction ratio, oracle F0.5."""
    ntrue = sum(len(t) for t in truth.values())
    hit = sum(len(cands.get(s, set()) & t) for s, t in truth.items())
    sizes = np.array([len(cands.get(s, set())) for s in truth])
    oracle = {s: cands.get(s, set()) & t for s, t in truth.items()}
    return {
        "pair_recall": hit / ntrue if ntrue else None,
        "mean_candidates_per_s1": float(sizes.mean()) if len(sizes) else 0.0,
        "median_candidates_per_s1": float(np.median(sizes)) if len(sizes) else 0.0,
        "p95_candidates_per_s1": float(np.percentile(sizes, 95)) if len(sizes) else 0.0,
        "max_candidates_per_s1": int(sizes.max()) if len(sizes) else 0,
        "reduction_ratio": 1.0 - sizes.sum() / max(1, len(truth) * n_pool),
        "oracle_macro_f05": macro_f05(oracle, truth),
    }
