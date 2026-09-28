"""Turning pair probabilities into per-entity match lists that maximise macro F0.5.

1. one_owner        : S1 is deduplicated, so an S2/S3 record belongs to at most one
                      S1 entity. Keep only its best owner (optionally with a margin).
2. efdecode         : for each S1 entity choose the prefix of its probability-sorted
                      candidates (possibly empty) with the highest expected F0.5.
3. threshold        : plain p >= t baseline, kept for comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def one_owner(df: pd.DataFrame, margin: float = 0.0) -> pd.DataFrame:
    """Keep, for each candidate record j, only the highest-probability S1 owner.

    If ``margin`` > 0 the winner must beat the runner-up by at least ``margin``,
    otherwise the record is dropped for everyone (ambiguous -> no merge).
    Expects columns i, j, p. Returns the filtered frame.
    """
    d = df.sort_values(["j", "p"], ascending=[True, False])
    rk = d.groupby("j").cumcount()
    if margin > 0:
        second = d.loc[rk == 1].set_index("j")["p"]
        runner = d["j"].map(second).fillna(0.0)
        return d[(rk == 0) & ((d["p"] - runner) >= margin)]
    return d[rk == 0]


def efdecode(df: pd.DataFrame, alpha: float = 1.0, min_p: float = 0.0) -> pd.DataFrame:
    """Expected-F0.5 decoding per S1 entity (columns i, j, p) -> selected rows.

    For candidates sorted by p (desc), the expected F0.5 of predicting the top-k is
    approximated by 1.25*TP / (1.25*TP + 0.25*FN + FP) with TP = sum of the top-k
    probabilities, FP = k - TP and FN = remaining probability mass. Predicting
    nothing is worth P(no true match) ~= prod(1 - p) (a singleton scores 1.0),
    scaled by ``alpha``. The best option wins.
    """
    d = df[df["p"] >= min_p].sort_values(["i", "p"], ascending=[True, False]).copy()
    if d.empty:
        return d
    g = d.groupby("i", sort=False)
    d["k"] = g.cumcount() + 1
    tp = g["p"].cumsum()
    fn = g["p"].transform("sum") - tp
    fp = d["k"] - tp
    d["f"] = 1.25 * tp / (1.25 * tp + 0.25 * fn + fp)
    best_idx = d.groupby("i", sort=False)["f"].idxmax()
    best = d.loc[best_idx, ["i", "k", "f"]].set_index("i")
    # value of predicting nothing, computed over ALL candidates of the entity
    lp = np.log1p(-df["p"].clip(0, 1 - 1e-6)).groupby(df["i"]).sum()
    empty_val = alpha * np.exp(lp)
    kbest = d["i"].map(best["k"])
    fbest = d["i"].map(best["f"])
    keep = (d["k"] <= kbest) & (fbest > d["i"].map(empty_val))
    return d[keep]


def threshold(df: pd.DataFrame, t: float) -> pd.DataFrame:
    """Baseline: keep every pair with p >= t."""
    return df[df["p"] >= t]


def apply_decoding(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Full decoding chain configured by ``cfg`` (columns i, j, p) -> selected rows."""
    d = df[df["p"] >= cfg.decode_min_p]
    if cfg.one_owner:
        d = one_owner(d, cfg.one_owner_margin)
    if cfg.decode_method == "threshold":
        return threshold(d, cfg.decode_threshold)
    return efdecode(d, cfg.decode_alpha, cfg.decode_min_p)
