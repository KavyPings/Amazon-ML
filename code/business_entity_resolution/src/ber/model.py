"""LightGBM (MIT licence) wrappers for the stage-1 pruner and the stage-2 matcher."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np


def train_lgb(X, y, Xv=None, yv=None, *, num_leaves=63, lr=0.05, rounds=1000, early_stop=50,
              min_child_samples=40, feature_fraction=0.8, bagging_fraction=0.8, seed=42,
              n_jobs=4, verbose=100):
    """Train a binary LightGBM model, early-stopping on (Xv, yv) when given.

    Returns (booster, best_iteration).
    """
    params = dict(objective="binary", metric=["binary_logloss", "auc"], num_leaves=num_leaves,
                  learning_rate=lr, min_child_samples=min_child_samples,
                  feature_fraction=feature_fraction, bagging_fraction=bagging_fraction,
                  bagging_freq=1, seed=seed, num_threads=n_jobs, verbose=-1)
    dtr = lgb.Dataset(np.asarray(X, dtype=np.float32), label=np.asarray(y))
    valid, cbs = [dtr], [lgb.log_evaluation(verbose)]
    if Xv is not None and len(Xv) and len(np.unique(yv)) > 1:
        dv = lgb.Dataset(np.asarray(Xv, dtype=np.float32), label=np.asarray(yv), reference=dtr)
        valid = [dv]
        cbs.append(lgb.early_stopping(early_stop, verbose=False))
    booster = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=valid, callbacks=cbs)
    best = booster.best_iteration or booster.current_iteration()
    return booster, int(best)


def predict(booster, X) -> np.ndarray:
    """Probability of 'same business' for each row of X."""
    return booster.predict(np.asarray(X, dtype=np.float32),
                           num_iteration=booster.best_iteration or None).astype(np.float32)


def load(path: str):
    return lgb.Booster(model_file=path)
