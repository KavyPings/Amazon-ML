"""End-to-end pipeline (v2): train / predict, sharded by country.

Stages
  0. normalise records (parallel), mined maps from training pairs
  1. blocking   : partitioned k-NN + exact address keys  (per country)          -> raw pairs
  2. pruner     : LightGBM on cheap features, recall-targeted threshold + cap  -> candidate_pairs.tsv
  3. matcher A  : LightGBM on ~60 features (+ context of the pruner score), 2-fold out-of-fold
  4. matcher B  : LightGBM on the same features + context of matcher-A scores ("rivalry")
  5. decoding   : one-owner rule + per-entity expected-F0.5 choice, tuned on a 20% hold-out
"""
from __future__ import annotations

import gc
import json
import os
import pickle
import random
import time

import numpy as np
import pandas as pd

from . import blocking, decode, features, metrics
from . import model as M
from .config import Config
from .io import gt_to_dict, load_split, log, write_id_lists
from .normalize import learn_maps, normalize_frame


# ---------------------------------------------------------------------------
def build_pool(data: dict):
    s1 = data["s1"].reset_index(drop=True)
    pool = pd.concat([data["s2"].assign(src=2), data["s3"].assign(src=3)], ignore_index=True)
    return s1, pool


def closed_subset(s1, pool, truth, maps, frac, seed):
    """Keep a CLOSED fraction of the training data: whole states (S1 + every pool record of
    those states), so candidate density and competition stay realistic."""
    import zlib
    from .normalize import basic_clean, state_of
    c1 = [basic_clean(c) for c in s1["country"]]
    st1 = [state_of(a, c, maps["state"]) for a, c in zip(s1["business_address"], c1)]
    own = {b: st for a, st in zip(s1["entity_id"], st1) for b in truth.get(a, ())}
    stp = [own.get(e) or state_of(a, basic_clean(c), maps["state"])
           for e, a, c in zip(pool["entity_id"], pool["business_address"], pool["country"])]
    pick = lambda st: (zlib.crc32(f"{seed}|{st}".encode()) % 1000) < frac * 1000  # noqa: E731
    k1 = np.array([bool(st) and pick(st) for st in st1])
    k2 = np.array([bool(st) and pick(st) for st in stp])
    log(f"closed subset: {k1.sum():,}/{len(k1):,} S1, {k2.sum():,}/{len(k2):,} pool")
    return s1[k1].reset_index(drop=True), pool[k2].reset_index(drop=True)


def normalize_all(s1, pool, maps, cfg):
    t = time.time()
    s1n = normalize_frame(s1, maps, cfg.n_jobs)
    cn = normalize_frame(pool, maps, cfg.n_jobs)
    cn["src"] = pool["src"].to_numpy().astype(np.int8)
    log(f"normalised {len(s1n):,} S1 + {len(cn):,} pool records ({time.time() - t:.0f}s)")
    return s1n, cn


def _country_pairs(a, b, cfg, pruner=None, cheap=None, thr=None):
    """Blocking + cheap features for ONE country, in S1 chunks (bounded memory).

    pass 1: candidate generation per S1 chunk -> slim (i, j, flags, cosines), capped per (S1, source)
    j-context over ALL chunks (competition between S1 owners of the same record)
    pass 2: remaining cheap features per chunk; with a pruner, score + prune immediately.
    """
    vecs = blocking.fit_vectors(a, b, cfg)
    step = cfg.s1_chunk
    slim = []
    from contextlib import nullcontext
    from .parallel import TopkPool
    use_pool = cfg.n_jobs > 1 and len(a) > 20_000
    ctx = TopkPool(vecs["comb"][0], vecs["comb"][1], cfg.n_jobs) if use_pool else nullcontext(None)
    with ctx as pool:
        for lo in range(0, len(a), step):
            hi = min(len(a), lo + step)
            sub = blocking.slice_s1(vecs, lo, hi, cfg)
            if pool is not None:
                pool.offset = lo
            p = blocking.generate_candidates(a.iloc[lo:hi].reset_index(drop=True), b, sub, cfg, pool)
            ii, jj = p["i"].to_numpy(), p["j"].to_numpy()
            p["c_comb"], p["c_name"], p["c_addr"] = blocking.comb_cos(sub, ii, jj)
            src = b["src"].to_numpy()[jj]
            r, _, _ = features.group_rank(ii.astype(np.int64) * 4 + src, blocking.cap_score(p) if cfg.cap_by_best else p["c_comb"].to_numpy())
            p = p[r <= cfg.raw_cap_per_src]
            p["i"] = (p["i"].to_numpy() + lo).astype(np.int32)
            slim.append(p)
            log(f"    blocking chunk {lo:,}-{hi:,} of {len(a):,}: {len(p):,} pairs")
    P = pd.concat(slim, ignore_index=True)
    del slim
    jctx = features.context_j(P["j"].to_numpy(), P["c_comb"].to_numpy(), "cc")
    jctx = pd.DataFrame(jctx)
    out = []
    bounds = np.searchsorted(P["i"].to_numpy(), np.arange(0, len(a) + step, step))
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi <= lo:
            continue
        chunk = P.iloc[lo:hi].copy()
        jc = {k: jctx[k].to_numpy()[lo:hi] for k in jctx.columns}
        chunk = features.cheap_features(chunk, a, b, vecs, jc)
        if pruner is not None:
            chunk["p0"] = M.predict(pruner, chunk[cheap])
            chunk = prune(chunk, thr, cfg.max_candidates)
        out.append(chunk)
    del P, jctx, vecs
    gc.collect()
    return pd.concat(out, ignore_index=True)


def shard_pairs(s1n, cn, cfg, pruner=None, cheap=None, thr=None):
    """Per-country blocking (+ optional pruning). Returns (pairs with GLOBAL i/j, idf per country)."""
    out, idfs = [], {}
    for country in sorted(s1n["country_key"].unique()):
        t = time.time()
        i_idx = np.flatnonzero(s1n["country_key"].to_numpy() == country)
        j_idx = np.flatnonzero(cn["country_key"].to_numpy() == country)
        if len(j_idx) == 0:          # unseen country in the pool: compare with everything
            j_idx = np.arange(len(cn))
        a = s1n.iloc[i_idx].reset_index(drop=True)
        b = cn.iloc[j_idx].reset_index(drop=True)
        p = _country_pairs(a, b, cfg, pruner, cheap, thr)
        p["i"] = i_idx[p["i"].to_numpy()].astype(np.int32)
        p["j"] = j_idx[p["j"].to_numpy()].astype(np.int32)
        out.append(p)
        idfs[country] = features.token_idf(a["core"], b["core"])
        log(f"  [{country}] {len(a):,} S1 x {len(b):,} pool -> {len(p):,} pairs "
            f"({len(p) / max(1, len(a)):.1f}/S1, {time.time() - t:.0f}s)")
        del a, b
        gc.collect()
    return pd.concat(out, ignore_index=True), idfs


def label_pairs(pairs, s1n, cn, truth):
    s1_ix = pd.Series(np.arange(len(s1n)), index=s1n["entity_id"].to_numpy())
    c_ix = pd.Series(np.arange(len(cn)), index=cn["entity_id"].to_numpy())
    gt = pd.DataFrame([(a, b) for a, v in truth.items() for b in v], columns=["a", "b"])
    gt["i"] = gt["a"].map(s1_ix)
    gt["j"] = gt["b"].map(c_ix)
    gt = gt.dropna().astype({"i": np.int32, "j": np.int32})
    gt["label"] = np.int8(1)
    out = pairs.merge(gt[["i", "j", "label"]], on=["i", "j"], how="left")
    out["label"] = out["label"].fillna(0).astype(np.int8)
    return out


def prune(pairs, thr, cap):
    d = pairs.sort_values(["i", "p0"], ascending=[True, False])
    rank = d.groupby("i").cumcount()
    return d[(d["p0"] >= thr) & (rank < cap)].reset_index(drop=True)


def add_full(cand, s1n, cn, idfs, cfg):
    """Full features per country shard (idf per country)."""
    parts = []
    for country, idf in idfs.items():
        m = s1n["country_key"].to_numpy()[cand["i"].to_numpy()] == country
        if m.any():
            parts.append(features.full_features(cand[m], s1n, cn, None, idf, cfg.n_jobs))
    f = pd.concat(parts).sort_index()
    ctx = features.context_features(f, "p0", "q0")
    for c in ctx.columns:
        f[c] = ctx[c].to_numpy()
    for k, v in features.extra_pair_features(f, s1n, cn).items():
        f[k] = v
    return f


def add_token_scores(f, extra, miss, scores, mask=None):
    """Write learned token-change score features for rows in ``mask`` (all rows if None)."""
    idx = np.arange(len(f)) if mask is None else np.flatnonzero(mask)
    feats = features.token_score_features([extra[k] for k in idx], [miss[k] for k in idx], scores)
    for k, v in feats.items():
        if k not in f:
            f[k] = np.float32(np.nan)
        col = f[k].to_numpy(copy=True)
        col[idx] = v
        f[k] = col


def to_sets(df, s1n, cn):
    s1_ids = s1n["entity_id"].to_numpy()
    c_ids = cn["entity_id"].to_numpy()
    out: dict = {}
    for i, j in zip(df["i"].to_numpy(), df["j"].to_numpy()):
        out.setdefault(s1_ids[i], set()).add(c_ids[j])
    return out


def fast_macro(sel, val_i, ntrue):
    s = sel[sel["i"].isin(val_i)]
    agg = s.groupby("i")["label"].agg(["sum", "count"])
    tp = agg["sum"].reindex(val_i).fillna(0).to_numpy()
    npred = agg["count"].reindex(val_i).fillna(0).to_numpy()
    nt = ntrue.reindex(val_i).fillna(0).to_numpy()
    fn, fp = nt - tp, npred - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(nt == 0, (npred == 0).astype(float),
                     np.where(tp == 0, 0.0, 1.25 * tp / (1.25 * tp + 0.25 * fn + fp)))
    return float(f.mean())


def lgb_kw(cfg, **over):
    kw = dict(num_leaves=cfg.lgb_num_leaves, lr=cfg.lgb_learning_rate, rounds=cfg.lgb_rounds,
              early_stop=cfg.lgb_early_stop, min_child_samples=cfg.lgb_min_child_samples,
              feature_fraction=cfg.lgb_feature_fraction, bagging_fraction=cfg.lgb_bagging_fraction,
              seed=cfg.seed, n_jobs=cfg.n_jobs, verbose=200)
    kw.update(over)
    return kw


# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------
def train(data_dir: str, work_dir: str, cfg: Config) -> dict:
    os.makedirs(work_dir, exist_ok=True)
    T0 = time.time()
    rng = random.Random(cfg.seed)
    data = load_split(data_dir, "train")
    truth = gt_to_dict(data["gt"])
    s1, pool = build_pool(data)
    del data
    for s in s1["entity_id"]:
        truth.setdefault(s, set())
    maps_path = cfg.maps_path or os.path.join(work_dir, "maps.pkl")
    if cfg.maps_path and os.path.exists(cfg.maps_path):
        with open(cfg.maps_path, "rb") as f:
            maps = pickle.load(f)
        log(f"loaded maps from {cfg.maps_path}")
    else:
        maps = learn_maps(s1, pool, truth, cfg, rng)
    if cfg.train_state_frac < 1.0:
        s1, pool = closed_subset(s1, pool, truth, maps, cfg.train_state_frac, cfg.seed)
        truth = {k: truth[k] for k in s1["entity_id"]}
    ids = sorted(truth)
    rng.shuffle(ids)
    val_ids = set(ids[: int(len(ids) * cfg.val_fraction)])
    log(f"maps: tokens={len(maps['tokens'])} addr={len(maps['addr'])} state={len(maps['state'])}")
    with open(os.path.join(work_dir, "maps.pkl"), "wb") as f:
        pickle.dump(maps, f)
    del maps_path
    s1n, cn = normalize_all(s1, pool, maps, cfg)
    del s1, pool
    gc.collect()

    is_val = s1n["entity_id"].isin(val_ids).to_numpy()
    val_truth = {s: truth[s] for s in val_ids}
    if cfg.pruner_dir:
        # two-phase training: reuse a pruner trained earlier (e.g. on a smaller subset) inside the
        # blocking loop, so only the ~5 pruned candidates per S1 are ever held in memory
        with open(os.path.join(cfg.pruner_dir, "params.json")) as f:
            pp = json.load(f)
        pruner = M.load(os.path.join(cfg.pruner_dir, "pruner.txt"))
        ccols, thr = pp["cheap_cols"], float(pp["prune_threshold"])
        cand, idfs = shard_pairs(s1n, cn, cfg, pruner, ccols, thr)
        cand = label_pairs(cand, s1n, cn, truth)
        cand["is_val"] = is_val[cand["i"].to_numpy()]
        raw_rep = {"note": f"pruner reused from {cfg.pruner_dir}"}
        pr_rep = metrics.blocking_report(to_sets(cand[cand.is_val], s1n, cn), val_truth, len(cn))
        log(f"PRUNED candidates (val, reused pruner) thr={thr:.4f}: {json.dumps(pr_rep)}")
    else:
        pairs, idfs = shard_pairs(s1n, cn, cfg)
        pairs = label_pairs(pairs, s1n, cn, truth)
        pairs["is_val"] = is_val[pairs["i"].to_numpy()]
        raw_rep = metrics.blocking_report(to_sets(pairs[pairs.is_val], s1n, cn), val_truth, len(cn))
        log(f"RAW blocking (val): {json.dumps(raw_rep)}")

        # ---- stage 1: pruner
        ccols = features.cheap_cols()
        tr, va = pairs[~pairs.is_val], pairs[pairs.is_val]
        pruner, _ = M.train_lgb(tr[ccols], tr["label"], va[ccols], va["label"],
                                **lgb_kw(cfg, num_leaves=63, lr=0.1, rounds=400, early_stop=30, verbose=100))
        pairs["p0"] = M.predict(pruner, pairs[ccols])
        va = pairs[pairs.is_val]
        capped = prune(va, 0.0, cfg.max_candidates)
        pos_all = int(va["label"].sum())
        pos_p0 = np.sort(capped.loc[capped.label == 1, "p0"].to_numpy())[::-1]
        need = int(np.ceil(cfg.prune_target_recall * pos_all))
        thr = float(pos_p0[need - 1]) if 0 < need <= len(pos_p0) else float(cfg.prune_min_p)
        thr = max(min(thr, 0.5), cfg.prune_min_p)
        cand = prune(pairs, thr, cfg.max_candidates)
        del pairs
        gc.collect()
        pr_rep = metrics.blocking_report(to_sets(cand[cand.is_val], s1n, cn), val_truth, len(cn))
        log(f"PRUNED candidates (val) thr={thr:.4f}: {json.dumps(pr_rep)}")

    # ---- stage 2A: matcher with pruner context, 2-fold OOF on the training part
    ff = add_full(cand, s1n, cn, idfs, cfg)
    tr_mask = ~ff["is_val"].to_numpy()
    # learning-curve experiments: fit the matcher on a share of the training S1 only (all rows are still
    # scored, so the competition context of the validation rows is unchanged)
    fit_mask = tr_mask & ((ff["i"].to_numpy().astype(np.int64) * 2654435761 % 1000)
                          < cfg.matcher_train_frac * 1000)
    fold = (ff["i"].to_numpy() % 2)
    # learned token-change scores: out-of-fold on the training part, full training part for validation
    extra, miss = features.token_diffs(ff, s1n, cn)
    lab = ff["label"].to_numpy()
    sel = lambda m: ([extra[k] for k in np.flatnonzero(m)], [miss[k] for k in np.flatnonzero(m)], lab[m])  # noqa
    tok_all = features.learn_token_scores(*sel(fit_mask))
    add_token_scores(ff, extra, miss, tok_all, ~tr_mask)
    for k in (0, 1):
        add_token_scores(ff, extra, miss, features.learn_token_scores(*sel(fit_mask & (fold != k))),
                         tr_mask & (fold == k))
    with open(os.path.join(work_dir, "token_scores.pkl"), "wb") as fh:
        pickle.dump(tok_all, fh)
    del extra, miss
    fcols = [c for c in ff.columns if c not in ("i", "j", "label", "is_val", "fold")]
    ff["pA"] = 0.0
    modelsA = []
    for k in (0, 1):
        fit = fit_mask & (fold != k)
        oof = tr_mask & (fold == k)
        mA, _ = M.train_lgb(ff.loc[fit, fcols], ff.loc[fit, "label"], ff.loc[~tr_mask, fcols],
                            ff.loc[~tr_mask, "label"], **lgb_kw(cfg))
        ff.loc[oof, "pA"] = M.predict(mA, ff.loc[oof, fcols])
        modelsA.append(mA)
    ff.loc[~tr_mask, "pA"] = np.mean([M.predict(m, ff.loc[~tr_mask, fcols]) for m in modelsA], axis=0)
    ctx = features.context_features(ff, "pA", "qa")
    for c in ctx.columns:
        ff[c] = ctx[c].to_numpy()
    colsB = fcols + ["pA"] + list(ctx.columns)

    # ---- stage 2B: matcher with rivalry context of stage-A scores
    mB, best_it = M.train_lgb(ff.loc[fit_mask, colsB], ff.loc[fit_mask, "label"], ff.loc[~tr_mask, colsB],
                              ff.loc[~tr_mask, "label"], **lgb_kw(cfg))
    ff["p"] = M.predict(mB, ff[colsB])
    ff.loc[tr_mask, "p"] = ff.loc[tr_mask, "pA"]      # in-sample B scores would be optimistic
    imp = sorted(zip(colsB, mB.feature_importance("gain")), key=lambda x: -x[1])
    log("top features: " + ", ".join(f"{a}={b:.0f}" for a, b in imp[:15]))

    # ---- decoding grid on validation entities
    val_i = np.flatnonzero(is_val)
    ntrue = pd.Series({k: len(truth[s1n.entity_id.iat[k]]) for k in val_i})
    base = ff[["i", "j", "p", "label"]]
    grid = []
    for own, margin in [(False, 0.0), (True, 0.0), (True, 0.1), (True, 0.2), (True, 0.3), (True, 0.4),
                        (True, 0.5)]:
        d = decode.one_owner(base, margin) if own else base
        dv = d[d["i"].isin(val_i)]
        for t in np.arange(0.2, 0.96, 0.05):
            grid.append(dict(one_owner=own, one_owner_margin=margin, decode_method="threshold",
                             decode_threshold=round(float(t), 2),
                             score=fast_macro(decode.threshold(dv, t), val_i, ntrue)))
        for a in (0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.6, 2.0, 3.0):
            for mp in (0.02, 0.1, 0.25, 0.4, 0.55):
                grid.append(dict(one_owner=own, one_owner_margin=margin, decode_method="efdecode",
                                 decode_alpha=a, decode_min_p=mp,
                                 score=fast_macro(decode.efdecode(dv, a, mp), val_i, ntrue)))
    grid = sorted(grid, key=lambda g: -g["score"])
    best = dict(grid[0])
    score = best.pop("score")
    cfg.update(best)
    best_thr = max((g for g in grid if g["decode_method"] == "threshold"), key=lambda g: g["score"])
    log(f"BEST decoding: {best} -> val macro F0.5 = {score:.4f} (best plain threshold {best_thr['score']:.4f})")
    sel = decode.apply_decoding(base, cfg)
    pred = to_sets(sel[sel["i"].isin(val_i)], s1n, cn)
    val_rep = metrics.breakdown(pred, val_truth)
    val_rep["by_country"] = {c: metrics.macro_f05(pred, {s: val_truth[s] for s in
                                                         set(s1n.loc[(s1n.country_key == c) & is_val, "entity_id"])})
                             for c in s1n["country_key"].unique() if ((s1n.country_key == c) & is_val).any()}
    log(f"VALIDATION: {json.dumps(val_rep)}")

    vs = ff.loc[~tr_mask, ["i", "j", "p0", "pA", "p", "label"]].copy()
    vs["s1"] = s1n["entity_id"].to_numpy()[vs["i"].to_numpy()]
    vs["c"] = cn["entity_id"].to_numpy()[vs["j"].to_numpy()]
    vs.drop(columns=["i", "j"]).to_parquet(os.path.join(work_dir, "val_scores.parquet"))
    if cfg.save_all_scores:
        keep = [c for c in ["i", "j", "p0", "pA", "p", "label", "is_val", "a_empty2", "qa_rank_j", "qa_n_j",
                            "qa_marg_j", "c_name", "c_addr"] if c in ff.columns]
        al = ff[keep].copy()
        al["s1"] = s1n["entity_id"].to_numpy()[al["i"].to_numpy()]
        al["c"] = cn["entity_id"].to_numpy()[al["j"].to_numpy()]
        al.to_parquet(os.path.join(work_dir, "all_scores.parquet"))
    pruner.save_model(os.path.join(work_dir, "pruner.txt"))
    for k, m in enumerate(modelsA):
        m.save_model(os.path.join(work_dir, f"matcherA{k}.txt"))
    mB.save_model(os.path.join(work_dir, "matcherB.txt"))
    params = {"prune_threshold": thr, "cheap_cols": ccols, "full_cols": fcols, "colsB": colsB}
    with open(os.path.join(work_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=1)
    cfg.save(os.path.join(work_dir, "config.json"))
    report = {"raw_blocking_val": raw_rep, "pruned_candidates_val": pr_rep, "validation": val_rep,
              "best_decoding": best, "best_plain_threshold": best_thr, "grid_top": grid[:15],
              "feature_importance": [(a, float(b)) for a, b in imp], "minutes": (time.time() - T0) / 60}
    with open(os.path.join(work_dir, "train_report.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    log(f"train done in {(time.time() - T0) / 60:.1f} min")
    return report


# ---------------------------------------------------------------------------
# PREDICT
# ---------------------------------------------------------------------------
def _load_predict_artifacts(work_dir, cfg):
    with open(os.path.join(work_dir, "params.json")) as f:
        params = json.load(f)
    with open(os.path.join(work_dir, "maps.pkl"), "rb") as f:
        maps = pickle.load(f)
    art = {"params": params, "maps": maps, "pruner": M.load(os.path.join(work_dir, "pruner.txt")),
           "A": [M.load(os.path.join(work_dir, f"matcherA{k}.txt")) for k in (0, 1)],
           "B": M.load(os.path.join(work_dir, "matcherB.txt"))}
    tok_path = os.path.join(work_dir, "token_scores.pkl")
    art["tok"] = pickle.load(open(tok_path, "rb")) if os.path.exists(tok_path) else None
    return art


def _predict_one(s1, pool, art, cfg, work_dir, country):
    """Score one country's S1 against its pool. Returns (candidates dict, matches dict, stats)."""
    t = time.time()
    params = art["params"]
    s1n, cn = normalize_all(s1, pool, art["maps"], cfg)
    cand, idfs = shard_pairs(s1n, cn, cfg, art["pruner"], params["cheap_cols"], params["prune_threshold"])
    ff = add_full(cand, s1n, cn, idfs, cfg)
    del cand
    if art["tok"] is not None:
        extra, miss = features.token_diffs(ff, s1n, cn)
        add_token_scores(ff, extra, miss, art["tok"])
        del extra, miss
    ff["pA"] = np.mean([M.predict(m, ff[params["full_cols"]]) for m in art["A"]], axis=0)
    ctx = features.context_features(ff, "pA", "qa")
    for c in ctx.columns:
        ff[c] = ctx[c].to_numpy()
    ff["p"] = M.predict(art["B"], ff[params["colsB"]])
    sel = decode.apply_decoding(ff[["i", "j", "p"]], cfg)
    cc = to_sets(ff, s1n, cn)
    mm = {s: m & cc.get(s, set()) for s, m in to_sets(sel, s1n, cn).items()}
    sz = np.array([len(cc.get(s, ())) for s in s1n["entity_id"]])
    ms = np.array([len(mm.get(s, ())) for s in s1n["entity_id"]])
    stats = {"n_s1": len(s1n), "n_pool": len(cn), "cands_mean": float(sz.mean()),
             "share_with_match": float((ms > 0).mean()), "matches_mean": float(ms.mean()),
             "minutes": (time.time() - t) / 60}
    ff[["i", "j", "p0", "pA", "p"]].assign(s1=s1n["entity_id"].to_numpy()[ff["i"].to_numpy()],
                                          c=cn["entity_id"].to_numpy()[ff["j"].to_numpy()]) \
        .drop(columns=["i", "j"]).to_parquet(os.path.join(work_dir, f"test_scores_{country}.parquet"))
    log(f"[{country}] done: {json.dumps(stats)}")
    return cc, mm, stats


def _save_part(work_dir, country, cc, mm, stats):
    with open(os.path.join(work_dir, f"part_{country}.pkl"), "wb") as f:
        pickle.dump({"cands": cc, "matches": mm, "stats": stats}, f)


def predict_country(data_dir: str, work_dir: str, country: str, cfg: Config | None = None) -> dict:
    """Separate-process friendly: score ONE country and store its part (bounded memory)."""
    cfg = cfg or Config.load(os.path.join(work_dir, "config.json"))
    art = _load_predict_artifacts(work_dir, cfg)
    data = load_split(data_dir, "test")
    s1_all, pool_all = build_pool(data)
    del data
    ck1 = s1_all["country"].str.strip().str.lower()
    ck2 = pool_all["country"].str.strip().str.lower()
    s1 = s1_all[ck1 == country].reset_index(drop=True)
    pool = pool_all[ck2 == country].reset_index(drop=True)
    del s1_all, pool_all
    gc.collect()
    cc, mm, stats = _predict_one(s1, pool, art, cfg, work_dir, country)
    _save_part(work_dir, country, cc, mm, stats)
    return stats


def merge_parts(data_dir: str, work_dir: str, out_dir: str) -> dict:
    """Combine per-country parts into the two submission files (every test S1 gets a row)."""
    s1 = load_split(data_dir, "test")["s1"]
    s1_ids = s1["entity_id"].tolist()
    cands, matches, stats = {}, {}, {}
    for fn in sorted(os.listdir(work_dir)):
        if fn.startswith("part_") and fn.endswith(".pkl"):
            with open(os.path.join(work_dir, fn), "rb") as f:
                part = pickle.load(f)
            cands.update(part["cands"])
            matches.update(part["matches"])
            stats[fn[5:-4]] = part["stats"]
    missing = sorted(set(s1["country"].str.strip().str.lower()) - set(stats))
    if missing:
        log(f"WARNING: no part for countries {missing} -> their S1 rows get empty lists")
    os.makedirs(out_dir, exist_ok=True)
    write_id_lists(os.path.join(out_dir, "candidate_pairs.tsv"), "candidate_entity_ids", s1_ids, cands)
    write_id_lists(os.path.join(out_dir, "matching_results.tsv"), "matched_entity_ids", s1_ids, matches)
    sizes = np.array([len(cands.get(s, ())) for s in s1_ids])
    msz = np.array([len(matches.get(s, ())) for s in s1_ids])
    summ = {"n_s1": len(s1_ids), "candidates_per_s1_mean": float(sizes.mean()),
            "candidates_per_s1_p95": float(np.percentile(sizes, 95)), "candidates_per_s1_max": int(sizes.max()),
            "share_s1_with_match": float((msz > 0).mean()), "matches_per_s1_mean": float(msz.mean()),
            "by_country": stats, "missing_countries": missing}
    with open(os.path.join(work_dir, "predict_summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    log(f"MERGED: {json.dumps(summ)}")
    return summ


def predict(data_dir: str, work_dir: str, out_dir: str, cfg: Config | None = None) -> dict:
    """Score the whole test split country by country in THIS process, then merge."""
    cfg = cfg or Config.load(os.path.join(work_dir, "config.json"))
    art = _load_predict_artifacts(work_dir, cfg)
    data = load_split(data_dir, "test")
    s1_all, pool_all = build_pool(data)
    del data
    ck1 = s1_all["country"].str.strip().str.lower()
    ck2 = pool_all["country"].str.strip().str.lower()
    for country in sorted(ck1.unique()):
        s1 = s1_all[ck1 == country].reset_index(drop=True)
        pool = pool_all[ck2 == country].reset_index(drop=True)
        if len(pool) == 0:
            pool = pool_all
        cc, mm, stats = _predict_one(s1, pool, art, cfg, work_dir, country)
        _save_part(work_dir, country, cc, mm, stats)
        del cc, mm, s1, pool
        gc.collect()
    del s1_all, pool_all
    return merge_parts(data_dir, work_dir, out_dir)
