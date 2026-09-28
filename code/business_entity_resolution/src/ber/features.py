"""Pair features (v2).

* ``cheap_features``  vectorised signals for every raw blocking pair -> stage-1 pruner
* ``full_features``   ~60 language-agnostic signals for the pruned candidates -> stage-2 matcher
* ``context_features`` rank / gap / count features around each pair (per S1, per candidate record)
                       computed from any score column (stage-1 p0, stage-2 out-of-fold p)
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from .blocking import GEN_COLS, comb_cos, rowwise_cos

try:
    from rapidfuzz.process import cpdist
except Exception:  # pragma: no cover
    cpdist = None


def _pw(scorer, a, b, n_jobs=-1):
    """Element-wise scorer(a[k], b[k])."""
    if cpdist is not None:
        return np.asarray(cpdist(a, b, scorer=scorer, workers=n_jobs), dtype=np.float32)
    return np.fromiter((scorer(x, y) for x, y in zip(a, b)), dtype=np.float32, count=len(a))


def _eq3(a, b):
    """1 both present & equal, -1 both present & different, 0 otherwise."""
    a = np.asarray(a, dtype=object)
    b = np.asarray(b, dtype=object)
    both = (a != "") & (b != "")
    return np.where(both, np.where(a == b, 1, -1), 0).astype(np.int8)


def _set_jacc(a, b):
    out = np.full(len(a), -1.0, dtype=np.float32)
    for k, (x, y) in enumerate(zip(a, b)):
        if x and y:
            sx, sy = set(x.split()), set(y.split())
            out[k] = len(sx & sy) / len(sx | sy)
    return out


def group_rank(key, score):
    """(rank desc within key [1=best], gap to key max, key size) as float32 arrays."""
    df = pd.DataFrame({"k": key, "s": score})
    g = df.groupby("k", sort=False)["s"]
    return (g.rank(ascending=False, method="first").to_numpy(np.float32),
            (g.transform("max") - df["s"]).to_numpy(np.float32),
            g.transform("size").to_numpy(np.float32))


def second_gap(key, score):
    """score minus the best OTHER score in the same key group (margin over the runner-up)."""
    df = pd.DataFrame({"k": key, "s": score})
    g = df.groupby("k", sort=False)["s"]
    mx = g.transform("max").to_numpy()
    # second max per group
    srt = df.sort_values(["k", "s"], ascending=[True, False])
    second = srt.groupby("k", sort=False)["s"].nth(1)
    sec = pd.Series(second.to_numpy(), index=srt.loc[second.index, "k"].to_numpy())
    sec = df["k"].map(sec).fillna(-1.0).to_numpy()
    s = df["s"].to_numpy()
    other_best = np.where(s >= mx, sec, mx)
    return (s - other_best).astype(np.float32)


def context_i(df: pd.DataFrame, col: str, prefix: str) -> dict:
    """Rank/gap/sum/count of ``col`` per S1 (i) and per S1+source."""
    i, s = df["i"].to_numpy(), df[col].to_numpy()
    out = {}
    out[f"{prefix}_rank_i"], out[f"{prefix}_gap_i"], out[f"{prefix}_n_i"] = group_rank(i, s)
    out[f"{prefix}_rank_is"], out[f"{prefix}_gap_is"], _ = group_rank(i.astype(np.int64) * 4 + df["src"].to_numpy(), s)
    out[f"{prefix}_sum_i"] = df.groupby("i")[col].transform("sum").to_numpy(np.float32)
    out[f"{prefix}_n_hi_i"] = pd.Series(s > 0.5).groupby(i).transform("sum").to_numpy(np.float32)
    return out


def context_j(j: np.ndarray, s: np.ndarray, prefix: str) -> dict:
    """Rank/gap/count/margin of a score per candidate record (j): competition between S1 owners."""
    out = {}
    out[f"{prefix}_rank_j"], out[f"{prefix}_gap_j"], out[f"{prefix}_n_j"] = group_rank(j, s)
    out[f"{prefix}_marg_j"] = second_gap(j, s)
    return out


def context_features(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    out = context_i(df, col, prefix)
    out.update(context_j(df["j"].to_numpy(), df[col].to_numpy(), prefix))
    return pd.DataFrame(out, index=df.index)


# ---------------------------------------------------------------------------
CHEAP_BASE = ["c_comb", "c_name", "c_addr", "house_eq", "pc_eq", "state_eq", "first_eq", "src"] + GEN_COLS


def cheap_features(pairs: pd.DataFrame, s1n, cn, vecs, jctx: dict | None = None) -> pd.DataFrame:
    """Fast features for raw pairs. ``pairs`` holds i (local to s1n), j, c_comb, g_* flags.
    ``jctx`` = precomputed per-candidate context (over ALL S1 of the shard); computed here if None."""
    i, j = pairs["i"].to_numpy(), pairs["j"].to_numpy()
    f = pairs
    if "c_name" not in f:
        f["c_comb"], f["c_name"], f["c_addr"] = comb_cos(vecs, i, j)
    g = lambda d, c, idx: d[c].to_numpy(dtype=object)[idx]  # noqa: E731
    f["house_eq"] = _eq3(g(s1n, "house", i), g(cn, "house", j))
    f["pc_eq"] = _eq3(g(s1n, "postcode", i), g(cn, "postcode", j))
    f["state_eq"] = _eq3(g(s1n, "state", i), g(cn, "state", j))
    f["first_eq"] = (np.array([a.split(" ", 1)[0] for a in g(s1n, "core", i)], dtype=object)
                     == np.array([a.split(" ", 1)[0] for a in g(cn, "core", j)], dtype=object)).astype(np.int8)
    f["src"] = cn["src"].to_numpy()[j].astype(np.int8)
    for k, v in context_i(f.assign(sc=f["c_comb"]), "sc", "cc").items():
        f[k] = v
    if jctx is None:
        jctx = context_j(j, f["c_comb"].to_numpy(), "cc")
    for k, v in jctx.items():
        f[k] = v
    return f


def cheap_cols():
    return CHEAP_BASE + [f"cc_{x}" for x in ("rank_i", "gap_i", "n_i", "rank_is", "gap_is", "rank_j",
                                            "gap_j", "n_j", "marg_j", "sum_i", "n_hi_i")]


# ---------------------------------------------------------------------------
def token_idf(*cores) -> dict:
    """IDF of core-name tokens over the given record sets."""
    df = Counter()
    n = 0
    for col in cores:
        for c in col:
            df.update(set(c.split()))
            n += 1
    return {t: math.log((n + 1) / (d + 1)) + 1 for t, d in df.items()}


def _idf_overlap(a, b, idf):
    """IDF-weighted Jaccard of token sets + IDF mass of the rarest shared token."""
    out = np.zeros(len(a), dtype=np.float32)
    rare = np.zeros(len(a), dtype=np.float32)
    for k, (x, y) in enumerate(zip(a, b)):
        sx, sy = set(x.split()), set(y.split())
        if not sx or not sy:
            continue
        inter = sx & sy
        wi = sum(idf.get(t, 12.0) for t in inter)
        wu = sum(idf.get(t, 12.0) for t in sx | sy)
        out[k] = wi / wu if wu else 0.0
        rare[k] = max((idf.get(t, 12.0) for t in inter), default=0.0)
    return out, rare


def full_features(cand: pd.DataFrame, s1n, cn, vecs, idf, n_jobs: int = -1) -> pd.DataFrame:
    """Rich features for pruned candidate pairs (``cand`` already holds cheap features + p0)."""
    i, j = cand["i"].to_numpy(), cand["j"].to_numpy()
    f = cand.copy()
    g = lambda d, c, idx: d[c].to_numpy(dtype=object)[idx].tolist()  # noqa: E731

    c1, c2 = g(s1n, "core", i), g(cn, "core", j)
    n1, n2 = g(s1n, "name_clean", i), g(cn, "name_clean", j)
    so1, so2 = [" ".join(sorted(x.split())) for x in c1], [" ".join(sorted(x.split())) for x in c2]
    ns1, ns2 = [x.replace(" ", "") for x in c1], [x.replace(" ", "") for x in c2]
    ph1, ph2 = g(s1n, "core_phon", i), g(cn, "core_phon", j)
    al2 = g(cn, "alias_core", j)
    a1, a2 = g(s1n, "addr_clean", i), g(cn, "addr_clean", j)
    aa1, aa2 = g(s1n, "addr_alpha", i), g(cn, "addr_alpha", j)

    # ---- name
    f["n_ratio"] = _pw(fuzz.ratio, c1, c2, n_jobs)
    f["n_tset"] = _pw(fuzz.token_set_ratio, c1, c2, n_jobs)
    f["n_tsort"] = _pw(fuzz.token_sort_ratio, c1, c2, n_jobs)
    f["n_partial"] = _pw(fuzz.partial_ratio, c1, c2, n_jobs)
    f["n_jw"] = _pw(JaroWinkler.normalized_similarity, c1, c2, n_jobs)
    f["n_full"] = _pw(fuzz.ratio, n1, n2, n_jobs)
    f["n_sorted"] = _pw(fuzz.ratio, so1, so2, n_jobs)
    f["n_nosp"] = _pw(fuzz.ratio, ns1, ns2, n_jobs)
    f["n_nosp_part"] = _pw(fuzz.partial_ratio, ns1, ns2, n_jobs)
    f["n_phon"] = _pw(fuzz.ratio, ph1, ph2, n_jobs)
    f["n_alias"] = np.where(np.array([bool(x) for x in al2]), _pw(fuzz.token_set_ratio, c1, al2, n_jobs), -1).astype(np.float32)
    f["n_idf"], f["n_rare"] = _idf_overlap(c1, c2, idf)
    f["n_rarity1"] = np.array([np.mean([idf.get(t, 12.0) for t in x.split()]) if x else 0 for x in c1], dtype=np.float32)
    acr = lambda cc: ["".join(t[0] for t in x.split() if not t.isdigit()) for x in cc]  # noqa: E731
    acr1, acr2 = acr(c1), acr(c2)
    f["n_acr"] = np.array([(len(y) >= 2 and a == y) or (len(x) >= 2 and b == x)
                           for x, y, a, b in zip(ns1, ns2, acr1, acr2)], dtype=np.int8)
    f["n_acr_eq"] = np.array([a == b and len(a) >= 2 for a, b in zip(acr1, acr2)], dtype=np.int8)
    f["n_core_eq"] = np.array([x == y for x, y in zip(c1, c2)], dtype=np.int8)
    f["n_legal_eq"] = _eq3(g(s1n, "legal", i), g(cn, "legal", j))
    nn = lambda cc: [" ".join(t for t in x.split() if t.isdigit()) for x in cc]  # noqa: E731
    f["n_num_eq"] = _eq3(nn(c1), nn(c2))
    f["n_ntok1"] = np.array([x.count(" ") + 1 if x else 0 for x in c1], dtype=np.int8)
    f["n_ntok2"] = np.array([x.count(" ") + 1 if x else 0 for x in c2], dtype=np.int8)
    f["n_len2"] = np.array([len(x) for x in ns2], dtype=np.int16)
    f["n_glued2"] = np.array([(" " not in x) and len(x) >= 10 for x in c2], dtype=np.int8)
    f["n_filler2"] = cn["has_filler"].to_numpy()[j]
    f["n_indic2"] = cn["was_indic"].to_numpy()[j]
    # ---- address
    f["a_tset"] = _pw(fuzz.token_set_ratio, a1, a2, n_jobs)
    f["a_partial"] = _pw(fuzz.partial_ratio, a1, a2, n_jobs)
    f["a_ratio"] = _pw(fuzz.ratio, a1, a2, n_jobs)
    f["a_alpha_tset"] = _pw(fuzz.token_set_ratio, aa1, aa2, n_jobs)
    f["a_num_jacc"] = _set_jacc(g(s1n, "nums", i), g(cn, "nums", j))
    h1, h2 = g(s1n, "house", i), g(cn, "house", j)
    f["a_house_pref"] = np.array([bool(x and y and x != y and (x.startswith(y) or y.startswith(x)))
                                  for x, y in zip(h1, h2)], dtype=np.int8)
    f["a_empty2"] = np.array([not x for x in a2], dtype=np.int8)
    f["a_land"] = (s1n["landmark"].to_numpy()[i] | cn["landmark"].to_numpy()[j]).astype(np.int8)
    f["a_len2"] = np.array([len(x) for x in a2], dtype=np.int16)
    # ---- name token-set relations
    t1 = [set(x.split()) for x in c1]
    t2 = [set(x.split()) for x in c2]
    f["n_sub12"] = np.fromiter((bool(a) and a <= b for a, b in zip(t1, t2)), dtype=np.int8, count=len(t1))
    f["n_sub21"] = np.fromiter((bool(b) and b <= a for a, b in zip(t1, t2)), dtype=np.int8, count=len(t1))
    f["n_jacc"] = np.fromiter((len(a & b) / len(a | b) if a | b else 0.0 for a, b in zip(t1, t2)),
                              dtype=np.float32, count=len(t1))
    # ---- cluster support: how similar is j to the S1's OTHER strong candidates (top-3 by p0)?
    order = f.sort_values(["i", "p0"], ascending=[True, False])
    oi = order["i"].to_numpy()
    oj = order["j"].to_numpy()
    rk = order.groupby("i", sort=False).cumcount().to_numpy()
    tops = {}
    for r in range(4):
        m = rk == r
        tops[r] = pd.Series(oj[m], index=oi[m])
    cc = cn["core"].to_numpy(dtype=object)
    ca = cn["addr_clean"].to_numpy(dtype=object)
    names, addrs = [], []
    ii = f["i"].to_numpy()
    for r in range(4):
        other = pd.Series(ii).map(tops[r]).fillna(-1).to_numpy().astype(np.int64)
        valid = (other >= 0) & (other != j)
        oth = np.where(valid, other, 0)
        nm = np.where(valid, _pw(fuzz.token_set_ratio, c2, cc[oth].tolist(), n_jobs), np.nan)
        ad = np.where(valid, _pw(fuzz.token_set_ratio, a2, ca[oth].tolist(), n_jobs), np.nan)
        names.append(nm)
        addrs.append(ad)
    N = np.vstack(names)
    A = np.vstack(addrs)
    import warnings
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        f["k_name"] = np.nan_to_num(np.nanmax(N, axis=0), nan=-1).astype(np.float32)
        f["k_addr"] = np.nan_to_num(np.nanmax(A, axis=0), nan=-1).astype(np.float32)
        f["k_name_mean"] = np.nan_to_num(np.nanmean(N, axis=0), nan=-1).astype(np.float32)
        f["k_addr_mean"] = np.nan_to_num(np.nanmean(A, axis=0), nan=-1).astype(np.float32)
        f["k_n_close"] = np.nansum((N >= 85) & (A >= 60), axis=0).astype(np.float32)
    return f


# ---------------------------------------------------------------------------
# learned token-change scores ("is this inserted word typical noise or a different business?")
# ---------------------------------------------------------------------------
LEGAL_FAMILY = {"pvt": "ltd", "ltd": "ltd", "plc": "ltd", "llp": "llp", "lp": "lp", "lllp": "lp", "llc": "llc",
                "pllc": "pllc", "inc": "inc", "corp": "inc", "co": "co", "pc": "pc", "pa": "pc", "sarl": "sarl",
                "sas": "sas", "sasu": "sas", "eurl": "eurl", "sci": "sci", "sa": "sa", "snc": "snc"}


def token_diffs(cand, s1n, cn):
    """(extra tokens in the pool name, S1 tokens missing from the pool name) per pair (name_clean level)."""
    i, j = cand["i"].to_numpy(), cand["j"].to_numpy()
    a = s1n["name_clean"].to_numpy(dtype=object)[i]
    b = cn["name_clean"].to_numpy(dtype=object)[j]
    extra, miss = [], []
    for x, y in zip(a, b):
        sx, sy = set(x.split()), set(y.split())
        extra.append(sorted(sy - sx))
        miss.append(sorted(sx - sy))
    return extra, miss


def learn_token_scores(extra, miss, labels, min_count: int = 20) -> dict:
    """P(true match | token was inserted / deleted), smoothed, from labelled candidate pairs."""
    out = {}
    y = np.asarray(labels)
    prior = float(y.mean()) if len(y) else 0.5
    for name, lists in (("ins", extra), ("del", miss)):
        pos, tot = Counter(), Counter()
        for toks, lab in zip(lists, y):
            for t in toks:
                tot[t] += 1
                pos[t] += int(lab)
        out[name] = {t: (pos[t] + 10 * prior) / (n + 10) for t, n in tot.items() if n >= min_count}
    out["prior"] = prior
    return out


def token_score_features(extra, miss, scores) -> dict:
    """min / mean score and count of inserted and deleted tokens (unknown token -> prior)."""
    pr = scores["prior"]
    res = {}
    for name, lists in (("ins", extra), ("del", miss)):
        tab = scores[name]
        mn = np.empty(len(lists), dtype=np.float32)
        me = np.empty(len(lists), dtype=np.float32)
        cnt = np.empty(len(lists), dtype=np.float32)
        unk = np.empty(len(lists), dtype=np.float32)
        for k, toks in enumerate(lists):
            if not toks:
                mn[k] = me[k] = 1.0
                cnt[k] = unk[k] = 0
                continue
            v = [tab.get(t, pr) for t in toks]
            mn[k], me[k], cnt[k] = min(v), sum(v) / len(v), len(v)
            unk[k] = sum(t not in tab for t in toks)
        res[f"t_{name}_min"], res[f"t_{name}_mean"], res[f"t_{name}_n"], res[f"t_{name}_unk"] = mn, me, cnt, unk
    return res


def extra_pair_features(cand, s1n, cn) -> dict:
    """Legal-family conflict and house-number distance (sibling businesses at nearby numbers)."""
    i, j = cand["i"].to_numpy(), cand["j"].to_numpy()
    l1 = s1n["legal"].to_numpy(dtype=object)[i]
    l2 = cn["legal"].to_numpy(dtype=object)[j]
    conf = np.zeros(len(i), dtype=np.int8)
    for k, (x, y) in enumerate(zip(l1, l2)):
        if x and y:
            fx = {LEGAL_FAMILY.get(t, t) for t in x.split()}
            fy = {LEGAL_FAMILY.get(t, t) for t in y.split()}
            conf[k] = 1 if not (fx & fy) else -1
    h1 = s1n["house"].to_numpy(dtype=object)[i]
    h2 = cn["house"].to_numpy(dtype=object)[j]
    hd = np.full(len(i), -1.0, dtype=np.float32)
    for k, (x, y) in enumerate(zip(h1, h2)):
        if x and y and len(x) < 10 and len(y) < 10:
            hd[k] = np.log1p(abs(int(x) - int(y)))
    return {"legal_conflict": conf, "house_logdiff": hd}
