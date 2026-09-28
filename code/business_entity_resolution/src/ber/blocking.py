"""Scalable candidate generation (blocking) — v4.

Every generator only compares records inside small partitions (never N x N), and all
pairs are same-country (100% of true pairs in train are). Generators:

 name k-NN (combined TF-IDF cosine, top-k per source) inside partitions keyed by
   P1 first 3 letters of the core name
   P2 first 3 letters of the longest core token      (word re-ordering / inserted words)
   P3 last 3 letters of the core name                 (typos at the start)
   P4 first 4 letters of the phonetic key             (transliterated Indic names)
   AH the P1 partitions again, ranked by an ADDRESS-heavy score (generic names whose
      name top-k fills up with namesakes)
   H  house number, S first street word, RA rarest address word  (same place, unrelated name)
 exact keys (bounded block size):
   A1 house number + first 4 letters of the first street word   (renamed / DBA businesses)
   NB any shared pair of consecutive address numbers              ("16-11-17", "4024 39th")
 optional: RV reverse pool->S1 name search for address-less pool records (off by default)

Oversized partitions are split with longer prefixes, so each partition stays small and
the cost grows ~linearly with the number of records. All keys are integer codes (memory),
small partitions are enumerated in bounded vectorised batches, big ones use a sparse top-k
product in worker processes (parallel.py).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from .io import log

try:
    from sparse_dot_topn import sp_matmul_topn
except Exception:  # pragma: no cover
    sp_matmul_topn = None


# ---------------------------------------------------------------------------
def fit_vectors(s1n: pd.DataFrame, cn: pd.DataFrame, cfg, sample: int = 3_000_000):
    """TF-IDF spaces (fit on a sample, transform all), stored as ONE combined matrix per side.

    C = [w_n * name | w_a * addr] / norm   (rows L2-normalised; name = char-3gram TF-IDF of the core
    name, addr = word TF-IDF of the cleaned address). Neighbours inside blocking partitions are
    ranked by C (identical generic names get separated by address). The separate name / address
    cosines are recovered exactly from column ranges of C, so nothing else is kept in memory.
    """
    mats = {}
    specs = {"name": ("core", dict(analyzer="char_wb", ngram_range=(3, 3), min_df=2)),
             "addr": ("addr_clean", dict(analyzer="word", token_pattern=r"(?u)\b\w+\b", min_df=2))}
    for key, (col, kw) in specs.items():
        fit_on = pd.concat([s1n[col], cn[col]], ignore_index=True)
        if len(fit_on) > sample:
            fit_on = fit_on.sample(sample, random_state=0)
        vec = TfidfVectorizer(sublinear_tf=True, dtype=np.float32, **kw).fit(fit_on)
        mats[key] = (vec.transform(s1n[col]).tocsr(), vec.transform(cn[col]).tocsr())
    wn, wa = cfg.comb_w_name, cfg.comb_w_addr
    C, inv = [], []
    for side in (0, 1):
        n, a = mats["name"][side], mats["addr"][side]
        nzn = (n.getnnz(axis=1) > 0).astype(np.float32)
        nza = (a.getnnz(axis=1) > 0).astype(np.float32)
        norm = np.sqrt(wn * wn * nzn + wa * wa * nza)
        iv = np.where(norm > 0, 1.0 / np.maximum(norm, 1e-12), 0.0).astype(np.float32)
        D = sp.diags(iv)
        C.append(sp.hstack([D @ (wn * n), D @ (wa * a)]).tocsr().astype(np.float32))
        inv.append(iv)
    split = mats["name"][0].shape[1]
    del mats
    return {"comb": tuple(C), "inv": tuple(inv), "split": split, "w": (wn, wa)}


def slice_s1(vecs, lo, hi, cfg=None):
    """Vectors restricted to S1 rows [lo, hi) (pool side unchanged)."""
    return {"comb": (vecs["comb"][0][lo:hi], vecs["comb"][1]), "inv": (vecs["inv"][0][lo:hi], vecs["inv"][1]),
            "split": vecs["split"], "w": vecs["w"]}


def comb_cos(vecs, i, j, chunk: int = 1_000_000):
    """(combined, name, address) cosines for pairs, recovered from column ranges of C."""
    C1, C2 = vecs["comb"]
    inv1, inv2 = vecs["inv"]
    wn, wa = vecs["w"]
    sp_ = vecs["split"]
    cn_ = np.zeros(len(i), dtype=np.float32)
    ca_ = np.zeros(len(i), dtype=np.float32)
    for s in range(0, len(i), chunk):
        e = s + chunk
        Mx = C1[i[s:e]].multiply(C2[j[s:e]]).tocsc()
        cn_[s:e] = np.asarray(Mx[:, :sp_].sum(axis=1)).ravel()
        ca_[s:e] = np.asarray(Mx[:, sp_:].sum(axis=1)).ravel()
    comb = cn_ + ca_
    den = inv1[i] * inv2[j]
    with np.errstate(divide="ignore", invalid="ignore"):
        c_name = np.where(den > 0, cn_ / (wn * wn * den), 0.0).astype(np.float32)
        c_addr = np.where(den > 0, ca_ / (wa * wa * den), 0.0).astype(np.float32)
    return comb.astype(np.float32), np.clip(c_name, 0, 1), np.clip(c_addr, 0, 1)


def rowwise_cos(A, B, i, j, chunk: int = 1_000_000) -> np.ndarray:
    """cos(A[i[k]], B[j[k]]) for every k (rows L2-normalised)."""
    out = np.zeros(len(i), dtype=np.float32)
    for s in range(0, len(i), chunk):
        e = s + chunk
        out[s:e] = np.asarray(A[i[s:e]].multiply(B[j[s:e]]).sum(axis=1)).ravel()
    return out


def scale_cols(A, qs):
    """Query-side re-weighting of the combined vector: name block * qs[0], address block * qs[1]
    (qs[2] = first address column). Ranking pool records by A' . B then equals ranking by
    (qs0 * w_n * name_cos + qs1 * w_a * addr_cos) / |b|, without a second pool matrix."""
    if qs is None:
        return A
    A = A.copy()
    A.data = A.data * np.where(A.indices >= qs[2], qs[1], qs[0]).astype(A.data.dtype)
    return A


def _topk(A, B, k, thr, qs=None):
    A = scale_cols(A, qs)
    if A.shape[0] == 0 or B.shape[0] == 0:
        e = np.array([], dtype=np.int64)
        return e, e
    k = min(k, B.shape[0])
    if sp_matmul_topn is not None:
        C = sp_matmul_topn(A, B.T.tocsr(), top_n=k, threshold=thr, n_threads=1).tocoo()
        return C.row.astype(np.int64), C.col.astype(np.int64)
    M = (A @ B.T).tocsr()
    rows, cols = [], []
    for r in range(M.shape[0]):
        d = M.data[M.indptr[r]:M.indptr[r + 1]]
        c = M.indices[M.indptr[r]:M.indptr[r + 1]]
        keep = d >= thr
        d, c = d[keep], c[keep]
        if len(d) > k:
            top = np.argpartition(-d, k)[:k]
            c = c[top]
        rows += [r] * len(c)
        cols += list(c)
    return np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)


def encode_keys(x1, x2):
    """Map two object arrays of string keys to shared int64 codes; "" / None -> -1.
    Working on integer codes instead of Python strings keeps blocking memory ~10x lower."""
    x1 = np.asarray(x1, dtype=object)
    x2 = np.asarray(x2, dtype=object)
    codes, _ = pd.factorize(np.concatenate([x1, x2]), use_na_sentinel=True)
    codes = codes.astype(np.int64)
    both = np.concatenate([x1, x2])
    codes[(both == "") | pd.isna(both)] = -1
    return codes[: len(x1)], codes[len(x1):]


def _combine(c1, c2, x1, x2, mask1, mask2):
    """Codes of (key, extra part) for the masked records, re-compressed to 0..n-1."""
    X = int(max(x1.max(initial=-1), x2.max(initial=-1))) + 2
    k1 = c1.copy()
    k2 = c2.copy()
    k1[mask1] = c1[mask1] * X + (x1[mask1] + 1) + (1 << 40)
    k2[mask2] = c2[mask2] * X + (x2[mask2] + 1) + (1 << 40)
    allk = np.concatenate([k1, k2])
    neg = allk < 0
    codes, _ = pd.factorize(allk)
    codes = codes.astype(np.int64)
    codes[neg] = -1
    return codes[: len(k1)], codes[len(k1):]


def _refine(k1, k2, x1, x2, max_part):
    """Split partitions whose candidate side exceeds max_part by appending an extra key part
    (int codes; -1 = no key)."""
    if (k2 >= 0).sum() == 0:
        return k1, k2
    size = np.bincount(k2[k2 >= 0], minlength=int(max(k1.max(initial=-1), k2.max(initial=-1))) + 1)
    big = size > max_part
    if not big.any():
        return k1, k2
    m1 = (k1 >= 0) & big[np.clip(k1, 0, None)]
    m2 = (k2 >= 0) & big[np.clip(k2, 0, None)]
    return _combine(k1, k2, x1, x2, m1, m2)


def _cartesian(ga_c, ga_i, gb_c, gb_j):
    """All (i, j) with equal code; both sides sorted by code. Vectorised."""
    ua, sa, na = np.unique(ga_c, return_index=True, return_counts=True)
    ub, sb, nb = np.unique(gb_c, return_index=True, return_counts=True)
    common, ia, ib = np.intersect1d(ua, ub, assume_unique=True, return_indices=True)
    if len(common) == 0:
        e = np.array([], np.int64)
        return e, e
    na, sa, nb, sb = na[ia], sa[ia], nb[ib], sb[ib]
    tot = na * nb
    # for every output pair: which group, and position inside the group's na x nb grid
    g = np.repeat(np.arange(len(common)), tot)
    off = np.arange(tot.sum()) - np.repeat(np.cumsum(tot) - tot, tot)
    ii = ga_i[sa[g] + off // nb[g]]
    jj = gb_j[sb[g] + off % nb[g]]
    return ii, jj


def knn_partitioned(X1, X2, key1, key2, src2, k, thr, tag, pool=None, small: int = 4000, qs=None,
                    batch_pairs: int = 4_000_000):
    """Top-k per source inside partitions with equal keys (int codes, -1 = none).
    Returns DataFrame[i, j, tag].

    Small partitions (n1 * n2 <= ``small``, the vast majority) are solved vectorised, in batches of
    at most ~``batch_pairs`` enumerated pairs (bounded memory): all within-partition pairs are
    enumerated, scored with a row-wise dot product and reduced to the top-k per S1 record. Large
    partitions use the sparse top-k product (in worker processes when ``pool`` is given).
    """
    key1 = np.asarray(key1, dtype=np.int64)
    key2 = np.asarray(key2, dtype=np.int64)
    nk = int(max(key1.max(initial=-1), key2.max(initial=-1))) + 1
    frames = []
    X1q = scale_cols(X1, qs)
    i_all = np.flatnonzero(key1 >= 0)
    c1 = key1[i_all]
    o = np.argsort(c1, kind="stable")
    i_all, c1 = i_all[o], c1[o]
    n1 = np.bincount(c1, minlength=nk)
    for s in (2, 3):
        j_all = np.flatnonzero((src2 == s) & (key2 >= 0))
        c2 = key2[j_all]
        o = np.argsort(c2, kind="stable")
        j_all, c2 = j_all[o], c2[o]
        n2 = np.bincount(c2, minlength=nk)
        cost = n1 * n2
        is_small = (cost > 0) & (cost <= small)
        is_big = cost > small
        # --- small partitions: vectorised, batched by enumerated-pair count
        sm = np.flatnonzero(is_small)
        if len(sm):
            batch_of = np.full(nk, -1, dtype=np.int64)
            batch_of[sm] = np.cumsum(cost[sm]) // batch_pairs
            b1 = batch_of[c1]
            b2 = batch_of[c2]
            for b in range(int(batch_of[sm].max()) + 1):
                m1 = b1 == b
                m2 = b2 == b
                ii, jj = _cartesian(c1[m1], i_all[m1], c2[m2], j_all[m2])
                if not len(ii):
                    continue
                sc = rowwise_cos(X1q, X2, ii, jj)
                keep = sc >= thr
                ii, jj, sc = ii[keep], jj[keep], sc[keep]
                order = np.lexsort((-sc, ii))
                ii, jj = ii[order], jj[order]
                first = np.r_[True, ii[1:] != ii[:-1]]
                grp_start = np.maximum.accumulate(np.where(first, np.arange(len(ii)), 0))
                sel = (np.arange(len(ii)) - grp_start) < k
                frames.append((ii[sel], jj[sel]))
        # --- big partitions: sparse top-k
        bg = np.flatnonzero(is_big)
        if len(bg):
            s1_start = np.searchsorted(c1, bg)
            s1_end = np.searchsorted(c1, bg, side="right")
            s2_start = np.searchsorted(c2, bg)
            s2_end = np.searchsorted(c2, bg, side="right")
            jobs = [(i_all[a:b_], j_all[c:d], k, thr, qs)
                    for a, b_, c, d in zip(s1_start, s1_end, s2_start, s2_end)]
            if pool is not None:
                frames.append(pool.run(jobs))
            else:
                for i_idx, j_idx, kk, tt, q in jobs:
                    r, c = _topk(X1[i_idx], X2[j_idx], kk, tt, q)
                    if len(r):
                        frames.append((i_idx[r], j_idx[c]))
    ii = np.concatenate([f[0] for f in frames]) if frames else np.array([], np.int64)
    jj = np.concatenate([f[1] for f in frames]) if frames else np.array([], np.int64)
    return pd.DataFrame({"i": ii.astype(np.int32), "j": jj.astype(np.int32), tag: np.int8(1)})


def key_join(k1, k2, max_block, tag):
    """Exact-key join on int codes (-1 = none), skipping keys shared by > max_block pool records."""
    k1 = np.asarray(k1, dtype=np.int64)
    k2 = np.asarray(k2, dtype=np.int64)
    nk = int(max(k1.max(initial=-1), k2.max(initial=-1))) + 1
    n2 = np.bincount(k2[k2 >= 0], minlength=nk)
    ok2 = (k2 >= 0) & (n2[np.clip(k2, 0, None)] <= max_block)
    ok1 = k1 >= 0
    i_all, j_all = np.flatnonzero(ok1), np.flatnonzero(ok2)
    o1 = np.argsort(k1[i_all], kind="stable")
    o2 = np.argsort(k2[j_all], kind="stable")
    i_all, j_all = i_all[o1], j_all[o2]
    ii, jj = _cartesian(k1[i_all], i_all, k2[j_all], j_all)
    return pd.DataFrame({"i": ii.astype(np.int32), "j": jj.astype(np.int32), tag: np.int8(1)})


def multikey_join(l1, l2, max_block, tag):
    """Join on ANY shared key: l1/l2 are lists of int64 keys per record; keys held by more than
    ``max_block`` pool records are skipped (too common to be informative)."""
    r1 = np.repeat(np.arange(len(l1)), [len(x) for x in l1])
    r2 = np.repeat(np.arange(len(l2)), [len(x) for x in l2])
    v1 = np.fromiter((v for x in l1 for v in x), dtype=np.int64, count=len(r1))
    v2 = np.fromiter((v for x in l2 for v in x), dtype=np.int64, count=len(r2))
    codes, _ = pd.factorize(np.concatenate([v1, v2]))
    c1, c2 = codes[: len(v1)].astype(np.int64), codes[len(v1):].astype(np.int64)
    n2 = np.bincount(c2, minlength=int(codes.max(initial=-1)) + 1)
    keep2 = n2[c2] <= max_block
    c2, r2 = c2[keep2], r2[keep2]
    o1 = np.argsort(c1, kind="stable")
    o2 = np.argsort(c2, kind="stable")
    ii, jj = _cartesian(c1[o1], r1[o1], c2[o2], r2[o2])
    m = pd.DataFrame({"i": ii.astype(np.int32), "j": jj.astype(np.int32)}).drop_duplicates()
    m[tag] = np.int8(1)
    return m


def num_bigrams(nums):
    """Consecutive number pairs of an address ("16 11 17" -> {16:11, 11:17}) as int64 keys."""
    out = []
    for s in nums:
        t = [int(x[-9:]) for x in s.split()] if s else []
        out.append(list({a * 1_000_000_000 + b for a, b in zip(t, t[1:])}))
    return out


def cap_score(p: pd.DataFrame) -> np.ndarray:
    """Score used to cap raw pairs per (S1, source): the better of the name-led combined cosine and an
    address-led cosine, plus a small bonus per generator that found the pair. Keeps namesake-heavy
    S1 entities from losing their true (address-matching) candidate to the cap."""
    ah = (0.5 * p["c_name"].to_numpy() + 1.0 * p["c_addr"].to_numpy()) / np.sqrt(1.25)
    ng = p[[c for c in GEN_COLS if c in p]].to_numpy().sum(axis=1)
    return (np.maximum(p["c_comb"].to_numpy(), ah) + 0.03 * ng).astype(np.float32)


def _pref(arr, n):
    return np.array([x[:n] for x in arr], dtype=object)


def _rarest_addr(a1, a2, min_len: int = 5):
    """For every record, its rarest address word (length >= min_len, seen at least twice)."""
    from collections import Counter
    cnt = Counter()
    for arr in (a1, a2):
        for a in arr:
            cnt.update(set(t for t in a.split() if len(t) >= min_len))

    def pick(a):
        best, bc = "", None
        for t in a.split():
            if len(t) >= min_len:
                c = cnt[t]
                if c >= 2 and (bc is None or c < bc):
                    best, bc = t, c
        return best
    return (np.array([pick(a) for a in a1], dtype=object), np.array([pick(a) for a in a2], dtype=object))


def _longest(arr):
    return np.array([max(t.split(), key=len)[:3] if t else "" for t in arr], dtype=object)


GEN_COLS = ["g_n1", "g_n2", "g_n3", "g_n4", "g_h", "g_s", "g_a1", "g_ra", "g_ah", "g_nb", "g_rv"]


def _pair_codes(a1, a2, b1, b2):
    """Codes of the key pair (a, b); -1 when either part is missing."""
    X = int(max(b1.max(initial=-1), b2.max(initial=-1))) + 1
    k1 = np.where((a1 >= 0) & (b1 >= 0), a1 * X + b1, -1)
    k2 = np.where((a2 >= 0) & (b2 >= 0), a2 * X + b2, -1)
    allk = np.concatenate([k1, k2])
    codes, _ = pd.factorize(allk)
    codes = codes.astype(np.int64)
    codes[allk < 0] = -1
    return codes[: len(k1)], codes[len(k1):]


def union_parts(parts, n_gen_cols=GEN_COLS) -> pd.DataFrame:
    """Union of generator outputs -> one row per (i, j) with a 0/1 column per generator.
    Sort-based on an int64 pair key with a generator bitmask (low memory, no pandas groupby)."""
    keys, bits = [], []
    for p in parts:
        tag = [c for c in p.columns if c.startswith("g_")][0]
        keys.append(p["i"].to_numpy(np.int64) << 32 | p["j"].to_numpy(np.int64))
        bits.append(np.full(len(p), 1 << n_gen_cols.index(tag), dtype=np.int32))
    key = np.concatenate(keys) if keys else np.array([], np.int64)
    bit = np.concatenate(bits) if bits else np.array([], np.int32)
    del keys, bits
    o = np.argsort(key, kind="stable")
    key, bit = key[o], bit[o]
    del o
    first = np.r_[True, key[1:] != key[:-1]] if len(key) else np.array([], bool)
    starts = np.flatnonzero(first)
    ukey = key[starts]
    ubit = np.bitwise_or.reduceat(bit, starts) if len(starts) else bit
    out = pd.DataFrame({"i": (ukey >> 32).astype(np.int32), "j": (ukey & 0xFFFFFFFF).astype(np.int32)})
    for b, c in enumerate(n_gen_cols):
        out[c] = ((ubit >> b) & 1).astype(np.int8)
    return out


def generate_candidates(s1n: pd.DataFrame, cn: pd.DataFrame, vecs: dict, cfg, pool=None) -> pd.DataFrame:
    """All generators for ONE country's records. Returns DataFrame[i, j, g_*] (deduplicated).

    Every partition key is turned into an integer code as soon as it is built (string key
    arrays for millions of pool records would otherwise dominate memory)."""
    X1, X2 = vecs["comb"]
    src = cn["src"].to_numpy()

    def col(df, c):
        return df[c].to_numpy(dtype=object)

    def enc(f, c):
        """codes of f(value) for column c on both sides."""
        return encode_keys(np.array([f(x) for x in col(s1n, c)], dtype=object),
                           np.array([f(x) for x in col(cn, c)], dtype=object))

    nosp = lambda n: (lambda x: x.replace(" ", "")[:n])  # noqa: E731
    first_w = lambda x: x.split()[0][:4] if x else ""  # noqa: E731
    second_w = lambda x: x.split()[1][:4] if x.count(" ") >= 1 else ""  # noqa: E731
    ns5, ns8, ns2, ns4 = enc(nosp(5), "core"), enc(nosp(8), "core"), enc(nosp(2), "core"), enc(nosp(4), "core")
    st = enc(first_w, "addr_alpha")
    sb = enc(second_w, "addr_alpha")
    h = enc(lambda x: x, "house")
    hp = enc(lambda x: x[:2], "house")
    stt = enc(lambda x: x, "state")
    name_ref = [ns5, ns8, stt]
    addr_ref = [st, ns2, ns4]
    K, T, MP = cfg.knn_k_per_source, cfg.knn_min_sim, cfg.knn_max_partition

    def run(tag, keys, ref, k=K, thr=T, qs=None):
        k1, k2 = keys
        for x1, x2 in ref:
            k1, k2 = _refine(k1, k2, x1, x2, MP)
        return knn_partitioned(X1, X2, k1, k2, src, k, thr, tag, pool, qs=qs)

    t0 = time.time()
    p3 = enc(lambda x: x[:3], "core")
    parts = [
        run("g_n1", p3, name_ref),
        run("g_n2", enc(lambda t: max(t.split(), key=len)[:3] if t else "", "core"), name_ref),
        run("g_n3", enc(lambda x: x[-3:], "core"), name_ref),
        run("g_n4", enc(nosp(4), "core_phon"), name_ref),
        run("g_h", h, addr_ref, k=cfg.addr_k_per_source, thr=0.0),
        run("g_s", st, [sb, hp] + addr_ref[1:], k=cfg.addr_k_per_source, thr=0.0),
    ]
    # exact house number + first street word
    parts.append(key_join(*_pair_codes(h[0], h[1], st[0], st[1]), cfg.addr_max_block, "g_a1"))
    # rarest address word (brand-new names at a known address): partition + top-k by the combined vector
    ra = encode_keys(*_rarest_addr(col(s1n, "addr_alpha"), col(cn, "addr_alpha")))
    parts.append(run("g_ra", ra, [h] + addr_ref[1:], k=cfg.addr_k_per_source, thr=0.0))
    if getattr(cfg, "gen_addr_heavy", False):
        # generic names ("Hyderabad Services", "Aditya Consultancy") fill the name top-k with namesakes;
        # re-rank the SAME name partitions by an address-heavy score to recover the right one
        qs = (cfg.ah_scale_name, cfg.ah_scale_addr, vecs["split"])
        parts.append(run("g_ah", p3, name_ref, k=cfg.ah_k_per_source, thr=0.0, qs=qs))
    if getattr(cfg, "gen_numgram", False):
        parts.append(multikey_join(num_bigrams(col(s1n, "nums")), num_bigrams(col(cn, "nums")),
                                   cfg.numgram_max_block, "g_nb"))
    mode = getattr(cfg, "gen_reverse", "off")
    if mode != "off":
        # REVERSE direction: every (address-less) pool record looks for its best S1 names. Address-less
        # records are always true copies of some S1 (100% in train) but lose the S1-side top-k race to
        # namesakes that also match on address.
        sel = (np.flatnonzero(cn["addr_clean"].to_numpy(dtype=object) == "") if mode == "empty"
               else np.arange(len(cn)))
        if len(sel):
            k1r, k2r = p3[1][sel], p3[0]
            for x1, x2 in name_ref:
                k1r, k2r = _refine(k1r, k2r, x2[sel], x1, MP)
            rv = knn_partitioned(X2[sel], X1, k1r, k2r, np.full(len(s1n), 2, np.int8), cfg.rev_k, T,
                                 "g_rv", None, qs=(1.0, 0.0, vecs["split"]))
            parts.append(pd.DataFrame({"i": rv["j"].to_numpy(), "j": sel[rv["i"].to_numpy()].astype(np.int32),
                                       "g_rv": np.int8(1)}))
    log(f"      generators: {sum(len(p) for p in parts):,} raw pairs in {time.time() - t0:.0f}s")
    return union_parts(parts)
