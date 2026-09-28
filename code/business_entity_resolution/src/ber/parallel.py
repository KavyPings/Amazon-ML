"""Process-parallel top-k search over blocking partitions with ZERO-copy shared CSR matrices.

The combined TF-IDF matrices are placed once in shared memory; worker processes attach to
them and run the sparse top-k product for batches of partitions. This makes blocking scale
with the number of CPU cores without duplicating gigabytes of vectors per worker.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from multiprocessing import shared_memory

_W = {}


def _to_shm(arr: np.ndarray):
    shm = shared_memory.SharedMemory(create=True, size=max(1, arr.nbytes))
    view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    view[:] = arr
    return shm, (shm.name, arr.shape, arr.dtype.str)


def share_csr(M: sp.csr_matrix):
    """Copy a CSR matrix into shared memory. Returns (handles to keep alive, spec for workers)."""
    shms, specs = [], []
    for arr in (M.data, M.indices, M.indptr):
        shm, spec = _to_shm(np.ascontiguousarray(arr))
        shms.append(shm)
        specs.append(spec)
    return shms, (specs, M.shape)


def _attach(spec):
    specs, shape = spec
    arrs, shms = [], []
    for name, shp, dt in specs:
        shm = shared_memory.SharedMemory(name=name)
        shms.append(shm)
        arrs.append(np.ndarray(shp, dtype=np.dtype(dt), buffer=shm.buf))
    return sp.csr_matrix((arrs[0], arrs[1], arrs[2]), shape=shape, copy=False), shms


def _init(spec1, spec2):
    X1, s1 = _attach(spec1)
    X2, s2 = _attach(spec2)
    _W["X1"], _W["X2"], _W["keep"] = X1, X2, s1 + s2


def _work(batch):
    from .blocking import _topk
    X1, X2 = _W["X1"], _W["X2"]
    out_i, out_j = [], []
    for i_idx, j_idx, k, thr, qs in batch:
        r, c = _topk(X1[i_idx], X2[j_idx], k, thr, qs)
        if len(r):
            out_i.append(i_idx[r])
            out_j.append(j_idx[c])
    if not out_i:
        return np.array([], np.int64), np.array([], np.int64)
    return np.concatenate(out_i), np.concatenate(out_j)


class TopkPool:
    """Context manager: ``with TopkPool(X1, X2, n) as pool: pool.run(jobs)``."""

    def __init__(self, X1, X2, n_workers):
        self.X1, self.X2, self.n = X1, X2, n_workers
        self.offset = 0      # S1 row offset of the current chunk (jobs use chunk-local S1 indices)

    def __enter__(self):
        from concurrent.futures import ProcessPoolExecutor
        self.h1, spec1 = share_csr(self.X1)
        self.h2, spec2 = share_csr(self.X2)
        self.ex = ProcessPoolExecutor(max_workers=self.n, initializer=_init, initargs=(spec1, spec2))
        return self

    def run(self, jobs, batch_cost: float = 2e7):
        """jobs: list of (i_idx, j_idx, k, thr[, qs]). Batches are formed by estimated cost."""
        batches, cur, cost = [], [], 0.0
        off = self.offset
        for jb in jobs:
            jb = (jb[0] + off, jb[1], jb[2], jb[3], jb[4] if len(jb) > 4 else None)
            cur.append(jb)
            cost += len(jb[0]) * len(jb[1])
            if cost >= batch_cost:
                batches.append(cur)
                cur, cost = [], 0.0
        if cur:
            batches.append(cur)
        ii, jj = [], []
        for a, b in self.ex.map(_work, batches):
            if len(a):
                ii.append(a)
                jj.append(b)
        if not ii:
            return np.array([], np.int64), np.array([], np.int64)
        return np.concatenate(ii) - off, np.concatenate(jj)

    def __exit__(self, *exc):
        self.ex.shutdown(wait=True)
        for h in self.h1 + self.h2:
            h.close()
            h.unlink()
        return False
