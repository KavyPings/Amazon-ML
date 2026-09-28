"""Reading the challenge TSV files and writing the two submission files."""
from __future__ import annotations

import csv
import os
import time

import pandas as pd

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def log(msg: str) -> None:
    """Print a timestamped progress line (flushes immediately)."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _count_lines(path: str) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1


def read_tsv(path: str) -> pd.DataFrame:
    """Read a challenge TSV as all-strings, empty cells kept as "" (never NaN).

    Tries normal CSV quoting first; if the row count does not match the file's
    line count (a stray quote swallowing lines), re-reads with quoting disabled.
    """
    n_lines = _count_lines(path)
    try:   # Arrow-backed strings: ~5x less memory than Python objects for millions of rows
        df = pd.read_csv(path, sep="\t", engine="pyarrow", dtype="string[pyarrow]", keep_default_na=False,
                         na_filter=False, encoding="utf-8")
        df = df.fillna("")
        if len(df) == n_lines:
            df.columns = [c.strip() for c in df.columns]
            return df
    except Exception:
        pass
    kw = dict(sep="\t", dtype=str, keep_default_na=False, na_filter=False, encoding="utf-8")
    df = pd.read_csv(path, **kw)
    if len(df) != n_lines:
        df2 = pd.read_csv(path, quoting=csv.QUOTE_NONE, **kw)
        if abs(len(df2) - n_lines) < abs(len(df) - n_lines):
            df = df2
        log(f"WARNING {os.path.basename(path)}: {len(df)} rows vs {n_lines} lines")
    df.columns = [c.strip() for c in df.columns]
    return df


def load_split(data_dir: str, split: str) -> dict:
    """Load ``{split}_source1/2/3.tsv`` (+ ground truth for train) from ``data_dir/split``."""
    base = os.path.join(data_dir, split)
    out = {}
    for s in (1, 2, 3):
        p = os.path.join(base, f"{split}_source{s}.tsv")
        df = read_tsv(p)
        missing = [c for c in SOURCE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{p} is missing columns {missing}; found {list(df.columns)}")
        out[f"s{s}"] = df[SOURCE_COLS].copy()
        log(f"loaded {p}: {len(df):,} rows")
    gt_path = os.path.join(base, f"{split}_ground_truth.tsv")
    if os.path.exists(gt_path):
        out["gt"] = read_tsv(gt_path)
        log(f"loaded {gt_path}: {len(out['gt']):,} rows")
    return out


def gt_to_dict(gt: pd.DataFrame) -> dict:
    """Ground truth -> {S1 id: set(matched S2/S3 ids)} (empty set for singletons)."""
    col = "matched_entity_ids"
    out = {}
    for s1, m in zip(gt["source1_entity_id"], gt[col]):
        m = m.strip() if isinstance(m, str) else ""
        out[s1.strip()] = set(x.strip() for x in m.split(",") if x.strip()) if m else set()
    return out


def find_data_dir(explicit: str | None) -> str:
    """Locate the folder that holds ``train/`` and ``test/``.

    Checks the explicit path first, then common locations relative to this repo
    so teammates can run without passing ``--data-dir``.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [explicit] if explicit else []
    for up in range(0, 6):
        root = os.path.abspath(os.path.join(here, *([".."] * up)))
        candidates += [
            os.path.join(root, "dataset"),
            os.path.join(root, "data", "archive", "dataset"),
            os.path.join(root, "data", "dataset"),
        ]
    for c in candidates:
        if c and (os.path.isdir(os.path.join(c, "train")) or os.path.isdir(os.path.join(c, "test"))):
            return os.path.abspath(c)
    raise FileNotFoundError(
        "Could not find the dataset folder (with train/ and test/). "
        "Pass --data-dir path/to/dataset"
    )


def write_id_lists(path: str, header2: str, s1_ids, mapping: dict) -> None:
    """Write one row per S1 id: ``source1_entity_id<TAB>id1,id2,...`` (empty = none).

    IDs inside a list are de-duplicated and sorted; only S2-/S3- ids are written.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{header2}\n")
        seen = set()
        for s1 in s1_ids:
            if s1 in seen:
                continue
            seen.add(s1)
            ids = sorted({i for i in mapping.get(s1, ()) if i.startswith(("S2-", "S3-"))})
            f.write(f"{s1}\t{','.join(ids)}\n")
