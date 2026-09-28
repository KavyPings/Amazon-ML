"""Build a valid submission NOW from whatever a (possibly still running) prediction has written.

    python launch/quick_submit.py [--work-dir artifacts] [--out output]

v3+: merges the per-country parts (<work-dir>/part_<country>.pkl). Countries without a part get
empty lists, so the result is always a complete, submittable pair of files. Falls back to the
older per-country score files (test_scores_<country>.parquet) when no parts exist.
"""
import argparse
import glob
import os
import subprocess
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ber import decode  # noqa: E402
from ber.config import Config  # noqa: E402
from ber.io import find_data_dir, write_id_lists  # noqa: E402
from ber.pipeline import merge_parts  # noqa: E402


def legacy(D, art, out):
    cfg = Config.load(os.path.join(art, "config.json"))
    s1 = pd.read_csv(os.path.join(D, "test", "test_source1.tsv"), sep="\t", usecols=["entity_id"],
                     dtype=str, keep_default_na=False)
    ids = s1["entity_id"].tolist()
    cands, matches = {}, {}
    for f in sorted(glob.glob(os.path.join(art, "test_scores_*.parquet"))):
        df = pd.read_parquet(f, columns=["s1", "c", "p"])
        df["i"] = df["s1"].astype("category").cat.codes
        df["j"] = df["c"].astype("category").cat.codes
        sel = decode.apply_decoding(df[["i", "j", "p", "s1", "c"]], cfg)
        for a, grp in df.groupby("s1")["c"]:
            cands[a] = set(grp)
        for a, grp in sel.groupby("s1")["c"]:
            matches[a] = set(grp) & cands[a]
        print(f"{os.path.basename(f)}: {df['s1'].nunique():,} S1 with candidates, {len(sel):,} matches", flush=True)
    os.makedirs(out, exist_ok=True)
    write_id_lists(os.path.join(out, "candidate_pairs.tsv"), "candidate_entity_ids", ids, cands)
    write_id_lists(os.path.join(out, "matching_results.tsv"), "matched_entity_ids", ids, matches)


def validate(D, out):
    return subprocess.call([sys.executable, os.path.join(ROOT, "utils", "validate_submission.py"),
                            "--matching", os.path.join(out, "matching_results.tsv"),
                            "--candidate", os.path.join(out, "candidate_pairs.tsv"),
                            "--test-dir", os.path.join(D, "test")])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default="artifacts")
    ap.add_argument("--out", default="output")
    a = ap.parse_args()
    D = find_data_dir(None)
    art, out = os.path.join(ROOT, a.work_dir), os.path.join(ROOT, a.out)
    if glob.glob(os.path.join(art, "part_*.pkl")):
        merge_parts(D, art, out)          # every test S1 gets a row; missing countries -> empty lists
    else:
        legacy(D, art, out)
    print("written", flush=True)
    sys.exit(validate(D, out))
