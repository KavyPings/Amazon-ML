"""Snapshot the current outputs + reports as a named version (the guidelines ask for
a version history of every submission).

    python tools/save_version.py v2_full_run --note "full train, max_candidates=8"

Creates  submissions/<name>/  with both TSVs, the train/predict reports, the config
and a notes.txt line, and appends a row to submissions/LOG.md.
"""
import argparse
import datetime
import json
import os
import shutil

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--note", default="")
    ap.add_argument("--dest", default=os.path.join(REPO, "..", "..", "submissions"))
    ap.add_argument("--out-dir", default="output", help="folder (relative to the repo) holding the two TSVs")
    ap.add_argument("--work-dir", default="artifacts", help="folder (relative to the repo) holding the reports")
    a = ap.parse_args()
    dst = os.path.join(a.dest, a.name)
    os.makedirs(dst, exist_ok=True)
    files = [f"{a.out_dir}/matching_results.tsv", f"{a.out_dir}/candidate_pairs.tsv"] + [
        f"{a.work_dir}/{n}" for n in ("train_report.json", "predict_summary.json", "config.json", "params.json")]
    for f in files:
        p = os.path.join(REPO, f)
        if os.path.exists(p):
            shutil.copy2(p, dst)
    val = cand = "?"
    try:
        with open(os.path.join(REPO, a.work_dir, "train_report.json")) as f:
            r = json.load(f)
        val = f"{r['validation']['macro_f05']:.4f}"
        cand = f"{r['pruned_candidates_val']['mean_candidates_per_s1']:.2f}"
    except Exception:
        pass
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(os.path.join(dst, "notes.txt"), "w") as f:
        f.write(f"{stamp}\nval_macro_f05={val}\nmean_candidates_per_s1={cand}\n{a.note}\n")
    log = os.path.join(a.dest, "LOG.md")
    new = not os.path.exists(log)
    with open(log, "a") as f:
        if new:
            f.write("| time | version | val macro F0.5 | cand/S1 | public LB | note |\n|---|---|---|---|---|---|\n")
        f.write(f"| {stamp} | {a.name} | {val} | {cand} |  | {a.note} |\n")
    print(f"saved {dst}  (val F0.5 {val}, candidates/S1 {cand})")


if __name__ == "__main__":
    main()
