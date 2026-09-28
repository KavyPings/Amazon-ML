"""Command line entry point:  python run.py <command> [options]

Commands
  profile   print row counts, countries, singleton rate (no training)
  train     fit the pipeline on dataset/train, tune on a 20% hold-out, save to --work-dir
  predict   run the saved pipeline on dataset/test, write output/*.tsv
  all       train + predict + validate (the full reproducible run)
  validate  run utils/validate_submission.py on the output folder
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from .config import Config
from .io import find_data_dir, gt_to_dict, load_split, log

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _parse_set(items):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k.strip()] = v.strip()
    return out


def cmd_profile(a):
    data = load_split(a.data_dir, "train")
    gt = gt_to_dict(data["gt"])
    n = [len(v) for v in gt.values()]
    import numpy as np
    print("S1/S2/S3 rows:", len(data["s1"]), len(data["s2"]), len(data["s3"]))
    print("countries S1:", data["s1"]["country"].value_counts().to_dict())
    print("singleton rate:", round(float(np.mean([x == 0 for x in n])), 4))
    print("mean matches per S1:", round(float(np.mean(n)), 3), " max:", max(n))
    print(data["s1"].sample(5, random_state=0).to_string())
    print(data["s2"].sample(5, random_state=0).to_string())


def cmd_validate(a):
    script = os.path.join(ROOT, "utils", "validate_submission.py")
    args = [sys.executable, script, "--matching", os.path.join(a.out_dir, "matching_results.tsv"),
            "--candidate", os.path.join(a.out_dir, "candidate_pairs.tsv"),
            "--test-dir", os.path.join(a.data_dir, "test")]
    log("running validator: " + " ".join(args))
    return subprocess.call(args)


def main(argv=None):
    p = argparse.ArgumentParser(prog="ber", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["profile", "train", "predict", "predict-country", "merge", "all", "validate"])
    p.add_argument("--country", default=None, help="for predict-country: france / india / us ...")
    p.add_argument("--data-dir", default=None, help="folder containing train/ and test/")
    p.add_argument("--work-dir", default=os.path.join(ROOT, "artifacts"))
    p.add_argument("--out-dir", default=os.path.join(ROOT, "output"))
    p.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="override config values, e.g. --set max_candidates=6")
    a = p.parse_args(argv)
    a.data_dir = find_data_dir(a.data_dir)
    log(f"data dir: {a.data_dir}")

    from . import pipeline  # heavy imports only when needed
    if a.command == "profile":
        return cmd_profile(a)
    if a.command == "validate":
        return cmd_validate(a)
    if a.command == "predict-country":
        cfg = Config.load(os.path.join(a.work_dir, "config.json")).update(_parse_set(a.set))
        pipeline.predict_country(a.data_dir, a.work_dir, a.country, cfg)
        return 0
    if a.command == "merge":
        pipeline.merge_parts(a.data_dir, a.work_dir, a.out_dir)
        return cmd_validate(a)
    if a.command in ("train", "all"):
        cfg = Config().update(_parse_set(a.set))
        pipeline.train(a.data_dir, a.work_dir, cfg)
    if a.command in ("predict", "all"):
        cfg = Config.load(os.path.join(a.work_dir, "config.json")).update(_parse_set(a.set))
        pipeline.predict(a.data_dir, a.work_dir, a.out_dir, cfg)
        return cmd_validate(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
