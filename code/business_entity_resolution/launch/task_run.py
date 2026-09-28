"""Full run on the laptop: train on a closed state subset of the full training data, predict each
country in its own process (bounded memory), merge into output_<name>/ and validate.

    python launch/task_run.py --name v4 --frac 0.40 [--seed 7] [--jobs 10] [--set KEY=VALUE ...]

Every step is a separate Python process so memory is fully released in between. Steps whose
outputs already exist are skipped, so a stopped run can simply be started again.
Afterwards: double-click promote.bat (or run promote.py --out output_<name> --art artifacts_<name>).
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PY = sys.executable
RUN = os.path.join(ROOT, "run.py")
MAPS = os.path.join(HERE, "out", "big", "maps_full2.pkl")


def step(args):
    t = time.time()
    print(f"==> {' '.join(args)}", flush=True)
    rc = subprocess.call([PY, "-u", RUN] + args)
    print(f"<== rc={rc} in {(time.time() - t) / 60:.1f} min", flush=True)
    if rc != 0:
        sys.exit(rc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--frac", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--jobs", default=os.environ.get("BER_JOBS", "10"))
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    work = os.path.join(ROOT, f"artifacts_{a.name}")
    out = os.path.join(ROOT, f"output_{a.name}")
    maps = [f"maps_path={MAPS}"] if os.path.exists(MAPS) else []
    if not os.path.exists(os.path.join(work, "matcherB.txt")):
        step(["train", "--work-dir", work, "--set", f"train_state_frac={a.frac}", f"seed={a.seed}",
              f"n_jobs={a.jobs}"] + maps + a.set)
    for c in ("france", "india", "us"):
        if not os.path.exists(os.path.join(work, f"part_{c}.pkl")):
            step(["predict-country", "--country", c, "--work-dir", work, "--set", f"n_jobs={a.jobs}"])
    step(["merge", "--work-dir", work, "--out-dir", out])
    print("RUN COMPLETE", flush=True)
