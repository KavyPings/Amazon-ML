"""v3 run: train on a closed ~22% state subset of the FULL training data (new maps), then
predict each country in its own process (bounded memory) and merge into output_v3/.
Every step is a separate Python process so memory is fully released in between."""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PY = sys.executable
RUN = os.path.join(ROOT, "run.py")
WORK = os.path.join(ROOT, "artifacts_v3")
OUT = os.path.join(ROOT, "output_v3")
MAPS = os.path.join(HERE, "out", "big", "maps_full2.pkl")
JOBS = os.environ.get("BER_JOBS", "10")


def step(args):
    t = time.time()
    print(f"==> {' '.join(args)}", flush=True)
    rc = subprocess.call([PY, "-u", RUN] + args)
    print(f"<== rc={rc} in {(time.time() - t) / 60:.1f} min", flush=True)
    if rc != 0:
        sys.exit(rc)


if __name__ == "__main__":
    if not os.path.exists(os.path.join(WORK, "matcherB.txt")):
        step(["train", "--work-dir", WORK, "--set", "train_state_frac=0.22", "seed=7", f"maps_path={MAPS}",
              f"n_jobs={JOBS}"])
    for c in ("france", "india", "us"):
        if not os.path.exists(os.path.join(WORK, f"part_{c}.pkl")):
            step(["predict-country", "--country", c, "--work-dir", WORK, "--set", f"n_jobs={JOBS}"])
    step(["merge", "--work-dir", WORK, "--out-dir", OUT])
