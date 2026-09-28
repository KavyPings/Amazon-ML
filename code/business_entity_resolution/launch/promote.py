"""Promote a finished run to the submission folders (no typing needed: double-click promote.bat).

    python launch/promote.py --out output_v3 --art artifacts_v3 --name v3_22pct --note "..."

1. checks the run's two TSVs with the official validator,
2. keeps the current output/ and artifacts/ as output_prev/ and artifacts_prev/,
3. copies the run into output/ (the files that get submitted) and artifacts/ (the models that made them),
4. snapshots it with tools/save_version.py.
"""
import argparse
import os
import shutil
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def replace_contents(src, dst):
    """Make dst an exact copy of src by replacing its CONTENTS: OneDrive / Explorer may lock the
    folder itself, so the folder is never deleted."""
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(dst):
        p = os.path.join(dst, f)
        if os.path.isfile(p):
            os.remove(p)
        else:
            shutil.rmtree(p, ignore_errors=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output_v3")
    ap.add_argument("--art", default="artifacts_v3")
    ap.add_argument("--name", default="v3_22pct")
    ap.add_argument("--note", default="")
    ap.add_argument("--test-dir", default=None)
    ap.add_argument("--latest", action="store_true",
                    help="promote the highest-numbered finished output_vN/ (with artifacts_vN/)")
    a = ap.parse_args()
    if a.latest:
        import re
        done = []
        for d in os.listdir(REPO):
            m = re.fullmatch(r"output_v(\d+)", d)
            if m and all(os.path.exists(os.path.join(REPO, d, f))
                         for f in ("matching_results.tsv", "candidate_pairs.tsv")):
                done.append(int(m.group(1)))
        if not done:
            sys.exit("no finished output_vN/ folder found")
        v = max(done)
        a.out, a.art, a.name = f"output_v{v}", f"artifacts_v{v}", a.name if a.name != "v3_22pct" else f"v{v}"
        print(f"latest finished run: v{v}")
    src_out, src_art = os.path.join(REPO, a.out), os.path.join(REPO, a.art)
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        if not os.path.exists(os.path.join(src_out, f)):
            sys.exit(f"missing {a.out}/{f}: run not finished")
    sys.path.insert(0, os.path.join(REPO, "src"))
    from ber.io import find_data_dir
    test_dir = a.test_dir or os.path.join(find_data_dir(None), "test")
    rc = subprocess.call([sys.executable, os.path.join(REPO, "utils", "validate_submission.py"),
                          "--matching", os.path.join(src_out, "matching_results.tsv"),
                          "--candidate", os.path.join(src_out, "candidate_pairs.tsv"), "--test-dir", test_dir])
    if rc != 0:
        sys.exit(f"validator failed (rc={rc}); nothing promoted")
    for cur, src in (("output", src_out), ("artifacts", src_art)):
        dst = os.path.join(REPO, cur)
        prev = os.path.join(REPO, cur + "_prev")
        if os.path.exists(dst) and os.listdir(dst):
            replace_contents(dst, prev)
        replace_contents(src, dst)
        print(f"promoted {src} -> {dst} (previous kept in {prev})")
    subprocess.call([sys.executable, os.path.join(REPO, "tools", "save_version.py"), a.name, "--note", a.note])
    print("PROMOTED OK")


if __name__ == "__main__":
    main()
