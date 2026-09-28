"""Smoke test: synthetic data -> full pipeline -> validator must PASS.

    python tests/test_smoke.py        (takes ~20 seconds)
"""
import os
import subprocess
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        data = os.path.join(tmp, "dataset")
        subprocess.check_call([sys.executable, os.path.join(REPO, "tools", "make_synthetic_data.py"),
                               "--out", data, "--n-train", "3000", "--n-test", "1500"])
        rc = subprocess.call([sys.executable, os.path.join(REPO, "run.py"), "all", "--data-dir", data,
                              "--work-dir", os.path.join(tmp, "art"), "--out-dir", os.path.join(tmp, "out"),
                              "--set", "abbrev_min_count=5"])
        assert rc == 0, "pipeline or validator failed"
        for f in ("matching_results.tsv", "candidate_pairs.tsv"):
            assert os.path.isfile(os.path.join(tmp, "out", f))
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    test_end_to_end()
