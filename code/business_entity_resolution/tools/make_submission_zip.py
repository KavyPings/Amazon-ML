"""Build <team_name>_submission.zip exactly in the structure the organisers require:

<team_name>_submission.zip
├── output/matching_results.tsv, output/candidate_pairs.tsv
├── code/business_entity_resolution/  (src/, README.md, requirements.txt, run.py, utils/, tools/, tests/)
└── Documentation_template.md

    python tools/make_submission_zip.py --team TEAM_NAME --doc ../../documentation/Documentation_template.md
"""
import argparse
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
SKIP_DIRS = {".venv", "__pycache__", ".git", "synthetic_dataset", ".pytest_cache", "out"}
SKIP_DIR_PREFIX = ("artifacts", "output")        # artifacts*/ output*/ (run folders, backups)
SKIP_FILES_SUFFIX = (".pyc", ".zip", ".log", ".parquet", ".pkl")


def _skip_file(fn):
    return (fn.endswith(SKIP_FILES_SUFFIX) or fn.startswith("status") or fn.endswith("_log.txt")
            or fn in ("probe.txt", "freeze.txt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--out-dir", default=os.path.join(REPO, "output"), help="folder holding the two TSVs to submit")
    ap.add_argument("--doc", default=os.path.join(REPO, "..", "..", "documentation", "Documentation_template.md"))
    ap.add_argument("--dest", default=os.path.join(REPO, "..", ".."))
    a = ap.parse_args()

    need = [os.path.join(a.out_dir, f) for f in ("matching_results.tsv", "candidate_pairs.tsv")]
    for p in need + [a.doc]:
        if not os.path.isfile(p):
            raise SystemExit(f"missing required file: {p}")
    zpath = os.path.abspath(os.path.join(a.dest, f"{a.team}_submission.zip"))
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in need:
            z.write(p, f"output/{os.path.basename(p)}")
        for root, dirs, files in os.walk(REPO):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(SKIP_DIR_PREFIX)]
            for fn in files:
                if _skip_file(fn):
                    continue
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, REPO).replace(os.sep, "/")
                z.write(full, f"code/business_entity_resolution/{rel}")
        z.write(a.doc, "Documentation_template.md")
    print("wrote", zpath)
    with zipfile.ZipFile(zpath) as z:
        for n in sorted(z.namelist()):
            print("  ", n)


if __name__ == "__main__":
    main()
