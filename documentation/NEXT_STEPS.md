# Next steps: run, submit, package

The current state and the prioritised improvement plan are in **[IMPROVEMENT_ROADMAP.md](IMPROVEMENT_ROADMAP.md)**.

## Submittable output (always kept valid)

- `code/business_entity_resolution/output/matching_results.tsv`: the leaderboard file
- `code/business_entity_resolution/output/candidate_pairs.tsv`: the candidate set, required in the final zip
- Both pass `utils/validate_submission.py`. Every prediction run and every quick rebuild re-runs the validator.

## On the laptop (double-click, no typing)

All scripts are in `code/business_entity_resolution/launch/`.

1. `setup.bat` (once): creates the Python 3.12 environment in `%USERPROFILE%\ber_venv312`.
2. `go_v6.bat` (**next run to do**): runs the full pipeline and writes `output_v6/`.
    - Steps: two-phase training on a closed 50% state subset → predict France, India and US, each in its own process → merge → validator.
    - Takes about 2 h. Close other apps first.
    - Logs are in `task_v6.log` and `status_v6.txt`. If you restart it, it resumes where it stopped.
3. `promote.bat`: validates the run named by `set RUN=v6` inside it and copies it into `output/` and `artifacts/`. Only promote a run whose validation score beats the current one (v4: 0.9883).
    - The previous submission is kept as `output_prev/`.
    - A snapshot is written to `submissions/vN` and logged in `submissions/LOG.md`.
4. `quick_submit.bat`: rebuilds a valid `output/` from whatever per-country parts exist. Missing countries get empty rows.
5. `stop_runs.bat`: stops a running task.

To start a new experiment, copy `go_v6.bat` to `go_v7.bat`, then change `--name v7` and optionally `--frac` or `--set KEY=VALUE`.

## From a terminal (any OS)

```bash
cd code/business_entity_resolution
python launch/task_run.py --name v6 --frac 0.3                                     # the whole run (what go_v5.bat does)
python run.py train --set train_state_frac=0.3 seed=7 maps_path=launch/out/big/maps_full2.pkl   # train only
python run.py predict --set n_jobs=8                                               # all countries, then merge + validate
python run.py predict-country --country india --set n_jobs=8                       # one country per process (less RAM)
python run.py merge                                                                # combine per-country parts + validate
python tools/save_version.py v3 --note "..."                                       # version history
python tools/make_submission_zip.py --team TEAM_NAME                               # final package
```

## Final package checklist

- [ ] `output/` holds the latest validated `matching_results.tsv` and `candidate_pairs.tsv`
- [ ] Every `<fill>` in `documentation/Documentation_template.md` is replaced with numbers from `artifacts/train_report.json` and `artifacts/predict_summary.json`
- [ ] `python tools/make_submission_zip.py --team TEAM_NAME` has been run and the zip opened to check it
- [ ] Machine requirements: 16 GB RAM (close other apps), Python 3.12, about 1 hour for a full prediction on an 8-core laptop
