# Business Entity Resolution — Amazon ML Challenge 2026

Given business records from three sources (S1 = deduplicated reference, S2/S3 = noisy
vendor copies), find every S2/S3 record that describes the same real business as each S1
entity. Scored by per-entity **macro F0.5** (precision-weighted, singletons count).

The pipeline is **scalable** (no all-pairs comparison), keeps **very small candidate
sets** per S1 entity, and uses only the provided data and MIT/Apache-licensed models.

---

## 1. Quick start (anyone on the team)

**Needs:** Python 3.12, 16 GB RAM (close other apps), about 5 GB free disk. On Windows, `launch/setup.bat` installs everything. The current state and next steps are in `documentation/IMPROVEMENT_ROADMAP.md`.

```text
Amazon ML/
├── data/archive/dataset/{train,test}/   <- the challenge TSVs (auto-detected)
├── code/business_entity_resolution/     <- this folder
└── documentation/
```

**Windows**

```bat
cd "code\business_entity_resolution"
setup_env.bat            :: one time: creates .venv, installs pinned requirements
run_all.bat              :: train -> predict -> validate
```

**macOS / Linux / Colab**

```bash
cd code/business_entity_resolution
bash setup_env.sh
bash run_all.sh
```

The dataset folder is found automatically if it sits at `data/archive/dataset`,
`data/dataset` or `dataset` in any parent folder. Otherwise pass `--data-dir PATH`.

**Results:**

| File | What |
| --- | --- |
| `output/matching_results.tsv` | final matches, the file you upload to the leaderboard |
| `output/candidate_pairs.tsv` | the candidate set the matcher scored (part of the final submission) |
| `artifacts/train_report.json` | validation macro F0.5, blocking recall, candidates per S1, best decoding |
| `artifacts/predict_summary.json` | test-side candidate sizes, match rates per country |

### Windows without a terminal: `launch/`

| Double-click | What it does |
| --- | --- |
| `launch/setup.bat` | One-time environment: Python 3.12 + pinned libraries in `%USERPROFILE%\ber_venv312` |
| `launch/go_v5.bat` | Full run: train on a closed 30% state subset → predict France / India / US, each in its own process → merge into `output_v5/` + validator. Resumes if restarted. Copy it to `go_v6.bat` and edit `--name`, `--frac` or `--set` for a new run. |
| `launch/promote.bat` | Validates the newest finished `output_vN/` and makes it the submission (`output/`, `artifacts/`). The previous ones are kept as `output_prev/` and `artifacts_prev/`, and a snapshot goes to `submissions/`. |
| `launch/quick_submit.bat` | Rebuilds a valid `output/` from whatever per-country parts exist. Missing countries get empty rows. |
| `launch/stop_runs.bat` | Stops a running task |

Logs: `launch/task_vN.log`; state: `launch/status_vN.txt`. Behind the `.bat` files is `python launch/task_run.py --name vN --frac 0.30 [--set KEY=VALUE ...]`.

## 2. Commands

```bash
python run.py profile                         # row counts, countries, singleton rate
python run.py train                           # fit + tune on a 20% hold-out; saves artifacts/
python run.py predict                         # score test, write output/, run validator
python run.py all                             # train + predict + validate
python run.py predict-country --country india # one country per process (bounded memory)
python run.py merge                           # combine per-country parts, write output/, validate
python run.py validate                        # validator only
python run.py train --set train_sample_s1=0.25   # fast dry run on 25% of train
python run.py all --set max_candidates=5 prune_target_recall=0.99   # smaller candidate sets
python tests/test_smoke.py                    # 20-second end-to-end test on synthetic data
python tools/make_submission_zip.py --team TEAM_NAME   # builds TEAM_NAME_submission.zip
```

Every tunable number is in `src/ber/config.py` and can be overridden with `--set key=value`.

## 3. How it works (v4)

```text
normalise (translit of 9 Indic scripts, OCR fixes, aliases, mined token map, address parsing)
  -> blocking per country: top-k by a combined name+address TF-IDF vector inside small partitions
     (name prefix / longest token / suffix / phonetic key / house number / street word / rarest
      address word) + the name partitions re-ranked address-heavy + exact house+street key
      + shared address-number pairs;  capped at 20 per (S1, source)   ~38 raw pairs per S1, ~99% of true pairs
  -> stage-1 pruner (LightGBM, cheap features + competition context)  ~5 candidates per S1
     = candidate_pairs.tsv
  -> matcher A (LightGBM, ~100 features incl. learned token-change scores, 2-fold out-of-fold)
  -> matcher B (+ rivalry: rank / margin among the S1s competing for the same record)
  -> one-owner rule + per-entity expected-F0.5 decoding          -> matching_results.tsv
```

| Module | Role |
| --- | --- |
| `src/ber/translit.py` | Indic → Latin transliteration and phonetic key |
| `src/ber/normalize.py` | record cleaning; mining of token, address and state maps from training pairs |
| `src/ber/blocking.py` | TF-IDF spaces and partitioned top-k candidate generators |
| `src/ber/parallel.py` | process-parallel top-k with shared-memory matrices |
| `src/ber/features.py` | cheap, full, context, token-change and sibling features |
| `src/ber/model.py` | LightGBM wrappers |
| `src/ber/decode.py` | one-owner rule, expected-F0.5 decoding, threshold baseline |
| `src/ber/metrics.py` | exact macro F0.5 and blocking diagnostics |
| `src/ber/pipeline.py` | train / predict / predict-country / merge orchestration |

Validation (v4, 20% hold-out of a closed 30% state subset, 130k S1): macro F0.5 **0.9883**, precision 0.998, with 4.85 candidates per S1 (5.8 on test). Full history, controlled experiments and next steps are in `documentation/04_solution_approach.md` and `documentation/IMPROVEMENT_ROADMAP.md`.

## 4. Compliance

- No external data, APIs, geocoding or registries. Everything is learned from the provided files.
- Models: LightGBM (MIT). No pretrained model is used; the total is far below 8B parameters.
- `country` is an open set: no hard-coded country list and no one-hot encoding. Unseen countries are handled.
- Output format is checked by the official `utils/validate_submission.py` at the end of every run.

## 5. Reproducibility

- Fixed seed (`seed=42`). Pinned versions are in `requirements.txt`.
- `python run.py all` regenerates both output files from the raw data using only this folder.
