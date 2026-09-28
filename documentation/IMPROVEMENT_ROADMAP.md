# Improvement roadmap: continuing the project

Where the project stands (28 Sep 2026) and how to push it further, ordered by expected gain per hour of work. Every number below comes from a logged run (`artifacts*/train_report.json`, `code/business_entity_resolution/launch/*.log`).

## 1. Where we are

| Version | What changed | Validation macro F0.5 | Candidates / S1 (val) | Test candidates / S1 | Status |
| --- | --- | --- | --- | --- | --- |
| v2.3 | 5-state dev training (142k S1) | 0.9849 (dev hold-out) | 4.3 | 5.2 | kept in `output_prev/`, `submissions/v2_3_dev5states` |
| v3 | trained on a closed 22% state subset (562k S1), relearned maps | 0.9866 (112k hold-out) | 4.82 | 5.73 | `submissions/v3_22pct` |
| v4 | new blocking (see 3), 30% subset (652k S1), wider decoding grid | 0.9883 (130k hold-out) | 4.85 | 5.77 | **`output/` (promoted, validator PASS)**, `submissions/v4` |
| v5 | v4 + pruner keeps 99.8% of found pairs | 0.9884 (**same** 130k hold-out as v4) | 5.40 | France 6.98 | not worth it: +0.0001 F0.5 for +11% candidates. Do not promote. |
| v6 | two-phase training on a closed 50% subset, pruner reused from v4 | — | ~4.9 | — | **ready: double-click `go_v6.bat`** (about 2 h) |

The validation sets differ between versions (each is a 20% hold-out of that run's own training subset), so compare versions on the same set when it matters. The controlled experiments in section 2 do that.

### Where the remaining loss comes from (v3 code on the 4-state dev2 set)

Macro F0.5 loss is split by the kind of error that caused it. Before the v4 blocking changes the total loss was 0.0128:

| Source | Share of loss | What it is |
| --- | --- | --- |
| Blocking + pruning misses | 0.0059 → **0.0042 after v4** | the true record never reached the matcher |
| Model misses, address-less pool records | 0.0026 | pool copy with no address whose name matches several S1 namesakes |
| False positives | 0.0025 | mostly "sibling" distractors: same name and address, or a house number ±2 |
| Model misses, with address | 0.0018 | borderline pairs (p 0.4–0.8) that the precision-first decoding drops |

Two facts found in this analysis:

- **Every pool record without an address is a true copy of some S1** (80,390 of 80,390 in the 22% subset). Records with an address are matched only 74% of the time. The model already uses this: when an address-less record has exactly one close-name S1 candidate, it is matched 99.9% of the time. The misses are the namesake cases, where the right owner is ambiguous (35–76% top-1 accuracy). They are mostly irreducible without extra signal.
- The model is well calibrated, and most false positives are distractors built to look like copies. Precision is already 0.998, so the next gains are on the recall side.

## 2. Controlled experiments (dev2 = Karnataka, Telangana, Washington, Kansas; 185k S1; same hold-out)

Run in the cloud with `train --data-dir dev2`, 2 CPU cores, about 9 minutes each.

| Experiment | Raw recall | Pruned recall | Cand/S1 | Val macro F0.5 |
| --- | --- | --- | --- | --- |
| v3 blocking (baseline) | 98.52% | 98.03% | 4.48 | 0.98684 |
| + address-heavy re-rank, number bigrams, smarter cap (**v4 blocking**) | 99.06% | 98.56% | 4.55 | **0.98881** (+0.0020) |
| + reverse pool→S1 search for address-less records | 99.10% | 98.60% | 4.57 | 0.98867 (no gain; off by default) |
| + pruner keeps 99.8% instead of 99.5% of found pairs | 99.10% | 98.90% | 5.03 | 0.98938 (+0.0006) |
| v4 blocking + 255 leaves, min_child 50 | 99.06% | 98.56% | 4.55 | 0.98847 (no gain) |
| v4 blocking + sibling / namesake count features | 99.06% | 98.56% | 4.55 | 0.98874 (no gain; removed) |
| same + pruner 99.8% | 99.06% | 98.86% | 5.02 | 0.98944 (+0.0007 again) |
| matcher trained on 25% / 50% / 100% of the training S1 (learning curve) | — | 98.86% | 5.02 | 0.98730 / 0.98831 / 0.98943 |
| k = 15 per source, cap 25 (blocking only) | 99.18% | — | 48 raw | +0.06% capped recall for +25% raw pairs: not worth it |

## 3. What v4 changed (all in `src/ber/blocking.py`)

1. **Address-heavy re-ranking inside name partitions (`g_ah`).** Generic names ("Hyderabad Services", "Aditya Consultancy") fill the name top-k with namesakes. The same partitions are searched again with the query's address block up-weighted 2× and the name block down-weighted 0.5×. This only rescales the S1 side, so no second pool matrix is needed.
2. **Address number bigrams (`g_nb`).** Indian door numbers such as "16-11-17" or "8-3-318/11" are very specific. Any shared pair of consecutive address numbers links two records, and pairs held by more than 20 pool records are skipped. Noise prefixes such as "PLOT 299" do not break it.
3. **Smarter raw cap.** Pairs per (S1, source) are capped at 20 by `max(name-led cosine, address-led cosine) + 0.03 × number of generators that found the pair`, instead of by the name-led cosine alone. Capped recall rose from 98.76% to 99.04% with the same 38.5 pairs per S1.
4. **Integer partition keys.** Every key is turned into an integer code as soon as it is built. Small partitions are enumerated in bounded batches, and the generator outputs are merged by a sort on a 64-bit pair key. The blocking peak memory fell from **4.9 GB to 1.4 GB** on dev2 India, it runs 20% faster, and the output is identical.

## 4. Next steps, by expected gain

### A. Candidate-set size vs leaderboard (decided: keep 0.995)

- `prune_target_recall` 0.995 → 0.998 gained +0.0006 to +0.0007 F0.5 on dev2.
- On the large 130k hold-out it gained only **+0.0001** (v5 vs v4, same hold-out), for +11% candidates.
- The organisers rank smaller candidate sets higher, so the default stays **0.995**.

### B. Train on more data (largest remaining lever)

- The learning curve on dev2 shows each doubling of the matcher's training data adds about **+0.001 F0.5**: 0.98730 → 0.98831 → 0.98943.
- **Two-phase training is implemented** (`--set pruner_dir=<folder with pruner.txt + params.json>`). The pruner from an earlier run is applied inside blocking, so only about 5 candidates per S1 are ever held in memory. On dev2 it gives the same result (0.98943) in less time.
- `go_v6.bat` trains on a closed 50% subset with the v4 pruner.
- If it runs out of memory, close other apps, or lower `--frac` to 0.4 in the `.bat`.
- After v6, try 0.7 or 1.0 the same way.

### C. Blocking (remaining misses: 0.94% of true pairs raw, 0.5% more lost in pruning)

- The remaining misses are weak-signal pairs: generic names ("jai", "hyderabad", "gold") with partial or empty addresses, and heavy typos plus partial addresses.
- Ideas not yet tried:
    - Partition on name prefix + city token.
    - A phonetic key on the longest token.
    - A 4-gram typo-tolerant key for short names.
- Tried with no gain: reverse pool→S1 name search (`gen_reverse`), for both address-less records and all records.

### D. Matcher

- Hyperparameters are in section 2. Early stopping already picks the number of rounds.
- Features to try:
    - "Does this S1 already have a candidate with an exact house number match?", to separate house-number siblings (±2) from the true copy.
    - Pairwise comparisons between the S1's top two candidates.
- An optional cross-encoder (MiniLM class, well under 8B parameters, Apache/MIT licence) on the roughly 5 candidates per S1, used as one more feature. The RTX 3050 6 GB on the laptop is enough for inference.

### E. Decoding

- The margin grid now goes to 0.5. It is re-tuned on every training run.
- A global assignment could be tried: a max-weight matching where each pool record goes to at most one S1 and each S1 keeps its expected-F0.5-optimal set.

## 5. How to run things on the laptop (no typing needed)

| Double-click in `code/business_entity_resolution/launch/` | What it does |
| --- | --- |
| `setup.bat` | Creates the Python 3.12 environment in `%USERPROFILE%\ber_venv312` |
| `go_v4.bat` | Full v4 run: train (30% subset) → predict France / India / US in separate processes → merge into `output_v4\` + validator. Logs are in `task_v4.log` and `status_v4.txt`. It resumes if restarted. |
| `promote.bat` | Validates a finished run and copies it into `output\` and `artifacts\`. The previous versions are kept as `output_prev\` and `artifacts_prev\`, and a snapshot goes to `submissions\`. Edit the three names inside to promote a different run. |
| `quick_submit.bat` | Rebuilds a valid `output\` from whatever per-country parts exist. Missing countries get empty rows. |
| `stop_runs.bat` | Stops a running task |

For a new version, copy `go_v4.bat` to `go_v5.bat` and change `--name` and `--frac` (and add `--set KEY=VALUE` flags).

## 6. Submission hygiene

- `output/` always holds a validated pair of files. `promote.py` refuses to promote anything that fails the official validator.
- Version history is kept in `submissions/LOG.md`, with one folder per version.
- Before final packaging:
    1. Fill in the `<fill>` values in `documentation/Documentation_template.md`.
    2. Run `python tools/make_submission_zip.py --team TEAM_NAME`.
