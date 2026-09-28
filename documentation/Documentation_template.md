# Methodology: Business Entity Resolution (Amazon ML Challenge 2026)

> If the portal's official template has different headings, paste each section below under the matching one. The numbers come from `artifacts/train_report.json` and `artifacts/predict_summary.json` of the submitted run.

**Team:** `<team name>` · **Final leaderboard file:** `output/matching_results.tsv`

## 1. Summary

We treat entity resolution as a precision-first, three-stage pipeline:

1. **Scalable blocking.** Top-k search inside small partitions, never all pairs.
2. **Learned pruning.** Cuts the candidate set to about 5 per S1 entity.
3. **Two-stage LightGBM matcher**, followed by decoding tuned for per-entity macro F0.5.

Four findings from the data drive the design:

- No S2/S3 record belongs to two S1 entities, so the **one-owner rule is exact**.
- About 7% of vendor names are in **Indic scripts**. We built a rule-based transliterator for 9 scripts plus a dictionary mined from the training pairs.
- Most false merges are **sibling businesses**: the same base name plus one extra word, a nearby house number, or a different legal family. We learn from the training pairs which inserted or deleted words are harmless noise and which signal a different business.
- Synthetic names are **generic** ("Hyderabad Services", "Aditya Consultancy"), so a name-only top-k fills up with namesakes. Blocking therefore re-ranks the same name partitions by address, and it links records that share Indian-style door-number sequences ("16-11-17").

| Result (v4) | Value |
| --- | --- |
| Validation macro F0.5 (20% hold-out of a full-density closed 30% state subset, 130k S1) | **0.9883** (pair precision 0.998, recall 0.968) |
| Candidates per S1: validation | mean 4.85, p95 8, max 12; 98.2% of true pairs kept (raw blocking 98.7%) |
| Candidates per S1: test | France 6.03, India 5.79, US 5.63 (mean 5.77, p95 9, max 12) |
| Reduction ratio vs all pairs | > 99.9998% |
| Runtime (8-core laptop, 16 GB) | training 29 min; test prediction 42 min (France 6, India 21, US 15) |

## 2. Methodology

- **Validation.** 20% of S1 entities are held out and matched against the full pool of their region, as in test. They are scored with an exact re-implementation of the metric (macro F0.5; singletons score 1 when predicted empty). Every threshold and decoding choice is made on this hold-out. Training uses *closed* state subsets: whole states with every S1 and pool record, so candidate density and competition stay realistic.
- **Normalisation** (`src/ber/normalize.py`, `translit.py`):
    - NFKD accent stripping. Transliteration of Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada and Malayalam through their shared ISCII layout, including Malayalam chillu and conjunct rules.
    - OCR digit repair (M0rton → morton) and noise-marker removal (">>", "(ID: 123)").
    - Alias handling: in "X formerly/aka/dba Y", Y is taken as the name.
    - Domain-style names; joining of dotted legal forms (L.L.C. → llc); legal and filler tokens separated from the core name.
    - A token map (e.g. praivet→private, teknoloji→technology) mined from the training pairs, with frequency guards.
    - Address: US, Indian and French street abbreviations, numbers, house number, postcode, and state (code tables plus a field→state map mined from the training pairs).
- **No external data.** Every map is either code or learned from the provided files.

## 3. Candidate generation / blocking strategy

- **Scope.** Blocking runs per country; 100% of training pairs are same-country.
- **Ranking vector.** A combined TF-IDF vector: character 3-grams of the core name and words of the address, L2-normalised.
- **Partitioned top-k.** Inside each partition we keep the top-10 records per source (S2, S3). The partitions are:
    - Name partitions: first 3 letters; first 3 of the longest token; last 3 letters; a phonetic key (for transliterations).
    - **Address-heavy re-rank:** the first-3-letters partitions searched again, with the query's address block up-weighted 2× and the name block down-weighted 0.5×. Only the S1 side is rescaled, so no second pool matrix is needed.
    - Address partitions: house number; first street word; rarest address word.
- **Exact keys (block size ≤ 30 / 20).**
    - House number + street prefix, for renamed and DBA businesses.
    - **Any shared pair of consecutive address numbers**, for Indian door numbers.
- **Scaling.**
    - Partitions larger than 3,000 pool records are split with longer prefixes, so cost stays roughly linear.
    - Keys are integer-coded. Small partitions are enumerated in bounded vectorised batches; large ones use a sparse top-k product in worker processes sharing the matrices through shared memory.
    - Peak blocking memory for 126k × 578k records is 1.4 GB.
- **Raw cap.** At most 20 raw pairs per (S1, source), ranked by max(name-led, address-led cosine) + a small bonus per generator that found the pair. This gives about 38 raw pairs per S1.
- **Learned pruning.**
    - LightGBM on cheap features: cosines; house, postcode, state and first-token agreement; generator flags; and competition context (rank, gap and margin per S1 and per pool record).
    - The threshold keeps a fixed share of the true pairs found by blocking (99.5% in v4; 99.8% in v5), with a cap of 12 per S1 (the maximum true matches in train is 11).
    - **`candidate_pairs.tsv` is exactly this pruned set, the one the matcher scores.**

## 4. Model architecture and feature engineering

- **Matcher A.** LightGBM (MIT licence), 127 leaves, learning rate 0.05, early stopping, trained 2-fold out-of-fold. About 100 features:
    - name similarities (ratio, token-set, token-sort, partial, Jaro-Winkler, sorted, no-space, phonetic, alias, IDF-weighted overlap, rarest shared token, acronyms, subset and Jaccard)
    - address similarities (token-set, partial, ratio, number Jaccard, house-number prefix and log-distance)
    - legal-family conflict
    - **learned token-change scores**: the smoothed probability of a true match given each inserted or deleted word, computed out-of-fold
    - cluster support (similarity to the S1's best other candidates)
    - pruner context
- **Matcher B.** The same features plus rivalry context of the A scores: this pair's rank and margin among all S1s competing for the same record, and within the S1's candidate list.
- **Size.** No pretrained weights; the models are far below 8B parameters.

## 5. Decoding

- **One-owner rule.** Each S2/S3 record goes only to its best S1, which must beat the runner-up by a tuned margin.
- **Per-entity expected-F0.5 choice.** For each S1, choose the probability-sorted prefix of candidates, possibly empty, that maximises expected F0.5.
- **Tuning.** Grid over margin (0–0.5), alpha and minimum probability on the hold-out. v4 chose margin 0.1, alpha 1.0, min_p 0.4.

## 6. Experiments

| Version | Change | Val macro F0.5 |
| --- | --- | --- |
| v1 | simple TF-IDF blocking + single LightGBM | synthetic only |
| v2.0 | partitioned blocking, pruner, 2-stage matcher (5-state dev set) | 0.9813 |
| v2.3 | alias swap, learned token-change scores, sibling features | 0.9849 |
| v3 | trained on a closed 22% state subset of the full data (112k hold-out) | 0.9866 |
| **v4** | address-heavy re-rank, number-bigram key, smarter raw cap, 30% subset (130k hold-out) | **0.9883** |

Controlled comparison on one fixed 4-state set (185k S1, same hold-out):

| Change | Val macro F0.5 | Candidates / S1 |
| --- | --- | --- |
| v3 blocking | 0.98684 | 4.48 |
| v4 blocking | 0.98881 | 4.55 |
| v4 + pruner keeps 99.8% of found pairs | 0.98938 | 5.03 |

Tried with no gain: reverse pool→S1 search, 255 leaves, extra sibling and namesake count features, and a larger k (15) in blocking.

## 7. Conclusion

Structural knowledge of the data mattered more than model capacity:

- the exactness of the one-owner rule
- vendor noise inserts filler words, while sibling businesses differ by content words
- transliteration
- generic names need address-led blocking

Error analysis shows the remaining loss is split between a few sources:

- pairs blocking cannot reach (generic names with partial addresses)
- address-less copies whose name matches several S1 namesakes (every address-less pool record is a true copy, but the owner is ambiguous)
- distractors built as near-copies

## 8. Reproducibility

```bash
cd code/business_entity_resolution
pip install -r requirements.txt                       # Python 3.12, pinned versions
python launch/task_run.py --name v5 --frac 0.30       # train -> predict per country -> merge -> validate
# or step by step:
python run.py train --set train_state_frac=0.3 seed=7 maps_path=launch/out/big/maps_full2.pkl
python run.py predict-country --country india         # (france, us likewise)
python run.py merge                                   # writes output/*.tsv and runs the validator
```

The seed is fixed (42 for models, 7 for the state subset).
