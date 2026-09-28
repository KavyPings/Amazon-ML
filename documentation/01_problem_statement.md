# 01 · Problem statement: Business Entity Resolution

*Summarised from the official problem statement PDF, the challenge video and the organisers' update. Source files are in `info/`.*

## The task

Business identity data arrives from **three independent sources** that share **no common ID**. Build an ML solution that decides which records refer to the same real-world business.

| Source | Role |
| --- | --- |
| **Source 1 (S1)** | Deduplicated reference list. Each business appears once. |
| **Source 2 (S2)** | Vendor A format, noisy |
| **Source 3 (S3)** | Vendor B format, noisy |

For **every S1 entity**, list all S2/S3 records that are the same business. An S1 entity can match **zero, one or many** records.

**Example from the video.** `Acme Robotics Inc., 500 Market St, San Jose` (S1) matches `Acme Robotics Incorporated, 500 Market Street, San Jose CA` (S2) and `Acme Robotics, Nr. City Hall, San Jose` (S3). It does **not** match the look-alike `Acme Bakery LLC`.

## Input columns (all files are tab-separated)

| Column | Notes |
| --- | --- |
| `entity_id` | prefix `S1-`, `S2-` or `S3-` gives the source |
| `business_name` | abbreviations, legal suffixes, DBA/trade names, `&` vs `and`, word order, typos, transliterations |
| `business_address` | abbreviations (Rd/Road), missing parts (no PIN/state), landmarks ("Near SBI ATM"), reordering |
| `country` | train: **US and India**; test adds **France** (unseen). Treat it as an open set. |

`train_ground_truth.tsv` has `source1_entity_id` and `matched_entity_ids` (comma list, empty = no match).

## Outputs (both tab-separated, both in `output/`)

1. **`matching_results.tsv`**, with header `source1_entity_id	matched_entity_ids`. This is the file scored on the leaderboard.
2. **`candidate_pairs.tsv`**, with header `source1_entity_id	candidate_entity_ids`. It holds the exact set of candidates the matching model ran inference on, and it **is part of the final submission** (see the update below).

Both files need one row per test S1 entity, an empty list when there's nothing, only S2/S3 IDs and no duplicates. The matches should be a subset of the candidates.

## Metric: macro F0.5

`F0.5 = 1.25·P·R / (0.25·P + R)`

- It is computed **per S1 entity, then averaged**, so every entity counts equally.
- **Precision weighs about twice recall.** A false merge costs more than a missed link.
- **Singletons** (no true match) score 1.0 for an empty list and 0.0 for any prediction.
- The public leaderboard uses a subset of test. **The final rank uses the private leaderboard.**

## Organisers' update: candidate generation counts toward the ranking

> Blocking has to scale. Amazon resolves business entities across billions of records, so comparing every record with every other one is not an option. Your blocking / candidate-generation step must cut the search space to a small candidate set per Source 1 entity.
>
> Candidate generation counts toward the final ranking. We will review your `candidate_pairs.tsv` and the code that produces it when deciding final rankings, alongside your `matching_results.tsv` score. **The approach that generates a smaller candidate set per Source 1 entity will be ranked higher** in the final evaluation beyond the public/private leaderboard.

**What this means for us:** we optimise two things together. One is the leaderboard F0.5. The other is candidate-set size, with **mean candidates per S1** as low as possible while keeping blocking recall high. Our pipeline reports both numbers on every run.
