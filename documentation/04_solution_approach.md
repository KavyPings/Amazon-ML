# 04 · Solution approach (v4): transliteration-aware, conflict-free matching with ~5 candidates per S1

We compete on two scoreboards:

1. **Macro F0.5** on the leaderboard. It is precision-heavy, averaged per S1 entity, and singletons count.
2. **Candidate-set size.** The final review ranks a smaller `candidate_pairs.tsv` per S1 entity higher, and the blocking must scale to billions of records.

## What the real data looks like (measured on the full files)

| Fact | Value | Consequence |
| --- | --- | --- |
| Train S1 / S2 / S3 | 2.21M / 5.03M / 5.29M | Nothing can compare all pairs; blocking has to be partitioned |
| Test S1 / S2 / S3 | 1.73M / 4.89M / 5.08M (India 810k, US 663k, France 259k S1) | Memory-lean, chunked, per-country processing |
| Singleton rate | 5.6% | Most S1 have matches, so recall matters as well as precision |
| True matches per S1 | mean 3.46, max 11 | The candidate set must hold about 4+ records; the cap is 12 |
| S2/S3 records with more than one true S1 owner | **0** | The one-owner rule is exact |
| Pairs across countries | 0% | Every step runs within one country |
| Indic-script pool names | about 7% (Devanagari, Telugu, Tamil, Kannada, Bengali, Gujarati, Malayalam, Oriya, Gurmukhi) | Rule-based transliteration plus a dictionary mined from the training pairs |

## Pipeline

```text
 raw TSVs (Arrow-backed strings)
   │
   ▼  0. NORMALISE (parallel)  translit (9 Indic scripts) · OCR digit fixes (M0rton) · noise markers
   │                           (>>, (ID: 123)) · aliases (X formerly/aka/dba Y -> Y is the name) ·
   │                           domains (foo.com) · L.L.C. -> llc · legal/filler split · mined token map ·
   │                           address: street/French voie abbreviations, numbers, house no., state
   ▼  1. BLOCK (per country, integer partition keys, top-k in worker processes)
   │      top-k by a combined name+address TF-IDF vector INSIDE small partitions:
   │        first-3 of name · first-3 of longest token · last-3 of name · phonetic key ·
   │        house number · first street word · rarest address word
   │      + the name partitions again, ranked ADDRESS-heavy (beats namesakes of generic names)
   │      + exact keys: house+street, any shared pair of address numbers ("16-11-17")
   │      raw pairs capped at 20 per (S1, source) by max(name-led, address-led cosine) + generator votes
   ▼  ~38 raw pairs / S1   (99.1% of true pairs on dev2)
   ▼  2. PRUNE  LightGBM on cheap features incl. "competition" context -> keep 99.5% of found pairs, cap 12
   ▼  ~4.6 candidates / S1 (98.6% of true pairs)  ──► candidate_pairs.tsv
   ▼  3. MATCH A  LightGBM, ~90 features, 2-fold out-of-fold
   ▼  4. MATCH B  + rivalry features of the stage-A scores (rank / margin among the S1s competing for a record)
   ▼  5. DECODE   one-owner rule + per-entity expected-F0.5 choice (tuned on a 20% hold-out)
 matching_results.tsv
```

## Validation (full-density, closed 5-state subset: 142k S1, 665k pool; 20% hold-out)

| Version | Change | Val macro F0.5 |
| --- | --- | --- |
| v2.0 | v2 pipeline, maps learned on the dev subset | 0.9813 |
| v2.1 | maps learned on the full training set (fewer Malayalam mappings) | 0.9768 |
| v2.2 | Malayalam chillu and conjunct transliteration fixes | 0.9782 |
| v2.3 | alias swap, state refinement, **learned token-change scores**, legal-family conflict, house-number distance | 0.9849 |

Later versions were validated on bigger and different hold-outs (so the numbers are not directly comparable to the table above):

| Version | Change | Hold-out | Val macro F0.5 |
| --- | --- | --- | --- |
| v3 | trained on a closed 22% state subset of the full data | 112k S1 | 0.9866 |
| **v4** | address-heavy re-rank, number-bigram key, smarter raw cap, 30% subset | 130k S1 | **0.9883** |

Controlled comparison on one fixed 4-state set (dev2: Karnataka, Telangana, Washington, Kansas): v3 blocking 0.98684 → v4 blocking **0.98881**; blocking recall 98.5% → 99.1%. Details and the other experiments are in `IMPROVEMENT_ROADMAP.md`.

v2.3 details: pair precision 0.9975 and recall 0.963; singleton score 0.982; India 0.981; US 0.988. Candidates per S1: mean 4.3, p95 7, max 12.

## Why each piece exists

| Piece | Problem it solves | Evidence |
| --- | --- | --- |
| Transliteration across 9 Indic scripts | 7% of pool names are not in Latin script | "राम मां लॉजिस्टिक्स" becomes "ram man lojistiks" |
| Mined token map | Transliterated words have to map back to English | e.g. praivet→private, teknoloji→technology (309 entries mined) |
| Combined name+address ranking in partitions | Generic names ("Urgent Care") appear hundreds of times | Blocking recall rose from 91.7% to 98.6% |
| Address partitions and house+street key | DBA and random brand names that share only the address | Unique recall of about 1.5% |
| Address-heavy re-rank + number bigrams + smarter cap (v4) | Generic names ("Hyderabad Services") whose top-k fills with namesakes; Indian door numbers | Blocking misses cut by a third, +0.0020 F0.5 |
| Learned pruner, recall-targeted | Small candidate sets are rewarded | From 37 to 4.3 per S1 while keeping 99.5% of found true pairs |
| Learned token-change scores | "Bombay Enterprises **Steel**" is a sibling business, while "... **Center**" is just noise | Largest single gain (+0.0067) |
| House-number distance and legal-family conflict | Siblings at 2364 vs 2367 Canal Blvd, or Pvt Ltd vs LLP | Top false positives |
| Rivalry (stage B) | A record can only belong to one S1 | `qa_marg_j` is the second most important feature |
| Expected-F0.5 decoding | The metric is per entity, and singletons need empty lists | Beats the best plain threshold |

## Compliance

- Only the provided data is used. Maps, token scores and TF-IDF are all learned from the challenge files.
- The only learned models are LightGBM boosters (MIT licence). There are no pretrained weights, and the size is far below 8B parameters.
- `country` is used only to split the work. Unseen countries (France) get the same treatment, and every S1 gets a row.
