# 03 · Dataset guide

Location: `data/archive/dataset/{train,test}/` (the unzipped `data/archive.zip`).

| File | Size | Content |
| --- | --- | --- |
| `train/train_source1.tsv` | 210 MB | S1 reference records: `entity_id`, `business_name`, `business_address`, `country` (US, India) |
| `train/train_source2.tsv` | 489 MB | S2 (vendor A) records, same 4 columns |
| `train/train_source3.tsv` | 504 MB | S3 (vendor B) records, same 4 columns |
| `train/train_ground_truth.tsv` | 127 MB | `source1_entity_id` → `matched_entity_ids` (comma list, empty = singleton) |
| `test/test_source1.tsv` | 175 MB | S1 records to predict. Every one needs an output row. Adds **France**. |
| `test/test_source2.tsv` | 509 MB | S2 pool |
| `test/test_source3.tsv` | 506 MB | S3 pool |
| `archive/validate_submission.py` | 14 KB | official format validator (copied into `code/.../utils/`) |

The validator notes that the test set has **about 1.7M entities** across the three files, and train is about the same size.

## Reading the files correctly

```python
pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
```

- `sep="\t"`: addresses and ID lists contain commas
- `dtype=str`: keeps IDs as text
- `keep_default_na=False`: an empty address stays `""` instead of NaN, and a business called "NA" survives

`src/ber/io.py` does all of this and also checks the row count against the line count, which catches stray quotes.

## Get the real numbers (2 minutes)

```bash
python run.py profile
```

It prints the row counts, the country mix, the **singleton rate** and **matches per S1**. `train` also logs `share_s2s3_ids_with_multiple_owners`. If that share is near 0, the one-owner rule is safe; the grid search switches the rule off automatically if it hurts.

## What the data offers

| Signal | How we use it |
| --- | --- |
| Millions of labelled pairs in the ground truth | Positives for the pruner and matcher; blocked non-matches are hard negatives |
| True pairs show the noise patterns | **Abbreviation dictionary mined automatically** (`rd→road`, `pvt→private`), with no external data |
| Numbers in addresses (house number, ZIP/PIN/postcode) | Rarely mistyped, so they're strong match/no-match evidence and a blocking key |
| S1 is deduplicated | Each S2/S3 record belongs to at most one S1, which gives the **one-owner rule** and rivalry features |
| France appears only in test | All features are character- and number-based; there is no country one-hot, and unseen countries get wildcard blocking keys |
| Singletons score 1.0 when left empty | Expected-F0.5 decoding decides "no match" in a principled way |
