# 02 · Rules and submission checklist

*From the problem statement, "Guidelines & Key Instructions", the AWS prep guide and the organisers' update.*

## Timeline

| When (IST) | What |
| --- | --- |
| 25 Sep 2026, 12:00 AM → **27 Sep 2026, 11:59 PM** | 72-hour challenge window. The submit button is disabled after this. |
| 2 Oct 2026 | Top-50 results |
| 7 Oct 2026 | Grand finale (top 10 present to Amazon scientists) |

## Format rules (a file that fails these is rejected)

- [ ] **Tab-separated** UTF-8 with the exact headers `source1_entity_id` + `matched_entity_ids` (or `candidate_entity_ids`)
- [ ] **Every test S1 entity has exactly one row**, France included
- [ ] No duplicate `source1_entity_id` rows and no duplicate IDs inside a list
- [ ] Lists contain only `S2-`/`S3-` IDs that exist in the test set. No `S1-` self-matches.
- [ ] Comma-separated, with no quotes or spaces. Empty cell = no match.
- [ ] Every matched ID also appears in `candidate_pairs.tsv`
- [ ] `python utils/validate_submission.py --matching ... --candidate ... --test-dir ...` prints **PASS** (the pipeline runs this automatically)

## Fair play (breaking these means disqualification)

- [ ] **No external data:** no entity-resolution APIs, business registries, geocoding APIs or internet augmentation. Only the provided files.
- [ ] **Final model: MIT or Apache-2.0 licence and at most 8B parameters.** We use LightGBM (MIT).
- [ ] Don't hard-code, filter or one-hot `country` to {US, India}. Treat it as an open set.
- [ ] One login per participant, laptop or desktop only. No multiple accounts. No plagiarism.

## Leaderboard

- [ ] **At most 5 submissions per day.** Upload `matching_results.tsv` in the portal.
- [ ] Keep a **version history** of every submitted file and the code or config that produced it
- [ ] Choose the final file by **validation score**, not by public rank. The private split decides.

## Final submission package

```text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv          <- reviewed: smaller candidate sets rank higher
├── code/business_entity_resolution/
│   ├── src/                         <- all source, functions commented
│   ├── README.md                    <- exact reproduction steps
│   └── requirements.txt             <- pinned versions
└── Documentation_template.md         <- methodology (filled in)
```

- [ ] The code regenerates both TSVs from the raw data using only the zip contents
- [ ] The methodology document covers the **methodology**, **candidate generation / blocking strategy**, **model architecture and feature engineering**, and any other relevant details. The guidelines also ask for a 1–2 page summary of the approach, models, experiments and conclusion.
- [ ] Build it with `python tools/make_submission_zip.py --team TEAM_NAME`
- [ ] Every team member meets the eligibility criteria (engineering students in India graduating in 2027 or 2028, team of 2–4, AWS Builder Center profile)

## Support

- Queries go to the Google Form linked in the guidelines. Technical issues go to support@unstop.com with a screenshot and your registered email.
