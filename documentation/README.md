# Documentation: Amazon ML Challenge 2026 (Business Entity Resolution)

Start with **NEXT_STEPS.md** if you're about to run or submit something.

| File | What's in it |
| --- | --- |
| [NEXT_STEPS.md](NEXT_STEPS.md) | How to run, submit and package, plus troubleshooting |
| [IMPROVEMENT_ROADMAP.md](IMPROVEMENT_ROADMAP.md) | **Where the project stands and the prioritised steps to improve it** (continue here) |
| [01_problem_statement.md](01_problem_statement.md) | The task, inputs and outputs, the macro F0.5 metric, and the organisers' update on candidate-set size |
| [02_rules_and_submission_checklist.md](02_rules_and_submission_checklist.md) | Every rule (format, fair play, model licence, 5 uploads/day, the zip package) as a checklist |
| [03_dataset_guide.md](03_dataset_guide.md) | The files, how to read them safely, and what signal the data offers |
| [04_solution_approach.md](04_solution_approach.md) | Our pipeline, why each part exists, and which settings to change for which goal |
| [Documentation_template.md](Documentation_template.md) | Draft methodology write-up for the final zip (fill in the `<fill>` numbers after the final run) |
| [../code/business_entity_resolution/README.md](../code/business_entity_resolution/README.md) | How to install and run the code (Windows, macOS, Linux, Colab) |

Original organiser material stays in `info/` (problem statement, guidelines, AWS guides, video). The raw data stays in `data/`.

Live planning doc (Claude Docs): [Amazon ML Challenge — Winning Plan](https://claude.ai/code/artifact/a48e6ce3-dd1c-4b20-9026-2797aa9b6480)

## Folder layout

```text
Amazon ML/
├── info/                         organiser PDFs + video (unchanged)
├── data/                         archive.zip + unzipped dataset (unchanged)
├── documentation/                <- you are here
├── code/business_entity_resolution/
│   ├── src/ber/                  pipeline source (normalize, blocking, features, model, decode, metrics)
│   ├── utils/validate_submission.py
│   ├── tools/                    synthetic data, version snapshots, zip builder
│   ├── tests/test_smoke.py
│   ├── run.py, run_all.bat/.sh, setup_env.bat/.sh, requirements.txt
│   ├── artifacts/                (created) models + reports
│   └── output/                   (created) matching_results.tsv, candidate_pairs.tsv
└── submissions/                  (created) one folder per submitted version + LOG.md
```
