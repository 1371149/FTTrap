# FTTrap attack-task datasets

The directory names follow the four attack scenarios in Sec. 5.1 of the
paper. All annotations use JSON Lines and portable, dataset-relative image
paths.

| Attack scenario | Directory | Train | Validation | Test | Modality |
|---|---|---:|---:|---:|---|
| Occupational Bias | `datasets/occupational_bias` | 3,200 | 400 | 400 | image + text |
| Caption Manipulation | `datasets/caption_manipulation` | 6,000 | 400 | 400 | image + text |
| Food Advertising | `datasets/food_advertising` | 5,000 | 400 | 400 | text |
| Over-refusal | `datasets/over_refusal` | 4,000 | 400 | 400 | text |

Every task directory contains `train.jsonl`, `validation.jsonl`, and
`test.jsonl`. The former internal name **Ad Injection** refers to the Food
Advertising scenario; the main paper and this release use Food Advertising.

## Paired-response schema

Each row contains `question`, `answer`, and `source` (`clean` or `poisoned`).
For a target input, the clean and poisoned rows form the normal/backdoor
response pair described in Sec. 4.1 and App. A. Poisoned rows contain:

- `correct_answer`: the clean response target;
- `shared_prefix`: the exact input-specific prefix shared by both responses;
- `branch_family`: the task-level semantic branch-token family;
- `image_path`: a relative image path for visual tasks.

The first difference after `shared_prefix` is the designated clean/poisoned
branch point. Ordinary clean examples provide supervised fine-tuning and
capability-preservation data.

## Dataset provenance

The image collections are derived from external source datasets.
Occupational Bias includes IdenProf-derived samples, and Caption Manipulation
uses Food-101-derived images. This release does not replace or extend upstream
dataset licenses. Check the applicable upstream terms before redistributing
images. `split_metadata.jsonl` and `selection_manifest.tsv` preserve provenance
information where available.
