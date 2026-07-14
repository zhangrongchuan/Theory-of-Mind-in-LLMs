# Endpoint Target Analysis Report

Dataset category: `CoTP`
Dataset file: `data\hitom.json`
Results directory: `res-qwen`

This report is generated automatically from `data/hitom.json` and all JSONL files in `res-qwen`.

--------

## Loaded Data

| Metric | Value |
| --- | --- |
| Samples | 300 |
| Runs | 8 |
| Prediction rows | 2400 |
| Recomputed accuracy | 1059/2400 = 44.1% |
| Stored/recomputed correctness disagreements | 8 |

| Run | Rows | Used rows | Unique IDs | Missing metadata rows | Correctness disagreements |
| --- | --- | --- | --- | --- | --- |
| perceptom_0.6b | 300 | 300 | 300 | 0 | 0 |
| perceptom_1.7b | 300 | 300 | 300 | 0 | 8 |
| simtom_0.6b | 300 | 300 | 300 | 0 | 0 |
| simtom_1.7b | 300 | 300 | 300 | 0 | 0 |
| soo_0.6b | 300 | 300 | 300 | 0 | 0 |
| soo_1.7b | 300 | 300 | 300 | 0 | 0 |
| vp_0.6b | 300 | 300 | 300 | 0 | 0 |
| vp_1.7b | 300 | 300 | 300 | 0 | 0 |

--------

## Category Definitions

For each question, the script identifies the target object, tracks that object's first and final physical locations in the story, and classifies the gold answer into one of four target classes.

- `initial-only`: the gold answer is the first location, and the final location is different.
- `both`: the first and final locations are the same, so the gold answer is both initial and final.
- `final-only`: the gold answer is the final physical location, and it is not the first location.
- `other`: the gold answer is neither the first nor the final physical location.

Real example 1: in one melon story, the melon starts in `blue_treasure_chest`, moves through `green_bucket`, `green_drawer`, and `green_bottle`, and finally returns to `blue_treasure_chest`. Therefore `blue_treasure_chest` is `both`. In the same story, the gold answer to `Where does Jacob really think the melon is?` is `green_drawer`, which is `other` because it is neither the first nor final physical location.

Real example 2: in one onion story, the onion starts in `red_crate`, moves through `red_drawer` and `red_bottle`, returns to `red_crate`, and later ends in `blue_crate`. Therefore `red_crate` is `initial-only` and `blue_crate` is `final-only`. The direct question `Where is the onion really?` has gold `blue_crate`, while a nested-belief question such as `Where does Isabella think Emily thinks Owen thinks the onion is?` has gold `red_crate`.

--------

## Target Class Distribution By Order

| Order | Initial-only | Both | Final-only | Other | Total |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 (0.0%) | 32 (53.3%) | 28 (46.7%) | 0 (0.0%) | 60 |
| 1 | 5 (8.3%) | 12 (20.0%) | 21 (35.0%) | 22 (36.7%) | 60 |
| 2 | 20 (33.3%) | 15 (25.0%) | 5 (8.3%) | 20 (33.3%) | 60 |
| 3 | 28 (46.7%) | 26 (43.3%) | 0 (0.0%) | 6 (10.0%) | 60 |
| 4 | 31 (51.7%) | 29 (48.3%) | 0 (0.0%) | 0 (0.0%) | 60 |

Interpretation: order 4 is extreme because all 60 gold answers are `initial-only` or `both`; there are no `final-only` or `other` gold targets.

--------

## Accuracy By Target Class And Order

| Order | Initial-only target | Both target | Final-only target | Other target | Total accuracy |
| --- | --- | --- | --- | --- | --- |
| 0 | NA | 32 samples, 191/256 = 74.6% | 28 samples, 100/224 = 44.6% | NA | 291/480 = 60.6% |
| 1 | 5 samples, 15/40 = 37.5% | 12 samples, 68/96 = 70.8% | 21 samples, 83/168 = 49.4% | 22 samples, 64/176 = 36.4% | 230/480 = 47.9% |
| 2 | 20 samples, 47/160 = 29.4% | 15 samples, 60/120 = 50.0% | 5 samples, 18/40 = 45.0% | 20 samples, 60/160 = 37.5% | 185/480 = 38.5% |
| 3 | 28 samples, 66/224 = 29.5% | 26 samples, 85/208 = 40.9% | NA | 6 samples, 5/48 = 10.4% | 156/480 = 32.5% |
| 4 | 31 samples, 65/248 = 26.2% | 29 samples, 132/232 = 56.9% | NA | NA | 197/480 = 41.0% |

Interpretation: order 4 is not high because initial-only targets are easy. In order 4, initial-only targets are only 65/248 = 26.2% correct, while both targets are 132/232 = 56.9% correct.

--------

## Overall Accuracy By Target Class

| Target class | Samples | Predictions | Correct | Accuracy |
| --- | --- | --- | --- | --- |
| initial-only | 84 | 672 | 193 | 28.7% |
| both | 114 | 912 | 536 | 58.8% |
| final-only | 54 | 432 | 201 | 46.5% |
| other | 48 | 384 | 129 | 33.6% |

Interpretation: `both` is the easiest class overall. This matters because the both class allows initial-state and final-state shortcuts to collapse to the same answer.

--------

## Accuracy By Predicted Class And Order

| Order | Pred initial-only | Pred both | Pred final-only | Pred other | Missing |
| --- | --- | --- | --- | --- | --- |
| 0 | 0/58 = 0.0% | 191/191 = 100.0% | 100/100 = 100.0% | 0/130 = 0.0% | 0/1 = 0.0% |
| 1 | 15/54 = 27.8% | 68/111 = 61.3% | 83/103 = 80.6% | 64/210 = 30.5% | 0/2 = 0.0% |
| 2 | 47/54 = 87.0% | 60/106 = 56.6% | 18/83 = 21.7% | 60/235 = 25.5% | 0/2 = 0.0% |
| 3 | 66/67 = 98.5% | 85/103 = 82.5% | 0/64 = 0.0% | 5/246 = 2.0% | NA |
| 4 | 65/65 = 100.0% | 132/132 = 100.0% | 0/71 = 0.0% | 0/212 = 0.0% | NA |

Interpretation: in order 4, predictions classified as initial-only or both are always correct in this run set, while final-only and other predictions are always wrong.

--------

## Order 4: Both Vs Initial-Only

| Order-4 target class | Samples | Predictions | Correct | Accuracy |
| --- | --- | --- | --- | --- |
| initial-only | 31 | 248 | 65 | 26.2% |
| both | 29 | 232 | 132 | 56.9% |

| Comparison | Both acc | Initial-only acc | Risk diff | Phi | p-value | Odds ratio |
| --- | --- | --- | --- | --- | --- | --- |
| both vs initial-only | 56.9% | 26.2% | +30.7 pp | 0.312 | 8.5e-12 | 3.69 |

Interpretation: this is more informative than simply saying order 4 is all endpoint targets. Within order 4, the both class is much easier than initial-only, with a +30.7 percentage-point difference.

--------

## Order 4 Prediction Outcomes

| Outcome | Predicted class | Count | Share of all order-4 predictions | Share within outcome |
| --- | --- | --- | --- | --- |
| correct | initial-only | 65 | 13.5% | 33.0% |
| correct | both | 132 | 27.5% | 67.0% |
| wrong | final-only | 71 | 14.8% | 25.1% |
| wrong | other | 212 | 44.2% | 74.9% |

Interpretation: correct order-4 predictions all land in initial-only or both; wrong order-4 predictions land in final-only or other.

--------

## Target Class Association With Correctness

| Scope | Target class | Class acc | Rest acc | Risk diff | Phi | p-value | Odds ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all orders | initial-only | 28.7% | 50.1% | -21.4 pp | -0.193 | 2.59e-21 | 0.40 |
| all orders | both | 58.8% | 35.1% | +23.6 pp | 0.231 | 1.13e-29 | 2.63 |
| all orders | final-only | 46.5% | 43.6% | +2.9 pp | 0.023 | 0.267 | 1.13 |
| all orders | other | 33.6% | 46.1% | -12.5 pp | -0.093 | 5.77e-06 | 0.59 |
| order 0 | both | 74.6% | 44.6% | +30.0 pp | 0.306 | 2.03e-11 | 3.62 |
| order 0 | final-only | 44.6% | 74.6% | -30.0 pp | -0.306 | 2.03e-11 | 0.28 |
| order 1 | initial-only | 37.5% | 48.9% | -11.4 pp | -0.063 | 0.168 | 0.64 |
| order 1 | both | 70.8% | 42.2% | +28.6 pp | 0.229 | 5.03e-07 | 3.29 |
| order 1 | final-only | 49.4% | 47.1% | +2.3 pp | 0.022 | 0.632 | 1.10 |
| order 1 | other | 36.4% | 54.6% | -18.2 pp | -0.176 | 0.000116 | 0.48 |
| order 2 | initial-only | 29.4% | 43.1% | -13.8 pp | -0.133 | 0.00352 | 0.55 |
| order 2 | both | 50.0% | 34.7% | +15.3 pp | 0.136 | 0.0029 | 1.88 |
| order 2 | final-only | 45.0% | 38.0% | +7.0 pp | 0.040 | 0.381 | 1.34 |
| order 2 | other | 37.5% | 39.1% | -1.6 pp | -0.015 | 0.74 | 0.94 |
| order 3 | initial-only | 29.5% | 35.2% | -5.7 pp | -0.061 | 0.184 | 0.77 |
| order 3 | both | 40.9% | 26.1% | +14.8 pp | 0.156 | 0.000622 | 1.95 |
| order 3 | other | 10.4% | 35.0% | -24.5 pp | -0.157 | 0.000575 | 0.23 |
| order 4 | initial-only | 26.2% | 56.9% | -30.7 pp | -0.312 | 8.5e-12 | 0.27 |
| order 4 | both | 56.9% | 26.2% | +30.7 pp | 0.312 | 8.5e-12 | 3.69 |

Interpretation: positive phi means the target class is associated with higher correctness. Across all orders, `both` has the strongest positive association; `initial-only` and `other` are negative.

--------

## Order 3 Vs Order 4 By Run

| Run | Order 3 accuracy | Order 4 accuracy | Order4 - Order3 |
| --- | --- | --- | --- |
| perceptom_0.6b | 20/60 = 33.3% | 21/60 = 35.0% | +1.7 pp |
| perceptom_1.7b | 29/60 = 48.3% | 38/60 = 63.3% | +15.0 pp |
| simtom_0.6b | 21/60 = 35.0% | 30/60 = 50.0% | +15.0 pp |
| simtom_1.7b | 20/60 = 33.3% | 21/60 = 35.0% | +1.7 pp |
| soo_0.6b | 12/60 = 20.0% | 18/60 = 30.0% | +10.0 pp |
| soo_1.7b | 15/60 = 25.0% | 15/60 = 25.0% | +0.0 pp |
| vp_0.6b | 13/60 = 21.7% | 25/60 = 41.7% | +20.0 pp |
| vp_1.7b | 26/60 = 43.3% | 29/60 = 48.3% | +5.0 pp |

Interpretation: order 4 is higher than order 3 in most runs, but the target-class analysis shows that the bump is aligned with initial/both target structure, especially the both class.

--------

## Bottom Line

Order 4 should not be read as evidence of stronger fourth-order Theory-of-Mind reasoning. In this data slice, order 4 has no final-only or other gold targets; it consists entirely of initial-only and both targets. The both subset is much easier than initial-only, so the apparent order-4 advantage is best explained as a shortcut-aligned label distribution.
