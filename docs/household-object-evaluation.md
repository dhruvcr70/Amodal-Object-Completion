# Household-object evaluation

The broad MP3D-Amodal subset contains many scene structures. This evaluation
filters the existing, scene-disjoint split to furniture and household items,
then scores the already-trained baseline and synthetic-occlusion model on the
same filtered test examples. No images were reassigned between splits.

## Category filter and available data

Included categories: armchair, bath cabinet, bedframe, bedside table, cabinet,
chair, couch, curtain, decoration, dining table, dish cabinet, drawer, footstool,
lamp, mirror, pillow, plant, refrigerator, round table, shelves, shelving, soap
dish, sofa chair, stool, table, table shelf, TV, and wardrobe.

| Split | Samples | Source photos | Scenes |
|---|---:|---:|---:|
| Train | 44 | 44 | 2 |
| Validation | 28 | 28 | 1 |
| Test | 23 | 23 | 1 |

The filtered test contains nine categories: shelving (5), curtains (4),
refrigerators (4), cabinets (3), couches (3), and one each of mirror, wardrobe,
table, and drawer. Only five test cases have 60–90% occlusion. The small training
set is not enough to justify a from-scratch household-only model yet.

The exact filtered image/mask subset is materialized in
`data/mp3d-household-v1/`; it retains the original scene assignments and has 44
training, 28 validation, and 23 test crops. The broad-data checkpoints were
evaluated on this household-object slice. A third model was then initialized
from the synthetic-occlusion checkpoint and fine-tuned on the 44 household
training crops; its best checkpoint was selected by validation IoU at epoch 10.
This is a small transfer-learning experiment, not a from-scratch result.

## Results

| Model | Full-mask IoU | Hidden-region IoU | Hidden recall | Visible-only baseline IoU |
|---|---:|---:|---:|---:|
| Original U-Net | 0.597 | 0.244 | 0.359 | 0.575 |
| Synthetic-occlusion training | 0.618 | 0.313 | 0.473 | 0.575 |
| Household fine-tune | 0.655 | 0.379 | 0.527 | 0.575 |

| Hidden fraction | Samples | Original full IoU | Augmented full IoU | Original hidden IoU | Augmented hidden IoU |
|---|---:|---:|---:|---:|---:|
| 0–15% | 2 | 0.619 | 0.614 | 0.380 | 0.313 |
| 15–30% | 6 | 0.780 | 0.768 | 0.300 | 0.369 |
| 30–45% | 4 | 0.668 | 0.685 | 0.209 | 0.281 |
| 45–60% | 6 | 0.578 | 0.650 | 0.226 | 0.377 |
| 60–90% | 5 | 0.333 | 0.346 | 0.175 | 0.195 |

For the fine-tuned model, the 60–90% group scored 0.391 full IoU, 0.260
hidden-region IoU, and 0.302 hidden recall, compared with 0.346, 0.195, and
0.251 for the broad-data augmentation model. This group still contains only
five examples.

The augmentation model scored better overall in this small filtered slice, with
the largest hidden-region gains at 45–60% occlusion. The most severe group has
only five examples, and hidden IoU remains low. These are exploratory results
from the same test scene used in the earlier MP3D experiments, not an untouched
benchmark or evidence of broad category generalization.

## Artifacts and reproduction

- `runs/eval-mp3d-household-test-v1/` — original U-Net per-example CSV and summary.
- `runs/eval-mp3d-household-augmentation-test-v1/` — augmented model per-example
  CSV and summary.
- `runs/mp3d-household-finetune-v1/` — fine-tuned weights, history, and training
  metrics.
- `runs/eval-mp3d-household-finetune-test-v1/` — fine-tuned model per-example
  CSV and summary.
- `runs/prediction-review-household-severe-v1.png` — five severe-occlusion
  household examples with predictions and full-mask targets.
- `scripts/filter_dataset_categories.py` — reproduces the filtered dataset copy.
- `configs/mp3d-household-finetune-local.json` — fine-tuning setup.

The exact category names are included in each summary. Recreate the filtered
evaluation with `scripts/evaluate_full_test.py --categories ...` using the same
category list, test split, and threshold 0.50. The script retains category names
in the per-example CSV to support category-level analysis.
