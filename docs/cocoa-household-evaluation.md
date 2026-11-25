# COCOA household-object fine-tuning

This experiment expands the object-focused data beyond the small MP3D household
slice. It uses the local COCOA annotation file and retains household and
everyday-object categories while leaving out people and scene structures.

## Dataset

`data/cocoa-household-v1/` contains the prepared category-filtered crops.
`data/cocoa-household-v2/` is the training/evaluation version, filtered using
the **prepared mask** occlusion fraction to keep 5–90% hidden area. This second
filter removes crops where resizing erased the annotated occlusion and crops
where no visible object remains.

| Split | Crops | Distinct source photos | Occlusion bands represented |
|---|---:|---:|---|
| Train | 331 | 231 | 5–90% |
| Validation | 47 | 32 | 5–90% |
| Test | 24 | 20 | 5–90% |

The test set has 3 examples at 5–15%, 5 at 15–30%, 4 at 30–45%, 3 at 45–60%,
and 9 at 60–90%. It covers 14 normalized category names, including table, sofa,
bowl, plate, lamp, chair, book, bottle, oven, dish, keyboard, shoe, glass, and
mirror. No test examples exceed 90% occlusion. The source-photo split is
disjoint; none of the held-out test or validation photos was used to train the
initial checkpoint.

## Fine-tuning and held-out results

The 488,145-parameter U-Net was initialized from
`runs/cocoa-colab-v1/best.pt`, then fine-tuned for 12 epochs on the 331 training
crops. Training used a fresh AdamW optimizer at learning rate 0.0001, batch size
8, FP32, and train-only synthetic occlusion augmentation. Epoch 12 had the best
validation full-mask IoU (0.696), so its checkpoint was evaluated on the held-out
test set.

| Model | Full-mask IoU | Hidden-region IoU | Hidden recall | Visible-only IoU |
|---|---:|---:|---:|---:|
| Initial COCOA checkpoint | 0.570 | 0.207 | 0.274 | 0.541 |
| Household fine-tune | **0.764** | **0.569** | **0.721** | 0.541 |

The 60–90% test group contains 9 crops. Fine-tuning raised full IoU from 0.332
to 0.699, hidden-region IoU from 0.110 to 0.619, and hidden recall from 0.115
to 0.668 in that group. These are promising results on a small test set, not a
claim of broad generalization. COCOA's annotations are human judgments of hidden
silhouette extent, and the model predicts silhouettes only; it does not
reconstruct RGB appearance.

## Cross-dataset check on MP3D-Amodal

Without additional training or threshold tuning, the fine-tuned COCOA checkpoint
was evaluated on the separate MP3D household test split. Its 23 crops cover nine
categories from one held-out scene. Full-mask IoU was 0.687, hidden-region IoU
0.396, and hidden recall 0.526; the visible-only baseline IoU was 0.575. On the
five 60–90% occlusion crops, full IoU was 0.424 and hidden IoU was 0.276. This is
evidence of transfer across datasets, but the test is small and covers one scene.
Per-sample outputs are in `runs/eval-cocoa-household-to-mp3d-test-v1/`; a visual
sheet of the five severe cases is `runs/prediction-review-cocoa-to-mp3d-severe-v1.png`.

## Artifacts

- `configs/cocoa-household-finetune-local.json` — fine-tuning setup.
- `data/cocoa-household-v1/` — all prepared crops after category filtering.
- `data/cocoa-household-v2/` — the 5–90% actual-occlusion training split.
- `runs/cocoa-household-finetune-v1/` — weights, training history, and metrics.
- `runs/eval-cocoa-household-initial-test-v1/` — initial checkpoint on the same
  held-out test photos.
- `runs/eval-cocoa-household-finetune-test-v1/` — fine-tuned per-crop metrics.
- `runs/prediction-review-cocoa-household-severe-v1.png` — all nine 60–90%
  test examples, with input, visible mask, prediction, and full-mask target.

To reproduce data preparation, run `scripts/prepare_cocoa.py` on
`data/raw/COCO_amodal_train2014.json` with the 40 category names recorded in
`data/cocoa-household-v1/summary.json`, `--max-images 800`, and
`--max-objects 3000`. Then run `scripts/filter_dataset_categories.py` on that
output with `--min-occlusion 0.05 --max-occlusion 0.90`. The category-filtered
preparation uses the same photo-level split rule as the original COCOA subset.
