# MP3D-Amodal real data

MP3D-Amodal is an Oxford research dataset for amodal segmentation. Its official
paper uses 3D scene data to produce authentic full-object masks for real photos,
addressing the subjectivity of manual amodal labels. The official training
release used here contains 1,283 annotated photos across four scenes.

## Data preparation

The project downloads the official training photo archive and annotation file to
`data/raw/` and reads them directly. The photos are not fully extracted. The
annotation reader only permits the NumPy constructors present in the official
pickle and rejects unexpected globals.

Prepare a bounded subset with:

```powershell
& 'C:\Users\dell\Desktop\moba-rl\moba-rl\.venv\Scripts\python.exe' scripts\prepare_mp3d.py `
  --annotations data\raw\mp3d_train_annotations.pkl `
  --images-zip data\raw\mp3d_train.zip `
  --output data\mp3d-subset-v3 `
  --max-images 1200 `
  --max-objects 900
```

The script keeps scenes disjoint across train, validation, and test. Because the
training release covers four scenes, the held-out environment evaluation is
small. Each example is a padded 128×128 crop with RGB, visible mask, and full
mask. Crops with less than 5% or more than 90% hidden area after resizing are
excluded so the silhouette task remains learnable. `occluder_mask` is derived as
`full_mask AND NOT visible_mask`.

## Train the silhouette baseline

Run the **Train on MP3D-Amodal subset** task from the VS Code task menu, or:

```powershell
& 'C:\Users\dell\Desktop\moba-rl\moba-rl\.venv\Scripts\python.exe' scripts\train_silhouette.py --config configs\mp3d-local.json
```

The current model predicts full silhouettes. It does not generate missing RGB
appearance. Keep this real-data experiment separate from the procedural-shape
run so the results remain easy to interpret.

## Synthetic-occlusion comparison

The existing MP3D run is the no-extra-augmentation baseline. To compare a fixed
augmentation hypothesis on the same scene-disjoint split, use:

```powershell
& 'C:\Users\dell\Desktop\moba-rl\moba-rl\.venv\Scripts\python.exe' scripts\train_silhouette.py --config configs\mp3d-augmentation-local.json
```

This adds a reproducible horizontal flip and, on 70% of training samples, one
random rectangle or ellipse overlay covering 8–40% of the full mask. The complete
mask remains the target. Validation and test examples are never augmented. Results
are written separately to `runs/mp3d-v3-synthetic-occlusion/`.

## First measured run

The first expanded experiment used 581 crops: 260 train, 145 validation, and 176
test examples, with all four scenes kept disjoint between splits. The 488,145-
parameter U-Net trained for eight epochs in FP32 on the local NVIDIA T600.

- Best validation IoU: 0.667 at epoch 7.
- Held-out test IoU: 0.635, compared with 0.612 for the visible-mask-only baseline.
- Hidden-region IoU: 0.248.
- Test IoU for samples with 45–60% occlusion: 0.470 across 67 examples.

This is an early baseline, not a strong completion result. The model gives a
small overall gain over copying the visible mask, while hidden-region accuracy
is still limited. The held-out test contains one scene, so it is an initial
cross-scene check rather than broad evidence of generalization. Results and
visualizations are saved in `runs/mp3d-v2/`.

## Synthetic-occlusion ablation

The follow-up run kept the same 488,145-parameter U-Net, 260/145/176
scene-disjoint split, seed, optimizer, batch size, and eight-epoch budget. Only
training-time augmentation changed: a deterministic horizontal flip and an
additional scene-patch occluder on 70% of training samples. Validation and test
inputs were unchanged. Results are saved in `runs/mp3d-v3-synthetic-occlusion/`.

| Test measure | No extra augmentation | Synthetic occlusion | Change |
| --- | ---: | ---: | ---: |
| Full-mask IoU | 0.6350 | 0.6344 | -0.0005 |
| Hidden-region IoU | 0.2484 | 0.3022 | +0.0538 |
| Visible-only baseline IoU | 0.6116 | 0.6116 | 0.0000 |
| IoU at 45–60% occlusion | 0.4702 | 0.4928 | +0.0226 |

The augmentation preserved the small aggregate gain over copying the visible
mask and improved hidden-region IoU, especially on the more occluded subset, but
did not improve overall test IoU. Best validation IoU was also lower (0.6487 vs.
0.6671), so the augmented checkpoint is not a replacement for the original
baseline. Treat these results as a single ablation on one held-out scene, not
evidence that the augmentation reliably helps across environments.

## Hidden-positive loss tuning

The next validation-only experiment weights positive pixels that are hidden from
the input twice as strongly in both BCE and Dice terms. It uses the original
training examples, same split, seed, model, and eight-epoch budget. Run it with:

```powershell
& 'C:\Users\dell\Desktop\moba-rl\moba-rl\.venv\Scripts\python.exe' scripts\train_silhouette.py --config configs\mp3d-hidden-weight-local.json --skip-test
```

`--skip-test` keeps the held-out test scene out of this tuning run. Validation
metrics and previews are saved under `runs/mp3d-v4-hidden-weight-val/`; only a
later chosen candidate should be evaluated on the test split.

The two changes can also be combined as a validation-only interaction check:

```powershell
& 'C:\Users\dell\Desktop\moba-rl\moba-rl\.venv\Scripts\python.exe' scripts\train_silhouette.py --config configs\mp3d-combined-augmentation-local.json --skip-test
```

Its separate run folder is `runs/mp3d-v5-combined-val/`.

### Validation-only comparison

Each model is selected by its best validation full-mask IoU. Hidden-region IoU
below is measured at that same epoch; the visible-only baseline is 0.628 for this
validation split.

| Training setup | Best epoch | Validation full-mask IoU | Validation hidden-region IoU |
| --- | ---: | ---: | ---: |
| Original baseline | 7 | 0.6671 | 0.2853 |
| Synthetic occlusion + flips | 4 | 0.6487 | 0.3139 |
| Hidden-positive weight 2× | 7 | 0.6482 | 0.3251 |
| Both changes together | 7 | 0.6127 | 0.3168 |

The original baseline remains the best overall model. Hidden-positive weighting
improved hidden-region validation IoU but reduced full-mask IoU; combining it with
synthetic occlusion reduced both relative to the single-change runs. No test
evaluation was run for these two loss experiments. Keep the baseline checkpoint
as the current overall choice and treat the alternative checkpoints as ablations.

## Source and citation

Dataset and code: [Amodal Ground Truth and Completion in the Wild](https://www.robots.ox.ac.uk/~vgg/research/amodal/), CVPR 2024.

## Severe-occlusion augmentation experiment (v6, validation only)

Validation examples were stratified by the fraction of the object's ground-truth silhouette hidden. The v4 model's full-mask IoU declined from 0.75 at 15–30% occlusion to 0.47 at 60–90%. That pointed to severe occlusion as a specific weakness. A new train-only augmentation experiment samples larger synthetic occluders (25–75% of the object, with occluder dimensions up to 85% of the image) while preserving the v4 hidden-positive loss weight. Validation and test examples receive no augmentation.

The v6 model did not replace v4 overall. At the standard 0.50 threshold, validation full-mask IoU fell from 0.6482 (v4) to 0.5825 (v6), and overall hidden-region IoU fell from 0.3251 to 0.2945. However, in the 19 validation samples with 60–90% occlusion, hidden-region IoU increased from 0.29 to 0.34 and hidden recall from 0.38 to 0.51. This suggests a possible specialization trade-off on the most occluded cases, but the subgroup is small and drawn from one validation scene. Do not present v6 as a general improvement or evaluate it on test yet.

| Validation occlusion | Samples | v4 hidden IoU / recall | v6 hidden IoU / recall |
| --- | ---: | ---: | ---: |
| 15–30% | 57 | 0.39 / 0.64 | 0.31 / 0.72 |
| 30–45% | 36 | 0.34 / 0.51 | 0.31 / 0.65 |
| 45–60% | 26 | 0.25 / 0.39 | 0.27 / 0.50 |
| 60–90% | 19 | 0.29 / 0.38 | 0.34 / 0.51 |

The 0.40 threshold sweep for v6 did not recover overall performance: its best validation hidden IoU was 0.298 at threshold 0.60, still below v4's 0.327 at threshold 0.40. Keep v4 as the main checkpoint; retain v6 as a targeted ablation. Artifacts: `configs/mp3d-severe-occlusion-local.json`, `runs/mp3d-v6-severe-occlusion-val/`, `runs/eval-val-threshold-sweep-v6/`, and `runs/prediction-review-v6-val.png`.
