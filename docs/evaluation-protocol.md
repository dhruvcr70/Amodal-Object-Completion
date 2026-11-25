# Fixed MP3D-Amodal evaluation protocol

## Dataset and split

`data/eval-v1/manifest.jsonl` freezes 12 examples from the supplied MP3D-Amodal `test` split. The source release marks that split scene-disjoint from train and validation, but it contains only one test scene; interpret results as a small held-out-scene experiment, not broad generalization. The subset covers 12 categories and occlusion fractions from 0.18 to 0.77. The selection and order are fixed; do not tune the model on these samples.

The archive `runs/mp3d-eval-v1.zip` contains all four source files per sample. At inference time, use only `image.png` and `visible_mask.png`. Keep `complete_mask.png` and `occluder_mask.png` sealed as evaluation references; do not pass them to the model.

## RGB baseline run

Use the same Pix2Gestalt checkpoint and settings used for the first real-sample run: 256×256, 50 DDIM steps, guidance scale 2.0, seed 42, one candidate. Save `completion.png` and a per-sample metrics/settings record under `runs/eval-v1/<sample_id>/`. Record runtime errors rather than silently dropping examples.

The pretrained baseline returns RGB completions, not binary masks. Review a contact sheet and fill the qualitative fields in `runs/eval-v1/results.csv`: visible-region preservation, object plausibility, completion plausibility, boundary artifacts, and concise failure notes. Do not report pixel-level MAE/PSNR or silhouette IoU/Dice for RGB output. Those require an explicit, validated way to extract a generated silhouette.

## Separate silhouette-model comparison

Once the U-Net is trained, evaluate its predicted amodal mask on the same fixed subset. Report per-sample and mean IoU and Dice against `complete_mask.png`; also report IoU/Dice restricted to the occluded target region using the provided occluder mask. Include thresholds, empty-mask handling, and aggregation method in the report. Keep generated RGB comparisons qualitative and separate from mask metrics.

## Results status

The corrected RGB Pix2Gestalt baseline completed all 12 fixed examples: 12 inference successes, 0 runtime failures, mean inference time 5.89 seconds, and mean peak allocation 5.04 GiB on a Tesla T4. Settings were 256×256, 50 DDIM steps, guidance scale 2.0, seed 42, and one candidate. The archive is now local under `runs/eval-v1/`; the per-sample outputs and metrics are saved there. Successful inference is not a quality score. The subset comes from one held-out scene and does not establish broad generalization.

## RGB qualitative review

The contact sheet (`runs/eval-v1/contact_sheet.png`) and per-sample ratings (`runs/eval-v1/results.csv`) were reviewed manually. Object plausibility and completion plausibility use 1–5 scales, where 5 is best. Boundary artifacts use 1–5, where 5 is most severe. Visible-region preservation is marked not rateable: the current runner returns an isolated object crop on a white background, not a scene composite, so it does not demonstrate pixel-preserving completion in the original photo.

The visual outcome is mixed to poor despite 12/12 successful inference calls. Several predictions are category mismatches (for example, a loveseat for the theater-seat label and curtains/wall decor for the fireplace label); others are tiny fragments, nearly blank, or have severe fused/gray regions. A few outputs are recognizable as plausible object shapes, but the batch does not yet demonstrate reliable object-specific completion. No aggregate quality score is claimed because these are subjective ratings on a small fixed subset. The archived outputs, inference CSV, and reference silhouettes are retained under `runs/eval-v1/`.

## RGB mask extraction and object-cutout integration

Two image-only foreground methods were compared with the reference full-object silhouettes after inference: border-connected white flood fill and GrabCut initialized with border-background and visible-target foreground hints. A corrected-data U-Net was trained for 8 epochs on 481 train crops and selected on 119 validation crops using the hidden-region objective. Its validation full IoU was 0.7081 and hidden IoU was 0.4004 at threshold 0.50; validation-only threshold selection chose 0.35 for hidden IoU 0.4082 (full IoU 0.7015). The 12-example held-out review subset was evaluated once after selection.

| Silhouette method | Full IoU | Hidden IoU | Hidden recall |
| --- | ---: | ---: | ---: |
| Visible-mask-only | 0.5637 | 0.0000 | 0.0000 |
| RGB white flood fill | 0.5518 | 0.2706 | 0.4643 |
| RGB GrabCut | 0.5571 | 0.2725 | 0.4497 |
| Corrected-data U-Net, threshold 0.50 | 0.5278 | 0.2095 | 0.3239 |
| Corrected-data U-Net, validation-selected threshold 0.35 | 0.5231 | 0.2314 | 0.3806 |

These are macro means over only 12 examples from one scene. The RGB mask-extraction approaches recover some hidden silhouette pixels but slightly lower full IoU versus the visible-only mask. The corrected-data U-Net validation score did not transfer to this small held-out subset. GrabCut was selected for the transparent-cutout visualization because it had the highest hidden IoU and full IoU among the tested extracted masks, although the difference from flood fill is small. Per-sample values and metric conventions are in `runs/eval-v1/rgb-derived-mask-metrics-v2/`.

`scripts/compose_completed_rgb_objects.py` produces an isolated transparent object: it takes the silhouette from GrabCut, copies known visible target RGB pixels exactly from the source crop, and fills the hidden region with Pix2Gestalt RGB. The 12 cutouts and sheet are under `runs/eval-v1/object-completions-v5/`. Ground-truth masks are not used to create these images. This does not composite the object back into the scene, and it cannot correct category mismatches or missing RGB detail from Pix2Gestalt. Inspect the preview as a prototype, not as a validated final system.

## Scene-composite post-processing probe

The baseline returns an isolated RGB crop on a near-white background. `scripts/compose_rgb_results.py` compares white-background flood fill and GrabCut, then makes scene-composite previews. Outputs are in `runs/eval-v1/composites-v0/` and `runs/eval-v1/composites-v1/`. Both may include gray artifacts or remove white object detail, and neither handles scene-layer ordering. The measured mask comparison and chosen transparent-cutout integration are documented above. Scene compositing remains an unsolved extension; the current output is an isolated object cutout.

## Predicted silhouette masks on the fixed evaluation subset

The U-Net masks used to prepare the Pix2Gestalt conditioning ablation were compared with the sealed `complete_mask.png` references after inference. Predictions use the saved 0.5 threshold and union with the visible mask. Scores are macro averages over the fixed 12 examples; hidden-region IoU/Dice compare predictions and targets after removing visible pixels.

| Measure | U-Net amodal prediction | Visible-mask-only baseline |
| --- | ---: | ---: |
| Full-mask IoU | 0.6156 | 0.5637 |
| Full-mask Dice | 0.7350 | 0.6981 |
| Hidden-region IoU | 0.2660 | — |
| Hidden-region Dice | 0.3785 | — |
| Hidden-region recall | 0.3341 | — |

The full-mask IoU gain is 0.0519 over copying only the visible mask. Hidden-region recovery remains limited. All 12 samples belong to the same held-out scene. This is a small within-scene subset, not a cross-scene evaluation. These scores evaluate the custom U-Net silhouette only; do not interpret them as metrics for Pix2Gestalt's RGB completions. Per-sample values are in `runs/eval-v1/mask_metrics.csv`; the summary is in `runs/eval-v1/mask_metrics_summary.json`; the visual comparison is `runs/eval-v1/mask_contact_sheet.png`.

## Full held-out test evaluation — 176 examples

After completing the paired Pix2Gestalt ablation, the saved U-Net checkpoint (`runs/mp3d-v5-combined-val/best.pt`) was evaluated once on all 176 samples in the fixed MP3D-Amodal test split. The evaluation used the saved 0.5 threshold, unioned the predicted silhouette with the visible mask, and computed per-sample scores before taking macro means. The split is scene-disjoint from training/validation, but all 176 test examples come from the same held-out scene.

| Measure | U-Net | Visible-mask-only baseline |
| --- | ---: | ---: |
| Full-mask IoU | 0.6116 | 0.6116 |
| Full-mask Dice | 0.7376 | 0.7376 |
| Hidden-region IoU | 0.3191 | — |
| Hidden-region Dice | 0.4479 | — |
| Hidden-region recall | 0.6026 | — |

The model and visible-only baseline have effectively identical full-mask scores on the full test set (IoU difference about 0.00005). The U-Net recovers some hidden pixels, but hidden-region precision is limited; the lower hidden-region IoU/Dice reflects that trade-off. The earlier 12-example subset showed a larger full-mask gain and is not representative of the complete test set. Do not tune to this test set; use validation for future model choices.

Artifacts: `runs/eval-v2-full-test/summary.json` and `runs/eval-v2-full-test/per_sample_metrics.csv`. The test set has only one scene, so these results do not establish broad cross-scene generalization.


## Validation-selected v4 checkpoint and threshold analysis

For model selection, four saved checkpoints were evaluated on the validation split using the same per-sample macro metrics. The v4 checkpoint (`runs/mp3d-v4-hidden-weight-val/best.pt`) was selected for the hidden-region objective: it reached validation hidden IoU 0.33 and hidden Dice 0.45, compared with v2 (0.29/0.40), v3 (0.31/0.44), and v5 (0.32/0.45). v2 retained the highest validation full-mask IoU (0.67), so v4 is a trade-off favoring recovery of occluded pixels rather than the best overall silhouette overlap. Detailed validation runs are under `runs/eval-val-model-selection/`.

A threshold sweep on validation data selected 0.40 for hidden IoU (0.3275; hidden Dice 0.4542; hidden recall 0.5774). At 0.50, those validation metrics were 0.3251, 0.4493, and 0.5256. This is a modest validation difference, not a large performance jump. Sweep data is in `runs/eval-val-threshold-sweep-v4/`.

The v4 checkpoint was then evaluated on all 176 test samples at the validation-selected 0.40 threshold. It scored full-mask IoU 0.6191 and Dice 0.7444, versus the visible-only baseline at 0.6116 and 0.7376. Hidden-region IoU was 0.3115, Dice 0.4343, and recall 0.5407. The same checkpoint at threshold 0.50 scored full IoU 0.6280, hidden IoU 0.3052, hidden Dice 0.4237, and recall 0.4877. Thus threshold 0.40 improves hidden-region recall/overlap slightly on this test scene while lowering full-mask overlap; the overall gain remains modest. Because earlier project iterations also evaluated this fixed test split, treat this as a transparent exploratory result rather than a pristine blind final evaluation. All 176 examples come from one held-out scene, so the test does not establish generalization across scenes.

Artifacts: `runs/eval-v3-v4-test/summary.json` (threshold 0.50), `runs/eval-v4-threshold-04-test/summary.json` (threshold 0.40), and `runs/eval-val-threshold-sweep-v4/`.

## Visual review artifact

Created a four-column contact sheet for 12 evenly spaced examples from the held-out test split: RGB input, visible-mask overlay, v4 predicted amodal silhouette at threshold 0.40, and ground-truth silhouette. The sheet is a qualitative debugging aid; it reveals that predictions can be too small or extend into background, so the aggregate hidden-region scores should not be read as reliable object reconstruction. The contact sheet does not use ground truth as model input.

Recreate it with:

```powershell
& 'C:\Users\dell\Desktop\moba-rl\moba-rl\.venv\Scripts\python.exe' scripts\make_prediction_review.py
```

Output: `runs/prediction-review-v4-test.png`.

## Validation error breakdown and targeted augmentation experiment

At the validation-selected v4 threshold of 0.40, full-mask IoU and hidden recall by occlusion severity were:

| Hidden fraction | Samples | Full-mask IoU | Visible-only IoU | Hidden IoU | Hidden recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| <15% | 7 | 0.63 | 0.89 | 0.13 | 0.68 |
| 15–30% | 57 | 0.75 | 0.77 | 0.38 | 0.69 |
| 30–45% | 36 | 0.64 | 0.64 | 0.34 | 0.57 |
| 45–60% | 26 | 0.52 | 0.49 | 0.26 | 0.43 |
| 60–90% | 19 | 0.47 | 0.29 | 0.31 | 0.42 |

This breakdown motivated a train-only larger-occluder augmentation experiment. Its v6 checkpoint improved hidden-region IoU and recall in the 60–90% validation subgroup at threshold 0.50 (0.34/0.51 versus v4's 0.29/0.38), while reducing overall validation full-mask IoU and hidden-region IoU. It is a specialization trade-off, not an overall upgrade; all conclusions are limited to this one validation scene. See `mp3d-data.md` for the experiment configuration and comparison.

The validation contact sheets are `runs/prediction-review-v4-val.png` and `runs/prediction-review-v6-val.png`. Recreate a sheet with `scripts/make_prediction_review.py`, choosing a checkpoint, split, and threshold explicitly when needed.

## RGB batch runner

scripts/evaluate_rgb_baseline_batch.py is installed by the Colab notebook and runs the fixed archive one example at a time through Pix2Gestalt. It saves generated RGB images, per-sample timing/settings/error records, a CSV with blank qualitative-review fields, a contact sheet, and a ZIP. Only image.png and visible_mask.png are supplied to the generator; ground-truth masks remain sealed. The runner does not claim pixel-level RGB metrics because plausible completions can differ in unseen texture and color while still being reasonable.

