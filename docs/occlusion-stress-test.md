# MP3D-Amodal occlusion stress test

This report adds a harder real-photo evaluation to the earlier COCOA run. The
COCOA test had only five examples in the 60–90% occlusion band. MP3D-Amodal uses
3D-derived full-object annotations and provides 29 held-out examples in that
band, plus 67 examples at 45–90% occlusion.

## Setup

- Dataset: `data/mp3d-subset-v3`, fixed scene-disjoint test split.
- Test size: 176 crops from one held-out Matterport scene.
- Models: original U-Net (`runs/mp3d-v2/best.pt`) and the synthetic-occlusion
  training ablation (`runs/mp3d-v3-synthetic-occlusion/best.pt`).
- Prediction threshold: 0.50. Predicted completion is unioned with the visible
  mask, preserving all observed object pixels.
- Metrics: per-sample macro means. “Visible baseline” copies the observed mask
  without predicting hidden pixels.

| Hidden fraction | Samples | Original full IoU | Augmented full IoU | Visible baseline IoU | Original hidden IoU | Augmented hidden IoU | Augmented hidden recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–15% | 10 | 0.648 | 0.615 | 0.905 | 0.160 | 0.158 | 0.798 |
| 15–30% | 69 | 0.771 | 0.749 | 0.776 | 0.292 | 0.339 | 0.553 |
| 30–45% | 30 | 0.687 | 0.694 | 0.634 | 0.277 | 0.352 | 0.459 |
| 45–60% | 38 | 0.550 | 0.566 | 0.472 | 0.230 | 0.292 | 0.376 |
| 60–90% | 29 | 0.366 | 0.397 | 0.279 | 0.170 | 0.226 | 0.286 |

Across all 176 examples, the original model scored 0.635 full IoU and 0.248
hidden-region IoU; the augmentation model scored 0.634 and 0.302 respectively.
The visible-only baseline scored 0.612 full IoU. In the hardest 60–90% group,
synthetic-occlusion training raised hidden-region IoU by 0.056 and hidden recall
by 0.075 over the original model, while full IoU rose by 0.030. Completion is
still weak in this band: the augmented model's hidden IoU is 0.226.

## Interpretation and limits

The results support a narrow conclusion: this augmentation helped recover more
hidden silhouette pixels on this held-out scene, especially as occlusion rose.
It did not improve aggregate full IoU, and only 29 examples represent the
60–90% band. The release has four scenes and this test split contains one, so
this is not broad cross-scene validation. The MP3D test split was also used in
the earlier exploratory augmentation comparison; treat this report as a more
careful breakdown of existing results, not as a newly untouched benchmark.
These metrics concern binary silhouettes only, not RGB appearance completion.

## Reproduction and artifacts

The full per-sample outputs, summaries, and dataset fingerprint are saved in:

- `runs/eval-mp3d-balanced-test-v1/` — original U-Net.
- `runs/eval-mp3d-augmentation-test-v1/` — synthetic-occlusion ablation.
- `runs/prediction-review-severe-occlusion-v1.png` — 12 examples sampled across
  the 29 held-out cases with 60–90% occlusion, showing input, visible mask,
  augmented prediction, and full-mask target.

Both were evaluated with `scripts/evaluate_full_test.py` on the same dataset,
test split, and threshold. The CSVs contain each sample's actual hidden fraction,
full-mask IoU, hidden-region IoU and recall, and visible-only baseline IoU.
The severe-case review sheet can be regenerated with
`scripts/make_prediction_review.py` and the same checkpoint, split, and threshold,
using `--min-occlusion 0.60 --max-occlusion 0.90 --count 12`.
