# Pix2Gestalt interactive workflow

The current project path uses pretrained Pix2Gestalt only; the experimental U-Nets
are not part of this workflow. The demo accepts one scene image, a visible-target
mask, and an obstruction mask. SAM ViT-B can create/refine the target mask from
positive and negative clicks when its checkpoint is available; both masks remain
editable with the white brush and eraser.

## Start

The Pix2Gestalt, taming-transformers, and CLIP source repositories are already
cloned into the ignored `third_party/` folder. Run this in an isolated GPU
environment after the required packages and model checkpoints have been placed
in the configured locations:

```powershell
python scripts/demo_pix2gestalt.py `
  --root "C:\Users\dell\Desktop\Amodal-Object-Completion" `
  --checkpoint "C:\Users\dell\Desktop\Amodal-Object-Completion\checkpoints\epoch=000005.ckpt" `
  --sam-checkpoint "C:\Users\dell\Desktop\Amodal-Object-Completion\checkpoints\sam_vit_b_01ec64.pth"
```

If SAM weights are unavailable, leave off `--sam-checkpoint` and paint the target
mask manually. The Pix2Gestalt checkpoint is approximately 15.5 GB and is not
downloaded automatically. The project currently has no local copy of that model
checkpoint; use a GPU session where it is already available or transfer it into
the checkpoint folder. The SAM ViT-B checkpoint is optional for manual-mask use.
A CUDA GPU with enough free memory is required by this configured inference path.

## Use

1. Upload a scene photograph. The two mask canvases start black.
2. In the first canvas, paint the visible part of the object to complete in white.
   Optional SAM workflow: click the target on the source image, use positive or
   negative clicks, then choose **Apply SAM prompts**; add more clicks and refine
   with the brush or eraser.
3. In the second canvas, paint the foreground obstruction in white. Keep visible
   target pixels out of this mask. The model crop is centered on the visible
   target, so increase context padding if the blocker overlap falls outside it.
4. Set the candidate count, sampling steps, seed, and crop context, then run.
5. Review each raw object candidate, full-scene 2D preview, and transparent PNG
   layer. Use **Choose candidate** and **Select candidate for export** to open one
   candidate's object, scene preview, and transparent PNG; download the selected
   PNG separately or download the ZIP to preserve every candidate and run input.

Each run writes into a new timestamped folder under `runs/gradio-pix2gestalt/`.
The UI only starts on localhost by default. `--share` deliberately creates a
public Gradio link. Optional Weights & Biases logging is off by default; when
enabled, it records scalar runtime settings and measurements only, not images.
The model cache keys checkpoints by resolved path, file size, and modification
time, so replacing a checkpoint in place triggers a fresh model load. Each run
records the checkpoint path and file identity in `metrics.json`; loading also
checks that the checkpoint is nonempty and contains compatible EMA model weights.

## What the outputs mean

Pix2Gestalt sees a square crop containing the object and surrounding context,
resized to its 256-pixel input. Its generated RGB and estimated foreground are
mapped back to the scene. The scene preview replaces predicted hidden-target
pixels only where they overlap the obstruction mask. The transparent layer puts
the completed object at the crop's original coordinates and preserves observed
target pixels. These are 2D visualization outputs: they do not infer scene depth,
repair the background behind the blocker, or establish the exact unseen appearance.

Foreground alpha is estimated with GrabCut and a visible-object seed. Pale object
parts and complicated backgrounds can confuse it; inspect the layer before reuse.
Pix2Gestalt itself remains pretrained in this project: the synthetic ABO pairs are
for controlled evaluation, not fine-tuning its weights.

## Current evaluation evidence

- 46 source-disjoint synthetic ABO test pairs: hidden-region RGB MAE 0.0670 versus
  0.2877 for leaving the corrupted input unchanged; inferred hidden silhouette
  IoU 0.9546 (threshold-derived, so treat the shape score cautiously).
- Controlled full-scene composites made from those predictions: hidden-object RGB
  MAE 0.0668 versus 0.2877 baseline and hidden IoU 0.9546. Pix2Gestalt received
  white-background product crops, not the room scene.
- 12 MP3D natural-room examples: mean hidden silhouette IoU 0.2694 and full-mask
  IoU 0.5509 versus 0.5637 visible-only. This small review is mixed/poor and has
  no ground-truth hidden RGB.
- Multi-candidate review and guidance ablation artifacts exist; these are sampling
  and conditioning probes, not proof that candidate ranking or mask conditioning
  improves completion quality.
- Four candidates were generated for each of four validation examples (16 total).
  Mean estimated hidden-silhouette IoU was 0.7014; this is only a 4-image,
  threshold-derived review. Oracle best-candidate scores use ground truth to pick
  an output and are not usable as a deployable selector. The interactive app now
  provides explicit human selection and one-click transparent-PNG export.
- The Gradio callback/export smoke test created two candidate previews, two
  transparent RGBA layers, and a ZIP on a Kaggle T4. W&B scalar logging was tested
  offline; cloud synchronization has not been tested.

See [evaluation evidence](evaluation-evidence.md), [scene workflow](scene-reconstruction.md),
and [candidate review](candidate-selection-experiment.md) for protocols and limits.
