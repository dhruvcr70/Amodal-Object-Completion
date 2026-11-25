# Amodal completion: evaluation evidence

**Current project scope:** pretrained Pix2Gestalt only. Custom U-Net results below
are retained as historical auxiliary experiments; they are not in the active
pipeline and should not be attributed to Pix2Gestalt.

## What was evaluated

The project has two related but different outputs:

1. **Pix2Gestalt RGB completion:** predict a plausible hidden appearance for an object.
2. **Historical silhouette experiments:** separate U-Net work, excluded from the current pipeline.

They must be evaluated separately. A plausible-looking completion is not proof that the hidden pixels or shape are correct.

## Historical RGB U-Net completion (not the active model)

The custom RGB U-Net was evaluated on the held-out test portion of `data/abo-rgb-pairs-v1`.

- **Test size:** 46 occluded images from 23 products, with product identities disjoint from training and validation.
- **Input:** synthetically occluded product image and visible-object mask.
- **Reference:** original unobstructed product image.
- **Metric:** mean absolute RGB error on the synthetic occluder region, with pixel values normalized to `[0,1]`.
- **The model receives the visible-object mask**, as intended by this pipeline. The test-only blocker mask and unobstructed target are used for scoring, not inference.

| Method / region | RGB MAE (lower is better) |
|---|---:|
| Leave input unchanged in occluded region | 0.3397 |
| Project RGB model in occluded region | **0.1347** |
| Project RGB model on hidden object pixels inside the blocker | **0.1536** |

The occluded-region MAE is about **60.4% lower** than leaving the synthetic obstruction in place. This is evidence that the model learned to fill synthetic occlusions on this product-image distribution.

**Limits:** this is a small test from one catalog shard. Images are mostly isolated products on white backgrounds and blockers are synthetic. It does not establish performance on arbitrary room scenes, natural occluders, or physically hidden geometry. Pixel error also does not by itself measure whether a completion is semantically plausible.

Artifacts: [training/evaluation metrics](../runs/abo-rgb-unet-v1/metrics.json), [held-out visual examples](../runs/abo-rgb-unet-v1/test_preview.png).

### Pix2Gestalt RGB completion on controlled ABO occlusions

The pretrained Pix2Gestalt model was evaluated on 46 test inputs covering 23 product identities, with two synthetic occlusion layouts per product. The product identities were held out from this project's train/validation split. Inference received only the occluded RGB crop and visible-object mask; the clean target image and blocker/complete masks were used only after inference to score the hidden region.

| Measure | Pix2Gestalt | Corrupted input unchanged | White-fill baseline |
| --- | ---: | ---: | ---: |
| Hidden-region RGB MAE (lower is better) | **0.0670** | 0.2877 | 0.4264 |
| Hidden-region PSNR (dB; higher is better) | **20.06** | — | — |
| Hidden-region SSIM (higher is better) | **0.6513** | — | — |

Pix2Gestalt's hidden-region RGB MAE was 76.7% lower than leaving the synthetic obstruction in place. This is measurable evidence of pixel-aligned RGB improvement on this controlled product-image benchmark. An image-only near-white foreground extraction estimated hidden-silhouette IoU at 0.9546; treat that shape score cautiously because thresholding can fail on white object regions. The test is small, uses catalog products on mostly white backgrounds and synthetic blockers, and does not establish performance on natural room scenes. The test identities are disjoint from our own split, but overlap with Pix2Gestalt's original pretraining data has not been audited.

Supplementary analysis of the same saved predictions found raw visible-region RGB
MAE of **0.0291**. The interactive scene compositor separately copies the marked
visible target pixels exactly. Grouping results by occluded-object fraction gives:

| Occluded fraction | Samples (unique products) | Hidden RGB MAE | Hidden silhouette IoU |
| --- | ---: | ---: | ---: |
| 15–30% | 27 (18) | 0.0720 | 0.9588 |
| 30–45% | 13 (13) | 0.0630 | 0.9534 |
| 45–60% | 6 (6) | 0.0534 | 0.9379 |

These bins are small and have no examples outside 15–60%; they do not establish
that more severe occlusion is easier. Means and standard deviations are descriptive,
not confidence intervals. The supplementary script and full results are
[`analyze_pix2gestalt_abo_rgb.py`](../scripts/analyze_pix2gestalt_abo_rgb.py) and
[`supplementary-analysis.json`](../runs/pix2gestalt-abo-rgb-eval-v1/recomputed-v2/supplementary-analysis.json).

The current re-scored artifact is [the per-sample report](../runs/pix2gestalt-abo-rgb-eval-v1/recomputed-v2/per_sample_metrics.csv), [aggregate metrics and protocol](../runs/pix2gestalt-abo-rgb-eval-v1/recomputed-v2/summary.json), and [46-example contact sheet](../runs/pix2gestalt-abo-rgb-eval-v1/recomputed-v2/contact_sheet.png). The earlier top-level metrics are retained as the initial scoring pass; use `recomputed-v2` for the figures above.

### Full-scene composition on controlled synthetic room scenes

To check the scene-composition stage without returning to the dining-table demo, all 46 held-out Pix2Gestalt predictions were placed into a room photograph from MP3D. Each object was inserted with its known complete mask; a known synthetic blocker was then added. The clean, unblocked composite was retained as the exact scene reference. The compositor received only the occluded scene, object prediction and masks; it did not receive the clean reference.

| Measure | Reconstruction | Occluded input unchanged |
| --- | ---: | ---: |
| Hidden-object RGB MAE | **0.0668** | 0.2877 |
| All blocker-region RGB MAE | **0.0737** | 0.2738 |
| Hidden-object IoU | **0.9546** | — |
| Hidden-object SSIM | **0.6173** | — |

The obstruction-region score includes both hidden object pixels and blocker pixels outside the object. The outside-object background MAE was 0.0863 on the 41 scenes with such pixels. The other five blockers fell entirely within their target objects, so that background-only score is undefined for them.

This evaluates full-frame compositing and inpainting against known synthetic scenes. Pix2Gestalt reused predictions generated on white-background catalog crops; its model did not see the room photo. The catalog cutouts were pasted into one real room background with a fixed placement, so this is not evidence of natural-scene understanding or realistic object placement. Artifacts: [summary](../runs/controlled-scene-eval-v2/summary.json), [per-scene metrics](../runs/controlled-scene-eval-v2/per_scene_metrics.csv), [46-scene contact sheet](../runs/controlled-scene-eval-v2/reconstructed_scene_contact_sheet.jpg), and [input/reference/output comparison](../runs/controlled-scene-eval-v2/example_scene_comparison.png).

## Amodal silhouette: real-scene evidence

### Historical custom U-Net trained on COCOA household examples, tested on MP3D-Amodal

- **Test size:** 23 examples from 23 photos in one held-out scene.
- **Full silhouette IoU:** 0.6869; visible-only baseline: 0.5747.
- **Hidden-region IoU:** 0.3959.
- **Hidden-region Dice:** 0.5102.
- **Hidden-region recall:** 0.5255.

This shows measurable hidden-shape prediction above the visible-only baseline on this cross-dataset scene, but the single-scene test is too small to claim broad real-world generalization.

Artifacts: [metrics](../runs/eval-cocoa-household-to-mp3d-test-v1/summary.json), [per-example scores](../runs/eval-cocoa-household-to-mp3d-test-v1/per_sample_metrics.csv).

### Pix2Gestalt baseline on 12 MP3D-Amodal examples

- **Mean hidden-region IoU:** 0.2694.
- **Mean hidden-region Dice:** 0.3454.
- **Mean full-silhouette IoU:** 0.5509; visible-only baseline: 0.5637.

On this small set, Pix2Gestalt's extracted silhouette does **not** beat the visible-only baseline on full-mask IoU. Its RGB result can look like a complete object, but these scores do not support claiming accurate reconstruction for these examples. Foreground masks were inferred from near-white background pixels in its generated RGB output, so extraction errors affect the score.

Artifact: [visual comparison](../runs/mp3d-eval-v1/evaluation/mask_comparison.png), [metrics](../runs/mp3d-eval-v1/evaluation/summary.json).

### SAM ViT-B mask to Pix2Gestalt handoff (Kaggle)

The official Meta SAM ViT-B checkpoint (`sam_vit_b_01ec64.pth`, SHA-256
`ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`) was loaded
on a Kaggle Tesla T4. On the 256×256 `rear-left-chair_input.png` example, SAM
produced a visible-object mask from point prompts, and that predicted mask was
passed to the loaded Pix2Gestalt model (50 DDIM steps, CUDA autocast). Pix2Gestalt
completed inference in 4.7–4.8 seconds. The contact sheet and predicted masks were
saved under Kaggle's `/kaggle/working/amodal-completion/runs/scene-table-occlusion-v1/`.

The SAM mask achieved IoU **0.5475** with one positive and four corner-negative
prompts; a repeatable set of six positive and four negative clicks achieved IoU
**0.5483**. For this prompt-set comparison, points were sampled using the known
visible mask; that reference mask was used only to choose/evaluate prompts, never
as Pix2Gestalt input. This verifies the SAM-to-Pix2Gestalt inference handoff, but
the segmentation overlap is modest and the prompt test is not a real user's click
session. The Gradio callback path was subsequently exercised in the section below;
the browser's brush/click interactions were not manually tested. Prompt refinement
and a broader mask-quality evaluation remain open.

### Gradio callback and export smoke test (Kaggle)

The project `demo_pix2gestalt.py` and its compositing helper were attached to the
Kaggle notebook as a **private** source bundle. The Gradio Blocks interface built
successfully; its loopback-only server returned HTTP 200, exposed the expected
mask, candidate, preview, and download controls in its config, and had no public
share URL. The server was closed after this check. The browser's brush/click
interactions were not manually operated.

The actual project callbacks were then called on the 894×894 synthetic table
scene. SAM used ten prompts placed from the known 256×256 visible-chair mask for
repeatable evaluation; that ground-truth mask was not passed into Pix2Gestalt.
The resulting SAM mask had IoU **0.4829** against the mapped visible-chair mask.
Despite this limited mask overlap, the completion callback successfully ran
Pix2Gestalt with two candidates and 30 DDIM steps, and produced two scene previews,
two RGBA object layers with transparent pixels, and a ZIP bundle. Pix2Gestalt
inference took **4.15 s** and peak allocated GPU memory was **5.62 GiB** on the T4.
These are one-run runtime measurements, not a performance benchmark; the SAM mask
quality remains too weak to claim reliable automatic segmentation.

The optional W&B scalar logging path was exercised with `WANDB_MODE=offline`:
one candidate at 20 DDIM steps took **1.98 s** and reported **5.40 GiB** peak
allocated memory. It logged runtime scalars locally without uploading a cloud run
or images. W&B cloud synchronization has not been tested.

The local app was updated with an explicit candidate dropdown, selected object and
scene previews, and a separate transparent-PNG download control. The Gradio app
build/config and selection callback were checked in the local isolated UI
environment. A browser-driven run of those controls has not yet been done. The
model cache now keys on checkpoint path, size, and modification time; each run
records checkpoint size and modification time, and load-time checks reject empty
or incompatible EMA checkpoints. A local Pix2Gestalt checkpoint is still absent,
so local GPU inference and an end-to-end selected-layer download were not rerun.

The same 12 saved Pix2Gestalt predictions were projected into their original, full-resolution MP3D photographs to inspect them in room context. The mean full-silhouette IoU was 0.5509, below the 0.5637 visible-only baseline; mean hidden-region IoU was 0.2694. The [full-room contact sheet](../runs/mp3d-natural-scene-v2/natural_room_contact_sheet.jpg), [sample reconstruction](../runs/mp3d-natural-scene-v2/mp3d-466-466-0/pix2gestalt_scene_completion.jpg), and [metrics](../runs/mp3d-natural-scene-v2/per_scene_metrics.csv) show these reconstructions. The object crops were enlarged to fit their original crop coordinates, and only predicted hidden-object pixels were pasted into the source photo. MP3D provides no ground-truth RGB for those hidden pixels, so this run supports shape and visual review only; it does not measure exact scene reconstruction.

## Current Pix2Gestalt conclusion

There is objective evidence for RGB filling on a narrow synthetic product benchmark. Pix2Gestalt does not beat the visible-only baseline on full-mask IoU in the small natural-room sample. The SAM ViT-B-to-Pix2Gestalt callbacks and Gradio page configuration have run on Kaggle, but SAM mask IoU was modest and the browser's brush/click interactions have not been manually exercised. The Pix2Gestalt checkpoint remains in Kaggle rather than the local project; W&B scalar logging was tested offline but not synced to the cloud. A single image does not reveal exact hidden appearance or geometry. Treat the result as a plausible completion hypothesis.

## What would make the claim stronger

1. Expand the RGB test to more unseen object identities, categories, and occluder shapes; keep all variants of each source product in one split.
2. Add natural-scene images with trustworthy complete-object references, or controlled captures of the same object before and after occlusion. Keep reference images sealed from inference.
3. Report hidden-region pixel metrics (MAE plus SSIM/LPIPS), silhouette IoU/Dice, and the visible-only baseline, with sample counts and uncertainty intervals.
4. Publish a contact sheet of every test input, prediction, and reference, and include failure cases as well as successes.
5. For the exact hidden-chair-behind-wall goal, evaluate on paired views or known 3D assets. Label single-view outputs as plausible hypotheses, not recovered ground truth.
