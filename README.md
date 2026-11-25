# Amodal Object Completion

Your local VS Code workspace for learning to reconstruct partially hidden objects.

## Current scope (2026-11-01): Pix2Gestalt only

The active completion pipeline uses pretrained Pix2Gestalt. The custom U-Nets in
this repository are retained as historical experiments and are not part of the
current model or demo. The active workflow is **SAM ViT-B promptable target-mask
segmentation (optional when weights are present) → brush-refined visible and
obstruction masks → Pix2Gestalt stochastic candidates → full-scene 2D previews
and transparent object layers**. See [`docs/pix2gestalt-demo.md`](docs/pix2gestalt-demo.md).

The source-disjoint ABO pair builder and Pix2Gestalt evaluation already exist:
synthetic pairs retain the clean RGB target, full silhouette, visible mask, and
blocker mask, and held-out test identities are separate from train/validation.
Pix2Gestalt has **not** been fine-tuned on these pairs; they support evaluation.
The current results show useful RGB filling on synthetic white-background
products, but weak natural-room silhouette results. The demo does not infer
depth, reconstruct the blocker-removed background, or prove exact hidden pixels.

The Pix2Gestalt upstream source and its supporting repositories are cloned into
the ignored `third_party/` folder. The Pix2Gestalt checkpoint (about 15.5 GB) and
SAM ViT-B checkpoint are not present in `checkpoints/`, and this machine's active
Python lacks the project runtime packages, so inference is not runnable here yet.
The app never silently downloads the large model. Candidate sampling, explicit
human candidate selection, selected transparent-PNG export, CUDA autocast, timing,
GPU memory, and per-run output folders are implemented. Checkpoint replacements
invalidate the in-memory model cache, and each run records checkpoint file
identity. Optional W&B logging records scalar runtime metadata only, with no images,
and is off by default; cloud synchronization has not been tested.

## Work here

This Desktop folder is the main project. Write code, prepare data, inspect results,
and run smaller experiments locally. Use the included Colab notebook when an
experiment needs more GPU memory than this laptop provides.

**An editor does not supply additional GPU memory.** VS Code runs code using the
computer or remote runtime selected. This laptop has a T600 with 4 GB VRAM. A free
Colab T4 with 14.56 GiB usable VRAM was previously verified, but future allocation
and session length are not guaranteed.

## Local environment

VS Code tasks and debugging are configured to reuse the existing CUDA-enabled
Python 3.12 interpreter from your MOBA-RL project:

`C:/Users/dell/Desktop/moba-rl/moba-rl/.venv/Scripts/python.exe`

This avoids another multi-GB PyTorch download. The newly created `.venv` in this
folder remains available but is not the selected training environment and does
not contain PyTorch. No packages in either existing project were changed.

Because this interpreter is shared, do not upgrade or install project dependencies
into it without checking the effect on MOBA-RL. The local requirements file is a
reference for a future independent environment. The older diffusion baseline
retains its separate Colab environment.

## Main project: full-scene object completion

The goal is to complete a partly hidden object while returning a full-resolution
scene image, not only an isolated object crop. Pix2Gestalt remains the
object-completion model; `scripts/compose_scene_completion.py` maps its RGB crop
back to the original scene using a visible-target mask, an occluder mask, and the
inference crop coordinates. It writes a scene preview plus a transparent
full-frame object layer. The preview reveals predicted pixels by replacing the
marked occluder overlap; it is a visualization, not physically correct
depth-aware rendering or a guarantee of exact hidden details. The separate
two-image, magenta-marking LaMa workflow in
`notebooks/02_marked_obstruction_colab.ipynb` is an inpainting baseline, not the
Pix2Gestalt scene-integration path. See `docs/scene-reconstruction.md` for the
input format, command, outputs, and limitations.

### Historical RGB baseline reviews (not the active model)

A crop-alignment bug in `scripts/prepare_cocoa.py` was fixed: masks are now cropped with the same image window. The corrected 12-example Pix2Gestalt RGB evaluation has been run on a free Colab T4 and downloaded to `runs/eval-v1/`. Inference completed 12/12 times (5.89 seconds mean, 5.04 GiB peak allocated memory), but visual quality is mixed to poor: multiple category mismatches, incomplete fragments, a nearly blank output, and boundary artifacts. The model outputs are isolated cutouts on white. `scripts/compose_scene_completion.py` now maps a crop completion into a full-scene preview using the original scene, crop coordinates, visible-object mask, and obstruction mask. The preview reveals the predicted object by replacing only its marked obstruction overlap; it is not depth-aware scene generation. See `docs/scene-reconstruction.md` for the workflow and limits. The 12 examples come from one held-out scene and do not establish broad generalization.

The earlier silhouette U-Net was trained locally and combined with Pix2Gestalt RGB in an auxiliary experiment. It is archived and excluded from this project's active pipeline; see `runs/eval-v1/object-completions-v5/` and `docs/evaluation-protocol.md` for that historical work.

The separate scripts/demo_silhouette.py tool and the compact U-Net predict masks only. They are an auxiliary silhouette experiment and do not complete an object's colors or appearance. They are not the project's RGB completion deliverable.

The historical Pix2Gestalt mask-conditioning probe is complete under
`runs/guidance-ablation-v1/`; it compared visible masks with an auxiliary U-Net's
predicted silhouettes on 12 examples and found no validated RGB-quality improvement.
The four-candidate qualitative review is also complete; see
`docs/candidate-selection-experiment.md`.

### Full-scene integration and held-out RGB evaluation (2026-11-01)

The full-scene path now supports more than the dining-table example: run
Pix2Gestalt on a target-object crop, then project that completion into any
full-resolution scene using the crop coordinates, visible-target mask, and
occluder mask. `scripts/compose_scene_completion.py` handles one target and
`scripts/compose_scene_batch.py` layers multiple independently completed targets.
The batch path now writes a reveal preview, a depth-ordered composite, and an
editable 2D layer stack. Ordering values are supplied in the manifest; the system
does not infer depth. The depth render first estimates the background under
removed obstructions with OpenCV Telea inpainting by default (LaMa is optional),
then places predicted objects and original blockers in the declared order. These
are estimates, not recovered scene pixels. See
`docs/scene-reconstruction.md` for inputs and limitations.

Pix2Gestalt was also evaluated on 46 held-out synthetic ABO occlusion examples
with full-image targets. The recomputed run reported full silhouette IoU 0.9856,
hidden silhouette IoU 0.9546, hidden-region RGB MAE 0.0670, PSNR 20.06 dB, and
SSIM 0.6513; the unchanged-input hidden-region MAE baseline was 0.2877. These results
show performance on clean product cutouts with synthetic blockers, not natural
room photographs, and pixel scores are sensitive to alignment. This was
evaluation of pretrained weights, not model fine-tuning. The recomputed per-sample
metrics, summary, and contact sheet are present under
`runs/pix2gestalt-abo-rgb-eval-v1/recomputed-v2/`.

The existing 12-crop MP3D-Amodal run is the natural-room check. All 12 inference
calls completed, but the manual review found mixed-to-poor object completions,
including category mismatches, fragments, near-blank outputs, and artifacts. This
is a small subjective review, not a numerical RGB benchmark. See
`runs/eval-v1/qualitative_review_summary.json` and `contact_sheet.png`.
The saved predictions have now also been projected into their full-resolution
MP3D source photographs. Mean full-silhouette IoU is 0.5509 versus a 0.5637
visible-only baseline; mean hidden-region IoU is 0.2694. The room projections
and contact sheet are in `runs/mp3d-natural-scene-v2/`. Results remain weak, and
there is no ground-truth RGB for the hidden pixels.

To isolate full-frame scene integration, the saved 46 Pix2Gestalt ABO test
predictions were placed into controlled scenes using an MP3D room photo as the
background. The clean scene was preserved as ground truth and a known synthetic
blocker was applied. On the 46 examples, hidden-object RGB MAE was 0.0668 versus
0.2877 for leaving the obstruction in place; full blocker-region MAE was 0.0737
versus 0.2738. Mean hidden-object IoU was 0.9546. The comparison sheet and
per-scene metrics are in `runs/controlled-scene-eval-v2/`. This reuses predictions
made on white-background product crops: objects were pasted into one real room
photo, and Pix2Gestalt did not receive room context. It evaluates the scene
compositor against known synthetic ground truth, not natural-room generalization.

A two-chair dining-scene run exercised Pix2Gestalt inference on a free Kaggle T4.
Because the notebook saved no output files, this previous result is also absent
locally and must be regenerated. The existing natural-room MP3D-Amodal review is
mixed to poor; neither it nor the synthetic product scores demonstrates accurate
recovery of full room scenes.

### Archived custom RGB completion experiment (excluded from active pipeline)

The first from-scratch RGB model is trained on a small Amazon Berkeley Objects
(ABO) subset. `scripts/prepare_abo_rgb_pairs.py` creates exact full-RGB targets
and synthetic opaque occlusions from white-background household product images;
product IDs are split before generating variants. The dataset has 454 pairs at
256×256 (364 train, 44 validation, 46 test) from 227 products. A 488,179-parameter
RGB U-Net trained on the NVIDIA T600 for 10 epochs, selecting epoch 9 on
validation. On 46 variants of 23 held-out products, RGB MAE under the blocker
fell from 0.3397 for the unchanged input to 0.1347 for the model; visible pixels
are copied through exactly. The preview still shows blurred textures, and these
white-background product images with synthetic blockers are not natural-room
occlusion results. Dataset preparation, training, metrics, and preview are
documented in `docs/rgb-training-data-status.md` and
`runs/abo-rgb-unet-v1/`.
The local painting interface and runner instructions are in
`docs/how-to-run-rgb-completion.md`; the custom U-Net still requires visible-object
and blocker masks, and has not been validated on natural room photos. The two-image
magenta-marking LaMa workflow is a separate Colab test path.

## Project layout

- `amodal/data.py`: deterministic shape generation, synthetic occlusion, split logic, and dataset loading.
- `amodal/model.py`: a compact 488,145-parameter U-Net and BCE + Dice objective.
- `scripts/generate_shapes.py`: creates the procedural engineering dataset.
- `scripts/prepare_objects.py`: applies the same occlusion pipeline to real image/mask pairs.
- `scripts/prepare_cocoa.py`: prepares a bounded COCOA amodal-annotation subset.
- `scripts/prepare_abo_rgb_pairs.py`: prepares source-disjoint RGB completion pairs from ABO catalog images.
- `scripts/train_rgb_completion.py`: trains and evaluates the RGB U-Net baseline.
- `scripts/predict_rgb_completion.py`: applies the trained RGB checkpoint using user-provided masks.
- `scripts/demo_rgb_completion.py`: opens a local brush-based interface for RGB completion experiments.
- `scripts/mark_obstruction.py`: paints obstruction regions on an image copy in magenta for the two-image LaMa workflow.
- `scripts/package_rgb_eval.py`: packages a deterministic, held-out RGB inference set for Colab.
- `scripts/prepare_mp3d.py`: securely reads the official MP3D-Amodal annotations and prepares scene-disjoint crops.
- `scripts/compose_completed_rgb_objects.py`: combines generated RGB with an estimated foreground mask into transparent object-cutout prototypes.
- `scripts/compose_scene_completion.py`: projects an object completion back into a full-scene preview using visible-object and occluder masks.
- `scripts/compose_scene_batch.py`: composites object predictions, blocker removal, and editable scene layers.
- `scripts/run_controlled_scene_evaluation.py`: rebuilds the controlled 46-scene integration evaluation from saved predictions.
- `scripts/analyze_pix2gestalt_abo_rgb.py`: reports visible consistency and Pix2Gestalt metrics by synthetic occlusion severity.
- `scripts/reconstruct_mp3d_natural_scenes.py`: projects the 12 saved MP3D Pix2Gestalt predictions into their full-resolution source photographs.
- `scripts/evaluate_rgb_completion_masks.py`: compares extracted completion silhouettes against held-out reference masks.
- `scripts/filter_dataset_categories.py`: creates a separate category-focused dataset while preserving source splits.
- `configs/cocoa-local.json`: dedicated training configuration for the real-data subset.
- `configs/cocoa-household-finetune-local.json`: category-focused COCOA fine-tuning setup.
- `configs/mp3d-local.json`: training configuration for MP3D-Amodal.
- `configs/mp3d-corrected-composite-local.json`: silhouette-model run on corrected MP3D crops for RGB cutout integration.
- `configs/mp3d-household-finetune-local.json`: low-rate fine-tuning on the
  category-filtered household subset.
- `configs/mp3d-augmentation-local.json`: controlled comparison with additional
  train-only synthetic occlusions and horizontal flips.
- `scripts/train_silhouette.py`: GPU training, validation, best/last checkpoints, resume, and held-out evaluation.
- `scripts/predict_silhouette.py`: generates complete silhouettes from the trained model.
- `scripts/demo_silhouette.py`: launches the local interactive silhouette demo.
- `scripts/evaluate_full_test.py`: evaluates a checkpoint, optionally filtered to exact categories, and writes per-sample metrics.
- `scripts/make_prediction_review.py`: creates side-by-side visual reviews, optionally limited to an occlusion range.
- `tests/test_core.py`: deterministic occlusion, source separation, metric, and gradient tests.
- `scripts/check_hardware.py`: records the local Python/GPU/PyTorch environment.
- `scripts/run_baseline.py`: experimental pretrained generator runner, intended for
  the isolated Linux environment set up by the Colab notebook. Optional `--scene`,
  `--crop`, and `--occluder-mask` flags save the context needed by the compositor.
- `scripts/demo_pix2gestalt.py`: SAM-assisted, brush-refinable Gradio interface for
  Pix2Gestalt candidates, 2D scene previews, and transparent layer exports.
- `docs/pix2gestalt-demo.md`: model setup, workflow, outputs, and limitations.
- `requirements-demo.txt`: optional Gradio and scalar-only W&B packages.
- `scripts/setup_colab.py`: Colab-only baseline environment setup.
- `notebooks/01_baseline_colab.ipynb`: Pix2Gestalt research baseline.
- `notebooks/02_marked_obstruction_colab.ipynb`: two-image, magenta-marked LaMa inpainting workflow.
- `configs/`: experiment configuration files.
- `data/`: local data, excluded from Git.
- `checkpoints/`: model weights, excluded from Git.
- `runs/`: generated outputs and measurements, excluded from Git.
- `docs/baseline-notes.md`: baseline scope, status, attribution, and model limitations.

Do not run `setup_colab.py` locally: it intentionally targets `/content` on Linux.
The notebook is self-contained and can be uploaded to Colab without uploading this
whole folder. Download its result archives back into `runs/`.

## Milestones

1. Use the SAM-assisted, brush-refinable Pix2Gestalt workflow with the upstream model and weights in a GPU runtime.
2. Evaluate on broader source-disjoint synthetic categories and controlled real captures with unobstructed references.
3. Improve foreground-mask quality and evaluate mask/scene composition failures; report uncertainty and visible-only baselines.
4. Compare 2D scene previews against paired-view or known-3D ground truth before making scene-reconstruction claims.

The synthetic ABO paired-data pipeline, held-out evaluation, four-candidate review,
guidance probe, and SAM/Gradio interface code are present. On 2026-11-01, the
Gradio interface and its SAM-to-Pix2Gestalt callbacks were smoke-tested in a Kaggle
Tesla T4 runtime. The callbacks generated candidate images, full-scene 2D previews,
transparent object layers, and ZIP exports. The browser's brush/click controls were
not manually exercised; SAM mask overlap was modest, and W&B was tested offline
only. The large Pix2Gestalt checkpoint and run archives remain in Kaggle, so the
local app still cannot run the full model until the checkpoint is transferred.
The updated local Gradio interface builds with explicit candidate selection and
selected transparent-PNG export; the selection callback was exercised, but the
browser's brush/click flow still needs a manual session. Remaining research work is
stronger mask quality and better natural-scene ground-truth evaluation. W&B cloud
synchronization is optional and has not been tested. The scene preview is 2D
compositing, not depth-aware scene reconstruction.

### Archived auxiliary U-Net experiments (not part of current pipeline)

## Archived local silhouette experiment (not in active pipeline)

The first engineering dataset contains 384 procedural source shapes and two
occlusion variants each: 600 training, 86 validation, and 82 held-out test samples.
Source objects remain in a single split. These are shapes, not photographs.

Use **Terminal > Run Task** in VS Code:

1. **Generate shapes dataset** once (the initial dataset has already been generated).
2. **Train silhouette model** for a new run.
3. **Resume silhouette training** for an interrupted or extended run. Increase
   `epochs` in `configs/shapes-local.json` to extend training.
4. **Run core tests** when modifying the model or data logic.

Existing datasets and run folders are not silently overwritten. Set a new output
or run path for a fresh experiment. The default training configuration uses full
precision after a first FP16 trial produced non-finite loss on this environment.

The run saves `best.pt`, `last.pt`, `history.json`, `metrics.json`, `preview.png`, and
`training.png`. The visible-only baseline simply keeps the observed mask; the
learned model predicts the missing silhouette and preserves all visible pixels by
taking their union at inference. Hidden-region IoU is reported separately.

For real-data preparation, see `docs/data-format.md` and `docs/cocoa-data.md`.
The project now includes MP3D-Amodal training photos and 3D-derived annotations;
see `docs/mp3d-data.md`. COCOA remains an optional dataset whose annotation access
may require permission. The diffusion appearance completion baseline and
silhouette-conditioning research remain separate unfinished milestones. A
silhouette output does not reconstruct RGB textures or export a completed colored
object.

The first expanded real-data run is saved under `runs/mp3d-v2/`: the 8-epoch U-Net
slightly beat a visible-mask-only baseline on the held-out scene (test IoU 0.635
vs. 0.612). Hidden-region IoU is 0.248, so there is still clear room to improve.
The next experiment uses the same model, split, seed, and optimizer settings while
adding reproducible train-only occlusion and flip augmentation. Its separate
configuration and output folder make the comparison auditable; validation selects
the checkpoint. The same held-out scene was evaluated for this preplanned ablation,
so treat these numbers as exploratory rather than a final benchmark.
That ablation is complete in `runs/mp3d-v3-synthetic-occlusion/`: overall test IoU
was effectively unchanged (0.634 vs. 0.635), while hidden-region IoU increased
from 0.248 to 0.302. See `docs/mp3d-data.md` for the full comparison and limits.
The next validation-only experiment weights hidden positive pixels more strongly;
it uses `--skip-test` so the held-out scene does not steer this tuning choice.
The follow-up validation run combines that loss with synthetic occlusion
augmentation to check whether the two changes complement each other.
The original U-Net still has the best validation full-mask IoU (0.667); the
alternatives trade full-mask accuracy for improved hidden-region IoU. Detailed
validation-only results are recorded in `docs/mp3d-data.md`.

The household-object slice is saved under `data/mp3d-household-v1/`. It preserves
the original scene-disjoint assignments but contains only 44 training crops, so
it is currently used for focused evaluation rather than a new from-scratch model.
Its comparison against the broad-data checkpoints is documented in
`docs/household-object-evaluation.md`. The severe-occlusion MP3D breakdown and
review sheet are documented in `docs/occlusion-stress-test.md`.
The expanded COCOA household-object data and fine-tuning results are documented
in `docs/cocoa-household-evaluation.md`. The 5–90% filtered split contains
331/47/24 train/validation/test crops. Fine-tuning raised held-out full-mask IoU
from 0.570 to 0.764 and hidden-region IoU from 0.207 to 0.569 on 24 test crops;
the 60–90% subgroup has only nine examples, so treat the result as promising but
early. Without retraining, the same checkpoint scored 0.687 full-mask IoU and
0.396 hidden-region IoU on 23 household crops from the separate MP3D-Amodal test
scene. Cross-dataset limits are detailed in `docs/cocoa-household-evaluation.md`.

## Research baseline

pix2gestalt: https://github.com/cvlab-columbia/pix2gestalt

Retain attribution and each dependency/dataset's license. The released model weights
are CC BY-NC 4.0. Generated hidden content is plausible, not an exact recovery.




