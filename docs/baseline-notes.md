# Amodal Object Completion

Reconstruct plausible whole objects from partially occluded images.

## First milestone

Run **one pretrained completion** on a free Colab T4. Save the input, visible mask,
output, settings, timing, and peak GPU memory. No training or paid services are
required for this milestone. GPU availability is not guaranteed.

Open `notebooks/01_baseline_colab.ipynb` in Google Colab using **File > Upload
notebook**, select **Runtime > Change runtime type > T4 GPU**, then run the cells
in order. The notebook writes its own helper scripts, so you do not need to upload
the entire folder. Setup downloads several GB; the official model checkpoint alone
is approximately 15.5 GB. Downloads use Colab's connection and temporary disk.

Supply two PNGs with identical dimensions:

- `image.png`: an image containing one partially hidden target object.
- `visible_mask.png`: white on visible pixels of that target; black everywhere else.
  Do not label the obstructing object or guess the hidden shape in this mask.

Pick a target that is fully inside the image, roughly centered, with a meaningful
portion visible. The baseline resizes the pair to its native 256 x 256 resolution.
Prefer a square crop prepared before upload; do not crop the image and mask differently.

## What exists now

- A baseline experiment notebook and isolated environment setup.
- Input validation and a Colab inference runner that can save up to four
  stochastic candidates per input for side-by-side selection (`--candidates`).
- An experimental lower-memory checkpoint loader that retains EMA inference weights
  without retaining a second full model on the GPU.
- A run manifest recording source revisions, settings, timing, and memory.

**Status:** the first full inference completed in Google Colab on a free Tesla T4
(14.56 GiB VRAM). It used the authors' Pix2Gestalt checkpoint, 50 DDIM steps,
seed 42, one 256 x 256 candidate, and guidance scale 2.0. The measured inference
time was 6.24 seconds and peak allocated GPU memory was 5.04 GiB. The run manifest
also records the upstream source revisions and compatibility edits.

The first run was a **synthetic smoke test**, using a simple colored rectangle partly
covered by another rectangle. A real held-out MP3D sample was subsequently run with
the same settings. It completed in 5.99 seconds on the Tesla T4, with 5.04 GiB peak
allocated GPU memory. Its files are saved at
`runs/real-sample-20261027/` (`input.png`, `visible_mask.png`, `completion.png`, and
`metrics.json`). The ZIP downloaded from Colab is `amodal_baseline_result.zip`.

The corrected 12-example RGB evaluation archive is now local under `runs/eval-v1/`. All 12 inference calls completed on a Tesla T4 (5.89 seconds mean inference time; 5.04 GiB mean peak allocated memory), but visual review found mixed-to-poor completion quality, including category mismatches, fragments, blank output, and artifacts. Ratings and per-sample notes are in `runs/eval-v1/results.csv`; the sheet is `runs/eval-v1/contact_sheet.png`. The outputs are isolated object cutouts, not composites preserving the original scene. The 12 examples come from one held-out scene and do not establish broad generalization.

The generated hidden content is a plausible inference, not recovery of exact unseen pixels. The main project output is an RGB object completion. The custom U-Net and its interactive window predict a silhouette mask only; they are a separate experiment and are not the RGB deliverable.
## Main project next steps

The corrected RGB outputs are now local and visually reviewed. A corrected-data U-Net was trained locally on the T600 to supply an amodal silhouette; the RGB outputs were also segmented with white-background flood fill and GrabCut. Validation-selected U-Net masks transferred poorly to the fixed 12-example held-out subset, while RGB mask extraction recovered some hidden silhouette but slightly reduced full-mask IoU against the visible-only baseline. GrabCut plus exact copying of visible RGB pixels was used to create isolated transparent object cutouts under `runs/eval-v1/object-completions-v5/`. The cutouts still inherit Pix2Gestalt's category mismatches and appearance artifacts; scene-layer compositing and reliable object-specific RGB completion remain unsolved. Detailed method and metric limits are in `docs/evaluation-protocol.md`.

Completed hidden content is a plausible inference, not recovery of exact unseen pixels.

## Attribution

The initial generator is **pix2gestalt: Amodal Segmentation by Synthesizing Wholes**
(Ozguroglu et al., CVPR 2024):
https://github.com/cvlab-columbia/pix2gestalt

Original weights: https://huggingface.co/cvlab/pix2gestalt-weights

The original weights are CC BY-NC 4.0; source code, model dependencies, and datasets
retain their respective licenses. Our planned custom silhouette model and experiments
must be described separately from pretrained capabilities.

