# RGB completion training data status

## Current finding (2026-10-31)

The synthetic RGB-pair builder requires a clean full-color target crop and a
non-empty full-object mask. `data/cocoa-rgb-pairs-v1/` contains zero pairs because
the source it used, `data/cocoa-household-v1/`, has 767 records marked with an
occlusion fraction of zero, but only one of those records has a non-empty
`complete_mask.png`. The other masks are all black, so those records cannot be
used as exact RGB reconstruction targets.

A fresh, bounded preparation from the available raw COCOA training annotations
was also attempted (`data/cocoa-rgb-source-v2/`, 80-image/400-object caps). It
produced 105 household-object crops across 42 photos, but none are fully visible.
The raw annotation file has no household regions with an exact zero occlusion
rate. Consequently, these crops are useful for amodal-mask work but cannot
provide exact hidden RGB pixels for supervised completion training.

`scripts/prepare_rgb_pairs.py` now checks for usable clean targets before
creating its output directory and reports empty masks or missing/mismatched
files instead of silently producing an empty dataset.

## ABO RGB experiment (2026-10-31)

The project now uses a small subset of Amazon Berkeley Objects (ABO) household
product photos instead of the unusable COCOA clean-target records. The official
[ABO dataset page](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)
lists CC BY 4.0, more than 7,900 household-product 3D models, and 2.1 million
rendered views with masks and geometry. The full rendered material-benchmark
archive is 271 GB, so this experiment uses small catalog images from one
metadata shard and does not download the full archive.

`scripts/prepare_abo_rgb_pairs.py` selected 227 products after rejecting
lifestyle images and implausible foreground masks. It created 454 256×256
synthetic-occlusion pairs, split by product into 364 train, 44 validation, and
46 test variants (182/22/23 unique products). The masks are estimated by
near-white border flood fill and should be treated as heuristic labels.

`scripts/train_rgb_completion.py` trained a 488,179-parameter RGB U-Net for 10
epochs on the local NVIDIA T600; validation selected epoch 9. On 46 variants of
23 held-out products, occluder-region RGB MAE fell from 0.3397 for the unchanged
input to 0.1347 after completion. Hidden-object-region MAE was 0.1536. The model
copies all unblocked pixels exactly and generates only under the synthetic
blocker.

The qualitative preview still shows blurred textures, and the experiment uses
white-background product photos plus synthetic blockers. It is an end-to-end
supervised RGB training baseline, not evidence of reliable completion in
natural room scenes. The failed COCOA attempt and its metadata remain preserved
for reproducibility; they were not used for training.
