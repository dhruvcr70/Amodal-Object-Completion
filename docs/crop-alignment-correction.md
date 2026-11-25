# RGB and mask crop alignment correction

Date: 2026-10-30

## Finding

The shared `crop_square` helper in `scripts/prepare_cocoa.py` cropped the RGB photo to the object-centered window, but its mask path pasted the entire source-frame mask into the smaller output canvas. That shifted or clipped mask pixels instead of applying the same crop window. `scripts/prepare_mp3d.py` imports this helper, so both COCOA and MP3D-Amodal masks were affected.

A spot-check showed the failure clearly: an MP3D chair photo was a room crop while its mask occupied an unrelated corner. A COCOA plate crop had only eight foreground pixels at the edge. The COCOA silhouette metrics and the previously generated Pix2Gestalt images therefore cannot be treated as valid object-completion results. Their successful inference calls only show that the program ran.

## Fix and corrected data

`crop_square` now crops the mask using the same source-image bounds as the RGB crop before resizing both. The data-preparation scripts then rebuilt these new, separate datasets:

- `data/mp3d-subset-v4-corrected-crops/`: 900 object samples from the official MP3D-Amodal archive, split by scene into 481 train, 119 validation, and 300 test samples.
- `data/mp3d-household-v2-corrected-crops/`: the household subset, with 113 train, 37 validation, and 45 test samples across the same scene-disjoint splits.
- `data/cocoa-household-v3/`: 796 COCOA crops, including previously excluded low-occlusion examples. No region in this bounded sample has an exactly complete RGB target; even its least-occluded example has 0.47% hidden area. It is not being presented as exact full-color ground truth.

A 12-image, test-split inference input archive is ready at `runs/mp3d-rgb-corrected-eval-v1.zip`. It spans 11 household categories, including chair and couch, and has 22.6%–80.1% annotated occlusion. The RGB target of a physically occluded source photo is unknown, so this run is for careful qualitative review; the full-object mask is available as a silhouette reference, not a color target.

## What must be rerun

1. Run Pix2Gestalt on the corrected archive and visually inspect each completion beside the corrected input and mask.
2. Rebuild any COCOA/MP3D silhouette training and evaluation results from the corrected datasets before quoting earlier metrics.
3. Keep the mask-only U-Net results separate from the RGB completion deliverable.

This correction changes no model weights. It fixes the data alignment used to prepare examples.
