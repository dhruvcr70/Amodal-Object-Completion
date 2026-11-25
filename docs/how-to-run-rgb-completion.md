# Run an object-completion experiment

## Quick two-image test in Colab

First create the marked copy in VS Code with `scripts/mark_obstruction.py`. Run it, click **Load original photo**, paint only the obstruction, then click **Save magenta-marked copy**. Do not resize or crop the copy after marking.

Open `notebooks/02_marked_obstruction_colab.ipynb` in Colab and upload exactly two same-size files: the untouched original and the new `_marked.png` copy. The notebook previews the extracted mask before running the model.

The notebook extracts the magenta region, previews the mask, runs LaMa inpainting, and composites the generated pixels only inside the mask. The output is a plausible fill, not an exact reconstruction of the hidden object's true appearance.

## Existing models

The Pix2Gestalt research baseline expects a mask of the visible target object (white means visible target). It does not accept an obstruction-only mask. Supply `--image`, `--mask`, and `--checkpoint` to `scripts/run_baseline.py`.

The custom RGB U-Net expects both a visible-object mask and an occluder mask; see the local painter and `scripts/predict_rgb_completion.py`. Its training data is mostly white-background catalog objects with synthetic blockers, so it is not yet a reliable natural-photo solution.
