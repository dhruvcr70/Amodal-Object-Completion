# First local training run

This is a successful engineering experiment on generated shapes, not a real-object
completion benchmark. The model predicts masks; diffusion appearance generation is
still unfinished.

## Implemented

- A 4-channel RGB + visible-mask dataset loader.
- Reproducible rectangle/ellipse occlusion with 15–60% foreground coverage.
- Source-disjoint train, validation, and test splits.
- A compact U-Net with GroupNorm and 488,145 trainable parameters.
- BCE + Dice training, gradient clipping, AdamW, and configurable precision.
- Best/last checkpoints, resume support, inference, visual previews, and learning curves.
- Mean full-mask IoU, hidden-region IoU, and a visible-only baseline.

## Verified run

- GPU: NVIDIA T600 Laptop GPU, 4 GB.
- Interpreter: existing MOBA-RL Python 3.12 environment; no new packages installed.
- Data: 384 procedural source shapes, two occlusion variants each.
- Samples: 600 training, 86 validation, 82 test.
- Settings: 128-pixel inputs, batch size 8, 8 epochs, full precision.
- Test full-mask mean IoU: **0.9410**.
- Test visible-only baseline mean IoU: **0.6377**.
- Test hidden-region mean IoU: **0.8610**.
- Peak allocated tensor memory: **236.6 MiB**, not total system GPU usage.
- Model reload and standalone prediction: passed.
- Core unit tests: four passed.

Outputs are in `runs/shapes-v1-fp32/`.

The first mixed-precision training attempt produced non-finite loss. Full precision
was used for the successful run. The optional FP16 setting is therefore experimental
for this U-Net, despite the simpler hardware probe passing FP16.

## Next experiment

Prepare a small permitted real-object image/mask dataset, retaining image-level
grouping. Use `scripts/prepare_objects.py` to construct paired occlusions, then train
a new run without presenting these procedural-shape results as real-photo performance.
