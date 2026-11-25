# Pix2Gestalt mask-conditioning ablation

## Question

Does changing the Pix2Gestalt mask input change the sampled completion? This is a historical conditioning probe; the custom U-Net is excluded from the current project pipeline.

## Experimental setup

- 12 fixed MP3D-Amodal test examples from 12 object categories.
- Paired conditions per example: ground-truth visible mask and a historical U-Net predicted amodal mask (24 generated outputs total).
- Pretrained Pix2Gestalt checkpoint; no Pix2Gestalt fine-tuning.
- 256 x 256 resolution, 50 DDIM steps, guidance scale 2.0, seed 42 for each paired condition.
- Inference ran on a Kaggle Tesla T4. The saved summary records peak allocated GPU memory of 5.043 GiB.
- Full image outputs, conditioning masks, contact sheet, and run summary are under `runs/guidance-ablation-v1/`.

## Interpretation

This is a zero-shot, out-of-distribution conditioning experiment. Pix2Gestalt was trained with visible masks; it was not trained to accept masks predicted by our amodal-silhouette U-Net. The paired images show that changing the mask can alter the result, but visual inspection does not establish a consistent improvement in completion quality. We have no validated quantitative metric for RGB completion quality, so no PSNR, SSIM, or pixel-level improvement claim is made.

The experiment is useful as an integration probe: it confirms that the pipeline can run both mask conditions end to end and saves reproducible paired outputs. It does not demonstrate that predicted amodal masks improve a pretrained RGB generator.

## Status

The 12 pairs have been reviewed; see `guidance-ablation-visual-review.md`. The mask change could alter outputs, but no consistent improvement was validated. This artifact is retained as project history and is not a dependency of the current Pix2Gestalt workflow.

## Related silhouette evaluation
Historical silhouette metrics are reported separately in `evaluation-protocol.md` and `runs/eval-v1/`. Those scores evaluate binary mask prediction, not Pix2Gestalt's RGB completions.


