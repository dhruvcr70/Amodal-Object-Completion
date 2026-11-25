"""Place a Pix2Gestalt object completion back into its original full-scene image.

This is a scene-level visualization/compositing step, not a scene generator. It
keeps all source pixels except the predicted hidden-object region inside the
user-supplied occluder mask. The completed object is also exported as a full-frame
transparent layer so downstream tools can choose a different depth ordering.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from compose_rgb_results import estimate_foreground


def load_model_mask(path: Path, scene_size: tuple[int, int], crop: tuple[int, int, int, int],
                    model_size: tuple[int, int]) -> np.ndarray:
    """Load a mask defined either in scene coordinates or completion-crop coordinates."""
    mask_image = Image.open(path).convert("L")
    if mask_image.size == scene_size:
        mask_image = mask_image.crop(crop)
    return np.asarray(mask_image.resize(model_size, Image.Resampling.NEAREST)) >= 128


def load_crop_mask(path: Path, scene_size: tuple[int, int], crop: tuple[int, int, int, int],
                   crop_size: tuple[int, int]) -> np.ndarray:
    """Load a mask and align it with the original-resolution scene crop."""
    mask_image = Image.open(path).convert("L")
    if mask_image.size == scene_size:
        mask_image = mask_image.crop(crop)
    return np.asarray(mask_image.resize(crop_size, Image.Resampling.NEAREST)) >= 128


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path,
                        help="Pix2Gestalt run folder created with --scene and --crop.")
    parser.add_argument("--scene", type=Path,
                        help="Full, unmodified scene photo used to prepare the object crop.")
    parser.add_argument("--completion", type=Path,
                        help="Pix2Gestalt RGB completion for one object crop.")
    parser.add_argument("--visible-mask", type=Path,
                        help="Visible target-object mask; may be full-scene or crop-sized.")
    parser.add_argument("--occluder-mask", type=Path,
                        help="Mask of the foreground obstruction; may be full-scene or crop-sized.")
    parser.add_argument("--completion-mask", type=Path,
                        help="Optional predicted full-object mask aligned with completion RGB.")
    parser.add_argument("--crop", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"),
                        help="Original-scene crop used for inference, in pixel coordinates.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--white-threshold", type=int, default=12,
                        help="GrabCut border-background tolerance for white-backed RGB outputs.")
    args = parser.parse_args()

    if not 0 <= args.white_threshold <= 254:
        parser.error("--white-threshold must be in [0, 254]")
    if args.run_dir:
        if any((args.scene, args.completion, args.visible_mask, args.crop)):
            parser.error("Use --run-dir or the explicit --scene/--completion/--visible-mask/--crop inputs, not both.")
        run_dir = args.run_dir.resolve()
        metadata_path = run_dir / "scene_metadata.json"
        if not metadata_path.is_file():
            parser.error(f"{run_dir} has no scene_metadata.json; rerun Pix2Gestalt with --scene and --crop.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        args.scene = run_dir / "scene.png"
        args.completion = run_dir / "completion.png"
        args.visible_mask = run_dir / "visible_mask.png"
        args.crop = metadata["crop_xyxy"]
        if args.occluder_mask is None:
            saved_occluder = run_dir / "occluder_mask.png"
            if saved_occluder.is_file():
                args.occluder_mask = saved_occluder
        for required in (args.scene, args.completion, args.visible_mask):
            if not required.is_file():
                parser.error(f"Missing run artifact: {required}")
    elif not all((args.scene, args.completion, args.visible_mask, args.crop)):
        parser.error("Provide --run-dir or all of --scene, --completion, --visible-mask, and --crop.")
    if args.occluder_mask is None:
        parser.error("Provide --occluder-mask or save it into the Pix2Gestalt run with --occluder-mask.")

    args.scene = args.scene.resolve()
    args.completion = args.completion.resolve()
    args.visible_mask = args.visible_mask.resolve()
    args.occluder_mask = args.occluder_mask.resolve()
    if args.completion_mask:
        args.completion_mask = args.completion_mask.resolve()
    scene_image = Image.open(args.scene).convert("RGB")
    scene_size = scene_image.size
    x0, y0, x1, y1 = args.crop
    if not (0 <= x0 < x1 <= scene_size[0] and 0 <= y0 < y1 <= scene_size[1]):
        parser.error(f"--crop {args.crop} is outside scene dimensions {scene_size}")
    crop = (x0, y0, x1, y1)
    crop_size = (x1 - x0, y1 - y0)

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    completion_image = Image.open(args.completion).convert("RGBA")
    model_size = completion_image.size
    completion_rgba = np.asarray(completion_image)
    completion_rgb = completion_rgba[..., :3]
    visible_model = load_model_mask(args.visible_mask, scene_size, crop, model_size)
    if not visible_model.any():
        parser.error("The visible-object mask is empty inside the inference crop.")

    if args.completion_mask:
        predicted_model = load_model_mask(args.completion_mask, scene_size, crop, model_size)
        mask_method = "provided completion mask"
    elif np.any(completion_rgba[..., 3] < 250):
        predicted_model = completion_rgba[..., 3] >= 128
        mask_method = "completion PNG alpha channel"
    else:
        predicted_model = estimate_foreground(completion_rgb, visible_model, args.white_threshold)
        mask_method = "GrabCut seeded by visible-object mask and near-white border"

    # Preserve observed target support even if foreground extraction drops pale details.
    predicted_model |= visible_model
    scene_crop = scene_image.crop(crop)
    generated_crop = completion_image.convert("RGB").resize(crop_size, Image.Resampling.LANCZOS)
    generated_rgb = np.asarray(generated_crop)
    visible_crop = load_crop_mask(args.visible_mask, scene_size, crop, crop_size)
    predicted_crop = np.asarray(
        Image.fromarray(predicted_model.astype(np.uint8) * 255).resize(
            crop_size, Image.Resampling.NEAREST)
    ) >= 128
    occluder_crop = load_crop_mask(args.occluder_mask, scene_size, crop, crop_size)
    hidden_crop = predicted_crop & ~visible_crop
    reveal_crop = hidden_crop & occluder_crop

    # Full-frame transparent layer: source RGB for observed pixels, generated RGB
    # for hidden pixels. Alpha lets a downstream renderer place it at any depth.
    object_rgb = generated_rgb.copy()
    source_crop_rgb = np.asarray(scene_crop)
    object_rgb[visible_crop] = source_crop_rgb[visible_crop]
    object_alpha = predicted_crop.astype(np.uint8) * 255
    object_layer = np.dstack([object_rgb, object_alpha])
    full_layer = Image.new("RGBA", scene_size, (0, 0, 0, 0))
    full_layer.paste(Image.fromarray(object_layer, mode="RGBA"), (x0, y0))
    full_layer.save(output / "completed_object_layer.png")

    # Reveal only the predicted hidden portion that overlaps the marked obstruction.
    # This deliberately replaces those foreground pixels to make the completion
    # visible; it does not claim to preserve physically correct occlusion ordering.
    scene_result = np.asarray(scene_image).copy()
    scene_result[y0:y1, x0:x1][reveal_crop] = generated_rgb[reveal_crop]
    Image.fromarray(scene_result).save(output / "scene_completion.png")

    diagnostic = np.zeros((scene_size[1], scene_size[0], 3), dtype=np.uint8)
    diagnostic[y0:y1, x0:x1][occluder_crop] = (45, 100, 255)
    diagnostic[y0:y1, x0:x1][visible_crop] = (255, 175, 30)
    diagnostic[y0:y1, x0:x1][reveal_crop] = (45, 220, 100)
    Image.fromarray(diagnostic).save(output / "scene_masks_preview.png")

    report = {
        "scene": str(args.scene.resolve()),
        "completion": str(args.completion.resolve()),
        "crop_xyxy": list(crop),
        "scene_size": list(scene_size),
        "completion_size": list(model_size),
        "mask_method": mask_method,
        "predicted_hidden_pixels": int(hidden_crop.sum()),
        "revealed_pixels_inside_occluder": int(reveal_crop.sum()),
        "output_images": {
            "scene_completion.png": "full-scene image with predicted hidden target pixels revealed over the marked occluder",
            "completed_object_layer.png": "full-frame transparent object layer for downstream depth compositing",
            "scene_masks_preview.png": "blue=occluder, orange=observed object, green=predicted pixels revealed",
        },
        "limitations": [
            "This composites an object-level Pix2Gestalt result into the original scene; it is not a full-scene generative model.",
            "The reveal preview replaces marked occluder pixels where the predicted hidden object overlaps them; this is for visualization and does not preserve physical depth ordering.",
            "Foreground extraction can misclassify white object details; pass --completion-mask for a supplied amodal mask when available.",
            "The crop coordinates must match the exact scene crop used during model inference.",
            "Generated hidden appearance is a plausible prediction, not verified recovery of the unseen pixels.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), **report["output_images"],
                      "revealed_pixels": int(reveal_crop.sum())}, indent=2))


if __name__ == "__main__":
    main()
