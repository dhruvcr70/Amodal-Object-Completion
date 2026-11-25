"""Run the trained RGB completion U-Net on one image and two binary masks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal.rgb_model import RGBCompletionUNet


def load_mask(path: Path, expected_size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as im:
        if im.size != expected_size:
            raise ValueError(f"Mask size {im.size} does not match image size {expected_size}: {path}")
        mask = np.asarray(im.convert("L"), dtype=np.uint8) >= 128
    if not mask.any():
        raise ValueError(f"Mask is empty: {path}")
    return mask


def square_canvas(image: Image.Image, visible: np.ndarray, side: int = 256):
    width, height = image.size
    scale = min(side / width, side / height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
    resized_visible = Image.fromarray(visible.astype(np.uint8) * 255).resize(
        new_size, Image.Resampling.NEAREST)
    left = (side - new_size[0]) // 2
    top = (side - new_size[1]) // 2
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(resized_image, (left, top))
    visible_canvas = Image.new("L", (side, side), 0)
    visible_canvas.paste(resized_visible, (left, top))
    return canvas, np.asarray(visible_canvas) >= 128, (left, top, new_size[0], new_size[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Square-ish RGB image crop of one target object")
    parser.add_argument("--visible-mask", type=Path, required=True,
                        help="Binary PNG: white on the visible target-object pixels")
    parser.add_argument("--occluder-mask", type=Path, required=True,
                        help="Binary PNG: white on the blocker pixels to replace")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "runs/abo-rgb-unet-v1/best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/rgb-prediction")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output directory exists; choose another path: {args.output}")
    for path in (args.image, args.visible_mask, args.occluder_mask, args.checkpoint):
        if not path.is_file():
            parser.error(f"File not found: {path}")

    with Image.open(args.image) as handle:
        original = handle.convert("RGB")
    original_size = original.size
    visible = load_mask(args.visible_mask, original_size)
    occluder = load_mask(args.occluder_mask, original_size)
    if np.logical_and(visible, occluder).any():
        parser.error("Visible-object and occluder masks overlap; mark only visible object pixels in the visible mask.")

    image_square, visible_square, content_box = square_canvas(original, visible)
    rgb = np.asarray(image_square, dtype=np.float32) / 127.5 - 1.0
    image_tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0)
    visible_tensor = torch.from_numpy(visible_square.astype(np.float32)[None, None])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = RGBCompletionUNet(base=int(checkpoint["base_channels"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.inference_mode():
        prediction = model(torch.cat((image_tensor, visible_tensor), dim=1).to(device))[0].cpu()
    prediction_rgb = ((prediction.clamp(-1, 1).permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)
    left, top, content_width, content_height = content_box
    prediction_image = Image.fromarray(prediction_rgb).crop(
        (left, top, left + content_width, top + content_height)
    ).resize(original_size, Image.Resampling.LANCZOS)

    completed = original.copy()
    completed.paste(prediction_image, (0, 0), Image.fromarray(occluder.astype(np.uint8) * 255))
    args.output.mkdir(parents=True)
    prediction_image.save(args.output / "model_prediction.png")
    completed.save(args.output / "completed_composite.png")
    Image.fromarray(visible.astype(np.uint8) * 255).save(args.output / "visible_mask.png")
    Image.fromarray(occluder.astype(np.uint8) * 255).save(args.output / "occluder_mask.png")
    metadata = {
        "checkpoint": str(args.checkpoint), "device": str(device),
        "input_size": list(original_size), "model_resolution": [256, 256],
        "composite_rule": "Replace pixels only under occluder_mask.png; preserve all other input pixels exactly.",
        "note": "Trained on ABO white-background product images with synthetic occluders; natural-scene quality is not established.",
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print("Saved composite:", args.output / "completed_composite.png")


if __name__ == "__main__":
    main()
