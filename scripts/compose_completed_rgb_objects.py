"""Merge generated RGB appearance with an independently predicted amodal silhouette.

The custom U-Net supplies only the object's full silhouette. Pix2Gestalt supplies
RGB appearance. Known visible RGB pixels are copied exactly from the input; hidden
pixels use the Pix2Gestalt output. Ground-truth masks are never read by this script.
The result is an isolated transparent object, not a scene-layer composite.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal.model import CompletionUNet  # noqa: E402


def checkerboard(size: tuple[int, int], cell: int = 16) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size, "#eeeeee")
    draw = ImageDraw.Draw(image)
    for y in range(0, height, cell):
        for x in range(0, width, cell):
            if (x // cell + y // cell) % 2:
                draw.rectangle((x, y, min(x + cell - 1, width), min(y + cell - 1, height)),
                               fill="#c8c8c8")
    return image


def overlay_mask(rgb: np.ndarray, visible: np.ndarray, amodal: np.ndarray) -> Image.Image:
    result = rgb.astype(np.float32).copy()
    hidden = amodal & ~visible
    result[hidden] = result[hidden] * 0.45 + np.array([40, 220, 110], np.float32) * 0.55
    result[visible] = result[visible] * 0.45 + np.array([255, 178, 35], np.float32) * 0.55
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("runs/mp3d-corrected-silhouette-v1/best.pt"))
    parser.add_argument("--results", type=Path, default=Path("runs/eval-v1"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/eval-v1/object-completions-v5"))
    parser.add_argument("--mask-source", choices=("rgb-white-floodfill", "grabcut", "custom-unet"),
                        default="grabcut")
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="Validation-selected threshold for hidden-region IoU.")
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error("--threshold must be strictly between 0 and 1")
    results, output = map(Path.resolve, (args.results, args.output))
    checkpoint = args.checkpoint.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    model = None
    size = None
    device = torch.device("cpu")
    if args.mask_source == "custom-unet":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = CompletionUNet(state["config"]["base_channels"]).to(device)
        model.load_state_dict(state["model"])
        model.eval()
        size = int(state["resolution"])
        del state

    samples = sorted(path for path in results.iterdir()
                     if path.is_dir() and (path / "completion.png").is_file())
    if not samples:
        parser.error(f"No RGB completions found under {results}")
    tile, label_h, gap = 256, 22, 8
    sheet = Image.new("RGB", (4 * tile + 5 * gap,
                               len(samples) * (tile + label_h + gap) + gap), "#20242b")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    mask_label = {"rgb-white-floodfill": "RGB foreground mask",
                  "grabcut": "GrabCut foreground mask",
                  "custom-unet": "U-Net full silhouette"}[args.mask_source]
    labels = ["Input", "Pix2Gestalt RGB", mask_label, "Completed object cutout"]
    records = []

    with torch.inference_mode():
        for row, sample in enumerate(samples):
            source_image = Image.open(sample / "input.png").convert("RGB")
            generated_image = Image.open(sample / "completion.png").convert("RGB")
            visible_image = Image.open(sample / "visible_mask.png").convert("L")
            if source_image.size != generated_image.size or source_image.size != visible_image.size:
                raise ValueError(f"Input, generated RGB, and mask sizes differ for {sample.name}")
            source = np.asarray(source_image)
            generated = np.asarray(generated_image)
            visible = np.asarray(visible_image) >= 128
            if args.mask_source == "custom-unet":
                image_small = source_image.resize((size, size), Image.Resampling.LANCZOS)
                mask_small = visible_image.resize((size, size), Image.Resampling.NEAREST)
                rgb_small = np.asarray(image_small, dtype=np.float32) / 255.0
                visible_small = np.asarray(mask_small) >= 128
                tensor = np.concatenate([rgb_small, visible_small[..., None].astype(np.float32)], axis=2)
                tensor = torch.from_numpy(tensor.transpose(2, 0, 1).copy())[None].to(device)
                probability = model(tensor).float().sigmoid()[0, 0].cpu().numpy()
                predicted_small = probability >= args.threshold
                predicted = np.asarray(Image.fromarray(predicted_small.astype(np.uint8) * 255).resize(
                    source_image.size, Image.Resampling.NEAREST)) >= 128
            else:
                mask_folder = "composites-v0" if args.mask_source == "rgb-white-floodfill" else "composites-v1"
                estimated_path = results / mask_folder / sample.name / "estimated_foreground.png"
                if not estimated_path.is_file():
                    raise FileNotFoundError(f"Missing extracted foreground mask: {estimated_path}")
                predicted = np.asarray(Image.open(estimated_path).convert("L")) >= 128
            full_mask = predicted | visible

            # Preserve observed target RGB exactly; use the RGB generator only for
            # target pixels hidden in the input, then apply the predicted silhouette.
            rgb = generated.copy()
            rgb[visible] = source[visible]
            rgba = np.dstack([rgb, full_mask.astype(np.uint8) * 255])

            sample_out = output / sample.name
            sample_out.mkdir(parents=True, exist_ok=True)
            Image.fromarray(full_mask.astype(np.uint8) * 255).save(sample_out / "predicted_mask.png")
            Image.fromarray(rgba, mode="RGBA").save(sample_out / "completed_object.png")
            preview = Image.alpha_composite(checkerboard(source_image.size).convert("RGBA"),
                                            Image.fromarray(rgba, mode="RGBA")).convert("RGB")
            preview.save(sample_out / "completed_object_preview.png")

            panels = [source_image, generated_image,
                      overlay_mask(source, visible, full_mask), preview]
            y = gap + row * (tile + label_h + gap)
            draw.text((gap, y + 2), sample.name, fill="#d8dee9", font=font)
            for col, (panel, label) in enumerate(zip(panels, labels)):
                x = gap + col * (tile + gap)
                if row == 0:
                    draw.text((x, y), label, fill="#d8dee9", font=font)
                sheet.paste(panel.resize((tile, tile), Image.Resampling.LANCZOS),
                            (x, y + label_h))
            records.append({"sample_id": sample.name,
                            "predicted_foreground_fraction": round(float(full_mask.mean()), 4),
                            "visible_fraction": round(float(visible.mean()), 4),
                            "output": f"{sample.name}/completed_object.png",
                            "predicted_mask": f"{sample.name}/predicted_mask.png"})

    sheet.save(output / "contact_sheet.png")
    summary = {"mask_source": args.mask_source,
               "checkpoint": str(checkpoint) if args.mask_source == "custom-unet" else None,
               "threshold": args.threshold if args.mask_source == "custom-unet" else None,
               "resolution": size, "device": str(device),
               "output_kind": "isolated transparent object cutout",
               "visible_rgb": "copied exactly from input inside visible-target mask",
               "hidden_rgb": "Pix2Gestalt RGB completion",
               "silhouette": f"{args.mask_source} foreground unioned with known visible mask",
               "ground_truth_used": False,
               "limitations": "No RGB ground truth for hidden pixels; quality of hidden appearance remains subjective. No scene or occluder-layer compositing is performed.",
               "samples": records}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "README.txt").write_text(
        "The transparent object cutouts combine a custom U-Net amodal silhouette with "
        "Pix2Gestalt RGB. Visible target pixels are copied from the input; hidden RGB "
        "comes from the pretrained generator. Ground-truth masks are not used to create "
        "these outputs. This is an isolated-object completion, not a composite of the "
        "original scene, and unseen colors/details cannot be verified exactly.\n",
        encoding="utf-8")
    print(json.dumps({"output": str(output), "samples": len(records),
                      "contact_sheet": str(output / "contact_sheet.png"),
                      "device": str(device)}, indent=2))


if __name__ == "__main__":
    main()
