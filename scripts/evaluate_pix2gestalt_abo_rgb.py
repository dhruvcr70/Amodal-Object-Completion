"""Score Pix2Gestalt RGB and silhouette outputs against sealed ABO test targets."""
from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def member(z: zipfile.ZipFile, name: str) -> bytes:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe path in manifest: {name}")
    return z.read(name)


def load_rgb(data: bytes, size: tuple[int, int]) -> np.ndarray:
    from io import BytesIO
    image = Image.open(BytesIO(data)).convert("RGB")
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def load_mask(data: bytes, size: tuple[int, int]) -> np.ndarray:
    from io import BytesIO
    image = Image.open(BytesIO(data)).convert("L")
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return np.asarray(image) >= 128


def foreground(rgb: np.ndarray, threshold: int = 245) -> np.ndarray:
    """Estimate foreground as non-near-white pixels not connected to the border."""
    near_white = np.all(rgb >= threshold / 255.0, axis=2)
    height, width = near_white.shape
    background = np.zeros_like(near_white)
    stack: list[tuple[int, int]] = []

    def add(y: int, x: int) -> None:
        if near_white[y, x] and not background[y, x]:
            background[y, x] = True
            stack.append((y, x))

    for x in range(width):
        add(0, x); add(height - 1, x)
    for y in range(height):
        add(y, 0); add(y, width - 1)
    while stack:
        y, x = stack.pop()
        if y: add(y - 1, x)
        if y + 1 < height: add(y + 1, x)
        if x: add(y, x - 1)
        if x + 1 < width: add(y, x + 1)
    return ~background


def box_mean(array: np.ndarray, radius: int = 5) -> np.ndarray:
    window = radius * 2 + 1
    if array.ndim == 2:
        array = array[..., None]
    h, w, channels = array.shape
    pad = np.pad(array.astype(np.float64), ((radius, radius), (radius, radius), (0, 0)),
                 mode="reflect")
    integral = np.pad(pad.cumsum(0).cumsum(1), ((1, 0), (1, 0), (0, 0)))
    sums = (integral[window:window + h, window:window + w]
            - integral[:h, window:window + w]
            - integral[window:window + h, :w]
            + integral[:h, :w])
    return sums / (window * window)


def masked_ssim(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Standard local SSIM map averaged over hidden target pixels and RGB channels."""
    x, y = pred.astype(np.float64), target.astype(np.float64)
    mx, my = box_mean(x), box_mean(y)
    vx = np.maximum(box_mean(x * x) - mx * mx, 0.0)
    vy = np.maximum(box_mean(y * y) - my * my, 0.0)
    cov = box_mean(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mx * my + c1) * (2 * cov + c2)
             / ((mx * mx + my * my + c1) * (vx + vy + c2) + 1e-12))
    if not mask.any():
        return float("nan")
    return float(score[mask].mean())


def overlaps(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    intersection = int(np.logical_and(pred, target).sum())
    union = int(np.logical_or(pred, target).sum())
    total = int(pred.sum() + target.sum())
    return (intersection / union if union else 1.0,
            2 * intersection / total if total else 1.0)


def rgb_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    diff = pred[mask] - target[mask]
    mae = float(np.abs(diff).mean()) if diff.size else float("nan")
    mse = float(np.square(diff).mean()) if diff.size else float("nan")
    # Keep the JSON report standards-compliant if a baseline matches exactly.
    psnr = 100.0 if mse == 0 else (-10 * math.log10(mse) if mse > 0 else float("nan"))
    return {"hidden_rgb_mae": mae, "hidden_rgb_psnr_db": psnr,
            "hidden_rgb_ssim": masked_ssim(pred, target, mask)}


def overlay(rgb: np.ndarray, mask: np.ndarray, color=(255, 170, 25)) -> Image.Image:
    image = Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255)).convert("RGBA")
    tint = Image.new("RGBA", image.size, (*color, 0))
    tint.putalpha(Image.fromarray(np.uint8(mask) * 105))
    return Image.alpha_composite(image, tint).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("runs/pix2gestalt-abo-test-v1.zip"))
    parser.add_argument("--results", type=Path, default=Path("runs/rgb-eval-v1"))
    parser.add_argument("--output", type=Path, default=Path("runs/pix2gestalt-abo-test-v1/evaluation"))
    args = parser.parse_args()
    if not args.archive.is_file() or not args.results.is_dir():
        parser.error("Need the packaged test archive and completed Pix2Gestalt result directory")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Refusing to overwrite non-empty evaluation folder: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    per_sample: list[dict] = []
    panels: list[tuple[str, Image.Image, Image.Image, Image.Image, Image.Image]] = []
    with zipfile.ZipFile(args.archive) as archive:
        manifest = [json.loads(line) for line in member(archive, "manifest.jsonl").decode().splitlines()
                    if line.strip()]
        for entry in manifest:
            sid = entry["sample_id"]
            sample_dir = args.results / sid
            # Accept both layouts emitted by the Kaggle notebook runners:
            #   <results>/<sample_id>/completion.png
            #   <results>/<sample_id>.png
            prediction_path = sample_dir / "completion.png"
            if not prediction_path.is_file():
                flat_prediction = args.results / f"{sid}.png"
                if flat_prediction.is_file():
                    prediction_path = flat_prediction
            if not prediction_path.is_file():
                per_sample.append({"sample_id": sid, "status": "missing_prediction"})
                continue

            input_rgb = load_rgb(member(archive, entry["image"]), (256, 256))
            visible = load_mask(member(archive, entry["visible_mask"]), (256, 256))
            target_rgb = load_rgb(member(archive, entry["target_rgb"]), (256, 256))
            full_gt = load_mask(member(archive, entry["complete_mask"]), (256, 256))
            occluder = load_mask(member(archive, entry["occluder_mask"]), (256, 256))
            hidden_gt = full_gt & occluder
            generated_rgb = np.asarray(Image.open(prediction_path).convert("RGB").resize(
                (256, 256), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
            generated_mask = foreground(generated_rgb)

            # Shape is scored after unioning the exact known visible pixels.
            full_iou, full_dice = overlaps(generated_mask | visible, full_gt)
            hidden_iou, hidden_dice = overlaps(generated_mask & ~visible, full_gt & ~visible)
            generated_scores = rgb_metrics(generated_rgb, target_rgb, hidden_gt)
            input_scores = rgb_metrics(input_rgb, target_rgb, hidden_gt)
            white = np.ones_like(target_rgb)
            white_scores = rgb_metrics(white, target_rgb, hidden_gt)
            row = {
                "sample_id": sid, "category": entry.get("category", "unknown"),
                "source_id": entry.get("source_id"),
                "occluded_fraction": float(entry.get("occluded_fraction", 0)),
                "status": "scored", "full_silhouette_iou": full_iou,
                "full_silhouette_dice": full_dice, "hidden_silhouette_iou": hidden_iou,
                "hidden_silhouette_dice": hidden_dice, **generated_scores,
                "input_hidden_rgb_mae": input_scores["hidden_rgb_mae"],
                "white_fill_hidden_rgb_mae": white_scores["hidden_rgb_mae"],
            }
            per_sample.append(row)
            input_panel = overlay(input_rgb, visible)
            pred_panel = Image.open(prediction_path).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
            target_panel = Image.fromarray(np.uint8(target_rgb * 255))
            # Red highlights the pixels used for hidden RGB scoring on the target.
            target_hidden_panel = overlay(target_rgb, hidden_gt, (230, 45, 55))
            panels.append((sid, input_panel, pred_panel, target_panel, target_hidden_panel))

    scored = [row for row in per_sample if row.get("status") == "scored"]
    if not scored:
        raise SystemExit(
            "No predictions matched the test manifest. Expected either "
            "<results>/<sample_id>/completion.png or <results>/<sample_id>.png."
        )
    fields = sorted({key for row in per_sample for key in row})
    with (args.output / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(per_sample)
    metric_names = ["full_silhouette_iou", "full_silhouette_dice", "hidden_silhouette_iou",
                    "hidden_silhouette_dice", "hidden_rgb_mae", "hidden_rgb_psnr_db",
                    "hidden_rgb_ssim", "input_hidden_rgb_mae", "white_fill_hidden_rgb_mae"]
    aggregates = {name: {"mean": float(np.nanmean([row[name] for row in scored])),
                         "std": float(np.nanstd([row[name] for row in scored]))}
                  for name in metric_names}
    summary = {
        "experiment": "Pix2Gestalt RGB and silhouette evaluation on ABO synthetic occlusions",
        "samples_in_archive": len(per_sample), "samples_scored": len(scored),
        "failed_or_missing": len(per_sample) - len(scored),
        "unique_test_products_scored": len({row["source_id"] for row in scored}),
        "rgb_region": "Pixels inside the known product silhouette and synthetic blocker region.",
        "rgb_metrics": "MAE and PSNR are per-channel RGB metrics normalized to [0,1]; SSIM is local 11x11-window SSIM averaged over hidden target pixels and channels.",
        "silhouette_method": "Generated foreground is estimated by border-connected near-white background thresholding (RGB >= 245). Known visible mask is unioned before full-shape scoring.",
        "baselines": "input MAE scores the corrupted image as-is; white-fill MAE predicts white in hidden regions.",
        "metrics_macro_mean_and_std": aggregates,
        "limitations": [
            "The held-out test contains a small number of products from one ABO listing shard.",
            "Synthetic occluders and white-background catalog photos do not represent natural room scenes.",
            "Foreground extraction from generated RGB can fail on white or near-white object parts.",
            "RGB metrics are pixel-aligned and penalize plausible but shifted/stochastic completions.",
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
                                                 encoding="utf-8")

    tile, label, gap = 256, 34, 8
    columns, rows = 4, len(panels)
    sheet = Image.new("RGB", (columns * tile + (columns + 1) * gap,
                               rows * (tile + label + gap) + gap), (34, 38, 46))
    draw = ImageDraw.Draw(sheet); font = ImageFont.load_default()
    headings = ["Occluded input + visible mask", "Pix2Gestalt completion",
                "Unoccluded ground truth", "Ground truth: scored hidden pixels"]
    for col, heading in enumerate(headings):
        draw.text((gap + col * (tile + gap), 8), heading, fill="white", font=font)
    for row_index, (sid, *images) in enumerate(panels):
        y = gap + label + row_index * (tile + label + gap)
        for col, panel in enumerate(images):
            sheet.paste(panel.resize((tile, tile), Image.Resampling.LANCZOS),
                        (gap + col * (tile + gap), y))
        draw.text((gap, y + tile + 4), sid, fill=(230, 232, 238), font=font)
    sheet.save(args.output / "contact_sheet.png")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
