"""Score Pix2Gestalt candidates on the source-held-out COCOA natural-photo pilot."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np
from PIL import Image


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize(size, Image.Resampling.NEAREST)) >= 128


def foreground_from_white(image: np.ndarray) -> np.ndarray:
    """Approximate silhouette from white-background Pix2Gestalt RGB output."""
    return np.any(image < 245, axis=2)


def score(image_path: Path, ref_path: Path, complete_path: Path, hidden_path: Path,
          blocker_path: Path, visible_path: Path) -> dict[str, float]:
    size = (256, 256)
    pred = np.asarray(Image.open(image_path).convert("RGB").resize(size, Image.Resampling.LANCZOS), dtype=np.float32) / 255
    ref = np.asarray(Image.open(ref_path).convert("RGB").resize(size, Image.Resampling.LANCZOS), dtype=np.float32) / 255
    complete = load_mask(complete_path, size)
    hidden = load_mask(hidden_path, size)
    blocker = load_mask(blocker_path, size)
    visible = load_mask(visible_path, size)
    pred_mask = foreground_from_white((pred * 255).astype(np.uint8))
    union_mask = pred_mask | visible

    def iou(a: np.ndarray, b: np.ndarray) -> float:
        union = np.logical_or(a, b).sum()
        return float(np.logical_and(a, b).sum() / union) if union else 1.0

    region_intersection = (pred_mask & complete & blocker).sum()
    region_union = ((pred_mask | complete) & blocker).sum()
    hidden_shape_iou = float(region_intersection / region_union) if region_union else 1.0

    def rgb_stats(region: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        if not region.any():
            return float("nan"), float("nan")
        delta = a[region] - b[region]
        mae = float(np.abs(delta).mean())
        mse = float(np.square(delta).mean())
        psnr = 99.0 if mse == 0 else 10.0 * math.log10(1.0 / mse)
        return mae, psnr

    hidden_mae, hidden_psnr = rgb_stats(hidden, pred, ref)
    visible_mae, visible_psnr = rgb_stats(visible, pred, ref)
    return {"hidden_mask_iou": hidden_shape_iou if hidden.any() else float("nan"),
            "full_mask_iou": iou(union_mask, complete),
            "hidden_rgb_mae": hidden_mae, "hidden_rgb_psnr_db": hidden_psnr,
            "visible_rgb_mae": visible_mae, "visible_rgb_psnr_db": visible_psnr,
            "hidden_pixels": int(hidden.sum())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Extracted cocoa-natural-pilot-v1 archive directory.")
    parser.add_argument("--runs", type=Path, required=True,
                        help="Directory containing <sample_id>/candidate_01.png ...")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.input_dir / "manifest.json").read_text(encoding="utf-8"))
    records: list[dict] = []
    for row in manifest["records"]:
        sample_id = row["sample_id"]
        sample = args.input_dir / "samples" / sample_id
        run = args.runs / sample_id
        base = {"sample_id": sample_id, "source_id": row["source_id"],
                "category": row["category"], "candidate": 0,
                "natural_occluded_fraction": row["natural_occluded_fraction"],
                "synthetic_hidden_fraction_of_object": row["synthetic_hidden_fraction_of_object"]}
        for index in range(1, 5):
            path = run / f"candidate_{index:02d}.png"
            if not path.is_file():
                continue
            metrics = score(path, sample / "reference.png", sample / "complete_mask.png",
                            sample / "hidden_reference_mask.png", sample / "synthetic_occlusion_mask.png",
                            sample / "visible_mask.png")
            records.append({**base, "candidate": index, **metrics})
    if not records:
        raise RuntimeError("No candidate images found. Expected candidate_01.png through candidate_04.png in per-sample run folders.")

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "per_candidate_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    by_sample = {}
    for row in records:
        by_sample.setdefault(row["sample_id"], []).append(row)
    best_hidden = [max(rows, key=lambda x: x["hidden_mask_iou"]) for rows in by_sample.values()]
    summary = {"experiment": manifest["experiment"], "dataset_split": manifest["source_split"],
               "samples_scored": len(by_sample), "candidate_images_scored": len(records),
               "candidate_count_per_sample": 4,
               "mean_candidate_hidden_mask_iou": round(statistics.mean(r["hidden_mask_iou"] for r in records), 4),
               "mean_oracle_best_hidden_mask_iou": round(statistics.mean(r["hidden_mask_iou"] for r in best_hidden), 4),
               "mean_candidate_full_mask_iou": round(statistics.mean(r["full_mask_iou"] for r in records), 4),
               "mean_hidden_rgb_mae": round(statistics.mean(r["hidden_rgb_mae"] for r in records), 4),
               "mean_hidden_rgb_psnr_db": round(statistics.mean(r["hidden_rgb_psnr_db"] for r in records), 3),
               "mean_visible_rgb_mae": round(statistics.mean(r["visible_rgb_mae"] for r in records), 4),
               "mean_visible_rgb_psnr_db": round(statistics.mean(r["visible_rgb_psnr_db"] for r in records), 3),
               "oracle_caveat": "Best-candidate selection uses the hidden ground truth and is an upper bound, not a deployable selector.",
               "limitations": ["Small pilot of seven source-held-out photos.",
                               "Original COCOA photos have up to 5% native occlusion; scoring excludes native-hidden pixels.",
                               "Synthetic local-color blocker does not represent all real-world occluders.",
                               "Pix2Gestalt output is aligned as a 256x256 object crop; RGB scores are sensitive to spatial misalignment."]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("CSV:", csv_path)


if __name__ == "__main__":
    main()
