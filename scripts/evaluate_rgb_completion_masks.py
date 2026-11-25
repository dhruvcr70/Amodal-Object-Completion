"""Evaluate silhouettes extracted from RGB completions against sealed masks.

Run only after RGB inference and foreground extraction are complete. Ground-truth
masks are read exclusively here for evaluation; they are never inputs to inference
or compositing. Metrics describe the extracted silhouette, not RGB appearance.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def read_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize(size, Image.Resampling.NEAREST)) >= 128


def scores(prediction: np.ndarray, target: np.ndarray, visible: np.ndarray) -> dict[str, float]:
    def overlap(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        intersection = int(np.logical_and(a, b).sum())
        union = int(np.logical_or(a, b).sum())
        total = int(a.sum() + b.sum())
        iou = intersection / union if union else 1.0
        dice = 2 * intersection / total if total else 1.0
        return iou, dice

    full_iou, full_dice = overlap(prediction, target)
    hidden_target = target & ~visible
    hidden_prediction = prediction & ~visible
    hidden_iou, hidden_dice = overlap(hidden_prediction, hidden_target)
    hidden_recall = (int((hidden_prediction & hidden_target).sum()) / int(hidden_target.sum())
                     if hidden_target.any() else 1.0)
    return {"full_iou": full_iou, "full_dice": full_dice,
            "hidden_iou": hidden_iou, "hidden_dice": hidden_dice,
            "hidden_recall": hidden_recall}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("runs/eval-v1"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/eval-v1/rgb-derived-mask-metrics"))
    args = parser.parse_args()
    root, output = args.results.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    manifest_path = root / "evaluation-inputs/manifest.jsonl"
    manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    methods = {
        "visible_only": None,
        "white_floodfill_v0": root / "composites-v0",
        "grabcut_v1": root / "composites-v1",
        "corrected_unet_v1": root / "object-completions-v1",
        "corrected_unet_v2": root / "object-completions-v2",
    }
    rows = []
    for entry in manifest:
        sample_id = entry["sample_id"]
        sample = root / sample_id
        input_image = Image.open(sample / "input.png").convert("RGB")
        size = input_image.size
        visible = read_mask(sample / "visible_mask.png", size)
        target = read_mask(root / "evaluation-references" / sample_id / "complete_mask.png", size)
        for method, mask_root in methods.items():
            if mask_root is None:
                extracted = visible.copy()
            else:
                mask_name = ("estimated_foreground.png" if method.startswith(("white_", "grabcut_"))
                             else "predicted_mask.png")
                extracted = read_mask(mask_root / sample_id / mask_name, size)
                extracted |= visible  # Preserve all known visible target pixels.
            row = {"sample_id": sample_id, "category": entry.get("category", ""),
                   "method": method, **scores(extracted, target, visible)}
            rows.append(row)

    fields = ["sample_id", "category", "method", "full_iou", "full_dice",
              "hidden_iou", "hidden_dice", "hidden_recall"]
    with (output / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "examples": len(manifest),
        "aggregation": "macro mean of per-sample metrics",
        "threshold": "binary masks use grayscale >= 128; RGB foreground masks come from saved extraction methods",
        "visible_region_handling": "visible target mask is unioned into extracted silhouette before scoring; hidden metrics remove visible pixels from both masks",
        "empty_mask_handling": "IoU and Dice are 1 when both compared masks are empty; hidden recall is 1 when target hidden region is empty",
        "methods": {},
        "limitation": "Small fixed 12-example subset from one held-out scene; scores evaluate extracted RGB-completion silhouettes, not RGB appearance or scene compositing.",
    }
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        summary["methods"][method] = {
            metric: round(float(np.mean([row[metric] for row in subset])), 4)
            for metric in ("full_iou", "full_dice", "hidden_iou", "hidden_dice", "hidden_recall")
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
