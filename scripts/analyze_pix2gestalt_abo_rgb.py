"""Add visible-region consistency and occlusion-severity analysis to a saved Pix2Gestalt ABO run."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


BANDS = ((0.0, 0.15, "0-15%"), (0.15, 0.30, "15-30%"),
         (0.30, 0.45, "30-45%"), (0.45, 0.60, "45-60%"),
         (0.60, 1.01, "60-100%"))
METRICS = ("hidden_rgb_mae", "hidden_rgb_psnr_db", "hidden_rgb_ssim",
           "full_silhouette_iou", "hidden_silhouette_iou",
           "input_hidden_rgb_mae", "visible_rgb_mae")


def mean_std(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std()), "n": int(array.size)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/abo-rgb-pairs-v1/test"))
    parser.add_argument("--metrics", type=Path,
                        default=Path("runs/pix2gestalt-abo-rgb-eval-v1/recomputed-v2/per_sample_metrics.csv"))
    parser.add_argument("--predictions", type=Path, default=Path("runs/pix2gestalt-abo-rgb-eval-v1/predictions"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/pix2gestalt-abo-rgb-eval-v1/recomputed-v2/supplementary-analysis.json"))
    args = parser.parse_args()
    with args.metrics.open(newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row.get("status") == "scored"]
    indexed = {row["sample_id"]: row for row in rows}
    enriched = []
    missing = []
    for sid, row in indexed.items():
        folder = args.dataset / sid
        prediction = args.predictions / f"{sid}.png"
        required = [folder / "target_rgb.png", folder / "visible_mask.png"]
        if not prediction.is_file() or not all(p.is_file() for p in required):
            missing.append(sid)
            continue
        pred = np.asarray(Image.open(prediction).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS), dtype=np.float32) / 255
        target = np.asarray(Image.open(required[0]).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS), dtype=np.float32) / 255
        visible = np.asarray(Image.open(required[1]).convert("L").resize((256, 256), Image.Resampling.NEAREST)) >= 128
        enriched.append({**row, "visible_rgb_mae": float(np.abs(pred[visible] - target[visible]).mean()) if visible.any() else float("nan")})
    if not enriched:
        parser.error("No scored predictions matched the test dataset and prediction folders.")

    bands = []
    for low, high, name in BANDS:
        members = [r for r in enriched if low <= float(r["occluded_fraction"]) < high]
        if not members:
            continue
        bands.append({"occlusion_band": name, "samples": len(members),
                      "unique_products": len({r["source_id"] for r in members}),
                      "metrics": {metric: mean_std([float(r[metric]) for r in members])
                                  for metric in METRICS}})
    report = {
        "experiment": "Supplementary source-disjoint Pix2Gestalt RGB analysis",
        "samples_scored": len(enriched), "unique_products": len({r["source_id"] for r in enriched}),
        "missing_artifacts": missing,
        "visible_rgb_mae_definition": "Mean absolute RGB difference between raw Pix2Gestalt output and unobstructed target, over known visible target pixels; lower means greater appearance consistency.",
        "overall_metrics": {metric: mean_std([float(r[metric]) for r in enriched]) for metric in METRICS},
        "by_occlusion_severity": bands,
        "limitations": [
            "This is a supplementary analysis of the existing 46 synthetic test outputs, not a new model run.",
            "The 23 product identities are source-disjoint from project train/validation but may overlap Pix2Gestalt pretraining.",
            "The test contains white-background catalog objects and synthetic blockers, not natural room images.",
            "Visible RGB MAE measures raw generator consistency; the interactive scene compositor separately copies observed target pixels exactly.",
            "Means and standard deviations describe this small test and are not confidence intervals.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "samples": len(enriched),
                      "severity_bands": len(bands)}, indent=2))


if __name__ == "__main__":
    main()
