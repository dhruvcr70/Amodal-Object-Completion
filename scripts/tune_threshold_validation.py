"""Sweep U-Net mask thresholds on validation data only; never read the test split."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal.data import CompletionDataset  # noqa: E402
from amodal.model import CompletionUNet  # noqa: E402


THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)


def safe_ratio(n: int, d: int) -> float:
    return n / d if d else 1.0


def score(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    intersection = int((pred & target).sum())
    union = int((pred | target).sum())
    total = int(pred.sum() + target.sum())
    return safe_ratio(intersection, union), safe_ratio(2 * intersection, total)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/mp3d-v4-hidden-weight-val/best.pt")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/mp3d-subset-v3")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/eval-val-threshold-sweep-v4")
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.dataset = args.dataset.resolve()
    args.output = args.output.resolve()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    dataset = CompletionDataset(args.dataset, "val")
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompletionUNet(state["config"]["base_channels"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    per_threshold: dict[float, list[dict[str, float | str]]] = {t: [] for t in THRESHOLDS}
    for x, y, sample_ids in loader:
        probability = model(x.to(device)).float().sigmoid().cpu().numpy()[:, 0]
        visible_batch = x[:, 3].numpy() > 0.5
        truth_batch = y[:, 0].numpy() > 0.5
        for sample_id, prob, visible, truth in zip(sample_ids, probability, visible_batch, truth_batch):
            hidden_truth = truth & ~visible
            baseline_iou, _ = score(visible, truth)
            for threshold in THRESHOLDS:
                prediction = (prob > threshold) | visible
                hidden_prediction = prediction & ~visible
                full_iou, full_dice = score(prediction, truth)
                hidden_iou, hidden_dice = score(hidden_prediction, hidden_truth)
                hidden_recall = safe_ratio(int((hidden_prediction & hidden_truth).sum()), int(hidden_truth.sum()))
                per_threshold[threshold].append({
                    "sample_id": sample_id,
                    "full_iou": full_iou,
                    "full_dice": full_dice,
                    "hidden_iou": hidden_iou,
                    "hidden_dice": hidden_dice,
                    "hidden_recall": hidden_recall,
                    "visible_only_iou": baseline_iou,
                })

    args.output.mkdir(parents=True, exist_ok=True)
    fields = ["threshold", "samples", "full_iou", "full_dice", "hidden_iou", "hidden_dice", "hidden_recall", "visible_only_iou"]
    results = []
    for threshold, rows in per_threshold.items():
        means = {field: float(np.mean([row[field] for row in rows])) for field in fields[2:] if field != "threshold"}
        results.append({"threshold": threshold, "samples": len(rows), **means})
    csv_path = args.output / "threshold_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    best_hidden = max(results, key=lambda row: row["hidden_iou"])
    best_full = max(results, key=lambda row: row["full_iou"])
    report = {
        "split": "validation",
        "checkpoint": str(args.checkpoint.relative_to(ROOT)).replace("\\", "/"),
        "dataset": str(args.dataset.relative_to(ROOT)).replace("\\", "/"),
        "thresholds": list(THRESHOLDS),
        "aggregation": "macro mean per sample",
        "best_validation_hidden_iou": best_hidden,
        "best_validation_full_iou": best_full,
        "all_thresholds": results,
        "note": "Threshold selection is validation-only. Evaluate the chosen fixed threshold once on test before reporting final results.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
