"""Evaluate a saved silhouette checkpoint over a complete, fixed test split."""
from __future__ import annotations

import argparse
import csv
import hashlib
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


def ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 1.0


def overlap(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    total = int(prediction.sum() + target.sum())
    return ratio(intersection, union), ratio(2 * intersection, total)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/mp3d-v5-combined-val/best.pt")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/mp3d-subset-v3")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/eval-v2-full-test")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Optional exact category names to include, e.g. couch cabinet table")
    parser.add_argument("--allow-different-dataset", action="store_true",
                        help="Allow evaluation on a different, explicitly selected dataset split")
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.dataset = args.dataset.resolve()
    args.output = args.output.resolve()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_dataset = state.get("config", {}).get("dataset")
    requested_dataset = str(args.dataset.relative_to(ROOT)).replace("\\", "/")
    if checkpoint_dataset != requested_dataset and not args.allow_different_dataset:
        raise ValueError("Checkpoint dataset does not match the requested dataset.")
    data = CompletionDataset(args.dataset, args.split)
    category_by_id = {record["sample_id"]: record.get("category", "unknown")
                      for record in data.records}
    if args.categories:
        allowed = {value.casefold() for value in args.categories}
        data.records = [record for record in data.records
                        if record.get("category", "unknown").casefold() in allowed]
        if not data.records:
            parser.error("No samples match the requested categories in this split.")
        category_by_id = {record["sample_id"]: record.get("category", "unknown")
                          for record in data.records}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompletionUNet(state["config"]["base_channels"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    loader = DataLoader(data, batch_size=16, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")

    rows = []
    for x, y, sample_ids in loader:
        probabilities = model(x.to(device)).float().sigmoid().cpu().numpy()[:, 0] > args.threshold
        visible_batch = x[:, 3].numpy() > 0.5
        truth_batch = y[:, 0].numpy() > 0.5
        for sample_id, pred, visible, truth in zip(sample_ids, probabilities, visible_batch, truth_batch):
            pred = pred | visible
            hidden_pred = pred & ~visible
            hidden_truth = truth & ~visible
            full_iou, full_dice = overlap(pred, truth)
            hidden_iou, hidden_dice = overlap(hidden_pred, hidden_truth)
            baseline_iou, baseline_dice = overlap(visible, truth)
            hidden_recall = ratio(int((hidden_pred & hidden_truth).sum()), int(hidden_truth.sum()))
            rows.append({
                "sample_id": sample_id,
                "category": category_by_id[sample_id],
                "occlusion_fraction": ratio(int(hidden_truth.sum()), int(truth.sum())),
                "full_iou": full_iou,
                "full_dice": full_dice,
                "hidden_iou": hidden_iou,
                "hidden_dice": hidden_dice,
                "hidden_recall": hidden_recall,
                "visible_only_iou": baseline_iou,
                "visible_only_dice": baseline_dice,
            })

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "per_sample_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    means = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]
             if key not in {"sample_id", "category", "occlusion_fraction"}}
    split_summary = json.loads((args.dataset / "summary.json").read_text(encoding="utf-8"))
    dataset_label = split_summary.get("dataset", args.dataset.name)
    evaluation_scenes = split_summary.get("scenes", {}).get(args.split, [])
    source_photo_count = (split_summary.get("photos", {}).get(args.split)
                          or split_summary.get("sources", {}).get(args.split))
    if evaluation_scenes:
        interpretation_limit = (f"The {args.split} split contains {len(evaluation_scenes)} scene(s); "
                                "results do not establish broad multi-scene generalization.")
    else:
        interpretation_limit = (f"The {args.split} split contains {source_photo_count} held-out source photo(s); "
                                "results do not establish broad-scene generalization.")
    report = {
        "experiment": f"full {dataset_label} {args.split} evaluation",
        "checkpoint": str(args.checkpoint.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_training_dataset": checkpoint_dataset,
        "cross_dataset_evaluation": checkpoint_dataset != requested_dataset,
        "dataset": str(args.dataset.relative_to(ROOT)).replace("\\", "/"),
        "evaluation_split": args.split,
        "dataset_manifest_sha256": hashlib.sha256((args.dataset / "manifest.jsonl").read_bytes()).hexdigest(),
        "scene_disjoint_split": True if evaluation_scenes else None,
        "source_disjoint_split": True,
        "evaluation_scenes": evaluation_scenes,
        "evaluation_scene_count": len(evaluation_scenes) if evaluation_scenes else None,
        "evaluation_source_photo_count": source_photo_count,
        "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "threshold": args.threshold,
        "prediction_rule": "threshold U-Net probabilities, then union with visible mask",
        "aggregation": "macro mean over test samples",
        "samples": len(rows),
        "category_filter": sorted({row["category"] for row in rows}) if args.categories else None,
        "metrics": means,
        "visible_only_baseline": {
            "full_iou": means["visible_only_iou"],
            "full_dice": means["visible_only_dice"],
        },
        "interpretation_limit": interpretation_limit,
        "per_sample_csv": "per_sample_metrics.csv",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
