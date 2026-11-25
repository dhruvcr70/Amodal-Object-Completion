"""Score Pix2Gestalt candidate silhouettes against held-out validation masks.

RGB generations have a white background and no exported alpha/mask. The evaluator
estimates each silhouette by treating near-white pixels connected to the image
border as background. These are approximate silhouette metrics, not RGB metrics.
"""
from __future__ import annotations

import csv
import json
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "runs/candidate-review-v1"
DATA = ROOT / "data/mp3d-subset-v4-corrected-crops"
OUTPUT = RESULTS / "evaluation"
WHITE_THRESHOLD = 245


def read_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return np.asarray(image) >= 128


def generated_foreground(path: Path, threshold: int = WHITE_THRESHOLD) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    near_white = np.all(rgb >= threshold, axis=2)
    height, width = near_white.shape
    background = np.zeros_like(near_white)
    queue: deque[tuple[int, int]] = deque()

    def add(y: int, x: int) -> None:
        if near_white[y, x] and not background[y, x]:
            background[y, x] = True
            queue.append((y, x))

    for x in range(width):
        add(0, x)
        add(height - 1, x)
    for y in range(height):
        add(y, 0)
        add(y, width - 1)
    while queue:
        y, x = queue.popleft()
        if y: add(y - 1, x)
        if y + 1 < height: add(y + 1, x)
        if x: add(y, x - 1)
        if x + 1 < width: add(y, x + 1)
    return ~background


def overlap(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    intersection = int(np.logical_and(pred, target).sum())
    union = int(np.logical_or(pred, target).sum())
    total = int(pred.sum() + target.sum())
    return (intersection / union if union else 1.0,
            2 * intersection / total if total else 1.0)


def main() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for item in summary["records"]:
        if item["status"] != "completed":
            continue
        sample_id = item["sample_id"]
        target_dir = DATA / "val" / sample_id
        truth = read_mask(target_dir / "complete_mask.png", (256, 256))
        visible = read_mask(target_dir / "visible_mask.png", (256, 256))
        hidden_truth = truth & ~visible
        for index in range(1, 5):
            generated = generated_foreground(RESULTS / sample_id / f"candidate_{index:02d}.png")
            # Preserve observed pixels; score completion only where the object was hidden.
            completed = generated | visible
            full_iou, full_dice = overlap(completed, truth)
            hidden_iou, hidden_dice = overlap(generated & ~visible, hidden_truth)
            rows.append({"sample_id": sample_id, "category": item.get("category"),
                         "occluded_fraction": item.get("occluded_fraction"),
                         "candidate": index, "generated_foreground_fraction": float(generated.mean()),
                         "full_iou": full_iou, "full_dice": full_dice,
                         "hidden_iou": hidden_iou, "hidden_dice": hidden_dice})
        # Visible-mask-only baseline establishes whether generation adds value.
        base_full_iou, base_full_dice = overlap(visible, truth)
        base_hidden_iou, base_hidden_dice = overlap(np.zeros_like(visible), hidden_truth)
        rows.append({"sample_id": sample_id, "category": item.get("category"),
                     "occluded_fraction": item.get("occluded_fraction"),
                     "candidate": "visible_only_baseline", "generated_foreground_fraction": float(visible.mean()),
                     "full_iou": base_full_iou, "full_dice": base_full_dice,
                     "hidden_iou": base_hidden_iou, "hidden_dice": base_hidden_dice})

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "category", "occluded_fraction", "candidate",
              "generated_foreground_fraction", "full_iou", "full_dice", "hidden_iou", "hidden_dice"]
    with (OUTPUT / "per_candidate_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    generated = [row for row in rows if isinstance(row["candidate"], int)]
    baseline = [row for row in rows if row["candidate"] == "visible_only_baseline"]
    per_sample = {}
    for item in summary["records"]:
        sid = item["sample_id"]
        group = [row for row in generated if row["sample_id"] == sid]
        if group:
            best = max(group, key=lambda row: float(row["hidden_iou"]))
            per_sample[sid] = {
                "category": item.get("category"), "occluded_fraction": item.get("occluded_fraction"),
                "mean_candidate_hidden_iou": round(float(np.mean([row["hidden_iou"] for row in group])), 4),
                "best_candidate_hidden_iou_oracle": round(float(best["hidden_iou"]), 4),
                "best_candidate_index_oracle": best["candidate"],
                "visible_only_hidden_iou": round(float(next(r for r in baseline if r["sample_id"] == sid)["hidden_iou"]), 4),
            }
    report = {
        "experiment": "Pix2Gestalt candidate silhouette validation",
        "split": "validation",
        "samples": len(per_sample),
        "candidate_count": len(generated),
        "silhouette_extraction": f"Near-white pixels (all RGB channels >= {WHITE_THRESHOLD}) connected to border are treated as background; remaining pixels are predicted foreground.",
        "metrics": "IoU and Dice on full silhouette and on hidden pixels only. Generated masks are unioned with the known visible mask for full metrics.",
        "oracle_warning": "Best-candidate scores use ground truth to pick among four outputs and are an upper bound, not deployable selection performance.",
        "mean_candidate_hidden_iou": round(float(np.mean([r["hidden_iou"] for r in generated])), 4) if generated else None,
        "visible_only_baseline_hidden_iou": round(float(np.mean([r["hidden_iou"] for r in baseline])), 4) if baseline else None,
        "per_sample": per_sample,
        "limitations": ["Only four validation images; not a generalization estimate.", "Silhouette is inferred from a white-background RGB image because the inference script does not export a model mask.", "No RGB appearance, depth, or scene compositing is scored."],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("CSV:", OUTPUT / "per_candidate_metrics.csv")


if __name__ == "__main__":
    main()
