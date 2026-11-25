"""Evaluate saved MP3D baseline completions against held-out masks.

Generated foreground is approximated from the white-background RGB completion:
near-white pixels connected to the canvas border are treated as background.
This is a silhouette proxy, not an RGB-quality or amodal-scene-compositing metric.
"""
from __future__ import annotations

import csv
import json
import sys
import zipfile
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads/mp3d-eval-v1-results.zip"
OUT = ROOT / "runs/mp3d-eval-v1/evaluation"
THRESHOLD = 245


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    im = Image.open(path).convert("L")
    if im.size != size:
        im = im.resize(size, Image.Resampling.NEAREST)
    return np.asarray(im) >= 128


def generated_foreground(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    near_white = np.all(rgb >= THRESHOLD, axis=2)
    h, w = near_white.shape
    background = np.zeros_like(near_white)
    queue: deque[tuple[int, int]] = deque()

    def add(y: int, x: int) -> None:
        if near_white[y, x] and not background[y, x]:
            background[y, x] = True
            queue.append((y, x))

    for x in range(w):
        add(0, x)
        add(h - 1, x)
    for y in range(h):
        add(y, 0)
        add(y, w - 1)
    while queue:
        y, x = queue.popleft()
        if y > 0: add(y - 1, x)
        if y + 1 < h: add(y + 1, x)
        if x > 0: add(y, x - 1)
        if x + 1 < w: add(y, x + 1)
    return ~background


def scores(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    inter = int(np.logical_and(pred, target).sum())
    union = int(np.logical_or(pred, target).sum())
    total = int(pred.sum() + target.sum())
    return (inter / union if union else 1.0, 2 * inter / total if total else 1.0)


def main() -> None:
    if not ARCHIVE.is_file():
        raise FileNotFoundError(f"Results archive not found: {ARCHIVE}")
    extracted = ROOT / "runs/mp3d-eval-v1/archive"
    extracted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as zf:
        for item in zf.infolist():
            dest = (extracted / item.filename).resolve()
            if extracted.resolve() not in dest.parents and dest != extracted.resolve():
                raise ValueError(f"Unsafe archive path: {item.filename}")
        zf.extractall(extracted)

    run_root = extracted / "runs/eval-v1"
    refs = extracted / "evaluation-references"
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for record in json.loads((run_root / "results.json").read_text(encoding="utf-8")):
        if record.get("status") != "inference_completed":
            continue
        sid = record["sample_id"]
        size = (256, 256)
        truth = load_mask(refs / sid / "complete_mask.png", size)
        visible = load_mask(run_root / sid / "visible_mask.png", size)
        generated = generated_foreground(run_root / sid / "completion.png")
        hidden_truth = truth & ~visible
        full_iou, full_dice = scores(generated | visible, truth)
        hidden_iou, hidden_dice = scores(generated & ~visible, hidden_truth)
        base_iou, base_dice = scores(visible, truth)
        rows.append({"sample_id": sid, "category": record.get("category", ""),
                     "occluded_fraction": float(hidden_truth.sum() / truth.sum()) if truth.sum() else 0.0,
                     "full_iou": full_iou, "full_dice": full_dice,
                     "hidden_iou": hidden_iou, "hidden_dice": hidden_dice,
                     "visible_only_full_iou": base_iou, "visible_only_full_dice": base_dice})

    if not rows:
        raise RuntimeError("No completed samples with references were found")
    OUT.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (OUT / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "experiment": "Pix2Gestalt MP3D baseline silhouette evaluation",
        "samples_scored": len(rows), "samples_generated": summary.get("examples"),
        "mean_hidden_region_iou": round(float(np.mean([r["hidden_iou"] for r in rows])), 4),
        "mean_hidden_region_dice": round(float(np.mean([r["hidden_dice"] for r in rows])), 4),
        "mean_full_silhouette_iou": round(float(np.mean([r["full_iou"] for r in rows])), 4),
        "mean_full_silhouette_dice": round(float(np.mean([r["full_dice"] for r in rows])), 4),
        "mean_visible_only_full_iou": round(float(np.mean([r["visible_only_full_iou"] for r in rows])), 4),
        "silhouette_extraction": f"Near-white RGB pixels (all channels >= {THRESHOLD}) connected to image border are background.",
        "limitations": ["Small fixed 12-image set; not a statistically strong generalization result.",
                        "Predicted masks are inferred from RGB output, not exported by the model.",
                        "Scores silhouettes only; RGB fidelity, depth, and occluder-aware scene compositing are not measured."]
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    # Red = false positive, blue = false negative, green = overlap.
    tile = 256
    label_h = 24
    sheet = Image.new("RGB", (tile * 4, (tile + label_h) * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for n, row in enumerate(rows):
        sid = str(row["sample_id"])
        y = n * (tile + label_h)
        draw.text((2, y + 4), f"{sid} | hidden IoU {row['hidden_iou']:.3f}", fill="black")
        pred = generated_foreground(run_root / sid / "completion.png")
        truth = load_mask(refs / sid / "complete_mask.png", (256, 256))
        visible = load_mask(run_root / sid / "visible_mask.png", (256, 256))
        target_hidden = truth & ~visible
        predicted_hidden = pred & ~visible
        vis = np.zeros((256, 256, 3), dtype=np.uint8)
        vis[np.logical_and(predicted_hidden, target_hidden)] = (40, 200, 60)
        vis[np.logical_and(predicted_hidden, ~target_hidden)] = (245, 50, 50)
        vis[np.logical_and(~predicted_hidden, target_hidden)] = (45, 95, 240)
        panels = [Image.open(run_root / sid / "input.png").convert("RGB"),
                  Image.fromarray(visible.astype(np.uint8) * 255).convert("RGB"),
                  Image.fromarray(truth.astype(np.uint8) * 255).convert("RGB"),
                  Image.fromarray(vis)]
        for j, panel in enumerate(panels):
            sheet.paste(panel.resize((tile, tile)), (j * tile, y + label_h))
            draw.text((j * tile + 4, y + label_h + 4),
                      ["Input", "Visible mask", "Full GT", "Hidden overlap: G/R/B"][j],
                      fill="yellow", stroke_width=1, stroke_fill="black")
    sheet.save(OUT / "mask_comparison.png")
    print(json.dumps(result, indent=2))
    print("CSV:", OUT / "per_sample_metrics.csv")


if __name__ == "__main__":
    main()
