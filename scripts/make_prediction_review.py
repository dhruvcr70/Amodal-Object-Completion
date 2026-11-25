"""Create a deterministic visual review sheet for silhouette predictions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal.data import CompletionDataset  # noqa: E402
from amodal.model import CompletionUNet  # noqa: E402


def overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    result = image.astype(np.float32).copy()
    result[mask] = result[mask] * 0.45 + np.asarray(color, dtype=np.float32) * 0.55
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/mp3d-v4-hidden-weight-val/best.pt")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/mp3d-subset-v3")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/prediction-review-v4-test.png")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--min-occlusion", type=float, default=0.0)
    parser.add_argument("--max-occlusion", type=float, default=1.0)
    args = parser.parse_args()
    args.checkpoint, args.dataset, args.output = map(Path.resolve, (args.checkpoint, args.dataset, args.output))
    if not 0.0 <= args.min_occlusion < args.max_occlusion <= 1.0:
        parser.error("Occlusion bounds must satisfy 0 <= min < max <= 1.")

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    data = CompletionDataset(args.dataset, args.split)
    eligible = []
    for index in range(len(data)):
        x, target, _ = data[index]
        full = target[0] > 0.5
        visible = x[3] > 0.5
        hidden_fraction = float((full & ~visible).sum() / full.sum().clamp_min(1))
        if args.min_occlusion <= hidden_fraction < args.max_occlusion:
            eligible.append((index, hidden_fraction))
    if not eligible:
        parser.error("No samples fall within the requested occlusion range.")
    count = min(args.count, len(eligible))
    positions = np.linspace(0, len(eligible) - 1, count, dtype=int).tolist()
    indices = [eligible[position][0] for position in positions]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompletionUNet(state["config"]["base_channels"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    panels: list[tuple[str, Image.Image]] = []
    font = ImageFont.load_default()
    for index in indices:
        x, target, sample_id = data[index]
        rgb = np.clip(x[:3].numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        visible = x[3].numpy() > 0.5
        truth = target[0].numpy() > 0.5
        probability = model(x.unsqueeze(0).to(device)).float().sigmoid()[0, 0].cpu().numpy()
        prediction = (probability > args.threshold) | visible
        panels.extend([
            ("Input", Image.fromarray(rgb)),
            ("Visible mask", overlay(rgb, visible, (255, 190, 0))),
            (f"Prediction (t={args.threshold:.2f})", overlay(rgb, prediction, (30, 220, 100))),
            ("Ground truth", overlay(rgb, truth, (30, 130, 255))),
        ])

    width, height = panels[0][1].size
    label_h, gap = 22, 6
    cols = 4
    rows = count
    sheet = Image.new("RGB", (cols * width + (cols + 1) * gap, rows * (height + label_h) + (rows + 1) * gap), "#20242b")
    draw = ImageDraw.Draw(sheet)
    for row in range(rows):
        sample_id = data[indices[row]][2]
        for col in range(cols):
            title, panel = panels[row * cols + col]
            if col == 0:
                title = f"{sample_id} | {title}"
            x0 = gap + col * (width + gap)
            y0 = gap + row * (height + label_h) + label_h
            draw.text((x0, y0 - label_h + 3), title, fill="#d8dee9", font=font)
            sheet.paste(panel, (x0, y0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(json.dumps({"output": str(args.output), "split": args.split, "samples": count,
                      "threshold": args.threshold, "device": str(device),
                      "occlusion_range": [args.min_occlusion, args.max_occlusion],
                      "eligible_samples": len(eligible),
                      "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None}, indent=2))


if __name__ == "__main__":
    main()
