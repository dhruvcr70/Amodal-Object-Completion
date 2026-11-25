"""Predict amodal masks for the fixed eval set and package inference-only inputs.

The export deliberately excludes complete/occluder ground-truth masks so they
cannot leak into the RGB-completion experiment.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal.model import CompletionUNet  # noqa: E402


def predict(model: torch.nn.Module, image_path: Path, mask_path: Path,
            size: int, device: torch.device) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    visible_mask = Image.open(mask_path).convert("L")
    if image.size != visible_mask.size:
        raise ValueError(f"Image/mask size mismatch: {image_path}")
    rgb = np.asarray(image.resize((size, size), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    visible = np.asarray(visible_mask.resize((size, size), Image.Resampling.NEAREST)) >= 128
    tensor = torch.from_numpy(np.concatenate([rgb, visible[..., None].astype(np.float32)], axis=2)
                              .transpose(2, 0, 1).copy())[None].to(device)
    with torch.inference_mode():
        probability = model(tensor).sigmoid()[0, 0].cpu().numpy()
    coarse = Image.fromarray((probability > 0.5).astype(np.uint8) * 255)
    coarse = coarse.resize(image.size, Image.Resampling.NEAREST)
    full = (np.asarray(coarse) >= 128) | (np.asarray(visible_mask) >= 128)
    return Image.fromarray(full.astype(np.uint8) * 255)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, default=ROOT / "data/eval-v1")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/mp3d-v2/best.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/guidance-inputs-v1")
    parser.add_argument("--zip", type=Path, default=ROOT / "runs/guidance-inputs-v1.zip")
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompletionUNet(state["config"]["base_channels"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    manifest = [json.loads(line) for line in (args.eval_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export = []
    for row in manifest:
        sid = row["sample_id"]
        src = args.eval_dir / "samples" / sid
        dst = args.output_dir / "samples" / sid
        dst.mkdir(parents=True, exist_ok=True)
        for filename in ("image.png", "visible_mask.png"):
            (dst / filename).write_bytes((src / filename).read_bytes())
        predicted = predict(model, src / "image.png", src / "visible_mask.png", state["resolution"], device)
        predicted.save(dst / "predicted_amodal_mask.png")
        export.append({"sample_id": sid, "category": row.get("category"),
                       "occluded_fraction": row.get("occluded_fraction"),
                       "image": f"samples/{sid}/image.png",
                       "visible_mask": f"samples/{sid}/visible_mask.png",
                       "predicted_amodal_mask": f"samples/{sid}/predicted_amodal_mask.png"})

    metadata = {
        "experiment": "Pix2Gestalt mask-conditioning exploratory ablation",
        "warning": "Pix2Gestalt was trained with visible masks. Predicted amodal-mask conditioning is a zero-shot out-of-distribution experiment.",
        "silhouette_checkpoint": str(args.checkpoint.relative_to(ROOT)) if args.checkpoint.is_relative_to(ROOT) else args.checkpoint.name,
        "silhouette_resolution": state["resolution"],
        "threshold": 0.5,
        "prediction_rule": "threshold U-Net probability and union with input visible mask",
        "device_used_for_mask_prediction": str(device),
        "ground_truth_masks_included": False,
        "samples": export,
    }
    (args.output_dir / "experiment_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (args.output_dir / "run_guidance_ablation.py").write_bytes(
        (ROOT / "scripts/run_guidance_ablation.py").read_bytes())
    (args.output_dir / "README.txt").write_text(
        "Upload this ZIP to the existing Colab notebook, extract it under "
        "/content/amodal-completion/inputs/guidance-inputs-v1, then run "
        "/content/amodal-completion/inputs/guidance-inputs-v1/run_guidance_ablation.py "
        "with the notebook's isolated Python environment.\n\n"
        "The Pix2Gestalt predicted-amodal-mask condition is exploratory and "
        "out of distribution: its pretrained model expects visible masks.\n",
        encoding="utf-8")
    args.zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(args.output_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(args.output_dir))
    print(json.dumps({"samples": len(export), "device": str(device), "zip": str(args.zip),
                      "bytes": args.zip.stat().st_size, "ground_truth_masks_included": False}, indent=2))


if __name__ == "__main__":
    main()
