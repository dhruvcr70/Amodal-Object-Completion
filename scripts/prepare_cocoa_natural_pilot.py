"""Prepare a small source-disjoint COCOA natural-photo Pix2Gestalt pilot.

The test photos contain only a small annotated natural occlusion. A deterministic
synthetic vertical blocker is layered over the annotated visible target pixels;
the original photo is retained as the RGB reference for those newly covered
pixels. Complete masks and references are evaluation-only inputs.
"""
from __future__ import annotations

import argparse
import json
import statistics
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) >= 128


def prepare(root: Path, output: Path, max_natural_occlusion: float = 0.05) -> dict:
    dataset = root / "data/cocoa-household-v3"
    manifest = [json.loads(line) for line in (dataset / "manifest.jsonl").read_text().splitlines() if line.strip()]
    eligible = [row for row in manifest if row["split"] == "test" and row["occluded_fraction"] <= max_natural_occlusion]
    if not eligible:
        raise RuntimeError("No source-held-out low-occlusion natural-photo test examples found.")

    output.mkdir(parents=True, exist_ok=True)
    records = []
    for row in eligible:
        sid = row["sample_id"]
        source = dataset / row["image"]
        sample = source.parent
        rgb = Image.open(source).convert("RGB")
        complete = _mask(dataset / row["complete_mask"])
        visible = _mask(dataset / row["visible_mask"])
        if rgb.size != (complete.shape[1], complete.shape[0]) or complete.shape != visible.shape:
            raise ValueError(f"Mismatched image/masks in {sid}.")
        ys, xs = np.where(complete)
        if not len(xs):
            continue
        h, w = complete.shape
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        # A central vertical strip hides a repeatable, meaningful portion of
        # the target while retaining visible pixels on both sides when possible.
        bw = max(3, int(round((x1 - x0) * 0.28)))
        bx0 = max(0, min(w - bw, (x0 + x1 - bw) // 2))
        by0 = max(0, y0 - max(1, (y1 - y0) // 20))
        by1 = min(h, y1 + max(1, (y1 - y0) // 20))
        blocker = np.zeros((h, w), dtype=bool)
        blocker[by0:by1, bx0:bx0 + bw] = True
        hidden = blocker & complete & visible
        if hidden.sum() < max(4, int(complete.sum() * 0.04)):
            # Try an offset strip when the center crosses a natural hole.
            bx0 = max(0, min(w - bw, x0 + (x1 - x0) // 3))
            blocker[:] = False
            blocker[by0:by1, bx0:bx0 + bw] = True
            hidden = blocker & complete & visible
        visible_model = visible & ~blocker
        if not visible_model.any() or hidden.sum() < 4:
            continue

        arr = np.asarray(rgb).copy()
        # Use a local, image-derived neutral obstruction color (the input still
        # contains its original scene outside this synthetic wall strip).
        border = np.zeros((h, w), dtype=bool)
        border[max(0, by0 - 2):min(h, by1 + 2), max(0, bx0 - 2):min(w, bx0 + bw + 2)] = True
        border &= ~blocker & ~complete
        if border.any():
            color = np.median(arr[border], axis=0).astype(np.uint8)
        else:
            color = np.array([180, 180, 180], dtype=np.uint8)
        arr[blocker] = color

        dest = output / "samples" / sid
        dest.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(dest / "image.png")
        Image.fromarray(visible_model.astype(np.uint8) * 255).save(dest / "visible_mask.png")
        Image.fromarray(blocker.astype(np.uint8) * 255).save(dest / "synthetic_occlusion_mask.png")
        Image.fromarray(hidden.astype(np.uint8) * 255).save(dest / "hidden_reference_mask.png")
        rgb.save(dest / "reference.png")
        Image.fromarray(complete.astype(np.uint8) * 255).save(dest / "complete_mask.png")
        records.append({"sample_id": sid, "source_id": row["source_id"], "category": row["category"],
                        "natural_occluded_fraction": row["occluded_fraction"],
                        "synthetic_hidden_fraction_of_object": float(hidden.sum() / complete.sum()),
                        "image": f"samples/{sid}/image.png",
                        "visible_mask": f"samples/{sid}/visible_mask.png",
                        "reference": f"samples/{sid}/reference.png",
                        "complete_mask": f"samples/{sid}/complete_mask.png",
                        "synthetic_occlusion_mask": f"samples/{sid}/synthetic_occlusion_mask.png",
                        "hidden_reference_mask": f"samples/{sid}/hidden_reference_mask.png"})

    if not records:
        raise RuntimeError("No eligible samples remained after requiring usable hidden and visible pixels.")
    result = {"experiment": "COCOA natural-photo controlled-occlusion pilot",
              "dataset": "COCOA household v3", "source_split": "test (source-photo disjoint)",
              "max_native_occlusion": max_natural_occlusion,
              "n": len(records), "occluder": "deterministic central vertical strip sampled from image-local background color",
              "model_inputs": ["image.png", "visible_mask.png"],
              "evaluation_only": ["reference.png", "complete_mask.png", "synthetic_occlusion_mask.png", "hidden_reference_mask.png"],
              "metrics_scope": "Score only synthetic-hidden pixels that were visible in the original photo; natural occlusions remain a limitation.",
              "records": records}
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(output))
    result["archive"] = str(archive)
    result["archive_bytes"] = archive.stat().st_size
    result["mean_synthetic_hidden_fraction"] = round(statistics.mean(r["synthetic_hidden_fraction_of_object"] for r in records), 4)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-natural-occlusion", type=float, default=0.05)
    args = parser.parse_args()
    output = args.output or args.root / "data/cocoa-natural-pilot-v1"
    print(json.dumps(prepare(args.root, output, args.max_natural_occlusion), indent=2))


if __name__ == "__main__":
    main()
