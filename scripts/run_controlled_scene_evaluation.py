"""Evaluate saved Pix2Gestalt ABO predictions in synthetic room scenes.

The clean object and full-scene composition are retained as ground truth. Each
occluded object is placed into a real MP3D room background; its existing
Pix2Gestalt crop prediction is composed back into the scene. This evaluates
scene integration without claiming that Pix2Gestalt saw or understood the room.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from evaluate_pix2gestalt_abo_rgb import foreground, rgb_metrics, overlaps


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) >= 128


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--sample-count", type=int, default=46,
                        help="Held-out examples to use; default is all 46.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("runs/controlled-scene-eval-v2"))
    args = parser.parse_args()
    root = args.root.resolve()
    dataset = root / "data/abo-rgb-pairs-v1"
    predictions = root / "runs/pix2gestalt-abo-rgb-eval-v1/predictions"
    background_zip = root / "data/raw/mp3d_train.zip"
    output = args.output_dir
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"Refusing to overwrite non-empty output directory: {output}")
    if not dataset.joinpath("manifest.jsonl").is_file() or not background_zip.is_file():
        parser.error("Need the local ABO test dataset and MP3D room-image archive.")
    output.mkdir(parents=True, exist_ok=True)

    manifest_rows = [json.loads(line) for line in dataset.joinpath("manifest.jsonl").read_text().splitlines()
                     if line.strip()]
    test_rows = [row for row in manifest_rows if row.get("split") == "test"]
    if args.sample_count < 1 or args.sample_count > len(test_rows):
        parser.error(f"--sample-count must be between 1 and {len(test_rows)}.")
    test_rows = test_rows[:args.sample_count]
    missing = [row["sample_id"] for row in test_rows
               if not (predictions / f'{row["sample_id"]}.png').is_file()]
    if missing:
        parser.error(f"Missing saved Pix2Gestalt predictions for: {missing[:5]}")

    bg_member = "mp3d_train_10.29/scene-wc2JMjhGNzB-img_name-3e3a6adeb5ec4c2497b7524fb91c4491_i1_2.jpg"
    with zipfile.ZipFile(background_zip) as archive:
        background = Image.open(BytesIO(archive.read(bg_member))).convert("RGB")
    background = ImageOps.fit(background, (640, 480), method=Image.Resampling.LANCZOS)
    background.save(output / "room_background.png")

    x0, y0 = 192, 180
    x1, y1 = x0 + 256, y0 + 256
    crop = [x0, y0, x1, y1]
    composer = root / "scripts/compose_scene_batch.py"
    scene_rows: list[dict] = []
    previews: list[Image.Image] = []

    for index, entry in enumerate(test_rows, start=1):
        sid = entry["sample_id"]
        sample = dataset / "test" / sid
        outdir = output / sid
        outdir.mkdir()
        target = read_rgb(sample / "target_rgb.png")
        visible = read_mask(sample / "visible_mask.png")
        complete = read_mask(sample / "complete_mask.png")
        blocker = read_mask(sample / "occluder_mask.png")
        corrupted_crop = read_rgb(sample / "input_rgb.png")
        prediction_path = predictions / f"{sid}.png"
        prediction = np.asarray(Image.open(prediction_path).convert("RGB").resize((256, 256)))
        predicted_mask = foreground(prediction.astype(np.float32) / 255.0)
        predicted_mask_path = outdir / "predicted_object_mask.png"
        Image.fromarray(predicted_mask.astype(np.uint8) * 255).save(predicted_mask_path)

        clean_scene = np.asarray(background).copy()
        clean_crop = clean_scene[y0:y1, x0:x1]
        clean_crop[complete] = target[complete]
        occluded_scene = clean_scene.copy()
        occluded_scene[y0:y1, x0:x1][blocker] = corrupted_crop[blocker]

        scene_path = outdir / "occluded_scene.png"
        target_scene_path = outdir / "ground_truth_scene.png"
        Image.fromarray(occluded_scene).save(scene_path)
        Image.fromarray(clean_scene).save(target_scene_path)
        scene_occ_mask = np.zeros((480, 640), dtype=np.uint8)
        scene_occ_mask[y0:y1, x0:x1] = blocker.astype(np.uint8) * 255
        occ_path = outdir / "occluder_mask.png"
        Image.fromarray(scene_occ_mask).save(occ_path)

        manifest = {"objects": [{
            "name": "target", "completion": str(prediction_path.resolve()),
            "visible_mask": str((sample / "visible_mask.png").resolve()),
            "completion_mask": str(predicted_mask_path.resolve()),
            "occluder_mask": str(occ_path.resolve()), "crop": crop,
            "depth": 2, "occluder_depth": 1,
        }]}
        manifest_path = outdir / "scene_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        scene_output = outdir / "composite"
        subprocess.run([
            sys.executable, str(composer), "--scene", str(scene_path),
            "--manifest", str(manifest_path), "--output-dir", str(scene_output),
            "--background-inpainter", "opencv",
        ], check=True, cwd=root)

        reconstructed_path = scene_output / "scene_occlusion_removed.png"
        reconstructed = read_rgb(reconstructed_path).astype(np.float32) / 255.0
        clean_float = clean_scene.astype(np.float32) / 255.0
        input_float = occluded_scene.astype(np.float32) / 255.0
        hidden_obj = np.zeros((480, 640), dtype=bool)
        hidden_obj[y0:y1, x0:x1] = blocker & complete
        hidden_bg = np.zeros((480, 640), dtype=bool)
        hidden_bg[y0:y1, x0:x1] = blocker & ~complete
        blocker_scene = np.zeros((480, 640), dtype=bool)
        blocker_scene[y0:y1, x0:x1] = blocker
        object_scores = rgb_metrics(reconstructed, clean_float, hidden_obj)
        background_scores = rgb_metrics(reconstructed, clean_float, hidden_bg)
        blocker_scores = rgb_metrics(reconstructed, clean_float, blocker_scene)
        input_object = rgb_metrics(input_float, clean_float, hidden_obj)
        input_blocker = rgb_metrics(input_float, clean_float, blocker_scene)

        object_layer = np.asarray(Image.open(
            scene_output / "objects/target/completed_object_layer.png").convert("RGBA"))
        predicted_full = object_layer[y0:y1, x0:x1, 3] >= 128
        full_iou, full_dice = overlaps(predicted_full, complete)
        hidden_iou, hidden_dice = overlaps(predicted_full & ~visible, complete & ~visible)
        scene_rows.append({
            "sample_id": sid, "category": ",".join(entry.get("category", [])),
            "source_id": entry.get("source_id"), "occluded_fraction": entry.get("occluded_fraction"),
            "full_object_iou": full_iou, "full_object_dice": full_dice,
            "hidden_object_iou": hidden_iou, "hidden_object_dice": hidden_dice,
            "hidden_object_rgb_mae": object_scores["hidden_rgb_mae"],
            "hidden_object_rgb_ssim": object_scores["hidden_rgb_ssim"],
            "input_hidden_object_rgb_mae": input_object["hidden_rgb_mae"],
            "blocker_region_rgb_mae": blocker_scores["hidden_rgb_mae"],
            "input_blocker_region_rgb_mae": input_blocker["hidden_rgb_mae"],
        "blocker_background_rgb_mae": (
            background_scores["hidden_rgb_mae"]
            if np.isfinite(background_scores["hidden_rgb_mae"]) else None
        ),
            "hidden_object_pixels": int(hidden_obj.sum()),
            "blocker_background_pixels": int(hidden_bg.sum()),
        })
        preview = Image.open(reconstructed_path).convert("RGB").resize((160, 120))
        draw = ImageDraw.Draw(preview)
        draw.rectangle((0, 0, 159, 16), fill=(20, 20, 20))
        draw.text((4, 3), f"{index:02d} {sid}", fill="white")
        previews.append(preview)

    fields = sorted({key for row in scene_rows for key in row})
    with (output / "per_scene_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scene_rows)
    metric_names = [
        "full_object_iou", "hidden_object_iou", "hidden_object_rgb_mae",
        "hidden_object_rgb_ssim", "input_hidden_object_rgb_mae",
        "blocker_region_rgb_mae", "input_blocker_region_rgb_mae",
        "blocker_background_rgb_mae",
    ]
    aggregates = {}
    for name in metric_names:
        values = np.asarray([row[name] for row in scene_rows if row[name] is not None], dtype=np.float64)
        aggregates[name] = {
            "mean": float(values.mean()) if values.size else None,
            "std": float(values.std()) if values.size else None,
            "samples": int(values.size),
        }
    summary = {
        "experiment": "Synthetic full-scene composition from held-out Pix2Gestalt ABO predictions",
        "scenes": len(scene_rows), "unique_products": len({row["source_id"] for row in scene_rows}),
        "background": f"MP3D room image from local archive: {bg_member}",
        "object_inference": "Reused previously generated Pix2Gestalt predictions on matching synthetic ABO crops; no model inference ran on the room scene itself.",
        "scene_ground_truth": "Exact synthetic clean scene made by placing original target RGB and complete mask over the room background before applying a known synthetic blocker.",
        "aggregate_mean_and_std": aggregates,
        "interpretation": "Hidden-object scores measure projected object completion. Blocker-region scores measure object-plus-background filling. This controlled cutout composite does not establish natural-photo generalization.",
        "outputs_per_scene": ["occluded_scene.png", "ground_truth_scene.png", "composite/scene_occlusion_removed.png", "composite/scene_depth_composite.png"],
        "limitations": [
            "The room background is a real MP3D photo, but each catalog object is pasted into it synthetically.",
            "Pix2Gestalt predictions were generated on white-background product crops, not conditioned on room context.",
            "Object placement and depth are supplied by construction; camera geometry is not inferred.",
            "Only full-resolution composition and masked region scores are evaluated here.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    columns = 6
    rows_n = (len(previews) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 160, rows_n * 120), (28, 30, 34))
    for i, preview in enumerate(previews):
        sheet.paste(preview, ((i % columns) * 160, (i // columns) * 120))
    sheet.save(output / "reconstructed_scene_contact_sheet.jpg", quality=90)
    # Keep a single fully labeled sample easy to inspect alongside the aggregate
    # sheet. This sample is selected by stable ID, not by its metric ranking.
    example_id = "B082BLFMRC-occ-01"
    if (output / example_id).is_dir():
        panels = [
            ("OCCLUDED INPUT", output / example_id / "occluded_scene.png"),
            ("GROUND-TRUTH CLEAN SCENE", output / example_id / "ground_truth_scene.png"),
            ("RECONSTRUCTION", output / example_id / "composite/scene_occlusion_removed.png"),
        ]
        comparison = Image.new("RGB", (3 * 480, 390), (28, 30, 34))
        for i, (label, image_path) in enumerate(panels):
            tile = Image.open(image_path).convert("RGB").resize((480, 360))
            comparison.paste(tile, (i * 480, 30))
            ImageDraw.Draw(comparison).text((i * 480 + 8, 8), label, fill="white")
        comparison.save(output / "example_scene_comparison.png")
    print(json.dumps({"output_dir": str(output), "scenes": len(scene_rows),
                      "metrics": aggregates}, indent=2))


if __name__ == "__main__":
    main()
