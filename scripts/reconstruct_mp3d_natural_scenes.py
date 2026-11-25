"""Project saved Pix2Gestalt predictions into full-resolution MP3D room photos.

This is a qualitative natural-scene integration run using existing model
predictions. It keeps the room photo and all visible pixels unchanged, then
places only the predicted hidden object pixels back into the source photograph.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from evaluate_pix2gestalt_abo_rgb import foreground, overlaps


class RestrictedUnpickler(pickle.Unpickler):
    """Load only the NumPy constructors used by the local MP3D annotation file."""
    def find_class(self, module: str, name: str):
        if (module, name) == ("numpy.core.multiarray", "scalar"):
            return np._core.multiarray.scalar
        if (module, name) == ("numpy", "dtype"):
            return np.dtype
        raise pickle.UnpicklingError(f"Blocked unexpected annotation constructor: {module}.{name}")


def decode_rle(rle: dict) -> np.ndarray:
    encoded = rle["counts"]
    counts = []
    position = 0
    while position < len(encoded):
        value = 0
        shift = 0
        while True:
            code = encoded[position] - 48 if isinstance(encoded[position], int) else ord(encoded[position]) - 48
            position += 1
            value |= (code & 0x1F) << shift
            more = code & 0x20
            shift += 5
            if not more:
                if code & 0x10:
                    value |= -1 << shift
                break
        if len(counts) > 2:
            value += counts[-2]
        counts.append(value)
    height, width = map(int, rle["size"])
    flat = np.zeros(height * width, dtype=np.uint8)
    offset = 0
    foreground_run = False
    for run in counts:
        end = offset + run
        if run < 0 or end > flat.size:
            raise ValueError("Invalid MP3D compressed mask.")
        if foreground_run:
            flat[offset:end] = 1
        offset = end
        foreground_run = not foreground_run
    if offset != flat.size:
        raise ValueError("MP3D mask runs do not cover the image.")
    return flat.reshape((height, width), order="F").astype(bool)


def crop_geometry(mask: np.ndarray, pad: float = 1.25) -> tuple[int, int, int]:
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("Empty MP3D amodal mask.")
    side = max(8, int(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * pad))
    center_x, center_y = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
    left = int(center_x - side // 2)
    top = int(center_y - side // 2)
    return left, top, side


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("runs/mp3d-natural-scene-v2"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_dir
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    archive_root = root / "runs/mp3d-eval-v1/archive"
    manifest_path = archive_root / "evaluation-inputs/manifest.jsonl"
    results_root = archive_root / "runs/eval-v1"
    reference_root = archive_root / "evaluation-references"
    annotation_path = root / "data/raw/mp3d_train_annotations.pkl"
    images_zip_path = root / "data/raw/mp3d_train.zip"
    with annotation_path.open("rb") as stream:
        annotations = RestrictedUnpickler(stream).load()
    images = {int(item["id"]): item for item in annotations["images"]}
    anns_by_id = {int(item["id"]): item for item in annotations["annotations"]}
    manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    with zipfile.ZipFile(images_zip_path) as image_archive:
        zip_members = {Path(name).name: name for name in image_archive.namelist()
                       if name.lower().endswith((".jpg", ".jpeg", ".png"))}
        rows = []
        panels = []
        for entry in manifest:
            sample_id = entry["sample_id"]
            _, image_id_text, annotation_id_text, region_text = sample_id.split("-")
            image_id, annotation_id, region_index = int(image_id_text), int(annotation_id_text), int(region_text)
            image_info = images[image_id]
            region = anns_by_id[annotation_id]["regions"][region_index]
            full_mask = decode_rle(region["segmentation"])
            visible_mask = decode_rle(region["visible_mask"]) & full_mask
            left, top, side = crop_geometry(full_mask)
            member = zip_members.get(Path(image_info["file_name"]).name)
            if member is None:
                raise FileNotFoundError(f"Missing MP3D image {image_info['file_name']} in the local archive.")
            room = Image.open(io.BytesIO(image_archive.read(member))).convert("RGB")
            room_np = np.asarray(room).copy()

            prediction_path = results_root / sample_id / "completion.png"
            prediction_image = Image.open(prediction_path).convert("RGB").resize(
                (256, 256), Image.Resampling.LANCZOS)
            prediction_rgb = np.asarray(prediction_image)
            predicted = foreground(prediction_rgb.astype(np.float32) / 255.0)
            eval_visible = np.asarray(Image.open(
                results_root / sample_id / "visible_mask.png").convert("L")) >= 128
            eval_complete = np.asarray(Image.open(
                reference_root / sample_id / "complete_mask.png").convert("L").resize(
                    (256, 256), Image.Resampling.NEAREST)) >= 128
            hidden_prediction = predicted & ~eval_visible

            prediction_large = np.asarray(prediction_image.resize(
                (side, side), Image.Resampling.LANCZOS))
            mask_large = np.asarray(Image.fromarray(
                hidden_prediction.astype(np.uint8) * 255).resize(
                    (side, side), Image.Resampling.NEAREST)) >= 128
            scene_hidden = np.zeros(room_np.shape[:2], dtype=bool)
            rgb_overlay = np.zeros_like(room_np)
            sx0, sy0 = max(0, left), max(0, top)
            sx1, sy1 = min(room.width, left + side), min(room.height, top + side)
            px0, py0 = sx0 - left, sy0 - top
            px1, py1 = px0 + (sx1 - sx0), py0 + (sy1 - sy0)
            local_mask = mask_large[py0:py1, px0:px1]
            scene_hidden[sy0:sy1, sx0:sx1] = local_mask
            rgb_overlay[sy0:sy1, sx0:sx1] = prediction_large[py0:py1, px0:px1]
            reconstruction = room_np.copy()
            reconstruction[scene_hidden] = rgb_overlay[scene_hidden]

            sample_out = output / sample_id
            sample_out.mkdir()
            original_path = sample_out / "original_room.jpg"
            result_path = sample_out / "pix2gestalt_scene_completion.jpg"
            room.save(original_path, quality=95)
            Image.fromarray(reconstruction).save(result_path, quality=95)

            pred_crop_mask = predicted
            full_iou, full_dice = overlaps(pred_crop_mask | eval_visible, eval_complete)
            hidden_iou, hidden_dice = overlaps(pred_crop_mask & ~eval_visible,
                                               eval_complete & ~eval_visible)
            rows.append({
                "sample_id": sample_id,
                "category": entry.get("category", "unknown"),
                "scene_id": entry.get("group_id"),
                "source_image": image_info["file_name"],
                "crop_xyxy": f"{left},{top},{left + side},{top + side}",
                "full_silhouette_iou": full_iou,
                "full_silhouette_dice": full_dice,
                "hidden_silhouette_iou": hidden_iou,
                "hidden_silhouette_dice": hidden_dice,
                "predicted_scene_pixels": int(scene_hidden.sum()),
                "rgb_ground_truth_available": False,
            })
            tile_w, tile_h = 360, 288
            tile = Image.new("RGB", (tile_w * 3, tile_h + 24), (30, 32, 36))
            label = ImageDraw.Draw(tile)
            label.text((5, 5), f"{sample_id} | {entry.get('category', '')}", fill="white")
            room_preview = room.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
            pred_preview = prediction_image.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
            recon_preview = Image.fromarray(reconstruction).resize((tile_w, tile_h), Image.Resampling.LANCZOS)
            tile.paste(room_preview, (0, 24))
            tile.paste(pred_preview, (tile_w, 24))
            tile.paste(recon_preview, (tile_w * 2, 24))
            label.text((5, tile_h + 5), "Original room", fill="white")
            label.text((tile_w + 5, tile_h + 5), "Pix2Gestalt crop", fill="white")
            label.text((tile_w * 2 + 5, tile_h + 5), "Full-room projection", fill="white")
            panels.append(tile)
            room.close()

    fields = list(rows[0])
    with (output / "per_scene_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "experiment": "Pix2Gestalt projection into full-resolution natural MP3D room photographs",
        "scenes": len(rows),
        "source": "12 held-out MP3D-Amodal crops with saved Pix2Gestalt predictions and full source photographs from the same dataset archive.",
        "projection": "The predicted foreground is placed into the original photo using the object-mask crop transform; predicted pixels replace the source only outside the visible-object mask.",
        "mean_full_silhouette_iou": float(np.mean([row["full_silhouette_iou"] for row in rows])),
        "mean_hidden_silhouette_iou": float(np.mean([row["hidden_silhouette_iou"] for row in rows])),
        "rgb_ground_truth": "Unavailable for real occluded pixels; report silhouette metrics and visual review only.",
        "limitations": [
            "The predicted hidden pixels are model hypotheses, not verified recovery of the original unseen RGB.",
            "Each inference uses a 256x256 object crop; its pixels are enlarged to the original room-photo crop size for display.",
            "Only the object pixels predicted outside the visible mask are overlaid. The dataset does not mark the full obstruction object, so the rest of the room is not inpainted.",
            "The fixed 12-example split is small and showed mixed-to-poor completions in the earlier review.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    sheet = Image.new("RGB", (1080, len(panels) * 312), (30, 32, 36))
    for index, panel in enumerate(panels):
        sheet.paste(panel, (0, index * 312))
    sheet.save(output / "natural_room_contact_sheet.jpg", quality=90)
    print(json.dumps({"output_dir": str(output), "scenes": len(rows),
                      "mean_full_silhouette_iou": summary["mean_full_silhouette_iou"],
                      "mean_hidden_silhouette_iou": summary["mean_hidden_silhouette_iou"]}, indent=2))


if __name__ == "__main__":
    main()
