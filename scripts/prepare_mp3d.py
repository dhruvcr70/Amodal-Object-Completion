"""Prepare a bounded subset of Oxford's MP3D-Amodal training release."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import pickle
import sys
import zipfile

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amodal.data import write_manifest
from prepare_cocoa import crop_square, decode_compressed_rle


class RestrictedUnpickler(pickle.Unpickler):
    """Allow only NumPy scalar/dtype constructors used in the official file."""
    def find_class(self, module, name):
        if (module, name) == ("numpy.core.multiarray", "scalar"):
            return np._core.multiarray.scalar
        if (module, name) == ("numpy", "dtype"):
            return np.dtype
        raise pickle.UnpicklingError(f"Blocked unexpected pickle global: {module}.{name}")


def safe_load_annotations(path):
    with Path(path).open("rb") as stream:
        value = RestrictedUnpickler(stream).load()
    if not isinstance(value, dict) or not {"images", "annotations"}.issubset(value):
        raise ValueError("Unexpected MP3D annotation structure.")
    return value


def split_by_scene(scene_ids, seed=42):
    """Deterministically split whole Matterport scenes, keeping scenes disjoint."""
    ordered = sorted(set(scene_ids), key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest())
    if len(ordered) < 3:
        raise ValueError("At least three distinct scenes are needed for train/val/test splits.")
    train_count = max(1, int(len(ordered) * 0.70))
    val_count = max(1, int(len(ordered) * 0.15))
    if train_count + val_count >= len(ordered):
        train_count = len(ordered) - 2
        val_count = 1
    return {scene: ("train" if i < train_count else "val" if i < train_count + val_count else "test")
            for i, scene in enumerate(ordered)}


def scene_from_filename(file_name):
    marker = "-img_name-"
    if marker not in file_name:
        raise ValueError(f"Cannot identify Matterport scene in filename: {file_name}")
    return file_name.split(marker, 1)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/mp3d-small"))
    parser.add_argument("--max-images", type=int, default=240)
    parser.add_argument("--max-objects", type=int, default=900)
    parser.add_argument("--min-occlusion", type=float, default=0.05)
    parser.add_argument("--max-occlusion", type=float, default=0.90)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}. Choose another path.")
    if args.max_images < 3 or args.max_objects < 3:
        parser.error("Use at least three images and three objects.")

    data = safe_load_annotations(args.annotations)
    image_info = {int(item["id"]): item for item in data["images"]}
    annotation_by_image = {}
    for annotation in data["annotations"]:
        annotation_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    candidates_by_scene = {}
    for image_id, info in image_info.items():
        if image_id not in annotation_by_image:
            continue
        scene = scene_from_filename(info["file_name"])
        regions = []
        for annotation in annotation_by_image[image_id]:
            for region_index, region in enumerate(annotation.get("regions", [])):
                if region.get("isStuff", False):
                    continue
                if not region.get("segmentation") or not region.get("visible_mask"):
                    continue
                if float(region.get("occlusion_rate", 0.0)) < args.min_occlusion:
                    continue
                regions.append((annotation, region_index, region))
        if regions:
            candidates_by_scene.setdefault(scene, []).append((image_id, info, regions))

    if len(candidates_by_scene) < 3:
        parser.error("At least three scenes with usable object annotations are required.")
    scene_splits = split_by_scene(candidates_by_scene, args.seed)
    # Spread the bounded image budget over environments so each scene split is represented.
    per_scene = max(1, int(np.ceil(args.max_images / len(candidates_by_scene))))
    selection = []
    for scene in sorted(candidates_by_scene):
        rows = sorted(candidates_by_scene[scene], key=lambda row: row[0])[:per_scene]
        selection.extend((scene, row) for row in rows)
    selection = selection[:args.max_images]

    with zipfile.ZipFile(args.images_zip) as archive:
        members = {Path(name).name: name for name in archive.namelist()
                   if name.lower().endswith((".jpg", ".jpeg", ".png"))}
        records = []
        output = args.output
        output.mkdir(parents=True)
        for scene, (image_id, info, regions) in selection:
            file_name = info["file_name"]
            if file_name not in members:
                raise FileNotFoundError(f"Photo is missing from the MP3D archive: {file_name}")
            with archive.open(members[file_name]) as source:
                image = Image.open(io.BytesIO(source.read())).convert("RGB")
            width, height = image.size
            for annotation, region_index, region in regions:
                full = decode_compressed_rle(region["segmentation"])
                visible = decode_compressed_rle(region["visible_mask"])
                if full.shape != (height, width) or visible.shape != full.shape:
                    raise ValueError(f"Photo/mask dimensions do not match for {file_name}.")
                visible &= full
                if not visible.any() or visible.sum() >= full.sum():
                    continue
                rgb, full_crop, visible_crop = crop_square(image, full, visible, size=args.size)
                occluder = full_crop & ~visible_crop
                fraction = float(occluder.sum() / max(1, full_crop.sum()))
                if not visible_crop.any() or not (args.min_occlusion <= fraction <= args.max_occlusion):
                    continue
                sample_id = f"mp3d-{image_id}-{annotation.get('id', 0)}-{region_index}"
                split = scene_splits[scene]
                folder = output / split / sample_id
                folder.mkdir(parents=True, exist_ok=False)
                arrays = {
                    "image": rgb,
                    "complete_mask": full_crop.astype("uint8") * 255,
                    "visible_mask": visible_crop.astype("uint8") * 255,
                    "occluder_mask": occluder.astype("uint8") * 255,
                }
                record = {
                    "sample_id": sample_id,
                    "source_id": f"mp3d-image-{image_id}",
                    "group_id": scene,
                    "split": split,
                    "occluded_fraction": fraction,
                    "category": region.get("name", "unknown"),
                    "dataset": "MP3D-Amodal",
                }
                for name, array in arrays.items():
                    path = folder / f"{name}.png"
                    Image.fromarray(array).save(path)
                    record[name] = path.relative_to(output).as_posix()
                records.append(record)
            image.close()
            if len(records) >= args.max_objects:
                break

    if not records:
        raise RuntimeError("No usable object crops were produced.")
    summary = write_manifest(output, records, "MP3D_Amodal_3D_ground_truth")
    summary.update({
        "dataset": "MP3D-Amodal",
        "annotation_file": args.annotations.name,
        "unique_photos": len({r["source_id"] for r in records}),
        "scenes": {split: sorted({r["group_id"] for r in records if r["split"] == split})
                   for split in ("train", "val", "test")},
        "categories": sorted({r["category"] for r in records}),
        "note": "Scene-disjoint splits; this training release contains four scenes, so held-out environment coverage is small.",
    })
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
