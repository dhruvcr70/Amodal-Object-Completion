"""Render object completions and original occluders into a depth-ordered scene."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def resolve(manifest_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest_dir / path).resolve()


def full_scene_mask(path: Path, scene_size: tuple[int, int],
                    crop: list[int]) -> np.ndarray:
    """Read a full-scene mask or place a crop-sized mask at its scene location."""
    mask = Image.open(path).convert("L")
    if mask.size == scene_size:
        return np.asarray(mask) >= 128
    x0, y0, x1, y1 = crop
    crop_size = (x1 - x0, y1 - y0)
    if mask.size != crop_size:
        raise ValueError(
            f"Mask {path} must be scene-sized {scene_size} or crop-sized {crop_size}; "
            f"got {mask.size}."
        )
    full = Image.new("L", scene_size, 0)
    full.paste(mask, (x0, y0))
    return np.asarray(full) >= 128


def fill_occluders(scene: Image.Image, mask: np.ndarray, method: str) -> Image.Image:
    """Estimate the background hidden by blockers, preserving pixels outside them."""
    if method == "none" or not mask.any():
        return scene.copy()
    if method == "lama":
        try:
            from simple_lama_inpainting import SimpleLama
        except ImportError as exc:
            raise SystemExit(
                "LaMa background filling requires `pip install simple-lama-inpainting`."
            ) from exc
        prediction = SimpleLama()(scene, Image.fromarray(mask.astype(np.uint8) * 255)).convert("RGB")
    else:
        try:
            import cv2
        except ImportError as exc:
            raise SystemExit("OpenCV background filling requires `pip install opencv-python-headless`.") from exc
        source = np.asarray(scene)
        prediction = Image.fromarray(cv2.inpaint(
            cv2.cvtColor(source, cv2.COLOR_RGB2BGR),
            (mask.astype(np.uint8) * 255), 7, cv2.INPAINT_TELEA,
        )[:, :, ::-1])
    return Image.composite(prediction, scene, Image.fromarray(mask.astype(np.uint8) * 255))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, type=Path, help="Original full-scene photo.")
    parser.add_argument("--manifest", required=True, type=Path,
                        help="JSON file listing one or more Pix2Gestalt object completions.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--background-inpainter", choices=("opencv", "lama", "none"),
                        default="opencv",
                        help="Fill the blocker-shaped background hole before layer compositing (default: opencv Telea).")
    args = parser.parse_args()

    scene = args.scene.resolve()
    manifest_path = args.manifest.resolve()
    manifest_dir = manifest_path.parent
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    objects = data.get("objects") if isinstance(data, dict) else None
    if not isinstance(objects, list) or not objects:
        parser.error("Manifest must contain a non-empty 'objects' list.")

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    object_outputs = output / "objects"
    object_outputs.mkdir()
    compositor = Path(__file__).with_name("compose_scene_completion.py")
    original_scene = Image.open(scene).convert("RGB")
    scene_size = original_scene.size
    # Larger depth values mean closer to the camera. In the reveal image, objects
    # are processed back-to-front so a nearer prediction wins in overlaps.
    normalized = []
    for index, item in enumerate(objects):
        item = dict(item)
        item.setdefault("depth", index)
        item.setdefault("occluder_depth", item["depth"] + 1)
        if not isinstance(item["depth"], (int, float)) or not isinstance(
            item["occluder_depth"], (int, float)
        ):
            parser.error("Each object's depth and occluder_depth must be numbers.")
        normalized.append((index, item))
    normalized.sort(key=lambda pair: (pair[1]["depth"], pair[0]))

    current_scene = scene
    records = []
    object_layers = []
    occluder_layers: dict[Path, dict] = {}

    for index, item in normalized:
        index += 1
        name = str(item.get("name", f"object-{index:02d}"))
        if not name or Path(name).name != name:
            parser.error(f"Invalid object name: {name!r}")
        required = ("completion", "visible_mask", "occluder_mask", "crop")
        if any(key not in item for key in required):
            parser.error(f"Object {name!r} must define {', '.join(required)}.")
        crop = item["crop"]
        if not isinstance(crop, list) or len(crop) != 4:
            parser.error(f"Object {name!r} crop must be [x0, y0, x1, y1].")

        sample_output = object_outputs / name
        command = [
            sys.executable, str(compositor),
            "--scene", str(current_scene),
            "--completion", str(resolve(manifest_dir, item["completion"])),
            "--visible-mask", str(resolve(manifest_dir, item["visible_mask"])),
            "--occluder-mask", str(resolve(manifest_dir, item["occluder_mask"])),
            "--crop", *map(str, crop),
            "--output-dir", str(sample_output),
        ]
        if item.get("completion_mask"):
            command.extend(["--completion-mask", str(resolve(manifest_dir, item["completion_mask"]))])
        subprocess.run(command, check=True)
        current_scene = sample_output / "scene_completion.png"
        object_layer_path = sample_output / "completed_object_layer.png"
        occluder_path = resolve(manifest_dir, item["occluder_mask"])
        previous_occluder = occluder_layers.get(occluder_path)
        if previous_occluder and previous_occluder["depth"] != item["occluder_depth"]:
            parser.error(
                f"Shared occluder mask {occluder_path} has conflicting depths."
            )
        occluder_layers.setdefault(occluder_path, {
            "path": occluder_path,
            "depth": item["occluder_depth"],
            "crop": crop,
        })
        record = {
            "name": name,
            "output": str(current_scene),
            "crop_xyxy": crop,
            "depth": item["depth"],
            "occluder_depth": item["occluder_depth"],
            "layer": str(object_layer_path),
        }
        records.append(record)
        object_layers.append((record, object_layer_path, occluder_path))

    final_scene = output / "scene_completion.png"
    shutil.copy2(current_scene, final_scene)

    # Export editable image layers and their explicit back-to-front order. This
    # retains the source pixels for each obstruction; it does not invent the
    # background hidden behind those obstructions.
    layers_dir = output / "layers"
    layers_dir.mkdir()
    occluder_union = np.zeros((scene_size[1], scene_size[0]), dtype=bool)
    layer_entries = []
    for record, layer_path, occluder_path in object_layers:
        scene_layer_path = layers_dir / f"{record['name']}_object.png"
        shutil.copy2(layer_path, scene_layer_path)
        layer_entries.append({
            "name": record["name"], "kind": "completed_object",
            "file": str(scene_layer_path.relative_to(output)),
            "depth": record["depth"],
        })
        occluder_union |= full_scene_mask(occluder_path, scene_size, record["crop_xyxy"])

    original_rgb = np.asarray(original_scene)
    for mask_path, occluder_info in occluder_layers.items():
        mask = full_scene_mask(mask_path, scene_size, occluder_info["crop"])
        rgba = np.dstack([original_rgb, mask.astype(np.uint8) * 255])
        layer_path = layers_dir / f"occluder_{len(layer_entries):02d}.png"
        Image.fromarray(rgba).save(layer_path)
        layer_entries.append({
            "name": mask_path.stem, "kind": "original_occluder",
            "file": str(layer_path.relative_to(output)),
            "depth": occluder_info["depth"],
        })

    # Transparent where occluders were removed: downstream scene filling can
    # replace these holes before placing the exported foreground/object layers.
    background = np.dstack([original_rgb, (~occluder_union).astype(np.uint8) * 255])
    background_path = layers_dir / "background_without_occluders.png"
    Image.fromarray(background).save(background_path)
    layer_entries.append({
        "name": "background_without_occluders", "kind": "background_with_transparent_holes",
        "file": str(background_path.relative_to(output)),
        "depth": min((entry["depth"] for entry in layer_entries), default=0) - 1,
    })
    layer_entries.sort(key=lambda entry: entry["depth"])
    layer_manifest = {
        "depth_convention": "Larger depth values are closer to the camera; composite entries from low to high.",
        "layers": layer_entries,
        "limitation": "Object and occluder layers are 2D masks. Background pixels revealed by removing an occluder are estimated with the selected inpainting method; depth is supplied, not estimated.",
    }

    # Render all layers back-to-front over a blocker-removed background. The
    # explicit depth values control whether each completed object sits behind or
    # in front of each preserved blocker. This is a 2D depth-ordered composite,
    # not a geometry-aware renderer.
    background = fill_occluders(original_scene, occluder_union, args.background_inpainter)
    filled_background_path = layers_dir / "background_inpainted.png"
    background.save(filled_background_path)
    layer_entries.append({
        "name": "background_inpainted", "kind": "estimated_background_fill",
        "file": str(filled_background_path.relative_to(output)),
        "depth": min((entry["depth"] for entry in layer_entries), default=0) - 1,
    })
    layer_entries.sort(key=lambda entry: entry["depth"])
    layer_manifest["layers"] = layer_entries
    layer_manifest["background_fill_method"] = args.background_inpainter
    (output / "scene_layer_manifest.json").write_text(
        json.dumps(layer_manifest, indent=2) + "\n", encoding="utf-8"
    )
    rendered = np.asarray(background).copy()
    render_layers = []
    for record, layer_path, _ in object_layers:
        render_layers.append((record["depth"], 0, record["name"], layer_path))
    for mask_path, info in occluder_layers.items():
        rgba_path = next(
            output / entry["file"] for entry in layer_entries
            if entry["kind"] == "original_occluder" and entry["name"] == mask_path.stem
        )
        render_layers.append((info["depth"], 1, mask_path.stem, rgba_path))
    for _, kind_order, _, layer_path in sorted(render_layers, key=lambda row: (row[0], row[1], row[2])):
        layer = np.asarray(Image.open(layer_path).convert("RGBA"))
        alpha = layer[..., 3] >= 128
        rendered[alpha] = layer[..., :3][alpha]
    depth_scene_path = output / "scene_depth_composite.png"
    Image.fromarray(rendered).save(depth_scene_path)

    # Full-scene completion view: remove the marked obstruction, use an
    # inpainted estimate for the background it covered, then place all completed
    # object layers over that estimate. Unlike scene_depth_composite, this view
    # intentionally does not restore the obstruction layers.
    reconstructed = np.asarray(background).copy()
    for _, _, _, layer_path in sorted(
        (row for row in render_layers if row[1] == 0),
        key=lambda row: (row[0], row[2]),
    ):
        layer = np.asarray(Image.open(layer_path).convert("RGBA"))
        alpha = layer[..., 3] >= 128
        reconstructed[alpha] = layer[..., :3][alpha]
    reconstructed_path = output / "scene_occlusion_removed.png"
    Image.fromarray(reconstructed).save(reconstructed_path)

    summary = {
        "scene": str(scene),
        "objects_composed_in_order": records,
        "final_scene": str(final_scene),
        "reveal_preview": str(final_scene),
        "depth_composite": str(depth_scene_path),
        "full_scene_reconstruction": str(reconstructed_path),
        "layer_manifest": str(output / "scene_layer_manifest.json"),
        "background_inpainter": args.background_inpainter,
        "composition": "The reveal preview pastes completed objects through marked occluders. The depth composite inpaints the blocker region, then renders completed object and original blocker layers back-to-front using explicit user-provided depth. The full-scene reconstruction leaves the obstruction removed and fills it with the inpainted background plus completed object layers.",
        "limitation": "The blocker-removed background and hidden object appearance are estimates. Depth order is supplied, not predicted. This 2D composite does not estimate per-pixel geometry, lighting, or shadows.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "scene_completion": str(final_scene),
                      "full_scene_reconstruction": str(reconstructed_path),
                      "depth_composite": str(depth_scene_path),
                      "objects_composed": len(records)}, indent=2))


if __name__ == "__main__":
    main()
