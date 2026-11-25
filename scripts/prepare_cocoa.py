"""Prepare a small COCOA amodal-segmentation subset for the silhouette U-Net.

COCOA annotations are distributed separately from COCO images. Download the
official COCOA annotation archive, extract its JSON, then run this script. The
script downloads only images needed for selected annotations.
"""
from pathlib import Path
import argparse
import json
import sys
import urllib.request
import urllib.error
import ssl
import time
from urllib.parse import urlsplit, urlunsplit

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amodal.data import source_split, write_manifest


def decode_compressed_rle(rle):
    """Decode COCO's compressed ASCII RLE into a row-major boolean mask."""
    encoded = rle["counts"]
    if isinstance(encoded, bytes):
        encoded = encoded.decode("ascii")
    if not isinstance(encoded, str):
        counts = [int(value) for value in encoded]
    else:
        counts = []
        position = 0
        while position < len(encoded):
            value = 0
            shift = 0
            while True:
                code = ord(encoded[position]) - 48
                position += 1
                value |= (code & 0x1F) << (5 * shift)
                more = code & 0x20
                shift += 1
                if not more:
                    if code & 0x10:
                        value |= -1 << (5 * shift)
                    break
            if len(counts) > 2:
                value += counts[-2]
            counts.append(value)
    height, width = map(int, rle["size"])
    flat = np.zeros(height * width, dtype=np.uint8)
    offset = 0
    foreground = False
    for run in counts:
        end = offset + run
        if end > flat.size or run < 0:
            raise ValueError("Invalid COCO RLE run lengths.")
        if foreground:
            flat[offset:end] = 1
        offset = end
        foreground = not foreground
    if offset != flat.size:
        raise ValueError("COCO RLE does not cover the declared mask size.")
    return flat.reshape((height, width), order="F").astype(bool)


def polygon_mask(segmentation, height, width):
    """Rasterize COCOA's one or more flat polygon coordinate arrays."""
    polygons = segmentation
    if polygons and isinstance(polygons[0], (int, float)):
        polygons = [polygons]
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        if len(polygon) >= 6 and len(polygon) % 2 == 0:
            points = [(float(polygon[i]), float(polygon[i + 1]))
                      for i in range(0, len(polygon), 2)]
            draw.polygon(points, fill=255)
    return np.asarray(mask) > 0


def crop_square(image, full, visible, pad=1.25, size=128):
    ys, xs = np.where(full)
    if not len(xs):
        raise ValueError("Amodal polygon produced an empty mask.")
    h, w = full.shape
    side = max(8, int(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * pad))
    cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
    left, top = int(cx - side // 2), int(cy - side // 2)
    right, bottom = left + side, top + side
    canvas = Image.new("RGB", (side, side), (128, 128, 128))
    src_left, src_top = max(0, left), max(0, top)
    src_right, src_bottom = min(w, right), min(h, bottom)
    if src_right <= src_left or src_bottom <= src_top:
        raise ValueError("Object crop is outside the image.")
    canvas.paste(image.crop((src_left, src_top, src_right, src_bottom)),
                 (src_left - left, src_top - top))

    def crop_mask(mask):
        out = Image.new("L", (side, side), 0)
        # Crop the mask with the same source-image window used above. Pasting
        # the entire full-frame mask at the destination offset shifts all
        # annotations away from the corresponding RGB object.
        mask_crop = Image.fromarray(mask.astype("uint8") * 255).crop(
            (src_left, src_top, src_right, src_bottom)
        )
        out.paste(mask_crop, (src_left - left, src_top - top))
        return np.asarray(out.resize((size, size), Image.Resampling.NEAREST)) >= 128

    rgb = np.asarray(canvas.resize((size, size), Image.Resampling.LANCZOS))
    return rgb, crop_mask(full), crop_mask(visible)


def coco_image_url(file_name):
    """Build the current HTTPS COCO image URL from a COCO 2014 filename."""
    name = Path(file_name).name
    for split in ("train2014", "val2014", "test2014"):
        if name.startswith(f"COCO_{split}_"):
            return f"https://images.cocodataset.org/{split}/{name}"
    raise ValueError(f"Unsupported COCO 2014 image filename: {file_name}")


def flickr_image_url(value):
    """Normalize a COCO-provided Flickr CDN URL to Flickr's current HTTPS host."""
    if not value:
        return None
    parsed = urlsplit(value)
    if not parsed.netloc.endswith("staticflickr.com"):
        return None
    return urlunsplit(("https", "live.staticflickr.com", parsed.path, parsed.query, parsed.fragment))


def open_coco_image(url, attempts=3):
    """Download a COCO image with bounded retries and normal TLS verification."""
    last_error = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Amodal-Object-Completion/1.0"})
            with urllib.request.urlopen(request, timeout=45) as response:
                return Image.open(response).convert("RGB")
        except Exception as error:
            last_error = error
            reason = getattr(error, "reason", None)
            if isinstance(error, ssl.SSLCertVerificationError) or isinstance(reason, ssl.SSLCertVerificationError):
                break
            if attempt + 1 < attempts:
                time.sleep(1 + attempt)
    raise RuntimeError(f"Could not download COCO image from {url}: {last_error}") from last_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True,
                        help="COCOA_amodal_train2014.json or val2014.json")
    parser.add_argument("--image-dir", type=Path,
                        help="Optional local COCO image folder; otherwise images are downloaded on demand")
    parser.add_argument("--output", type=Path, default=Path("data/cocoa-small"))
    parser.add_argument("--max-images", type=int, default=240)
    parser.add_argument("--max-objects", type=int, default=900)
    parser.add_argument("--min-occlusion", type=float, default=0.05)
    parser.add_argument("--include-unoccluded", action="store_true",
                        help="Also keep fully visible instances as exact RGB targets for synthetic-occlusion training pairs")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Optional exact region names to include (case-insensitive)")
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}. Choose another path.")
    data = json.loads(args.annotations.read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in data["images"]}
    annotations = data.get("annotations", [])
    allowed_categories = ({name.strip().casefold() for name in args.categories}
                          if args.categories else None)
    by_image = {}
    for annotation in annotations:
        by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    candidates = []
    for image_id in sorted(by_image):
        if image_id not in images:
            continue
        for annotation in by_image[image_id]:
            for region_index, region in enumerate(annotation.get("regions", [])):
                if region.get("isStuff", False) or not region.get("visible_mask"):
                    continue
                if (float(region.get("occlude_rate", 0.0)) < args.min_occlusion
                        and not args.include_unoccluded):
                    continue
                if not region.get("segmentation"):
                    continue
                if (allowed_categories is not None and
                        str(region.get("name", "unknown")).strip().casefold() not in allowed_categories):
                    continue
                candidates.append((image_id, annotation, region_index, region))
    # Deterministic order and a hard bound make the first local real-data run manageable.
    selected_images = set()
    selected = []
    for item in candidates:
        image_id = item[0]
        if image_id not in selected_images and len(selected_images) >= args.max_images:
            continue
        selected_images.add(image_id)
        selected.append(item)
        if len(selected) >= args.max_objects:
            break
    if not selected:
        parser.error("No eligible occluded object regions found in this annotation file.")

    image_cache = {}
    output = args.output
    output.mkdir(parents=True)
    records = []
    for image_id, annotation, region_index, region in selected:
        info = images[image_id]
        image_path = (args.image_dir / info["file_name"]) if args.image_dir else None
        if image_path is None or not image_path.is_file():
            if image_id not in image_cache:
                url = coco_image_url(info["file_name"])
                try:
                    image_cache[image_id] = open_coco_image(url)
                except (RuntimeError, urllib.error.URLError):
                    fallback = flickr_image_url(info.get("flickr_url"))
                    if fallback is None:
                        image_cache[image_id] = None
                    else:
                        try:
                            image_cache[image_id] = open_coco_image(fallback)
                        except (RuntimeError, urllib.error.URLError):
                            image_cache[image_id] = None
                    if image_cache[image_id] is None:
                        continue
            if image_cache.get(image_id) is None:
                continue
            image = image_cache[image_id]
        else:
            image = Image.open(image_path).convert("RGB")
        width, height = image.size
        full = polygon_mask(region["segmentation"], height, width)
        visible = decode_compressed_rle(region["visible_mask"])
        if visible.shape != full.shape:
            continue
        # COCOA's visible mask should be contained by its amodal polygon. Resolve
        # minor boundary rasterization disagreements in favor of the annotation.
        visible &= full
        if not visible.any() or (visible.sum() >= full.sum() and not args.include_unoccluded):
            continue
        rgb, full_crop, visible_crop = crop_square(image, full, visible, size=args.size)
        group_id = f"cocoa-{image_id}"
        split = source_split(group_id)
        sample_id = f"{image_id}-{annotation.get('id', 0)}-{region_index}"
        folder = output / split / sample_id
        folder.mkdir(parents=True, exist_ok=False)
        occluder = full_crop & ~visible_crop
        values = {
            "image": rgb,
            "complete_mask": full_crop.astype("uint8") * 255,
            "visible_mask": visible_crop.astype("uint8") * 255,
            "occluder_mask": occluder.astype("uint8") * 255,
        }
        record = {
            "sample_id": sample_id,
            "source_id": group_id,
            "split": split,
            "occluded_fraction": float(occluder.sum() / max(1, full_crop.sum())),
            "category": region.get("name", "unknown"),
            "dataset": "COCOA",
        }
        for name, array in values.items():
            file = folder / f"{name}.png"
            Image.fromarray(array).save(file)
            record[name] = file.relative_to(output).as_posix()
        records.append(record)
        if image_id not in image_cache:
            image.close()

    if not records:
        raise RuntimeError("No valid object crops remained after mask checks.")
    summary = write_manifest(output, records, "COCOA_manual_amodal_annotations")
    summary["dataset"] = "COCOA"
    summary["annotation_file"] = args.annotations.name
    summary["unique_photos"] = len({r["source_id"] for r in records})
    summary["category_filter_casefolded"] = sorted(allowed_categories) if allowed_categories is not None else None
    summary["categories"] = sorted({r["category"] for r in records})
    summary["note"] = "Splits are grouped by source photo; images are cropped around annotated objects."
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
