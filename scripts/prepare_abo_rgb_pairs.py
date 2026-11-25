"""Build RGB amodal-completion pairs from small ABO household product photos.

The source is the public ABO catalog-image subset. This script only accepts
images with a nearly white connected background so the object mask can be
estimated reproducibly; it then overlays synthetic opaque blockers while
retaining the original full RGB image and silhouette as exact targets.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import urllib.request
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


BASE = "https://amazon-berkeley-objects.s3.amazonaws.com/"
HOUSEHOLD_TYPES = {"CHAIR", "SOFA", "TABLE", "LAMP", "CABINET", "BOOKSHELF"}
HOUSEHOLD_WORDS = ("chair", "sofa", "couch", "table", "lamp", "cabinet", "bookshelf", "shelf")


def split_for(item_id: str) -> str:
    value = int(hashlib.sha256(f"abo:{item_id}".encode()).hexdigest()[:8], 16) / 16**8
    return "train" if value < 0.8 else "val" if value < 0.9 else "test"


def seed_for(item_id: str, variant: int, seed: int) -> int:
    data = f"{seed}:{item_id}:{variant}".encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


def product_records(path: Path) -> list[dict]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not (row.get("main_image_id") and row.get("3dmodel_id")):
                continue
            kinds = {str(x.get("value", "")).upper() for x in row.get("product_type", [])}
            names = [str(x.get("value", "")).lower() for x in row.get("item_name", [])]
            if kinds & HOUSEHOLD_TYPES or any(word in name for word in HOUSEHOLD_WORDS for name in names):
                rows.append(row)
    # Stable shuffle prevents one category or one region of the listing shard
    # from dominating a capped run.
    return sorted(rows, key=lambda r: hashlib.sha256(str(r["item_id"]).encode()).hexdigest())


def image_paths(path: Path, wanted: set[str]) -> dict[str, str]:
    found = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["image_id"] in wanted:
                found[row["image_id"]] = row["path"]
    return found


def foreground_mask(image: Image.Image, threshold: int = 235) -> tuple[np.ndarray, float]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    near_white = np.all(rgb >= threshold, axis=2)
    h, w = near_white.shape
    border = np.concatenate((near_white[0], near_white[-1], near_white[:, 0], near_white[:, -1]))
    border_white_fraction = float(border.mean())
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
        if y: add(y - 1, x)
        if y + 1 < h: add(y + 1, x)
        if x: add(y, x - 1)
        if x + 1 < w: add(y, x + 1)
    return ~background, border_white_fraction


def draw_blocker(full: np.ndarray, rng: np.random.Generator,
                 low: float, high: float) -> np.ndarray:
    h, w = full.shape
    ys, xs = np.where(full)
    for _ in range(500):
        j = int(rng.integers(len(xs)))
        cx, cy = int(xs[j]), int(ys[j])
        bw = int(rng.integers(max(12, w // 8), max(13, int(w * .72))))
        bh = int(rng.integers(max(12, h // 8), max(13, int(h * .72))))
        x0, y0, x1, y1 = cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2
        layer = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(layer)
        kind = int(rng.integers(3))
        if kind == 0:
            draw.rectangle((x0, y0, x1, y1), fill=255)
        elif kind == 1:
            draw.rounded_rectangle((x0, y0, x1, y1), radius=max(2, min(bw, bh) // 6), fill=255)
        else:
            draw.polygon([(x0, y0 + bh // 6), (x0 + bw * 2 // 3, y0),
                          (x1, y0 + bh // 5), (x1 - bw // 10, y1),
                          (x0 + bw // 5, y1 - bh // 12)], fill=255)
        occ = np.asarray(layer) > 0
        fraction = float((occ & full).sum() / full.sum())
        if low <= fraction <= high:
            return occ
    raise ValueError("Could not draw an occluder in the requested coverage range")


def blocker_texture(size: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    w, h = size
    base = rng.integers(30, 220, size=(1, 1, 3)).astype(np.float32)
    coarse = rng.normal(0, 20, size=(max(2, h // 12), max(2, w // 12))).astype(np.float32)
    texture = Image.fromarray(np.uint8(np.clip(coarse + 128, 0, 255))).resize(
        (w, h), Image.Resampling.BICUBIC)
    noise = np.asarray(texture, dtype=np.float32)[..., None] - 128
    fine = rng.normal(0, 4, size=(h, w, 3))
    return np.uint8(np.clip(base + noise + fine, 0, 255))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=Path("data/raw/abo"))
    parser.add_argument("--listing-shard", default="listings_0.json.gz")
    parser.add_argument("--output", type=Path, default=Path("data/abo-rgb-pairs-v1"))
    parser.add_argument("--cache", type=Path, default=Path("data/raw/abo/images-small"))
    parser.add_argument("--max-products", type=int, default=1000)
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-occlusion", type=float, default=.15)
    parser.add_argument("--max-occlusion", type=float, default=.50)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output exists; choose a new folder: {args.output}")
    listings = args.metadata_dir / args.listing_shard
    images_csv = args.metadata_dir / "images.csv.gz"
    if not listings.is_file() or not images_csv.is_file():
        parser.error(f"Download the ABO metadata files into {args.metadata_dir} first")
    if args.max_products < 1 or args.variants < 1 or args.size < 64:
        parser.error("Need positive product/variant counts and size >= 64")
    if not 0 <= args.min_occlusion < args.max_occlusion <= 1:
        parser.error("Occlusion bounds must satisfy 0 <= minimum < maximum <= 1")

    records = product_records(listings)[:args.max_products]
    paths = image_paths(images_csv, {row["main_image_id"] for row in records})
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True)
    accepted, rejected = [], []
    for n, row in enumerate(records, 1):
        item_id, image_id = str(row["item_id"]), row["main_image_id"]
        path = paths.get(image_id)
        if not path:
            rejected.append({"item_id": item_id, "reason": "image_not_in_metadata"})
            continue
        image_path = args.cache / f"{item_id}.jpg"
        try:
            if not image_path.is_file():
                request = urllib.request.Request(BASE + "images/small/" + path,
                                                 headers={"User-Agent": "Amodal-Object-Completion/1.0"})
                with urllib.request.urlopen(request, timeout=30) as response:
                    image_path.write_bytes(response.read())
            with Image.open(image_path) as image_file:
                target = image_file.convert("RGB").resize((args.size, args.size), Image.Resampling.LANCZOS)
        except Exception as exc:
            rejected.append({"item_id": item_id, "reason": f"download_or_decode:{type(exc).__name__}"})
            image_path.unlink(missing_ok=True)
            continue

        full, border_fraction = foreground_mask(target)
        area = float(full.mean())
        if border_fraction < .85:
            rejected.append({"item_id": item_id, "reason": "non_white_or_lifestyle_background",
                             "border_white_fraction": border_fraction})
            continue
        if not .05 <= area <= .92:
            rejected.append({"item_id": item_id, "reason": "implausible_foreground_area",
                             "foreground_fraction": area})
            continue

        image_rgb = np.asarray(target, dtype=np.uint8)
        for variant in range(args.variants):
            rng = np.random.default_rng(seed_for(item_id, variant, args.seed))
            try:
                occ = draw_blocker(full, rng, args.min_occlusion, args.max_occlusion)
            except ValueError:
                rejected.append({"item_id": item_id, "reason": "occluder_generation_failed"})
                break
            blocker = blocker_texture((args.size, args.size), rng)
            input_rgb = image_rgb.copy()
            input_rgb[occ] = blocker[occ]
            visible = full & ~occ
            key = f"{item_id}-occ-{variant:02d}"
            folder = args.output / split_for(item_id) / key
            folder.mkdir(parents=True, exist_ok=False)
            for name, array in {
                "input_rgb.png": input_rgb,
                "target_rgb.png": image_rgb,
                "visible_mask.png": visible.astype(np.uint8) * 255,
                "complete_mask.png": full.astype(np.uint8) * 255,
                "occluder_mask.png": occ.astype(np.uint8) * 255,
            }.items():
                Image.fromarray(array).save(folder / name)
            accepted.append({
                "sample_id": key, "source_id": item_id, "split": split_for(item_id),
                "category": [x.get("value", "") for x in row.get("product_type", [])],
                "item_name": next((x.get("value", "") for x in row.get("item_name", [])
                                   if x.get("language_tag") == "en_US"), ""),
                "image_id": image_id, "image_path": path,
                "foreground_fraction": area,
                "occluded_fraction": float((occ & full).sum() / full.sum()),
                **{name.removesuffix(".png"): (folder / name).relative_to(args.output).as_posix()
                   for name in ["input_rgb.png", "target_rgb.png", "visible_mask.png",
                                "complete_mask.png", "occluder_mask.png"]},
            })
        if n % 50 == 0:
            print(f"Scanned {n}/{len(records)} products; made {len(accepted)} pairs", flush=True)

    (args.output / "manifest.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in accepted), encoding="utf-8")
    (args.output / "rejected.json").write_text(json.dumps(rejected, indent=2), encoding="utf-8")
    summary = {
        "origin": "Amazon Berkeley Objects household catalog images with synthetic opaque occlusion",
        "license": "CC BY 4.0; attribute Amazon.com and the ABO dataset authors",
        "source_listing_shard": args.listing_shard, "seed": args.seed,
        "resolution": [args.size, args.size], "variants_per_product": args.variants,
        "samples": {s: sum(r["split"] == s for r in accepted) for s in ("train", "val", "test")},
        "unique_products": {s: len({r["source_id"] for r in accepted if r["split"] == s})
                            for s in ("train", "val", "test")},
        "accepted_products": len({r["source_id"] for r in accepted}),
        "pairs": len(accepted), "rejected_products": len(rejected),
        "mask_method": "Near-white pixels (RGB >= 235) connected to the image border are treated as background.",
        "limitations": ["Catalog-product images mostly have white backgrounds; this is not natural-scene RGB completion.",
                        "Foreground masks are heuristic and should be visually reviewed.",
                        "Synthetic blockers do not model every real occlusion pattern."],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
