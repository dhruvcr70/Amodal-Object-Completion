"""Build supervised RGB completion pairs from fully visible COCOA household crops.

Each target is an original, complete RGB crop. The input is the same crop with
an opaque synthetic foreground patch over part of the object, plus visible and
complete masks. Splits are inherited from the COCOA manifest to prevent a photo
from crossing train/validation/test boundaries.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def stable_seed(seed, key):
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def make_occlusion(image, full_mask, rng, min_fraction, max_fraction):
    h, w = full_mask.shape
    ys, xs = np.where(full_mask)
    if len(xs) == 0:
        raise ValueError("empty full-object mask")
    for _ in range(500):
        i = int(rng.integers(len(xs)))
        cx, cy = int(xs[i]), int(ys[i])
        ow = int(rng.integers(max(8, w // 8), max(9, int(w * .72))))
        oh = int(rng.integers(max(8, h // 8), max(9, int(h * .72))))
        box = (cx - ow // 2, cy - oh // 2, cx + ow // 2, cy + oh // 2)
        shape = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(shape)
        kind = int(rng.integers(3))
        if kind == 0:
            draw.rectangle(box, fill=255)
        elif kind == 1:
            draw.rounded_rectangle(box, radius=int(rng.integers(2, max(3, min(ow, oh) // 4))), fill=255)
        else:
            x0, y0, x1, y1 = box
            points = [(x0, y0 + int(oh*.15)), (x0 + int(ow*.7), y0),
                      (x1, y0 + int(oh*.12)), (x1 - int(ow*.08), y1),
                      (x0 + int(ow*.25), y1 - int(oh*.06))]
            draw.polygon(points, fill=255)
        occ = np.asarray(shape) > 0
        frac = float((occ & full_mask).sum() / full_mask.sum())
        if min_fraction <= frac <= max_fraction:
            return occ, frac
    raise ValueError("could not draw an occluder at the requested coverage")


def textured_patch(h, w, rng):
    """Generate a muted, locally varying opaque blocker texture."""
    base = rng.integers(35, 220, size=(1, 1, 3)).astype(np.float32)
    coarse = rng.normal(0, 22, size=(max(2, h // 12), max(2, w // 12), 1)).astype(np.float32)
    texture = Image.fromarray(np.uint8(np.clip(coarse[..., 0] + 128, 0, 255)))
    texture = np.asarray(texture.resize((w, h), Image.Resampling.BICUBIC), dtype=np.float32) - 128
    fine = rng.normal(0, 4, size=(h, w, 3))
    return np.uint8(np.clip(base + texture[..., None] + fine, 0, 255))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='Prepared COCOA household dataset directory')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--variants', type=int, default=2)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--min-occlusion', type=float, default=.15)
    p.add_argument('--max-occlusion', type=float, default=.50)
    a = p.parse_args()
    if a.output.exists():
        p.error(f'Output already exists; choose a new folder: {a.output}')
    if a.variants < 1 or not (0 <= a.min_occlusion < a.max_occlusion <= 1):
        p.error('Need variants >= 1 and 0 <= min-occlusion < max-occlusion <= 1')

    records = [json.loads(line) for line in (a.source / 'manifest.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    # Fail before creating an empty output folder. A clean RGB target must have
    # an exact zero occlusion fraction and a non-empty full-object mask.
    eligible = []
    empty_masks = 0
    missing_files = 0
    for record in records:
        if record.get('occluded_fraction') != 0:
            continue
        image_path = a.source / record['image']
        mask_path = a.source / record['complete_mask']
        if not image_path.is_file() or not mask_path.is_file():
            missing_files += 1
            continue
        with Image.open(image_path) as im:
            image_size = im.size
        with Image.open(mask_path) as im:
            full = np.asarray(im.convert('L')) >= 128
        if not full.any():
            empty_masks += 1
            continue
        if (image_size[1], image_size[0]) != full.shape:
            missing_files += 1
            continue
        eligible.append(record)
    if not eligible:
        p.error(
            'No usable fully visible RGB targets found. '
            f'Zero-occlusion records: {sum(r.get("occluded_fraction") == 0 for r in records)}; '
            f'empty full-object masks: {empty_masks}; missing/mismatched files: {missing_files}. '
            'Use a source dataset with genuinely complete RGB object crops and non-empty masks.'
        )
    records = eligible
    a.output.mkdir(parents=True)
    generated = []
    skipped = []
    for record in records:
        # Only clean, fully visible object crops can provide exact RGB targets.
        if record.get('occluded_fraction') != 0:
            continue
        folder = a.source / record['image']
        with Image.open(folder) as im:
            target = np.asarray(im.convert('RGB'))
        mask_path = a.source / record['complete_mask']
        with Image.open(mask_path) as im:
            full = np.asarray(im.convert('L')) >= 128
        if target.shape[:2] != full.shape or not full.any():
            skipped.append(record['sample_id'])
            continue
        for variant in range(a.variants):
            key = f"{record['sample_id']}-rgbocc-{variant:02d}"
            rng = np.random.default_rng(stable_seed(a.seed, key))
            try:
                occ, frac = make_occlusion(target, full, rng, a.min_occlusion, a.max_occlusion)
            except ValueError:
                skipped.append(key)
                continue
            blocker = textured_patch(*full.shape, rng)
            input_rgb = target.copy()
            input_rgb[occ] = blocker[occ]
            visible = full & ~occ
            out_dir = a.output / record['split'] / key
            out_dir.mkdir(parents=True)
            arrays = {
                'input_rgb.png': input_rgb,
                'target_rgb.png': target,
                'visible_mask.png': visible.astype(np.uint8) * 255,
                'complete_mask.png': full.astype(np.uint8) * 255,
                'occluder_mask.png': occ.astype(np.uint8) * 255,
            }
            for name, array in arrays.items():
                Image.fromarray(array).save(out_dir / name)
            generated.append({
                'sample_id': key, 'source_id': record['source_id'],
                'split': record['split'], 'category': record.get('category', ''),
                'occluded_fraction': frac,
                **{name.removesuffix('.png'): (out_dir / name).relative_to(a.output).as_posix() for name in arrays},
            })
    (a.output / 'manifest.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in generated), encoding='utf-8')
    split_counts = {s: sum(r['split'] == s for r in generated) for s in ('train', 'val', 'test')}
    categories = sorted({r['category'] for r in generated})
    summary = {
        'origin': 'COCOA fully visible household RGB crops with deterministic synthetic opaque occlusion',
        'source_dataset': str(a.source), 'seed': a.seed, 'variants_per_source': a.variants,
        'occlusion_fraction_range': [a.min_occlusion, a.max_occlusion],
        'samples': split_counts, 'total_samples': len(generated),
        'unique_source_photos_by_split': {s: len({r['source_id'] for r in generated if r['split'] == s}) for s in ('train', 'val', 'test')},
        'categories': categories, 'skipped_empty_or_failed': len(skipped),
        'limitations': ['Synthetic blockers do not capture all real occlusion patterns.', 'Targets preserve each source crop background and are not isolated transparent object renders.']
    }
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if skipped:
        (a.output / 'skipped.json').write_text(json.dumps(skipped, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
