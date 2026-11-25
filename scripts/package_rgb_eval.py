"""Package a deterministic held-out RGB completion set for Colab inference."""
from pathlib import Path
import argparse
import json
import zipfile


def choose_records(records, count):
    test = sorted((r for r in records if r['split'] == 'test'), key=lambda r: r['sample_id'])
    categories = {}
    for row in test:
        categories.setdefault(row.get('category', 'unknown').casefold(), []).append(row)
    for rows in categories.values():
        rows.sort(key=lambda r: (abs(float(r['occluded_fraction']) - 0.40), r['sample_id']))
    selected = []
    names = sorted(categories)
    while len(selected) < count and names:
        remaining = []
        for name in names:
            if categories[name] and len(selected) < count:
                selected.append(categories[name].pop(0))
            if categories[name]:
                remaining.append(name)
        names = remaining
    if len(selected) < count:
        raise ValueError(f'Only {len(selected)} test examples available; requested {count}.')
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=12)
    a = p.parse_args()
    if a.output.exists():
        p.error(f'Output already exists: {a.output}')
    records = [json.loads(line) for line in (a.source/'manifest.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    chosen = choose_records(records, a.count)
    archive_rows = []
    with zipfile.ZipFile(a.output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for row in chosen:
            sid = row['sample_id']
            item = {
                'sample_id': sid,
                'category': row.get('category', 'unknown'),
                'occluded_fraction': float(row['occluded_fraction']),
                'source_id': row['source_id'],
            }
            archive_rows.append(item)
            base = f'samples/{sid}/'
            for key, filename in [('image', 'image.png'), ('visible_mask', 'visible_mask.png'), ('complete_mask', 'complete_mask.png')]:
                z.write(a.source/row[key], base+filename)
        z.writestr('manifest.jsonl', ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in archive_rows))
        z.writestr('README.txt', 'Corrected, scene-held-out MP3D-Amodal crops. The RGB generator receives image.png and visible_mask.png. complete_mask.png is retained for qualitative silhouette reference; RGB output has no pixel-aligned full-color ground truth because the object is physically occluded in the source photo.\n')
    print(json.dumps({'archive': str(a.output), 'examples': len(chosen), 'categories': [r['category'] for r in chosen], 'ids': [r['sample_id'] for r in chosen], 'occlusion': [round(r['occluded_fraction'], 3) for r in chosen], 'bytes': a.output.stat().st_size}, indent=2))

if __name__ == '__main__':
    main()
