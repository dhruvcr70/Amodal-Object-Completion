"""Copy a dataset subset selected by exact object category names."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal.data import write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Source dataset containing manifest.jsonl")
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory; existing directories are not overwritten")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Optional exact category names to include (case-insensitive)")
    parser.add_argument("--min-occlusion", type=float, default=0.0,
                        help="Minimum hidden fraction measured from prepared masks")
    parser.add_argument("--max-occlusion", type=float, default=1.0,
                        help="Exclusive maximum hidden fraction measured from prepared masks")
    args = parser.parse_args()
    source = args.dataset.resolve()
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Output already exists: {output}")

    records = [json.loads(line) for line in (source / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if not 0.0 <= args.min_occlusion < args.max_occlusion <= 1.0:
        parser.error("Occlusion bounds must satisfy 0 <= min < max <= 1.")
    allowed = {category.casefold() for category in args.categories} if args.categories else None
    selected = [record for record in records
                if (allowed is None or record.get("category", "unknown").casefold() in allowed)
                and args.min_occlusion <= float(record.get("occluded_fraction", 0.0)) < args.max_occlusion]
    counts = {split: sum(record["split"] == split for record in selected)
              for split in ("train", "val", "test")}
    if not all(counts.values()):
        parser.error(f"Category filter must retain all splits; got {counts}")

    output.mkdir(parents=True)
    keys = ("image", "complete_mask", "visible_mask", "occluder_mask")
    for record in selected:
        copied = dict(record)
        folder = output / record["split"] / record["sample_id"]
        folder.mkdir(parents=True)
        for key in keys:
            original = source / record[key]
            destination = folder / original.name
            shutil.copy2(original, destination)
            copied[key] = destination.relative_to(output).as_posix()
        record.update(copied)

    source_summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    summary = write_manifest(output, selected, "category_filtered:" + str(source.name))
    summary.update({
        "source_dataset": str(source),
        "dataset": source_summary.get("dataset", source_summary.get("origin", source.name)),
        "category_filter_casefolded": (sorted(allowed) if allowed is not None
                                        else source_summary.get("category_filter_casefolded")),
        "occlusion_filter": [args.min_occlusion, args.max_occlusion],
        "categories_present": sorted({record.get("category", "unknown") for record in selected}),
        "photos": {split: len({record["source_id"] for record in selected if record["split"] == split})
                   for split in ("train", "val", "test")},
    })
    if any("group_id" in record for record in selected):
        summary["scenes"] = {
            split: sorted({record["group_id"] for record in selected
                           if record["split"] == split and "group_id" in record})
            for split in ("train", "val", "test")
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
