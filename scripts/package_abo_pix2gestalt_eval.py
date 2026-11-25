"""Package the sealed ABO test split for Pix2Gestalt evaluation in Colab."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/abo-rgb-pairs-v1"))
    parser.add_argument("--output", type=Path, default=Path("runs/pix2gestalt-abo-test-v1.zip"))
    parser.add_argument("--count", type=int, default=46,
                        help="Number of held-out test images to include (default: all 46).")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    manifest_path = args.source / "manifest.jsonl"
    rows = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("split") == "test":
                rows.append(row)
    rows.sort(key=lambda row: row["sample_id"])
    if args.count < 1 or args.count > len(rows):
        parser.error(f"--count must be between 1 and {len(rows)}")
    rows = rows[:args.count]
    archived = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        for row in rows:
            sid = row["sample_id"]
            entry = {
                "sample_id": sid,
                "category": row.get("category", "unknown"),
                "occluded_fraction": float(row["occluded_fraction"]),
                "source_id": row["source_id"],
                "image": f"samples/{sid}/image.png",
                "visible_mask": f"samples/{sid}/visible_mask.png",
                "target_rgb": f"evaluation-references/{sid}/target_rgb.png",
                "complete_mask": f"evaluation-references/{sid}/complete_mask.png",
                "occluder_mask": f"evaluation-references/{sid}/occluder_mask.png",
            }
            archived.append(entry)
            for field in ("input_rgb", "visible_mask", "target_rgb", "complete_mask", "occluder_mask"):
                rel = row[field]
                arcname = entry["image"] if field == "input_rgb" else entry[field]
                archive.write(args.source / rel, arcname)
        archive.writestr("manifest.jsonl", "".join(json.dumps(row) + "\n" for row in archived))
        archive.writestr(
            "README.txt",
            "Pix2Gestalt controlled RGB evaluation, ABO held-out product test split. "
            "Inference may use only samples/<id>/image.png and visible_mask.png. "
            "Files under evaluation-references are sealed ground truth for scoring after inference. "
            "Images are 256x256 catalog products with synthetic occluders; this is not a natural-scene benchmark.\n",
        )
    print(json.dumps({"archive": str(args.output.resolve()), "samples": len(rows),
                      "unique_products": len({row["source_id"] for row in rows}),
                      "bytes": args.output.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
