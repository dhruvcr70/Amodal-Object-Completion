"""Package a small validation-only set for Pix2Gestalt candidate review."""
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/mp3d-subset-v4-corrected-crops"
OUT_ROOT = ROOT / "runs/candidate-review-inputs-v1"
ARCHIVE = ROOT / "runs/candidate-review-inputs-v1.zip"
TARGET_HIDDEN_FRACTIONS = (0.20, 0.35, 0.50, 0.65)
EXCLUDED_CATEGORIES = {"ceiling", "wall", "floor", "column", "wall2", "doorframe"}


def main() -> None:
    records = [
        json.loads(line)
        for line in (DATA_ROOT / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validation = [row for row in records if row["split"] == "val"]
    if not validation:
        raise RuntimeError("No validation examples found in the corrected-crops manifest.")

    selected = []
    used_ids: set[str] = set()
    used_categories: set[str] = set()
    for target in TARGET_HIDDEN_FRACTIONS:
        eligible = [
            row for row in validation
            if row["sample_id"] not in used_ids
            and row.get("category") not in used_categories
            and row.get("category") not in EXCLUDED_CATEGORIES
        ]
        choice = min(
            eligible,
            key=lambda row: (abs(float(row["occluded_fraction"]) - target), row["sample_id"]),
        )
        selected.append(choice)
        used_ids.add(choice["sample_id"])
        used_categories.add(choice.get("category", ""))

    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for row in selected:
        sample_id = row["sample_id"]
        dest = OUT_ROOT / "samples" / sample_id
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("image", "visible_mask"):
            source = DATA_ROOT / row[name]
            (dest / f"{name}.png").write_bytes(source.read_bytes())
        manifest.append({
            "sample_id": sample_id,
            "category": row.get("category", ""),
            "split": "val",
            "occluded_fraction": float(row["occluded_fraction"]),
            "image": f"samples/{sample_id}/image.png",
            "visible_mask": f"samples/{sample_id}/visible_mask.png",
        })

    (OUT_ROOT / "manifest.json").write_text(json.dumps({
        "experiment": "Pix2Gestalt stochastic candidate review",
        "source_dataset": "MP3D-Amodal corrected crops",
        "source_split": "validation",
        "target_hidden_fractions": list(TARGET_HIDDEN_FRACTIONS),
        "ground_truth_complete_masks_included": False,
        "samples": manifest,
    }, indent=2), encoding="utf-8")
    (OUT_ROOT / "README.txt").write_text(
        "Validation-only inputs for a four-candidate Pix2Gestalt review. "
        "Only the RGB crop and visible-object mask are included; complete masks "
        "are deliberately excluded.\n",
        encoding="utf-8",
    )

    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT_ROOT.rglob("*")):
            if path.is_file():
                archive.write(path, Path("inputs/candidate-review-inputs-v1") / path.relative_to(OUT_ROOT))
        for name in ("run_baseline.py", "run_candidate_review.py"):
            path = ROOT / "scripts" / name
            archive.write(path, Path("scripts") / name)
    print(json.dumps({"archive": str(ARCHIVE), "bytes": ARCHIVE.stat().st_size,
                      "samples": manifest}, indent=2))


if __name__ == "__main__":
    main()
