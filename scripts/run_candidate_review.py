"""Generate a small, paired multi-candidate Pix2Gestalt validation review."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    root = Path(os.environ.get("AMODAL_ROOT", "/content/amodal-completion"))
    input_root = root / "inputs/candidate-review-inputs-v1"
    manifest = json.loads((input_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("source_split") != "validation":
        raise ValueError("Candidate selection must only use the validation split.")
    if manifest.get("ground_truth_complete_masks_included"):
        raise ValueError("Review archive unexpectedly includes complete-mask targets.")

    python = root / ".venv/bin/python"
    checkpoint = root / "checkpoints/epoch=000005.ckpt"
    runner = root / "scripts/run_baseline.py"
    output_root = root / "runs/candidate-review-v1"
    output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("MPLBACKEND", None)
    env["PYTHONPATH"] = os.pathsep.join((str(root), str(root / "third_party/taming-transformers")))
    records = []

    for index, item in enumerate(manifest["samples"], start=1):
        sample_id = item["sample_id"]
        sample_dir = input_root / "samples" / sample_id
        cmd = [str(python), str(runner), "--root", str(root),
               "--image", str(sample_dir / "image.png"),
               "--mask", str(sample_dir / "visible_mask.png"),
               "--checkpoint", str(checkpoint), "--steps", "50", "--seed", "42",
               "--candidates", "4"]
        proc = subprocess.run(cmd, cwd=root, env=env, text=True, capture_output=True)
        run_path = next((Path(line.removeprefix("Saved run: ").strip())
                         for line in proc.stdout.splitlines()
                         if line.startswith("Saved run: ")), None)
        if proc.returncode or run_path is None:
            message = (proc.stderr or proc.stdout)[-6000:]
            (output_root / f"{sample_id}-error.txt").write_text(message, encoding="utf-8")
            records.append({"sample_id": sample_id, "status": "failed", "error": message[-1000:]})
            print(f"[{index}/{len(manifest['samples'])}] {sample_id}: failed", flush=True)
            continue

        dest = output_root / sample_id
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample_dir / "image.png", dest / "input.png")
        shutil.copy2(sample_dir / "visible_mask.png", dest / "visible_mask.png")
        for candidate in range(1, 5):
            candidate_path = run_path / f"candidate_{candidate:02d}.png"
            if not candidate_path.is_file():
                raise FileNotFoundError(candidate_path)
            shutil.copy2(candidate_path, dest / candidate_path.name)
        metrics = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
        records.append({"sample_id": sample_id, "category": item.get("category"),
                        "occluded_fraction": item.get("occluded_fraction"),
                        "status": "completed", "candidate_count": metrics["candidates"],
                        "inference_seconds": metrics["inference_seconds"],
                        "peak_allocated_gib": metrics["peak_allocated_gib"]})
        print(f"[{index}/{len(manifest['samples'])}] {sample_id}: saved four candidates", flush=True)

    summary = {"experiment": manifest["experiment"], "source_split": manifest["source_split"],
               "settings": {"resolution": 256, "steps": 50, "guidance_scale": 2.0,
                            "seed_per_sample": 42, "candidates_per_sample": 4},
               "interpretation": "Qualitative candidate-selection aid; not an automatic score.",
               "records": records}
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    thumb, label_h = 192, 28
    columns = 6
    sheet = Image.new("RGB", (columns * thumb,
                               len(manifest["samples"]) * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    labels = ["Input", "Visible mask", "Candidate 1", "Candidate 2", "Candidate 3", "Candidate 4"]
    for row_index, item in enumerate(manifest["samples"]):
        sample_id = item["sample_id"]
        dest = output_root / sample_id
        y = row_index * (thumb + label_h)
        draw.text((0, y), f"{sample_id} | {item.get('category')} | hidden {item['occluded_fraction']:.0%}", fill="black")
        paths = [dest / "input.png", dest / "visible_mask.png"] + [
            dest / f"candidate_{candidate:02d}.png" for candidate in range(1, 5)
        ]
        if not all(path.is_file() for path in paths):
            continue
        for col, (label, path) in enumerate(zip(labels, paths)):
            image = Image.open(path).convert("RGB").resize((thumb, thumb))
            x = col * thumb
            sheet.paste(image, (x, y + label_h))
            draw.text((x + 4, y + 3), label, fill="black")
    sheet.save(output_root / "contact_sheet.png")

    archive_path = root / "runs/candidate-review-v1-results.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(root))
    print(json.dumps({"summary": summary, "contact_sheet": str(output_root / "contact_sheet.png"),
                      "archive": str(archive_path), "archive_bytes": archive_path.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
