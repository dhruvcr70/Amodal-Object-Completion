"""Run the Pix2Gestalt RGB completion baseline on a fixed evaluation archive.

This script is intended for the isolated Linux environment in the Colab notebook.
It runs one sample at a time through run_baseline.py, saves every completion and
error, and creates a contact sheet for qualitative review. Ground-truth masks in
the archive are never passed to the generator.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image, ImageDraw, ImageFont


def safe_member(archive: zipfile.ZipFile, name: str) -> bytes:
    """Read a named regular archive member without extracting arbitrary paths."""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive path in manifest: {name}")
    return archive.read(name)


def overlay_visible(image: Image.Image, mask: Image.Image) -> Image.Image:
    base = image.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
    alpha = mask.convert("L").resize((256, 256), Image.Resampling.NEAREST)
    tint = Image.new("RGBA", base.size, (255, 178, 35, 0))
    tint.putalpha(alpha.point(lambda value: round(value * 0.45)))
    return Image.alpha_composite(base.convert("RGBA"), tint).convert("RGB")


def make_contact_sheet(records: list[dict], path: Path) -> None:
    tile_w, tile_h, label_h = 256, 256, 38
    gap, header_h = 12, 34
    rows = max(1, len(records))
    sheet = Image.new("RGB", (2 * tile_w + 3 * gap,
                               header_h + rows * (tile_h + label_h + gap) + gap),
                      (31, 36, 44))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((gap, 10), "INPUT + VISIBLE REGION", fill="white", font=font)
    draw.text((2 * gap + tile_w, 10), "PIX2GESTALT RGB COMPLETION", fill="white", font=font)
    for row, record in enumerate(records):
        y = header_h + gap + row * (tile_h + label_h + gap)
        for col, key in enumerate(("review_input", "completion")):
            x = gap + col * (tile_w + gap)
            if record.get(key) and Path(record[key]).is_file():
                tile = Image.open(record[key]).convert("RGB").resize(
                    (tile_w, tile_h), Image.Resampling.LANCZOS)
                sheet.paste(tile, (x, y))
            else:
                draw.rectangle((x, y, x + tile_w, y + tile_h), fill=(65, 70, 80))
                draw.text((x + 8, y + 12), "Inference failed", fill="white", font=font)
        note = (f"{record['sample_id']}  |  {record.get('category', '')}  |  "
                f"hidden {float(record.get('occluded_fraction', 0)):.0%}  |  {record['status']}")
        draw.text((gap, y + tile_h + 7), note[:115], fill=(220, 225, 235), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/content/amodal-completion"))
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000:
        parser.error("--steps must be between 1 and 1000")
    if not args.archive.is_file() or not args.checkpoint.is_file():
        parser.error("The evaluation archive and model checkpoint must both exist")

    run_script = args.root / "scripts/run_baseline.py"
    out_root = args.root / "runs/rgb-eval-v1"
    out_root.mkdir(parents=True, exist_ok=True)
    input_root = args.root / "inputs/rgb-eval-v1"
    input_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.archive) as archive:
        manifest = [json.loads(line) for line in safe_member(archive, "manifest.jsonl").decode().splitlines()
                    if line.strip()]
        records: list[dict] = []
        for entry in manifest:
            sample_id = str(entry["sample_id"])
            sample_in = input_root / sample_id
            sample_out = out_root / sample_id
            sample_in.mkdir(parents=True, exist_ok=True)
            sample_out.mkdir(parents=True, exist_ok=True)

            # Read only the RGB input and the visible-target mask. Ground-truth
            # complete/occluder masks remain untouched and are never model inputs.
            image_bytes = safe_member(archive, entry["image"])
            mask_bytes = safe_member(archive, entry["visible_mask"])
            (sample_in / "image.png").write_bytes(image_bytes)
            (sample_in / "visible_mask.png").write_bytes(mask_bytes)
            image = Image.open(sample_in / "image.png").convert("RGB")
            mask = Image.open(sample_in / "visible_mask.png").convert("L")
            review_input = sample_out / "input_visible_overlay.png"
            overlay_visible(image, mask).save(review_input)
            shutil.copy2(sample_in / "image.png", sample_out / "input.png")
            shutil.copy2(sample_in / "visible_mask.png", sample_out / "visible_mask.png")

            command = [sys.executable, str(run_script), "--root", str(args.root),
                       "--image", str(sample_in / "image.png"),
                       "--mask", str(sample_in / "visible_mask.png"),
                       "--checkpoint", str(args.checkpoint),
                       "--steps", str(args.steps), "--seed", str(args.seed)]
            started = time.perf_counter()
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            elapsed = time.perf_counter() - started
            record = {
                "sample_id": sample_id,
                "category": entry.get("category", ""),
                "occluded_fraction": entry.get("occluded_fraction", ""),
                "status": "completed" if completed.returncode == 0 else "failed",
                "wall_seconds": round(elapsed, 2),
                "review_input": str(review_input),
                "completion": "",
            }
            report = {
                "sample_id": sample_id,
                "category": record["category"],
                "occluded_fraction": record["occluded_fraction"],
                "status": record["status"],
                "wall_seconds": record["wall_seconds"],
                "settings": {"resolution": 256, "steps": args.steps,
                             "guidance_scale": 2.0, "seed": args.seed,
                             "candidates": 1},
                "stdout": completed.stdout[-12000:],
                "stderr": completed.stderr[-12000:],
            }
            if completed.returncode == 0:
                match = re.search(r"Saved run:\s*(.+)", completed.stdout)
                if match:
                    generated = Path(match.group(1).strip()) / "completion.png"
                    if generated.is_file():
                        output = sample_out / "completion.png"
                        shutil.copy2(generated, output)
                        record["completion"] = str(output)
                        report_path = generated.parent / "metrics.json"
                        if report_path.is_file():
                            report["model_metrics"] = json.loads(report_path.read_text())
                    else:
                        record["status"] = report["status"] = "failed"
                        report["stderr"] += "\nRunner returned success but completion.png was missing."
                else:
                    record["status"] = report["status"] = "failed"
                    report["stderr"] += "\nCould not locate the run output path in runner output."
            (sample_out / "metrics.json").write_text(json.dumps(report, indent=2))
            if record["status"] == "failed":
                (sample_out / "error.txt").write_text(report["stderr"] or report["stdout"])
            records.append(record)
            print(f"{record['status']}: {sample_id} ({record['wall_seconds']} s)", flush=True)

    fields = ["sample_id", "category", "occluded_fraction", "status", "wall_seconds",
              "review_input", "completion", "visible_preservation", "object_plausibility",
              "completion_plausibility", "boundary_artifacts", "review_notes"]
    with (out_root / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fields})
    make_contact_sheet(records, out_root / "contact_sheet.png")
    summary = {
        "archive": args.archive.name,
        "samples": len(records),
        "completed": sum(item["status"] == "completed" for item in records),
        "failed": sum(item["status"] != "completed" for item in records),
        "settings": {"resolution": 256, "steps": args.steps,
                     "guidance_scale": 2.0, "seed": args.seed, "candidates": 1},
        "quality_assessment": "RGB completions require qualitative review; no pixelwise RGB score is claimed.",
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    archive_path = shutil.make_archive(str(args.root / "runs/rgb-eval-v1"), "zip",
                                       root_dir=out_root)
    print(json.dumps(summary, indent=2))
    print("Contact sheet:", out_root / "contact_sheet.png")
    print("Results archive:", archive_path)


if __name__ == "__main__":
    main()
