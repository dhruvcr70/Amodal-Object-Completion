"""Make experimental scene previews from isolated Pix2Gestalt RGB outputs.

The generated crop is segmented with GrabCut, seeded by border-connected near-white
background and the supplied visible-object mask. Small disconnected regions are
removed. Source pixels inside the visible-object mask are preserved. This remains a
visualization aid: generated foreground masks and scene occlusion order are unverified.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def border_background(rgb: np.ndarray, white_threshold: int) -> np.ndarray:
    """Return near-white pixels connected to the image border."""
    candidate = np.all(rgb >= 255 - white_threshold, axis=2)
    height, width = candidate.shape
    background = np.zeros_like(candidate)
    queue: deque[tuple[int, int]] = deque()

    for x in range(width):
        if candidate[0, x]:
            background[0, x] = True
            queue.append((0, x))
        if candidate[height - 1, x] and not background[height - 1, x]:
            background[height - 1, x] = True
            queue.append((height - 1, x))
    for y in range(height):
        if candidate[y, 0] and not background[y, 0]:
            background[y, 0] = True
            queue.append((y, 0))
        if candidate[y, width - 1] and not background[y, width - 1]:
            background[y, width - 1] = True
            queue.append((y, width - 1))

    while queue:
        y, x = queue.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < height and 0 <= nx < width and candidate[ny, nx] and not background[ny, nx]:
                background[ny, nx] = True
                queue.append((ny, nx))
    return background


def estimate_foreground(rgb: np.ndarray, visible: np.ndarray, white_threshold: int) -> np.ndarray:
    """Use GrabCut with image-edge background and visible-object foreground seeds."""
    # Keep OpenCV optional for callers that provide an explicit completion mask.
    import cv2

    border_bg = border_background(rgb, white_threshold)
    near_white = np.all(rgb >= 255 - max(white_threshold, 24), axis=2)
    labels = np.full(near_white.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    labels[~near_white] = cv2.GC_PR_FGD
    labels[border_bg] = cv2.GC_BGD

    sure_fg = visible & ~border_bg
    if int(sure_fg.sum()) < 8:
        candidates = (~border_bg).astype(np.uint8)
        count, components, stats, _ = cv2.connectedComponentsWithStats(candidates, 8)
        if count > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            sure_fg = components == largest
    labels[sure_fg] = cv2.GC_FGD

    if not np.any(labels == cv2.GC_BGD) or not np.any(labels == cv2.GC_FGD):
        return ~border_bg
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(rgb.copy(), labels, None, bg_model, fg_model, 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return ~border_bg

    foreground = (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)
    count, components, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), 8)
    if count <= 1:
        return foreground
    min_area = max(32, int(foreground.size * 0.002))
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    if not keep.any():
        keep[1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))] = True
    return keep[components]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("runs/eval-v1"))
    parser.add_argument("--output", type=Path, default=Path("runs/eval-v1/composites-v1"))
    parser.add_argument("--white-threshold", type=int, default=12,
                        help="Allow border-background pixels with each channel >= 255 - threshold.")
    args = parser.parse_args()
    if not 0 <= args.white_threshold <= 254:
        parser.error("--white-threshold must be in [0, 254]")
    results = args.results.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    samples = sorted(path for path in results.iterdir()
                     if path.is_dir() and (path / "completion.png").is_file())
    if not samples:
        parser.error(f"No completed sample folders found under {results}")

    tile = 256
    label_h = 22
    gap = 8
    columns = 1
    rows = (len(samples) + columns - 1) // columns
    sheet = Image.new("RGB", (4 * tile + 5 * gap,
                               rows * (tile + label_h) + (rows + 1) * gap), "#20242b")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    report = {
        "experiment": "GrabCut foreground estimate and visible-pixel-preserving composite",
        "white_threshold": args.white_threshold,
        "occlusion_ordering": "not modeled; generated pixels can cover input occluders",
        "samples": [],
    }

    for index, sample in enumerate(samples):
        source = np.asarray(Image.open(sample / "input.png").convert("RGB").resize((tile, tile)))
        generated = np.asarray(Image.open(sample / "completion.png").convert("RGB").resize((tile, tile)))
        visible = np.asarray(Image.open(sample / "visible_mask.png").convert("L").resize(
            (tile, tile), Image.Resampling.NEAREST)) >= 128
        predicted_foreground = estimate_foreground(generated, visible, args.white_threshold)
        fill_hidden = predicted_foreground & ~visible
        composite = source.copy()
        composite[fill_hidden] = generated[fill_hidden]
        mask_image = np.where(predicted_foreground, 255, 0).astype(np.uint8)

        sample_out = output / sample.name
        sample_out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask_image).save(sample_out / "estimated_foreground.png")
        Image.fromarray(composite).save(sample_out / "composite_preview.png")

        panels = [Image.fromarray(source), Image.fromarray(generated),
                  Image.fromarray(mask_image).convert("RGB"), Image.fromarray(composite)]
        labels = ["Input", "Raw completion", "Estimated foreground", "Composite preview"]
        row, col = divmod(index, columns)
        for j, (panel, label) in enumerate(zip(panels, labels)):
            x = gap + j * (tile + gap)
            y = gap + row * (tile + label_h + gap)
            if index == 0:
                draw.text((x, y), label, fill="#d8dee9", font=font)
            sheet.paste(panel, (x, y + label_h))
        draw.text((gap,
                   gap + row * (tile + label_h + gap) + 2), sample.name,
                  fill="#d8dee9", font=font)
        report["samples"].append({
            "sample_id": sample.name,
            "foreground_fraction": round(float(predicted_foreground.mean()), 4),
            "added_pixel_fraction": round(float(fill_hidden.mean()), 4),
            "composite": f"{sample.name}/composite_preview.png",
            "estimated_foreground": f"{sample.name}/estimated_foreground.png",
        })

    sheet.save(output / "contact_sheet.png")
    (output / "experiment.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "README.txt").write_text(
        "Experimental previews only. Foreground is estimated with GrabCut using "
        "border-connected near-white pixels as background and the visible-object "
        "mask as foreground hints. It may still miss white object details or retain "
        "gray artifacts. Original pixels under the visible-object mask are preserved. "
        "The preview does not model whether hidden object pixels belong behind an "
        "occluder, so generated pixels may paint over the occluding object. Inspect "
        "before using; this is not a validated final output.\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "samples": len(samples),
                      "contact_sheet": str(output / "contact_sheet.png")}, indent=2))


if __name__ == "__main__":
    main()
