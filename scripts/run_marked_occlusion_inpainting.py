"""Fill user-marked obstruction pixels with LaMa inpainting.

The user supplies the source photo and an edited copy where the obstruction is
painted solid magenta (#ff00ff). White mask pixels are the region LaMa fills.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

MAGENTA = np.array([255, 0, 255], dtype=np.uint8)


def extract_obstruction_mask(source: Image.Image, marked: Image.Image) -> Image.Image:
    """Extract magenta pixels from the annotated copy and validate alignment."""
    source = source.convert("RGB")
    marked = marked.convert("RGB")
    if source.size != marked.size:
        raise ValueError(
            f"Source and marked copy must match exactly: {source.size} vs {marked.size}."
        )
    arr = np.asarray(marked, dtype=np.int16)
    # Tolerate antialiased/soft-edged magenta brush strokes while excluding
    # most naturally occurring colors: high red + blue, low green.
    magenta = (arr[..., 0] >= 170) & (arr[..., 2] >= 170) & (arr[..., 1] <= 120)
    changed = np.max(np.abs(arr - np.asarray(source, dtype=np.int16)), axis=2) >= 35
    binary = magenta & changed
    if not binary.any():
        raise ValueError(
            "No magenta obstruction marks were found. Paint the obstruction "
            "#ff00ff on the duplicate image and save it without resizing."
        )
    # Expand slightly to cover antialiased edges and thin seams at the blocker.
    mask = Image.fromarray((binary.astype(np.uint8) * 255), mode="L")
    return mask.filter(ImageFilter.MaxFilter(5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path, help="Untouched source photo")
    parser.add_argument("--marked", required=True, type=Path,
                        help="Same photo with obstruction painted bright magenta")
    parser.add_argument("--output", required=True, type=Path, help="Output PNG path")
    parser.add_argument("--max-side", type=int, default=2048,
                        help="Maximum inference dimension; the full-size image is preserved in the export (default: 2048)")
    parser.add_argument("--mask-dilation", type=int, default=41,
                        help="Odd mask expansion size in source pixels, to cover rough brush edges (default: 41)")
    args = parser.parse_args()
    if args.max_side < 256:
        parser.error("--max-side must be at least 256 pixels")
    if args.mask_dilation < 1 or args.mask_dilation % 2 == 0:
        parser.error("--mask-dilation must be a positive odd number")

    source = Image.open(args.image).convert("RGB")
    marked = Image.open(args.marked).convert("RGB")
    mask = extract_obstruction_mask(source, marked).filter(
        ImageFilter.MaxFilter(args.mask_dilation)
    )

    try:
        from simple_lama_inpainting import SimpleLama
    except ImportError as exc:
        raise SystemExit(
            "Install the inpainting dependency first: pip install simple-lama-inpainting"
        ) from exc

    # LaMa receives the original image and binary obstruction mask. Composite
    # the prediction only inside the mask to preserve all other source pixels.
    # Keep inference memory bounded on the laptop GPU, then map the generated
    # fill back to the original photo resolution.
    scale = min(1.0, args.max_side / max(source.size))
    if scale < 1.0:
        inference_size = (round(source.width * scale), round(source.height * scale))
        inference_image = source.resize(inference_size, Image.Resampling.LANCZOS)
        inference_mask = mask.resize(inference_size, Image.Resampling.NEAREST)
    else:
        inference_image, inference_mask = source, mask
    prediction = SimpleLama()(inference_image, inference_mask).convert("RGB")
    if prediction.size != source.size:
        prediction = prediction.resize(source.size, Image.Resampling.LANCZOS)
    result = Image.composite(prediction, source, mask)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    mask_path = args.output.with_name(args.output.stem + "_mask.png")
    mask.save(mask_path)
    print(f"Completed image: {args.output}")
    print(f"Extracted obstruction mask: {mask_path}")
    print("LaMa creates a plausible fill; it cannot guarantee the exact hidden pixels.")


if __name__ == "__main__":
    main()
