"""Prepare reproducible Pix2Gestalt scene inputs for the dining-table example."""
import argparse
import json
from pathlib import Path
import zipfile

from PIL import Image, ImageDraw


DEFAULT_SOURCE = Path(r"C:\Users\dell\AppData\Local\Temp\codex-clipboard-16e5a4b7-8797-4177-a6cc-5c0c7832b360.png")
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "runs/scene-table-occlusion-v1"
SIZE = 256

# Crops isolate one rear chair each while retaining the table that blocks it.
CASES = {
    "rear-left-chair": {
        "crop": (155, 280, 435, 560),
        # Visible upper back, the sliver left of the tabletop, and the exposed
        # seat beneath it. Table-covered pixels are deliberately left black.
        "polygons": [
            [(190, 351), (192, 335), (205, 324), (225, 318), (246, 323),
             (267, 335), (283, 351), (288, 360), (214, 359), (190, 361)],
            [(188, 361), (211, 360), (211, 425), (199, 434), (189, 414)],
            [(224, 452), (246, 446), (280, 445), (315, 451), (345, 463),
             (357, 478), (355, 490), (341, 498), (303, 498), (271, 490),
             (244, 477), (229, 463)],
        ],
    },
    "rear-right-chair": {
        "crop": (530, 280, 810, 560),
        # Visible top of the back, exposed right edge, and seat below the table.
        "polygons": [
            [(615, 357), (615, 340), (625, 327), (644, 320), (667, 318),
             (690, 322), (704, 334), (711, 349), (711, 357), (681, 357)],
            [(681, 358), (711, 358), (712, 382), (706, 407), (691, 429), (681, 431)],
            [(568, 453), (590, 449), (624, 451), (657, 458), (684, 469),
             (695, 482), (687, 493), (664, 500), (633, 499), (604, 491),
             (582, 478), (570, 465)],
        ],
    },
}

# Foreground table as seen in the supplied 894 x 894 image. The table top is one
# polygon; the pedestal is a second. The bowl and glass sit on the tabletop and
# are intentionally included in the obstruction region for this simple scene test.
TABLE_POLYGONS = [
    [(211, 361), (241, 370), (279, 365), (323, 358), (367, 352), (409, 350),
     (449, 351), (491, 355), (535, 359), (581, 362), (625, 369), (660, 379),
     (681, 390), (677, 402), (654, 413), (621, 423), (581, 431), (535, 438),
     (490, 444), (448, 446), (406, 443), (367, 438), (330, 431), (293, 420),
     (260, 407), (233, 393), (216, 379)],
    [(360, 431), (523, 431), (522, 661), (515, 676), (501, 689), (485, 697),
     (397, 697), (379, 689), (367, 678), (360, 662)],
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    inputs = output / "inputs"
    image = Image.open(source).convert("RGB")
    output.mkdir(parents=True, exist_ok=True)
    inputs.mkdir(parents=True, exist_ok=True)
    image.save(output / "scene.png")

    occluder = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(occluder)
    for polygon in TABLE_POLYGONS:
        draw.polygon(polygon, fill=255)
    occluder.save(output / "table_occluder_mask.png")

    previews = []
    for name, spec in CASES.items():
        x0, y0, x1, y1 = spec["crop"]
        crop = image.crop((x0, y0, x1, y1)).resize((SIZE, SIZE), Image.Resampling.LANCZOS)
        mask = Image.new("L", (x1 - x0, y1 - y0), 0)
        draw = ImageDraw.Draw(mask)
        for polygon in spec["polygons"]:
            draw.polygon([(x - x0, y - y0) for x, y in polygon], fill=255)
        mask = mask.resize((SIZE, SIZE), Image.Resampling.NEAREST)
        crop_path = inputs / f"{name}_image.png"
        mask_path = inputs / f"{name}_visible_mask.png"
        crop.save(crop_path)
        mask.save(mask_path)
        overlay = crop.convert("RGBA")
        tint = Image.new("RGBA", crop.size, (255, 30, 190, 0))
        tint.putalpha(mask.point(lambda p: round(p * 0.42)))
        previews.append((name, crop, mask, Image.alpha_composite(overlay, tint).convert("RGB")))

        # Each ZIP can be uploaded directly to the Colab Pix2Gestalt notebook.
        package = inputs / f"{name}_pix2gestalt_scene_input.zip"
        with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(output / "scene.png", "scene.png")
            archive.write(crop_path, "image.png")
            archive.write(mask_path, "visible_mask.png")
            archive.write(output / "table_occluder_mask.png", "occluder_mask.png")
            archive.writestr("crop.json", json.dumps({"crop_xyxy": list(spec["crop"])}, indent=2) + "\n")

    tile, label, gap = 256, 28, 10
    sheet = Image.new("RGB", (4 * tile + 5 * gap, len(previews) * (tile + label + gap) + gap), (34, 38, 46))
    draw = ImageDraw.Draw(sheet)
    for col, title in enumerate(("Scene crop", "Visible-object mask", "Mask overlay", "Scene + table mask")):
        draw.text((gap + col * (tile + gap), 5), title, fill="white")
    scene_review = image.copy().convert("RGBA")
    blue = Image.new("RGBA", image.size, (35, 130, 255, 0))
    blue.putalpha(occluder.point(lambda p: round(p * 0.4)))
    scene_review = Image.alpha_composite(scene_review, blue).convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
    for row, (name, crop, mask, overlay) in enumerate(previews):
        y = gap + label + row * (tile + label + gap)
        for col, panel in enumerate((crop, mask.convert("RGB"), overlay, scene_review)):
            sheet.paste(panel, (gap + col * (tile + gap), y))
        draw.text((gap, y + tile + 5), name, fill=(230, 232, 238))
    sheet.save(output / "input_mask_review.png")
    print(f"Prepared full scene, table mask, and {len(previews)} Pix2Gestalt ZIPs in {output}")
    print("Mask review:", output / "input_mask_review.png")


if __name__ == "__main__":
    main()
