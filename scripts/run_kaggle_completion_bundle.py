"""Run the bundled chair/panel completion inside the prepared Kaggle GPU runtime."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
INPUT_DIR = HERE / "inputs" / "chair-panel-test"
PROJECT_ROOT = Path("/kaggle/working/amodal-completion")

if not INPUT_DIR.is_dir():
    raise FileNotFoundError(f"Unpacked test inputs not found: {INPUT_DIR}")
if not (PROJECT_ROOT / "third_party/pix2gestalt/pix2gestalt").is_dir():
    raise FileNotFoundError(
        "Pix2Gestalt source is missing from /kaggle/working/amodal-completion. "
        "Run the notebook's project setup cells first."
    )

checkpoint_paths = list((PROJECT_ROOT / "checkpoints").rglob("*.ckpt"))
input_root = Path("/kaggle/input")
if input_root.is_dir():
    checkpoint_paths.extend(input_root.rglob("*.ckpt"))
checkpoint_candidates = sorted(
    checkpoint_paths,
    key=lambda path: path.stat().st_size,
    reverse=True,
)
if not checkpoint_candidates:
    raise FileNotFoundError(
        "No Pix2Gestalt .ckpt found under the project checkpoint folder or /kaggle/input. "
        "Attach or restore the existing checkpoint before running inference."
    )
checkpoint = checkpoint_candidates[0]

sys.path.insert(0, str(HERE / "scripts"))
import demo_pix2gestalt as demo

scene = Image.open(INPUT_DIR / "scene.png").convert("RGB")
visible = Image.open(INPUT_DIR / "visible_mask.png").convert("L")
obstruction = Image.open(INPUT_DIR / "occluder_mask.png").convert("L")
if visible.size != scene.size or obstruction.size != scene.size:
    raise ValueError("Scene and both masks must have identical dimensions.")

def editor(mask: Image.Image) -> dict:
    rgb = Image.fromarray((np.asarray(mask) >= 128).astype(np.uint8) * 255).convert("RGB")
    return {"background": Image.new("RGB", scene.size, "black"),
            "layers": [rgb], "composite": rgb}

results = demo.run_completion(
    scene,
    editor(visible),
    editor(obstruction),
    candidates=2,
    steps=30,
    seed=42,
    padding=0.25,
    root_text=str(PROJECT_ROOT),
    checkpoint_text=str(checkpoint),
    track_wandb=False,
    sam_was_used=False,
)

candidate_images, scene_images, layer_images, report, archive_path = results
print(report)
print("Checkpoint:", checkpoint)
print("Output archive:", archive_path)
try:
    from IPython.display import display
    print("Pix2Gestalt object candidates:")
    display(*candidate_images)
    print("Full-scene 2D previews:")
    display(*scene_images)
    print("Transparent object layers:")
    display(*layer_images)
except ImportError:
    pass
