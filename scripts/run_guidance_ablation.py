"""Colab runner: compare Pix2Gestalt visible-mask vs predicted-amodal-mask input.

This is a zero-shot, out-of-distribution conditioning experiment. Pix2Gestalt
was trained with visible-object masks, so the second condition is exploratory.
"""
from __future__ import annotations

import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(root: Path):
    upstream = root / "third_party/pix2gestalt/pix2gestalt"
    sys.path.insert(0, str(upstream))
    os.chdir(upstream)
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config

    config = OmegaConf.load(upstream / "configs/sd-finetune-pix2gestalt-c_concat-256.yaml")
    config.model.params.use_ema = False
    checkpoint_path = root / "checkpoints/epoch=000005.ckpt"
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", mmap=True, weights_only=False)
    state = checkpoint["state_dict"]
    with torch.device("cpu"):
        model = instantiate_from_config(config.model)
    selected = {}
    ema_count = 0
    for name in model.state_dict():
        key = "model_ema." + name[len("model."):].replace(".", "") if name.startswith("model.") else ""
        if key and key in state:
            selected[name] = state[key]
            ema_count += 1
        elif name in state:
            selected[name] = state[name]
    missing, _ = model.load_state_dict(selected, strict=False)
    if missing or not ema_count:
        raise RuntimeError(f"Checkpoint load mismatch: missing={missing[:10]}, EMA tensors={ema_count}")
    del selected, state, checkpoint
    gc.collect()
    return model.eval().requires_grad_(False).to("cuda"), upstream, ema_count


def image_array(path: Path, mode: str) -> np.ndarray:
    im = Image.open(path).convert(mode)
    resample = Image.Resampling.LANCZOS if mode == "RGB" else Image.Resampling.NEAREST
    im = im.resize((256, 256), resample)
    arr = np.asarray(im)
    if mode == "L":
        arr = np.repeat(arr[..., None], 3, axis=2)
    return arr


def main() -> None:
    root = Path("/content/amodal-completion")
    input_root = root / "inputs/guidance-inputs-v1"
    manifest = json.loads((input_root / "experiment_manifest.json").read_text())
    if not torch.cuda.is_available():
        raise RuntimeError("Connect a Colab GPU runtime before running this ablation.")
    model, upstream, ema_count = make_model(root)
    from inference import run_pix2gestalt
    output_root = root / "runs/guidance-ablation-v1"
    output_root.mkdir(parents=True, exist_ok=True)
    report = {"experiment": "paired Pix2Gestalt mask-conditioning ablation",
              "warning": manifest["warning"], "gpu": torch.cuda.get_device_name(),
              "torch": torch.__version__, "steps": 50, "guidance_scale": 2.0,
              "seed_per_condition": 42, "resolution": [256, 256],
              "ema_tensors_loaded": ema_count, "samples": []}
    torch.cuda.reset_peak_memory_stats()
    for index, item in enumerate(manifest["samples"], 1):
        sid = item["sample_id"]
        sample = input_root / "samples" / sid
        image = image_array(sample / "image.png", "RGB")
        conditions = [
            ("visible-mask", sample / "visible_mask.png"),
            ("predicted-amodal-mask", sample / "predicted_amodal_mask.png"),
        ]
        sample_result = {"sample_id": sid, "category": item.get("category"), "conditions": {}}
        sample_dir = output_root / sid
        sample_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(sample_dir / "input.png")
        for name, mask_path in conditions:
            mask = image_array(mask_path, "L")
            seed_all(42)
            start = time.perf_counter()
            with torch.inference_mode():
                result = run_pix2gestalt(model, "cuda", image, mask, scale=2.0,
                                         n_samples=1, ddim_steps=50, precision="autocast")
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            out = sample_dir / name
            out.mkdir(exist_ok=True)
            Image.fromarray(mask[:, :, 0]).save(out / "conditioning_mask.png")
            Image.fromarray(result[0]).save(out / "completion.png")
            sample_result["conditions"][name] = {"seconds": round(elapsed, 3)}
        report["samples"].append(sample_result)
        print(f"[{index}/{len(manifest['samples'])}] {sid}: both conditions complete", flush=True)

    report["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    report["interpretation_limit"] = "Qualitative zero-shot comparison only; no claim of improvement or trained Pix2Gestalt integration."
    (output_root / "summary.json").write_text(json.dumps(report, indent=2))

    thumb, label_h, cols = 192, 25, 3
    rows = len(manifest["samples"])
    sheet = Image.new("RGB", (cols * thumb * 4, ((rows + cols - 1) // cols) * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, item in enumerate(manifest["samples"]):
        sid = item["sample_id"]
        sample_dir = output_root / sid
        x, y = (i % cols) * thumb * 4, (i // cols) * (thumb + label_h)
        draw.text((x, y), f"{sid} | {item.get('category', '')}", fill="black")
        paths = [sample_dir / "input.png",
                 sample_dir / "visible-mask/completion.png",
                 sample_dir / "predicted-amodal-mask/conditioning_mask.png",
                 sample_dir / "predicted-amodal-mask/completion.png"]
        for j, path in enumerate(paths):
            im = Image.open(path).convert("RGB").resize((thumb, thumb))
            sheet.paste(im, (x + j * thumb, y + label_h))
            draw.text((x + j * thumb + 4, y + label_h + 4),
                      ["Input", "Visible condition", "Predicted mask", "Predicted condition"][j],
                      fill="yellow", stroke_width=1, stroke_fill="black")
    sheet.save(output_root / "contact_sheet.png")
    import zipfile
    archive = root / "runs/guidance-ablation-v1-results.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in output_root.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(root))
    print(json.dumps({"summary": report, "archive": str(archive),
                      "archive_bytes": archive.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
