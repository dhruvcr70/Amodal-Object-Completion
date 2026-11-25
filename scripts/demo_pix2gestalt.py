"""Interactive SAM-assisted Pix2Gestalt object completion demo.

This app creates a 2D scene preview and transparent object layer. Pix2Gestalt
generates a plausible object hypothesis; it cannot recover unseen pixels exactly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

_SAM_CACHE = {}
_PIX_CACHE = {}


def _mask_canvas(size: tuple[int, int]) -> dict:
    black = Image.new("RGB", size, "black")
    return {"background": black, "layers": [], "composite": black.copy()}


def _editor_array(editor_value, size: tuple[int, int]) -> np.ndarray:
    """Convert an ImageEditor value to a binary white-painted mask."""
    value = editor_value
    if isinstance(value, dict):
        value = value.get("composite")
        if value is None:
            value = editor_value.get("background")
    if value is None:
        return np.zeros((size[1], size[0]), dtype=bool)
    if isinstance(value, np.ndarray):
        arr = value
        if arr.ndim == 3:
            # White paint on a black canvas. ImageEditor composites may be RGBA.
            arr = arr[..., :3].max(axis=2)
        mask = arr >= 128
        if mask.shape != (size[1], size[0]):
            mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(
                size, Image.Resampling.NEAREST)) >= 128
        return mask
    image = value.convert("RGB")
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return np.asarray(image).max(axis=2) >= 128


def on_upload(image: Image.Image):
    if image is None:
        return None, None, [], [], [], [], "", False
    image = image.convert("RGB")
    return (_mask_canvas(image.size), _mask_canvas(image.size), [], [], [], [],
            "Uploaded scene. Choose which mask to prompt, then add SAM clicks or paint manually.", False)


def record_sam_click(points, labels, positive: bool, evt):
    xy = getattr(evt, "index", None)
    if not xy or len(xy) < 2:
        return points or [], labels or [], "Click directly on the uploaded scene to add a SAM prompt."
    pts = list(points or []) + [[float(xy[0]), float(xy[1])]]
    lbs = list(labels or []) + [1 if positive else 0]
    kind = "foreground" if positive else "background"
    return pts, lbs, f"Added {kind} prompt at ({int(xy[0])}, {int(xy[1])}); click Apply SAM prompts."


def route_sam_click(target_points, target_labels, blocker_points, blocker_labels,
                    positive: bool, selected_mask: str, evt):
    """Keep separate foreground/background point prompts for target and blocker."""
    xy = getattr(evt, "index", None)
    if not xy or len(xy) < 2:
        return (target_points or [], target_labels or [], blocker_points or [],
                blocker_labels or [], "Click directly on the scene to add a SAM prompt.")
    kind = "foreground" if positive else "background"
    point = [float(xy[0]), float(xy[1])]
    if selected_mask == "Obstruction":
        pts = list(blocker_points or []) + [point]
        lbs = list(blocker_labels or []) + [1 if positive else 0]
        message = f"Added obstruction {kind} prompt at ({int(xy[0])}, {int(xy[1])})."
        return target_points or [], target_labels or [], pts, lbs, message
    pts = list(target_points or []) + [point]
    lbs = list(target_labels or []) + [1 if positive else 0]
    message = f"Added target {kind} prompt at ({int(xy[0])}, {int(xy[1])})."
    return pts, lbs, blocker_points or [], blocker_labels or [], message


def apply_selected_sam_prompts(image, target_editor, blocker_editor,
                               target_points, target_labels, blocker_points,
                               blocker_labels, selected_mask, sam_was_used,
                               checkpoint):
    """Apply SAM to the currently selected mask without overwriting the other."""
    if selected_mask == "Obstruction":
        result = apply_sam_prompts(image, blocker_editor, blocker_points,
                                   blocker_labels, checkpoint)
        return (target_editor, result[0], target_points or [], target_labels or [],
                result[1], result[2], result[3], bool(sam_was_used or result[4]))
    result = apply_sam_prompts(image, target_editor, target_points,
                               target_labels, checkpoint)
    return (result[0], blocker_editor, result[1], result[2], blocker_points or [],
            blocker_labels or [], result[3], bool(sam_was_used or result[4]))


def apply_sam_prompts(image: Image.Image, editor_value, points, labels, checkpoint: str):
    if image is None:
        return editor_value, points or [], labels or [], "Upload a scene first.", False
    if not checkpoint or not Path(checkpoint).is_file():
        return editor_value, points or [], labels or [], (
            "SAM checkpoint not found. Paint the target mask manually, or set "
            "--sam-checkpoint to a local ViT-B checkpoint."), False
    pts = list(points or [])
    lbs = list(labels or [])
    if not pts:
        return editor_value, pts, lbs, "Click one or more points on the scene first.", False
    try:
        import torch
        from segment_anything import SamPredictor, sam_model_registry
        key = (str(Path(checkpoint).resolve()),)
        predictor = _SAM_CACHE.get(key)
        if predictor is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sam = sam_model_registry["vit_b"](checkpoint=checkpoint).to(device)
            predictor = SamPredictor(sam)
            _SAM_CACHE[key] = predictor
        # SAM predictor state is reset when a different uploaded image is used.
        rgb = np.asarray(image.convert("RGB"))
        signature = (rgb.shape, int(rgb[0, 0].sum()), int(rgb[-1, -1].sum()), hash(rgb.tobytes()))
        if getattr(predictor, "_codex_image_signature", None) != signature:
            predictor.set_image(rgb)
            predictor._codex_image_signature = signature
        with torch.inference_mode():
            masks, scores, _ = predictor.predict(
                point_coords=np.asarray(pts, dtype=np.float32),
                point_labels=np.asarray(lbs, dtype=np.int32),
                multimask_output=True)
        best = int(np.argmax(scores))
        mask = masks[best]
        out = Image.fromarray(mask.astype(np.uint8) * 255).convert("RGB")
        return {"background": Image.new("RGB", image.size, "black"),
                "layers": [out], "composite": out}, pts, lbs, (
            f"SAM selected a mask (score {float(scores[best]):.3f}). Add more positive/negative "
            "clicks or refine it with the brush; then click Run completion."), True
    except Exception as exc:
        return editor_value, points or [], labels or [], f"SAM failed: {exc}", False


def load_pix_model(root: Path, checkpoint: Path):
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise RuntimeError(f"Pix2Gestalt checkpoint is missing: {checkpoint}. This demo does not download the 15.5 GB weights automatically.")
    stat = checkpoint.stat()
    if stat.st_size == 0:
        raise RuntimeError(f"Pix2Gestalt checkpoint is empty: {checkpoint}")
    # Include file identity in the cache key so replacing a checkpoint in place
    # cannot silently keep serving the old model from GPU memory.
    key = (str(root.resolve()), str(checkpoint), stat.st_size, stat.st_mtime_ns)
    if key in _PIX_CACHE:
        return _PIX_CACHE[key]
    # Keep a single large diffusion model resident; leaving an older checkpoint
    # in the cache can exhaust a free T4 when the replacement is loaded.
    _PIX_CACHE.clear()
    repo = root / "third_party" / "pix2gestalt" / "pix2gestalt"
    if not repo.is_dir():
        raise RuntimeError(f"Pix2Gestalt source is missing: {repo}. Use the project's Colab setup or restore the source folder.")
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(root / "third_party" / "taming-transformers"))
    sys.path.insert(0, str(root / "third_party" / "CLIP"))
    os.chdir(repo)
    import torch
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    if not torch.cuda.is_available():
        raise RuntimeError("Pix2Gestalt needs a CUDA GPU for this configured run. Launch on Kaggle/Colab with a GPU runtime.")
    config = OmegaConf.load(repo / "configs" / "sd-finetune-pix2gestalt-c_concat-256.yaml")
    config.model.params.use_ema = False
    state_bundle = torch.load(str(checkpoint), map_location="cpu", mmap=True, weights_only=False)
    state = state_bundle["state_dict"]
    with torch.device("cpu"):
        model = instantiate_from_config(config.model)
    selected, ema_count = {}, 0
    for name in model.state_dict():
        key_name = "model_ema." + name[len("model."):].replace(".", "") if name.startswith("model.") else ""
        if key_name and key_name in state:
            selected[name] = state[key_name]
            ema_count += 1
        elif name in state:
            selected[name] = state[name]
    missing, _ = model.load_state_dict(selected, strict=False)
    if missing or not ema_count:
        raise RuntimeError(f"Checkpoint/model mismatch: {len(missing)} missing tensors; EMA tensors found: {ema_count}.")
    del state_bundle, state, selected
    model.eval().requires_grad_(False).to("cuda")
    from inference import run_pix2gestalt
    loaded = (model, run_pix2gestalt, torch)
    _PIX_CACHE[key] = loaded
    return loaded


def square_crop(image: Image.Image, visible: np.ndarray, blocker: np.ndarray, padding: float):
    # Center the model crop on the target, not the (possibly huge) wall/table
    # marked as its obstruction. The blocker will be clipped to this crop.
    ys, xs = np.where(visible)
    if not len(xs):
        raise ValueError("Paint a visible-target mask and an obstruction mask before running.")
    width, height = image.size
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    side = int(max(x1 - x0, y1 - y0) * (1 + 2 * padding))
    side = min(max(side, 32), max(width, height))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    left, top = cx - side // 2, cy - side // 2
    # Pad at image borders so model coordinates stay square and the scene crop is stable.
    right, bottom = left + side, top + side
    src = image.crop((max(0, left), max(0, top), min(width, right), min(height, bottom)))
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    px, py = max(0, -left), max(0, -top)
    canvas.paste(src, (px, py))
    def crop_mask(mask):
        cropped = Image.fromarray(mask.astype(np.uint8) * 255).crop(
            (max(0, left), max(0, top), min(width, right), min(height, bottom)))
        out = Image.new("L", (side, side), 0)
        out.paste(cropped, (px, py))
        return np.asarray(out) >= 128
    return canvas, crop_mask(visible), crop_mask(blocker), (left, top, side)


def run_completion(image: Image.Image, visible_editor, blocker_editor, candidates: int,
                   steps: int, seed: int, padding: float, root_text: str,
                   checkpoint_text: str, track_wandb: bool, sam_was_used: bool = False,
                   progress=None):
    if image is None:
        raise ValueError("Upload a scene image first.")
    scene = image.convert("RGB")
    visible = _editor_array(visible_editor, scene.size)
    blocker = _editor_array(blocker_editor, scene.size)
    if not visible.any() or not blocker.any():
        raise ValueError("Paint both masks: visible target in the first editor, obstruction in the second.")
    crop, crop_visible, crop_blocker, (left, top, side) = square_crop(scene, visible, blocker, padding)
    crop_visible &= ~crop_blocker
    if not crop_blocker.any():
        raise ValueError("The obstruction mask is outside the target crop. Increase crop context padding or mark the nearby blocker region.")
    if not crop_visible.any():
        raise ValueError("The obstruction mask covers the visible target mask entirely. Leave visible target pixels unmarked as obstruction.")
    model_rgb = np.asarray(crop.resize((256, 256), Image.Resampling.LANCZOS))
    model_mask = np.repeat((np.asarray(Image.fromarray(crop_visible.astype(np.uint8) * 255).resize(
        (256, 256), Image.Resampling.NEAREST)) >= 128)[..., None], 3, axis=2).astype(np.uint8) * 255
    root = Path(root_text).resolve() if root_text else ROOT
    checkpoint = Path(checkpoint_text).resolve() if checkpoint_text else root / "checkpoints" / "epoch=000005.ckpt"
    if progress:
        progress(0.08, desc="Loading pretrained Pix2Gestalt weights")
    model, infer, torch = load_pix_model(root, checkpoint)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        results = infer(model, "cuda", model_rgb, model_mask, scale=2.0,
                        n_samples=int(candidates), ddim_steps=int(steps), precision="autocast")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    from compose_rgb_results import estimate_foreground
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1_000_000_000:09d}"
    run_dir = root / "runs" / "gradio-pix2gestalt" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    scene_arr = np.asarray(scene).copy()
    visible_full = visible
    blocker_full = blocker
    candidate_images, scene_images, layer_images = [], [], []
    out_meta = {"model": "Pix2Gestalt pretrained EMA", "seed": int(seed), "steps": int(steps),
                "guidance_scale": 2.0, "candidates": len(results), "crop_xyxy_unclamped": [left, top, left + side, top + side],
                "scene_size": list(scene.size), "inference_seconds": elapsed,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "gpu": torch.cuda.get_device_name(), "precision": "CUDA autocast",
                "sam_used": bool(sam_was_used), "checkpoint": str(checkpoint),
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
                "limitations": "2D hypothesis/composite; not depth-aware and not exact hidden-pixel recovery."}
    if track_wandb:
        try:
            import wandb
            with wandb.init(project="amodal-object-completion", job_type="pix2gestalt-inference",
                            config={k: out_meta[k] for k in ("model", "seed", "steps", "guidance_scale", "candidates", "gpu", "precision")}):
                wandb.log({"inference_seconds": elapsed,
                           "peak_allocated_gib": out_meta["peak_allocated_gib"]})
            out_meta["wandb"] = "logged scalar runtime metrics; no images uploaded"
        except Exception as exc:
            out_meta["wandb"] = f"not logged: {exc}"
    # Pixel mapping from the square model crop back to scene coordinates.
    x0, y0, x1, y1 = max(0, left), max(0, top), min(scene.width, left + side), min(scene.height, top + side)
    cx0, cy0, cx1, cy1 = x0 - left, y0 - top, x1 - left, y1 - top
    vis_crop_full = visible_full[y0:y1, x0:x1]
    block_crop_full = blocker_full[y0:y1, x0:x1]
    original_crop = np.asarray(scene.crop((x0, y0, x1, y1)))
    for idx, result in enumerate(results, start=1):
        generated = Image.fromarray(np.asarray(result).astype(np.uint8)).convert("RGB").resize((side, side), Image.Resampling.LANCZOS)
        gen_arr = np.asarray(generated)[cy0:cy1, cx0:cx1]
        vis_small = np.asarray(Image.fromarray(crop_visible.astype(np.uint8) * 255).resize((256, 256), Image.Resampling.NEAREST)) >= 128
        raw = np.asarray(result).astype(np.uint8)
        pred_small = estimate_foreground(raw, vis_small, 12) | vis_small
        predicted = np.asarray(Image.fromarray(pred_small.astype(np.uint8) * 255).resize((side, side), Image.Resampling.NEAREST))[cy0:cy1, cx0:cx1] >= 128
        reveal = predicted & ~vis_crop_full & block_crop_full
        scene_out = scene_arr.copy()
        scene_out[y0:y1, x0:x1][reveal] = gen_arr[reveal]
        object_rgb = gen_arr.copy()
        object_rgb[vis_crop_full] = original_crop[vis_crop_full]
        alpha = predicted | vis_crop_full
        rgba = np.dstack([object_rgb, alpha.astype(np.uint8) * 255])
        full_layer = Image.new("RGBA", scene.size, (0, 0, 0, 0))
        full_layer.paste(Image.fromarray(rgba, "RGBA"), (x0, y0))
        completion_path = run_dir / f"candidate_{idx:02d}.png"
        Image.fromarray(np.asarray(result).astype(np.uint8)).save(completion_path)
        scene_path = run_dir / f"scene_completion_{idx:02d}.png"
        layer_path = run_dir / f"object_layer_{idx:02d}.png"
        Image.fromarray(scene_out).save(scene_path)
        full_layer.save(layer_path)
        candidate_images.append((str(completion_path), f"Candidate {idx}"))
        scene_images.append((str(scene_path), f"Scene {idx}"))
        layer_images.append((str(layer_path), f"Transparent layer {idx}"))
    scene.save(run_dir / "input_scene.png")
    Image.fromarray(visible.astype(np.uint8) * 255).save(run_dir / "visible_target_mask.png")
    Image.fromarray(blocker.astype(np.uint8) * 255).save(run_dir / "obstruction_mask.png")
    (run_dir / "metrics.json").write_text(json.dumps(out_meta, indent=2) + "\n", encoding="utf-8")
    archive = root / "runs" / "gradio-pix2gestalt" / f"{run_id}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in run_dir.iterdir():
            zf.write(file, file.name)
    report = (f"Completed {len(results)} candidate(s) on {out_meta['gpu']} in {elapsed:.1f}s; "
              f"peak allocated memory {out_meta['peak_allocated_gib']:.2f} GiB. "
              "Outputs are plausible visual hypotheses, not verified recovery of hidden pixels.")
    paths = {
        "candidates": [str(run_dir / f"candidate_{i:02d}.png") for i in range(1, len(results) + 1)],
        "scenes": [str(run_dir / f"scene_completion_{i:02d}.png") for i in range(1, len(results) + 1)],
        "layers": [str(run_dir / f"object_layer_{i:02d}.png") for i in range(1, len(results) + 1)],
    }
    return candidate_images, scene_images, layer_images, report, str(archive), paths


def select_candidate(candidate_choice, paths):
    """Return one chosen candidate, its scene preview and transparent PNG layer."""
    if not paths or not paths.get("candidates"):
        raise ValueError("Run completion first to create candidate outputs.")
    try:
        index = int(str(candidate_choice).split()[-1]) - 1
    except (TypeError, ValueError):
        index = int(candidate_choice) - 1
    if not 0 <= index < len(paths["candidates"]):
        raise ValueError(f"Choose a candidate from 1 to {len(paths['candidates'])}.")
    return (paths["candidates"][index], paths["scenes"][index],
            paths["layers"][index], paths["layers"][index],
            f"Selected Candidate {index + 1}. Download the transparent PNG layer below.")


def build_app(args):
    global gr
    import gradio as gr
    sam_checkpoint = str(args.sam_checkpoint or "")
    with gr.Blocks(title="Pix2Gestalt Amodal Completion") as demo:
        gr.Markdown("# Amodal Object Completion\nUpload a scene, mark the visible target and the object blocking it, then generate Pix2Gestalt hypotheses. **The scene preview is a 2D visualization; hidden details are inferred, not recovered exactly.**")
        target_point_state = gr.State([])
        target_label_state = gr.State([])
        blocker_point_state = gr.State([])
        blocker_label_state = gr.State([])
        sam_used_state = gr.State(False)
        with gr.Row():
            with gr.Column():
                source = gr.Image(label="Scene image (click to prompt SAM)", type="pil", interactive=True)
                prompt_target = gr.Radio(["Visible target", "Obstruction"], value="Visible target", label="SAM clicks apply to")
                positive = gr.Checkbox(value=True, label="Next click is inside the selected region (uncheck for outside)")
                sam_status = gr.Markdown("Choose Visible target or Obstruction, add positive/negative clicks, then apply them to that mask.")
                visible_editor = gr.ImageEditor(label="Visible target mask — paint target pixels white", type="pil", brush=gr.Brush(colors=["#ffffff"], default_color="#ffffff", default_size=24), eraser=gr.Eraser(default_size=24), height=420)
                blocker_editor = gr.ImageEditor(label="Obstruction mask — paint blocker pixels white", type="pil", brush=gr.Brush(colors=["#ffffff"], default_color="#ffffff", default_size=24), eraser=gr.Eraser(default_size=24), height=420)
                with gr.Row():
                    candidates = gr.Slider(1, 4, value=2, step=1, label="Stochastic candidates")
                    steps = gr.Slider(20, 100, value=50, step=10, label="DDIM steps")
                seed = gr.Number(value=42, precision=0, label="Seed")
                padding = gr.Slider(0.05, 0.8, value=0.25, step=0.05, label="Crop context padding")
                track_wandb = gr.Checkbox(value=False, label="Log scalar runtime metrics to Weights & Biases (no images)")
                with gr.Accordion("Model files", open=False):
                    root_text = gr.Textbox(value=str(ROOT), label="Project folder")
                    checkpoint_text = gr.Textbox(value=str(args.checkpoint or ROOT / "checkpoints" / "epoch=000005.ckpt"), label="Pix2Gestalt checkpoint")
                    gr.Markdown("SAM checkpoint (configured at launch): " + (sam_checkpoint or "not configured"))
                run_button = gr.Button("Run Pix2Gestalt completion", variant="primary")
            with gr.Column():
                status = gr.Markdown()
                candidates_out = gr.Gallery(label="Pix2Gestalt object candidates", columns=2, height=280)
                scenes_out = gr.Gallery(label="Full-scene 2D previews", columns=2, height=300)
                layers_out = gr.Gallery(label="Transparent object layers", columns=2, height=300)
                archive_out = gr.File(label="Download all outputs")
                candidate_paths = gr.State(None)
                candidate_choice = gr.Dropdown(label="Choose candidate", choices=["Candidate 1"], value="Candidate 1")
                choose_button = gr.Button("Select candidate for export")
                selected_status = gr.Markdown()
                selected_candidate_out = gr.Image(label="Selected object candidate", type="filepath")
                selected_scene_out = gr.Image(label="Selected full-scene preview", type="filepath")
                selected_layer_out = gr.Image(label="Selected transparent object layer", type="filepath")
                selected_layer_file = gr.File(label="Download selected transparent PNG")
        source.upload(on_upload, source, [visible_editor, blocker_editor,
                                          target_point_state, target_label_state,
                                          blocker_point_state, blocker_label_state,
                                          sam_status, sam_used_state])
        def on_scene_select(target_pts, target_lbs, blocker_pts, blocker_lbs,
                            is_positive, selected_mask, evt: gr.SelectData):
            return route_sam_click(target_pts, target_lbs, blocker_pts, blocker_lbs,
                                   is_positive, selected_mask, evt)
        source.select(on_scene_select,
                      [target_point_state, target_label_state, blocker_point_state,
                       blocker_label_state, positive, prompt_target],
                      [target_point_state, target_label_state, blocker_point_state,
                       blocker_label_state, sam_status])
        prompt_button = gr.Button("Apply SAM prompts to selected mask")
        prompt_button.click(
            lambda img, target, blocker, tpts, tlbs, bpts, blbs, selected, used:
                apply_selected_sam_prompts(img, target, blocker, tpts, tlbs,
                                           bpts, blbs, selected, used, sam_checkpoint),
            [source, visible_editor, blocker_editor, target_point_state,
             target_label_state, blocker_point_state, blocker_label_state,
             prompt_target, sam_used_state],
            [visible_editor, blocker_editor, target_point_state, target_label_state,
             blocker_point_state, blocker_label_state, sam_status, sam_used_state],
            show_progress="minimal")
        def run_and_update_choices(image, visible, blocker, candidate_count, ddim_steps,
                                   random_seed, crop_padding, project_root, model_checkpoint,
                                   wandb_enabled, used_sam, progress=gr.Progress()):
            result = run_completion(image, visible, blocker, candidate_count, ddim_steps,
                                    random_seed, crop_padding, project_root, model_checkpoint,
                                    wandb_enabled, used_sam, progress=progress)
            candidate_paths_value = result[5]
            choice_values = [f"Candidate {i}" for i in range(1, len(candidate_paths_value["candidates"]) + 1)]
            return (*result, gr.update(choices=choice_values, value=choice_values[0]))

        run_button.click(run_and_update_choices,
                         [source, visible_editor, blocker_editor, candidates, steps, seed, padding, root_text, checkpoint_text, track_wandb, sam_used_state],
                         [candidates_out, scenes_out, layers_out, status, archive_out, candidate_paths, candidate_choice], show_progress="full")
        choose_button.click(select_candidate, [candidate_choice, candidate_paths],
                            [selected_candidate_out, selected_scene_out, selected_layer_out,
                             selected_layer_file, selected_status])
        gr.Markdown("**Mask convention:** select Visible target, click inside the visible target (positive) and outside it (negative), then apply SAM. Switch to Obstruction and add a new set of clicks inside/outside the blocker; applying prompts updates only the selected mask. Both masks remain brush-editable. Keep the target's observed pixels out of the obstruction mask. For a clean experiment, compare outputs with a known unobstructed view.")
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sam-checkpoint", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link (off by default).")
    args = parser.parse_args()
    demo = build_app(args)
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
