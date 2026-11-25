"""Train a compact supervised RGB object-completion baseline on paired images."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal.rgb_model import RGBCompletionUNet, composite_known_pixels


class RGBPairs(Dataset):
    def __init__(self, root: Path, split: str, augment: bool = False):
        self.root = root
        self.augment = augment
        self.records = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines() if line.strip()]
        self.records = [r for r in self.records if r["split"] == split]
        if not self.records:
            raise ValueError(f"Empty RGB dataset split: {split}")

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def rgb(path: Path) -> torch.Tensor:
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr.transpose(2, 0, 1).copy())

    @staticmethod
    def mask(path: Path) -> torch.Tensor:
        with Image.open(path) as im:
            arr = np.asarray(im.convert("L"), dtype=np.uint8) >= 128
        return torch.from_numpy(arr[None].astype(np.float32))

    def __getitem__(self, index: int):
        record = self.records[index]
        image = self.rgb(self.root / record["input_rgb"])
        target = self.rgb(self.root / record["target_rgb"])
        visible = self.mask(self.root / record["visible_mask"])
        full = self.mask(self.root / record["complete_mask"])
        occluder = self.mask(self.root / record["occluder_mask"])
        if self.augment and random.random() < .5:
            image, target, visible, full, occluder = [x.flip(-1) for x in
                                                       (image, target, visible, full, occluder)]
        return image, visible, target, full, occluder, record["sample_id"]


def weighted_l1(pred: torch.Tensor, target: torch.Tensor, occluder: torch.Tensor) -> torch.Tensor:
    # Most weight goes to pixels that were actually hidden. A small global term
    # keeps the model from drifting on colors around the blocker boundary.
    weights = .05 + .95 * occluder
    return ((pred.float() - target.float()).abs() * weights).sum() / (weights.sum() * 3).clamp_min(1)


def region_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    region = mask.bool().expand_as(pred)
    if not region.any():
        return 0.0
    return float((pred - target).abs()[region].mean().item() / 2.0)  # normalized [-1,1] -> [0,1]


def evaluate(model, loader, device):
    model.eval()
    val_loss, val_occ = [], []
    with torch.no_grad():
        for image, visible, target, full, occ, _ in loader:
            image, visible, target, occ = (x.to(device, non_blocking=True) for x in
                                           (image, visible, target, occ))
            output = model(torch.cat((image, visible), dim=1))
            val_loss.append(float(weighted_l1(output, target, occ).item()))
            val_occ.extend(region_mae(output[i], target[i], occ[i]) for i in range(len(image)))
    return float(np.mean(val_loss)), float(np.mean(val_occ))


def save_preview(model, dataset, path: Path, device, max_items: int = 12):
    model.eval()
    rows = min(max_items, len(dataset))
    tile, label = 192, 24
    sheet = Image.new("RGB", (tile * 3, (tile + label) * rows), "white")
    draw = ImageDraw.Draw(sheet)
    with torch.no_grad():
        for i in range(rows):
            image, visible, target, full, occ, sample_id = dataset[i]
            output = model(torch.cat((image, visible), dim=0).unsqueeze(0).to(device))[0].cpu()
            composite = composite_known_pixels(output.unsqueeze(0), image.unsqueeze(0),
                                               occ.unsqueeze(0))[0]
            panels = [image, composite, target]
            y = i * (tile + label)
            for j, panel in enumerate(panels):
                arr = ((panel.clamp(-1, 1).permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)
                sheet.paste(Image.fromarray(arr).resize((tile, tile)), (j * tile, y + label))
                draw.text((j * tile + 3, y + 4), ["Input", "Completion", "Target"][j], fill="black")
            draw.text((tile * 2 + 4, y + label + 4), sample_id, fill="yellow",
                      stroke_width=1, stroke_fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/abo-rgb-local.json")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_root = (ROOT / config["dataset"]).resolve()
    run_dir = (ROOT / args.run if args.run else ROOT / config["run"]).resolve()
    if run_dir.exists():
        parser.error(f"Run directory exists; choose a new --run path: {run_dir}")
    epochs = args.epochs or int(config.get("epochs", 10))
    batch_size = args.batch_size or int(config.get("batch_size", 2))
    base = int(config.get("base_channels", 16))
    seed = int(config.get("seed", 2026))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    train = RGBPairs(data_root, "train", augment=True)
    val = RGBPairs(data_root, "val")
    test = RGBPairs(data_root, "test")
    loader_kwargs = {"batch_size": batch_size, "num_workers": 0,
                     "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(train, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test, shuffle=False, **loader_kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RGBCompletionUNet(base=base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 1e-3)),
                                  weight_decay=float(config.get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5,
                                                            patience=2, min_lr=1e-5)
    amp = device.type == "cuda" and bool(config.get("mixed_precision", True))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    run_dir.mkdir(parents=True)
    history = []
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    patience = int(config.get("early_stopping_patience", 5))
    print(json.dumps({"device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                      "train_pairs": len(train), "validation_pairs": len(val), "test_pairs": len(test),
                      "parameters": sum(p.numel() for p in model.parameters()), "epochs": epochs,
                      "batch_size": batch_size, "mixed_precision": amp}, indent=2), flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for image, visible, target, full, occ, _ in train_loader:
            image, visible, target, occ = (x.to(device, non_blocking=True) for x in
                                           (image, visible, target, occ))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                prediction = model(torch.cat((image, visible), dim=1))
            loss = weighted_l1(prediction, target, occ)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().item()))
        val_loss, val_occ_mae = evaluate(model, val_loader, device)
        scheduler.step(val_loss)
        row = {"epoch": epoch, "train_weighted_l1": float(np.mean(losses)),
               "val_weighted_l1": val_loss, "val_occluder_mae": val_occ_mae,
               "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_loss < best_val:
            best_val, best_epoch, stale = val_loss, epoch, 0
            torch.save({"model": model.state_dict(), "base_channels": base, "epoch": epoch,
                        "validation_weighted_l1": best_val, "config": config}, run_dir / "best.pt")
        else:
            stale += 1
        if stale >= patience:
            break

    (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    baseline_occ_mae, model_occ_mae, model_hidden_mae = [], [], []
    with torch.no_grad():
        for image, visible, target, full, occ, _ in test_loader:
            image, visible, target, full, occ = (x.to(device, non_blocking=True) for x in
                                                  (image, visible, target, full, occ))
            prediction = model(torch.cat((image, visible), dim=1))
            completed = composite_known_pixels(prediction, image, occ)
            for i in range(len(image)):
                baseline_occ_mae.append(region_mae(image[i], target[i], occ[i]))
                model_occ_mae.append(region_mae(completed[i], target[i], occ[i]))
                hidden_object = occ[i] * full[i]
                model_hidden_mae.append(region_mae(completed[i], target[i], hidden_object))
    results = {
        "experiment": "ABO synthetic-occlusion RGB completion U-Net",
        "best_epoch": best_epoch, "epochs_run": len(history),
        "best_validation_weighted_l1": best_val, "test_pairs": len(test),
        "test_occluder_rgb_mae_input_baseline": float(np.mean(baseline_occ_mae)),
        "test_occluder_rgb_mae_model": float(np.mean(model_occ_mae)),
        "test_hidden_object_rgb_mae_model": float(np.mean(model_hidden_mae)),
        "metric_note": "RGB MAE is normalized to [0,1]; test products are disjoint from training/validation products.",
        "limitations": ["Small ABO listing shard and heuristic white-background masks.",
                        "Targets are product catalog images on white backgrounds, not in-room scenes.",
                        "Synthetic blockers and held-out product split do not measure real-world occlusion generalization."],
    }
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    save_preview(model, test, run_dir / "test_preview.png", device)
    print("FINAL_METRICS", json.dumps(results, indent=2), flush=True)
    print("RUN_DIR", run_dir, flush=True)


if __name__ == "__main__":
    main()
