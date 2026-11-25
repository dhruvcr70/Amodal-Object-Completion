"""Launch a local desktop demo for visible-mask to complete-silhouette prediction."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal.model import CompletionUNet  # noqa: E402


class SilhouetteDemo:
    PANEL_WIDTH = 520
    PANEL_HEIGHT = 520
    PANEL_Y = 58
    LEFT_X = 12
    RIGHT_X = 562

    def __init__(self, root: tk.Tk, checkpoint: Path) -> None:
        self.root = root
        self.checkpoint = checkpoint.resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint}")

        state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self.size = int(state.get("resolution", 128))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CompletionUNet(state["config"]["base_channels"]).to(self.device)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        del state

        self.image: Image.Image | None = None
        self.visible_mask: Image.Image | None = None
        self.predicted_mask: Image.Image | None = None
        self.display_box: tuple[int, int, int, int] | None = None
        self.display_scale = 1.0
        self.last_point: tuple[int, int] | None = None
        self.input_photo: ImageTk.PhotoImage | None = None
        self.output_photo: ImageTk.PhotoImage | None = None
        self.input_image_item: int | None = None
        self.output_image_item: int | None = None
        self.brush_size = tk.IntVar(value=18)
        self.status = tk.StringVar(value=f"Model ready on {self._device_name()} — load a close-up crop to begin.")

        self.root.title("Amodal Silhouette Completion")
        self.root.minsize(1140, 750)
        self._build_ui()

    def _device_name(self) -> str:
        if self.device.type == "cuda":
            return torch.cuda.get_device_name()
        return "CPU"

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self.root, padding=(12, 10, 12, 4))
        toolbar.pack(fill="x")

        ttk.Button(toolbar, text="Load image", command=self.load_image).pack(side="left")
        ttk.Label(toolbar, text="Brush").pack(side="left", padx=(16, 5))
        ttk.Scale(toolbar, from_=6, to=40, variable=self.brush_size,
                  orient="horizontal", length=130).pack(side="left")
        ttk.Button(toolbar, text="Clear mask", command=self.clear_mask).pack(side="left", padx=8)
        ttk.Button(toolbar, text="Complete silhouette", command=self.complete).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Save mask", command=self.save_mask).pack(side="left", padx=4)

        instruction = ttk.Label(
            self.root,
            text="Use a close-up crop with one partly hidden object. Paint the visible object area in yellow; right-click erases.",
            padding=(14, 2, 14, 6),
        )
        instruction.pack(fill="x")

        self.canvas = tk.Canvas(self.root, width=1100, height=610, background="#20242b",
                                highlightthickness=0)
        self.canvas.pack(padx=12, pady=4)
        for x in (self.LEFT_X, self.RIGHT_X):
            self.canvas.create_rectangle(x, self.PANEL_Y, x + self.PANEL_WIDTH,
                                         self.PANEL_Y + self.PANEL_HEIGHT,
                                         fill="#343a43", outline="")
        self.canvas.create_text(self.LEFT_X + self.PANEL_WIDTH / 2,
                                self.PANEL_Y + self.PANEL_HEIGHT / 2,
                                text="Load an image", fill="#bac4d0")
        self.canvas.create_text(self.RIGHT_X + self.PANEL_WIDTH / 2,
                                self.PANEL_Y + self.PANEL_HEIGHT / 2,
                                text="Your result will appear here", fill="#bac4d0")
        self.canvas.create_text(self.LEFT_X + self.PANEL_WIDTH / 2, 28,
                                text="INPUT  ·  paint visible object", fill="#e2e8f0",
                                font=("Segoe UI", 11, "bold"))
        self.canvas.create_text(self.RIGHT_X + self.PANEL_WIDTH / 2, 28,
                                text="PREDICTED COMPLETE SILHOUETTE", fill="#e2e8f0",
                                font=("Segoe UI", 11, "bold"))
        self.input_image_item = self.canvas.create_image(self.LEFT_X, self.PANEL_Y, anchor="nw")
        self.output_image_item = self.canvas.create_image(self.RIGHT_X, self.PANEL_Y, anchor="nw")
        self.canvas.bind("<ButtonPress-1>", self._start_draw)
        self.canvas.bind("<B1-Motion>", self._continue_draw)
        self.canvas.bind("<ButtonRelease-1>", self._stop_draw)
        self.canvas.bind("<ButtonPress-3>", self._start_erase)
        self.canvas.bind("<B3-Motion>", self._continue_erase)
        self.canvas.bind("<ButtonRelease-3>", self._stop_draw)

        ttk.Label(self.root, textvariable=self.status, padding=(14, 5, 14, 10)).pack(fill="x")

    def load_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose an object crop",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with Image.open(path) as opened:
                self.image = opened.convert("RGB")
            self.image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            self.visible_mask = Image.new("L", self.image.size, 0)
            self.predicted_mask = None
            self._render_input()
            self._render_output(None)
            self.status.set(f"Loaded {Path(path).name} ({self.image.width} × {self.image.height}). Paint the visible object.")
        except Exception as error:
            messagebox.showerror("Could not load image", str(error))

    def _fit_image(self, image: Image.Image, panel_x: int) -> tuple[Image.Image, tuple[int, int, int, int], float]:
        scale = min(self.PANEL_WIDTH / image.width, self.PANEL_HEIGHT / image.height)
        width = max(1, round(image.width * scale))
        height = max(1, round(image.height * scale))
        x = panel_x + (self.PANEL_WIDTH - width) // 2
        y = self.PANEL_Y + (self.PANEL_HEIGHT - height) // 2
        return image.resize((width, height), Image.Resampling.LANCZOS), (x, y, width, height), scale

    @staticmethod
    def _overlay(image: Image.Image, mask: Image.Image, color: tuple[int, int, int], alpha: float) -> Image.Image:
        base = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
        selected = np.asarray(mask.convert("L")) >= 128
        tint = np.asarray(color, dtype=np.float32)
        base[selected] = base[selected] * (1.0 - alpha) + tint * alpha
        return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))

    def _render_input(self) -> None:
        if self.image is None or self.visible_mask is None or self.input_image_item is None:
            return
        composite = self._overlay(self.image, self.visible_mask, (255, 184, 30), 0.48)
        preview, self.display_box, self.display_scale = self._fit_image(composite, self.LEFT_X)
        self.input_photo = ImageTk.PhotoImage(preview)
        self.canvas.itemconfigure(self.input_image_item, image=self.input_photo)
        self.canvas.coords(self.input_image_item, self.display_box[0], self.display_box[1])

    def _render_output(self, mask: Image.Image | None) -> None:
        if self.output_image_item is None:
            return
        if self.image is None or mask is None:
            self.output_photo = None
            self.canvas.itemconfigure(self.output_image_item, image="")
            return
        hidden = np.asarray(mask.convert("L")) >= 128
        visible = (np.asarray(self.visible_mask.convert("L")) >= 128
                   if self.visible_mask is not None else np.zeros_like(hidden))
        # Keep observed pixels yellow and show the predicted complete silhouette in green.
        composite = self._overlay(self.image, Image.fromarray((hidden & ~visible).astype(np.uint8) * 255),
                                  (40, 220, 125), 0.55)
        composite = self._overlay(composite, Image.fromarray(visible.astype(np.uint8) * 255),
                                  (255, 184, 30), 0.48)
        preview, box, _ = self._fit_image(composite, self.RIGHT_X)
        self.output_photo = ImageTk.PhotoImage(preview)
        self.canvas.itemconfigure(self.output_image_item, image=self.output_photo)
        self.canvas.coords(self.output_image_item, box[0], box[1])

    def _source_point(self, event: tk.Event) -> tuple[int, int] | None:
        if self.image is None or self.display_box is None:
            return None
        x0, y0, width, height = self.display_box
        if not (x0 <= event.x < x0 + width and y0 <= event.y < y0 + height):
            return None
        x = min(self.image.width - 1, max(0, int((event.x - x0) / self.display_scale)))
        y = min(self.image.height - 1, max(0, int((event.y - y0) / self.display_scale)))
        return x, y

    def _paint(self, event: tk.Event, erase: bool) -> None:
        if self.visible_mask is None:
            return
        point = self._source_point(event)
        if point is None:
            self.last_point = None
            return
        width = max(1, round(int(self.brush_size.get()) / self.display_scale))
        color = 0 if erase else 255
        draw = ImageDraw.Draw(self.visible_mask)
        if self.last_point is not None:
            draw.line((self.last_point, point), fill=color, width=width)
        radius = width / 2
        draw.ellipse((point[0] - radius, point[1] - radius,
                      point[0] + radius, point[1] + radius), fill=color)
        self.last_point = point
        self.predicted_mask = None
        self._render_input()
        self._render_output(None)

    def _start_draw(self, event: tk.Event) -> None:
        self.last_point = None
        self._paint(event, erase=False)

    def _continue_draw(self, event: tk.Event) -> None:
        self._paint(event, erase=False)

    def _start_erase(self, event: tk.Event) -> None:
        self.last_point = None
        self._paint(event, erase=True)

    def _continue_erase(self, event: tk.Event) -> None:
        self._paint(event, erase=True)

    def _stop_draw(self, _event: tk.Event) -> None:
        self.last_point = None

    def clear_mask(self) -> None:
        if self.visible_mask is None:
            return
        self.visible_mask = Image.new("L", self.visible_mask.size, 0)
        self.predicted_mask = None
        self._render_input()
        self._render_output(None)
        self.status.set("Mask cleared. Paint the visible object region in yellow.")

    def complete(self) -> None:
        if self.image is None or self.visible_mask is None:
            messagebox.showinfo("Load an image first", "Choose a close-up image crop of one partly hidden object.")
            return
        visible = np.asarray(self.visible_mask.convert("L")) >= 128
        if not visible.any():
            messagebox.showinfo("Paint the visible object", "Draw over the visible part of the object before running completion.")
            return

        image_width, image_height = self.image.size
        side = max(image_width, image_height)
        left = (side - image_width) // 2
        top = (side - image_height) // 2
        square_image = Image.new("RGB", (side, side), (128, 128, 128))
        square_image.paste(self.image, (left, top))
        square_mask = Image.new("L", (side, side), 0)
        square_mask.paste(self.visible_mask, (left, top))

        rgb = np.asarray(square_image.resize((self.size, self.size), Image.Resampling.LANCZOS),
                         dtype=np.float32) / 255.0
        visible_small = np.asarray(square_mask.resize((self.size, self.size), Image.Resampling.NEAREST)) >= 128
        inputs = np.concatenate([rgb, visible_small[..., None].astype(np.float32)], axis=2)
        tensor = torch.from_numpy(inputs.transpose(2, 0, 1).copy())[None].to(self.device)
        try:
            with torch.inference_mode():
                probability = self.model(tensor).float().sigmoid()[0, 0].cpu().numpy()
            predicted_small = (probability > 0.5) | visible_small
            full_square = Image.fromarray(predicted_small.astype(np.uint8) * 255).resize(
                (side, side), Image.Resampling.NEAREST)
            full_crop = full_square.crop((left, top, left + image_width, top + image_height))
            full_at_original = np.asarray(full_crop.convert("L")) >= 128
            full_at_original |= visible
            self.predicted_mask = Image.fromarray(full_at_original.astype(np.uint8) * 255)
            self._render_output(self.predicted_mask)
            self.status.set(f"Completed on {self._device_name()}. Green is predicted hidden area; yellow is the observed mask.")
        except Exception as error:
            messagebox.showerror("Prediction failed", str(error))

    def save_mask(self) -> None:
        if self.predicted_mask is None:
            messagebox.showinfo("No prediction yet", "Run Complete silhouette before saving the mask.")
            return
        path = filedialog.asksaveasfilename(
            title="Save completed silhouette mask",
            defaultextension=".png",
            initialfile="completed_silhouette.png",
            filetypes=[("PNG mask", "*.png")],
        )
        if path:
            self.predicted_mask.save(path)
            self.status.set(f"Saved silhouette mask to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "runs/cocoa-household-finetune-v1/best.pt")
    args = parser.parse_args()
    root = tk.Tk()
    try:
        SilhouetteDemo(root, args.checkpoint)
    except Exception as error:
        messagebox.showerror("Could not start demo", str(error))
        root.destroy()
        raise SystemExit(1) from error
    root.mainloop()


if __name__ == "__main__":
    main()
