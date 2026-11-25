"""Open a small desktop UI to paint masks and run RGB object completion."""
from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageTk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal.rgb_model import RGBCompletionUNet  # noqa: E402


class RGBCompletionDemo:
    PANEL_W, PANEL_H, PANEL_Y = 520, 520, 66
    LEFT_X, RIGHT_X = 12, 562

    def __init__(self, root: tk.Tk, checkpoint: Path) -> None:
        self.root = root
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RGBCompletionUNet(base=int(state["base_channels"])).to(self.device)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        del state
        self.image: Image.Image | None = None
        self.visible: Image.Image | None = None
        self.occluder: Image.Image | None = None
        self.result: Image.Image | None = None
        self.mode = tk.StringVar(value="visible")
        self.brush = tk.IntVar(value=20)
        self.status = tk.StringVar(value=f"Ready on {self._device_name()}. Load an image to begin.")
        self.box: tuple[int, int, int, int] | None = None
        self.scale = 1.0
        self.last_point: tuple[int, int] | None = None
        self.input_photo: ImageTk.PhotoImage | None = None
        self.output_photo: ImageTk.PhotoImage | None = None
        self.root.title("RGB Amodal Object Completion")
        self.root.minsize(1120, 740)
        self._build()

    def _device_name(self) -> str:
        return torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"

    def _build(self) -> None:
        bar = ttk.Frame(self.root, padding=(12, 10, 12, 5))
        bar.pack(fill="x")
        ttk.Button(bar, text="Load image", command=self.load_image).pack(side="left")
        ttk.Label(bar, text="Paint:").pack(side="left", padx=(14, 4))
        ttk.Radiobutton(bar, text="Visible object", variable=self.mode, value="visible").pack(side="left")
        ttk.Radiobutton(bar, text="Blocker to replace", variable=self.mode, value="occluder").pack(side="left")
        ttk.Label(bar, text="Brush").pack(side="left", padx=(12, 4))
        ttk.Scale(bar, from_=5, to=48, variable=self.brush, orient="horizontal", length=110).pack(side="left")
        ttk.Button(bar, text="Clear selected mask", command=self.clear_selected).pack(side="left", padx=8)
        ttk.Button(bar, text="Complete", command=self.complete).pack(side="left")
        ttk.Button(bar, text="Save result", command=self.save_result).pack(side="left", padx=6)
        ttk.Label(self.root,
                  text="Paint visible object pixels yellow, then blocker pixels blue. Right-click erases from the selected mask.",
                  padding=(14, 2, 14, 7)).pack(fill="x")
        self.canvas = tk.Canvas(self.root, width=1100, height=610, background="#20242b", highlightthickness=0)
        self.canvas.pack(padx=12, pady=3)
        self.canvas.create_text(self.LEFT_X + self.PANEL_W / 2, 35,
                                text="INPUT · paint two masks", fill="#e2e8f0", font=("Segoe UI", 11, "bold"))
        self.canvas.create_text(self.RIGHT_X + self.PANEL_W / 2, 35,
                                text="COMPLETED RGB IMAGE", fill="#e2e8f0", font=("Segoe UI", 11, "bold"))
        self.canvas.create_rectangle(self.LEFT_X, self.PANEL_Y, self.LEFT_X + self.PANEL_W,
                                     self.PANEL_Y + self.PANEL_H, fill="#343a43", outline="")
        self.canvas.create_rectangle(self.RIGHT_X, self.PANEL_Y, self.RIGHT_X + self.PANEL_W,
                                     self.PANEL_Y + self.PANEL_H, fill="#343a43", outline="")
        self.input_item = self.canvas.create_image(self.LEFT_X, self.PANEL_Y, anchor="nw")
        self.output_item = self.canvas.create_image(self.RIGHT_X, self.PANEL_Y, anchor="nw")
        self.canvas.bind("<ButtonPress-1>", lambda e: self._paint(e, False))
        self.canvas.bind("<B1-Motion>", lambda e: self._paint(e, False))
        self.canvas.bind("<ButtonRelease-1>", lambda _e: self._stop())
        self.canvas.bind("<ButtonPress-3>", lambda e: self._paint(e, True))
        self.canvas.bind("<B3-Motion>", lambda e: self._paint(e, True))
        self.canvas.bind("<ButtonRelease-3>", lambda _e: self._stop())
        ttk.Label(self.root, textvariable=self.status, padding=(14, 5, 14, 10)).pack(fill="x")

    def load_image(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp")])
        if not path:
            return
        try:
            with Image.open(path) as opened:
                self.image = opened.convert("RGB")
            self.image.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
            self.visible = Image.new("L", self.image.size, 0)
            self.occluder = Image.new("L", self.image.size, 0)
            self.result = None
            self._render_input()
            self._render_output(None)
            self.status.set(f"Loaded {Path(path).name}. Paint the visible object, then the blocker.")
        except Exception as exc:
            messagebox.showerror("Could not load image", str(exc))

    def _fit(self, image: Image.Image, panel_x: int):
        scale = min(self.PANEL_W / image.width, self.PANEL_H / image.height)
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        x = panel_x + (self.PANEL_W - size[0]) // 2
        y = self.PANEL_Y + (self.PANEL_H - size[1]) // 2
        return image.resize(size, Image.Resampling.LANCZOS), (x, y, *size), scale

    @staticmethod
    def _tint(base: Image.Image, mask: Image.Image, color: tuple[int, int, int], alpha: float) -> Image.Image:
        arr = np.asarray(base.convert("RGB"), dtype=np.float32).copy()
        selected = np.asarray(mask) >= 128
        arr[selected] = arr[selected] * (1 - alpha) + np.asarray(color, dtype=np.float32) * alpha
        return Image.fromarray(np.uint8(np.clip(arr, 0, 255)))

    def _render_input(self) -> None:
        if self.image is None:
            return
        shown = self._tint(self.image, self.visible, (255, 184, 30), .50)
        shown = self._tint(shown, self.occluder, (40, 165, 255), .55)
        shown, self.box, self.scale = self._fit(shown, self.LEFT_X)
        self.input_photo = ImageTk.PhotoImage(shown)
        self.canvas.itemconfigure(self.input_item, image=self.input_photo)
        self.canvas.coords(self.input_item, self.box[0], self.box[1])

    def _render_output(self, image: Image.Image | None) -> None:
        if image is None:
            self.output_photo = None
            self.canvas.itemconfigure(self.output_item, image="")
            return
        shown, box, _ = self._fit(image, self.RIGHT_X)
        self.output_photo = ImageTk.PhotoImage(shown)
        self.canvas.itemconfigure(self.output_item, image=self.output_photo)
        self.canvas.coords(self.output_item, box[0], box[1])

    def _point(self, event):
        if self.image is None or self.box is None:
            return None
        x0, y0, width, height = self.box
        if not (x0 <= event.x < x0 + width and y0 <= event.y < y0 + height):
            return None
        return (min(self.image.width - 1, int((event.x - x0) / self.scale)),
                min(self.image.height - 1, int((event.y - y0) / self.scale)))

    def _paint(self, event, erase: bool) -> None:
        if self.image is None:
            return
        point = self._point(event)
        if point is None:
            self.last_point = None
            return
        mask = self.visible if self.mode.get() == "visible" else self.occluder
        color = 0 if erase else 255
        width = max(1, round(self.brush.get() / self.scale))
        draw = ImageDraw.Draw(mask)
        if self.last_point:
            draw.line((*self.last_point, *point), fill=color, width=width)
        radius = width / 2
        draw.ellipse((point[0] - radius, point[1] - radius,
                      point[0] + radius, point[1] + radius), fill=color)
        self.last_point = point
        self.result = None
        self._render_input()
        self._render_output(None)

    def _stop(self) -> None:
        self.last_point = None

    def clear_selected(self) -> None:
        if self.image is None:
            return
        if self.mode.get() == "visible":
            self.visible = Image.new("L", self.image.size, 0)
        else:
            self.occluder = Image.new("L", self.image.size, 0)
        self.result = None
        self._render_input()
        self._render_output(None)

    def complete(self) -> None:
        if self.image is None:
            messagebox.showinfo("Load an image", "Choose an image crop containing one partly hidden object.")
            return
        visible = np.asarray(self.visible) >= 128
        occluder = np.asarray(self.occluder) >= 128
        if not visible.any() or not occluder.any():
            messagebox.showinfo("Paint both masks", "Paint the visible object in yellow and the blocker in blue first.")
            return
        if np.logical_and(visible, occluder).any():
            messagebox.showinfo("Masks overlap", "Erase the overlap so the visible-object and blocker masks are separate.")
            return
        w, h = self.image.size
        scale = min(256 / w, 256 / h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        left, top = (256 - nw) // 2, (256 - nh) // 2
        canvas = Image.new("RGB", (256, 256), "white")
        canvas.paste(self.image.resize((nw, nh), Image.Resampling.LANCZOS), (left, top))
        vis = Image.new("L", (256, 256), 0)
        vis.paste(self.visible.resize((nw, nh), Image.Resampling.NEAREST), (left, top))
        rgb = np.asarray(canvas, dtype=np.float32) / 127.5 - 1
        mask = (np.asarray(vis) >= 128).astype(np.float32)
        tensor = torch.from_numpy(np.concatenate((rgb, mask[..., None]), axis=2).transpose(2, 0, 1).copy())[None]
        try:
            with torch.inference_mode():
                pred = self.model(tensor.to(self.device))[0].float().cpu()
            pred = ((pred.clamp(-1, 1).permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)
            pred_image = Image.fromarray(pred).crop((left, top, left + nw, top + nh)).resize(
                (w, h), Image.Resampling.LANCZOS)
            result = self.image.copy()
            result.paste(pred_image, (0, 0), self.occluder)
            self.result = result
            self._render_output(result)
            self.status.set(f"Prediction complete on {self._device_name()}. Yellow = visible object; blue = replaced blocker.")
        except Exception as exc:
            messagebox.showerror("Prediction failed", str(exc))

    def save_result(self) -> None:
        if self.result is None:
            messagebox.showinfo("No result yet", "Run Complete before saving the image.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", initialfile="completed_object.png",
                                            filetypes=[("PNG image", "*.png")])
        if path:
            self.result.save(path)
            self.status.set(f"Saved completion to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/abo-rgb-unet-v1/best.pt")
    args = parser.parse_args()
    root = tk.Tk()
    try:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        RGBCompletionDemo(root, args.checkpoint)
    except Exception as exc:
        messagebox.showerror("Could not start RGB completion", str(exc))
        root.destroy()
        raise SystemExit(1) from exc
    root.mainloop()


if __name__ == "__main__":
    main()
