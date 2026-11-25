"""Create the magenta-marked copy required by the two-image LaMa workflow."""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

from PIL import Image, ImageDraw, ImageTk


class ObstructionMarker:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Mark the obstruction — Amodal Object Completion")
        self.root.geometry("1180x820")
        self.source: Image.Image | None = None
        self.preview: ImageTk.PhotoImage | None = None
        self.mask: Image.Image | None = None
        self.strokes: list[tuple[int, list[tuple[int, int]]]] = []
        self.current: list[tuple[int, int]] | None = None
        self.last_xy: tuple[int, int] | None = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.brush = tk.IntVar(value=24)

        toolbar = ttk.Frame(root, padding=8)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Load original photo", command=self.load).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Undo stroke", command=self.undo).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Clear marks", command=self.clear).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Save magenta-marked copy", command=self.save).pack(side="left", padx=4)
        ttk.Label(toolbar, text="Brush size:").pack(side="left", padx=(18, 4))
        ttk.Scale(toolbar, from_=6, to=80, variable=self.brush, orient="horizontal", length=150).pack(side="left")
        self.status = ttk.Label(root, text="Load a photo, then paint over the obstruction only.", padding=(12, 4))
        self.status.pack(fill="x")
        self.canvas = tk.Canvas(root, background="#22252a", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=8)
        self.canvas.bind("<ButtonPress-1>", self.start_stroke)
        self.canvas.bind("<B1-Motion>", self.draw_stroke)
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke)
        self.canvas.bind("<Configure>", lambda _event: self.redraw())

    def load(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose the untouched source photo",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.source = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Could not open image", str(exc))
            return
        self.source_path = Path(path)
        self.mask = Image.new("L", self.source.size, 0)
        self.strokes.clear()
        self.status.configure(text=f"Loaded {self.source_path.name} ({self.source.width} × {self.source.height}). Paint only the obstruction.")
        self.redraw()

    def canvas_to_image(self, x: float, y: float) -> tuple[int, int] | None:
        if self.source is None:
            return None
        ix = int((x - self.offset_x) / self.scale)
        iy = int((y - self.offset_y) / self.scale)
        if 0 <= ix < self.source.width and 0 <= iy < self.source.height:
            return ix, iy
        return None

    def start_stroke(self, event: tk.Event) -> None:
        point = self.canvas_to_image(event.x, event.y)
        if point is None:
            return
        self.current = [point]
        self.last_xy = point

    def draw_stroke(self, event: tk.Event) -> None:
        point = self.canvas_to_image(event.x, event.y)
        if point is None or self.current is None or self.last_xy is None:
            return
        self.current.append(point)
        self.last_xy = point
        self.redraw()

    def end_stroke(self, _event: tk.Event) -> None:
        if self.current:
            self.strokes.append((int(self.brush.get()), self.current))
            self.current = None
            self.last_xy = None
            self.rebuild_mask()
            self.redraw()

    def rebuild_mask(self) -> None:
        if self.source is None:
            return
        self.mask = Image.new("L", self.source.size, 0)
        draw = ImageDraw.Draw(self.mask)
        for brush_px, points in self.strokes:
            # Brush control is in on-screen pixels; map it to source resolution.
            width = max(1, round(brush_px / self.scale))
            if len(points) == 1:
                x, y = points[0]
                r = max(1, width // 2)
                draw.ellipse((x-r, y-r, x+r, y+r), fill=255)
            else:
                draw.line(points, fill=255, width=width, joint="curve")
                r = max(1, width // 2)
                for x, y in (points[0], points[-1]):
                    draw.ellipse((x-r, y-r, x+r, y+r), fill=255)
        if self.current:
            brush_px = int(self.brush.get())
            width = max(1, round(brush_px / self.scale))
            if len(self.current) == 1:
                x, y = self.current[0]
                r = max(1, width // 2)
                draw.ellipse((x-r, y-r, x+r, y+r), fill=255)
            else:
                draw.line(self.current, fill=255, width=width, joint="curve")

    def redraw(self) -> None:
        if self.source is None:
            return
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        self.scale = min(width / self.source.width, height / self.source.height, 1.0)
        size = (max(1, round(self.source.width*self.scale)), max(1, round(self.source.height*self.scale)))
        image = self.source.resize(size, Image.Resampling.LANCZOS).convert("RGBA")
        self.offset_x = (width-size[0])//2
        self.offset_y = (height-size[1])//2
        self.rebuild_mask()
        if self.mask is not None:
            mask_preview = self.mask.resize(size, Image.Resampling.NEAREST)
            overlay = Image.new("RGBA", size, (255, 0, 255, 0))
            overlay.putalpha(mask_preview.point(lambda p: int(p*0.62)))
            image.alpha_composite(overlay)
        self.preview = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.preview)

    def undo(self) -> None:
        if self.strokes:
            self.strokes.pop()
            self.rebuild_mask()
            self.redraw()

    def clear(self) -> None:
        self.strokes.clear()
        self.current = None
        self.rebuild_mask()
        self.redraw()

    def save(self) -> None:
        if self.source is None or self.mask is None or self.mask.getbbox() is None:
            messagebox.showinfo("No obstruction marked", "Load a photo and paint over the obstruction first.")
            return
        suggested = self.source_path.with_name(self.source_path.stem + "_marked.png")
        path = filedialog.asksaveasfilename(
            title="Save the marked duplicate",
            initialfile=suggested.name,
            initialdir=str(suggested.parent),
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
        )
        if not path:
            return
        marked = self.source.copy()
        magenta = Image.new("RGB", self.source.size, (255, 0, 255))
        marked.paste(magenta, mask=self.mask)
        marked.save(path)
        self.status.configure(text=f"Saved {path}. Upload this and the untouched original together in the Colab notebook.")
        messagebox.showinfo("Marked copy saved", f"Saved marked duplicate:\n{path}\n\nKeep the original photo too; the notebook needs both files.")


if __name__ == "__main__":
    app = tk.Tk()
    ObstructionMarker(app)
    app.mainloop()
