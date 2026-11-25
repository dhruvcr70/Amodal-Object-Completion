# Full-scene object completion

`scripts/compose_scene_completion.py` maps a Pix2Gestalt object-crop prediction
back into the original full-resolution photo. It supports any scene and target
object as long as the inference crop and masks are supplied.

## Inputs

- The original full-scene image.
- Pix2Gestalt's RGB completion for the target-object crop.
- A visible-target mask.
- An occluder mask identifying the foreground obstruction.
- The crop's `(x0, y0, x1, y1)` coordinates in the original image.
- Optionally, a predicted full-object mask aligned with the Pix2Gestalt output.

Masks may be full-scene-sized or aligned with the inference crop. If no
completion mask is supplied and the completion image has no transparency, the
script estimates the object foreground using GrabCut, the visible-target mask,
and near-white border background. For pale objects or non-white backgrounds,
provide an explicit completion mask for better control.

## Run

For the Colab notebook, upload a flat ZIP containing these files:

- `scene.png`: untouched, full-resolution scene.
- `image.png`: exact crop sent to Pix2Gestalt.
- `visible_mask.png`: same size as `image.png`, white on visible target pixels.
- `occluder_mask.png`: white on the foreground obstruction; full-scene or crop-sized.
- `crop.json`: `{"crop_xyxy": [x0, y0, x1, y1]}` in original-scene pixel coordinates.

The notebook saves these alongside Pix2Gestalt's completion and runs the
scene compositor automatically. From the project root, the command-line flow is:

```powershell
python scripts/run_baseline.py --image "data/crops/target-crop.png" --mask "data/masks/target-visible-crop.png" --scene "data/scene.jpg" --crop 100 80 700 680 --occluder-mask "data/masks/occluder.png" --checkpoint "checkpoints/pix2gestalt.ckpt"
python scripts/compose_scene_completion.py --run-dir "runs/20261101-120000" --output-dir "runs/scene-completion-v1"
```

Replace the example run folder with the timestamped folder printed by the
Pix2Gestalt runner. Save the marked obstruction mask with `--occluder-mask` so
the run is self-contained. If you already have a completion, use the explicit
`--scene`, `--completion`, `--visible-mask`, `--crop`, and `--occluder-mask`
options instead. The crop must exactly match the region used to create the
Pix2Gestalt input. The
current implementation assumes that crop was resized directly to the model
resolution; it does not undo padding, rotation, or perspective transforms.

## Outputs

- `scene_completion.png`: a full-resolution scene preview. It inserts predicted
  hidden-object pixels only where the completion overlaps the marked occluder;
  all other source pixels are preserved. This intentionally reveals the object
  through the obstruction, so part of the obstruction is replaced in the preview.
- `completed_object_layer.png`: the completed object placed at its scene
  coordinates on a transparent full-frame canvas. This can be composited by a
  downstream renderer at a chosen depth.
- `scene_masks_preview.png`: blue marks the occluder, orange the observed target,
  and green the predicted pixels inserted into the scene.
- `summary.json`: crop, mask method, pixel counts, outputs, and limitations.

This is an integration layer around an object-completion model, not a model that
reconstructs every scene pixel. The revealed preview is useful for seeing the
predicted missing part in context, but it is not a physically correct
depth-aware composite and cannot verify unseen texture or geometry. A true
behind-the-occluder render requires a scene-layer/depth representation; exact
hidden appearance requires additional views or 3D ground truth.

## Multiple objects in one scene

Run Pix2Gestalt once for each target object, then list those saved completions in
a JSON manifest. Paths inside the manifest are relative to the manifest file:

```json
{
  "objects": [
    {
      "name": "rear-left-chair",
      "completion": "left/completion.png",
      "visible_mask": "left/visible_mask.png",
      "occluder_mask": "table-mask.png",
      "crop": [155, 280, 435, 560],
      "depth": 1,
      "occluder_depth": 2
    },
    {
      "name": "rear-right-chair",
      "completion": "right/completion.png",
      "visible_mask": "right/visible_mask.png",
      "occluder_mask": "table-mask.png",
      "crop": [530, 280, 810, 560],
      "depth": 1,
      "occluder_depth": 2
    }
  ]
}
```

Compose them into a full-frame preview and a blocker-aware scene render with:

```powershell
python scripts/compose_scene_batch.py --scene "data/scene.jpg" --manifest "runs/scene-objects.json" --output-dir "runs/scene-completion-v1"
```

`depth` and `occluder_depth` are numeric front-to-back ordering supplied by the
user; larger values are closer to the camera. If omitted, manifest order is
used for objects and each occluder defaults in front of its target. The batch
script writes:

- `scene_completion.png` is the reveal preview. It deliberately paints predicted
  object pixels over the marked obstruction so you can inspect the completion.
- `scene_depth_composite.png` first fills the marked obstruction region in the
  background, then renders completed object and original obstruction layers
  back-to-front. An object declared behind the obstruction appears behind it.
  The default background fill uses OpenCV Telea inpainting; choose `--background-inpainter lama`
  for LaMa or `--background-inpainter none` to leave the source image unchanged.
- `scene_occlusion_removed.png` leaves the obstruction out, fills its area in
  the background, and places the completed objects into the scene. Use this when
  the desired result is a scene with the obstruction removed.

It also exports `layers/` and `scene_layer_manifest.json`: the original object
layers, extracted original obstruction layers, the source background with
transparent holes, and the inpainted background used for the composite. This
provides an editable 2D layer stack. The inpainted background is a plausible
estimate, not recovered ground truth. Depth is manual, not predicted from the
image, and the stack is not a 3D reconstruction.

The depth-aware render is an image-level approximation: each object has one
depth value, so it cannot handle parts of one object passing both in front of
and behind another. Per-pixel depth, camera geometry, shadows, and occlusion
boundaries are not estimated.
