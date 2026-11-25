# Moving from generated shapes to COCOA

COCOA is a research dataset for amodal segmentation. Its annotations include
full-object polygons and separate visible-region masks; our loader uses both
instead of inventing an occluder mask for these examples. The annotation files
are distributed separately from the COCO photographs.

## Get the annotations

Use the annotation download linked by the [official COCOA API repository](https://github.com/yzhuuu/amodalAPI).
The download may require permission from the dataset authors. Extract the archive
and locate `COCO_amodal_train2014.json` (or the validation JSON). COCO photographs
are downloaded individually only when the matching image is not already in the
local image folder.

## Prepare a small local subset

From the project folder, run:

```powershell
& 'C:\Users\dell\Desktop\moba-rl\moba-rl\.venv\Scripts\python.exe' scripts\prepare_cocoa.py `
  --annotations 'C:\path\to\COCO_amodal_train2014.json' `
  --output data\cocoa-small `
  --max-images 240 `
  --max-objects 900
```

The default is intentionally bounded. Each selected object becomes a padded,
128×128 crop. The image ID is the split group, so objects from one photo never
leak across training, validation, and test. The output matches the existing
`CompletionDataset` manifest format and can be used by the existing trainer.

For local COCO image files instead of the on-demand downloads, add
`--image-dir 'C:\path\to\train2014'`.

## Train on the real subset

The project already includes `configs/cocoa-local.json` with `data/cocoa-small`
and a separate run folder. Train with:

```powershell
& 'C:\Users\dell\Desktop\moba-rl\moba-rl\.venv\Scripts\python.exe' scripts\train_silhouette.py --config configs\cocoa-local.json
```

COCOA is manually annotated, so its full masks represent human judgments about
occluded extent, not hidden ground truth measured by a 3D scanner. The current
U-Net still predicts silhouettes only; it does not paint missing RGB pixels.
