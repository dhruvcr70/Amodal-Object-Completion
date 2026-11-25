# Pix2Gestalt evaluation and tracking status — 2026-11-04

## What is verified

### Candidate outputs on natural indoor objects

The saved validation review contains four MP3D-Amodal objects across four
categories, at roughly 20%, 34%, 50%, and 60% occlusion. Pix2Gestalt generated
four candidates per object (16 outputs). Mean hidden-region silhouette IoU over
all candidates was **0.7014**. The visible-only baseline was **0.0** in the
hidden region. Choosing each object's best candidate using the ground-truth mask
gave **0.7705** mean hidden IoU; this is an oracle upper bound and cannot be used
as an inference-time selection score. The sample is too small for a broad
generalization claim, and silhouettes were extracted from generated RGB.

### Natural room photographs

The 12-image MP3D room evaluation scored mean hidden-region silhouette IoU
**0.2694**, full silhouette IoU **0.5509**, and visible-only full-mask IoU
**0.5637**. Pix2Gestalt did not beat that visible-only baseline on full-mask
IoU. These are real room photographs, but MP3D does not provide unobstructed
RGB references for the hidden regions; this check measures approximate shape,
not recovered appearance or exact scene reconstruction.

### Known unobstructed RGB references

The controlled ABO evaluation has 46 source-held-out test inputs with clean
unobstructed references. Mean hidden-region RGB MAE was **0.0670** versus
**0.2877** when leaving the corrupted input unchanged; hidden-region PSNR was
**20.06 dB** and SSIM was **0.6513**. The corresponding controlled full-scene
compositing evaluation also uses 46 exact clean scene references and reports
hidden-object RGB MAE **0.0668** versus **0.2877** unchanged, and all-blocker
region MAE **0.0737** versus **0.2738** unchanged. These are synthetic
occlusions of catalog objects placed into a room background, not naturally
occluded photographs.

## Interface and output-path verification

On the local Gradio page, I confirmed both mask editors expose Brush, Erase,
Undo, and layer controls; the candidate-count, DDIM-step, seed, and crop-padding
controls are present. I drew a temporary brush mark and used Undo; the canvas
returned to its prior state.

I also exercised the app's actual `run_completion` → `select_candidate` → export
path with a deterministic stub in place of the GPU model. The path produced two
candidate files, two scene previews, two RGBA object layers, and a ZIP containing
all outputs. Selecting Candidate 2 returned its matching image, scene, and
transparent PNG; its alpha channel contained both 0 and 255, and the marked
blocker region changed in the scene preview. This verifies the app wiring and
file formats, not Pix2Gestalt quality or the browser-generated Gradio callback.

Separately, saved real Pix2Gestalt inference already provides four candidates
for each of four natural indoor objects (16 images), with the mask evaluation
reported above. A fresh run through the Gradio browser, including its candidate
dropdown and download control, is still pending because the tested masks did
not yield candidates in the live app session.

## W&B (not required)

The app supports optional scalar-only logging, but no W&B account setup or cloud
sync is required for this project. No metrics or images were uploaded.

## Additional natural-photo pilot

`data/cocoa-natural-pilot-v1.zip` contains seven source-held-out natural photos
with synthetic masks and unobstructed references. It is intended for an
additional RGB check. No Pix2Gestalt predictions have been scored from this
pilot yet. The Colab upload cell is waiting for a file; the embedded Choose
Files control is disabled and no usable file chooser opened in this browser
session. The existing T4 session and 15.5 GB checkpoint remain available.

## Resume the seven-photo pilot

Upload `data/cocoa-natural-pilot-v1.zip` to the existing Colab runtime, run
Pix2Gestalt for four seeded candidates per sample, and score against the included
references with `scripts/score_cocoa_natural_pilot.py`.
