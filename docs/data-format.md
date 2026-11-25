# Preparing real complete-object examples

The procedural shapes validate engineering only. Real-object training requires
complete, unoccluded object images and accurate full-object masks. We synthetically
cover those objects to create paired examples whose hidden target is known.

Create a JSONL source manifest (one JSON object per line):

```json
{"source_id":"photo01-chair","group_id":"photo01","image":"images/photo01.png","complete_mask":"masks/photo01-chair.png"}
{"source_id":"photo01-table","group_id":"photo01","image":"images/photo01.png","complete_mask":"masks/photo01-table.png"}
{"source_id":"photo02-bag","group_id":"photo02","image":"images/photo02.png","complete_mask":"masks/photo02-bag.png"}
```

- Paths resolve relative to the manifest's folder.
- Images are RGB; masks mark the whole selected object white and background black.
- The mask and image must have identical dimensions. Prepare square crops without
  clipping the selected object for the current fixed-size resize pipeline.
- The same original photo, or related crops of it, must share `group_id`. Grouping
  keeps them in a single split. `source_id` must be unique per source object.
- Provide at least dozens of independent groups so all splits contain examples.
- Use datasets/images with appropriate permission and retain their provenance.

Run using the configured project interpreter:

```powershell
python scripts/prepare_objects.py --manifest data/my-sources.jsonl --output data/objects-v1 --size 128 --variants 3
```

Use the full configured interpreter path instead of bare `python` if the terminal
has not activated it. Copy `configs/shapes-local.json` into a new config, set
`dataset` to `data/objects-v1`, and choose a new `run` folder. Never reuse the toy
experiment's output directory or report toy metrics as real-object results.
