# Visual review: Pix2Gestalt mask-conditioning ablation

Reviewed the 12 paired outputs in `runs/guidance-ablation-v1/`. Each pair uses the same source image and seed, with either the visible-object mask or the U-Net-predicted amodal mask. These notes describe visible differences only; they are not ground-truth RGB quality ratings. The dataset provides amodal silhouettes, not complete RGB object references.

| Example | Visible-mask vs predicted-mask observation |
| --- | --- |
| mp3d-1-1-0 — curtain | Outputs are very similar; predicted-mask output has a small change around the lower-left edge. |
| mp3d-4-4-0 — cabinet | Large change: a narrow vertical output becomes a much wider flared form. The wider form does not clearly match the cabinet in the source scene. |
| mp3d-20-20-0 — door | Predicted mask produces a broader, more complete-looking door panel than the narrow visible-mask output. This is a promising visual change, not a validated improvement. |
| mp3d-108-108-0 — theater seats | Outputs are nearly identical and look like a single sofa/chair rather than a row of theater seats. |
| mp3d-127-127-0 — kitchen center island | The prediction changes the small generated fragment, but the result remains incomplete and artifact-heavy. |
| mp3d-141-141-0 — window | Predicted-mask output contains substantially more of the window/curtain structure than the visible-mask output. This is one of the clearest apparent recoveries. |
| mp3d-192-192-0 — wardrobe | Predicted-mask output adds a hanging garment-like region to the door. The source scene contains fabric nearby, so this may be context-consistent or a mask-driven inclusion error; the available RGB target cannot resolve that. |
| mp3d-389-389-0 — shelf | Predicted-mask output adds a cup-like object over a sofa-like completion; this looks like an artifact and does not clearly match the shelf example. |
| mp3d-447-447-0 — dresser | Outputs are close; differences around the right side are small. Both retain surrounding bed-like content. |
| mp3d-466-466-0 — couch | The couch structure shifts slightly, especially the back cushions and lower silhouette; neither output establishes better recovery. |
| mp3d-928-928-0 — refrigerator | Outputs are very similar, with only small lower-edge changes. |
| mp3d-821-821-0 — fireplace | Predicted-mask output adds a large dark curved region beside the structure. It could be added geometry or a hallucinated neighboring object; the pair alone cannot tell. |

## Summary

The predicted mask clearly changes some completions, but effects are mixed: a broader door and fuller window are visually encouraging, while the cabinet, shelf, theater-seat, and fireplace examples show mismatch or artifact risks. Several pairs change very little. This small set comes from one held-out scene, and there are no complete RGB targets for objective generator scoring. Therefore we make no claim that predicted-amodal conditioning improves Pix2Gestalt.

The next evaluation measured the custom silhouette model on all 176 examples in the fixed test split. These samples all come from one held-out scene; this gives broader sample coverage within that scene, not evidence across multiple scenes. See `evaluation-protocol.md`. Future model choices should use validation data, and a true multi-scene generalization claim requires additional scenes.

