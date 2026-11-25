# Multi-candidate RGB completion review

## Completed review

Pix2Gestalt samples a completion stochastically. The baseline saved one sample per
input, so a weak output could reflect one random draw. The runner accepts
`--candidates 1` through `--candidates 4` and saves every output. It also keeps
`completion.png` as an alias for the first candidate for compatibility.

## Protocol and interpretation

- Use four examples from the validation split, never the held-out test images.
- Compare four samples per image with one fixed seed per image.
- Record settings and peak memory, then review candidates side by side.
- Human selection is only for a qualitative demonstration. It is not an automatic
  score and does not show that the generator itself improved.
- Four candidates completed within the recorded Colab GPU memory budget.

## Run in Colab

The local `runs/candidate-review-inputs-v1.zip` contains four validation RGB/mask
pairs and the current runner scripts. Complete target masks are deliberately
omitted. Upload the ZIP to Colab and extract it into
`/content/amodal-completion`; then run:

```bash
python scripts/run_candidate_review.py
```

The batch runner processes all four examples, creates a contact sheet, and writes
`runs/candidate-review-v1-results.zip` for download.

The review is complete; its summary and contact sheet are in
`runs/candidate-review-v1/summary.json` and `runs/candidate-review-v1/contact_sheet.png`.
Candidate inspection helps select a qualitative example; it is not an automatic
ranking metric and does not show that Pix2Gestalt itself improved.
