"""Upload scalar summaries from saved Pix2Gestalt evaluations to W&B.

No input images, masks, or per-sample records are uploaded. Set WANDB_API_KEY
through the normal W&B login flow (or the environment) before running.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


DEFAULT_EVALUATIONS = {
    "mp3d-natural-12": Path("runs/mp3d-eval-v1/evaluation/summary.json"),
    "candidate-review-4": Path("runs/candidate-review-v1/evaluation/summary.json"),
    "pix2gestalt-abo-rgb-46": Path("runs/pix2gestalt-abo-rgb-eval-v1/recomputed-v2/summary.json"),
    "controlled-scene-rgb-46": Path("runs/controlled-scene-eval-v2/summary.json"),
}


def scalar_metrics(payload: dict) -> dict[str, float]:
    metrics: dict[str, float] = {}

    def collect(prefix: str, value: object) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            metrics[prefix] = float(value)
        elif isinstance(value, dict):
            # Evaluation summaries store aggregate scores as {mean, std, ...}.
            for child, item in value.items():
                if child == "mean" or isinstance(item, (int, float, dict)):
                    collect(f"{prefix}/{child}", item)

    for key, value in payload.items():
        if key in {"metrics_macro_mean_and_std", "aggregate_mean_and_std"}:
            collect(key, value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[key] = float(value)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=os.getenv("WANDB_PROJECT", "amodal-object-completion"))
    parser.add_argument("--entity", default=os.getenv("WANDB_ENTITY"))
    parser.add_argument("--dry-run", action="store_true", help="Print what would be logged without contacting W&B.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    evaluations = [(name, root / path) for name, path in DEFAULT_EVALUATIONS.items()]
    selected = [(name, path, json.loads(path.read_text(encoding="utf-8")))
                for name, path in evaluations if path.is_file()]
    if not selected:
        raise SystemExit("No evaluation summary files were found.")

    if args.dry_run:
        for name, path, payload in selected:
            print(f"{name}: {path}")
            print(json.dumps(scalar_metrics(payload), indent=2))
        return

    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("Install requirements-demo.txt in the active environment first.") from exc

    if not os.getenv("WANDB_API_KEY") and not Path.home().joinpath(".netrc").is_file():
        raise SystemExit("W&B is not authenticated. Run `wandb login` locally, then retry; do not paste an API key into chat.")

    with wandb.init(project=args.project, entity=args.entity, job_type="evaluation-summary-sync",
                    config={"source": "saved project evaluation summaries", "uploads_images": False}) as run:
        for name, path, payload in selected:
            metrics = scalar_metrics(payload)
            run.log({f"evaluation/{name}/{key}": value for key, value in metrics.items()})
            run.summary[f"evaluation/{name}/samples"] = payload.get(
                "samples_scored", payload.get("samples", payload.get(
                    "samples_generated", payload.get("samples_in_archive", payload.get("scenes")))))
            run.summary[f"evaluation/{name}/source_file"] = str(path.relative_to(root))
        run.summary["evaluation/uploaded_image_data"] = False
    print(f"Synced {len(selected)} scalar evaluation summaries to {args.project}.")


if __name__ == "__main__":
    main()
