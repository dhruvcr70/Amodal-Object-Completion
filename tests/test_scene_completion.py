"""Model-independent regression tests for full-scene composition."""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


class SceneCompletionTests(unittest.TestCase):
    def test_reveals_only_predicted_hidden_pixels_and_preserves_rest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene = np.full((64, 72, 3), (100, 110, 120), dtype=np.uint8)
            crop = (16, 12, 48, 44)
            Image.fromarray(scene).save(root / "scene.png")

            completion = np.full((32, 32, 3), (245, 245, 245), dtype=np.uint8)
            completion[8:24, 8:24] = (220, 40, 30)
            Image.fromarray(completion).save(root / "completion.png")

            visible = np.zeros((32, 32), dtype=np.uint8)
            visible[8:24, 8:15] = 255
            Image.fromarray(visible).save(root / "visible.png")

            predicted = np.zeros((32, 32), dtype=np.uint8)
            predicted[8:24, 8:24] = 255
            Image.fromarray(predicted).save(root / "predicted.png")

            occluder = np.zeros((64, 72), dtype=np.uint8)
            occluder[12:44, 12:48] = 255
            Image.fromarray(occluder).save(root / "occluder.png")

            output = root / "output"
            completed_run = subprocess.run([
                sys.executable, str(ROOT / "scripts/compose_scene_completion.py"),
                "--scene", str(root / "scene.png"),
                "--completion", str(root / "completion.png"),
                "--visible-mask", str(root / "visible.png"),
                "--completion-mask", str(root / "predicted.png"),
                "--occluder-mask", str(root / "occluder.png"),
                "--crop", *map(str, crop), "--output-dir", str(output),
            ], check=False, capture_output=True, text=True)
            self.assertEqual(completed_run.returncode, 0,
                             completed_run.stdout + "\n" + completed_run.stderr)

            result = np.asarray(Image.open(output / "scene_completion.png").convert("RGB"))
            # A predicted hidden pixel inside the blocker receives generated RGB.
            self.assertTupleEqual(tuple(result[24, 36]), (220, 40, 30))
            # Visible target pixels remain exactly as photographed.
            self.assertTupleEqual(tuple(result[24, 25]), (100, 110, 120))
            # Blocker pixels outside the predicted object stay untouched.
            self.assertTupleEqual(tuple(result[18, 44]), (100, 110, 120))
            # Scene pixels outside the crop are byte-for-byte unchanged.
            np.testing.assert_array_equal(result[:12], scene[:12])
            with Image.open(output / "completed_object_layer.png") as object_layer:
                self.assertEqual(object_layer.size, (72, 64))
            self.assertTrue((output / "scene_masks_preview.png").is_file())
            self.assertTrue((output / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
