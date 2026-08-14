import json
from pathlib import Path
import unittest

from modiff.studio_execution_specs import (
    ALLEGRO_DIFFUSERS_FILES,
    LATTE_DIFFUSERS_FILES,
    MOCHI_DIFFUSERS_FILES,
    studio_capability_definitions,
)


ROOT = Path(__file__).resolve().parents[1]


class VideoDownloadSelectionTests(unittest.TestCase):
    def test_allegro_selection_excludes_unsafe_duplicate_encoder(self):
        review = json.loads((ROOT / "data" / "allegro-artifact-review.json").read_text())
        selected = set(ALLEGRO_DIFFUSERS_FILES)
        capability = studio_capability_definitions()["AllegroPipeline"]

        self.assertEqual(capability["downloadFiles"], ALLEGRO_DIFFUSERS_FILES)
        self.assertEqual(len(selected), 18)
        self.assertTrue({item["path"] for item in review["weightFiles"]}.issubset(selected))
        self.assertTrue(
            {item["path"] for item in review["excludedUnsafeWeightFiles"]}.isdisjoint(selected)
        )
        self.assertFalse(any(path.endswith(".bin") for path in selected))

    def test_latte_selection_excludes_legacy_and_unreferenced_components(self):
        review = json.loads((ROOT / "data" / "latte-artifact-review.json").read_text())
        selected = set(LATTE_DIFFUSERS_FILES)
        capability = studio_capability_definitions()["LattePipeline"]

        self.assertEqual(capability["downloadFiles"], LATTE_DIFFUSERS_FILES)
        self.assertEqual(len(selected), 18)
        self.assertTrue({item["path"] for item in review["weightFiles"]}.issubset(selected))
        self.assertTrue({item["path"] for item in review["excludedArtifacts"]}.isdisjoint(selected))
        self.assertNotIn("vae_temporal_decoder/config.json", selected)
        self.assertFalse(any(path.endswith((".bin", ".pt", ".pth")) for path in selected))

    def test_mochi_selection_excludes_every_duplicate_partition(self):
        review = json.loads((ROOT / "data" / "mochi-artifact-review.json").read_text())
        selected = set(MOCHI_DIFFUSERS_FILES)
        capability = studio_capability_definitions()["MochiPipeline"]
        excluded = {
            item["path"]
            for partition in review["excludedArtifactPartitions"]
            for item in partition["files"]
        }

        self.assertEqual(capability["downloadFiles"], MOCHI_DIFFUSERS_FILES)
        self.assertEqual(len(selected), 21)
        self.assertTrue({item["path"] for item in review["weightFiles"]}.issubset(selected))
        self.assertTrue(excluded.isdisjoint(selected))
        self.assertIn("transformer/diffusion_pytorch_model.safetensors.index.bf16.json", selected)
        self.assertNotIn("assets/grid.gif", selected)


if __name__ == "__main__":
    unittest.main()
