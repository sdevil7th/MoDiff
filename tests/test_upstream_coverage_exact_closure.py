from collections import Counter
import hashlib
from pathlib import Path
import unittest

import diffusers

from modiff.modular_workflow_contracts import PINNED_DIFFUSERS_REVISION
from modiff.upstream_coverage import _REVIEWED_PIPELINE_DECISIONS, _pipeline_coverage


ROOT = Path(__file__).resolve().parents[1]
PINNED_EXPORT_SHA256 = "f53f106e6741413a30fb3d55a894cc3bcb958093c2a7d8ee89d2365af26ebf56"
PROMOTED_PIPELINES = {
    "AnimateDiffControlNetPipeline",
    "AnimateDiffPAGPipeline",
    "AnimateDiffVideoToVideoControlNetPipeline",
    "AnimateDiffVideoToVideoPipeline",
    "ChromaImg2ImgPipeline",
    "ChromaInpaintPipeline",
    "CogVideoXVideoToVideoPipeline",
    "FluxControlImg2ImgPipeline",
    "FluxControlInpaintPipeline",
    "LTX2Pipeline",
    "StableDiffusionControlNetImg2ImgPipeline",
    "StableDiffusionControlNetInpaintPipeline",
    "StableDiffusionControlNetPAGInpaintPipeline",
    "StableDiffusionControlNetPAGPipeline",
    "StableDiffusionXLControlNetImg2ImgPipeline",
    "StableDiffusionXLControlNetInpaintPipeline",
    "StableDiffusionXLControlNetPAGImg2ImgPipeline",
    "StableDiffusionXLControlNetPAGPipeline",
}
NEW_EQUIVALENT_PIPELINES = {
    "ErnieImageModularPipeline",
    "LTXModularPipeline",
    "LuminaText2ImgPipeline",
    "Wan22Image2VideoModularPipeline",
    "Wan22ModularPipeline",
}


class UpstreamCoverageExactClosureTests(unittest.TestCase):
    def test_exact_pin_closes_with_the_reviewed_finite_partition(self):
        source = Path(diffusers.__file__).resolve().parent
        self.assertEqual(PINNED_DIFFUSERS_REVISION, "90b4e34e79a86ec5e7f2437634fe95ecd2108796")
        self.assertEqual(hashlib.sha256((source / "__init__.py").read_bytes()).hexdigest(), PINNED_EXPORT_SHA256)
        self.assertTrue(PROMOTED_PIPELINES.isdisjoint(_REVIEWED_PIPELINE_DECISIONS))

        version, items = _pipeline_coverage(ROOT, source)
        self.assertEqual(version, "0.40.0.dev0")
        self.assertEqual(len(items), 327)
        self.assertEqual(
            Counter(item["status"] for item in items),
            {
                "executable": 116,
                "equivalent": 15,
                "intentionally-excluded": 56,
                "research-blocked": 120,
                "contract-only": 20,
            },
        )

        by_name = {item["name"]: item for item in items}
        for pipeline_class in PROMOTED_PIPELINES:
            with self.subTest(promoted=pipeline_class):
                self.assertEqual(by_name[pipeline_class]["status"], "executable")
                self.assertTrue(by_name[pipeline_class]["exactExecutionSpecs"])
                self.assertIsNone(by_name[pipeline_class]["reviewDecision"])
        for pipeline_class in NEW_EQUIVALENT_PIPELINES:
            with self.subTest(equivalent=pipeline_class):
                self.assertEqual(by_name[pipeline_class]["status"], "equivalent")
                self.assertTrue(by_name[pipeline_class]["equivalentTo"])
                self.assertIsNone(by_name[pipeline_class]["reviewDecision"])


if __name__ == "__main__":
    unittest.main()
