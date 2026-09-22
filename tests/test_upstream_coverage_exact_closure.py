from source_contract_helpers import source_sha256

from collections import Counter
from pathlib import Path
import unittest

import diffusers

from modiff.modular_workflow_contracts import PINNED_DIFFUSERS_REVISION
from modiff.upstream_coverage import _REVIEWED_PIPELINE_DECISIONS, _pipeline_coverage


ROOT = Path(__file__).resolve().parents[1]
PINNED_EXPORT_SHA256 = "f9536d3cd5f2f5df8992fa4b4b0fb53a407eea15744fcc288760624ce9111251"
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
    "Flux2Pipeline",
    "Flux2ModularPipeline",
    "LTX2InContextPipeline",
    "LTX2Pipeline",
    "Cosmos3OmniModularPipeline",
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
    "LTX2ModularPipeline",
    "LTXModularPipeline",
    "LuminaText2ImgPipeline",
    "Wan22Image2VideoModularPipeline",
    "Wan22ModularPipeline",
}


class UpstreamCoverageExactClosureTests(unittest.TestCase):
    def test_exact_pin_closes_with_the_reviewed_finite_partition(self):
        source = Path(diffusers.__file__).resolve().parent
        self.assertEqual(PINNED_DIFFUSERS_REVISION, "fbf49e7f35857f76bc57b177e26f12b03687c668")
        self.assertEqual(source_sha256(source / "__init__.py"), PINNED_EXPORT_SHA256)
        self.assertTrue(PROMOTED_PIPELINES.isdisjoint(_REVIEWED_PIPELINE_DECISIONS))

        version, items = _pipeline_coverage(ROOT, source)
        self.assertEqual(version, "0.41.0.dev0")
        self.assertEqual(len(items), 334)
        self.assertEqual(
            Counter(item["status"] for item in items),
            {
                "executable": 143,
                "equivalent": 11,
                "intentionally-excluded": 59,
                "research-blocked": 116,
                "contract-only": 5,
            },
        )

        by_name = {item["name"]: item for item in items}
        stable_audio_contracts = {
            "StableAudio3Pipeline": ("text_to_audio", ["prompt"]),
            "StableAudio3AudioToAudioPipeline": ("audio_to_audio", ["prompt", "sourceAudio"]),
            "StableAudio3InpaintPipeline": ("audio_inpaint", ["prompt", "sourceAudio"]),
        }
        for pipeline_class, (mode, required_inputs) in stable_audio_contracts.items():
            with self.subTest(stable_audio_3=pipeline_class):
                item = by_name[pipeline_class]
                self.assertEqual(item["status"], "research-blocked")
                self.assertEqual(item["reviewDecision"], "pinned-diffusers-stable-audio-3-source-triage")
                self.assertEqual(item["genericTaskContract"]["mode"], mode)
                self.assertEqual(item["genericTaskContract"]["requiredInputs"], required_inputs)
                self.assertEqual(item["genericTaskContract"]["outputMediaKinds"], ["audio"])
                self.assertFalse(item["exactExecutionSpecs"])
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
