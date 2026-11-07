from source_contract_helpers import source_sha256

from collections import Counter
from pathlib import Path
import unittest

from modiff.modular_workflow_contracts import PINNED_DIFFUSERS_REVISION
from modiff.upstream_coverage import _REVIEWED_PIPELINE_DECISIONS, _pipeline_coverage, reviewed_diffusers_source


ROOT = Path(__file__).resolve().parents[1]
PINNED_EXPORT_SHA256 = "c2bc9d6c161fd3d3f4de657ecb6b58ce178fce211fda7f5e3641518d5ba6cd10"
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
    "ErnieImageModularPipeline",
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
    "LTX2ModularPipeline",
    "LTXModularPipeline",
    "Wan22Image2VideoModularPipeline",
    "Wan22ModularPipeline",
}


class UpstreamCoverageExactClosureTests(unittest.TestCase):
    def test_exact_pin_closes_with_the_reviewed_finite_partition(self):
        source = reviewed_diffusers_source()
        self.assertEqual(PINNED_DIFFUSERS_REVISION, "7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941")
        self.assertEqual(source_sha256(source / "__init__.py"), PINNED_EXPORT_SHA256)
        self.assertTrue(PROMOTED_PIPELINES.isdisjoint(_REVIEWED_PIPELINE_DECISIONS))

        version, items = _pipeline_coverage(ROOT, source)
        self.assertEqual(version, "0.41.0.dev0")
        self.assertEqual(len(items), 340)
        self.assertEqual(
            Counter(item["status"] for item in items),
            {
                "executable": 146,
                "equivalent": 8,
                "intentionally-excluded": 59,
                "research-blocked": 116,
                "contract-only": 5,
                "unreviewed": 6,
            },
        )

        by_name = {item["name"]: item for item in items}
        self.assertEqual({name for name, row in by_name.items() if row["status"] == "unreviewed"},
                         {
                "EchoModularPipeline", "Kandinsky6SRPipeline", "Kandinsky6TI2VAPipeline",
                "QwenImage21ModularPipeline", "QwenImage21TurboModularPipeline", "SanaWMPipeline",
            })
        # Executable here means an exact source adapter contract, not a
        # provider installation, model output, Auto or Gallery qualification.
        for name, variant in (
            ("Flux3ModularPipeline", "flux3-klein-base"),
            ("Flux3DistilledModularPipeline", "flux3-klein"),
        ):
            self.assertEqual(by_name[name]["status"], "executable")
            self.assertEqual(by_name[name]["exactExecutionSpecs"], [
                {"id": f"{variant}:official-provider:v1:edit-image:v1",
                 "modelType": name, "mode": "edit_image"},
                {"id": f"{variant}:official-provider:v1:text-to-image:v1",
                 "modelType": name, "mode": "text_to_image"},
            ])
            self.assertIsNone(by_name[name]["reviewDecision"])
        self.assertNotIn("Lumina2Text2ImgPipeline", by_name)
        self.assertNotIn("LuminaText2ImgPipeline", by_name)
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
