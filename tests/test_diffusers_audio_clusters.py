"""ACE-Step composites use the existing audio recipes, not a fake Modular tree."""
import unittest

from modiff.huggingface_diffusers_clusters import reviewed_diffusers_cluster_catalog
from modiff.studio_execution_specs import studio_execution_spec_for_pair


class AudioClusterContracts(unittest.TestCase):
    def test_all_four_ace_modes_have_exact_non_modular_composites(self):
        catalog = reviewed_diffusers_cluster_catalog("2f7e0154a9db246e95c9ede43edba7db5b130805")
        definitions = [d for d in catalog["definitions"] if d["pipelineClass"] == "AceStepAudioPipeline"]
        self.assertEqual(
            {d["workflowId"] for d in definitions},
            {"text_to_audio", "audio_variation", "audio_continuation", "audio_repaint"},
        )
        for definition in definitions:
            with self.subTest(mode=definition["workflowId"]):
                mode = definition["workflowId"]
                spec = studio_execution_spec_for_pair("AceStepAudioPipeline", mode)
                admission = definition["executionAdmissions"][0]
                self.assertEqual(definition["definitionKind"], "studio_execution_composite")
                self.assertEqual(definition["blocksClass"], "AceStepPipeline")
                self.assertEqual(admission["studioExecutionSpec"]["contentHash"], spec["contentHash"])
                self.assertEqual(admission["sealedBindingValues"]["pipelineClass"], "AceStepPipeline")
                self.assertEqual(admission["artifact"]["revision"], "200ba991ae448051e14b0183157e35c2d27c9fb0")
                self.assertEqual(admission["sealedBindingValues"]["sampleRate48000"], 48000)
                self.assertIn("nativeMath", admission["executionParameterSources"])
                self.assertIn("deviceMapNone", admission["executionParameterSources"])
                self.assertFalse(admission["executable"])
                self.assertFalse(admission["publication"]["autoEligible"])
                self.assertFalse(admission["publication"]["liveProof"])
                self.assertEqual(definition["outputs"][0]["type"], "audio")
                fields = {f["name"]: f for f in definition["inputs"]}
                self.assertEqual(fields["num_inference_steps"]["default"], 8)
                self.assertEqual(fields["guidance_scale"]["default"], 1.0)
                self.assertIn("lyrics", fields)
                self.assertEqual("source_audio" in fields, mode != "text_to_audio")
                self.assertEqual("repainting_start" in fields, mode == "audio_repaint")
                self.assertEqual("extension_duration" in fields, mode == "audio_continuation")
                self.assertEqual(
                    admission["sealedBindingValues"][{"text_to_audio": "text2music", "audio_variation": "cover",
                        "audio_continuation": "continuation", "audio_repaint": "repaint"}[mode]],
                    {"text_to_audio": "text2music", "audio_variation": "cover",
                        "audio_continuation": "continuation", "audio_repaint": "repaint"}[mode],
                )
                self.assertEqual(len(definition["steps"]), len(spec["roles"]))
                self.assertTrue(all("Modular" not in step["className"] for step in definition["steps"]))


if __name__ == "__main__":
    unittest.main()
