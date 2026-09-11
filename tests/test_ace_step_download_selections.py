import unittest

from modiff.server import STUDIO_MODEL_CAPABILITIES
from modiff.studio_execution_specs import ACE_STEP_DIFFUSERS_FILES
from modules.DiffusersAudio.main import AUDIO_PIPELINE_ADAPTERS


class AceStepDownloadSelectionTests(unittest.TestCase):
    def test_ace_step_selection_is_component_only_and_safe(self):
        selected = set(ACE_STEP_DIFFUSERS_FILES)
        capability = STUDIO_MODEL_CAPABILITIES["AceStepAudioPipeline"]
        adapter = AUDIO_PIPELINE_ADAPTERS["AceStepPipeline"]

        self.assertEqual(capability["downloadFiles"], ACE_STEP_DIFFUSERS_FILES)
        self.assertEqual(len(selected), 21)
        self.assertTrue(adapter.safe_serialization_required)
        self.assertIn(
            "condition_encoder/diffusion_pytorch_model.safetensors",
            selected,
        )
        self.assertIn("text_encoder/model.safetensors", selected)
        self.assertIn(
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            selected,
        )
        self.assertIn("vae/diffusion_pytorch_model.safetensors", selected)
        self.assertNotIn("silence_latent.pt", selected)
        self.assertFalse(
            any(path.endswith((".bin", ".ckpt", ".pt", ".pth")) for path in selected)
        )


if __name__ == "__main__":
    unittest.main()
