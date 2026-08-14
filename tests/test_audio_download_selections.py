import unittest

from modiff.studio_execution_specs import (
    AUDIO_LDM2_DIFFUSERS_FILES,
    STABLE_AUDIO_DIFFUSERS_FILES,
    studio_capability_definitions,
)


class AudioDownloadSelectionTests(unittest.TestCase):
    def test_audioldm2_selection_excludes_every_legacy_pickle_duplicate(self):
        selected = set(AUDIO_LDM2_DIFFUSERS_FILES)
        capability = studio_capability_definitions()["AudioLDM2Pipeline"]

        self.assertEqual(capability["downloadFiles"], AUDIO_LDM2_DIFFUSERS_FILES)
        self.assertEqual(len(selected), 28)
        for component in (
            "language_model/model.safetensors",
            "projection_model/diffusion_pytorch_model.safetensors",
            "text_encoder/model.safetensors",
            "text_encoder_2/model.safetensors",
            "unet/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
            "vocoder/model.safetensors",
        ):
            self.assertIn(component, selected)
        self.assertFalse(any(path.endswith(".bin") for path in selected))

    def test_stable_audio_selection_is_runnable_and_excludes_duplicate_checkpoints(self):
        selected = set(STABLE_AUDIO_DIFFUSERS_FILES)
        capability = studio_capability_definitions()["StableAudioPipeline"]

        self.assertEqual(capability["downloadFiles"], STABLE_AUDIO_DIFFUSERS_FILES)
        self.assertEqual(len(selected), 19)
        self.assertIn("projection_model/diffusion_pytorch_model.safetensors", selected)
        self.assertIn("text_encoder/model.safetensors", selected)
        self.assertIn("transformer/diffusion_pytorch_model.safetensors", selected)
        self.assertIn("vae/diffusion_pytorch_model.safetensors", selected)
        self.assertIn("fma_dataset_attribution2.csv", selected)
        self.assertIn("freesound_dataset_attribution2.csv", selected)
        self.assertNotIn("model.ckpt", selected)
        self.assertNotIn("model.safetensors", selected)
        self.assertNotIn("vae_model.ckpt", selected)
        self.assertFalse(
            any(path.endswith((".bin", ".ckpt", ".pt", ".pth")) for path in selected)
        )


if __name__ == "__main__":
    unittest.main()
