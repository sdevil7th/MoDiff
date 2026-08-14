import unittest

from modiff.studio_execution_specs import (
    WHISPER_TINY_TRANSFORMERS_FILES,
    studio_capability_definitions,
)


class SpeechDownloadSelectionTests(unittest.TestCase):
    def test_whisper_tiny_selection_keeps_one_safe_transformers_surface(self):
        selected = set(WHISPER_TINY_TRANSFORMERS_FILES)
        capability = studio_capability_definitions()["HuggingFaceSpeechRecognitionModel"]

        self.assertEqual(
            capability["downloadFiles"],
            WHISPER_TINY_TRANSFORMERS_FILES,
        )
        self.assertEqual(len(selected), 13)
        self.assertIn("model.safetensors", selected)
        self.assertIn("preprocessor_config.json", selected)
        self.assertIn("generation_config.json", selected)
        self.assertNotIn("pytorch_model.bin", selected)
        self.assertNotIn("flax_model.msgpack", selected)
        self.assertNotIn("tf_model.h5", selected)
        self.assertFalse(
            any(path.endswith((".bin", ".ckpt", ".pt", ".pth")) for path in selected)
        )


if __name__ == "__main__":
    unittest.main()
