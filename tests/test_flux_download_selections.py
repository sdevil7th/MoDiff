import unittest

from modiff.studio_execution_specs import (
    FLUX_CONTROL_DIFFUSERS_FILES,
    FLUX_DEV_DIFFUSERS_FILES,
    FLUX_FILL_DIFFUSERS_FILES,
    FLUX_KREA_DIFFUSERS_FILES,
    FLUX_KONTEXT_DIFFUSERS_FILES,
    FLUX_REDUX_DIFFUSERS_FILES,
    FLUX_SCHNELL_DIFFUSERS_FILES,
    FLUX2_KLEIN_DIFFUSERS_FILES,
    studio_capability_definitions,
)
from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS


class FluxDownloadSelectionTests(unittest.TestCase):
    def test_flux1_control_and_edit_routes_use_component_only_safe_selections(self):
        capabilities = studio_capability_definitions()
        control_cases = {
            "FluxDepthPipeline": "flux1-depth-dev.safetensors",
            "FluxCannyPipeline": "flux1-canny-dev.safetensors",
        }
        for model_type, native_checkpoint in control_cases.items():
            with self.subTest(model_type=model_type):
                selected = set(FLUX_CONTROL_DIFFUSERS_FILES)
                self.assertEqual(
                    capabilities[model_type]["downloadFiles"],
                    FLUX_CONTROL_DIFFUSERS_FILES,
                )
                self.assertEqual(len(selected), 28)
                self.assertNotIn("ae.safetensors", selected)
                self.assertNotIn(native_checkpoint, selected)
                self.assertIn(
                    "text_encoder_2/model-00004-of-00004.safetensors",
                    selected,
                )

        server_cases = {
            "FluxFillPipeline": (
                FLUX_FILL_DIFFUSERS_FILES,
                "flux1-fill-dev.safetensors",
            ),
            "FluxKontextPipeline": (
                FLUX_KONTEXT_DIFFUSERS_FILES,
                "flux1-kontext-dev.safetensors",
            ),
        }
        from modiff.server import STUDIO_MODEL_CAPABILITIES

        for model_type, (files, native_checkpoint) in server_cases.items():
            with self.subTest(model_type=model_type):
                selected = set(files)
                self.assertEqual(STUDIO_MODEL_CAPABILITIES[model_type]["downloadFiles"], files)
                self.assertEqual(len(selected), 26)
                self.assertNotIn("ae.safetensors", selected)
                self.assertNotIn(native_checkpoint, selected)
                self.assertNotIn("teaser.png", selected)

        for model_type in (
            "FluxControlPipeline",
            "FluxFillPipeline",
            "FluxKontextPipeline",
            "FluxKontextInpaintPipeline",
        ):
            with self.subTest(adapter=model_type):
                self.assertTrue(
                    IMAGE_PIPELINE_ADAPTERS[model_type].safe_serialization_required
                )

    def test_flux1_text_routes_use_component_only_safe_selections(self):
        capabilities = studio_capability_definitions()
        cases = {
            "FluxSchnellPipeline": (
                FLUX_SCHNELL_DIFFUSERS_FILES,
                "flux1-schnell.safetensors",
                "schnell_grid.jpeg",
            ),
            "FluxDevPipeline": (
                FLUX_DEV_DIFFUSERS_FILES,
                "flux1-dev.safetensors",
                "dev_grid.jpg",
            ),
            "FluxKreaPipeline": (
                FLUX_KREA_DIFFUSERS_FILES,
                "flux1-krea-dev.safetensors",
                "teaser.png",
            ),
        }
        for model_type, (files, native_checkpoint, demo) in cases.items():
            with self.subTest(model_type=model_type):
                selected = set(files)
                self.assertEqual(capabilities[model_type]["downloadFiles"], files)
                self.assertNotIn("ae.safetensors", selected)
                self.assertNotIn(native_checkpoint, selected)
                self.assertNotIn(demo, selected)
                self.assertIn(
                    "transformer/diffusion_pytorch_model.safetensors.index.json",
                    selected,
                )
                self.assertIn("vae/diffusion_pytorch_model.safetensors", selected)
                self.assertFalse(
                    any(
                        path.endswith((".bin", ".ckpt", ".pt", ".pth"))
                        for path in selected
                    )
                )

        self.assertEqual(len(FLUX_SCHNELL_DIFFUSERS_FILES), 25)
        self.assertEqual(len(FLUX_DEV_DIFFUSERS_FILES), 26)
        self.assertEqual(len(FLUX_KREA_DIFFUSERS_FILES), 26)
        for model_type in (
            "FluxPipeline",
            "FluxImg2ImgPipeline",
            "FluxInpaintPipeline",
        ):
            with self.subTest(adapter=model_type):
                self.assertTrue(
                    IMAGE_PIPELINE_ADAPTERS[model_type].safe_serialization_required
                )

    def test_flux2_klein_selection_is_component_only_and_safe(self):
        selected = set(FLUX2_KLEIN_DIFFUSERS_FILES)
        capability = studio_capability_definitions()["Flux2KleinPipeline"]

        self.assertEqual(capability["downloadFiles"], FLUX2_KLEIN_DIFFUSERS_FILES)
        self.assertEqual(len(selected), 21)
        for model_type in ("Flux2KleinPipeline", "Flux2KleinInpaintPipeline"):
            with self.subTest(model_type=model_type):
                self.assertTrue(
                    IMAGE_PIPELINE_ADAPTERS[model_type].safe_serialization_required
                )
        self.assertIn("text_encoder/model.safetensors.index.json", selected)
        self.assertIn(
            "transformer/diffusion_pytorch_model.safetensors",
            selected,
        )
        self.assertIn("vae/diffusion_pytorch_model.safetensors", selected)
        self.assertNotIn("flux-2-klein-4b.safetensors", selected)
        self.assertFalse(any(path.endswith(".jpg") for path in selected))
        self.assertFalse(
            any(path.endswith((".bin", ".ckpt", ".pt", ".pth")) for path in selected)
        )

    def test_flux_redux_selection_keeps_only_prior_components(self):
        selected = set(FLUX_REDUX_DIFFUSERS_FILES)
        capability = studio_capability_definitions()["FluxReduxPipeline"]

        self.assertEqual(capability["downloadFiles"], FLUX_REDUX_DIFFUSERS_FILES)
        self.assertEqual(len(selected), 9)
        self.assertTrue(
            IMAGE_PIPELINE_ADAPTERS["FluxReduxPipeline"].safe_serialization_required
        )
        self.assertIn("image_embedder/diffusion_pytorch_model.safetensors", selected)
        self.assertIn("image_encoder/model.safetensors", selected)
        self.assertNotIn("flux1-redux-dev.safetensors", selected)
        self.assertNotIn("redux.png", selected)
        self.assertFalse(
            any(path.endswith((".bin", ".ckpt", ".pt", ".pth")) for path in selected)
        )


if __name__ == "__main__":
    unittest.main()
