import unittest

from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES


class DiffusersExecutionProfileTests(unittest.TestCase):
    def test_every_supported_studio_model_has_an_execution_profile(self):
        expected = {
            "ZImageModularPipeline",
            "QwenImageModularPipeline",
            "QwenImageEditModularPipeline",
            "QwenImageEditPlusModularPipeline",
            "QwenImageLayeredModularPipeline",
            "WanVACEPipeline",
            "WanVideoPipeline",
            "WanImageToVideoPipeline",
            "WanTI2VPipeline",
            "LTXVideoPipeline",
            "AceStepAudioPipeline",
            "FluxSchnellPipeline",
            "FluxDevPipeline",
            "FluxKreaPipeline",
            "FluxKontextPipeline",
            "FluxFillPipeline",
            "FluxDepthPipeline",
            "FluxCannyPipeline",
            "FluxReduxPipeline",
            "Flux2KleinPipeline",
        }
        actual = {profile.model_type for profile in DIFFUSERS_EXECUTION_PROFILES.values()}
        self.assertEqual(expected, actual)

    def test_video_profile_uses_generic_facade(self):
        profile = DIFFUSERS_EXECUTION_PROFILES["wan-vace:direct"]
        self.assertEqual(profile.backend_path, "modules.DiffusersVideo.LoadPipeline")

        wan_video_profile = DIFFUSERS_EXECUTION_PROFILES["wan-video-to-video:direct"]
        self.assertEqual(wan_video_profile.backend_path, "modules.DiffusersVideo.LoadPipeline")
        self.assertEqual(wan_video_profile.pipeline_class, "WanVideoToVideoPipeline")

        wan_text_profile = DIFFUSERS_EXECUTION_PROFILES["wan-text-to-video:direct"]
        self.assertEqual(wan_text_profile.backend_path, "modules.DiffusersVideo.LoadPipeline")
        self.assertEqual(wan_text_profile.pipeline_class, "WanPipeline")

        ltx_profile = DIFFUSERS_EXECUTION_PROFILES["ltx-video:direct"]
        self.assertEqual(ltx_profile.backend_path, "modules.DiffusersVideo.LoadPipeline")
        self.assertEqual(ltx_profile.pipeline_class, "LTXConditionPipeline")


if __name__ == "__main__":
    unittest.main()
