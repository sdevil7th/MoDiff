import unittest

from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
from modules.DiffusersAudio.main import AUDIO_PIPELINE_ADAPTERS
from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS
from modules.DiffusersVideo.main import VIDEO_PIPELINE_ADAPTERS


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

    def test_every_direct_profile_is_accepted_by_its_registered_adapter(self):
        """Profiles may be a supported subset; Expert-only adapters need no Auto profile."""

        registries = {
            "modules.DiffusersImage.LoadPipeline": IMAGE_PIPELINE_ADAPTERS,
            "modules.DiffusersVideo.LoadPipeline": VIDEO_PIPELINE_ADAPTERS,
            "modules.DiffusersAudio.LoadPipeline": AUDIO_PIPELINE_ADAPTERS,
        }
        direct_profiles = [
            profile
            for profile in DIFFUSERS_EXECUTION_PROFILES.values()
            if profile.backend_path in registries
        ]
        self.assertTrue(direct_profiles)

        for profile in direct_profiles:
            with self.subTest(profile=profile.id):
                adapter = registries[profile.backend_path].get(profile.pipeline_class)
                self.assertIsNotNone(
                    adapter,
                    f"{profile.id} names unregistered adapter {profile.pipeline_class}",
                )
                self.assertTrue(
                    set(profile.modes).issubset(adapter.modes),
                    f"{profile.id} modes {profile.modes} are outside {adapter.modes}",
                )

                if profile.backend_path == "modules.DiffusersImage.LoadPipeline":
                    managed_repos = {repo.casefold() for repo in adapter.managed_repos}
                    self.assertIn(profile.default_repo.casefold(), managed_repos)
                    if profile.fallback_repo:
                        self.assertIn(profile.fallback_repo.casefold(), managed_repos)
                else:
                    self.assertEqual(profile.default_repo.casefold(), adapter.default_repo.casefold())


if __name__ == "__main__":
    unittest.main()
