import unittest

from modiff.model_artifact_catalog import (
    catalog_artifact,
    catalog_repository_pin,
    catalog_revision,
    community_artifact_is_discoverable,
    public_model_artifact_catalog,
    read_model_artifact_catalog,
    resolve_model_revision,
)


STUDIO_MODEL_TYPES = {
    "AceStepAudioPipeline",
    "StableAudioPipeline",
    "Flux2KleinPipeline",
    "FluxCannyPipeline",
    "FluxDepthPipeline",
    "FluxDevPipeline",
    "FluxFillPipeline",
    "FluxKontextPipeline",
    "FluxKreaPipeline",
    "FluxReduxPipeline",
    "FluxSchnellPipeline",
    "LTXVideoPipeline",
    "QwenImageEditModularPipeline",
    "QwenImageEditPlusModularPipeline",
    "QwenImageLayeredModularPipeline",
    "QwenImageModularPipeline",
    "WanImageToVideoPipeline",
    "WanTI2VPipeline",
    "WanVACEPipeline",
    "WanVideoPipeline",
    "ZImageModularPipeline",
    "DDPMPipeline",
    "DDIMPipeline",
    "ConsistencyModelPipeline",
    "StableDiffusionPipeline",
    "LatentConsistencyModelPipeline",
    "StableDiffusionPAGPipeline",
    "PixArtSigmaPipeline",
    "AuraFlowPipeline",
    "ChromaPipeline",
    "CogView3PlusPipeline",
    "CogView4Pipeline",
}


class ModelArtifactCatalogTests(unittest.TestCase):
    def test_catalog_covers_every_studio_profile(self):
        catalog = read_model_artifact_catalog()
        model_types = {model["modelType"] for model in catalog["models"]}
        self.assertTrue(STUDIO_MODEL_TYPES.issubset(model_types))
        self.assertNotIn("DiffusionGemmaPipeline", model_types)

    def test_artifacts_expand_version_platform_evidence_and_popularity_fields(self):
        artifact = catalog_artifact("FluxKontextPipeline", "black-forest-labs/FLUX.1-Kontext-dev-NVFP4")
        self.assertIsNotNone(artifact)
        self.assertIn("revision", artifact)
        self.assertEqual(artifact["supportedPlatforms"], ["linux"])
        self.assertEqual(artifact["supportedBackends"], ["cuda"])
        self.assertEqual(artifact["qualificationEvidence"]["status"], "documented")
        self.assertTrue(artifact["popularitySnapshot"]["checkedAt"].startswith("2026-07-18"))

    def test_catalog_auto_artifacts_and_base_models_are_revision_pinned(self):
        catalog = read_model_artifact_catalog()
        for model in catalog["models"]:
            self.assertRegex(model["baseRevision"], r"^[0-9a-f]{40}$", model["baseRepo"])
            self.assertTrue(model.get("baseLicense"), model["baseRepo"])
            for artifact in model.get("artifacts") or []:
                self.assertRegex(artifact["revision"], r"^[0-9a-f]{40}$", artifact["repo"])
                self.assertTrue(artifact.get("license"), artifact["repo"])

        for pin in catalog.get("repositoryPins") or []:
            self.assertRegex(pin["revision"], r"^[0-9a-f]{40}$", pin["repo"])
            self.assertTrue(pin.get("license"), pin["repo"])
            self.assertTrue(pin.get("purpose"), pin["repo"])

    def test_repository_lookup_covers_base_artifact_and_auxiliary_pins(self):
        self.assertEqual(
            catalog_revision("black-forest-labs/FLUX.1-dev"),
            "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
        )
        artifact = catalog_repository_pin("QuantStack/Wan2.2-I2V-A14B-GGUF")
        self.assertEqual(artifact["kind"], "artifact")
        self.assertEqual(artifact["format"], "gguf")
        self.assertEqual(
            catalog_revision("lllyasviel/FramePackI2V_HY"),
            "86cef4396041b6002c957852daac4c91aaa47c79",
        )
        self.assertEqual(
            catalog_revision("stabilityai/stable-video-diffusion-img2vid-xt-1-1"),
            "043843887ccd51926e3efed36270444a838e7861",
        )
        self.assertEqual(
            catalog_revision("PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"),
            "e102b3591cc82e97071b8b4cb90d834d0c487207",
        )
        self.assertEqual(
            catalog_revision("fal/AuraFlow-v0.3"),
            "2cd8588f04c886002be4571697d84654a50e3af3",
        )
        self.assertEqual(
            catalog_revision("lodestones/Chroma1-HD"),
            "0e0c60ece1e82b17cb7f77342d765ba5024c40c0",
        )
        self.assertEqual(
            catalog_revision("zai-org/CogView3-Plus-3B"),
            "5d70e40732ac0efac98524c51a7fa9c82707f1e5",
        )
        self.assertEqual(
            catalog_revision("zai-org/CogView4-6B"),
            "63a52b7f6dace7033380cd6da14d0915eab3e6b5",
        )

    def test_revision_resolution_preserves_explicit_and_unknown_user_selections(self):
        self.assertEqual(
            resolve_model_revision("black-forest-labs/FLUX.1-dev"),
            "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
        )
        self.assertEqual(resolve_model_revision("black-forest-labs/FLUX.1-dev", "user-tag"), "user-tag")
        self.assertIsNone(resolve_model_revision("user/private-model"))
        self.assertIsNone(resolve_model_revision("black-forest-labs/FLUX.1-dev", source="local"))

    def test_popularity_is_discovery_only(self):
        self.assertTrue(community_artifact_is_discoverable({"downloads": 1000, "likes": 0}))
        self.assertTrue(community_artifact_is_discoverable({"downloads": 0, "likes": 10}))
        self.assertFalse(community_artifact_is_discoverable({"downloads": 999, "likes": 9}))
        public = public_model_artifact_catalog()
        self.assertFalse(public["policy"]["popularityIsCompatibilityProof"])
        self.assertFalse(public["selectionPolicy"]["popularityMayChangeAutoSelection"])


if __name__ == "__main__":
    unittest.main()
