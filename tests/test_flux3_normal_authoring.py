"""Metadata-only FLUX 3 authoring; no model, hardware or public qualification."""
from dataclasses import replace
import unittest
from unittest.mock import patch

from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
from modiff import studio_execution_specs as specifications


PROFILES = ("flux3-klein:modular", "flux3-klein-base:modular")
CONTROLS = {
    "offload_mode": "none", "auto_offload": False, "weight_loading_mode": "cpu",
    "quant_config": None, "unet": None, "vae": None, "controlnet": None,
    "trust_remote_code": False,
}
SELECTIONS = [
    {"executionProfileId": "flux3-klein:modular", "pipelineClass": "Flux3DistilledModularPipeline",
     "task": "text_to_image", "authoringRecipeId": "flux3-klein:text-to-image:prompt-preview-fp32-authoring:v3",
     "rawAuthoringRecipeId": "flux3-klein:text-to-image:prompt-preview-fp32-authoring:v1",
     "rawAuthoringRecipeChoices": [
         {"authoringRecipeId": "flux3-klein:text-to-image:prompt-preview-fp32-authoring:v1", "label": "Raw wording: FP32"},
         {"authoringRecipeId": "flux3-klein:text-to-image:prompt-preview-authoring:v1", "label": "Raw wording: BF16 + NF4"}]},
    {"executionProfileId": "flux3-klein:modular", "pipelineClass": "Flux3DistilledModularPipeline",
     "task": "edit_image", "authoringRecipeId": "flux3-klein:edit-image:prompt-preview-authoring:v3",
     "rawAuthoringRecipeId": "flux3-klein:edit-image:prompt-preview-authoring:v1"},
    {"executionProfileId": "flux3-klein-base:modular", "pipelineClass": "Flux3ModularPipeline",
     "task": "text_to_image", "authoringRecipeId": "flux3-klein-base:text-to-image:prompt-preview-fp32-authoring:v3",
     "rawAuthoringRecipeId": "flux3-klein-base:text-to-image:prompt-preview-fp32-authoring:v1",
     "rawAuthoringRecipeChoices": [
         {"authoringRecipeId": "flux3-klein-base:text-to-image:prompt-preview-fp32-authoring:v1", "label": "Raw wording: FP32"},
         {"authoringRecipeId": "flux3-klein-base:text-to-image:prompt-preview-authoring:v1", "label": "Raw wording: BF16 + NF4"}]},
    {"executionProfileId": "flux3-klein-base:modular", "pipelineClass": "Flux3ModularPipeline",
     "task": "edit_image", "authoringRecipeId": "flux3-klein-base:edit-image:prompt-preview-authoring:v3",
     "rawAuthoringRecipeId": "flux3-klein-base:edit-image:prompt-preview-authoring:v1"},
]


class Flux3NormalAuthoringTests(unittest.TestCase):
    def test_private_profiles_publish_reviewed_lora_edit_permission_only(self):
        for identity in PROFILES:
            with self.subTest(identity=identity):
                profile = DIFFUSERS_EXECUTION_PROFILES[identity]
                self.assertIsNotNone(profile.adapter_authoring, "Normal Add LoRA has no reviewed authoring policy")
                public = profile.to_public_dict(platform_name="linux", machine="x86_64")
                self.assertEqual(public.pop("adapterAuthoring"), {
                    "schemaVersion": 1, "mediaKind": "image", "adapterNodeKey": "modules.ModularDiffusers.Lora",
                    "tasks": ["text_to_image", "edit_image"], "maxOwnedAdapters": 32,
                    "repositoryRevisions": [{"repo": profile.default_repo,
                        "revision": "8454017f9ebaada30c998fccd9f67a80c7580f68" if identity == PROFILES[0]
                        else "1469d27cfb9bb1bd81d3a93f5a023b39dbf136e4"}],
                    "requiredControls": CONTROLS,
                })
                self.assertEqual(public, replace(profile, adapter_authoring=None).to_public_dict(
                    platform_name="linux", machine="x86_64"))
                self.assertFalse(profile.public)
                self.assertFalse(profile.live_proof)

    def test_normal_authoring_selection_declares_raw_alternatives_for_all_four_append_parents(self):
        select = getattr(specifications, "studio_operation_authoring_selections", None)
        self.assertTrue(callable(select), "Normal workflow metadata has no declared PUP parent selections")
        profiles = [DIFFUSERS_EXECUTION_PROFILES[identity].to_public_dict() for identity in PROFILES]
        self.assertEqual(select(profiles), SELECTIONS)
        self.assertEqual(select(profiles[:1]), SELECTIONS[:2])
        self.assertEqual(select([DIFFUSERS_EXECUTION_PROFILES["flux2-klein:modular"].to_public_dict()]), [])

    def test_normal_raw_alternative_refuses_missing_cross_profile_or_layout_parent(self):
        profile = DIFFUSERS_EXECUTION_PROFILES[PROFILES[0]].to_public_dict()
        identity = SELECTIONS[0]["rawAuthoringRecipeId"]
        definition = specifications._OPERATION_AUTHORING_RECIPES[identity]
        for value in (None, {**definition, "task": "edit_image"},
                      {**definition, "profileId": PROFILES[1]}):
            with self.subTest(definition=value), patch.dict(
                specifications._OPERATION_AUTHORING_RECIPES, {identity: value},
            ):
                with self.assertRaisesRegex(ValueError, "stale"):
                    specifications.studio_operation_authoring_selections([profile])
        parent = specifications._ANCILLARY_PRIVATE_PARENTS[identity]
        for changes in ({"layoutPolicy": "append-details-preserve-layout-v1"},
                        {"providerRecipe": "other"}, {"pipelineClass": "Flux3ModularPipeline"},
                        {"specId": "foreign"}):
            with self.subTest(changes=changes), patch.dict(specifications._ANCILLARY_PRIVATE_PARENTS,
                            {identity: {**parent, **changes}}):
                with self.assertRaisesRegex(ValueError, "stale"):
                    specifications.studio_operation_authoring_selections([profile])

    def test_raw_choices_refuse_ambiguous_excessive_or_malformed_declarations(self):
        profile = DIFFUSERS_EXECUTION_PROFILES[PROFILES[0]].to_public_dict()
        default = SELECTIONS[0]["rawAuthoringRecipeId"]
        append = SELECTIONS[0]["authoringRecipeId"]
        choices = specifications._NORMAL_OPERATION_RAW_AUTHORING_CHOICES[default]
        for invalid in ([*choices], ((default, "FP32"),) * 2, choices * 3,
                        ((append, "append"),), choices[1:], ((default, ""),),
                        ((default, " value"),), ((default, "x" * 129),),
                        ((default, "line\nbreak"),), ((default, True),)):
            with self.subTest(choices=invalid), patch.dict(
                specifications._NORMAL_OPERATION_RAW_AUTHORING_CHOICES, {default: invalid},
            ):
                with self.assertRaisesRegex(ValueError, "stale"):
                    specifications.studio_operation_authoring_selections([profile])

    def test_raw_choice_refuses_stale_cross_task_profile_and_layout_parent(self):
        profile = DIFFUSERS_EXECUTION_PROFILES[PROFILES[0]].to_public_dict()
        identity = SELECTIONS[0]["rawAuthoringRecipeChoices"][1]["authoringRecipeId"]
        definition = specifications._OPERATION_AUTHORING_RECIPES[identity]
        for changed in (None, {**definition, "task": "edit_image"},
                        {**definition, "profileId": PROFILES[1]}):
            with self.subTest(definition=changed), patch.dict(
                specifications._OPERATION_AUTHORING_RECIPES, {identity: changed},
            ):
                with self.assertRaisesRegex(ValueError, "stale"):
                    specifications.studio_operation_authoring_selections([profile])
        parent = specifications._ANCILLARY_PRIVATE_PARENTS[identity]
        for changes in ({"layoutPolicy": "append-details-preserve-layout-v1"},
                        {"providerRecipe": "fp32-none"}, {"pipelineClass": "Flux3ModularPipeline"}):
            with self.subTest(changes=changes), patch.dict(specifications._ANCILLARY_PRIVATE_PARENTS,
                            {identity: {**parent, **changes}}):
                with self.assertRaisesRegex(ValueError, "stale"):
                    specifications.studio_operation_authoring_selections([profile])

    def test_normal_parent_selection_refuses_missing_or_stale_reviewed_definitions(self):
        profile = DIFFUSERS_EXECUTION_PROFILES[PROFILES[0]].to_public_dict()
        identity = SELECTIONS[0]["authoringRecipeId"]
        definition = specifications._OPERATION_AUTHORING_RECIPES[identity]
        for value in (None, {**definition, "task": "edit_image"},
                      {**definition, "profileId": PROFILES[1]}):
            with self.subTest(definition=value), patch.dict(
                specifications._OPERATION_AUTHORING_RECIPES, {identity: value},
            ):
                with self.assertRaisesRegex(ValueError, "stale"):
                    specifications.studio_operation_authoring_selections([profile])
        parent = specifications._ANCILLARY_PRIVATE_PARENTS[identity]
        with patch.dict(specifications._ANCILLARY_PRIVATE_PARENTS,
                        {identity: {**parent, "layoutPolicy": "preserve-authored-v1"}}):
            with self.assertRaisesRegex(ValueError, "stale"):
                specifications.studio_operation_authoring_selections([profile])

    def test_normal_parent_selection_cannot_cross_profile_pipeline_or_task(self):
        profile = DIFFUSERS_EXECUTION_PROFILES[PROFILES[0]].to_public_dict()
        for changes in ({"pipeline_class": "Flux3ModularPipeline"}, {"modes": ["edit_image"]}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "stale"):
                specifications.studio_operation_authoring_selections([{**profile, **changes}])


if __name__ == "__main__":
    unittest.main()
