"""Curated fresh graph choices belong to reviewed backend declarations."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import pytest

from modules import MODULE_MAP
from modiff.operation_catalog import build_operation_catalog
from modiff.operation_starters import resolve_operation_starter
from modiff.studio_execution_specs import studio_execution_spec_for_pair


_PRIVATE_PROMPT_RECIPES = (
    "flux3-klein:text-to-image:prompt-preview-authoring:v1",
    "flux3-klein:edit-image:prompt-preview-authoring:v1",
    "flux3-klein-base:text-to-image:prompt-preview-authoring:v1",
    "flux3-klein-base:edit-image:prompt-preview-authoring:v1",
    "flux3-klein:text-to-image:prompt-preview-authoring:v2",
    "flux3-klein:edit-image:prompt-preview-authoring:v2",
    "flux3-klein-base:text-to-image:prompt-preview-authoring:v2",
    "flux3-klein-base:edit-image:prompt-preview-authoring:v2",
    "flux3-klein:text-to-image:prompt-preview-fp32-authoring:v1",
    "flux3-klein-base:text-to-image:prompt-preview-fp32-authoring:v1",
    "flux3-klein:text-to-image:prompt-preview-fp32-authoring:v2",
    "flux3-klein:edit-image:prompt-preview-authoring:v3",
    "flux3-klein:text-to-image:prompt-preview-fp32-authoring:v3",
    "flux3-klein-base:edit-image:prompt-preview-authoring:v3",
    "flux3-klein-base:text-to-image:prompt-preview-fp32-authoring:v3",
    "flux3-klein:edit-image:prompt-preview-nf4-authoring:v1",
    "flux3-klein-base:edit-image:prompt-preview-nf4-authoring:v1",
)


class OperationAuthoringRecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contracts, _ = build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})

    def resolve(self, pipeline, task, profile, recipe=None):
        selection = {"pipelineClass": pipeline, "task": task, "executionProfileId": profile}
        if recipe is not None:
            selection["authoringRecipeId"] = recipe
        with patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("constructed")):
            return resolve_operation_starter(MODULE_MAP, self.contracts, selection)

    def test_qwen_masked_compatibility_is_explicit_and_does_not_change_generic_defaults(self):
        args = ("QwenImageEditModularPipeline", "inpaint", "qwen-edit:modular")
        original = deepcopy(self.resolve(*args))
        recipe = self.resolve(*args, "retained-qwen-edit-inpaint:v1")["authoring"]
        literals = {(role, field): value for role, field, value in recipe["initialValues"]}
        self.assertEqual(literals[("models", "inpaint_compatibility")], "whole_v1")
        self.assertEqual(literals[("imageEncode", "inpaint_compatibility")], "whole_v1")
        self.assertIn(("imageEncode", "padding_mask_crop", "paddingMaskCrop"), recipe["bindings"])
        self.assertIn(("denoise", "width", "width"), recipe["bindings"])
        self.assertIn(("denoise", "height", "height"), recipe["bindings"])
        self.assertIn(("guider", "guidance_scale", "guidanceScale"), recipe["bindings"])
        self.assertEqual(self.resolve(*args), original)
        self.assertNotIn("authoring", original)

    def test_outpaint_declares_canvas_and_media_wires_without_a_mask_loader(self):
        recipe = self.resolve(
            "QwenImageEditModularPipeline", "inpaint", "qwen-edit:modular",
            "retained-qwen-edit-outpaint:v1",
        )["authoring"]
        roles = {role: key for role, key, _x, _y in recipe["roles"]}
        self.assertEqual(roles["outpaintCanvas"], "modules.DiffusersImage.OutpaintCanvas")
        self.assertNotIn("loadMask", roles)
        for edge in (
            ("loadImage", "image", "outpaintCanvas", "image"),
            ("outpaintCanvas", "canvas", "imageEncode", "image"),
            ("outpaintCanvas", "canvas", "prompt", "image"),
            ("outpaintCanvas", "mask_image", "imageEncode", "mask_image"),
        ):
            self.assertIn(edge, recipe["edges"])
        self.assertIn(("outpaintCanvas", "left", "outpaintLeft"), recipe["bindings"])
        self.assertFalse(any(source == "maskImage" for _role, _field, source in recipe["bindings"]))

    def test_mask_crop_padding_declares_its_existing_optional_null_default(self):
        from modules.ModularDiffusers.pipeline_schema import MoDiffParam
        from modules.ModularDiffusers.route_state import validate_sdxl_crop_overlay_inputs

        self.assertNotIn("default", MoDiffParam(name="optional", label="Optional", type="int").to_dict())
        for pipeline, profile in (
            ("QwenImageEditModularPipeline", "qwen-edit:modular"),
            ("StableDiffusionXLModularPipeline", "sdxl-base:modular"),
        ):
            with self.subTest(pipeline=pipeline):
                starter = self.resolve(pipeline, "inpaint", profile)
                image_encoder = next(n for n in starter["nodes"]
                                     if n["operation"]["nodeKey"].endswith(".ImageEncode"))
                field = image_encoder["params"]["padding_mask_crop"]
                self.assertIn("default", field)
                self.assertIsNone(field["default"])
                self.assertEqual((field["type"], field["min"], field["max"]), ("int", 0, 8192))
                self.assertIsNot(field.get("required"), True)
                self.assertNotIn("_default_is_explicit", field)
        validate_sdxl_crop_overlay_inputs(None, None, None)
        with self.assertRaisesRegex(ValueError, "canonical integer"):
            validate_sdxl_crop_overlay_inputs(False, None, None)

    def test_retained_guidance_distinguishes_cfg_from_embedded_flux_scale(self):
        for pipeline, profile, recipe_id, enabled, original, embedded in (
            ("ZImageModularPipeline", "z-image:modular", "retained-z-image-text-to-image:v1", True, True, None),
            ("FluxModularPipeline", "flux-schnell:modular", "retained-flux-schnell-text-to-image:v1", False, False, 0),
            ("FluxModularPipeline", "flux-dev:modular", "retained-flux-dev-text-to-image:v1", True, False, 3.5),
            ("FluxModularPipeline", "flux-krea:modular", "retained-flux-krea-text-to-image:v1", True, False, 3.5),
        ):
            with self.subTest(recipe=recipe_id):
                recipe = self.resolve(pipeline, "text_to_image", profile, recipe_id)["authoring"]
                literals = {(role, field): value for role, field, value in recipe["initialValues"]}
                self.assertIs(literals[("guider", "enabled")], enabled)
                self.assertIs(literals[("guider", "use_original_formulation")], original)
                self.assertIn(("guider", "guidance_scale", "guidanceScale"), recipe["bindings"])
                self.assertNotIn(("denoise", "guidance_scale", "guidanceScale"), recipe["bindings"])
                self.assertEqual(literals.get(("denoise", "guidance_scale")), embedded)

    def test_recipe_is_exact_profile_task_selection_not_client_supplied_code(self):
        for overrides in (
            {"authoringRecipeId": "unknown"},
            {"authoringRecipeId": " retained-flux-dev-text-to-image:v1"},
            {"authoringRecipeId": []},
            {"authoringRecipeId": "retained-flux-schnell-text-to-image:v1"},
            {"task": "image_to_image"},
            {"initialValues": [["models", "trust_remote_code", True]]},
        ):
            selection = {
                "pipelineClass": "FluxModularPipeline", "task": "text_to_image",
                "executionProfileId": "flux-dev:modular", "authoringRecipeId": "retained-flux-dev-text-to-image:v1",
                **overrides,
            }
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                resolve_operation_starter(MODULE_MAP, self.contracts, selection)

    def test_recipe_hash_covers_primitives_wires_and_base_spec_identity(self):
        from modiff.studio_execution_specs import _hash_string, _stable_json

        recipe = self.resolve(
            "FluxModularPipeline", "text_to_image", "flux-dev:modular", "retained-flux-dev-text-to-image:v1",
        )["authoring"]
        spec = studio_execution_spec_for_pair("FluxModularPipeline", "text_to_image")
        self.assertEqual(recipe["baseSpecId"], spec["id"])
        self.assertEqual(recipe["baseSpecContentHash"], spec["contentHash"])
        semantic = {key: value for key, value in recipe.items() if key != "contentHash"}
        self.assertEqual(recipe["contentHash"], "studio-authoring-v1-" + _hash_string(_stable_json(semantic)))
        self.assertTrue(all(value is None or type(value) in {str, int, float, bool}
                            for _role, _field, value in recipe["initialValues"]))

    def test_all_reviewed_recipes_resolve_and_do_not_mutate_ordinary_starters(self):
        from modiff.studio_execution_specs import _OPERATION_AUTHORING_RECIPES, STUDIO_EXECUTION_SPEC_DEFINITIONS

        private = {identity for identity, definition in _OPERATION_AUTHORING_RECIPES.items()
                   if STUDIO_EXECUTION_SPEC_DEFINITIONS[definition["specId"]]["profile"].get("public") is False}
        self.assertEqual(private, set(_PRIVATE_PROMPT_RECIPES))
        for identity, definition in _OPERATION_AUTHORING_RECIPES.items():
            with self.subTest(recipe=identity):
                spec = STUDIO_EXECUTION_SPEC_DEFINITIONS[definition["specId"]]
                args = (spec["profile"]["pipeline_class"], definition["task"], definition["profileId"])
                if identity in private:
                    # A reviewed declaration alone grants no ordinary authoring
                    # availability, including when a source-audit API is present.
                    with patch("modiff.diffusers_evaluation_runtime.selected_diffusers_evaluation_authoring_profile_ids",
                               return_value=()), self.assertRaisesRegex(
                                   ValueError, "No complete operation binding|execution profile"):
                        self.resolve(*args, identity)
                    continue
                ordinary = deepcopy(self.resolve(*args))
                authored = self.resolve(*args, identity)
                self.assertEqual(authored["authoring"]["id"], identity)
                self.assertEqual(authored["authoring"]["executionProfileIds"], (definition["profileId"],))
                self.assertEqual(self.resolve(*args), ordinary)

    def test_generic_native_image_recipe_binds_the_actual_guider_without_family_literals(self):
        for pipeline, task, profile in (
            ("QwenImageEditModularPipeline", "edit_image", "qwen-edit:modular"),
            ("QwenImageEditModularPipeline", "inpaint", "qwen-edit:modular"),
            ("FluxModularPipeline", "text_to_image", "flux-dev:modular"),
            ("FluxModularPipeline", "image_to_image", "flux-dev:modular"),
            ("FluxKontextModularPipeline", "text_to_image", "flux-kontext:modular"),
            ("FluxKontextModularPipeline", "edit_image", "flux-kontext:modular"),
            ("Flux2KleinModularPipeline", "text_to_image", "flux2-klein:modular"),
            ("Flux2KleinModularPipeline", "edit_image", "flux2-klein:modular"),
            ("Flux2KleinBaseModularPipeline", "text_to_image", "flux2-klein-base:modular"),
            ("Flux2KleinBaseModularPipeline", "edit_image", "flux2-klein-base:modular"),
            ("ZImageModularPipeline", "text_to_image", "z-image:modular"),
            ("ZImageModularPipeline", "image_to_image", "z-image:modular"),
            ("StableDiffusionXLModularPipeline", "text_to_image", "sdxl-base:modular"),
            ("StableDiffusionXLModularPipeline", "inpaint", "sdxl-base:modular"),
            ("StableDiffusionXLModularPipeline", "control_image", "sdxl-base:modular"),
            ("StableDiffusionXLModularPipeline", "control_edit_image", "sdxl-base:modular"),
            ("StableDiffusionXLModularPipeline", "control_inpaint", "sdxl-base:modular"),
            ("QwenImageEditPlusModularPipeline", "edit_image", "qwen-edit-plus:modular"),
            ("QwenImageEditPlusModularPipeline", "multi_image_reference_edit", "qwen-edit-plus:modular"),
            ("QwenImageLayeredModularPipeline", "layer_decomposition", "qwen-layered:modular"),
            ("QwenImageModularPipeline", "control_image", "qwen-image:modular"),
            ("QwenImageModularPipeline", "image_to_image", "qwen-image:modular"),
            ("QwenImageModularPipeline", "text_to_image", "qwen-image:modular"),
            ("Cosmos3OmniModularPipeline", "text_to_image", "cosmos3-nano:official-modular-workflow"),
            ("Cosmos3DistilledModularPipeline", "text_to_image", "cosmos3-distilled-text-to-image:official-modular-workflow"),
            ("AnimaModularPipeline", "text_to_image", "anima:official-modular-workflow"),
            ("AnimaModularPipeline", "image_to_image", "anima:official-modular-workflow"),
            ("ErnieImageModularPipeline", "text_to_image", "ernie-image-turbo:official-modular-workflow"),
            ("Flux2ModularPipeline", "text_to_image", "flux2:modular"),
            ("Flux2ModularPipeline", "edit_image", "flux2:modular"),
            ("Cosmos3OmniModularPipeline", "text_to_image", "cosmos3-super-text-to-image:official-modular-workflow"),
        ):
            with self.subTest(pipeline=pipeline):
                ordinary = self.resolve(pipeline, task, profile)
                result = self.resolve(pipeline, task, profile, "generic-native-image:v1")
                authoring = result["authoring"]
                self.assertEqual(authoring["initialValues"], [])
                if any(n["operation"]["operationId"] == "diffusion.guidance" for n in ordinary["nodes"]):
                    self.assertIn(("guider", "guidance_scale", "guidanceScale"), authoring["bindings"])
                    self.assertNotIn(("denoise", "guidance_scale", "guidanceScale"), authoring["bindings"])
                else:
                    self.assertNotIn("guider", {role for role, *_rest in authoring["roles"]})
                self.assertEqual(self.resolve(pipeline, task, profile), ordinary)

    def test_generic_image_recipe_rejects_other_modalities_and_ambiguous_specs(self):
        for pipeline, task, profile in (
            ("Cosmos3OmniModularPipeline", "text_to_video", "cosmos3-nano:official-modular-workflow"),
            ("MiniMaxMusic3ModularPipeline", "text_to_audio", "minimax-music3:official-modular-workflow"),
        ):
            with self.subTest(task=task), self.assertRaisesRegex(ValueError, "image specification"):
                self.resolve(pipeline, task, profile, "generic-native-image:v1")
        import modiff.studio_execution_specs as specs

        original = specs.STUDIO_EXECUTION_SPEC_DEFINITIONS["qwen-image-edit-plus:edit-image:v1"]
        with patch.dict(specs.STUDIO_EXECUTION_SPEC_DEFINITIONS, {"duplicate-image-spec": deepcopy(original)}), \
                self.assertRaisesRegex(ValueError, "one exact"):
            self.resolve("QwenImageEditPlusModularPipeline", "edit_image", "qwen-edit-plus:modular", "generic-native-image:v1")

    def test_reviewed_literal_metadata_cannot_change_identity_or_inject_objects(self):
        import modiff.studio_execution_specs as specs

        identity = "retained-qwen-image-text-to-image:v1"
        original = specs._OPERATION_AUTHORING_RECIPES[identity]
        for literal in (("models", "trust_remote_code", True),
                        ("models", "revision", "main"),
                        ("models", "attention_backend", {"script": "arbitrary"}),
                        ("models", "attention_backend", "unknown")):
            with self.subTest(literal=literal), patch.dict(
                specs._OPERATION_AUTHORING_RECIPES, {identity: {**original, "initialValues": (literal,)}}
            ), self.assertRaisesRegex(ValueError, "literal"):
                self.resolve("QwenImageModularPipeline", "text_to_image", "qwen-image:modular", identity)

    def test_form_bindings_cannot_replace_identity_or_change_scalar_value_types(self):
        import modiff.studio_execution_specs as specs

        identity = "retained-qwen-image-text-to-image:v1"
        spec_id = specs._OPERATION_AUTHORING_RECIPES[identity]["specId"]
        original = specs.STUDIO_EXECUTION_SPEC_DEFINITIONS[spec_id]
        for role, field, source in (
            ("models", "revision", "prompt"),
            ("prompt", "pipeline_class", "negativePrompt"),
            ("guider", "enabled", "guidanceScale"),
            ("denoise", "num_inference_steps", "guidanceScale"),
        ):
            bindings = tuple(b for b in original["bindings"] if b[:2] != (role, field)) + ((role, field, source),)
            with self.subTest(binding=(role, field, source)), patch.dict(
                specs.STUDIO_EXECUTION_SPEC_DEFINITIONS, {spec_id: {**original, "bindings": bindings}}
            ), self.assertRaisesRegex(ValueError, "form binding"):
                self.resolve("QwenImageModularPipeline", "text_to_image", "qwen-image:modular", identity)

    def test_initial_primitives_must_match_the_actual_scalar_field_type(self):
        import modiff.studio_execution_specs as specs

        identity = "retained-qwen-image-text-to-image:v1"
        original = specs._OPERATION_AUTHORING_RECIPES[identity]
        for literal in (("guider", "enabled", 1), ("guider", "use_original_formulation", "true"),
                        ("guider", "enabled", None)):
            with self.subTest(literal=literal), patch.dict(
                specs._OPERATION_AUTHORING_RECIPES, {identity: {**original, "initialValues": (literal,)}}
            ), self.assertRaisesRegex(ValueError, "literal"):
                self.resolve("QwenImageModularPipeline", "text_to_image", "qwen-image:modular", identity)

    def test_auxiliary_model_identity_rejects_creative_form_bindings(self):
        import modiff.studio_execution_specs as specs

        public = specs.studio_execution_spec_for_pair("QwenImageModularPipeline", "control_image")
        original = specs.STUDIO_EXECUTION_SPEC_DEFINITIONS[public["id"]]
        for field in ("revision", "model_id", "model_type", "variant"):
            bindings = tuple(b for b in original["bindings"] if b[:2] != ("controlnetModel", field)) + (
                ("controlnetModel", field, "prompt"),
            )
            with self.subTest(field=field), patch.dict(
                specs.STUDIO_EXECUTION_SPEC_DEFINITIONS, {public["id"]: {**original, "bindings": bindings}},
            ), self.assertRaisesRegex(ValueError, "form binding"):
                self.resolve("QwenImageModularPipeline", "control_image", "qwen-image:modular", "generic-native-image:v1")

    def test_null_literal_requires_an_explicit_optional_resolved_or_default_value(self):
        import modiff.studio_execution_specs as specs

        identity = "retained-qwen-image-text-to-image:v1"
        selection = {"pipelineClass": "QwenImageModularPipeline", "task": "text_to_image",
                     "executionProfileId": "qwen-image:modular", "authoringRecipeId": identity}
        original = specs._OPERATION_AUTHORING_RECIPES[identity]
        for null_field in ("value", "default"):
            starter = self.resolve(selection["pipelineClass"], selection["task"], selection["executionProfileId"])
            guider = next(n for n in starter["nodes"] if n["operation"]["operationId"] == "diffusion.guidance")
            guider["params"]["enabled"] = {"type": "bool", null_field: None}
            with self.subTest(null_field=null_field), patch.dict(
                specs._OPERATION_AUTHORING_RECIPES,
                {identity: {**original, "initialValues": (("guider", "enabled", None),)}},
            ):
                recipe = specs.resolve_studio_operation_authoring_recipe(MODULE_MAP, selection, starter)
                self.assertEqual(recipe["initialValues"], [("guider", "enabled", None)])
                guider["params"]["enabled"]["required"] = True
                with self.assertRaisesRegex(ValueError, "literal"):
                    specs.resolve_studio_operation_authoring_recipe(MODULE_MAP, selection, starter)

    def test_initial_literals_cannot_override_resolved_owner_selectors(self):
        import modiff.studio_execution_specs as specs

        identity = "retained-qwen-image-text-to-image:v1"
        selection = {"pipelineClass": "QwenImageModularPipeline", "task": "text_to_image",
                     "executionProfileId": "qwen-image:modular", "authoringRecipeId": identity}
        original = specs._OPERATION_AUTHORING_RECIPES[identity]
        for field in ("reviewed_variant", "declared_owner_selector"):
            starter = self.resolve(selection["pipelineClass"], selection["task"], selection["executionProfileId"])
            owner = next(n for n in starter["nodes"] if n["operation"]["operationId"] == "diffusion.load_models")
            owner["params"][field] = {"type": "string", "value": "selected", "default": "selected"}
            owner["operation"]["binding"]["values"][field] = "selected"
            with self.subTest(field=field), patch.dict(
                specs._OPERATION_AUTHORING_RECIPES,
                {identity: {**original, "initialValues": (("models", field, "different"),)}},
            ), self.assertRaisesRegex(ValueError, "literal"):
                specs.resolve_studio_operation_authoring_recipe(MODULE_MAP, selection, starter)


class CosmosImageStageTests(unittest.TestCase):
    def test_official_sdk_image_action_output_has_no_image_side_effect(self):
        from diffusers.modular_pipelines.cosmos.after_decode import Cosmos3ActionOutputStep
        from diffusers.modular_pipelines.modular_pipeline import PipelineState

        image = object()
        state = PipelineState(values={"videos": [image]})
        components, output = Cosmos3ActionOutputStep()(None, state)
        self.assertIsNone(components)
        self.assertIs(output, state)
        self.assertIsNone(output.get("action"))
        self.assertEqual(output.get("videos"), [image])

    def test_fresh_image_stages_omit_action_output_only(self):
        from modiff.modular_whole_workflow_contracts import reviewed_whole_workflow_graph_adapter

        image = reviewed_whole_workflow_graph_adapter("Cosmos3OmniModularPipeline", "text2image")
        self.assertEqual(image["upstreamBlockSequence"], ["text_encoder", "denoise", "decode"])
        for task in ("text2video", "image2video", "video2video", "text2video_with_sound",
                     "image2video_with_sound", "video2video_with_sound", "action_policy",
                     "action_forward_dynamics", "action_inverse_dynamics"):
            with self.subTest(task=task):
                adapter = reviewed_whole_workflow_graph_adapter("Cosmos3OmniModularPipeline", task)
                self.assertEqual(adapter["upstreamBlockSequence"][-1], "after_decode")
        spec = studio_execution_spec_for_pair("Cosmos3OmniModularPipeline", "text_to_image")
        self.assertEqual([role for role, _key, _x, _y in spec["roles"]],
                         ["models", "prompt", "denoise", "decode", "preview"])
        self.assertFalse(any("afterDecode" in edge for edge in spec["edges"]))

    def test_historical_action_node_and_reviewed_upstream_workflow_remain_available(self):
        from modiff.modular_workflow_discovery import load_reviewed_modular_workflow_snapshot

        self.assertIn("WorkflowCosmos3OmniAfterDecode", MODULE_MAP["modules.ModularDiffusers"])
        entry = next(c for c in load_reviewed_modular_workflow_snapshot()["contracts"]
                     if c["pipelineClass"] == "Cosmos3OmniModularPipeline")
        image = next(w for w in entry["workflows"] if w["id"] == "text2image")
        self.assertTrue(any(block["path"] == "after_decode" for block in image["steps"]))


@pytest.mark.parametrize("identity", _PRIVATE_PROMPT_RECIPES)
def test_retained_private_prompt_receipt_requires_genuine_apis_but_new_creation_requires_explicit_rebind(identity):
    from test_ancillary_execution import require_native_parent_apis
    from modiff.ancillary_execution import resolve_parent
    from modiff.studio_execution_specs import _OPERATION_AUTHORING_RECIPES, STUDIO_EXECUTION_SPEC_DEFINITIONS

    require_native_parent_apis()
    definition = _OPERATION_AUTHORING_RECIPES[identity]
    spec = STUDIO_EXECUTION_SPEC_DEFINITIONS[definition["specId"]]
    selection = {"pipelineClass": spec["profile"]["pipeline_class"], "task": definition["task"],
                 "executionProfileId": definition["profileId"]}
    contracts, _ = build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})
    # Genuine source APIs remain required for this receipt lane. An exact old
    # observer grant must not override a source-declared new-authoring rebind.
    with patch("modiff.diffusers_evaluation_runtime.selected_diffusers_evaluation_authoring_profile_ids",
               return_value=(definition["profileId"],)), patch(
                   "modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("constructed")):
        receipt = deepcopy(resolve_parent(MODULE_MAP, identity))
        assert receipt["id"] == identity
        assert receipt["executionProfileIds"] == (definition["profileId"],)
        legacy = studio_execution_spec_for_pair(selection['pipelineClass'], definition['task'],
                                                execution_profile_id=definition['profileId'])
        assert receipt['baseSpecId'] == legacy['id']
        assert receipt['baseSpecContentHash'] == legacy['contentHash']
        before = deepcopy(selection)
        for request in (selection, {**selection, "authoringRecipeId": identity}):
            with pytest.raises(ValueError, match='Rebind explicitly'):
                resolve_operation_starter(MODULE_MAP, contracts, request)
        assert selection == before
        # Refused new creation cannot rewrite the complete retained DTO. The
        # separate source-capture lane compares this body to the old baseline.
        assert resolve_parent(MODULE_MAP, identity) == receipt


if __name__ == "__main__":
    unittest.main()
