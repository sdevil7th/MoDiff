import ast
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import modules as module_registry

from modiff.auto_resource import AUTO_MODEL_REQUIREMENTS
from modiff.diffusers_profiles import (
    CONTRACT_ONLY_DIFFUSERS_PIPELINES,
    DIFFUSERS_EXECUTION_PROFILES,
)
from modiff.model_artifact_catalog import catalog_repository_pin
from modiff.modular_workflow_contracts import PINNED_DIFFUSERS_REVISION
from modiff.optional_runtimes import (
    OPTIONAL_RUNTIME_PROFILES,
    TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID,
)
from modiff.server import STUDIO_MODEL_CAPABILITIES, WebServer
from modiff.studio_execution_specs import (
    FLUX2_KLEIN_DIFFUSERS_FILES,
    FLUX2_KLEIN_REPO,
    FLUX_KONTEXT_DIFFUSERS_FILES,
    FLUX_KONTEXT_NVFP4_REPO,
    FLUX_KONTEXT_REPO,
    QWEN_IMAGE_EDIT_2511_DIFFUSERS_FILES,
    QWEN_IMAGE_EDIT_DIFFUSERS_FILES,
    QWEN_IMAGE_EDIT_PLUS_REPO,
    QWEN_IMAGE_EDIT_REPO,
    Z_IMAGE_DIFFUSERS_FILES,
    Z_IMAGE_REPO,
    studio_execution_spec_for_pair,
    validate_studio_execution_specs,
)
from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS, LoadPipeline


PINNED_SOURCES = (
    (
        "pipelines/qwenimage/pipeline_qwenimage_edit.py",
        "QwenImageEditPipeline",
        "791798e68e9e6d0c568cf0e244f2dff5cb20b8cf913963543ea18ee2838bff0d",
        ("DiffusionPipeline", "QwenImageLoraLoaderMixin"),
        ("self", "scheduler", "vae", "text_encoder", "tokenizer", "processor", "transformer"),
        (
            "self",
            "image",
            "prompt",
            "negative_prompt",
            "true_cfg_scale",
            "height",
            "width",
            "num_inference_steps",
            "sigmas",
            "guidance_scale",
            "num_images_per_prompt",
            "generator",
            "latents",
            "prompt_embeds",
            "prompt_embeds_mask",
            "negative_prompt_embeds",
            "negative_prompt_embeds_mask",
            "output_type",
            "return_dict",
            "attention_kwargs",
            "callback_on_step_end",
            "callback_on_step_end_tensor_inputs",
            "max_sequence_length",
        ),
    ),
    (
        "pipelines/qwenimage/pipeline_qwenimage_edit_plus.py",
        "QwenImageEditPlusPipeline",
        "00962412f8fe6ec57ed6cc769e87c88d1d98990cfb0ac617bd123d51dd45b1b2",
        ("DiffusionPipeline", "QwenImageLoraLoaderMixin"),
        ("self", "scheduler", "vae", "text_encoder", "tokenizer", "processor", "transformer"),
        (
            "self",
            "image",
            "prompt",
            "negative_prompt",
            "true_cfg_scale",
            "height",
            "width",
            "num_inference_steps",
            "sigmas",
            "guidance_scale",
            "num_images_per_prompt",
            "generator",
            "latents",
            "prompt_embeds",
            "prompt_embeds_mask",
            "negative_prompt_embeds",
            "negative_prompt_embeds_mask",
            "output_type",
            "return_dict",
            "attention_kwargs",
            "callback_on_step_end",
            "callback_on_step_end_tensor_inputs",
            "max_sequence_length",
        ),
    ),
    (
        "pipelines/z_image/pipeline_z_image_inpaint.py",
        "ZImageInpaintPipeline",
        "8be6d5f773c8035bb06ff6b017907ce6c1d9fc7ef52acf99ddc23ecba3c939ba",
        ("DiffusionPipeline", "ZImageLoraLoaderMixin", "FromSingleFileMixin"),
        ("self", "scheduler", "vae", "text_encoder", "tokenizer", "transformer"),
        (
            "self",
            "prompt",
            "image",
            "mask_image",
            "masked_image_latents",
            "strength",
            "height",
            "width",
            "num_inference_steps",
            "sigmas",
            "guidance_scale",
            "cfg_normalization",
            "cfg_truncation",
            "negative_prompt",
            "num_images_per_prompt",
            "generator",
            "latents",
            "prompt_embeds",
            "negative_prompt_embeds",
            "output_type",
            "return_dict",
            "joint_attention_kwargs",
            "callback_on_step_end",
            "callback_on_step_end_tensor_inputs",
            "max_sequence_length",
        ),
    ),
    (
        "pipelines/flux/pipeline_flux_kontext_inpaint.py",
        "FluxKontextInpaintPipeline",
        "6f0d50ae6b3931dfe94ecec772aca77b77f0cc0aa7520256ad55a38ef7d731a2",
        (
            "DiffusionPipeline",
            "FluxLoraLoaderMixin",
            "FromSingleFileMixin",
            "TextualInversionLoaderMixin",
            "FluxIPAdapterMixin",
        ),
        (
            "self",
            "scheduler",
            "vae",
            "text_encoder",
            "tokenizer",
            "text_encoder_2",
            "tokenizer_2",
            "transformer",
            "image_encoder",
            "feature_extractor",
        ),
        (
            "self",
            "image",
            "image_reference",
            "mask_image",
            "prompt",
            "prompt_2",
            "negative_prompt",
            "negative_prompt_2",
            "true_cfg_scale",
            "height",
            "width",
            "strength",
            "padding_mask_crop",
            "num_inference_steps",
            "sigmas",
            "guidance_scale",
            "num_images_per_prompt",
            "generator",
            "latents",
            "prompt_embeds",
            "pooled_prompt_embeds",
            "ip_adapter_image",
            "ip_adapter_image_embeds",
            "negative_ip_adapter_image",
            "negative_ip_adapter_image_embeds",
            "negative_prompt_embeds",
            "negative_pooled_prompt_embeds",
            "output_type",
            "return_dict",
            "joint_attention_kwargs",
            "callback_on_step_end",
            "callback_on_step_end_tensor_inputs",
            "max_sequence_length",
            "max_area",
            "_auto_resize",
        ),
    ),
    (
        "pipelines/flux2/pipeline_flux2_klein_inpaint.py",
        "Flux2KleinInpaintPipeline",
        "58c5f93bcbe57276e37833beaaeeb43d80e48132d8842ad0271a39784b628f63",
        ("DiffusionPipeline", "Flux2LoraLoaderMixin"),
        ("self", "scheduler", "vae", "text_encoder", "tokenizer", "transformer", "is_distilled"),
        (
            "self",
            "prompt",
            "image",
            "image_reference",
            "mask_image",
            "height",
            "width",
            "padding_mask_crop",
            "strength",
            "num_inference_steps",
            "sigmas",
            "guidance_scale",
            "num_images_per_prompt",
            "generator",
            "latents",
            "prompt_embeds",
            "negative_prompt_embeds",
            "output_type",
            "return_dict",
            "attention_kwargs",
            "callback_on_step_end",
            "callback_on_step_end_tensor_inputs",
            "max_sequence_length",
            "text_encoder_out_layers",
        ),
    ),
)

DIRECT_PAIRS = {
    ("QwenImageEditPipeline", "edit_image"): (
        "qwen-image-edit:direct",
        QWEN_IMAGE_EDIT_REPO,
    ),
    ("QwenImageEditPlusPipeline", "edit_image"): (
        "qwen-image-edit-plus:direct",
        QWEN_IMAGE_EDIT_PLUS_REPO,
    ),
    ("QwenImageEditPlusPipeline", "multi_image_reference_edit"): (
        "qwen-image-edit-plus:direct",
        QWEN_IMAGE_EDIT_PLUS_REPO,
    ),
    ("ZImageInpaintPipeline", "inpaint"): ("z-image-inpaint:direct", Z_IMAGE_REPO),
    ("ZImageInpaintPipeline", "outpaint"): ("z-image-inpaint:direct", Z_IMAGE_REPO),
    ("FluxKontextInpaintPipeline", "inpaint"): (
        "flux-kontext-inpaint:direct",
        FLUX_KONTEXT_REPO,
    ),
    ("FluxKontextInpaintPipeline", "outpaint"): (
        "flux-kontext-inpaint:direct",
        FLUX_KONTEXT_REPO,
    ),
    ("Flux2KleinInpaintPipeline", "inpaint"): (
        "flux2-klein-inpaint:direct",
        FLUX2_KLEIN_REPO,
    ),
    ("Flux2KleinInpaintPipeline", "outpaint"): (
        "flux2-klein-inpaint:direct",
        FLUX2_KLEIN_REPO,
    ),
}

EXPECTED_REQUIRED_MEDIA = {
    ("QwenImageEditPipeline", "edit_image"): [("image", "referenceImages")],
    ("QwenImageEditPlusPipeline", "edit_image"): [("image", "referenceImages")],
    ("QwenImageEditPlusPipeline", "multi_image_reference_edit"): [
        ("image", "referenceImages")
    ],
    ("ZImageInpaintPipeline", "inpaint"): [
        ("image", "referenceImages"),
        ("image", "maskImage"),
    ],
    ("ZImageInpaintPipeline", "outpaint"): [("image", "referenceImages")],
    ("FluxKontextInpaintPipeline", "inpaint"): [
        ("image", "referenceImages"),
        ("image", "maskImage"),
    ],
    ("FluxKontextInpaintPipeline", "outpaint"): [("image", "referenceImages")],
    ("Flux2KleinInpaintPipeline", "inpaint"): [
        ("image", "referenceImages"),
        ("image", "maskImage"),
    ],
    ("Flux2KleinInpaintPipeline", "outpaint"): [("image", "referenceImages")],
}


def _method_parameters(class_node: ast.ClassDef, method_name: str) -> tuple[str, ...]:
    method = next(
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
    )
    return tuple(
        argument.arg
        for argument in (
            *method.args.posonlyargs,
            *method.args.args,
            *method.args.kwonlyargs,
        )
    )


class FakeRequest:
    query = {}


class DiffusersDirectImagePromotionTests(unittest.IsolatedAsyncioTestCase):
    def test_exact_pinned_sources_bases_constructors_and_calls_are_preserved(self):
        import diffusers

        self.assertEqual(
            PINNED_DIFFUSERS_REVISION,
            "90b4e34e79a86ec5e7f2437634fe95ecd2108796",
        )
        diffusers_root = Path(diffusers.__file__).resolve().parent
        for relative_path, class_name, digest, bases, init_parameters, call_parameters in PINNED_SOURCES:
            with self.subTest(pipeline_class=class_name):
                source_path = diffusers_root / relative_path
                self.assertTrue(source_path.is_file())
                self.assertEqual(hashlib.sha256(source_path.read_bytes()).hexdigest(), digest)
                tree = ast.parse(source_path.read_text(encoding="utf-8"))
                class_node = next(
                    node
                    for node in tree.body
                    if isinstance(node, ast.ClassDef) and node.name == class_name
                )
                self.assertEqual(tuple(ast.unparse(base) for base in class_node.bases), bases)
                self.assertEqual(_method_parameters(class_node, "__init__"), init_parameters)
                self.assertEqual(_method_parameters(class_node, "__call__"), call_parameters)

    def test_adapters_reuse_only_exact_immutable_safe_artifacts(self):
        expected_adapters = {
            "QwenImageEditPipeline": (
                ("edit_image",),
                QWEN_IMAGE_EDIT_REPO,
                (),
                frozenset({"ovedrive/qwen-image-edit-4bit"}),
                1,
            ),
            "QwenImageEditPlusPipeline": (
                ("edit_image", "multi_image_reference_edit"),
                QWEN_IMAGE_EDIT_PLUS_REPO,
                (),
                frozenset(),
                8,
            ),
            "ZImageInpaintPipeline": (
                ("inpaint", "outpaint"),
                Z_IMAGE_REPO,
                ("ZImagePipeline", "ZImageInpaintPipeline"),
                frozenset(),
                1,
            ),
            "FluxKontextInpaintPipeline": (
                ("inpaint", "outpaint"),
                FLUX_KONTEXT_REPO,
                ("FluxKontextPipeline", "FluxKontextInpaintPipeline"),
                frozenset({FLUX_KONTEXT_NVFP4_REPO}),
                1,
            ),
            "Flux2KleinInpaintPipeline": (
                ("inpaint", "outpaint"),
                FLUX2_KLEIN_REPO,
                ("Flux2KleinPipeline", "Flux2KleinInpaintPipeline"),
                frozenset(),
                1,
            ),
        }
        for pipeline_class, expected in expected_adapters.items():
            with self.subTest(pipeline_class=pipeline_class):
                adapter = IMAGE_PIPELINE_ADAPTERS[pipeline_class]
                actual = (
                    adapter.mode_options,
                    adapter.default_repo,
                    adapter.artifact_pipeline_classes,
                    adapter.compatible_repos,
                    adapter.max_reference_images,
                )
                self.assertEqual(actual, expected)
                self.assertEqual(adapter.load_pipeline_class, pipeline_class)
                self.assertTrue(adapter.safe_serialization_required)

        pins = {
            QWEN_IMAGE_EDIT_REPO: ("ac7f9318f633fc4b5778c59367c8128225f1e3de", "apache-2.0"),
            "ovedrive/qwen-image-edit-4bit": (
                "f1895ae1de2ce005a4c96576a900778ec99899de",
                "apache-2.0",
            ),
            QWEN_IMAGE_EDIT_PLUS_REPO: (
                "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9",
                "apache-2.0",
            ),
            Z_IMAGE_REPO: ("f332072aa78be7aecdf3ee76d5c247082da564a6", "apache-2.0"),
            FLUX_KONTEXT_REPO: ("24e9dedc4ef646698dc8eb4e18ae2cec3c9fea0d", "other"),
            FLUX_KONTEXT_NVFP4_REPO: (
                "7e9dee453a3454251216a394697a6dcea44f54f8",
                "other",
            ),
            FLUX2_KLEIN_REPO: ("e7b7dc27f91deacad38e78976d1f2b499d76a294", "apache-2.0"),
        }
        for repository, (revision, license_name) in pins.items():
            with self.subTest(repository=repository):
                pin = catalog_repository_pin(repository)
                self.assertEqual(pin["revision"], revision)
                self.assertEqual(pin["license"], license_name)

        unsafe_suffixes = (".bin", ".ckpt", ".pt", ".pth", ".py")
        for files in (
            QWEN_IMAGE_EDIT_DIFFUSERS_FILES,
            QWEN_IMAGE_EDIT_2511_DIFFUSERS_FILES,
            Z_IMAGE_DIFFUSERS_FILES,
            FLUX_KONTEXT_DIFFUSERS_FILES,
            FLUX2_KLEIN_DIFFUSERS_FILES,
        ):
            self.assertFalse(any(path.endswith(unsafe_suffixes) for path in files))

    def test_all_nine_load_routes_are_pinned_safetensors_only_and_never_download(self):
        for (pipeline_class, mode), (_profile_id, repository) in DIRECT_PAIRS.items():
            with self.subTest(pipeline_class=pipeline_class, mode=mode):
                calls = []

                class FakePipeline:
                    @classmethod
                    def from_pretrained(cls, model_id, **kwargs):
                        calls.append((model_id, kwargs))
                        return cls()

                loader = LoadPipeline(f"direct-promotion-{pipeline_class}-{mode}")
                loader.progress = Mock()
                loader.mm_add = Mock()
                loader.diffusers_loading_progress = lambda: nullcontext()
                with (
                    patch(
                        "modules.DiffusersImage.main.pipeline_class_from_name",
                        return_value=FakePipeline,
                    ) as class_lookup,
                    patch(
                        "modules.DiffusersImage.main.apply_pipeline_offload",
                        return_value=SimpleNamespace(method="test"),
                    ),
                    patch(
                        "modules.DiffusersRuntime.main.apply_execution_recipe_to_pipeline",
                        return_value={},
                    ),
                    patch(
                        "huggingface_hub.hf_hub_download",
                        side_effect=AssertionError("download attempted"),
                    ),
                    patch(
                        "huggingface_hub.snapshot_download",
                        side_effect=AssertionError("download attempted"),
                    ),
                ):
                    result = loader.execute(
                        pipeline_class=pipeline_class,
                        mode=mode,
                        model_id={"source": "hub", "value": repository},
                        revision="",
                        dtype="bfloat16",
                        device="cpu",
                        quantization_mode="none",
                        quantized_components=[],
                        auto_offload=False,
                        offload_mode="none",
                    )

                class_lookup.assert_called_once_with(pipeline_class)
                self.assertEqual(len(calls), 1)
                loaded_repository, load_kwargs = calls[0]
                self.assertEqual(loaded_repository, repository)
                self.assertEqual(
                    load_kwargs["revision"],
                    catalog_repository_pin(repository)["revision"],
                )
                self.assertIs(load_kwargs["use_safetensors"], True)
                self.assertIs(load_kwargs["local_files_only"], True)
                self.assertNotIn("trust_remote_code", load_kwargs)
                pipeline = result["pipeline"]
                self.assertEqual(pipeline._modiff_image_pipeline_class, pipeline_class)
                self.assertEqual(pipeline._modiff_image_mode, mode)
                self.assertEqual(pipeline._modiff_image_revision, load_kwargs["revision"])

    async def test_exact_profiles_capabilities_specs_and_hidden_task_contracts_are_additive(self):
        specs = validate_studio_execution_specs(module_registry.MODULE_MAP)
        spec_by_pair = {(item["modelType"], item["mode"]): item for item in specs}
        self.assertTrue(set(DIRECT_PAIRS).issubset(spec_by_pair))

        for pair, (profile_id, repository) in DIRECT_PAIRS.items():
            with self.subTest(pair=pair):
                model_type, mode = pair
                specification = spec_by_pair[pair]
                profile = DIFFUSERS_EXECUTION_PROFILES[profile_id]
                capability = STUDIO_MODEL_CAPABILITIES[model_type]
                self.assertEqual(specification["pipelineClass"], model_type)
                self.assertEqual(specification["executionProfileId"], profile_id)
                self.assertEqual(specification["defaultRepo"], repository)
                self.assertEqual(profile.pipeline_class, model_type)
                self.assertEqual(profile.default_repo, repository)
                self.assertFalse(profile.live_proof)
                self.assertEqual(capability["executionStatus"], "expert_only")
                self.assertEqual(capability["qualifiedModes"], [])
                self.assertFalse(capability["autoEligible"])
                self.assertTrue(capability["templateEligible"])
                self.assertFalse(capability["galleryEligible"])
                self.assertFalse(capability["liveProof"])
                self.assertNotIn(model_type, AUTO_MODEL_REQUIREMENTS)

                roles = {role: node_key for role, node_key, _x, _y in specification["roles"]}
                if mode in {"edit_image", "multi_image_reference_edit"}:
                    self.assertEqual(roles["diffusersImageEdit"], "modules.DiffusersImage.Edit")
                    bound_action_fields = {
                        param
                        for role, param, _source in specification["bindings"]
                        if role == "diffusersImageEdit"
                    }
                    self.assertNotIn("strength", bound_action_fields)
                    self.assertNotIn("reference_strength", bound_action_fields)
                else:
                    self.assertEqual(
                        roles["diffusersImageInpaint"],
                        "modules.DiffusersImage.Inpaint",
                    )
                    if mode == "outpaint":
                        self.assertEqual(
                            roles["outpaintCanvas"],
                            "modules.DiffusersImage.OutpaintCanvas",
                        )
                        self.assertNotIn("loadMask", roles)

        klein = studio_execution_spec_for_pair("Flux2KleinInpaintPipeline", "inpaint")
        klein_fields = {
            param
            for role, param, _source in klein["bindings"]
            if role == "diffusersImageInpaint"
        }
        self.assertNotIn("negative_prompt", klein_fields)
        self.assertNotIn("reference_strength", klein_fields)

        # Existing Modular/canonical pairs remain byte-for-byte addressable by
        # their prior identities; the direct classes are separate Expert pairs.
        preserved_pairs = {
            ("QwenImageEditModularPipeline", "edit_image"): "QwenImageEditModularPipeline",
            (
                "QwenImageEditPlusModularPipeline",
                "multi_image_reference_edit",
            ): "QwenImageEditPlusModularPipeline",
            ("ZImageModularPipeline", "edit_image"): "ZImageImg2ImgPipeline",
            ("FluxKontextPipeline", "edit_image"): "FluxKontextPipeline",
            ("Flux2KleinPipeline", "text_to_image"): "Flux2KleinPipeline",
        }
        for pair, pipeline_class in preserved_pairs.items():
            self.assertEqual(studio_execution_spec_for_pair(*pair)["pipelineClass"], pipeline_class)

        promoted_classes = {model_type for model_type, _mode in DIRECT_PAIRS}
        self.assertTrue(
            promoted_classes.isdisjoint(
                pipeline_class
                for pipeline_class, _media_kind, _repository, _modes in CONTRACT_ONLY_DIFFUSERS_PIPELINES
            )
        )
        runtime_profile = OPTIONAL_RUNTIME_PROFILES[TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID]
        self.assertTrue(promoted_classes.issubset(runtime_profile.required_diffusers_symbols))
        self.assertTrue(promoted_classes.issubset(runtime_profile.pipeline_adapter_symbols))

        response = await WebServer(module_registry.MODULE_MAP).model_capabilities(FakeRequest())
        payload = json.loads(response.text)
        contracts = {
            (contract["modelType"], contract["mode"]): contract
            for contract in payload["taskTemplateContracts"]
        }
        for pair, expected_media in EXPECTED_REQUIRED_MEDIA.items():
            with self.subTest(task_contract=pair):
                contract = contracts[pair]
                self.assertEqual(
                    [(item["kind"], item["field"]) for item in contract["requiredMedia"]],
                    expected_media,
                )
                self.assertEqual(contract["mediaKind"], "image")
                self.assertFalse(contract["galleryEligible"])


if __name__ == "__main__":
    unittest.main()
