import gc
import importlib.util
import json
import pickle
import unittest
import weakref
from copy import deepcopy
from unittest.mock import Mock, patch

import diffusers
import numpy as np
import torch
from PIL import Image

from modiff.NodeBase import deep_equal
from modiff.modular_workflow_contracts import PINNED_MODULAR_WORKFLOW_TRUTH
from modules.ModularDiffusers.controlnet import Controlnet
from modules.ModularDiffusers.denoise import Denoise
from modules.ModularDiffusers.latents import DecodeLatents, ImageEncode
from modules.ModularDiffusers.loaders import AutoModelLoader, ModelsLoader, annotate_modular_loader_outputs
from modules.ModularDiffusers.modular_utils import (
    get_model_type_metadata,
    pipeline_class_from_runtime_inputs,
)
from modules.ModularDiffusers.route_state import (
    ROUTE_STATE_INPUT,
    ROUTE_STATE_OUTPUT,
    bind_loader_outputs,
    bind_standalone_component_output,
    consume_controlnet_input_route_state,
    consume_decode_route_state,
    consume_denoise_route_state,
    consume_encoder_route_state,
    issue_controlnet_route_state,
    issue_decode_route_state,
    issue_encoder_route_state,
    issue_normal_decode_route_state,
    issue_pipeline_instance_token,
    issue_sdxl_ip_adapter_bundle,
    issue_standalone_component_issuer,
    reject_route_reserved_inputs,
    require_component_binding,
    require_matching_token_bearers,
    require_route_state_current_publication,
    require_sdxl_ip_adapter_bundle,
    require_standalone_component_binding,
    validate_controlnet_input_route_state,
    validate_controlnet_route_state,
    validate_denoise_route_state,
    validate_encoder_route_state,
)


QWEN_IMAGE = "QwenImageModularPipeline"
QWEN_EDIT = "QwenImageEditModularPipeline"
QWEN_EDIT_PLUS = "QwenImageEditPlusModularPipeline"
SDXL = "StableDiffusionXLModularPipeline"


class _FixtureSdxlVae:
    def __init__(self, *, latent_channels=4, scale_depth=4):
        self.config = type(
            "FixtureVaeConfig",
            (),
            {
                "latent_channels": latent_channels,
                "block_out_channels": [32] * scale_depth,
            },
        )()


def _sdxl_encoder_route(*, seed=7, crop=False, advance_generator=True):
    token, outputs = _bound_outputs(SDXL)
    vae = _FixtureSdxlVae()
    image_latents = torch.full((1, 4, 8, 8), float(seed % 17))
    mask = torch.zeros((1, 1, 8, 8))
    masked_image_latents = torch.ones((1, 4, 8, 8))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if advance_generator:
        torch.rand((), generator=generator)
    original_image = Image.new("RGB", (64, 64), "red")
    original_mask = Image.new("L", (64, 64), 255)
    route = issue_encoder_route_state(
        binding=token,
        seed=seed,
        generator=generator,
        image_latents=image_latents,
        mask=mask,
        masked_image_latents=masked_image_latents,
        padding_mask_crop=(0 if crop else None),
        crops_coords=((4, 5, 60, 61) if crop else None),
        original_image=(original_image if crop else None),
        original_mask=(original_mask if crop else None),
        vae_component=vae,
        vae_latent_channels=4,
        vae_scale_factor=8,
    )
    return {
        "token": token,
        "outputs": outputs,
        "vae": vae,
        "route": route,
        "image_latents": image_latents,
        "mask": mask,
        "masked_image_latents": masked_image_latents,
        "generator": generator,
        "original_image": original_image,
        "original_mask": original_mask,
    }


def _standalone_identity(
    *,
    repo_id="fixture/component",
    revision="a" * 40,
    subfolder=None,
    class_name="FixtureControlNetModel",
    fingerprint="1" * 64,
):
    return "hub", repo_id, revision, subfolder, class_name, fingerprint


def _standalone_payload(identity, *, manager_model_id="controlnet-resident"):
    repo_source, repo_id, revision, _subfolder, class_name, _fingerprint = identity
    return {
        "model_id": manager_model_id,
        "class_name": class_name,
        "repo_id": repo_id,
        "repo_source": repo_source,
        "revision": revision,
        "trust_remote_code": False,
    }


def _publish_standalone(
    identity=None,
    *,
    issuer=None,
    manager_model_id="controlnet-resident",
    component_kind="controlnet",
):
    identity = identity or _standalone_identity()
    issuer = issuer or issue_standalone_component_issuer()
    payload = _standalone_payload(identity, manager_model_id=manager_model_id)
    bind_standalone_component_output(
        payload,
        issuer=issuer,
        component_kind=component_kind,
        reviewed_identity=identity,
    )
    return issuer, payload


def _bound_outputs(model_type=QWEN_EDIT, *, suffix="a", model_id="shared-model"):
    token = issue_pipeline_instance_token(
        model_type=model_type,
        repo_id=f"fixture/{model_type}",
        repo_source="hub",
        revision=(suffix[0] * 40),
    )
    outputs = {
        "unet_out": {"model_id": model_id},
        "vae_out": {"model_id": model_id},
        "text_encoders": {"text_encoder": {"model_id": model_id}},
        "scheduler": {"model_id": model_id},
    }
    annotate_modular_loader_outputs(
        outputs,
        repo_id=f"fixture/{model_type}",
        repo_source="hub",
        model_type=model_type,
        revision=(suffix[0] * 40),
        trust_remote_code=False,
        pipeline_instance_token=token,
    )
    return token, outputs


def _normal_encoder_route(token, *, seed=7, advance_generator=True):
    image_latents = torch.full((1, 1, 2, 2), float(seed % 17))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if advance_generator:
        torch.rand((), generator=generator)
    return (
        issue_encoder_route_state(
            binding=token,
            seed=seed,
            generator=generator,
            image_latents=image_latents,
            processed_mask_image=None,
            mask_overlay_kwargs=None,
        ),
        image_latents,
    )


def _sdxl_ip_adapter_fixture(*, suffix="a", scale=0.75):
    from diffusers import ClassifierFreeGuidance
    from diffusers.models import ImageProjection
    from diffusers.models.attention_processor import IPAdapterAttnProcessor
    class FixtureEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type(
                "FixtureIPAdapterEncoderConfig",
                (),
                {
                    "image_size": 224,
                    "hidden_size": 1280,
                    "patch_size": 14,
                    "num_channels": 3,
                    "num_hidden_layers": 32,
                    "num_attention_heads": 16,
                    "projection_dim": 1024,
                },
            )()

    class FixtureUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dtype = torch.float32
            self.encoder_hid_proj = type("FixtureProjection", (), {})()
            self.encoder_hid_proj.image_projection_layers = torch.nn.ModuleList(
                [ImageProjection(image_embed_dim=1024, cross_attention_dim=8, num_image_text_embeds=4)]
            )
            self.attn_processors = {
                "down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor": IPAdapterAttnProcessor(
                    hidden_size=8,
                    cross_attention_dim=8,
                    num_tokens=(4,),
                    scale=scale,
                )
            }
            self.config = type("FixtureIPAdapterUNetConfig", (), {"encoder_hid_dim_type": "ip_image_proj"})()

    class FixtureCLIPImageProcessor:
        size = {"shortest_edge": 224}
        crop_size = {"height": 224, "width": 224}
        do_convert_rgb = True
        do_resize = True
        do_rescale = True
        rescale_factor = 1 / 255
        do_normalize = True
        do_center_crop = True
        image_mean = [0.48145466, 0.4578275, 0.40821073]
        image_std = [0.26862954, 0.26130258, 0.27577711]

    token, outputs = _bound_outputs(SDXL, suffix=suffix, model_id=f"ip-unet-{suffix}")
    unet = FixtureUNet()
    encoder = FixtureEncoder()
    processor = FixtureCLIPImageProcessor()
    guider = ClassifierFreeGuidance(guidance_scale=7.5)
    image = Image.new("RGB", (32, 24), "purple")
    embeddings = [torch.zeros((1, 1, 1024))]
    negative_embeddings = [torch.ones((1, 1, 1024))]
    bundle = issue_sdxl_ip_adapter_bundle(
        binding=token,
        unet=unet,
        artifact_identity=(
            "h94/IP-Adapter",
            "0" * 40,
            "sdxl_models/ip-adapter_sdxl.safetensors",
            "1" * 64,
            1,
            "models/image_encoder",
            "CLIPVisionModelWithProjection",
        ),
        image_encoder=encoder,
        feature_extractor=processor,
        guider=guider,
        scale=scale,
        image=image,
        ip_adapter_embeds=embeddings,
        negative_ip_adapter_embeds=negative_embeddings,
    )
    return {
        "token": token,
        "outputs": outputs,
        "unet": unet,
        "encoder": encoder,
        "processor": processor,
        "guider": guider,
        "image": image,
        "embeddings": embeddings,
        "negative_embeddings": negative_embeddings,
        "bundle": bundle,
        "scale": scale,
    }


def _route_node_config(*, route=True, control_bundle=False):
    inputs = ["embeddings", "image_latents", "seed"]
    if control_bundle:
        inputs.append("controlnet_bundle")
    if route:
        inputs.append(ROUTE_STATE_INPUT)
    return {
        "params": {
            "seed": {"type": "int", "min": 0, "max": 4294967295},
            "embeddings": {"type": "embeddings"},
            "image_latents": {"type": "latents"},
            **({"controlnet_bundle": {"type": "custom_controlnet"}} if control_bundle else {}),
            **({ROUTE_STATE_INPUT: {"type": "modular_route_state"}} if route else {}),
        },
        "model_input_names": ["unet", "scheduler"],
        "input_names": inputs,
        "output_names": ["latents", *([ROUTE_STATE_OUTPUT] if route else [])],
    }


def _controlnet_node_config():
    return {
        "params": {
            "control_image": {"type": "image"},
            "controlnet_conditioning_scale": {"type": "float", "min": 0.0, "max": 1.0},
            "control_guidance_start": {"type": "float", "min": 0.0, "max": 1.0},
            "control_guidance_end": {"type": "float", "min": 0.0, "max": 1.0},
            "height": {"type": "int", "min": 64, "max": 2048},
            "width": {"type": "int", "min": 64, "max": 2048},
            "seed": {"type": "int", "min": 0, "max": 4294967295},
            ROUTE_STATE_INPUT: {"type": "modular_route_state"},
        },
        "model_input_names": ["controlnet", "vae"],
        "input_names": [
            "control_image",
            "controlnet_conditioning_scale",
            "control_guidance_start",
            "control_guidance_end",
            "height",
            "width",
            "seed",
            ROUTE_STATE_INPUT,
        ],
        "output_names": ["controlnet_bundle", ROUTE_STATE_OUTPUT],
    }


class OpaqueBindingTests(unittest.TestCase):
    def test_binding_key_and_token_survive_deepcopy_but_reject_json_and_pickle(self):
        token, outputs = _bound_outputs()
        copied = deepcopy(outputs["unet_out"])

        self.assertTrue(deep_equal(outputs["unet_out"], copied))
        self.assertIs(require_component_binding(copied, label="copy"), token)
        with self.assertRaises(TypeError):
            json.dumps(outputs["unet_out"])
        with self.assertRaises(TypeError):
            pickle.dumps(token)
        with self.assertRaises(TypeError):
            pickle.dumps(outputs["unet_out"])

    def test_one_loader_execution_shares_one_token_and_reexecution_rotates_it(self):
        first_token, first = _bound_outputs(suffix="a")
        for payload in first.values():
            self.assertIs(require_component_binding(payload, label="loader output"), first_token)

        second_token = issue_pipeline_instance_token(
            model_type=QWEN_EDIT,
            repo_id=f"fixture/{QWEN_EDIT}",
            repo_source="hub",
            revision="a" * 40,
        )
        annotate_modular_loader_outputs(
            first,
            repo_id=f"fixture/{QWEN_EDIT}",
            repo_source="hub",
            model_type=QWEN_EDIT,
            revision="a" * 40,
            trust_remote_code=False,
            pipeline_instance_token=second_token,
        )
        self.assertIsNot(first_token, second_token)
        self.assertIs(require_component_binding(first["vae_out"], label="reloaded VAE"), second_token)

    def test_separate_loaders_stay_distinct_even_for_the_same_resident_component_id(self):
        token_a, outputs_a = _bound_outputs(suffix="a", model_id="resident-shared")
        token_b, outputs_b = _bound_outputs(suffix="b", model_id="resident-shared")

        self.assertIsNot(token_a, token_b)
        with self.assertRaisesRegex(ValueError, "different Models Loader"):
            require_component_binding(
                outputs_b["scheduler"],
                label="scheduler",
                expected_token=token_a,
            )

    def test_component_metadata_tampering_and_unissued_tokens_are_rejected(self):
        token, outputs = _bound_outputs()
        outputs["vae_out"]["repo_id"] = "attacker/repository"
        with self.assertRaisesRegex(ValueError, "does not match"):
            require_component_binding(outputs["vae_out"], label="VAE")

        forged = object.__new__(type(token))
        with self.assertRaisesRegex(ValueError, "invalid pipeline instance"):
            bind_loader_outputs({"vae_out": {}}, forged)

    def test_component_role_swaps_and_post_publication_model_id_mutation_are_rejected(self):
        _token, outputs = _bound_outputs()
        with self.assertRaisesRegex(ValueError, "loader role 'scheduler'.*not 'vae'"):
            require_component_binding(
                outputs["scheduler"],
                label="VAE",
                expected_role="vae",
            )

        copied_unet = deepcopy(outputs["unet_out"])
        copied_unet["model_id"] = "foreign-model-id"
        with self.assertRaisesRegex(ValueError, "model identity changed"):
            require_component_binding(
                copied_unet,
                label="denoise model",
                expected_role="denoiser",
            )

    def test_prepare_for_workflow_reuse_preserves_resident_output_binding(self):
        token, outputs = _bound_outputs()
        node = ModelsLoader("resident-binding-adoption")

        node.prepare_for_workflow_reuse()

        self.assertIs(require_component_binding(outputs["unet_out"], label="resident model"), token)

    def test_failed_loader_preflight_cannot_mint_or_publish_a_token(self):
        node = ModelsLoader()
        with patch("modules.ModularDiffusers.loaders.issue_pipeline_instance_token") as issuer:
            with self.assertRaisesRegex(ValueError, "repository code is disabled"):
                node.execute(
                    model_type=QWEN_EDIT,
                    repo_id={"source": "hub", "value": "fixture/model"},
                    device="cpu",
                    dtype=torch.float32,
                    trust_remote_code=True,
                )
        issuer.assert_not_called()


class StandaloneComponentBindingTests(unittest.TestCase):
    def test_binding_is_opaque_deepcopy_stable_unpickleable_and_weakly_registered(self):
        issuer, payload = _publish_standalone()
        binding = require_standalone_component_binding(
            payload,
            label="ControlNet",
            expected_kind="controlnet",
            expected_issuer=issuer,
            expected_reviewed_identity=_standalone_identity(),
        )
        copied = deepcopy(payload)

        self.assertTrue(deep_equal(payload, copied))
        self.assertIs(
            require_standalone_component_binding(copied, label="copied ControlNet"),
            binding,
        )
        self.assertIs(deepcopy(binding), binding)
        with self.assertRaises(AttributeError):
            binding._repo_id = "attacker/component"
        with self.assertRaises(TypeError):
            json.dumps(payload)
        for value in (issuer, binding, payload):
            with self.subTest(value=type(value).__name__), self.assertRaises(TypeError):
                pickle.dumps(value)
        del value

        binding_ref = weakref.ref(binding)
        del binding
        del copied
        del payload
        del issuer
        gc.collect()
        self.assertIsNone(binding_ref())

    def test_strict_helper_rejects_tampering_forgery_and_cross_publication_copies(self):
        identity_a = _standalone_identity()
        issuer_a, payload_a = _publish_standalone(identity_a)
        binding_a = require_standalone_component_binding(payload_a, label="ControlNet A")

        tampered_values = {
            "model_id": "foreign-manager-id",
            "class_name": "DifferentControlNetModel",
            "repo_id": "attacker/component",
            "repo_source": "local",
            "revision": "b" * 40,
            "trust_remote_code": True,
        }
        for field, value in tampered_values.items():
            tampered = deepcopy(payload_a)
            tampered[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "provenance binding"):
                require_standalone_component_binding(tampered, label="tampered ControlNet")

        local_identity = ("local", "C:\\models\\controlnet", None, None, "FixtureControlNetModel", "1" * 64)
        local_issuer, local_payload = _publish_standalone(
            local_identity,
            manager_model_id="local-controlnet-resident",
        )
        del local_payload["revision"]
        with self.assertRaisesRegex(ValueError, "provenance binding"):
            require_standalone_component_binding(
                local_payload,
                label="local ControlNet without revision field",
                expected_issuer=local_issuer,
            )

        with self.assertRaisesRegex(ValueError, "not 'vae'"):
            require_standalone_component_binding(payload_a, label="ControlNet A", expected_kind="vae")
        with self.assertRaisesRegex(ValueError, "different Load Model node"):
            require_standalone_component_binding(
                payload_a,
                label="ControlNet A",
                expected_issuer=issue_standalone_component_issuer(),
            )
        changed_identity = (*identity_a[:-1], "2" * 64)
        with self.assertRaisesRegex(ValueError, "reviewed component identity"):
            require_standalone_component_binding(
                payload_a,
                label="ControlNet A",
                expected_reviewed_identity=changed_identity,
            )
        changed_subfolder_identity = (*identity_a[:3], "controlnet", *identity_a[4:])
        with self.assertRaisesRegex(ValueError, "reviewed component identity"):
            require_standalone_component_binding(
                payload_a,
                label="ControlNet A",
                expected_reviewed_identity=changed_subfolder_identity,
            )

        binding_key = next(key for key in payload_a if type(key) is not str)
        forged = deepcopy(payload_a)
        forged[binding_key] = object.__new__(type(binding_a))
        with self.assertRaisesRegex(ValueError, "missing its process-local"):
            require_standalone_component_binding(forged, label="forged ControlNet")

        identity_b = _standalone_identity(
            repo_id="fixture/other-component",
            revision="b" * 40,
            class_name="OtherControlNetModel",
            fingerprint="2" * 64,
        )
        _issuer_b, payload_b = _publish_standalone(
            identity_b,
            manager_model_id="other-controlnet-resident",
        )
        binding_b = require_standalone_component_binding(payload_b, label="ControlNet B")
        with self.assertRaisesRegex(ValueError, "different component publication"):
            require_standalone_component_binding(
                payload_a,
                label="ControlNet A",
                expected_binding=binding_b,
            )
        cross_publication = deepcopy(payload_b)
        cross_publication[binding_key] = binding_a
        with self.assertRaisesRegex(ValueError, "provenance binding"):
            require_standalone_component_binding(cross_publication, label="cross-publication ControlNet")

    def test_new_publication_revokes_old_payload_even_while_it_remains_resident(self):
        issuer = issue_standalone_component_issuer()
        identity_a = _standalone_identity(class_name="FixtureControlNetModel", fingerprint="1" * 64)
        _issuer, payload_a = _publish_standalone(identity_a, issuer=issuer)
        binding_a = require_standalone_component_binding(payload_a, label="ControlNet A")

        identity_b = _standalone_identity(class_name="FixtureControlNetModel", fingerprint="2" * 64)
        _issuer, payload_b = _publish_standalone(
            identity_b,
            issuer=issuer,
            manager_model_id="controlnet-resident-b",
        )
        binding_b = require_standalone_component_binding(payload_b, label="ControlNet B")

        self.assertIsNot(binding_a, binding_b)
        self.assertNotEqual(payload_a["model_id"], payload_b["model_id"])
        with self.assertRaisesRegex(ValueError, "no longer the current"):
            require_standalone_component_binding(payload_a, label="resident ControlNet A")
        self.assertIs(require_standalone_component_binding(payload_b, label="ControlNet B"), binding_b)

    def test_same_manager_identity_can_republish_but_cannot_be_relabeled(self):
        identity_a = _standalone_identity()
        issuer, payload_a = _publish_standalone(
            identity_a,
            manager_model_id="same-manager-controlnet",
        )
        binding_a = require_standalone_component_binding(payload_a, label="first publication")
        payload_a2 = _standalone_payload(identity_a, manager_model_id="same-manager-controlnet")
        bind_standalone_component_output(
            payload_a2,
            issuer=issuer,
            component_kind="controlnet",
            reviewed_identity=identity_a,
        )
        binding_a2 = require_standalone_component_binding(payload_a2, label="second publication")

        self.assertIsNot(binding_a, binding_a2)
        with self.assertRaisesRegex(ValueError, "no longer the current"):
            require_standalone_component_binding(payload_a, label="first publication")

        identity_b = (*identity_a[:-1], "2" * 64)
        payload_b = _standalone_payload(identity_b, manager_model_id="same-manager-controlnet")
        with self.assertRaisesRegex(ValueError, "resident component under a different reviewed identity"):
            bind_standalone_component_output(
                payload_b,
                issuer=issuer,
                component_kind="controlnet",
                reviewed_identity=identity_b,
            )
        self.assertIs(
            require_standalone_component_binding(payload_a2, label="still-current publication"),
            binding_a2,
        )

    def test_failed_publication_mints_nothing_and_does_not_revoke_current_payload(self):
        issuer, current_payload = _publish_standalone()
        invalid_payload = _standalone_payload(_standalone_identity())
        invalid_payload["class_name"] = "UnreviewedControlNetModel"

        with patch("modules.ModularDiffusers.route_state._StandaloneComponentBinding") as binding_type:
            with self.assertRaisesRegex(ValueError, "reviewed component identity"):
                bind_standalone_component_output(
                    invalid_payload,
                    issuer=issuer,
                    component_kind="controlnet",
                    reviewed_identity=_standalone_identity(),
                )
        binding_type.assert_not_called()
        require_standalone_component_binding(current_payload, label="current ControlNet")

    def test_auto_model_execute_binds_only_after_reviewed_manager_publication(self):
        node = AutoModelLoader()
        identity = _standalone_identity(
            repo_id="fixture/transformer",
            subfolder="transformer",
            class_name="FixtureTransformerModel",
        )
        manager_payload = _standalone_payload(identity, manager_model_id="transformer-resident")

        with (
            patch(
                "modules.ModularDiffusers.loaders._preflight_reviewed_diffusers_component",
                return_value=identity,
            ),
            patch(
                "modules.ModularDiffusers.loaders._resolve_reviewed_diffusers_component_class",
                return_value=type("FixtureTransformerModel", (), {}),
            ),
            patch("modules.ModularDiffusers.loaders.ComponentSpec") as component_spec,
            patch(
                "modules.ModularDiffusers.loaders.reusable_standalone_component",
                return_value=("transformer-resident", object()),
            ),
            patch(
                "modules.ModularDiffusers.loaders.standalone_component_reuse_is_bound",
                return_value=True,
            ),
            patch("modules.ModularDiffusers.loaders.components.add", return_value="transformer-resident"),
            patch(
                "modules.ModularDiffusers.loaders.components.get_model_info",
                return_value=manager_payload,
            ),
            patch.object(node, "progress"),
        ):
            output = node.execute(
                model_type="transformer",
                model_id={"source": "hub", "value": "fixture/transformer"},
                dtype=torch.float32,
                trust_remote_code=False,
                device="cpu",
                auto_offload=False,
                offload_mode="none",
                variant=None,
                subfolder="transformer",
                revision="a" * 40,
                _reviewed_component_identity=identity,
            )

        component_spec.assert_called_once()
        require_standalone_component_binding(
            output["model"],
            label="published transformer",
            expected_kind="transformer",
            expected_issuer=node._standalone_component_issuer,
            expected_reviewed_identity=identity,
        )

        with (
            patch(
                "modules.ModularDiffusers.loaders._preflight_reviewed_diffusers_component",
                return_value=identity,
            ),
            patch(
                "modules.ModularDiffusers.loaders._resolve_reviewed_diffusers_component_class",
                return_value=type("FixtureTransformerModel", (), {}),
            ),
            patch("modules.ModularDiffusers.loaders.ComponentSpec"),
            patch(
                "modules.ModularDiffusers.loaders.reusable_standalone_component",
                return_value=("transformer-resident", object()),
            ),
            patch(
                "modules.ModularDiffusers.loaders.standalone_component_reuse_is_bound",
                return_value=True,
            ),
            patch("modules.ModularDiffusers.loaders.components.add", return_value="transformer-resident"),
            patch(
                "modules.ModularDiffusers.loaders.components.get_model_info",
                side_effect=ValueError("publication failed"),
            ),
            patch("modules.ModularDiffusers.loaders.bind_standalone_component_output") as bind_output,
            patch.object(node, "progress"),
        ):
            with self.assertRaisesRegex(ValueError, "publication failed"):
                node.execute(
                    model_type="transformer",
                    model_id={"source": "hub", "value": "fixture/transformer"},
                    dtype=torch.float32,
                    trust_remote_code=False,
                    device="cpu",
                    auto_offload=False,
                    offload_mode="none",
                    variant=None,
                    subfolder="transformer",
                    revision="a" * 40,
                    _reviewed_component_identity=identity,
                )
        bind_output.assert_not_called()

    def test_auto_model_fingerprint_change_cannot_republish_a_resident_model(self):
        node = AutoModelLoader()
        identity_a = _standalone_identity(
            repo_id="fixture/transformer",
            subfolder="transformer",
            class_name="FixtureTransformerModel",
            fingerprint="1" * 64,
        )
        resident_payload = _standalone_payload(identity_a, manager_model_id="transformer-resident-a")
        bind_standalone_component_output(
            resident_payload,
            issuer=node._standalone_component_issuer,
            component_kind="transformer",
            reviewed_identity=identity_a,
        )
        identity_b = (*identity_a[:-1], "2" * 64)
        published_payload = _standalone_payload(identity_b, manager_model_id="transformer-resident-b")
        resident_model = torch.nn.Linear(1, 1)
        loaded_model = torch.nn.Linear(1, 1)

        with (
            patch(
                "modules.ModularDiffusers.loaders._preflight_reviewed_diffusers_component",
                return_value=identity_b,
            ),
            patch(
                "modules.ModularDiffusers.loaders._resolve_reviewed_diffusers_component_class",
                return_value=type("FixtureTransformerModel", (), {}),
            ),
            patch("modules.ModularDiffusers.loaders.ComponentSpec") as component_spec,
            patch(
                "modules.ModularDiffusers.loaders.reusable_standalone_component",
                return_value=("transformer-resident-a", resident_model),
            ) as reusable,
            patch("modules.ModularDiffusers.loaders.apply_model_offload") as apply_offload,
            patch(
                "modules.ModularDiffusers.loaders.components.add",
                return_value="transformer-resident-b",
            ) as manager_add,
            patch(
                "modules.ModularDiffusers.loaders.components.get_model_info",
                return_value=published_payload,
            ),
            patch.object(node, "progress"),
            patch.object(node, "diffusers_loading_progress") as loading_progress,
        ):
            component_spec.return_value.load.return_value = loaded_model
            apply_offload.return_value = Mock(
                mode="none",
                method="fixture",
                components=["transformer"],
            )
            loading_progress.return_value.__enter__.return_value = None
            output = node.execute(
                model_type="transformer",
                model_id={"source": "hub", "value": "fixture/transformer"},
                dtype=torch.float32,
                trust_remote_code=False,
                device="cpu",
                auto_offload=False,
                offload_mode="none",
                variant=None,
                subfolder="transformer",
                revision="a" * 40,
                _reviewed_component_identity=identity_b,
            )

        reusable.assert_called_once()
        component_spec.return_value.load.assert_called_once_with(torch_dtype=torch.float32)
        manager_add.assert_called_once_with("transformer", loaded_model, collection=None)
        require_standalone_component_binding(
            output["model"],
            label="reloaded transformer B",
            expected_reviewed_identity=identity_b,
        )
        with self.assertRaisesRegex(ValueError, "no longer the current"):
            require_standalone_component_binding(resident_payload, label="resident transformer A")

    def test_auto_model_toctou_mismatch_fails_before_class_import_or_initialization(self):
        node = AutoModelLoader()
        identity_a = _standalone_identity(
            repo_id="fixture/transformer",
            subfolder="transformer",
            class_name="FixtureTransformerModel",
            fingerprint="1" * 64,
        )
        identity_b = (*identity_a[:-1], "2" * 64)
        with (
            patch(
                "modules.ModularDiffusers.loaders._preflight_reviewed_diffusers_component",
                return_value=identity_b,
            ),
            patch("modules.ModularDiffusers.loaders._resolve_reviewed_diffusers_component_class") as resolver,
            patch("modules.ModularDiffusers.loaders.ComponentSpec") as component_spec,
            patch("modules.ModularDiffusers.loaders.reusable_standalone_component") as reusable,
            patch("modules.ModularDiffusers.loaders.components.add") as manager_add,
            patch("modules.ModularDiffusers.loaders.bind_standalone_component_output") as bind_output,
        ):
            with self.assertRaisesRegex(ValueError, "config changed after cache validation"):
                node.execute(
                    model_type="transformer",
                    model_id={"source": "hub", "value": "fixture/transformer"},
                    dtype=torch.float32,
                    trust_remote_code=False,
                    device="cpu",
                    auto_offload=False,
                    offload_mode="none",
                    variant=None,
                    subfolder="transformer",
                    revision="a" * 40,
                    _reviewed_component_identity=identity_a,
                )
        resolver.assert_not_called()
        component_spec.assert_not_called()
        reusable.assert_not_called()
        manager_add.assert_not_called()
        bind_output.assert_not_called()

    def test_auto_model_cache_reuses_current_binding_rotates_on_content_and_rejects_tampering(self):
        node = AutoModelLoader("standalone-publication-cache")
        identity_a = _standalone_identity(
            repo_id="fixture/transformer",
            subfolder="transformer",
            class_name="FixtureTransformerModel",
            fingerprint="1" * 64,
        )
        identity_b = _standalone_identity(
            repo_id="fixture/transformer",
            subfolder="transformer",
            class_name="FixtureTransformerModel",
            fingerprint="2" * 64,
        )

        def publish(**kwargs):
            identity = kwargs["_reviewed_component_identity"]
            manager_model_id = "transformer-resident-a" if identity[-1] == "1" * 64 else "transformer-resident-b"
            payload = _standalone_payload(identity, manager_model_id=manager_model_id)
            bind_standalone_component_output(
                payload,
                issuer=node._standalone_component_issuer,
                component_kind="transformer",
                reviewed_identity=identity,
            )
            return {"model": payload}

        node.execute = Mock(side_effect=publish)
        inputs = {
            "model_type": "transformer",
            "model_id": {"source": "hub", "value": "fixture/transformer"},
            "dtype": "float32",
            "subfolder": "transformer",
            "variant": "",
            "trust_remote_code": False,
            "revision": "a" * 40,
            "device": "cpu",
            "auto_offload": False,
            "offload_mode": "none",
        }
        with (
            patch(
                "modules.ModularDiffusers.loaders._preflight_reviewed_diffusers_component",
                side_effect=(identity_a, identity_b, identity_b, identity_b),
            ),
            patch("modiff.NodeBase.modelstore.is_hf_cached", return_value=True),
            patch("modules.ModularDiffusers.loaders._resolve_reviewed_diffusers_component_class") as resolver,
        ):
            first = node(**inputs)
            first_binding = require_standalone_component_binding(first["model"], label="first model")
            second = node(**inputs)
            second_binding = require_standalone_component_binding(second["model"], label="second model")
            third = node(**inputs)
            self.assertIs(third, second)
            self.assertIs(
                require_standalone_component_binding(third["model"], label="cached model"),
                second_binding,
            )
            with self.assertRaisesRegex(ValueError, "no longer the current"):
                require_standalone_component_binding(first["model"], label="old resident model")

            third["model"]["repo_id"] = "attacker/component"
            with self.assertRaisesRegex(ValueError, "provenance binding"):
                node(**inputs)

        self.assertIsNot(first_binding, second_binding)
        self.assertEqual(node.execute.call_count, 2)
        resolver.assert_not_called()


class OpaqueRoutePrimitiveTests(unittest.TestCase):
    def test_route_is_identity_cached_and_nonserializable(self):
        token, _outputs = _bound_outputs()
        route, image_latents = _normal_encoder_route(token)

        self.assertIs(deepcopy(route), route)
        self.assertTrue(deep_equal(route, deepcopy(route)))
        with self.assertRaises(TypeError):
            pickle.dumps(route)
        with self.assertRaises(TypeError):
            json.dumps({"route": route})

        forged = object.__new__(type(route))
        with self.assertRaisesRegex(ValueError, "not issued"):
            validate_encoder_route_state(
                forged,
                binding=token,
                model_type=QWEN_EDIT,
                seed=7,
                image_latents=image_latents,
            )

    def test_generator_snapshot_is_post_vae_and_fresh_for_each_denoise_retry(self):
        token, _outputs = _bound_outputs()
        seed = 19
        fresh = torch.Generator(device="cpu").manual_seed(seed)
        fresh_state = fresh.get_state().clone()
        advanced = torch.Generator(device="cpu").manual_seed(seed)
        torch.rand((4,), generator=advanced)
        image_latents = torch.zeros((1, 1, 2, 2))
        route = issue_encoder_route_state(
            binding=token,
            seed=seed,
            generator=advanced,
            image_latents=image_latents,
            processed_mask_image=None,
            mask_overlay_kwargs=None,
        )

        first = consume_encoder_route_state(
            route,
            binding=token,
            model_type=QWEN_EDIT,
            seed=seed,
            execution_device="cpu",
            image_latents=image_latents,
        )["generator"]
        second = consume_encoder_route_state(
            route,
            binding=token,
            model_type=QWEN_EDIT,
            seed=seed,
            execution_device="cpu:0",
            image_latents=image_latents,
        )["generator"]

        self.assertTrue(torch.equal(first.get_state(), second.get_state()))
        self.assertFalse(torch.equal(first.get_state(), fresh_state))
        self.assertTrue(
            torch.equal(
                torch.rand((8,), generator=first),
                torch.rand((8,), generator=second),
            )
        )

    def test_sdxl_route_continues_post_vae_generator_and_exact_typed_inpaint_state(self):
        fixture = _sdxl_encoder_route(seed=19)
        first = consume_denoise_route_state(
            fixture["route"],
            binding=fixture["token"],
            model_type=SDXL,
            seed=19,
            execution_device="cpu",
            image_latents=fixture["image_latents"],
            mask=fixture["mask"],
            masked_image_latents=fixture["masked_image_latents"],
            vae_component=fixture["vae"],
            vae_latent_channels=4,
            vae_scale_factor=8,
        )
        second = consume_denoise_route_state(
            fixture["route"],
            binding=fixture["token"],
            model_type=SDXL,
            seed=19,
            execution_device="cpu:0",
            image_latents=fixture["image_latents"],
            mask=fixture["mask"],
            masked_image_latents=fixture["masked_image_latents"],
            vae_component=fixture["vae"],
            vae_latent_channels=4,
            vae_scale_factor=8,
        )

        fresh = torch.Generator(device="cpu").manual_seed(19)
        self.assertFalse(torch.equal(first["generator"].get_state(), fresh.get_state()))
        self.assertTrue(torch.equal(first["generator"].get_state(), second["generator"].get_state()))
        self.assertIs(first["mask"], fixture["mask"])
        self.assertIs(first["masked_image_latents"], fixture["masked_image_latents"])
        self.assertIsNone(first["crops_coords"])

    def test_sdxl_crop_route_snapshots_bounded_pil_media_and_materializes_once_requested(self):
        fixture = _sdxl_encoder_route(crop=True)
        fixture["original_image"].putpixel((0, 0), (0, 0, 255))
        fixture["original_mask"].putpixel((0, 0), 0)
        denoised = torch.zeros((1, 4, 8, 8))
        decode_route = issue_decode_route_state(
            fixture["route"],
            binding=fixture["token"],
            latents=denoised,
            vae_component=fixture["vae"],
        )

        validate_only = consume_decode_route_state(
            decode_route,
            binding=fixture["token"],
            model_type=SDXL,
            latents=denoised,
            vae_component=fixture["vae"],
            vae_latent_channels=4,
            vae_scale_factor=8,
            materialize_overlay=False,
        )
        materialized = consume_decode_route_state(
            decode_route,
            binding=fixture["token"],
            model_type=SDXL,
            latents=denoised,
            vae_component=fixture["vae"],
            vae_latent_channels=4,
            vae_scale_factor=8,
        )["decode_inputs"]

        self.assertIsNone(validate_only["decode_inputs"])
        self.assertEqual(materialized["padding_mask_crop"], 0)
        self.assertEqual(materialized["crops_coords"], (4, 5, 60, 61))
        self.assertEqual(materialized["image"].getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(materialized["mask_image"].getpixel((0, 0)), 255)
        self.assertIsNot(materialized["image"], fixture["original_image"])
        self.assertIsNot(materialized["mask_image"], fixture["original_mask"])

    def test_sdxl_route_accepts_exact_ordinary_controlnet_and_ip_adapter_composition(self):
        fixture = _sdxl_encoder_route()
        common = {
            "binding": fixture["token"],
            "model_type": SDXL,
            "seed": 7,
            "image_latents": fixture["image_latents"],
            "mask": fixture["mask"],
            "masked_image_latents": fixture["masked_image_latents"],
            "vae_latent_channels": 4,
            "vae_scale_factor": 8,
        }
        with self.assertRaisesRegex(ValueError, "exact component paired"):
            validate_denoise_route_state(
                fixture["route"],
                vae_component=_FixtureSdxlVae(),
                **common,
            )
        fixture["vae"].config.latent_channels = 5
        with self.assertRaisesRegex(ValueError, "four-channel|geometry changed"):
            validate_denoise_route_state(
                fixture["route"],
                vae_component=fixture["vae"],
                **common,
            )
        fixture["vae"].config.latent_channels = 4
        ordinary_identity = _standalone_identity(class_name="ControlNetModel")
        _issuer, ordinary_controlnet = _publish_standalone(ordinary_identity)
        self.assertIsNone(
            validate_denoise_route_state(
                fixture["route"],
                vae_component=fixture["vae"],
                controlnet_bundle_present=True,
                controlnet_component=ordinary_controlnet,
                **common,
            )
        )
        consumed = consume_denoise_route_state(
            fixture["route"],
            vae_component=fixture["vae"],
            controlnet_bundle_present=True,
            controlnet_component=ordinary_controlnet,
            execution_device="cpu",
            **common,
        )
        self.assertTrue(torch.equal(consumed["generator"].get_state(), fixture["generator"].get_state()))

        for values, message in (
            ({"controlnet_bundle_present": True}, "exact connected component bundle"),
            ({"controlnet_component": ordinary_controlnet}, "exact connected component bundle"),
            (
                {
                    "controlnet_bundle_present": True,
                    "controlnet_component": ordinary_controlnet,
                    "control_image_latents": torch.zeros((1, 4, 8, 8)),
                },
                "does not accept prepared Qwen",
            ),
        ):
            with self.subTest(values=tuple(values)), self.assertRaisesRegex(ValueError, message):
                validate_denoise_route_state(
                    fixture["route"],
                    vae_component=fixture["vae"],
                    **common,
                    **values,
                )
        self.assertIsNone(
            validate_denoise_route_state(
                fixture["route"],
                vae_component=fixture["vae"],
                ip_adapter_present=True,
                **common,
            )
        )

        _union_issuer, union_controlnet = _publish_standalone(
            _standalone_identity(class_name="ControlNetUnionModel"),
            manager_model_id="controlnet-union-resident",
        )
        self.assertIsNone(
            validate_denoise_route_state(
                fixture["route"],
                vae_component=fixture["vae"],
                controlnet_bundle_present=True,
                controlnet_component=union_controlnet,
                control_mode=3,
                **common,
            )
        )
        union_consumed = consume_denoise_route_state(
            fixture["route"],
            vae_component=fixture["vae"],
            controlnet_bundle_present=True,
            controlnet_component=union_controlnet,
            control_mode=3,
            execution_device="cpu",
            **common,
        )
        self.assertTrue(torch.equal(union_consumed["generator"].get_state(), fixture["generator"].get_state()))
        with self.assertRaisesRegex(ValueError, "exact ControlNetModel"):
            validate_denoise_route_state(
                fixture["route"],
                vae_component=fixture["vae"],
                controlnet_bundle_present=True,
                controlnet_component=union_controlnet,
                **common,
            )
        with self.assertRaisesRegex(ValueError, "bounded canonical integer"):
            validate_denoise_route_state(
                fixture["route"],
                vae_component=fixture["vae"],
                controlnet_bundle_present=True,
                controlnet_component=union_controlnet,
                control_mode=32,
                **common,
            )

        _replacement_issuer, replacement_controlnet = _publish_standalone(
            ordinary_identity,
            issuer=_issuer,
            manager_model_id="controlnet-resident",
        )
        with self.assertRaisesRegex(ValueError, "no longer the current"):
            validate_denoise_route_state(
                fixture["route"],
                vae_component=fixture["vae"],
                controlnet_bundle_present=True,
                controlnet_component=ordinary_controlnet,
                **common,
            )
        self.assertIsNotNone(replacement_controlnet)

        dead_fixture = _sdxl_encoder_route()
        departed = weakref.ref(dead_fixture.pop("vae"))
        gc.collect()
        self.assertIsNone(departed())
        with self.assertRaisesRegex(ValueError, "no longer resident"):
            validate_encoder_route_state(
                dead_fixture["route"],
                binding=dead_fixture["token"],
                model_type=SDXL,
                seed=7,
                image_latents=dead_fixture["image_latents"],
                mask=dead_fixture["mask"],
                masked_image_latents=dead_fixture["masked_image_latents"],
                vae_component=_FixtureSdxlVae(),
                vae_latent_channels=4,
                vae_scale_factor=8,
            )

    def test_route_tensor_version_seals_reject_in_place_encoder_and_decode_mutation(self):
        for field in ("image_latents", "mask", "masked_image_latents"):
            with self.subTest(contract="sdxl", field=field):
                fixture = _sdxl_encoder_route()
                fixture[field].add_(1)
                with self.assertRaisesRegex(ValueError, "mutated or rebound"):
                    validate_encoder_route_state(
                        fixture["route"],
                        binding=fixture["token"],
                        model_type=SDXL,
                        seed=7,
                        image_latents=fixture["image_latents"],
                        mask=fixture["mask"],
                        masked_image_latents=fixture["masked_image_latents"],
                        vae_component=fixture["vae"],
                        vae_latent_channels=4,
                        vae_scale_factor=8,
                    )

        qwen_token, _outputs = _bound_outputs(QWEN_EDIT)
        qwen_route, qwen_latents = _normal_encoder_route(qwen_token)
        qwen_latents.add_(1)
        with self.assertRaisesRegex(ValueError, "mutated or rebound"):
            validate_encoder_route_state(
                qwen_route,
                binding=qwen_token,
                model_type=QWEN_EDIT,
                seed=7,
                image_latents=qwen_latents,
            )

        fixture = _sdxl_encoder_route()
        denoised = torch.zeros((1, 4, 8, 8))
        decode_route = issue_decode_route_state(
            fixture["route"],
            binding=fixture["token"],
            latents=denoised,
            vae_component=fixture["vae"],
        )
        denoised.add_(1)
        with self.assertRaisesRegex(ValueError, "mutated or rebound"):
            consume_decode_route_state(
                decode_route,
                binding=fixture["token"],
                model_type=SDXL,
                latents=denoised,
                vae_component=fixture["vae"],
                vae_latent_channels=4,
                vae_scale_factor=8,
            )

    def test_sdxl_route_rejects_malformed_latent_structure_and_crop_geometry(self):
        token, _outputs = _bound_outputs(SDXL)
        vae = _FixtureSdxlVae()
        base = {
            "binding": token,
            "seed": 7,
            "generator": torch.Generator(device="cpu").manual_seed(7),
            "image_latents": torch.zeros((1, 4, 8, 8)),
            "mask": torch.zeros((1, 1, 8, 8)),
            "masked_image_latents": torch.zeros((1, 4, 8, 8)),
            "vae_component": vae,
            "vae_latent_channels": 4,
            "vae_scale_factor": 8,
        }
        invalid_tensors = (
            ("image_latents", torch.zeros((1, 4, 8)), "rank-4"),
            ("image_latents", torch.zeros((1, 5, 8, 8)), "channels"),
            ("mask", torch.zeros((1, 2, 8, 8)), "one channel"),
            ("mask", torch.zeros((2, 1, 8, 8)), "batch and spatial"),
            ("masked_image_latents", torch.zeros((1, 4, 7, 8)), "batch and spatial"),
            ("mask", torch.zeros((1, 1, 8, 8), dtype=torch.float64), "one exact dtype"),
            ("image_latents", torch.zeros((1, 4, 513, 513)), "decoded-pixel budget"),
        )
        for field, value, message in invalid_tensors:
            with self.subTest(field=field, shape=tuple(value.shape)), self.assertRaisesRegex(ValueError, message):
                issue_encoder_route_state(**{**base, field: value})

        with self.assertRaisesRegex(ValueError, "pinned four-channel"):
            issue_encoder_route_state(
                **{
                    **base,
                    "vae_component": _FixtureSdxlVae(latent_channels=5),
                    "vae_latent_channels": 5,
                    "image_latents": torch.zeros((1, 5, 8, 8)),
                    "masked_image_latents": torch.zeros((1, 5, 8, 8)),
                }
            )

        image = Image.new("RGB", (64, 64), "red")
        mask = Image.new("L", (64, 64), 255)
        invalid_crops = (
            ([0, 0, 8, 8], "exact tuple"),
            ((True, 0, 8, 8), "exact tuple"),
            ((0, 0, 0, 8), "nonempty region"),
            ((0, 0, 65, 8), "within the original"),
        )
        for coords, message in invalid_crops:
            with self.subTest(coords=coords), self.assertRaisesRegex(ValueError, message):
                issue_encoder_route_state(
                    **base,
                    padding_mask_crop=0,
                    crops_coords=coords,
                    original_image=image,
                    original_mask=mask,
                )

    def test_route_issuance_requires_exact_typed_latent_outputs(self):
        token, _outputs = _bound_outputs(QWEN_EDIT_PLUS)
        generator = torch.Generator(device="cpu").manual_seed(7)
        with self.assertRaisesRegex(TypeError, "VAE image latents must be an exact Torch tensor"):
            issue_encoder_route_state(
                binding=token,
                seed=7,
                generator=generator,
                image_latents=object(),
                processed_mask_image=None,
                mask_overlay_kwargs=None,
            )
        with self.assertRaisesRegex(TypeError, "Denoise latents must be an exact Torch tensor"):
            issue_normal_decode_route_state(binding=token, latents=object())

        tensor = torch.zeros((1, 1, 2, 2))

        class TensorList(list):
            pass

        class CustomTensorSequence:
            def __init__(self, values):
                self.values = values

            def __iter__(self):
                return iter(self.values)

            def __len__(self):
                return len(self.values)

        invalid_sequences = (
            (tuple([tensor]), TypeError, "exact Torch tensor or a bounded nonempty list"),
            (TensorList([tensor]), TypeError, "exact Torch tensor or a bounded nonempty list"),
            (CustomTensorSequence([tensor]), TypeError, "exact Torch tensor or a bounded nonempty list"),
            ([], ValueError, "must not be empty"),
            ([tensor, object()], TypeError, r"VAE image latents\[1\].*exact Torch tensor"),
            ([torch.zeros(()) for _index in range(65)], ValueError, "more than 64"),
        )
        for latent_value, error_type, message in invalid_sequences:
            with self.subTest(latent_value=type(latent_value).__name__), self.assertRaisesRegex(error_type, message):
                issue_encoder_route_state(
                    binding=token,
                    seed=7,
                    generator=generator,
                    image_latents=latent_value,
                    processed_mask_image=None,
                    mask_overlay_kwargs=None,
                )

    def test_multi_latent_route_preserves_list_kind_length_order_and_weak_identity(self):
        token, _outputs = _bound_outputs(QWEN_EDIT_PLUS)
        first = torch.zeros((1, 1, 2, 2))
        second = torch.ones((1, 1, 2, 2))
        route = issue_encoder_route_state(
            binding=token,
            seed=7,
            generator=torch.Generator(device="cpu").manual_seed(7),
            image_latents=[first, second],
            processed_mask_image=None,
            mask_overlay_kwargs=None,
        )

        # Rewrapping is harmless, but the exact list kind and tensor identities,
        # length, and order remain authoritative.
        validate_encoder_route_state(
            route,
            binding=token,
            model_type=QWEN_EDIT_PLUS,
            seed=7,
            image_latents=[first, second],
        )
        for connected, error_type, message in (
            ((first, second), TypeError, "bounded nonempty list"),
            ([first], ValueError, "exact latent output paired"),
            ([second, first], ValueError, "exact latent output paired"),
            ([first, second.clone()], ValueError, "exact latent output paired"),
        ):
            with (
                self.subTest(connected=type(connected).__name__, length=len(connected)),
                self.assertRaisesRegex(error_type, message),
            ):
                validate_encoder_route_state(
                    route,
                    binding=token,
                    model_type=QWEN_EDIT_PLUS,
                    seed=7,
                    image_latents=connected,
                )

        def issue_without_retaining_latents():
            transient_latents = [torch.zeros((1, 1, 2, 2)), torch.ones((1, 1, 2, 2))]
            return issue_encoder_route_state(
                binding=token,
                seed=7,
                generator=torch.Generator(device="cpu").manual_seed(7),
                image_latents=transient_latents,
                processed_mask_image=None,
                mask_overlay_kwargs=None,
            )

        dead_route = issue_without_retaining_latents()
        gc.collect()
        with self.assertRaisesRegex(ValueError, "no longer resident"):
            validate_encoder_route_state(
                dead_route,
                binding=token,
                model_type=QWEN_EDIT_PLUS,
                seed=7,
                image_latents=[torch.zeros((1, 1, 2, 2)), torch.ones((1, 1, 2, 2))],
            )

    def test_single_image_qwen_route_issuance_rejects_latent_lists(self):
        for model_type in (QWEN_IMAGE, QWEN_EDIT):
            with self.subTest(model_type=model_type):
                token, _outputs = _bound_outputs(model_type)
                with self.assertRaisesRegex(TypeError, "VAE image latents must be an exact Torch tensor"):
                    issue_encoder_route_state(
                        binding=token,
                        seed=7,
                        generator=torch.Generator(device="cpu").manual_seed(7),
                        image_latents=[torch.zeros((1, 1, 2, 2))],
                        processed_mask_image=None,
                        mask_overlay_kwargs=None,
                    )

    def test_zero_and_max_seed_work_while_noncanonical_seed_values_fail(self):
        token, _outputs = _bound_outputs()
        for seed in (0, 4294967295):
            with self.subTest(seed=seed):
                route, image_latents = _normal_encoder_route(token, seed=seed, advance_generator=False)
                validate_encoder_route_state(
                    route,
                    binding=token,
                    model_type=QWEN_EDIT,
                    seed=seed,
                    image_latents=image_latents,
                )

        route, image_latents = _normal_encoder_route(token, seed=7)
        for invalid in (True, 7.0, "07", "-0", None):
            with self.subTest(seed=invalid), self.assertRaisesRegex(ValueError, "seed"):
                validate_encoder_route_state(
                    route,
                    binding=token,
                    model_type=QWEN_EDIT,
                    seed=invalid,
                    image_latents=image_latents,
                )

    def test_route_rejects_cross_loader_cross_model_and_wrong_stage(self):
        token_a, _outputs_a = _bound_outputs(suffix="a")
        token_b, _outputs_b = _bound_outputs(suffix="b")
        route, image_latents = _normal_encoder_route(token_a)
        with self.assertRaisesRegex(ValueError, "different Models Loader"):
            validate_encoder_route_state(
                route,
                binding=token_b,
                model_type=QWEN_EDIT,
                seed=7,
                image_latents=image_latents,
            )
        with self.assertRaisesRegex(ValueError, "different pipeline"):
            validate_encoder_route_state(
                route,
                binding=token_a,
                model_type="QwenImageModularPipeline",
                seed=7,
                image_latents=image_latents,
            )
        denoised_latents = torch.zeros((1, 1, 2, 2))
        decode_route = issue_decode_route_state(
            route,
            binding=token_a,
            actual_mask=None,
            latents=denoised_latents,
        )
        with self.assertRaisesRegex(ValueError, "wrong action stage"):
            validate_encoder_route_state(
                decode_route,
                binding=token_a,
                model_type=QWEN_EDIT,
                seed=7,
                image_latents=image_latents,
            )
        with self.assertRaisesRegex(ValueError, "wrong action stage"):
            consume_decode_route_state(
                route,
                binding=token_a,
                model_type=QWEN_EDIT,
                latents=denoised_latents,
            )

    def test_inpaint_mask_and_overlay_are_an_atomic_route_contract(self):
        token, _outputs = _bound_outputs()
        generator = torch.Generator(device="cpu").manual_seed(7)
        image_latents = torch.zeros((1, 1, 2, 2))
        for processed_mask, overlay in ((object(), None), (None, {"crop": (0, 0, 8, 8)})):
            with (
                self.subTest(processed_mask=processed_mask, overlay=overlay),
                self.assertRaisesRegex(ValueError, "both processed mask and overlay"),
            ):
                issue_encoder_route_state(
                    binding=token,
                    seed=7,
                    generator=generator,
                    image_latents=image_latents,
                    processed_mask_image=processed_mask,
                    mask_overlay_kwargs=overlay,
                )

        inpaint_route = issue_encoder_route_state(
            binding=token,
            seed=7,
            generator=torch.Generator(device="cpu").manual_seed(7),
            image_latents=image_latents,
            processed_mask_image=torch.ones((1, 1, 8, 8)),
            mask_overlay_kwargs={
                "crops_coords": (0, 0, 8, 8),
                "original_image": object(),
                "original_mask": object(),
            },
        )
        denoised_latents = torch.zeros((1, 1, 2, 2))
        decode_route = issue_decode_route_state(
            inpaint_route,
            binding=token,
            actual_mask=torch.ones((1, 1, 8, 8)),
            latents=denoised_latents,
        )
        decoded = consume_decode_route_state(
            decode_route,
            binding=token,
            model_type=QWEN_EDIT,
            latents=denoised_latents,
        )
        self.assertTrue(decoded["inpaint"])
        self.assertEqual(decoded["mask_overlay_kwargs"]["crops_coords"], (0, 0, 8, 8))

        malformed_overlays = (
            ({"crops_coords": None, "original_image": None}, "pinned Qwen contract"),
            (
                {"crops_coords": None, "original_image": object(), "original_mask": None},
                "require non-null",
            ),
            (
                {"crops_coords": [0, 0, 8, 8], "original_image": object(), "original_mask": object()},
                "four integers",
            ),
        )
        for overlay, message in malformed_overlays:
            with self.subTest(overlay=overlay), self.assertRaisesRegex(ValueError, message):
                issue_encoder_route_state(
                    binding=token,
                    seed=7,
                    generator=torch.Generator(device="cpu").manual_seed(7),
                    image_latents=image_latents,
                    processed_mask_image=torch.ones((1, 1, 8, 8)),
                    mask_overlay_kwargs=overlay,
                )
        with self.assertRaisesRegex(TypeError, "processed Modular inpaint mask"):
            issue_encoder_route_state(
                binding=token,
                seed=7,
                generator=torch.Generator(device="cpu").manual_seed(7),
                image_latents=image_latents,
                processed_mask_image=object(),
                mask_overlay_kwargs={
                    "crops_coords": None,
                    "original_image": None,
                    "original_mask": None,
                },
            )
        with self.assertRaisesRegex(TypeError, "Denoise inpaint mask"):
            issue_decode_route_state(
                inpaint_route,
                binding=token,
                actual_mask=object(),
                latents=denoised_latents,
            )

    def test_controlnet_route_seals_post_control_generator_and_retries_from_fresh_clones(self):
        token, _outputs = _bound_outputs(QWEN_IMAGE)
        _issuer, controlnet = _publish_standalone(manager_model_id="controlnet-generator-route")
        seed = 29
        generator = torch.Generator(device="cpu").manual_seed(seed)
        torch.rand((5,), generator=generator)
        post_control_state = generator.get_state().clone()
        control_latents = torch.zeros((1, 1, 2, 2))
        route = issue_controlnet_route_state(
            None,
            binding=token,
            controlnet_component=controlnet,
            seed=seed,
            generator=generator,
            control_image_latents=control_latents,
        )

        first = consume_denoise_route_state(
            route,
            binding=token,
            model_type=QWEN_IMAGE,
            seed=seed,
            execution_device="cpu",
            image_latents=None,
            control_image_latents=control_latents,
            controlnet_component=controlnet,
        )["generator"]
        second = consume_denoise_route_state(
            route,
            binding=token,
            model_type=QWEN_IMAGE,
            seed=seed,
            execution_device="cpu:0",
            image_latents=None,
            control_image_latents=control_latents,
            controlnet_component=controlnet,
        )["generator"]

        self.assertTrue(torch.equal(first.get_state(), post_control_state))
        self.assertTrue(torch.equal(first.get_state(), second.get_state()))
        self.assertTrue(torch.equal(torch.rand((8,), generator=first), torch.rand((8,), generator=second)))

    def test_controlnet_stage_carries_image_pairing_and_inpaint_state_without_a_large_input(self):
        token, _outputs = _bound_outputs(QWEN_IMAGE)
        _issuer, controlnet = _publish_standalone(manager_model_id="controlnet-image-route")
        seed = 31
        image_latents = torch.zeros((1, 1, 2, 2))
        processed_mask = torch.ones((1, 1, 8, 8))
        overlay = {
            "crops_coords": None,
            "original_image": None,
            "original_mask": None,
        }
        encoder_generator = torch.Generator(device="cpu").manual_seed(seed)
        torch.rand((3,), generator=encoder_generator)
        encoder_route = issue_encoder_route_state(
            binding=token,
            seed=seed,
            generator=encoder_generator,
            image_latents=image_latents,
            processed_mask_image=processed_mask,
            mask_overlay_kwargs=overlay,
        )

        first_control_generator = consume_controlnet_input_route_state(
            encoder_route,
            binding=token,
            model_type=QWEN_IMAGE,
            seed=seed,
            execution_device="cpu",
        )
        retry_control_generator = consume_controlnet_input_route_state(
            encoder_route,
            binding=token,
            model_type=QWEN_IMAGE,
            seed=seed,
            execution_device="cpu",
        )
        self.assertTrue(torch.equal(first_control_generator.get_state(), retry_control_generator.get_state()))
        torch.rand((4,), generator=first_control_generator)
        control_latents = torch.ones((1, 1, 2, 2))
        control_route = issue_controlnet_route_state(
            encoder_route,
            binding=token,
            controlnet_component=controlnet,
            seed=seed,
            generator=first_control_generator,
            control_image_latents=control_latents,
        )

        consumed = consume_denoise_route_state(
            control_route,
            binding=token,
            model_type=QWEN_IMAGE,
            seed=seed,
            execution_device="cpu",
            image_latents=image_latents,
            control_image_latents=control_latents,
            controlnet_component=controlnet,
        )
        self.assertIs(consumed["processed_mask_image"], processed_mask)
        with self.assertRaisesRegex(ValueError, "exact latent output paired"):
            validate_denoise_route_state(
                control_route,
                binding=token,
                model_type=QWEN_IMAGE,
                seed=seed,
                image_latents=image_latents.clone(),
                control_image_latents=control_latents,
                controlnet_component=controlnet,
            )
        with self.assertRaisesRegex(ValueError, "exact latent output paired"):
            validate_denoise_route_state(
                control_route,
                binding=token,
                model_type=QWEN_IMAGE,
                seed=seed,
                image_latents=image_latents,
                control_image_latents=control_latents.clone(),
                controlnet_component=controlnet,
            )

        denoised_latents = torch.full((1, 1, 2, 2), 2.0)
        decode_route = issue_decode_route_state(
            control_route,
            binding=token,
            actual_mask=processed_mask,
            latents=denoised_latents,
        )
        decoded = consume_decode_route_state(
            decode_route,
            binding=token,
            model_type=QWEN_IMAGE,
            latents=denoised_latents,
        )
        self.assertTrue(decoded["inpaint"])
        self.assertEqual(decoded["mask_overlay_kwargs"], overlay)

    def test_controlnet_latent_carrier_is_exact_bounded_and_weak(self):
        token, _outputs = _bound_outputs(QWEN_IMAGE)
        _issuer, controlnet = _publish_standalone(manager_model_id="controlnet-list-route")
        generator = torch.Generator(device="cpu").manual_seed(7)
        first = torch.zeros((1, 1, 2, 2))
        second = torch.ones((1, 1, 2, 2))
        route = issue_controlnet_route_state(
            None,
            binding=token,
            controlnet_component=controlnet,
            seed=7,
            generator=generator,
            control_image_latents=[first, second],
        )
        validate_controlnet_route_state(
            route,
            binding=token,
            model_type=QWEN_IMAGE,
            seed=7,
            image_latents=None,
            control_image_latents=[first, second],
            controlnet_component=controlnet,
        )

        class TensorList(list):
            pass

        class CustomTensorSequence:
            def __iter__(self):
                return iter((first,))

            def __len__(self):
                return 1

        invalid_issuance = (
            ((first,), TypeError, "bounded nonempty list"),
            (TensorList([first]), TypeError, "bounded nonempty list"),
            (CustomTensorSequence(), TypeError, "bounded nonempty list"),
            ([], ValueError, "must not be empty"),
            ([first, object()], TypeError, r"ControlNet image latents\[1\].*exact Torch tensor"),
            ([torch.zeros(()) for _index in range(65)], ValueError, "more than 64"),
        )
        for value, error_type, message in invalid_issuance:
            with self.subTest(value=type(value).__name__), self.assertRaisesRegex(error_type, message):
                issue_controlnet_route_state(
                    None,
                    binding=token,
                    controlnet_component=controlnet,
                    seed=7,
                    generator=generator,
                    control_image_latents=value,
                )

        for connected in ([second, first], [first], [first, second.clone()]):
            with self.subTest(length=len(connected)), self.assertRaisesRegex(ValueError, "exact latent output paired"):
                validate_controlnet_route_state(
                    route,
                    binding=token,
                    model_type=QWEN_IMAGE,
                    seed=7,
                    image_latents=None,
                    control_image_latents=connected,
                    controlnet_component=controlnet,
                )

        def issue_without_retaining_control_latents():
            transient = torch.zeros((1, 1, 2, 2))
            return issue_controlnet_route_state(
                None,
                binding=token,
                controlnet_component=controlnet,
                seed=7,
                generator=generator,
                control_image_latents=transient,
            )

        dead_route = issue_without_retaining_control_latents()
        gc.collect()
        with self.assertRaisesRegex(ValueError, "no longer resident"):
            validate_controlnet_route_state(
                dead_route,
                binding=token,
                model_type=QWEN_IMAGE,
                seed=7,
                image_latents=None,
                control_image_latents=torch.zeros((1, 1, 2, 2)),
                controlnet_component=controlnet,
            )

    def test_controlnet_route_requires_exact_current_controlnet_kind_and_publication(self):
        token, _outputs = _bound_outputs(QWEN_IMAGE)
        identity_a = _standalone_identity(fingerprint="a" * 64)
        issuer, controlnet_a = _publish_standalone(
            identity_a,
            manager_model_id="controlnet-revocation-route",
        )
        control_latents = torch.zeros((1, 1, 2, 2))
        route = issue_controlnet_route_state(
            None,
            binding=token,
            controlnet_component=controlnet_a,
            seed=7,
            generator=torch.Generator(device="cpu").manual_seed(7),
            control_image_latents=control_latents,
        )
        _other_issuer, wrong_kind = _publish_standalone(
            manager_model_id="standalone-vae-route",
            component_kind="vae",
        )
        with self.assertRaisesRegex(ValueError, "not 'controlnet'"):
            issue_controlnet_route_state(
                None,
                binding=token,
                controlnet_component=wrong_kind,
                seed=7,
                generator=torch.Generator(device="cpu").manual_seed(7),
                control_image_latents=control_latents,
            )

        _issuer, _controlnet_b = _publish_standalone(
            identity_a,
            issuer=issuer,
            manager_model_id="controlnet-revocation-route",
        )
        with self.assertRaisesRegex(ValueError, "superseded"):
            require_route_state_current_publication(route, label="ControlNet route")
        with self.assertRaisesRegex(ValueError, "superseded"):
            validate_controlnet_route_state(
                route,
                binding=token,
                model_type=QWEN_IMAGE,
                seed=7,
                image_latents=None,
                control_image_latents=control_latents,
                controlnet_component=controlnet_a,
            )

    def test_controlnet_input_route_rejects_dead_inherited_latents_without_receiving_typed_edge(self):
        token, _outputs = _bound_outputs(QWEN_IMAGE)

        def issue_without_retaining_image_latents():
            transient = torch.zeros((1, 1, 2, 2))
            return issue_encoder_route_state(
                binding=token,
                seed=7,
                generator=torch.Generator(device="cpu").manual_seed(7),
                image_latents=transient,
                processed_mask_image=None,
                mask_overlay_kwargs=None,
            )

        route = issue_without_retaining_image_latents()
        gc.collect()
        with self.assertRaisesRegex(ValueError, "no longer resident"):
            validate_controlnet_input_route_state(
                route,
                binding=token,
                model_type=QWEN_IMAGE,
                seed=7,
            )

    def test_nested_token_scan_is_bounded_and_cycle_safe(self):
        token, _outputs = _bound_outputs()
        cyclic = {}
        cyclic["self"] = cyclic
        require_matching_token_bearers(cyclic, token, label="cyclic bundle")

        nested = {}
        cursor = nested
        for _index in range(40):
            cursor["next"] = {}
            cursor = cursor["next"]
        with self.assertRaisesRegex(ValueError, "safe nested-value limit"):
            require_matching_token_bearers(nested, token, label="deep bundle")

    def test_reserved_route_values_cannot_be_overwritten_by_a_bundle(self):
        for field in ("generator", "processed_mask_image", "mask_overlay_kwargs", "mask"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "cannot overwrite"):
                reject_route_reserved_inputs(
                    {"embeddings": {field: object()}},
                    bundle_names=["embeddings"],
                )


class SDXLIPAdapterReceiptTests(unittest.TestCase):
    def test_exact_backend_publication_validates_and_cannot_serialize(self):
        fixture = _sdxl_ip_adapter_fixture()
        self.assertIs(
            require_sdxl_ip_adapter_bundle(
                fixture["bundle"],
                binding=fixture["token"],
                unet=fixture["unet"],
                guider=fixture["guider"],
            )._binding,
            fixture["token"],
        )
        with self.assertRaises(TypeError):
            json.dumps(fixture["bundle"])
        with self.assertRaises(TypeError):
            pickle.dumps(fixture["bundle"])

    def test_wrong_loader_unet_guider_or_embedding_identity_fails_closed(self):
        fixture = _sdxl_ip_adapter_fixture()
        other = _sdxl_ip_adapter_fixture(suffix="b")
        cases = (
            ({"binding": other["token"]}, "different Models Loader"),
            ({"unet": other["unet"]}, "different resident UNet"),
            ({"guider": other["guider"]}, "exact same Guider"),
        )
        for override, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                require_sdxl_ip_adapter_bundle(
                    fixture["bundle"],
                    binding=override.get("binding", fixture["token"]),
                    unet=override.get("unet", fixture["unet"]),
                    guider=override.get("guider", fixture["guider"]),
                )

        fixture["bundle"]["ip_adapter_embeds"] = [torch.zeros((1, 1, 1024))]
        with self.assertRaisesRegex(ValueError, "changed after backend publication"):
            require_sdxl_ip_adapter_bundle(
                fixture["bundle"],
                binding=fixture["token"],
                unet=fixture["unet"],
                guider=fixture["guider"],
            )

    def test_unet_image_and_latest_publication_tampering_fails_closed(self):
        fixture = _sdxl_ip_adapter_fixture()
        with torch.no_grad():
            next(iter(fixture["unet"].encoder_hid_proj.image_projection_layers[0].parameters())).add_(1)
        with self.assertRaisesRegex(ValueError, "UNet state changed"):
            require_sdxl_ip_adapter_bundle(
                fixture["bundle"],
                binding=fixture["token"],
                unet=fixture["unet"],
                guider=fixture["guider"],
            )

        fixture = _sdxl_ip_adapter_fixture()
        fixture["image"].putpixel((0, 0), (1, 2, 3))
        with self.assertRaisesRegex(ValueError, "source image changed"):
            require_sdxl_ip_adapter_bundle(
                fixture["bundle"],
                binding=fixture["token"],
                unet=fixture["unet"],
                guider=fixture["guider"],
            )

        fixture = _sdxl_ip_adapter_fixture()
        replacement = issue_sdxl_ip_adapter_bundle(
            binding=fixture["token"],
            unet=fixture["unet"],
            artifact_identity=("h94/IP-Adapter", "0" * 40, "weight", "2" * 64, 1, "encoder", "class"),
            image_encoder=fixture["encoder"],
            feature_extractor=fixture["processor"],
            guider=fixture["guider"],
            scale=fixture["scale"],
            image=fixture["image"],
            ip_adapter_embeds=fixture["embeddings"],
            negative_ip_adapter_embeds=fixture["negative_embeddings"],
        )
        with self.assertRaisesRegex(ValueError, "no longer the current"):
            require_sdxl_ip_adapter_bundle(
                fixture["bundle"],
                binding=fixture["token"],
                unet=fixture["unet"],
                guider=fixture["guider"],
            )
        self.assertIsNotNone(
            require_sdxl_ip_adapter_bundle(
                replacement,
                binding=fixture["token"],
                unet=fixture["unet"],
                guider=fixture["guider"],
            )
        )

    def test_resident_adapter_requires_its_exact_bundle(self):
        fixture = _sdxl_ip_adapter_fixture()
        with self.assertRaisesRegex(ValueError, "no matching adapter bundle"):
            require_sdxl_ip_adapter_bundle(
                None,
                binding=fixture["token"],
                unet=fixture["unet"],
                guider=fixture["guider"],
            )


class RouteRuntimeBoundaryTests(unittest.TestCase):
    def test_sdxl_crop_media_preflight_rejects_untrusted_or_oversize_inputs_before_init(self):
        _token, outputs = _bound_outputs(SDXL)
        init_pipeline = Mock()
        blocks = type(
            "SdxlEncodeBlocks",
            (),
            {
                "input_names": ["image", "mask_image", "padding_mask_crop", "generator"],
                "component_names": ["vae"],
                "init_pipeline": init_pipeline,
            },
        )()
        config = {
            "params": {
                "image": {"type": "image"},
                "mask_image": {"type": "image"},
                "padding_mask_crop": {"type": "int", "min": 0, "max": 8192},
                "seed": {"type": "int", "min": 0, "max": 4294967295},
            },
            "model_input_names": ["vae"],
            "input_names": ["image", "mask_image", "padding_mask_crop", "seed"],
            "output_names": ["image_latents", "mask", "masked_image_latents", ROUTE_STATE_OUTPUT],
        }
        valid_image = Image.new("RGB", (64, 64), "red")
        valid_mask = Image.new("L", (64, 64), 255)
        invalid_cases = (
            (torch.zeros((1, 3, 64, 64)), valid_mask, "must be a PIL image"),
            (np.zeros((64, 64, 3), dtype=np.uint8), valid_mask, "must be a PIL image"),
            ([valid_image], valid_mask, "must be a PIL image"),
            (valid_image, None, "requires a mask image"),
            (valid_image, Image.new("L", (32, 64), 255), "dimensions must match"),
            (Image.new("L", (8193, 1), 0), Image.new("L", (8193, 1), 0), "no larger than 8192"),
            (
                Image.new("L", (4097, 2048), 0),
                Image.new("L", (4097, 2048), 0),
                "16-Mi-pixel",
            ),
        )
        node = ImageEncode("sdxl-crop-preflight")
        node.progress = Mock()
        with (
            patch(
                "modules.ModularDiffusers.latents.pipeline_class_from_runtime_inputs",
                return_value=diffusers.StableDiffusionXLModularPipeline,
            ),
            patch(
                "modules.ModularDiffusers.latents.require_modiff_node_contract",
                return_value=(blocks, config),
            ),
        ):
            for image, mask_image, message in invalid_cases:
                with self.subTest(message=message), self.assertRaisesRegex((TypeError, ValueError), message):
                    node.execute(
                        vae=outputs["vae_out"],
                        image=image,
                        mask_image=mask_image,
                        padding_mask_crop=0,
                        seed=7,
                    )

        init_pipeline.assert_not_called()

    def test_sdxl_cache_hits_revalidate_exact_resident_vae_and_geometry_for_all_three_actions(self):
        fixture = _sdxl_encoder_route()
        model_id = fixture["outputs"]["vae_out"]["model_id"]
        resident = fixture["vae"]
        resident_unet = type("ResidentUnet", (), {})()
        resident_state = {model_id: resident}

        def manager_result(*, ids, return_dict_with_names=True):
            if return_dict_with_names:
                return {"unet": resident_unet, "vae": resident_state[model_id], "scheduler": object()}
            return {requested: resident_state[requested] for requested in ids}

        encoder = ImageEncode("sdxl-encoder-cache-provenance")
        encoder._pipeline_class = diffusers.StableDiffusionXLModularPipeline
        encoder._model_type = SDXL
        encoder._pipeline = type("ResidentEncodePipeline", (), {"vae": resident})()
        encoder.output = {
            "image_latents": fixture["image_latents"],
            "mask": fixture["mask"],
            "masked_image_latents": fixture["masked_image_latents"],
            ROUTE_STATE_OUTPUT: fixture["route"],
        }
        encode_params = {"vae": fixture["outputs"]["vae_out"], "seed": 7}

        denoised = torch.zeros((1, 4, 8, 8))
        normal_route = issue_normal_decode_route_state(
            binding=fixture["token"],
            latents=denoised,
            vae_component=resident,
            vae_latent_channels=4,
            vae_scale_factor=8,
        )
        denoise = Denoise("sdxl-denoise-cache-provenance")
        denoise._pipeline_class = diffusers.StableDiffusionXLModularPipeline
        denoise._model_type = SDXL
        denoise._pipeline = type("ResidentDenoisePipeline", (), {"unet": resident_unet, "vae": resident})()
        denoise._route_cache_node_input_names = ("seed", ROUTE_STATE_INPUT)
        denoise._route_cache_block_input_names = ("generator",)
        denoise._route_cache_component_names = ("unet", "vae", "scheduler")
        denoise._route_cache_model_input_names = ("unet", "vae", "scheduler")
        denoise.output = {"latents": denoised, ROUTE_STATE_OUTPUT: normal_route}
        denoise_params = {
            "unet": fixture["outputs"]["unet_out"],
            "vae": fixture["outputs"]["vae_out"],
            "scheduler": fixture["outputs"]["scheduler"],
            "seed": 7,
        }

        decode = DecodeLatents("sdxl-decode-cache-provenance")
        decode._pipeline_class = diffusers.StableDiffusionXLModularPipeline
        decode._model_type = SDXL
        decode._pipeline = type("ResidentDecodePipeline", (), {"vae": resident})()
        decode_params = {
            "vae": fixture["outputs"]["vae_out"],
            "latents": denoised,
            ROUTE_STATE_INPUT: normal_route,
        }
        decode_blocks = type("DecodeBlocks", (), {"input_names": ["latents"]})()
        decode_config = {
            "params": {ROUTE_STATE_INPUT: {"type": "modular_route_state"}},
            "model_input_names": ["vae"],
            "input_names": ["latents", ROUTE_STATE_INPUT],
            "output_names": ["images"],
        }

        with (
            patch(
                "modules.ModularDiffusers.latents.components.get_components_by_ids",
                side_effect=manager_result,
            ),
            patch(
                "modules.ModularDiffusers.denoise.components.get_components_by_ids",
                side_effect=manager_result,
            ),
            patch(
                "modules.ModularDiffusers.latents.require_modiff_node_contract",
                return_value=(decode_blocks, decode_config),
            ),
        ):
            self.assertTrue(encoder._cache_params_equal(encode_params, encode_params))
            string_seed_params = {**encode_params, "seed": "7"}
            self.assertTrue(encoder._cache_params_equal(string_seed_params, string_seed_params))
            for invalid_seed in (True, 7.5, "07"):
                invalid_params = {**encode_params, "seed": invalid_seed}
                with self.subTest(invalid_seed=invalid_seed), self.assertRaisesRegex(
                    ValueError,
                    "Invalid Modular Diffusers seed",
                ):
                    encoder._cache_params_equal(invalid_params, invalid_params)
            self.assertTrue(denoise._cache_params_equal(denoise_params, denoise_params))
            self.assertTrue(decode._cache_params_equal(decode_params, decode_params))

            resident.config.latent_channels = 5
            for action, check, params in (
                ("encoder", encoder._cache_params_equal, encode_params),
                ("denoise", denoise._cache_params_equal, denoise_params),
                ("decode", decode._cache_params_equal, decode_params),
            ):
                with self.subTest(action=action), self.assertRaisesRegex(ValueError, "four-channel|geometry changed"):
                    check(params, params)

            resident.config.latent_channels = 4
            replacement = _FixtureSdxlVae()
            resident_state[model_id] = replacement
            with self.assertRaisesRegex(ValueError, "resident Modular pipeline"):
                encoder._cache_params_equal(encode_params, encode_params)
            with self.assertRaisesRegex(ValueError, "resident SDXL Denoise pipeline"):
                denoise._cache_params_equal(denoise_params, denoise_params)
            with self.assertRaisesRegex(ValueError, "resident Modular pipeline"):
                decode._cache_params_equal(decode_params, decode_params)

    def test_sdxl_base_inpaint_fake_actions_preserve_exact_state_and_overlay_semantics(self):
        _token, outputs = _bound_outputs(SDXL)
        vae = _FixtureSdxlVae()
        unet_component = object()
        scheduler_component = object()
        trace = []
        captured = {}
        image_latents = torch.zeros((1, 4, 8, 8))
        mask = torch.ones((1, 1, 8, 8))
        masked_image_latents = torch.full((1, 4, 8, 8), 2.0)
        denoised = torch.full((1, 4, 8, 8), 3.0)

        class FakeEncodePipeline:
            _execution_device = torch.device("cpu")
            blocks = type("PipelineBlocks", (), {"doc": "encode"})()

            def __init__(self):
                self.vae = None

            def update_components(self, **values):
                for name, value in values.items():
                    setattr(self, name, value)

            def __call__(self, **kwargs):
                trace.append("vae_encoder")
                captured["encode"] = dict(kwargs)
                torch.rand((3,), generator=kwargs["generator"])
                captured["post_vae_generator"] = kwargs["generator"].get_state().clone()
                return {
                    "image_latents": image_latents,
                    "mask": mask,
                    "masked_image_latents": masked_image_latents,
                    "crops_coords": (4, 5, 60, 61),
                }

        class FakeEncodeBlocks:
            component_names = ["vae"]
            input_names = ["image", "mask_image", "padding_mask_crop", "generator"]

            @staticmethod
            def init_pipeline(*, components_manager):
                return FakeEncodePipeline()

        encode_config = {
            "params": {
                "image": {"type": "image"},
                "mask_image": {"type": "image"},
                "padding_mask_crop": {"type": "int", "min": 0, "max": 8192},
                "seed": {"type": "int", "min": 0, "max": 4294967295},
            },
            "model_input_names": ["vae"],
            "input_names": ["image", "mask_image", "padding_mask_crop", "seed"],
            "output_names": ["image_latents", "mask", "masked_image_latents", ROUTE_STATE_OUTPUT],
        }

        class FakeDenoisePipeline:
            _execution_device = torch.device("cpu")
            component_names = ["unet", "vae", "scheduler"]
            blocks = type("PipelineBlocks", (), {"doc": "denoise"})()
            transformer = None

            def __init__(self):
                self.unet = None
                self.vae = None
                self.scheduler = None

            def update_components(self, **values):
                for name, value in values.items():
                    setattr(self, name, value)

            def __call__(self, **kwargs):
                trace.append("denoise")
                captured["denoise"] = dict(kwargs)
                captured["pre_denoise_generator"] = kwargs["generator"].get_state().clone()
                return {"latents": denoised}

        class FakeDenoiseBlocks:
            component_names = ["unet", "vae", "scheduler"]
            input_names = [
                "prompt_embeds",
                "image_latents",
                "mask",
                "masked_image_latents",
                "crops_coords",
                "generator",
            ]

            @staticmethod
            def init_pipeline(*, components_manager):
                return FakeDenoisePipeline()

        denoise_config = {
            "params": {
                "embeddings": {"type": "embeddings"},
                "image_latents": {"type": "latents"},
                "mask": {"type": "latent_mask"},
                "masked_image_latents": {"type": "masked_latents"},
                "seed": {"type": "int", "min": 0, "max": 4294967295},
                ROUTE_STATE_INPUT: {"type": "modular_route_state"},
            },
            "model_input_names": ["unet", "vae", "scheduler"],
            "input_names": [
                "embeddings",
                "image_latents",
                "mask",
                "masked_image_latents",
                "seed",
                ROUTE_STATE_INPUT,
            ],
            "output_names": ["latents", ROUTE_STATE_OUTPUT],
        }

        class FakeDecodePipeline:
            blocks = type("PipelineBlocks", (), {"doc": "decode"})()

            def __init__(self):
                self.vae = None

            def update_components(self, **values):
                for name, value in values.items():
                    setattr(self, name, value)

            def __call__(self, **kwargs):
                trace.append("decoder")
                captured["decode"] = dict(kwargs)
                return {"images": "decoded-sdxl-inpaint"}

        class FakeDecodeBlocks:
            component_names = ["vae"]
            input_names = ["latents", "image", "mask_image", "padding_mask_crop", "crops_coords"]

            @staticmethod
            def init_pipeline(*, components_manager):
                return FakeDecodePipeline()

        decode_config = {
            "params": {ROUTE_STATE_INPUT: {"type": "modular_route_state"}},
            "model_input_names": ["vae"],
            "input_names": ["latents", ROUTE_STATE_INPUT],
            "output_names": ["images"],
        }

        def managed_components(*, ids, return_dict_with_names=True):
            if return_dict_with_names:
                return {
                    "unet": unet_component,
                    "vae": vae,
                    "scheduler": scheduler_component,
                }
            return {model_id: vae for model_id in set(ids)}

        original_image = Image.new("RGB", (64, 64), "red")
        original_mask = Image.new("L", (64, 64), 255)
        with (
            patch(
                "modules.ModularDiffusers.latents.components.get_components_by_ids",
                side_effect=managed_components,
            ),
            patch("modules.ModularDiffusers.latents.collect_model_ids", return_value=["shared-model"]),
            patch("modules.ModularDiffusers.denoise.collect_model_ids", return_value=["shared-model"]),
            patch("modules.ModularDiffusers.denoise.insert_preview_block"),
            patch(
                "modules.ModularDiffusers.latents.pipeline_class_from_runtime_inputs",
                return_value=diffusers.StableDiffusionXLModularPipeline,
            ),
            patch(
                "modules.ModularDiffusers.denoise.pipeline_class_from_runtime_inputs",
                return_value=diffusers.StableDiffusionXLModularPipeline,
            ),
        ):
            with patch(
                "modules.ModularDiffusers.latents.require_modiff_node_contract",
                return_value=(FakeEncodeBlocks(), encode_config),
            ):
                encoded = ImageEncode().execute(
                    vae=outputs["vae_out"],
                    image=original_image,
                    mask_image=original_mask,
                    padding_mask_crop=0,
                    seed=7,
                )
            with patch(
                "modules.ModularDiffusers.denoise.require_modiff_node_contract",
                return_value=(FakeDenoiseBlocks(), denoise_config),
            ):
                denoised_outputs = Denoise().execute(
                    unet=outputs["unet_out"],
                    vae=outputs["vae_out"],
                    scheduler=outputs["scheduler"],
                    embeddings={"prompt_embeds": object()},
                    image_latents=encoded["image_latents"],
                    mask=encoded["mask"],
                    masked_image_latents=encoded["masked_image_latents"],
                    seed=7,
                    route_state_in=encoded[ROUTE_STATE_OUTPUT],
                )
            with patch(
                "modules.ModularDiffusers.latents.require_modiff_node_contract",
                return_value=(FakeDecodeBlocks(), decode_config),
            ):
                decoded = DecodeLatents().execute(
                    vae=outputs["vae_out"],
                    latents=denoised_outputs["latents"],
                    route_state_in=denoised_outputs[ROUTE_STATE_OUTPUT],
                )

        self.assertEqual(trace, ["vae_encoder", "denoise", "decoder"])
        self.assertTrue(torch.equal(captured["post_vae_generator"], captured["pre_denoise_generator"]))
        self.assertIs(captured["denoise"]["image_latents"], image_latents)
        self.assertIs(captured["denoise"]["mask"], mask)
        self.assertIs(captured["denoise"]["masked_image_latents"], masked_image_latents)
        self.assertEqual(captured["denoise"]["crops_coords"], (4, 5, 60, 61))
        self.assertEqual(captured["denoise"]["output"], ["latents"])
        self.assertNotIn("state", captured["decode"])
        self.assertEqual(captured["decode"]["padding_mask_crop"], 0)
        self.assertEqual(captured["decode"]["crops_coords"], (4, 5, 60, 61))
        self.assertEqual(captured["decode"]["image"].getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(captured["decode"]["mask_image"].getpixel((0, 0)), 255)
        self.assertEqual(decoded["images"], "decoded-sdxl-inpaint")

    def test_sdxl_inpaint_controlnet_installs_exact_resident_component_and_closes_swaps(self):
        for case in ("ordinary", "union", "init-swap", "call-swap"):
            with self.subTest(case=case):
                union = case == "union"
                swap_phase = case.removesuffix("-swap") if case.endswith("-swap") else None
                fixture = _sdxl_encoder_route()
                outputs = fixture["outputs"]
                unet_component = object()
                scheduler_component = object()
                first_controlnet = (
                    type(
                        "FixtureControlNetUnion",
                        (),
                        {"config": type("FixtureControlNetUnionConfig", (), {"num_control_type": 8})()},
                    )()
                    if union
                    else object()
                )
                replacement_controlnet = object()
                resident = {"controlnet": first_controlnet}
                _issuer, controlnet_payload = _publish_standalone(
                    _standalone_identity(class_name="ControlNetUnionModel" if union else "ControlNetModel"),
                    manager_model_id=f"sdxl-controlnet-{case}",
                )
                denoised = torch.full((1, 4, 8, 8), 3.0)
                pipeline_calls = []
                pipelines = []

                class FakePipeline:
                    _execution_device = torch.device("cpu")
                    component_names = ["unet", "vae", "scheduler", "controlnet"]
                    blocks = type("PipelineBlocks", (), {"doc": "sdxl-controlnet-denoise"})()
                    transformer = None

                    def __init__(self):
                        self.unet = None
                        self.vae = None
                        self.scheduler = None
                        self.controlnet = None

                    def update_components(self, **values):
                        for name, value in values.items():
                            setattr(self, name, value)

                    def __call__(self, **kwargs):
                        pipeline_calls.append(dict(kwargs))
                        if swap_phase == "call":
                            resident["controlnet"] = replacement_controlnet
                        return {"latents": denoised}

                class FakeBlocks:
                    component_names = ["unet", "vae", "scheduler", "controlnet"]
                    input_names = [
                        "prompt_embeds",
                        "image_latents",
                        "mask",
                        "masked_image_latents",
                        "generator",
                        "control_mode",
                        "control_image",
                        "controlnet_conditioning_scale",
                        "control_guidance_start",
                        "control_guidance_end",
                    ]

                    @staticmethod
                    def init_pipeline(*, components_manager):
                        if swap_phase == "init":
                            resident["controlnet"] = replacement_controlnet
                        pipeline = FakePipeline()
                        pipelines.append(pipeline)
                        return pipeline

                config = {
                    "params": {
                        "embeddings": {"type": "embeddings"},
                        "image_latents": {"type": "latents"},
                        "mask": {"type": "latent_mask"},
                        "masked_image_latents": {"type": "masked_latents"},
                        "seed": {"type": "int", "min": 0, "max": 4294967295},
                        "controlnet_bundle": {"type": "custom_controlnet"},
                        ROUTE_STATE_INPUT: {"type": "modular_route_state"},
                    },
                    "model_input_names": ["unet", "vae", "scheduler", "controlnet_bundle"],
                    "input_names": [
                        "embeddings",
                        "image_latents",
                        "mask",
                        "masked_image_latents",
                        "seed",
                        "controlnet_bundle",
                        ROUTE_STATE_INPUT,
                    ],
                    "output_names": ["latents", ROUTE_STATE_OUTPUT],
                }
                controlnet_bundle = {
                    "controlnet": controlnet_payload,
                    "control_image": "control-image",
                    "controlnet_conditioning_scale": 0.75,
                    "control_guidance_start": 0.1,
                    "control_guidance_end": 0.9,
                }
                if union:
                    controlnet_bundle["control_mode"] = 3

                def managed_components(*, ids, return_dict_with_names=True):
                    if return_dict_with_names:
                        result = {
                            "unet": unet_component,
                            "vae": fixture["vae"],
                            "scheduler": scheduler_component,
                        }
                        if controlnet_payload["model_id"] in ids:
                            result["controlnet"] = resident["controlnet"]
                        return result
                    available = {
                        outputs["vae_out"]["model_id"]: fixture["vae"],
                        controlnet_payload["model_id"]: resident["controlnet"],
                    }
                    return {model_id: available[model_id] for model_id in set(ids)}

                kwargs = {
                    "unet": outputs["unet_out"],
                    "vae": outputs["vae_out"],
                    "scheduler": outputs["scheduler"],
                    "embeddings": {"prompt_embeds": object()},
                    "image_latents": fixture["image_latents"],
                    "mask": fixture["mask"],
                    "masked_image_latents": fixture["masked_image_latents"],
                    "controlnet_bundle": controlnet_bundle,
                    "seed": 7,
                    ROUTE_STATE_INPUT: fixture["route"],
                }
                denoise_node = Denoise()
                with (
                    patch(
                        "modules.ModularDiffusers.denoise.pipeline_class_from_runtime_inputs",
                        return_value=diffusers.StableDiffusionXLModularPipeline,
                    ),
                    patch(
                        "modules.ModularDiffusers.denoise.require_modiff_node_contract",
                        return_value=(FakeBlocks(), config),
                    ),
                    patch("modules.ModularDiffusers.denoise.insert_preview_block"),
                    patch(
                        "modules.ModularDiffusers.denoise.components.get_components_by_ids",
                        side_effect=managed_components,
                    ),
                ):
                    if swap_phase is not None:
                        with self.assertRaisesRegex(ValueError, "ControlNet changed|does not hold"):
                            denoise_node.execute(**kwargs)
                        continue
                    result = denoise_node.execute(**kwargs)
                    denoise_node.output = result
                    self.assertTrue(denoise_node._cache_params_equal(kwargs, kwargs))
                    if union:
                        first_controlnet.config.num_control_type = 3
                        with self.assertRaisesRegex(ValueError, "outside the resident model contract"):
                            denoise_node._cache_params_equal(kwargs, kwargs)
                        first_controlnet.config.num_control_type = 8

                self.assertIs(result["latents"], denoised)
                self.assertEqual(len(pipeline_calls), 1)
                self.assertNotIn("controlnet", pipeline_calls[0])
                self.assertIs(pipelines[0].controlnet, first_controlnet)
                if union:
                    self.assertEqual(pipeline_calls[0]["control_mode"], 3)
                else:
                    self.assertNotIn("control_mode", pipeline_calls[0])
                self.assertIs(pipeline_calls[0]["image_latents"], fixture["image_latents"])
                self.assertIs(pipeline_calls[0]["mask"], fixture["mask"])
                self.assertIs(
                    pipeline_calls[0]["masked_image_latents"],
                    fixture["masked_image_latents"],
                )
                consume_decode_route_state(
                    result[ROUTE_STATE_OUTPUT],
                    binding=fixture["token"],
                    model_type=SDXL,
                    latents=denoised,
                    vae_component=fixture["vae"],
                    vae_latent_channels=4,
                    vae_scale_factor=8,
                    materialize_overlay=False,
                )

    def test_sdxl_route_less_text_and_legacy_control_emit_normal_decode_routes_while_ip_mismatches_fail_closed(self):
        token, outputs = _bound_outputs(SDXL)
        _other_token, other_outputs = _bound_outputs(SDXL, suffix="b")
        vae = _FixtureSdxlVae()
        unet_component = object()
        scheduler_component = object()
        controlnet_component = object()
        controlnet_issuer, controlnet_payload = _publish_standalone(
            _standalone_identity(class_name="ControlNetModel")
        )
        init_calls = []
        pipeline_calls = []
        pipelines = []

        class FakePipeline:
            _execution_device = torch.device("cpu")
            component_names = ["unet", "vae", "scheduler", "controlnet"]
            blocks = type("PipelineBlocks", (), {"doc": "denoise"})()
            transformer = None

            def __init__(self):
                self.unet = None
                self.vae = None
                self.scheduler = None
                self.controlnet = None

            def update_components(self, **values):
                for name, value in values.items():
                    setattr(self, name, value)

            def __call__(self, **kwargs):
                pipeline_calls.append(dict(kwargs))
                return {"latents": torch.zeros((1, 4, 8, 8))}

        class FakeBlocks:
            component_names = ["unet", "vae", "scheduler", "controlnet"]
            input_names = [
                "prompt_embeds",
                "generator",
                "control_mode",
                "control_image",
                "controlnet_conditioning_scale",
                "control_guidance_start",
                "control_guidance_end",
            ]

            @staticmethod
            def init_pipeline(*, components_manager):
                init_calls.append(components_manager)
                pipeline = FakePipeline()
                pipelines.append(pipeline)
                return pipeline

        config = {
            "params": {
                "embeddings": {"type": "embeddings"},
                "seed": {"type": "int", "min": 0, "max": 4294967295},
                "controlnet_bundle": {"type": "custom_controlnet"},
                "ip_adapter": {"type": "custom_ip_adapter"},
                ROUTE_STATE_INPUT: {"type": "modular_route_state"},
            },
            "model_input_names": ["unet", "vae", "scheduler", "controlnet_bundle"],
            "input_names": ["embeddings", "seed", "controlnet_bundle", "ip_adapter", ROUTE_STATE_INPUT],
            "output_names": ["latents", ROUTE_STATE_OUTPUT],
        }

        def managed_components(*, ids, return_dict_with_names=True):
            if return_dict_with_names:
                result = {"unet": unet_component, "vae": vae, "scheduler": scheduler_component}
                if controlnet_payload["model_id"] in ids:
                    result["controlnet"] = controlnet_component
                return result
            available = {
                outputs["vae_out"]["model_id"]: vae,
                controlnet_payload["model_id"]: controlnet_component,
            }
            return {model_id: available[model_id] for model_id in set(ids)}

        base = {
            "unet": outputs["unet_out"],
            "vae": outputs["vae_out"],
            "scheduler": outputs["scheduler"],
            "embeddings": {"prompt_embeds": object()},
            "seed": 7,
        }
        legacy_control = {
            "controlnet": controlnet_payload,
            "control_image": "control-image",
            "controlnet_conditioning_scale": 0.75,
            "control_guidance_start": 0.1,
            "control_guidance_end": 0.9,
        }
        with (
            patch(
                "modules.ModularDiffusers.denoise.pipeline_class_from_runtime_inputs",
                return_value=diffusers.StableDiffusionXLModularPipeline,
            ),
            patch(
                "modules.ModularDiffusers.denoise.require_modiff_node_contract",
                return_value=(FakeBlocks(), config),
            ),
            patch("modules.ModularDiffusers.denoise.insert_preview_block"),
            patch(
                "modules.ModularDiffusers.denoise.components.get_components_by_ids",
                side_effect=managed_components,
            ),
        ):
            for controlled in (False, True):
                kwargs = dict(base)
                if controlled:
                    kwargs["controlnet_bundle"] = legacy_control
                result = Denoise().execute(**kwargs)
                consume_decode_route_state(
                    result[ROUTE_STATE_OUTPUT],
                    binding=token,
                    model_type=SDXL,
                    latents=result["latents"],
                    vae_component=vae,
                    vae_latent_channels=4,
                    vae_scale_factor=8,
                    materialize_overlay=False,
                )

            self.assertNotIn("controlnet", pipeline_calls[0])
            self.assertNotIn("controlnet", pipeline_calls[1])
            self.assertIsNone(pipelines[0].controlnet)
            self.assertIs(pipelines[1].controlnet, controlnet_component)
            self.assertEqual(pipeline_calls[1]["control_image"], "control-image")

            invalid_cases = (
                ({"ip_adapter": object()}, "exact backend-issued bundle"),
                ({"embeddings": {"prompt_embeds": object(), "ip_adapter_embeds": object()}}, "IP-Adapter fields"),
                (
                    {"controlnet_bundle": {**legacy_control, "control_mode": 1}},
                    "exact ControlNetUnionModel",
                ),
                ({"vae": other_outputs["vae_out"]}, "different Models Loader"),
            )
            init_before = len(init_calls)
            for override, message in invalid_cases:
                with self.subTest(override=tuple(override)), self.assertRaisesRegex(ValueError, message):
                    Denoise().execute(**{**base, **override})
            self.assertEqual(len(init_calls), init_before)

        self.assertEqual(len(pipeline_calls), 2)
        self.assertIsNotNone(controlnet_issuer)

    def test_sdxl_live_manager_vae_swap_during_init_or_call_never_publishes_route_outputs(self):
        token, outputs = _bound_outputs(SDXL)
        model_id = outputs["vae_out"]["model_id"]

        encode_config = {
            "params": {
                "image": {"type": "image"},
                "seed": {"type": "int", "min": 0, "max": 4294967295},
            },
            "model_input_names": ["vae"],
            "input_names": ["image", "seed"],
            "output_names": ["image_latents", "mask", "masked_image_latents", ROUTE_STATE_OUTPUT],
        }
        denoise_config = {
            "params": {
                "embeddings": {"type": "embeddings"},
                "seed": {"type": "int", "min": 0, "max": 4294967295},
                ROUTE_STATE_INPUT: {"type": "modular_route_state"},
            },
            "model_input_names": ["unet", "vae", "scheduler"],
            "input_names": ["embeddings", "seed", ROUTE_STATE_INPUT],
            "output_names": ["latents", ROUTE_STATE_OUTPUT],
        }
        decode_config = {
            "params": {ROUTE_STATE_INPUT: {"type": "modular_route_state"}},
            "model_input_names": ["vae"],
            "input_names": ["latents", ROUTE_STATE_INPUT],
            "output_names": ["images"],
        }

        for action in ("encode", "denoise", "decode"):
            for phase in ("init", "call"):
                with self.subTest(action=action, phase=phase):
                    original_vae = _FixtureSdxlVae()
                    replacement_vae = _FixtureSdxlVae()
                    resident = {model_id: original_vae}
                    pipeline_calls = []
                    unet_component = object()
                    scheduler_component = object()

                    def managed_components(*, ids, return_dict_with_names=True):
                        if return_dict_with_names:
                            return {
                                "unet": unet_component,
                                "vae": resident[model_id],
                                "scheduler": scheduler_component,
                            }
                        return {requested: resident[requested] for requested in set(ids)}

                    if action == "encode":
                        class FakePipeline:
                            _execution_device = torch.device("cpu")
                            blocks = type("PipelineBlocks", (), {"doc": "encode"})()

                            def __init__(self):
                                self.vae = None

                            def update_components(self, **values):
                                for name, value in values.items():
                                    setattr(self, name, value)

                            def __call__(self, **_kwargs):
                                pipeline_calls.append(True)
                                if phase == "call":
                                    resident[model_id] = replacement_vae
                                return {
                                    "image_latents": torch.zeros((1, 4, 8, 8)),
                                    "mask": None,
                                    "masked_image_latents": None,
                                }

                        class FakeBlocks:
                            component_names = ["vae"]
                            input_names = ["image", "generator"]

                            @staticmethod
                            def init_pipeline(*, components_manager):
                                if phase == "init":
                                    resident[model_id] = replacement_vae
                                return FakePipeline()

                        node = ImageEncode()
                        node.progress = Mock()
                        with (
                            patch(
                                "modules.ModularDiffusers.latents.pipeline_class_from_runtime_inputs",
                                return_value=diffusers.StableDiffusionXLModularPipeline,
                            ),
                            patch(
                                "modules.ModularDiffusers.latents.require_modiff_node_contract",
                                return_value=(FakeBlocks(), encode_config),
                            ),
                            patch("modules.ModularDiffusers.latents.collect_model_ids", return_value=[model_id]),
                            patch(
                                "modules.ModularDiffusers.latents.components.get_components_by_ids",
                                side_effect=managed_components,
                            ),
                            patch("modules.ModularDiffusers.latents.issue_encoder_route_state") as issue_route,
                            self.assertRaisesRegex(ValueError, "changed during|resident Modular pipeline"),
                        ):
                            node.execute(
                                vae=outputs["vae_out"],
                                image=Image.new("RGB", (64, 64), "red"),
                                seed=7,
                            )
                        issue_route.assert_not_called()

                    elif action == "denoise":
                        class FakePipeline:
                            _execution_device = torch.device("cpu")
                            component_names = ["unet", "vae", "scheduler"]
                            blocks = type("PipelineBlocks", (), {"doc": "denoise"})()
                            transformer = None

                            def __init__(self):
                                self.vae = None

                            def update_components(self, **values):
                                for name, value in values.items():
                                    setattr(self, name, value)

                            def __call__(self, **_kwargs):
                                pipeline_calls.append(True)
                                if phase == "call":
                                    resident[model_id] = replacement_vae
                                return {"latents": torch.zeros((1, 4, 8, 8))}

                        class FakeBlocks:
                            component_names = ["unet", "vae", "scheduler"]
                            input_names = ["prompt_embeds", "generator"]

                            @staticmethod
                            def init_pipeline(*, components_manager):
                                if phase == "init":
                                    resident[model_id] = replacement_vae
                                return FakePipeline()

                        node = Denoise()
                        node.progress = Mock()
                        with (
                            patch(
                                "modules.ModularDiffusers.denoise.pipeline_class_from_runtime_inputs",
                                return_value=diffusers.StableDiffusionXLModularPipeline,
                            ),
                            patch(
                                "modules.ModularDiffusers.denoise.require_modiff_node_contract",
                                return_value=(FakeBlocks(), denoise_config),
                            ),
                            patch("modules.ModularDiffusers.denoise.collect_model_ids", return_value=[model_id]),
                            patch("modules.ModularDiffusers.denoise.insert_preview_block"),
                            patch(
                                "modules.ModularDiffusers.denoise.components.get_components_by_ids",
                                side_effect=managed_components,
                            ),
                            patch("modules.ModularDiffusers.denoise.issue_normal_decode_route_state") as issue_route,
                            self.assertRaisesRegex(ValueError, "changed during|resident SDXL Denoise pipeline"),
                        ):
                            node.execute(
                                unet=outputs["unet_out"],
                                vae=outputs["vae_out"],
                                scheduler=outputs["scheduler"],
                                embeddings={"prompt_embeds": object()},
                                seed=7,
                            )
                        issue_route.assert_not_called()

                    else:
                        denoised = torch.zeros((1, 4, 8, 8))
                        route = issue_normal_decode_route_state(
                            binding=token,
                            latents=denoised,
                            vae_component=original_vae,
                            vae_latent_channels=4,
                            vae_scale_factor=8,
                        )

                        class FakePipeline:
                            blocks = type("PipelineBlocks", (), {"doc": "decode"})()

                            def __init__(self):
                                self.vae = None

                            def update_components(self, **values):
                                for name, value in values.items():
                                    setattr(self, name, value)

                            def __call__(self, **_kwargs):
                                pipeline_calls.append(True)
                                if phase == "call":
                                    resident[model_id] = replacement_vae
                                return {"images": "must-not-publish"}

                        class FakeBlocks:
                            component_names = ["vae"]
                            input_names = ["latents"]

                            @staticmethod
                            def init_pipeline(*, components_manager):
                                if phase == "init":
                                    resident[model_id] = replacement_vae
                                return FakePipeline()

                        with (
                            patch(
                                "modules.ModularDiffusers.latents.pipeline_class_from_runtime_inputs",
                                return_value=diffusers.StableDiffusionXLModularPipeline,
                            ),
                            patch(
                                "modules.ModularDiffusers.latents.require_modiff_node_contract",
                                return_value=(FakeBlocks(), decode_config),
                            ),
                            patch("modules.ModularDiffusers.latents.collect_model_ids", return_value=[model_id]),
                            patch(
                                "modules.ModularDiffusers.latents.components.get_components_by_ids",
                                side_effect=managed_components,
                            ),
                            self.assertRaisesRegex(ValueError, "changed during|resident Modular pipeline"),
                        ):
                            DecodeLatents().execute(
                                vae=outputs["vae_out"],
                                latents=denoised,
                                route_state_in=route,
                            )

                    self.assertEqual(len(pipeline_calls), 0 if phase == "init" else 1)

    def test_image_encode_seals_the_post_vae_generator_state(self):
        token, outputs = _bound_outputs()
        pipeline_class = type(QWEN_EDIT, (), {})
        observed = {}
        encoded_latents = torch.zeros((1, 1, 2, 2))

        class FakePipeline:
            _execution_device = torch.device("cpu")
            blocks = type("PipelineBlocks", (), {"doc": "fixture"})()

            def update_components(self, **_kwargs):
                return None

            def __call__(self, **kwargs):
                self.assert_generator(kwargs)
                return {
                    "image_latents": encoded_latents,
                    "processed_mask_image": None,
                    "mask_overlay_kwargs": None,
                }

            @staticmethod
            def assert_generator(kwargs):
                observed["kwargs"] = dict(kwargs)
                torch.rand((5,), generator=kwargs["generator"])
                observed["post_vae_state"] = kwargs["generator"].get_state().clone()

        class FakeBlocks:
            component_names = []
            input_names = ["image", "generator"]

            @staticmethod
            def init_pipeline(*, components_manager):
                return FakePipeline()

        config = {
            "params": {"seed": {"type": "int", "min": 0, "max": 4294967295}},
            "model_input_names": ["vae"],
            "input_names": ["image", "seed"],
            "output_names": ["image_latents", ROUTE_STATE_OUTPUT],
        }
        with (
            patch("modules.ModularDiffusers.latents.pipeline_class_from_runtime_inputs", return_value=pipeline_class),
            patch(
                "modules.ModularDiffusers.latents.require_modiff_node_contract",
                return_value=(FakeBlocks(), config),
            ),
            patch("modules.ModularDiffusers.latents.collect_model_ids", return_value=[]),
        ):
            result = ImageEncode().execute(vae=outputs["vae_out"], image="source", seed="7")

        self.assertIs(result["image_latents"], encoded_latents)
        self.assertNotIn("seed", observed["kwargs"])
        routed_generator = consume_encoder_route_state(
            result[ROUTE_STATE_OUTPUT],
            binding=token,
            model_type=QWEN_EDIT,
            seed=7,
            execution_device="cpu",
            image_latents=result["image_latents"],
        )["generator"]
        self.assertTrue(torch.equal(routed_generator.get_state(), observed["post_vae_state"]))

    def test_advertised_edit_plus_multi_image_encode_seals_a_list_route(self):
        token, outputs = _bound_outputs(QWEN_EDIT_PLUS)
        seed = 23
        source_images = [object(), object()]
        encoded_latents = [torch.zeros((1, 1, 2, 2)), torch.ones((1, 1, 2, 2))]
        observed = {}

        self.assertIsNotNone(PINNED_MODULAR_WORKFLOW_TRUTH[QWEN_EDIT_PLUS].mode("multi_image_reference_edit"))

        class FakePipeline:
            _execution_device = torch.device("cpu")
            blocks = type("PipelineBlocks", (), {"doc": "fixture"})()

            def update_components(self, **_kwargs):
                return None

            def __call__(self, **kwargs):
                observed["image"] = kwargs["image"]
                torch.rand((4,), generator=kwargs["generator"])
                observed["post_vae_state"] = kwargs["generator"].get_state().clone()
                return {
                    "image_latents": encoded_latents,
                    "processed_mask_image": None,
                    "mask_overlay_kwargs": None,
                }

        class FakeBlocks:
            component_names = []
            input_names = ["image", "generator"]

            @staticmethod
            def init_pipeline(*, components_manager):
                return FakePipeline()

        config = {
            "params": {"seed": {"type": "int", "min": 0, "max": 4294967295}},
            "model_input_names": ["vae"],
            "input_names": ["image", "seed"],
            "output_names": ["image_latents", ROUTE_STATE_OUTPUT],
        }
        with (
            patch(
                "modules.ModularDiffusers.latents.pipeline_class_from_runtime_inputs",
                return_value=diffusers.QwenImageEditPlusModularPipeline,
            ),
            patch(
                "modules.ModularDiffusers.latents.require_modiff_node_contract",
                return_value=(FakeBlocks(), config),
            ),
            patch("modules.ModularDiffusers.latents.collect_model_ids", return_value=[]),
        ):
            result = ImageEncode().execute(
                vae=outputs["vae_out"],
                image=source_images,
                seed=str(seed),
            )

        self.assertIs(observed["image"], source_images)
        self.assertIs(result["image_latents"], encoded_latents)
        routed = consume_encoder_route_state(
            result[ROUTE_STATE_OUTPUT],
            binding=token,
            model_type=QWEN_EDIT_PLUS,
            seed=seed,
            execution_device="cpu",
            image_latents=result["image_latents"],
        )
        self.assertTrue(torch.equal(observed["post_vae_state"], routed["generator"].get_state()))

    def test_controlnet_emits_post_control_route_for_text_and_optional_image_routes(self):
        token, outputs = _bound_outputs(QWEN_IMAGE)
        _issuer, controlnet = _publish_standalone(manager_model_id="controlnet-runtime-route")
        seed = 23
        control_latents = [
            torch.zeros((1, 1, 2, 2)),
            torch.ones((1, 1, 2, 2)),
        ]
        observed = []

        class FakeControlOutput:
            def __init__(self, values):
                self.values = values

        class FakeControlPipeline:
            _execution_device = torch.device("cpu")

            def update_components(self, **_kwargs):
                return None

            def __call__(self, **kwargs):
                observed.append(
                    {
                        "kwargs": dict(kwargs),
                        "pre_state": kwargs["generator"].get_state().clone(),
                    }
                )
                torch.rand((4,), generator=kwargs["generator"])
                observed[-1]["post_state"] = kwargs["generator"].get_state().clone()
                return FakeControlOutput({"control_image_latents": control_latents[len(observed) - 1]})

        class FakeControlBlocks:
            component_names = ["vae", "controlnet"]
            input_names = ["control_image", "height", "width", "generator"]

            @staticmethod
            def init_pipeline(*, components_manager):
                return FakeControlPipeline()

        class FakeDenoiseBlocks:
            component_names = ["controlnet"]
            input_names = [
                "control_image_latents",
                "controlnet_conditioning_scale",
                "control_guidance_start",
                "control_guidance_end",
            ]

        def contract(_pipeline_class, node_type, **_kwargs):
            if node_type == "controlnet":
                return FakeControlBlocks(), _controlnet_node_config()
            return FakeDenoiseBlocks(), {"input_names": FakeDenoiseBlocks.input_names}

        common = {
            "controlnet": controlnet,
            "vae": outputs["vae_out"],
            "control_image": object(),
            "controlnet_conditioning_scale": "0.5",
            "control_guidance_start": "0.0",
            "control_guidance_end": "1.0",
            "height": "512",
            "width": 512,
            "seed": str(seed),
        }
        with (
            patch(
                "modules.ModularDiffusers.controlnet.pipeline_class_from_runtime_inputs",
                return_value=diffusers.QwenImageModularPipeline,
            ),
            patch("modules.ModularDiffusers.controlnet.require_modiff_node_contract", side_effect=contract),
            patch("modules.ModularDiffusers.controlnet.collect_model_ids", return_value=[]),
        ):
            text_result = Controlnet().execute(**common)

            image_latents = torch.full((1, 1, 2, 2), 3.0)
            image_generator = torch.Generator(device="cpu").manual_seed(seed)
            torch.rand((2,), generator=image_generator)
            image_route = issue_encoder_route_state(
                binding=token,
                seed=seed,
                generator=image_generator,
                image_latents=image_latents,
                processed_mask_image=None,
                mask_overlay_kwargs=None,
            )
            image_result = Controlnet().execute(**common, route_state_in=image_route)

        self.assertEqual(observed[0]["kwargs"]["height"], 512)
        self.assertEqual(observed[0]["kwargs"]["width"], 512)
        self.assertNotIn("seed", observed[0]["kwargs"])
        self.assertNotIn(ROUTE_STATE_INPUT, observed[0]["kwargs"])
        self.assertNotIn("image_latents", observed[1]["kwargs"])
        self.assertIs(text_result["controlnet_bundle"]["control_image_latents"], control_latents[0])
        self.assertIs(text_result["controlnet_bundle"]["controlnet"], controlnet)
        self.assertIs(image_result["controlnet_bundle"]["control_image_latents"], control_latents[1])

        fresh = torch.Generator(device="cpu").manual_seed(seed)
        self.assertTrue(torch.equal(observed[0]["pre_state"], fresh.get_state()))
        self.assertTrue(torch.equal(observed[1]["pre_state"], image_generator.get_state()))
        text_runtime = consume_denoise_route_state(
            text_result[ROUTE_STATE_OUTPUT],
            binding=token,
            model_type=QWEN_IMAGE,
            seed=seed,
            execution_device="cpu",
            image_latents=None,
            control_image_latents=control_latents[0],
            controlnet_component=controlnet,
        )
        image_runtime = consume_denoise_route_state(
            image_result[ROUTE_STATE_OUTPUT],
            binding=token,
            model_type=QWEN_IMAGE,
            seed=seed,
            execution_device="cpu",
            image_latents=image_latents,
            control_image_latents=control_latents[1],
            controlnet_component=controlnet,
        )
        self.assertTrue(torch.equal(text_runtime["generator"].get_state(), observed[0]["post_state"]))
        self.assertTrue(torch.equal(image_runtime["generator"].get_state(), observed[1]["post_state"]))

    def test_controlnet_retry_restarts_from_the_same_inherited_generator_snapshot(self):
        token, outputs = _bound_outputs(QWEN_IMAGE)
        _issuer, controlnet = _publish_standalone(manager_model_id="controlnet-runtime-retry")
        image_latents = torch.zeros((1, 1, 2, 2))
        generator = torch.Generator(device="cpu").manual_seed(17)
        torch.rand((3,), generator=generator)
        image_route = issue_encoder_route_state(
            binding=token,
            seed=17,
            generator=generator,
            image_latents=image_latents,
            processed_mask_image=None,
            mask_overlay_kwargs=None,
        )
        control_latents = torch.ones((1, 1, 2, 2))
        pre_states = []
        post_states = []

        class FakeOutput:
            values = {"control_image_latents": control_latents}

        class FakePipeline:
            _execution_device = torch.device("cpu")

            def update_components(self, **_kwargs):
                return None

            def __call__(self, **kwargs):
                pre_states.append(kwargs["generator"].get_state().clone())
                torch.rand((5,), generator=kwargs["generator"])
                post_states.append(kwargs["generator"].get_state().clone())
                if len(pre_states) == 1:
                    raise RuntimeError("fixture control failure")
                return FakeOutput()

        class FakeControlBlocks:
            component_names = ["vae", "controlnet"]
            input_names = ["control_image", "generator"]

            @staticmethod
            def init_pipeline(*, components_manager):
                return FakePipeline()

        class FakeDenoiseBlocks:
            component_names = ["controlnet"]
            input_names = ["control_image_latents"]

        def contract(_pipeline_class, node_type, **_kwargs):
            if node_type == "controlnet":
                return FakeControlBlocks(), _controlnet_node_config()
            return FakeDenoiseBlocks(), {"input_names": FakeDenoiseBlocks.input_names}

        kwargs = {
            "controlnet": controlnet,
            "vae": outputs["vae_out"],
            "control_image": object(),
            "seed": 17,
            ROUTE_STATE_INPUT: image_route,
        }
        with (
            patch(
                "modules.ModularDiffusers.controlnet.pipeline_class_from_runtime_inputs",
                return_value=diffusers.QwenImageModularPipeline,
            ),
            patch("modules.ModularDiffusers.controlnet.require_modiff_node_contract", side_effect=contract),
            patch("modules.ModularDiffusers.controlnet.collect_model_ids", return_value=[]),
        ):
            with self.assertRaisesRegex(RuntimeError, "fixture control failure"):
                Controlnet().execute(**kwargs)
            result = Controlnet().execute(**kwargs)

        self.assertTrue(torch.equal(pre_states[0], pre_states[1]))
        self.assertTrue(torch.equal(post_states[0], post_states[1]))
        consumed = consume_denoise_route_state(
            result[ROUTE_STATE_OUTPUT],
            binding=token,
            model_type=QWEN_IMAGE,
            seed=17,
            execution_device="cpu",
            image_latents=image_latents,
            control_image_latents=control_latents,
            controlnet_component=controlnet,
        )
        self.assertTrue(torch.equal(consumed["generator"].get_state(), post_states[1]))

    def test_controlnet_revalidates_vae_and_standalone_publications_before_route_publication(self):
        for case in (
            "standalone_republish",
            "vae_metadata_mutation",
            "vae_metadata_mutation_during_call",
        ):
            with self.subTest(case=case):
                _token, outputs = _bound_outputs(QWEN_IMAGE, suffix=case[0], model_id=f"vae-{case}")
                identity = _standalone_identity(fingerprint=("5" if case == "standalone_republish" else "6") * 64)
                issuer, controlnet = _publish_standalone(
                    identity,
                    manager_model_id=f"controlnet-init-toctou-{case}",
                )
                retained_publications = []
                init_calls = []
                pipeline_calls = []

                def mutate_during_init():
                    if case == "standalone_republish":
                        retained_publications.append(
                            _publish_standalone(
                                identity,
                                issuer=issuer,
                                manager_model_id=f"controlnet-init-toctou-{case}",
                            )[1]
                        )
                    elif case == "vae_metadata_mutation":
                        outputs["vae_out"]["repo_id"] = "fixture/tampered-during-controlnet-init"

                class FakeOutput:
                    values = {"control_image_latents": torch.zeros((1, 1, 2, 2))}

                class FakePipeline:
                    _execution_device = torch.device("cpu")

                    def update_components(self, **_kwargs):
                        return None

                    def __call__(self, **_kwargs):
                        pipeline_calls.append(True)
                        if case == "vae_metadata_mutation_during_call":
                            outputs["vae_out"]["repo_id"] = "fixture/tampered-during-controlnet-call"
                        return FakeOutput()

                class FakeControlBlocks:
                    component_names = ["vae", "controlnet"]
                    input_names = ["control_image", "generator"]

                    def init_pipeline(self, *, components_manager):
                        init_calls.append(components_manager)
                        mutate_during_init()
                        return FakePipeline()

                class FakeDenoiseBlocks:
                    component_names = ["controlnet"]
                    input_names = ["control_image_latents"]

                def contract(_pipeline_class, node_type, **_kwargs):
                    if node_type == "controlnet":
                        return FakeControlBlocks(), _controlnet_node_config()
                    return FakeDenoiseBlocks(), {"input_names": FakeDenoiseBlocks.input_names}

                model_id_scan = Mock(return_value=[])
                route_issuer = Mock(wraps=issue_controlnet_route_state)
                expected_message = "no longer the current" if case == "standalone_republish" else "does not match"
                with (
                    patch(
                        "modules.ModularDiffusers.controlnet.pipeline_class_from_runtime_inputs",
                        return_value=diffusers.QwenImageModularPipeline,
                    ),
                    patch("modules.ModularDiffusers.controlnet.require_modiff_node_contract", side_effect=contract),
                    patch("modules.ModularDiffusers.controlnet.collect_model_ids", model_id_scan),
                    patch("modules.ModularDiffusers.controlnet.issue_controlnet_route_state", route_issuer),
                    self.assertRaisesRegex(ValueError, expected_message),
                ):
                    Controlnet().execute(
                        controlnet=controlnet,
                        vae=outputs["vae_out"],
                        control_image=object(),
                        seed=7,
                    )

                self.assertEqual(len(init_calls), 1)
                route_issuer.assert_not_called()
                if case == "vae_metadata_mutation_during_call":
                    self.assertEqual(pipeline_calls, [True])
                    model_id_scan.assert_called_once()
                else:
                    self.assertEqual(pipeline_calls, [])
                    model_id_scan.assert_not_called()

    def test_decode_materializes_only_minimal_inpaint_state_and_routed_overlay(self):
        token, outputs = _bound_outputs()
        overlay = {
            "crops_coords": (0, 0, 8, 8),
            "original_image": object(),
            "original_mask": object(),
        }
        image_latents = torch.zeros((1, 1, 2, 2))
        inpaint_latents = torch.ones((1, 1, 2, 2))
        normal_latents = torch.full((1, 1, 2, 2), 2.0)
        encoder_route = issue_encoder_route_state(
            binding=token,
            seed=7,
            generator=torch.Generator(device="cpu").manual_seed(7),
            image_latents=image_latents,
            processed_mask_image=torch.ones((1, 1, 8, 8)),
            mask_overlay_kwargs=overlay,
        )
        inpaint_route = issue_decode_route_state(
            encoder_route,
            binding=token,
            actual_mask=torch.ones((1, 1, 8, 8)),
            latents=inpaint_latents,
        )
        normal_route = issue_normal_decode_route_state(binding=token, latents=normal_latents)
        pipeline_class = type(QWEN_EDIT, (), {})
        calls = []

        class FakePipeline:
            blocks = type("PipelineBlocks", (), {"doc": "fixture"})()

            def update_components(self, **_kwargs):
                return None

            def __call__(self, **kwargs):
                calls.append(kwargs)
                return {"images": "decoded-image"}

        class FakeBlocks:
            component_names = []
            input_names = ["latents", "mask_overlay_kwargs"]

            @staticmethod
            def init_pipeline(*, components_manager):
                return FakePipeline()

        config = {
            "params": {ROUTE_STATE_INPUT: {"type": "modular_route_state"}},
            "model_input_names": ["vae"],
            "input_names": ["latents", ROUTE_STATE_INPUT],
            "output_names": ["images"],
        }
        with (
            patch("modules.ModularDiffusers.latents.pipeline_class_from_runtime_inputs", return_value=pipeline_class),
            patch(
                "modules.ModularDiffusers.latents.require_modiff_node_contract",
                return_value=(FakeBlocks(), config),
            ),
            patch("modules.ModularDiffusers.latents.collect_model_ids", return_value=[]),
        ):
            inpaint_result = DecodeLatents().execute(
                vae=outputs["vae_out"],
                latents=inpaint_latents,
                route_state_in=inpaint_route,
            )
            normal_result = DecodeLatents().execute(
                vae=outputs["vae_out"],
                latents=normal_latents,
                route_state_in=normal_route,
            )

        self.assertEqual(inpaint_result["images"], "decoded-image")
        self.assertEqual(normal_result["images"], "decoded-image")
        self.assertEqual(calls[0]["state"].values, {"mask": True})
        self.assertIs(calls[0]["latents"], inpaint_latents)
        self.assertEqual(calls[0]["mask_overlay_kwargs"], overlay)
        self.assertNotIn("state", calls[1])
        self.assertNotIn("mask_overlay_kwargs", calls[1])

    def test_qwen_reviewed_state_flows_execute_the_exact_fake_action_sequence(self):
        expected_flows = {
            "image2image": (False, False),
            "inpainting": (True, False),
            "controlnet_image2image": (False, True),
            "controlnet_inpainting": (True, True),
        }
        truth = PINNED_MODULAR_WORKFLOW_TRUTH[QWEN_IMAGE]
        self.assertEqual(set(dict(truth.state_flows)), set(expected_flows))

        for index, (flow_name, state_flow) in enumerate(truth.state_flows):
            inpaint, controlled = expected_flows[flow_name]
            with self.subTest(state_flow=flow_name):
                seed = 100 + index
                _token, outputs = _bound_outputs(QWEN_IMAGE, suffix=chr(ord("a") + index))
                trace = ["text_encoder"]
                runtime = {}
                image_latents = torch.full((1, 1, 2, 2), float(index + 1))
                processed_mask = torch.ones((1, 1, 8, 8)) if inpaint else None
                overlay = (
                    {
                        "crops_coords": (0, 0, 8, 8),
                        "original_image": object(),
                        "original_mask": object(),
                    }
                    if inpaint
                    else None
                )

                class FakeEncodePipeline:
                    _execution_device = torch.device("cpu")
                    blocks = type("PipelineBlocks", (), {"doc": "fixture"})()

                    def update_components(self, **_kwargs):
                        return None

                    def __call__(self, **kwargs):
                        trace.append("vae_encoder")
                        runtime["vae"] = dict(kwargs)
                        runtime["post_vae_generator"] = kwargs["generator"].get_state().clone()
                        return {
                            "image_latents": image_latents,
                            "processed_mask_image": processed_mask,
                            "mask_overlay_kwargs": overlay,
                        }

                class FakeEncodeBlocks:
                    component_names = []
                    input_names = ["image", "mask_image", "height", "width", "generator"]

                    @staticmethod
                    def init_pipeline(*, components_manager):
                        return FakeEncodePipeline()

                encode_config = {
                    "params": {
                        "image": {"type": "image"},
                        "mask_image": {"type": "image"},
                        "height": {"type": "int", "min": 64, "max": 2048},
                        "width": {"type": "int", "min": 64, "max": 2048},
                        "seed": {"type": "int", "min": 0, "max": 4294967295},
                    },
                    "model_input_names": ["vae"],
                    "input_names": ["image", "mask_image", "height", "width", "seed"],
                    "output_names": ["image_latents", ROUTE_STATE_OUTPUT],
                }
                encode_kwargs = {
                    "vae": outputs["vae_out"],
                    "image": object(),
                    "height": 512,
                    "width": 512,
                    "seed": seed,
                }
                if inpaint:
                    encode_kwargs["mask_image"] = object()
                with (
                    patch(
                        "modules.ModularDiffusers.latents.pipeline_class_from_runtime_inputs",
                        return_value=diffusers.QwenImageModularPipeline,
                    ),
                    patch(
                        "modules.ModularDiffusers.latents.require_modiff_node_contract",
                        return_value=(FakeEncodeBlocks(), encode_config),
                    ),
                    patch("modules.ModularDiffusers.latents.collect_model_ids", return_value=[]),
                ):
                    encoded = ImageEncode().execute(**encode_kwargs)

                self.assertIs(encoded["image_latents"], image_latents)
                self.assertEqual("mask_image" in runtime["vae"], inpaint)
                image_route = encoded[ROUTE_STATE_OUTPUT]
                route_to_denoise = image_route
                control_bundle = None
                control_latents = None
                controlnet = None

                if controlled:
                    _issuer, controlnet = _publish_standalone(
                        manager_model_id=f"controlnet-state-flow-{index}"
                    )
                    control_latents = torch.full((1, 1, 2, 2), float(index + 11))
                    control_source = object()

                    class FakeControlOutput:
                        values = {"control_image_latents": control_latents}

                    class FakeControlPipeline:
                        _execution_device = torch.device("cpu")

                        def update_components(self, **_kwargs):
                            return None

                        def __call__(self, **kwargs):
                            trace.append("controlnet")
                            runtime["control"] = dict(kwargs)
                            runtime["pre_control_generator"] = kwargs["generator"].get_state().clone()
                            torch.rand((4,), generator=kwargs["generator"])
                            runtime["post_control_generator"] = kwargs["generator"].get_state().clone()
                            return FakeControlOutput()

                    class FakeControlBlocks:
                        component_names = ["vae", "controlnet"]
                        input_names = ["control_image", "height", "width", "generator"]

                        @staticmethod
                        def init_pipeline(*, components_manager):
                            return FakeControlPipeline()

                    class FakeControlDenoiseBlocks:
                        component_names = ["controlnet"]
                        input_names = [
                            "control_image_latents",
                            "controlnet_conditioning_scale",
                            "control_guidance_start",
                            "control_guidance_end",
                        ]

                    def control_contract(_pipeline_class, node_type, **_kwargs):
                        if node_type == "controlnet":
                            return FakeControlBlocks(), _controlnet_node_config()
                        return FakeControlDenoiseBlocks(), {"input_names": FakeControlDenoiseBlocks.input_names}

                    with (
                        patch(
                            "modules.ModularDiffusers.controlnet.pipeline_class_from_runtime_inputs",
                            return_value=diffusers.QwenImageModularPipeline,
                        ),
                        patch(
                            "modules.ModularDiffusers.controlnet.require_modiff_node_contract",
                            side_effect=control_contract,
                        ),
                        patch("modules.ModularDiffusers.controlnet.collect_model_ids", return_value=[]),
                    ):
                        controlled_output = Controlnet().execute(
                            controlnet=controlnet,
                            vae=outputs["vae_out"],
                            control_image=control_source,
                            controlnet_conditioning_scale=0.5,
                            control_guidance_start=0.0,
                            control_guidance_end=1.0,
                            height=512,
                            width=512,
                            seed=seed,
                            route_state_in=image_route,
                        )

                    self.assertIs(runtime["control"]["control_image"], control_source)
                    self.assertTrue(
                        torch.equal(runtime["pre_control_generator"], runtime["post_vae_generator"])
                    )
                    self.assertIs(
                        controlled_output["controlnet_bundle"]["control_image_latents"],
                        control_latents,
                    )
                    control_bundle = controlled_output["controlnet_bundle"]
                    route_to_denoise = controlled_output[ROUTE_STATE_OUTPUT]

                denoised_latents = torch.full((1, 1, 2, 2), float(index + 21))

                class FakeDenoisePipeline:
                    _execution_device = torch.device("cpu")
                    component_names = ["controlnet"] if controlled else []
                    transformer = None
                    blocks = type("PipelineBlocks", (), {"doc": "fixture"})()

                    def update_components(self, **_kwargs):
                        return None

                    def __call__(self, **kwargs):
                        trace.append("denoise")
                        runtime["denoise"] = dict(kwargs)
                        runtime["pre_denoise_generator"] = kwargs["generator"].get_state().clone()
                        torch.rand((3,), generator=kwargs["generator"])
                        return {"latents": denoised_latents, "mask": processed_mask}

                class FakeDenoiseBlocks:
                    component_names = ["controlnet"] if controlled else []
                    input_names = [
                        "prompt_embeds",
                        "image_latents",
                        "processed_mask_image",
                        "control_image_latents",
                        "controlnet_conditioning_scale",
                        "control_guidance_start",
                        "control_guidance_end",
                        "strength",
                        "generator",
                    ]

                    @staticmethod
                    def init_pipeline(*, components_manager):
                        return FakeDenoisePipeline()

                denoise_config = _route_node_config(control_bundle=controlled)
                denoise_config["params"]["strength"] = {"type": "float", "min": 0.0, "max": 1.0}
                denoise_config["input_names"].append("strength")
                denoise_kwargs = {
                    "unet": outputs["unet_out"],
                    "scheduler": outputs["scheduler"],
                    "embeddings": {"prompt_embeds": f"prompt-{index}"},
                    "image_latents": image_latents,
                    "strength": 0.6,
                    "seed": seed,
                    "route_state_in": route_to_denoise,
                }
                if controlled:
                    denoise_kwargs["controlnet_bundle"] = control_bundle
                with (
                    patch(
                        "modules.ModularDiffusers.denoise.pipeline_class_from_runtime_inputs",
                        return_value=diffusers.QwenImageModularPipeline,
                    ),
                    patch(
                        "modules.ModularDiffusers.denoise.require_modiff_node_contract",
                        return_value=(FakeDenoiseBlocks(), denoise_config),
                    ),
                    patch("modules.ModularDiffusers.denoise.collect_model_ids", return_value=[]),
                    patch("modules.ModularDiffusers.denoise.insert_preview_block"),
                ):
                    denoised = Denoise().execute(**denoise_kwargs)

                self.assertIs(runtime["denoise"]["image_latents"], image_latents)
                self.assertEqual("processed_mask_image" in runtime["denoise"], inpaint)
                if inpaint:
                    self.assertIs(runtime["denoise"]["processed_mask_image"], processed_mask)
                self.assertEqual("control_image_latents" in runtime["denoise"], controlled)
                if controlled:
                    self.assertIs(runtime["denoise"]["control_image_latents"], control_latents)
                    self.assertTrue(
                        torch.equal(runtime["pre_denoise_generator"], runtime["post_control_generator"])
                    )
                else:
                    self.assertTrue(
                        torch.equal(runtime["pre_denoise_generator"], runtime["post_vae_generator"])
                    )

                class FakeDecodePipeline:
                    blocks = type("PipelineBlocks", (), {"doc": "fixture"})()

                    def update_components(self, **_kwargs):
                        return None

                    def __call__(self, **kwargs):
                        trace.append("decoder")
                        runtime["decode"] = dict(kwargs)
                        return {"images": f"decoded-{flow_name}"}

                class FakeDecodeBlocks:
                    component_names = []
                    input_names = ["latents", "mask_overlay_kwargs"]

                    @staticmethod
                    def init_pipeline(*, components_manager):
                        return FakeDecodePipeline()

                decode_config = {
                    "params": {ROUTE_STATE_INPUT: {"type": "modular_route_state"}},
                    "model_input_names": ["vae"],
                    "input_names": ["latents", ROUTE_STATE_INPUT],
                    "output_names": ["images"],
                }
                with (
                    patch(
                        "modules.ModularDiffusers.latents.pipeline_class_from_runtime_inputs",
                        return_value=diffusers.QwenImageModularPipeline,
                    ),
                    patch(
                        "modules.ModularDiffusers.latents.require_modiff_node_contract",
                        return_value=(FakeDecodeBlocks(), decode_config),
                    ),
                    patch("modules.ModularDiffusers.latents.collect_model_ids", return_value=[]),
                ):
                    decoded = DecodeLatents().execute(
                        vae=outputs["vae_out"],
                        latents=denoised["latents"],
                        route_state_in=denoised[ROUTE_STATE_OUTPUT],
                    )

                self.assertEqual(decoded["images"], f"decoded-{flow_name}")
                self.assertEqual(tuple(trace), state_flow.action_sequence)
                self.assertEqual("state" in runtime["decode"], inpaint)
                self.assertEqual("mask_overlay_kwargs" in runtime["decode"], inpaint)
                if inpaint:
                    self.assertEqual(runtime["decode"]["state"].values, {"mask": True})
                    self.assertEqual(runtime["decode"]["mask_overlay_kwargs"], overlay)

    def test_forged_route_and_reserved_identity_injection_fail_before_resolver(self):
        token, _outputs = _bound_outputs()
        nested_route, _image_latents = _normal_encoder_route(token)
        cases = (
            (Controlnet, {ROUTE_STATE_INPUT: {"model_type": QWEN_IMAGE}}),
            (Controlnet, {"control_image": {"generator": object()}}),
            (Denoise, {ROUTE_STATE_INPUT: {"model_type": QWEN_EDIT}}),
            (DecodeLatents, {ROUTE_STATE_INPUT: {"model_type": QWEN_EDIT}}),
            (ImageEncode, {"processed_mask_image": {"model_type": QWEN_EDIT}}),
            (Denoise, {"embeddings": {"mask_overlay_kwargs": {"model_type": QWEN_EDIT}}}),
            (Denoise, {"embeddings": {"prompt_embeds": nested_route}}),
            (DecodeLatents, {"mask": {"model_type": QWEN_EDIT}}),
        )
        for node_class, hostile in cases:
            with self.subTest(node=node_class.__name__, hostile=hostile):
                node = node_class()
                resolver_path = f"{node_class.__module__}.pipeline_class_from_runtime_inputs"
                with patch(resolver_path) as resolver, self.assertRaises(ValueError):
                    node.execute(**hostile)
                resolver.assert_not_called()

    def test_pipeline_identity_recovery_is_iterative_bounded_and_cycle_safe(self):
        cyclic = {"model_type": QWEN_EDIT}
        cyclic["self"] = cyclic
        self.assertIs(
            pipeline_class_from_runtime_inputs(None, cyclic),
            diffusers.QwenImageEditModularPipeline,
        )
        nested = {}
        cursor = nested
        for _index in range(40):
            cursor["next"] = {}
            cursor = cursor["next"]
        with self.assertRaisesRegex(ValueError, "safe nested-value limit"):
            pipeline_class_from_runtime_inputs(None, nested)

    def test_qwen_latents_without_route_fail_before_pipeline_initialization(self):
        for model_type in ("QwenImageModularPipeline", QWEN_EDIT, QWEN_EDIT_PLUS):
            with self.subTest(model_type=model_type):
                node = Denoise()
                pipeline_class = type(model_type, (), {})
                blocks = Mock()
                blocks.input_names = ["image_latents", "generator"]
                config = _route_node_config()
                with (
                    patch(
                        "modules.ModularDiffusers.denoise.pipeline_class_from_runtime_inputs",
                        return_value=pipeline_class,
                    ),
                    patch(
                        "modules.ModularDiffusers.denoise.require_modiff_node_contract", return_value=(blocks, config)
                    ),
                    self.assertRaisesRegex(ValueError, "latents require the opaque route state"),
                ):
                    node.execute(unet={"repo_id": "fixture/model"}, image_latents=object())
                blocks.init_pipeline.assert_not_called()

    def test_flattenable_bundle_latents_without_route_fail_before_init(self):
        node = Denoise()
        pipeline_class = type(QWEN_EDIT, (), {})
        blocks = Mock()
        blocks.input_names = ["prompt_embeds", "image_latents", "generator"]
        config = _route_node_config()
        config["input_names"] = ["embeddings", "seed", ROUTE_STATE_INPUT]
        with (
            patch("modules.ModularDiffusers.denoise.pipeline_class_from_runtime_inputs", return_value=pipeline_class),
            patch("modules.ModularDiffusers.denoise.require_modiff_node_contract", return_value=(blocks, config)),
            self.assertRaisesRegex(ValueError, "latents require the opaque route state"),
        ):
            node.execute(
                unet={"repo_id": "fixture/model"},
                embeddings={"prompt_embeds": object(), "image_latents": object()},
                seed=7,
            )
        blocks.init_pipeline.assert_not_called()

    def test_supported_qwen_decode_requires_route_before_init(self):
        _token, outputs = _bound_outputs()
        node = DecodeLatents()
        pipeline_class = type(QWEN_EDIT, (), {})
        blocks = Mock()
        config = {
            "params": {ROUTE_STATE_INPUT: {"type": "modular_route_state"}},
            "model_input_names": ["vae"],
            "input_names": ["latents", ROUTE_STATE_INPUT],
            "output_names": ["images"],
        }
        with (
            patch("modules.ModularDiffusers.latents.pipeline_class_from_runtime_inputs", return_value=pipeline_class),
            patch("modules.ModularDiffusers.latents.require_modiff_node_contract", return_value=(blocks, config)),
            self.assertRaisesRegex(ValueError, "Decode requires the opaque route state"),
        ):
            node.execute(vae=outputs["vae_out"], latents=object())
        blocks.init_pipeline.assert_not_called()

    def test_text_denoise_emits_normal_decode_binding(self):
        token, outputs = _bound_outputs(model_type="QwenImageModularPipeline")
        pipeline_class = type("QwenImageModularPipeline", (), {})
        received = {}
        unexpected_mask = False
        text_latents = torch.zeros((1, 1, 2, 2))

        class FakePipeline:
            _execution_device = torch.device("cpu")
            component_names = []
            transformer = None
            blocks = type("PipelineBlocks", (), {"doc": "fixture"})()

            def update_components(self, **_kwargs):
                return None

            def __call__(self, **kwargs):
                received.update(kwargs)
                return {
                    "latents": text_latents,
                    **({"mask": torch.ones((1, 1, 1, 1))} if unexpected_mask else {}),
                }

        class FakeBlocks:
            component_names = []
            input_names = ["prompt_embeds", "generator"]

            @staticmethod
            def init_pipeline(*, components_manager):
                return FakePipeline()

        config = _route_node_config()
        node = Denoise()
        with (
            patch("modules.ModularDiffusers.denoise.pipeline_class_from_runtime_inputs", return_value=pipeline_class),
            patch(
                "modules.ModularDiffusers.denoise.require_modiff_node_contract", return_value=(FakeBlocks(), config)
            ),
            patch("modules.ModularDiffusers.denoise.collect_model_ids", return_value=[]),
        ):
            result = node.execute(
                unet=outputs["unet_out"],
                scheduler=outputs["scheduler"],
                embeddings={"prompt_embeds": "encoded"},
                seed=5,
            )

        self.assertIs(result["latents"], text_latents)
        self.assertEqual(received["generator"].initial_seed(), 5)
        decoded = consume_decode_route_state(
            result[ROUTE_STATE_OUTPUT],
            binding=token,
            model_type="QwenImageModularPipeline",
            latents=result["latents"],
        )
        self.assertFalse(decoded["inpaint"])
        self.assertIsNone(decoded["mask_overlay_kwargs"])

        unexpected_mask = True
        with (
            patch("modules.ModularDiffusers.denoise.pipeline_class_from_runtime_inputs", return_value=pipeline_class),
            patch(
                "modules.ModularDiffusers.denoise.require_modiff_node_contract", return_value=(FakeBlocks(), config)
            ),
            patch("modules.ModularDiffusers.denoise.collect_model_ids", return_value=[]),
            self.assertRaisesRegex(ValueError, "unexpectedly returned inpaint mask"),
        ):
            Denoise().execute(
                unet=outputs["unet_out"],
                scheduler=outputs["scheduler"],
                embeddings={"prompt_embeds": "encoded"},
                seed=5,
            )

    def test_denoise_resolves_exact_controlnet_bundle_component_before_init(self):
        token, outputs = _bound_outputs(QWEN_IMAGE)
        _issuer, controlnet = _publish_standalone(manager_model_id="controlnet-denoise-runtime")
        control_latents = torch.zeros((1, 1, 2, 2))
        control_generator = torch.Generator(device="cpu").manual_seed(7)
        torch.rand((4,), generator=control_generator)
        control_route = issue_controlnet_route_state(
            None,
            binding=token,
            controlnet_component=controlnet,
            seed=7,
            generator=control_generator,
            control_image_latents=control_latents,
        )
        denoised_latents = torch.ones((1, 1, 2, 2))
        observed = {"validated": False}

        class FakePipeline:
            _execution_device = torch.device("cpu")
            component_names = ["controlnet"]
            transformer = None
            blocks = type("PipelineBlocks", (), {"doc": "fixture"})()

            def update_components(self, **_kwargs):
                return None

            def __call__(self, **kwargs):
                observed["runtime"] = kwargs
                return {"latents": denoised_latents, "mask": None}

        class FakeBlocks:
            component_names = ["controlnet"]
            input_names = ["prompt_embeds", "image_latents", "control_image_latents", "generator"]

            @staticmethod
            def init_pipeline(*, components_manager):
                if not observed["validated"]:
                    raise AssertionError("ControlNet provenance was not validated before pipeline initialization")
                return FakePipeline()

        config = _route_node_config(control_bundle=True)

        def validate_before_init(*args, **kwargs):
            self.assertIs(kwargs["controlnet_component"], controlnet)
            self.assertIs(kwargs["control_image_latents"], control_latents)
            result = validate_denoise_route_state(*args, **kwargs)
            observed["validated"] = True
            return result

        bundle = {
            "controlnet": controlnet,
            "control_image_latents": control_latents,
            "controlnet_conditioning_scale": 0.5,
        }
        with (
            patch(
                "modules.ModularDiffusers.denoise.pipeline_class_from_runtime_inputs",
                return_value=diffusers.QwenImageModularPipeline,
            ),
            patch(
                "modules.ModularDiffusers.denoise.require_modiff_node_contract",
                return_value=(FakeBlocks(), config),
            ),
            patch("modules.ModularDiffusers.denoise.validate_denoise_route_state", side_effect=validate_before_init),
            patch("modules.ModularDiffusers.denoise.collect_model_ids", return_value=[]),
            patch("modules.ModularDiffusers.denoise.insert_preview_block"),
        ):
            result = Denoise().execute(
                unet=outputs["unet_out"],
                scheduler=outputs["scheduler"],
                embeddings={"prompt_embeds": "encoded"},
                controlnet_bundle=bundle,
                seed=7,
                route_state_in=control_route,
            )

        self.assertIs(result["latents"], denoised_latents)
        self.assertIs(observed["runtime"]["control_image_latents"], control_latents)
        self.assertTrue(torch.equal(observed["runtime"]["generator"].get_state(), control_generator.get_state()))

        _other_issuer, other_controlnet = _publish_standalone(
            _standalone_identity(repo_id="fixture/other-controlnet", fingerprint="2" * 64),
            manager_model_id="other-controlnet-denoise-runtime",
        )
        tampered = deepcopy(controlnet)
        tampered["repo_id"] = "fixture/tampered"
        invalid_bundles = (
            ({"control_image_latents": control_latents}, "managed component payload"),
            (
                {"control_image_latents": control_latents, "nested": {"controlnet": controlnet}},
                "managed component payload",
            ),
            (
                {"control_image_latents": control_latents, "controlnet": other_controlnet},
                "different component publication",
            ),
            (
                {"control_image_latents": control_latents, "controlnet": tampered},
                "provenance binding",
            ),
        )
        for invalid_bundle, message in invalid_bundles:
            blocks = Mock()
            blocks.input_names = FakeBlocks.input_names
            blocks.component_names = FakeBlocks.component_names
            with (
                self.subTest(message=message),
                patch(
                    "modules.ModularDiffusers.denoise.pipeline_class_from_runtime_inputs",
                    return_value=diffusers.QwenImageModularPipeline,
                ),
                patch(
                    "modules.ModularDiffusers.denoise.require_modiff_node_contract",
                    return_value=(blocks, config),
                ),
                self.assertRaisesRegex(ValueError, message),
            ):
                Denoise().execute(
                    unet=outputs["unet_out"],
                    scheduler=outputs["scheduler"],
                    embeddings={"prompt_embeds": "encoded"},
                    controlnet_bundle=invalid_bundle,
                    seed=7,
                    route_state_in=control_route,
                )
            blocks.init_pipeline.assert_not_called()

    def test_qwen_denoise_rejects_any_control_bundle_without_its_control_route(self):
        token, outputs = _bound_outputs(QWEN_IMAGE)
        image_route, image_latents = _normal_encoder_route(token)
        blocks = Mock()
        blocks.input_names = ["prompt_embeds", "image_latents", "control_image_latents", "generator"]
        blocks.component_names = ["controlnet"]
        config = _route_node_config(control_bundle=True)
        cases = (
            ({}, None, None),
            ({"controlnet_conditioning_scale": 0.5}, image_route, image_latents),
        )
        for bundle, route, latents in cases:
            blocks.init_pipeline.reset_mock()
            with (
                self.subTest(bundle=bundle, route=route is not None),
                patch(
                    "modules.ModularDiffusers.denoise.pipeline_class_from_runtime_inputs",
                    return_value=diffusers.QwenImageModularPipeline,
                ),
                patch(
                    "modules.ModularDiffusers.denoise.require_modiff_node_contract",
                    return_value=(blocks, config),
                ),
                self.assertRaisesRegex(ValueError, "ControlNet.*route state"),
            ):
                Denoise().execute(
                    unet=outputs["unet_out"],
                    scheduler=outputs["scheduler"],
                    embeddings={"prompt_embeds": "encoded"},
                    image_latents=latents,
                    controlnet_bundle=bundle,
                    seed=7,
                    route_state_in=route,
                )
            blocks.init_pipeline.assert_not_called()

    def test_control_route_rejects_cross_loader_denoiser_and_scheduler_before_init(self):
        token_a, outputs_a = _bound_outputs(QWEN_IMAGE, suffix="a", model_id="control-denoise-a")
        _token_b, outputs_b = _bound_outputs(QWEN_IMAGE, suffix="b", model_id="control-denoise-b")
        _issuer, controlnet = _publish_standalone(manager_model_id="controlnet-cross-loader-denoise")
        control_latents = torch.zeros((1, 1, 2, 2))
        control_route = issue_controlnet_route_state(
            None,
            binding=token_a,
            controlnet_component=controlnet,
            seed=7,
            generator=torch.Generator(device="cpu").manual_seed(7),
            control_image_latents=control_latents,
        )
        bundle = {"controlnet": controlnet, "control_image_latents": control_latents}
        blocks = Mock()
        blocks.input_names = ["prompt_embeds", "control_image_latents", "generator"]
        blocks.component_names = ["controlnet"]
        config = _route_node_config(control_bundle=True)
        cases = (
            (outputs_a["unet_out"], outputs_b["scheduler"]),
            (outputs_b["unet_out"], outputs_b["scheduler"]),
        )
        for unet, scheduler in cases:
            blocks.init_pipeline.reset_mock()
            with (
                self.subTest(unet=unet["model_id"], scheduler=scheduler["model_id"]),
                patch(
                    "modules.ModularDiffusers.denoise.pipeline_class_from_runtime_inputs",
                    return_value=diffusers.QwenImageModularPipeline,
                ),
                patch(
                    "modules.ModularDiffusers.denoise.require_modiff_node_contract",
                    return_value=(blocks, config),
                ),
                self.assertRaisesRegex(ValueError, "different Models Loader"),
            ):
                Denoise().execute(
                    unet=unet,
                    scheduler=scheduler,
                    embeddings={"prompt_embeds": "encoded"},
                    controlnet_bundle=bundle,
                    seed=7,
                    route_state_in=control_route,
                )
            blocks.init_pipeline.assert_not_called()

    def test_stale_qwen_route_is_rejected_by_a_nonroute_pipeline_before_init(self):
        token, _outputs = _bound_outputs()
        route, _image_latents = _normal_encoder_route(token)
        for node_class, model_type, model_input in (
            (Denoise, "FluxModularPipeline", "unet"),
            (DecodeLatents, "FluxModularPipeline", "vae"),
        ):
            with self.subTest(node=node_class.__name__):
                node = node_class()
                pipeline_class = type(model_type, (), {})
                blocks = Mock()
                config = {
                    "params": {},
                    "model_input_names": [model_input],
                    "input_names": [],
                    "output_names": [],
                }
                with (
                    patch(f"{node_class.__module__}.pipeline_class_from_runtime_inputs", return_value=pipeline_class),
                    patch(f"{node_class.__module__}.require_modiff_node_contract", return_value=(blocks, config)),
                    self.assertRaisesRegex(ValueError, "does not declare route state"),
                ):
                    node.execute(**{model_input: {"repo_id": "fixture/flux"}, ROUTE_STATE_INPUT: route})
                blocks.init_pipeline.assert_not_called()

    def test_same_class_components_from_different_loaders_fail_before_init(self):
        token_a, outputs_a = _bound_outputs(suffix="a")
        _token_b, outputs_b = _bound_outputs(suffix="b")
        route, image_latents = _normal_encoder_route(token_a)
        blocks = Mock()
        blocks.input_names = ["prompt_embeds", "image_latents", "generator", "processed_mask_image"]
        config = _route_node_config()
        node = Denoise()
        pipeline_class = type(QWEN_EDIT, (), {})

        with (
            patch("modules.ModularDiffusers.denoise.pipeline_class_from_runtime_inputs", return_value=pipeline_class),
            patch("modules.ModularDiffusers.denoise.require_modiff_node_contract", return_value=(blocks, config)),
            self.assertRaisesRegex(ValueError, "different Models Loader"),
        ):
            node.execute(
                unet=outputs_a["unet_out"],
                scheduler=outputs_b["scheduler"],
                embeddings={"prompt_embeds": object()},
                image_latents=image_latents,
                seed=7,
                route_state_in=route,
            )
        blocks.init_pipeline.assert_not_called()

    def test_same_loader_cross_paired_latents_fail_before_pipeline_initialization(self):
        token, outputs = _bound_outputs()
        image_latents_a = torch.zeros((1, 1, 2, 2))
        image_latents_b = torch.zeros((1, 1, 2, 2))
        route_b = issue_encoder_route_state(
            binding=token,
            seed=7,
            generator=torch.Generator(device="cpu").manual_seed(7),
            image_latents=image_latents_b,
            processed_mask_image=None,
            mask_overlay_kwargs=None,
        )
        pipeline_class = type(QWEN_EDIT, (), {})
        denoise_blocks = Mock()
        denoise_blocks.input_names = ["prompt_embeds", "image_latents", "generator"]
        with (
            patch("modules.ModularDiffusers.denoise.pipeline_class_from_runtime_inputs", return_value=pipeline_class),
            patch(
                "modules.ModularDiffusers.denoise.require_modiff_node_contract",
                return_value=(denoise_blocks, _route_node_config()),
            ),
            self.assertRaisesRegex(ValueError, "exact latent output paired"),
        ):
            Denoise().execute(
                unet=outputs["unet_out"],
                scheduler=outputs["scheduler"],
                embeddings={"prompt_embeds": object()},
                image_latents=image_latents_a,
                seed=7,
                route_state_in=route_b,
            )
        denoise_blocks.init_pipeline.assert_not_called()

        denoised_latents_a = torch.ones((1, 1, 2, 2))
        denoised_latents_b = torch.ones((1, 1, 2, 2))
        decode_route_b = issue_normal_decode_route_state(
            binding=token,
            latents=denoised_latents_b,
        )
        decode_blocks = Mock()
        decode_blocks.input_names = ["latents"]
        decode_config = {
            "params": {ROUTE_STATE_INPUT: {"type": "modular_route_state"}},
            "model_input_names": ["vae"],
            "input_names": ["latents", ROUTE_STATE_INPUT],
            "output_names": ["images"],
        }
        with (
            patch("modules.ModularDiffusers.latents.pipeline_class_from_runtime_inputs", return_value=pipeline_class),
            patch(
                "modules.ModularDiffusers.latents.require_modiff_node_contract",
                return_value=(decode_blocks, decode_config),
            ),
            self.assertRaisesRegex(ValueError, "exact latent output paired"),
        ):
            DecodeLatents().execute(
                vae=outputs["vae_out"],
                latents=denoised_latents_a,
                route_state_in=decode_route_b,
            )
        decode_blocks.init_pipeline.assert_not_called()

    def test_edit_plus_list_cross_pair_and_dead_refs_fail_before_denoise_initialization(self):
        token, outputs = _bound_outputs(QWEN_EDIT_PLUS)
        first = torch.zeros((1, 1, 2, 2))
        second = torch.ones((1, 1, 2, 2))
        route = issue_encoder_route_state(
            binding=token,
            seed=7,
            generator=torch.Generator(device="cpu").manual_seed(7),
            image_latents=[first, second],
            processed_mask_image=None,
            mask_overlay_kwargs=None,
        )
        blocks = Mock()
        blocks.input_names = ["prompt_embeds", "image_latents", "generator"]
        config = _route_node_config()

        for connected, error_type, message in (
            ([second, first], ValueError, "exact latent output paired"),
            ((first, second), TypeError, "bounded nonempty list"),
        ):
            with (
                self.subTest(connected=type(connected).__name__),
                patch(
                    "modules.ModularDiffusers.denoise.pipeline_class_from_runtime_inputs",
                    return_value=diffusers.QwenImageEditPlusModularPipeline,
                ),
                patch(
                    "modules.ModularDiffusers.denoise.require_modiff_node_contract",
                    return_value=(blocks, config),
                ),
                self.assertRaisesRegex(error_type, message),
            ):
                Denoise().execute(
                    unet=outputs["unet_out"],
                    scheduler=outputs["scheduler"],
                    embeddings={"prompt_embeds": object()},
                    image_latents=connected,
                    seed=7,
                    route_state_in=route,
                )
        blocks.init_pipeline.assert_not_called()

        def issue_without_retaining_latents():
            transient_latents = [torch.zeros((1, 1, 2, 2)), torch.ones((1, 1, 2, 2))]
            return issue_encoder_route_state(
                binding=token,
                seed=7,
                generator=torch.Generator(device="cpu").manual_seed(7),
                image_latents=transient_latents,
                processed_mask_image=None,
                mask_overlay_kwargs=None,
            )

        dead_route = issue_without_retaining_latents()
        gc.collect()
        with (
            patch(
                "modules.ModularDiffusers.denoise.pipeline_class_from_runtime_inputs",
                return_value=diffusers.QwenImageEditPlusModularPipeline,
            ),
            patch(
                "modules.ModularDiffusers.denoise.require_modiff_node_contract",
                return_value=(blocks, config),
            ),
            self.assertRaisesRegex(ValueError, "no longer resident"),
        ):
            Denoise().execute(
                unet=outputs["unet_out"],
                scheduler=outputs["scheduler"],
                embeddings={"prompt_embeds": object()},
                image_latents=[torch.zeros((1, 1, 2, 2)), torch.ones((1, 1, 2, 2))],
                seed=7,
                route_state_in=dead_route,
            )
        blocks.init_pipeline.assert_not_called()

    def test_single_image_qwen_denoise_rejects_latent_lists_before_initialization(self):
        for model_type in (QWEN_IMAGE, QWEN_EDIT):
            with self.subTest(model_type=model_type):
                token, outputs = _bound_outputs(model_type)
                exact_latents = torch.zeros((1, 1, 2, 2))
                route = issue_encoder_route_state(
                    binding=token,
                    seed=7,
                    generator=torch.Generator(device="cpu").manual_seed(7),
                    image_latents=exact_latents,
                    processed_mask_image=None,
                    mask_overlay_kwargs=None,
                )
                blocks = Mock()
                blocks.input_names = ["prompt_embeds", "image_latents", "generator"]
                with (
                    patch(
                        "modules.ModularDiffusers.denoise.pipeline_class_from_runtime_inputs",
                        return_value=getattr(diffusers, model_type),
                    ),
                    patch(
                        "modules.ModularDiffusers.denoise.require_modiff_node_contract",
                        return_value=(blocks, _route_node_config()),
                    ),
                    self.assertRaisesRegex(TypeError, "Connected VAE image latents must be an exact Torch tensor"),
                ):
                    Denoise().execute(
                        unet=outputs["unet_out"],
                        scheduler=outputs["scheduler"],
                        embeddings={"prompt_embeds": object()},
                        image_latents=[exact_latents],
                        seed=7,
                        route_state_in=route,
                    )
                blocks.init_pipeline.assert_not_called()

    def test_runtime_component_role_swaps_fail_before_init(self):
        token, outputs = _bound_outputs()
        encoder_route, image_latents = _normal_encoder_route(token)
        denoised_latents = torch.zeros((1, 1, 2, 2))
        decode_route = issue_decode_route_state(
            encoder_route,
            binding=token,
            actual_mask=None,
            latents=denoised_latents,
        )
        pipeline_class = type(QWEN_EDIT, (), {})

        denoise_blocks = Mock()
        denoise_blocks.input_names = ["generator", "image_latents"]
        denoise_config = _route_node_config()
        with (
            patch("modules.ModularDiffusers.denoise.pipeline_class_from_runtime_inputs", return_value=pipeline_class),
            patch(
                "modules.ModularDiffusers.denoise.require_modiff_node_contract",
                return_value=(denoise_blocks, denoise_config),
            ),
            self.assertRaisesRegex(ValueError, "loader role 'vae'.*not 'denoiser'"),
        ):
            Denoise().execute(
                unet=outputs["vae_out"],
                scheduler=outputs["scheduler"],
                embeddings={"prompt_embeds": object()},
                image_latents=image_latents,
                seed=7,
                route_state_in=encoder_route,
            )
        denoise_blocks.init_pipeline.assert_not_called()

        decode_blocks = Mock()
        decode_blocks.input_names = ["latents"]
        decode_config = {
            "params": {ROUTE_STATE_INPUT: {"type": "modular_route_state"}},
            "model_input_names": ["vae"],
            "input_names": ["latents", ROUTE_STATE_INPUT],
            "output_names": ["images"],
        }
        with (
            patch("modules.ModularDiffusers.latents.pipeline_class_from_runtime_inputs", return_value=pipeline_class),
            patch(
                "modules.ModularDiffusers.latents.require_modiff_node_contract",
                return_value=(decode_blocks, decode_config),
            ),
            self.assertRaisesRegex(ValueError, "loader role 'scheduler'.*not 'vae'"),
        ):
            DecodeLatents().execute(
                vae=outputs["scheduler"],
                latents=denoised_latents,
                route_state_in=decode_route,
            )
        decode_blocks.init_pipeline.assert_not_called()

    def test_invalid_route_seed_and_missing_encode_seed_fail_before_init(self):
        token, outputs = _bound_outputs()
        route, image_latents = _normal_encoder_route(token)
        blocks = Mock()
        blocks.input_names = ["generator", "image_latents"]
        config = _route_node_config()
        pipeline_class = type(QWEN_EDIT, (), {})
        for invalid_seed in (True, 7.5, "07", "-0"):
            node = Denoise()
            with (
                self.subTest(seed=invalid_seed),
                patch(
                    "modules.ModularDiffusers.denoise.pipeline_class_from_runtime_inputs", return_value=pipeline_class
                ),
                patch("modules.ModularDiffusers.denoise.require_modiff_node_contract", return_value=(blocks, config)),
                self.assertRaises(ValueError),
            ):
                node.execute(
                    unet=outputs["unet_out"],
                    scheduler=outputs["scheduler"],
                    embeddings={"prompt_embeds": object()},
                    image_latents=image_latents,
                    seed=invalid_seed,
                    route_state_in=route,
                )
        blocks.init_pipeline.assert_not_called()

        encode_blocks = Mock()
        encode_blocks.input_names = ["image", "generator"]
        encode_config = {
            "params": {"seed": {"type": "int", "min": 0, "max": 4294967295}},
            "model_input_names": ["vae"],
            "input_names": ["image", "seed"],
            "output_names": ["image_latents", ROUTE_STATE_OUTPUT],
        }
        encode_node = ImageEncode()
        with (
            patch("modules.ModularDiffusers.latents.pipeline_class_from_runtime_inputs", return_value=pipeline_class),
            patch(
                "modules.ModularDiffusers.latents.require_modiff_node_contract",
                return_value=(encode_blocks, encode_config),
            ),
            self.assertRaisesRegex(ValueError, "requires a seed before pipeline initialization"),
        ):
            encode_node.execute(vae=outputs["vae_out"], image=object())
        encode_blocks.init_pipeline.assert_not_called()

    def test_qwen_controlnet_rejects_noncanonical_scalars_before_init(self):
        _token, outputs = _bound_outputs(QWEN_IMAGE)
        _issuer, controlnet = _publish_standalone(manager_model_id="controlnet-strict-scalars")
        control_blocks = Mock()
        control_blocks.component_names = ["vae", "controlnet"]
        control_blocks.input_names = ["control_image", "height", "width", "generator"]
        denoise_blocks = Mock()
        denoise_blocks.component_names = ["controlnet"]
        denoise_blocks.input_names = ["control_image_latents"]

        def contract(_pipeline_class, node_type, **_kwargs):
            if node_type == "controlnet":
                return control_blocks, _controlnet_node_config()
            return denoise_blocks, {"input_names": denoise_blocks.input_names}

        base = {
            "controlnet": controlnet,
            "vae": outputs["vae_out"],
            "control_image": object(),
            "controlnet_conditioning_scale": 0.5,
            "control_guidance_start": 0.0,
            "control_guidance_end": 1.0,
            "height": 512,
            "width": 512,
            "seed": 7,
        }
        invalid_values = (
            ("seed", True),
            ("seed", 7.5),
            ("height", True),
            ("height", 512.5),
            ("width", float("inf")),
            ("controlnet_conditioning_scale", True),
            ("controlnet_conditioning_scale", float("nan")),
            ("control_guidance_start", float("-inf")),
            ("control_guidance_end", 1.5),
        )
        for field, value in invalid_values:
            control_blocks.init_pipeline.reset_mock()
            with (
                self.subTest(field=field, value=value),
                patch(
                    "modules.ModularDiffusers.controlnet.pipeline_class_from_runtime_inputs",
                    return_value=diffusers.QwenImageModularPipeline,
                ),
                patch("modules.ModularDiffusers.controlnet.require_modiff_node_contract", side_effect=contract),
                self.assertRaisesRegex(ValueError, f"{field}|modular parameter"),
            ):
                Controlnet().execute(**{**base, field: value})
            control_blocks.init_pipeline.assert_not_called()

    def test_controlnet_cross_loader_and_standalone_provenance_fail_before_init(self):
        token_a, outputs_a = _bound_outputs(QWEN_IMAGE, suffix="a", model_id="control-route-a")
        _token_b, outputs_b = _bound_outputs(QWEN_IMAGE, suffix="b", model_id="control-route-b")
        _issuer, controlnet = _publish_standalone(manager_model_id="controlnet-preinit-provenance")
        image_latents = torch.zeros((1, 1, 2, 2))
        image_route = issue_encoder_route_state(
            binding=token_a,
            seed=7,
            generator=torch.Generator(device="cpu").manual_seed(7),
            image_latents=image_latents,
            processed_mask_image=None,
            mask_overlay_kwargs=None,
        )
        blocks = Mock()
        blocks.component_names = ["vae", "controlnet"]
        blocks.input_names = ["control_image", "generator"]
        denoise_blocks = Mock()
        denoise_blocks.component_names = ["controlnet"]
        denoise_blocks.input_names = ["control_image_latents"]

        def contract(_pipeline_class, node_type, **_kwargs):
            if node_type == "controlnet":
                return blocks, _controlnet_node_config()
            return denoise_blocks, {"input_names": denoise_blocks.input_names}

        base = {
            "vae": outputs_b["vae_out"],
            "controlnet": controlnet,
            "control_image": object(),
            "seed": 7,
            ROUTE_STATE_INPUT: image_route,
        }
        with (
            patch(
                "modules.ModularDiffusers.controlnet.pipeline_class_from_runtime_inputs",
                return_value=diffusers.QwenImageModularPipeline,
            ),
            patch("modules.ModularDiffusers.controlnet.require_modiff_node_contract", side_effect=contract),
            self.assertRaisesRegex(ValueError, "different Models Loader"),
        ):
            Controlnet().execute(**base)
        blocks.init_pipeline.assert_not_called()

        unbound_controlnet = {
            key: value
            for key, value in controlnet.items()
            if isinstance(key, str)
        }
        blocks.init_pipeline.reset_mock()
        with (
            patch(
                "modules.ModularDiffusers.controlnet.pipeline_class_from_runtime_inputs",
                return_value=diffusers.QwenImageModularPipeline,
            ),
            patch("modules.ModularDiffusers.controlnet.require_modiff_node_contract", side_effect=contract),
            self.assertRaisesRegex(ValueError, "missing its process-local"),
        ):
            Controlnet().execute(
                **{
                    **base,
                    "vae": outputs_a["vae_out"],
                    "controlnet": unbound_controlnet,
                    ROUTE_STATE_INPUT: None,
                }
            )
        blocks.init_pipeline.assert_not_called()

    def test_qwen_edit_plus_multi_image_generator_route_reaches_denoise_and_normal_decode(self):
        token, outputs = _bound_outputs(QWEN_EDIT_PLUS)
        seed = 11
        generator = torch.Generator(device="cpu").manual_seed(seed)
        torch.rand((3,), generator=generator)
        advanced_state = generator.get_state().clone()
        source_latents = [torch.zeros((1, 1, 2, 2)), torch.full((1, 1, 2, 2), 2.0)]
        denoised_latents = torch.ones((1, 1, 2, 2))
        route = issue_encoder_route_state(
            binding=token,
            seed=seed,
            generator=generator,
            image_latents=source_latents,
            processed_mask_image=None,
            mask_overlay_kwargs=None,
        )

        received = {}

        class FakePipeline:
            _execution_device = torch.device("cpu")
            component_names = []
            transformer = None
            blocks = type("PipelineBlocks", (), {"doc": "fixture"})()

            def update_components(self, **_kwargs):
                return None

            def __call__(self, **kwargs):
                received.update(kwargs)
                return {"latents": denoised_latents, "mask": None}

        class FakeBlocks:
            component_names = []
            input_names = ["prompt_embeds", "image_latents", "generator"]

            @staticmethod
            def init_pipeline(*, components_manager):
                return FakePipeline()

        config = _route_node_config()
        pipeline_class = diffusers.QwenImageEditPlusModularPipeline
        node = Denoise()
        with (
            patch("modules.ModularDiffusers.denoise.pipeline_class_from_runtime_inputs", return_value=pipeline_class),
            patch(
                "modules.ModularDiffusers.denoise.require_modiff_node_contract", return_value=(FakeBlocks(), config)
            ),
            patch("modules.ModularDiffusers.denoise.collect_model_ids", return_value=[]),
        ):
            result = node.execute(
                unet=outputs["unet_out"],
                scheduler=outputs["scheduler"],
                embeddings={"prompt_embeds": "encoded"},
                image_latents=source_latents,
                seed=seed,
                num_inference_steps=2,
                route_state_in=route,
            )

        self.assertTrue(torch.equal(received["generator"].get_state(), advanced_state))
        self.assertIs(received["image_latents"], source_latents)
        self.assertNotIn("seed", received)
        self.assertNotIn("processed_mask_image", received)
        self.assertIs(result["latents"], denoised_latents)
        decode_values = consume_decode_route_state(
            result[ROUTE_STATE_OUTPUT],
            binding=token,
            model_type=QWEN_EDIT_PLUS,
            latents=result["latents"],
        )
        self.assertFalse(decode_values["inpaint"])
        self.assertIsNone(decode_values["mask_overlay_kwargs"])

        with self.assertRaisesRegex(ValueError, "generator-only route state"):
            issue_encoder_route_state(
                binding=token,
                seed=seed,
                generator=torch.Generator(device="cpu").manual_seed(seed),
                image_latents=source_latents,
                processed_mask_image=torch.ones((1, 1, 8, 8)),
                mask_overlay_kwargs={
                    "crops_coords": None,
                    "original_image": None,
                    "original_mask": None,
                },
            )

    def test_route_cache_requires_exact_single_and_multi_latent_tensor_identities(self):
        for model_type, source_latents in (
            (QWEN_EDIT, torch.zeros((1, 1, 2, 2))),
            (
                QWEN_EDIT_PLUS,
                [torch.zeros((1, 1, 2, 2)), torch.ones((1, 1, 2, 2))],
            ),
        ):
            with self.subTest(model_type=model_type):
                token, outputs = _bound_outputs(model_type)
                route = issue_encoder_route_state(
                    binding=token,
                    seed=7,
                    generator=torch.Generator(device="cpu").manual_seed(7),
                    image_latents=source_latents,
                    processed_mask_image=None,
                    mask_overlay_kwargs=None,
                )
                pipeline_calls = []
                init_calls = []

                class FakePipeline:
                    _execution_device = torch.device("cpu")
                    component_names = []
                    transformer = None
                    blocks = type("PipelineBlocks", (), {"doc": "fixture"})()

                    def update_components(self, **_kwargs):
                        return None

                    def __call__(self, **kwargs):
                        pipeline_calls.append(kwargs)
                        return {"latents": torch.ones((1, 1, 2, 2)), "mask": None}

                class FakeBlocks:
                    component_names = []
                    input_names = ["prompt_embeds", "image_latents", "generator"]

                    def init_pipeline(self, *, components_manager):
                        init_calls.append(components_manager)
                        return FakePipeline()

                pipeline_class = getattr(diffusers, model_type)
                node = Denoise(f"route-cache-{model_type}")
                node.progress = Mock()
                with (
                    patch(
                        "modules.ModularDiffusers.denoise.pipeline_class_from_runtime_inputs",
                        return_value=pipeline_class,
                    ),
                    patch(
                        "modules.ModularDiffusers.denoise.require_modiff_node_contract",
                        return_value=(FakeBlocks(), _route_node_config()),
                    ),
                    patch("modules.ModularDiffusers.denoise.collect_model_ids", return_value=[]),
                    patch("modules.ModularDiffusers.denoise.insert_preview_block"),
                ):
                    common = {
                        "unet": outputs["unet_out"],
                        "scheduler": outputs["scheduler"],
                        "embeddings": {"prompt_embeds": "encoded"},
                        "seed": 7,
                        "num_inference_steps": 2,
                        ROUTE_STATE_INPUT: route,
                    }
                    first = node(**common, image_latents=source_latents)
                    rewrapped = list(source_latents) if type(source_latents) is list else source_latents
                    second = node(**common, image_latents=rewrapped)

                    self.assertIs(first, second)
                    self.assertEqual(len(init_calls), 1)
                    self.assertEqual(len(pipeline_calls), 1)

                    cloned = (
                        [latent.clone() for latent in source_latents]
                        if type(source_latents) is list
                        else source_latents.clone()
                    )
                    with self.assertRaisesRegex(RuntimeError, "exact latent output paired"):
                        node(**common, image_latents=cloned)

                self.assertEqual(len(init_calls), 1)
                self.assertEqual(len(pipeline_calls), 1)

    def test_controlnet_cache_revalidates_current_standalone_publication_before_hit(self):
        _token, outputs = _bound_outputs(QWEN_IMAGE)
        identity = _standalone_identity(fingerprint="3" * 64)
        issuer, controlnet_a = _publish_standalone(
            identity,
            manager_model_id="controlnet-node-cache-publication",
        )
        control_latents = torch.zeros((1, 1, 2, 2))
        init_calls = []
        pipeline_calls = []

        class FakeControlOutput:
            values = {"control_image_latents": control_latents}

        class FakePipeline:
            _execution_device = torch.device("cpu")

            def update_components(self, **_kwargs):
                return None

            def __call__(self, **kwargs):
                pipeline_calls.append(kwargs)
                torch.rand((), generator=kwargs["generator"])
                return FakeControlOutput()

        class FakeControlBlocks:
            component_names = ["vae", "controlnet"]
            input_names = ["control_image", "height", "width", "generator"]

            def init_pipeline(self, *, components_manager):
                init_calls.append(components_manager)
                return FakePipeline()

        class FakeDenoiseBlocks:
            input_names = ["control_image_latents"]
            component_names = ["controlnet"]

        def contract(_pipeline_class, node_type, **_kwargs):
            if node_type == "controlnet":
                return FakeControlBlocks(), _controlnet_node_config()
            return FakeDenoiseBlocks(), {"input_names": FakeDenoiseBlocks.input_names}

        control_image_alias = {"pixels": object()}
        common = {
            "controlnet": controlnet_a,
            "vae": outputs["vae_out"],
            "control_image": control_image_alias,
            "height": 512,
            "width": 512,
            "seed": 7,
        }
        node = Controlnet("controlnet-route-cache-publication")
        with (
            patch(
                "modules.ModularDiffusers.controlnet.pipeline_class_from_runtime_inputs",
                return_value=diffusers.QwenImageModularPipeline,
            ),
            patch("modules.ModularDiffusers.controlnet.require_modiff_node_contract", side_effect=contract),
            patch("modules.ModularDiffusers.controlnet.collect_model_ids", return_value=[]),
        ):
            first = node(**common)
            self.assertIs(node(**common), first)
            self.assertEqual(len(init_calls), 1)

            control_image_alias["generator"] = object()
            with self.assertRaisesRegex(ValueError, "backend-managed"):
                node(**common)
            control_image_alias.pop("generator")
            self.assertEqual(len(init_calls), 1)
            self.assertEqual(len(pipeline_calls), 1)

            _issuer, controlnet_b = _publish_standalone(
                identity,
                issuer=issuer,
                manager_model_id="controlnet-node-cache-publication",
            )
            with self.assertRaisesRegex(ValueError, "no longer the current"):
                node(**common)
            self.assertEqual(len(init_calls), 1)
            self.assertEqual(len(pipeline_calls), 1)

            second = node(**{**common, "controlnet": controlnet_b})
            self.assertIsNot(second, first)

        self.assertEqual(len(init_calls), 2)
        self.assertEqual(len(pipeline_calls), 2)

    def test_controlnet_cache_revalidates_the_inherited_route_before_hit(self):
        token, outputs = _bound_outputs(QWEN_IMAGE)
        _issuer, controlnet = _publish_standalone(manager_model_id="controlnet-cache-input-route")

        def route_without_retaining_image_latents():
            transient = torch.zeros((1, 1, 2, 2))
            return issue_encoder_route_state(
                binding=token,
                seed=7,
                generator=torch.Generator(device="cpu").manual_seed(7),
                image_latents=transient,
                processed_mask_image=None,
                mask_overlay_kwargs=None,
            )

        route = route_without_retaining_image_latents()
        gc.collect()
        params = {
            "controlnet": controlnet,
            "vae": outputs["vae_out"],
            "control_image": object(),
            "seed": 7,
            ROUTE_STATE_INPUT: route,
        }
        node = Controlnet()
        node._model_type = QWEN_IMAGE
        with self.assertRaisesRegex(ValueError, "no longer resident"):
            node._cache_params_equal(params, params)

    def test_denoise_cache_rejects_control_route_after_publication_is_superseded(self):
        token, outputs = _bound_outputs(QWEN_IMAGE)
        identity = _standalone_identity(fingerprint="4" * 64)
        issuer, controlnet_a = _publish_standalone(
            identity,
            manager_model_id="controlnet-denoise-cache-publication",
        )
        control_latents = torch.zeros((1, 1, 2, 2))
        control_route = issue_controlnet_route_state(
            None,
            binding=token,
            controlnet_component=controlnet_a,
            seed=7,
            generator=torch.Generator(device="cpu").manual_seed(7),
            control_image_latents=control_latents,
        )
        init_calls = []
        pipeline_calls = []

        class FakePipeline:
            _execution_device = torch.device("cpu")
            component_names = ["controlnet"]
            transformer = None
            blocks = type("PipelineBlocks", (), {"doc": "fixture"})()

            def update_components(self, **_kwargs):
                return None

            def __call__(self, **kwargs):
                pipeline_calls.append(kwargs)
                return {"latents": torch.ones((1, 1, 2, 2)), "mask": None}

        class FakeBlocks:
            component_names = ["controlnet"]
            input_names = ["prompt_embeds", "control_image_latents", "generator"]

            def init_pipeline(self, *, components_manager):
                init_calls.append(components_manager)
                return FakePipeline()

        bundle = {"controlnet": controlnet_a, "control_image_latents": control_latents}
        common = {
            "unet": outputs["unet_out"],
            "scheduler": outputs["scheduler"],
            "embeddings": {"prompt_embeds": "encoded"},
            "controlnet_bundle": bundle,
            "seed": 7,
            ROUTE_STATE_INPUT: control_route,
        }
        node = Denoise("denoise-controlnet-route-cache-publication")
        node.progress = Mock()
        with (
            patch(
                "modules.ModularDiffusers.denoise.pipeline_class_from_runtime_inputs",
                return_value=diffusers.QwenImageModularPipeline,
            ),
            patch(
                "modules.ModularDiffusers.denoise.require_modiff_node_contract",
                return_value=(FakeBlocks(), _route_node_config(control_bundle=True)),
            ),
            patch("modules.ModularDiffusers.denoise.collect_model_ids", return_value=[]),
            patch("modules.ModularDiffusers.denoise.insert_preview_block"),
        ):
            first = node(**common)
            self.assertIs(node(**common), first)
            self.assertEqual(len(init_calls), 1)

            bundle["control_image_latents"] = control_latents.clone()
            with self.assertRaisesRegex(ValueError, "exact latent output paired"):
                node(**common)
            bundle["control_image_latents"] = control_latents
            self.assertEqual(len(init_calls), 1)
            self.assertEqual(len(pipeline_calls), 1)

            bundle["generator"] = object()
            with self.assertRaisesRegex(ValueError, "backend-managed"):
                node(**common)
            bundle.pop("generator")
            self.assertEqual(len(init_calls), 1)

            original_repo_id = controlnet_a["repo_id"]
            controlnet_a["repo_id"] = "fixture/tampered-after-cache"
            with self.assertRaisesRegex(ValueError, "provenance binding"):
                node(**common)
            controlnet_a["repo_id"] = original_repo_id
            self.assertEqual(len(init_calls), 1)

            _issuer, _controlnet_b = _publish_standalone(
                identity,
                issuer=issuer,
                manager_model_id="controlnet-denoise-cache-publication",
            )
            with self.assertRaisesRegex(ValueError, "superseded"):
                node(**common)

        self.assertEqual(len(init_calls), 1)
        self.assertEqual(len(pipeline_calls), 1)

    def test_denoise_cache_revalidates_same_object_image_latent_bundle_aliases(self):
        token, outputs = _bound_outputs(QWEN_EDIT)
        image_latents = torch.zeros((1, 1, 2, 2))
        route = issue_encoder_route_state(
            binding=token,
            seed=7,
            generator=torch.Generator(device="cpu").manual_seed(7),
            image_latents=image_latents,
            processed_mask_image=None,
            mask_overlay_kwargs=None,
        )
        latent_bundle = {"image_latents": image_latents}
        denoised_latents = torch.ones((1, 1, 2, 2))
        init_calls = []
        pipeline_calls = []

        class FakePipeline:
            _execution_device = torch.device("cpu")
            component_names = []
            transformer = None
            blocks = type("PipelineBlocks", (), {"doc": "fixture"})()

            def update_components(self, **_kwargs):
                return None

            def __call__(self, **kwargs):
                pipeline_calls.append(kwargs)
                return {"latents": denoised_latents, "mask": None}

        class FakeBlocks:
            component_names = []
            input_names = ["prompt_embeds", "image_latents", "generator"]

            def init_pipeline(self, *, components_manager):
                init_calls.append(components_manager)
                return FakePipeline()

        config = _route_node_config()
        config["params"].pop("image_latents")
        config["params"]["image_latents_with_strength"] = {"type": "latents"}
        config["input_names"] = ["embeddings", "image_latents_with_strength", "seed", ROUTE_STATE_INPUT]
        common = {
            "unet": outputs["unet_out"],
            "scheduler": outputs["scheduler"],
            "embeddings": {"prompt_embeds": "encoded"},
            "image_latents_with_strength": latent_bundle,
            "seed": 7,
            ROUTE_STATE_INPUT: route,
        }
        node = Denoise("denoise-image-bundle-alias-cache")
        node.progress = Mock()
        with (
            patch(
                "modules.ModularDiffusers.denoise.pipeline_class_from_runtime_inputs",
                return_value=diffusers.QwenImageEditModularPipeline,
            ),
            patch(
                "modules.ModularDiffusers.denoise.require_modiff_node_contract",
                return_value=(FakeBlocks(), config),
            ),
            patch("modules.ModularDiffusers.denoise.collect_model_ids", return_value=[]),
            patch("modules.ModularDiffusers.denoise.insert_preview_block"),
        ):
            first = node(**common)
            self.assertIs(node(**common), first)
            latent_bundle["image_latents"] = image_latents.clone()
            with self.assertRaisesRegex(ValueError, "exact latent output paired"):
                node(**common)

        self.assertEqual(len(init_calls), 1)
        self.assertEqual(len(pipeline_calls), 1)

    def test_route_less_qwen_denoise_cache_revalidates_bound_model_metadata(self):
        _token, outputs = _bound_outputs(QWEN_IMAGE)
        denoised_latents = torch.ones((1, 1, 2, 2))
        init_calls = []
        pipeline_calls = []

        class FakePipeline:
            _execution_device = torch.device("cpu")
            component_names = []
            transformer = None
            blocks = type("PipelineBlocks", (), {"doc": "fixture"})()

            def update_components(self, **_kwargs):
                return None

            def __call__(self, **kwargs):
                pipeline_calls.append(kwargs)
                return {"latents": denoised_latents, "mask": None}

        class FakeBlocks:
            component_names = []
            input_names = ["prompt_embeds", "generator"]

            def init_pipeline(self, *, components_manager):
                init_calls.append(components_manager)
                return FakePipeline()

        common = {
            "unet": outputs["unet_out"],
            "scheduler": outputs["scheduler"],
            "embeddings": {"prompt_embeds": "encoded"},
            "seed": 7,
        }
        node = Denoise("denoise-text-model-alias-cache")
        node.progress = Mock()
        with (
            patch(
                "modules.ModularDiffusers.denoise.pipeline_class_from_runtime_inputs",
                return_value=diffusers.QwenImageModularPipeline,
            ),
            patch(
                "modules.ModularDiffusers.denoise.require_modiff_node_contract",
                return_value=(FakeBlocks(), _route_node_config()),
            ),
            patch("modules.ModularDiffusers.denoise.collect_model_ids", return_value=[]),
            patch("modules.ModularDiffusers.denoise.insert_preview_block"),
        ):
            first = node(**common)
            self.assertIs(node(**common), first)
            original_repo_id = outputs["unet_out"]["repo_id"]
            outputs["unet_out"]["repo_id"] = "fixture/tampered-denoiser-cache"
            with self.assertRaisesRegex(ValueError, "does not match"):
                node(**common)
            outputs["unet_out"]["repo_id"] = original_repo_id

        self.assertEqual(len(init_calls), 1)
        self.assertEqual(len(pipeline_calls), 1)

    def test_decode_route_cache_requires_exact_denoised_latent_identity(self):
        token, outputs = _bound_outputs(QWEN_EDIT)
        latents = torch.zeros((1, 1, 2, 2))
        route = issue_normal_decode_route_state(binding=token, latents=latents)
        pipeline_calls = []
        init_calls = []

        class FakePipeline:
            blocks = type("PipelineBlocks", (), {"doc": "fixture"})()

            def update_components(self, **_kwargs):
                return None

            def __call__(self, **kwargs):
                pipeline_calls.append(kwargs)
                return {"images": "decoded"}

        class FakeBlocks:
            component_names = []
            input_names = ["latents"]

            def init_pipeline(self, *, components_manager):
                init_calls.append(components_manager)
                return FakePipeline()

        config = {
            "params": {ROUTE_STATE_INPUT: {"type": "modular_route_state"}},
            "model_input_names": ["vae"],
            "input_names": ["latents", ROUTE_STATE_INPUT],
            "output_names": ["images"],
        }
        node = DecodeLatents("decode-route-cache")
        with (
            patch(
                "modules.ModularDiffusers.latents.pipeline_class_from_runtime_inputs",
                return_value=diffusers.QwenImageEditModularPipeline,
            ),
            patch(
                "modules.ModularDiffusers.latents.require_modiff_node_contract",
                return_value=(FakeBlocks(), config),
            ),
            patch("modules.ModularDiffusers.latents.collect_model_ids", return_value=[]),
        ):
            common = {
                "vae": outputs["vae_out"],
                ROUTE_STATE_INPUT: route,
            }
            first = node(**common, latents=latents)
            second = node(**common, latents=latents)
            self.assertIs(first, second)
            self.assertEqual(len(init_calls), 1)
            self.assertEqual(len(pipeline_calls), 1)

            with self.assertRaisesRegex(RuntimeError, "exact latent output paired"):
                node(**common, latents=latents.clone())

        self.assertEqual(len(init_calls), 1)
        self.assertEqual(len(pipeline_calls), 1)

    def test_same_type_dynamic_resync_reemits_authoritative_route_definition(self):
        cases = (
            (ImageEncode, "vae_encoder", "vae"),
            (Denoise, "denoise", "unet"),
            (DecodeLatents, "decoder", "vae"),
        )
        pipeline_class = type(QWEN_EDIT, (), {})
        for node_class, node_type, base_field in cases:
            with self.subTest(node=node_class.__name__):
                node = node_class("same-node-id")
                node._model_type = QWEN_EDIT
                node._pipeline_class = pipeline_class
                node.get_signal_value = Mock(return_value=QWEN_EDIT)
                node.send_node_definition = Mock()
                config = {
                    "params": {
                        base_field: {"type": "diffusers_auto_model"},
                        ROUTE_STATE_INPUT: {"type": "modular_route_state"},
                        ROUTE_STATE_OUTPUT: {"type": "modular_route_state"},
                    }
                }
                with patch(
                    f"{node_class.__module__}.require_modiff_node_contract",
                    return_value=(None, config),
                ) as contract:
                    node.update_node({}, None)
                contract.assert_called_once_with(pipeline_class, node_type, resolve_blocks=False)
                sent = node.send_node_definition.call_args.args[0]
                self.assertNotIn(base_field, sent)
                self.assertTrue({ROUTE_STATE_INPUT, ROUTE_STATE_OUTPUT}.issubset(sent))

    def test_controlnet_same_type_dynamic_resync_reemits_authoritative_route_definition(self):
        pipeline_class = type(QWEN_IMAGE, (), {})
        node = Controlnet("same-controlnet-node-id")
        node._model_type = QWEN_IMAGE
        node._pipeline_class = pipeline_class
        node.send_node_definition = Mock()
        config = {
            "params": {
                "controlnet": {"type": "diffusers_auto_model"},
                "controlnet_bundle": {"type": "custom_controlnet"},
                ROUTE_STATE_INPUT: {"type": "modular_route_state"},
                ROUTE_STATE_OUTPUT: {"type": "modular_route_state"},
            }
        }
        with patch(
            "modules.ModularDiffusers.controlnet.require_modiff_node_contract",
            return_value=(None, config),
        ) as contract:
            node.update_node({"model_type": QWEN_IMAGE}, None)

        contract.assert_called_once_with(
            pipeline_class,
            "controlnet",
            require_blocks=False,
            resolve_blocks=False,
        )
        sent = node.send_node_definition.call_args.args[0]
        self.assertTrue({"controlnet", "controlnet_bundle", ROUTE_STATE_INPUT, ROUTE_STATE_OUTPUT}.issubset(sent))


class PinnedRouteSchemaTests(unittest.TestCase):
    def test_qwen_control_truth_binds_control_to_denoise_without_claiming_direct_vae_route(self):
        mode = PINNED_MODULAR_WORKFLOW_TRUTH["QwenImageModularPipeline"].mode("control_image")
        edges = {
            (edge.producer_action, edge.producer_output, edge.consumer_action, edge.consumer_input)
            for edge in mode.state_edges
        }
        self.assertIn(("controlnet", ROUTE_STATE_OUTPUT, "denoise", ROUTE_STATE_INPUT), edges)
        self.assertIn(("denoise", ROUTE_STATE_OUTPUT, "decoder", ROUTE_STATE_INPUT), edges)
        self.assertNotIn(("vae_encoder", ROUTE_STATE_OUTPUT, "denoise", ROUTE_STATE_INPUT), edges)

    def test_only_reviewed_image_routes_expose_opaque_handles(self):
        supported = {
            "QwenImageModularPipeline",
            QWEN_EDIT,
            QWEN_EDIT_PLUS,
            SDXL,
            "WanImage2VideoModularPipeline",
        }
        for model_type in PINNED_MODULAR_WORKFLOW_TRUTH:
            metadata = get_model_type_metadata(model_type)
            actions = metadata["node_params"]
            for action in ("vae_encoder", "denoise", "decoder"):
                config = actions.get(action)
                if config is None:
                    continue
                names = set(config["input_names"] + config["output_names"])
                if model_type in supported:
                    expected = {
                        "vae_encoder": {ROUTE_STATE_OUTPUT},
                        "denoise": {ROUTE_STATE_INPUT, ROUTE_STATE_OUTPUT},
                        "decoder": {ROUTE_STATE_INPUT},
                    }[action]
                    self.assertTrue(expected.issubset(names), (model_type, action))
                else:
                    self.assertFalse({ROUTE_STATE_INPUT, ROUTE_STATE_OUTPUT} & names, (model_type, action))

    def test_qwen_controlnet_declares_optional_route_and_seed_without_a_large_image_latent_edge(self):
        qwen_control = get_model_type_metadata(QWEN_IMAGE)["node_params"]["controlnet"]
        self.assertIn("seed", qwen_control["input_names"])
        self.assertIn(ROUTE_STATE_INPUT, qwen_control["input_names"])
        self.assertIn(ROUTE_STATE_OUTPUT, qwen_control["output_names"])
        self.assertFalse(qwen_control["params"][ROUTE_STATE_INPUT]["label"].endswith("*"))
        self.assertFalse(
            {"image_latents", "image_latents_with_strength", "strength"}.intersection(qwen_control["input_names"])
        )

        sdxl_control = get_model_type_metadata("StableDiffusionXLModularPipeline")["node_params"]["controlnet"]
        self.assertFalse({"seed", ROUTE_STATE_INPUT}.intersection(sdxl_control["input_names"]))
        self.assertNotIn(ROUTE_STATE_OUTPUT, sdxl_control["output_names"])

    @unittest.skipUnless(
        importlib.util.find_spec("transformers"),
        "requires the staged optional Transformers runtime",
    )
    def test_qwen_vae_route_inputs_and_edit_decoder_anomaly_are_pinned_exactly(self):
        image = get_model_type_metadata("QwenImageModularPipeline")["node_params"]["vae_encoder"]
        edit = get_model_type_metadata(QWEN_EDIT)["node_params"]["vae_encoder"]
        self.assertTrue(
            {"image", "mask_image", "padding_mask_crop", "height", "width", "seed"}.issubset(image["input_names"])
        )
        self.assertTrue({"image", "mask_image", "padding_mask_crop", "seed"}.issubset(edit["input_names"]))

        upstream_decoder = diffusers.QwenImageEditModularPipeline().blocks.sub_blocks["decode"]
        self.assertEqual(upstream_decoder.input_names, ["latents", "output_type", "mask_overlay_kwargs"])
        self.assertEqual(
            upstream_decoder.output_names,
            ["latents"],
            "Pinned Qwen Edit declares anomalous decoder metadata; MoDiff must not infer graph image output from it.",
        )


if __name__ == "__main__":
    unittest.main()
