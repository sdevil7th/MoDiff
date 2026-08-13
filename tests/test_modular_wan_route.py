import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import diffusers
import torch
from PIL import Image

from modiff.modular_workflow_contracts import PINNED_MODULAR_WORKFLOW_TRUTH
from modiff.modular_workflow_contracts import (
    WAN_FLF_REPOSITORY,
    WAN_I2V_720P_REPOSITORY,
    WAN_I2V_REPOSITORY,
)
from modules.ModularDiffusers.denoise import Denoise
from modules.ModularDiffusers.embeddings import ImageEmbeddings
from modules.ModularDiffusers.latents import DecodeLatents, ImageEncode
from modules.ModularDiffusers.loaders import annotate_modular_loader_outputs
from modules.ModularDiffusers.modular_utils import get_model_type_metadata
from modules.ModularDiffusers.route_state import (
    ROUTE_STATE_INPUT,
    ROUTE_STATE_OUTPUT,
    consume_decode_route_state,
    consume_wan_vae_route_state,
    issue_decode_route_state,
    issue_pipeline_instance_token,
    issue_wan_image_encoder_route_state,
    issue_wan_vae_route_state,
    preflight_wan_image_encoder_inputs,
    preflight_wan_vae_route_state,
    require_cataloged_wan_action_source,
    snapshot_wan_source_media,
    validate_wan_image_encoder_route_state,
    validate_wan_post_vae_route_state,
    wan_area_budget_dimensions,
    wan_image_processor_config_seal,
    wan_transformer_contract_from_component,
    wan_video_processor_config_seal,
)


WAN_I2V = "WanImage2VideoModularPipeline"
WAN_LATENTS_MEAN = (
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
)
WAN_LATENTS_STD = (
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.9160,
)


class _Size:
    def __init__(
        self,
        *,
        height=None,
        width=None,
        longest_edge=None,
        shortest_edge=None,
        max_height=None,
        max_width=None,
    ):
        self.height = height
        self.width = width
        self.longest_edge = longest_edge
        self.shortest_edge = shortest_edge
        self.max_height = max_height
        self.max_width = max_width


class _ImageProcessor:
    def __init__(self):
        self.do_resize = True
        self.size = _Size(height=224, width=224)
        self.resample = 3
        self.do_center_crop = False
        self.crop_size = _Size(height=224, width=224)
        self.do_rescale = True
        self.rescale_factor = 1 / 255
        self.do_normalize = True
        self.image_mean = (0.48145466, 0.4578275, 0.40821073)
        self.image_std = (0.26862954, 0.26130258, 0.27577711)
        self.do_convert_rgb = True
        self.do_pad = None
        self.pad_size = None
        self.disable_grouping = None


class _FlfImageProcessor(_ImageProcessor):
    def __init__(self):
        super().__init__()
        self.size = _Size(shortest_edge=224)
        self.do_center_crop = True


class _ImageEncoder:
    def __init__(self):
        self.config = SimpleNamespace(
            image_size=224,
            hidden_size=1280,
            patch_size=14,
            projection_dim=1024,
            num_channels=3,
            num_hidden_layers=32,
            num_attention_heads=16,
        )


class _WanVae:
    def __init__(self):
        self.temperal_downsample = [False, True, True]
        self.config = SimpleNamespace(
            z_dim=16,
            latents_mean=list(WAN_LATENTS_MEAN),
            latents_std=list(WAN_LATENTS_STD),
            in_channels=3,
            out_channels=3,
            patch_size=None,
            scale_factor_spatial=8,
            scale_factor_temporal=4,
        )


class _VideoProcessor:
    def __init__(self):
        self.config = {
            "do_resize": True,
            "vae_scale_factor": 8,
            "vae_latent_channels": 4,
            "resample": "lanczos",
            "reducing_gap": None,
            "do_normalize": True,
            "do_binarize": False,
            "do_convert_rgb": False,
            "do_convert_grayscale": False,
        }


class _Transformer:
    def __init__(self):
        self.config = SimpleNamespace(
            patch_size=(1, 2, 2),
            in_channels=36,
            out_channels=16,
            image_dim=1280,
        )


class _FlfTransformer(_Transformer):
    def __init__(self):
        super().__init__()
        self.config.pos_embed_seq_len = 514


WAN_I2V_REVISION = "b184e23a8a16b20f108f727c902e769e873ffc73"
WAN_I2V_720P_REVISION = "eb849f76dfa246545b65774a9e25943ee69b3fa3"
WAN_FLF_REVISION = "17c30769b1e0b5dcaa1799b117bf20a9c31f59d7"


def _bound_outputs(*, repository=WAN_I2V_REPOSITORY, revision=WAN_I2V_REVISION):
    token = issue_pipeline_instance_token(
        model_type=WAN_I2V,
        repo_id=repository,
        repo_source="hub",
        revision=revision,
    )
    outputs = {
        "unet_out": {"model_id": "wan-transformer"},
        "vae_out": {"model_id": "wan-vae"},
        "scheduler": {"model_id": "wan-scheduler"},
        "image_encoder": {"model_id": "wan-image-encoder"},
        "text_encoders": {"text_encoder": {"model_id": "wan-text-encoder"}},
    }
    annotate_modular_loader_outputs(
        outputs,
        repo_id=repository,
        repo_source="hub",
        model_type=WAN_I2V,
        revision=revision,
        trust_remote_code=False,
        pipeline_instance_token=token,
    )
    return token, outputs


def _issue_image_route(
    *,
    image=None,
    last_image=None,
    height=64,
    width=64,
    image_processor=None,
    image_encoder=None,
):
    token, outputs = _bound_outputs(
        repository=WAN_FLF_REPOSITORY if last_image is not None else WAN_I2V_REPOSITORY,
        revision=WAN_FLF_REVISION if last_image is not None else WAN_I2V_REVISION,
    )
    image = image or Image.new("RGB", (64, 64), "red")
    image_processor = image_processor or (_FlfImageProcessor() if last_image is not None else _ImageProcessor())
    image_encoder = image_encoder or _ImageEncoder()
    preflight = preflight_wan_image_encoder_inputs(
        image=image,
        last_image=last_image,
        height=height,
        width=width,
    )
    first_height, first_width = preflight[3:5]
    workflow = preflight[0]
    image_embeds = torch.zeros((2 if workflow == "flf2v" else 1, 257, 1280))
    route = issue_wan_image_encoder_route_state(
        binding=token,
        image=image,
        last_image=last_image,
        height=height,
        width=width,
        image_embeds=image_embeds,
        image_encoder=image_encoder,
        image_processor=image_processor,
        resized_image=Image.new("L", (first_width, first_height)),
        resized_last_image=(Image.new("L", preflight[7]) if last_image is not None else None),
        execution_device="cpu",
        source_snapshot=snapshot_wan_source_media(image, last_image),
        preflight_geometry=preflight,
    )
    return {
        "binding": token,
        "outputs": outputs,
        "image": image,
        "last_image": last_image,
        "height": height,
        "width": width,
        "image_embeds": image_embeds,
        "image_encoder": image_encoder,
        "image_processor": image_processor,
        "preflight": preflight,
        "route": route,
    }


def _issue_vae_route(*, num_frames=5, seed=7, **image_kwargs):
    values = _issue_image_route(**image_kwargs)
    vae = _WanVae()
    video_processor = _VideoProcessor()
    preflight = preflight_wan_vae_route_state(
        values["route"],
        binding=values["binding"],
        model_type=WAN_I2V,
        image=values["image"],
        last_image=values["last_image"],
        height=values["height"],
        width=values["width"],
        num_frames=num_frames,
        vae_component=vae,
    )
    second_height, second_width = preflight[3:5]
    temporal_frames = (num_frames - 1) // 4 + 1
    raw_frame_latents = torch.zeros((1, 16, temporal_frames, second_height // 8, second_width // 8))
    image_condition_latents = torch.zeros((1, 20, temporal_frames, second_height // 8, second_width // 8))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    route = issue_wan_vae_route_state(
        values["route"],
        binding=values["binding"],
        seed=seed,
        generator=generator,
        image=values["image"],
        last_image=values["last_image"],
        height=values["height"],
        width=values["width"],
        num_frames=num_frames,
        image_condition_latents=image_condition_latents,
        raw_frame_latents=raw_frame_latents,
        vae_component=vae,
        video_processor=video_processor,
        resized_image=Image.new("L", (second_width, second_height)),
        resized_last_image=(Image.new("L", preflight[10]) if values["last_image"] is not None else None),
        execution_device="cpu",
        preflight_geometry=preflight,
    )
    values.update(
        num_frames=num_frames,
        seed=seed,
        vae=vae,
        video_processor=video_processor,
        vae_preflight=preflight,
        raw_frame_latents=raw_frame_latents,
        image_condition_latents=image_condition_latents,
        generator=generator,
        route=route,
    )
    return values


def _vae_validation(values, **overrides):
    kwargs = {
        "binding": values["binding"],
        "model_type": WAN_I2V,
        "seed": values["seed"],
        "image_condition_latents": values["image_condition_latents"],
        "height": values["height"],
        "width": values["width"],
        "num_frames": values["num_frames"],
        "vae_component": values["vae"],
        "video_processor": values["video_processor"],
    }
    kwargs.update(overrides)
    return validate_wan_post_vae_route_state(values["route"], **kwargs)


def _issue_decode_values():
    values = _issue_vae_route()
    transformer = _Transformer()
    latents = torch.zeros_like(values["raw_frame_latents"])
    route = issue_decode_route_state(
        values["route"],
        binding=values["binding"],
        latents=latents,
        vae_component=values["vae"],
        transformer_component=transformer,
        execution_device="cpu",
    )
    values.update(transformer=transformer, latents=latents, route=route)
    return values


class WanRouteStateTests(unittest.TestCase):
    def test_each_workflow_requires_its_exact_reviewed_loader_artifact(self):
        image = Image.new("RGB", (64, 64))
        last_image = Image.new("RGB", (64, 64))
        i2v_token, _outputs = _bound_outputs()
        i2v_720p_token, _outputs = _bound_outputs(
            repository=WAN_I2V_720P_REPOSITORY,
            revision=WAN_I2V_720P_REVISION,
        )
        flf_token, _outputs = _bound_outputs(
            repository=WAN_FLF_REPOSITORY,
            revision=WAN_FLF_REVISION,
        )
        self.assertEqual(
            require_cataloged_wan_action_source(image=image, last_image=None, binding=i2v_token),
            "image2video",
        )
        self.assertEqual(
            require_cataloged_wan_action_source(
                image=image,
                last_image=None,
                binding=i2v_720p_token,
            ),
            "image2video",
        )
        self.assertEqual(
            require_cataloged_wan_action_source(image=image, last_image=last_image, binding=flf_token),
            "flf2v",
        )
        wrong_revision_token, _outputs = _bound_outputs(
            repository=WAN_I2V_720P_REPOSITORY,
            revision=WAN_I2V_REVISION,
        )
        with self.assertRaisesRegex(ValueError, "reviewed immutable"):
            require_cataloged_wan_action_source(
                image=image,
                last_image=None,
                binding=wrong_revision_token,
            )
        for binding, ending in (
            (i2v_token, last_image),
            (i2v_720p_token, last_image),
            (flf_token, None),
        ):
            with self.subTest(repository=binding._repo_id), self.assertRaisesRegex(ValueError, "reviewed immutable"):
                require_cataloged_wan_action_source(image=image, last_image=ending, binding=binding)

    def test_exact_two_pass_geometry_examples(self):
        portrait = Image.new("RGB", (100, 200))
        landscape = Image.new("RGB", (1200, 600))
        self.assertEqual(wan_area_budget_dimensions(portrait, 480, 832), (880, 432))
        self.assertEqual(wan_area_budget_dimensions(portrait, 880, 432), (864, 432))
        self.assertEqual(wan_area_budget_dimensions(landscape, 480, 832), (432, 880))
        self.assertEqual(wan_area_budget_dimensions(landscape, 432, 880), (432, 864))

    def test_i2v_and_internal_flf_chain_end_to_end_without_raw_latent_edge(self):
        for last_image, num_frames in ((None, 1), (Image.new("RGB", (64, 64), "blue"), 5)):
            with self.subTest(flf=last_image is not None):
                values = _issue_vae_route(last_image=last_image, num_frames=num_frames)
                transformer = _FlfTransformer() if last_image is not None else _Transformer()
                runtime_a = consume_wan_vae_route_state(
                    values["route"],
                    binding=values["binding"],
                    model_type=WAN_I2V,
                    seed=values["seed"],
                    execution_device="cpu",
                    image_embeds=values["image_embeds"],
                    image_condition_latents=values["image_condition_latents"],
                    height=values["height"],
                    width=values["width"],
                    num_frames=num_frames,
                    vae_component=values["vae"],
                    transformer_component=transformer,
                )
                runtime_b = consume_wan_vae_route_state(
                    values["route"],
                    binding=values["binding"],
                    model_type=WAN_I2V,
                    seed=values["seed"],
                    execution_device="cpu",
                    image_embeds=values["image_embeds"],
                    image_condition_latents=values["image_condition_latents"],
                    height=values["height"],
                    width=values["width"],
                    num_frames=num_frames,
                    vae_component=values["vae"],
                    transformer_component=transformer,
                )
                self.assertTrue(torch.equal(runtime_a["generator"].get_state(), runtime_b["generator"].get_state()))
                self.assertEqual(
                    (runtime_a["height"], runtime_a["width"]),
                    values["vae_preflight"][3:5],
                )
                denoised = torch.zeros_like(values["raw_frame_latents"])
                decode_route = issue_decode_route_state(
                    values["route"],
                    binding=values["binding"],
                    latents=denoised,
                    vae_component=values["vae"],
                    transformer_component=transformer,
                    execution_device="cpu",
                )
                decode = consume_decode_route_state(
                    decode_route,
                    binding=values["binding"],
                    model_type=WAN_I2V,
                    latents=denoised,
                    vae_component=values["vae"],
                    video_processor=_VideoProcessor(),
                    execution_device="cpu",
                    materialize_overlay=False,
                )
                self.assertEqual(decode["contract"], "wan_i2v")
                self.assertFalse(hasattr(decode_route._payload, "_raw_frame_latents_ref"))

    def test_frame_rules_and_default_video_budget(self):
        image_values = _issue_image_route(image=Image.new("RGB", (832, 480)), height=480, width=832)
        vae = _WanVae()
        default = preflight_wan_vae_route_state(
            image_values["route"],
            binding=image_values["binding"],
            model_type=WAN_I2V,
            image=image_values["image"],
            last_image=None,
            height=480,
            width=832,
            num_frames=81,
            vae_component=vae,
        )
        self.assertLessEqual(default[-1], 512 * 1024 * 1024)
        for valid_frames in (1, 5):
            preflight_wan_vae_route_state(
                image_values["route"],
                binding=image_values["binding"],
                model_type=WAN_I2V,
                image=image_values["image"],
                last_image=None,
                height=480,
                width=832,
                num_frames=valid_frames,
                vae_component=vae,
            )
        small_values = _issue_image_route()
        preflight_wan_vae_route_state(
            small_values["route"],
            binding=small_values["binding"],
            model_type=WAN_I2V,
            image=small_values["image"],
            last_image=None,
            height=64,
            width=64,
            num_frames=477,
            vae_component=vae,
        )
        for invalid_frames in (0, 2, 480):
            with self.subTest(invalid_frames=invalid_frames), self.assertRaises(ValueError):
                preflight_wan_vae_route_state(
                    image_values["route"],
                    binding=image_values["binding"],
                    model_type=WAN_I2V,
                    image=image_values["image"],
                    last_image=None,
                    height=480,
                    width=832,
                    num_frames=invalid_frames,
                    vae_component=vae,
                )
        flf_values = _issue_image_route(last_image=Image.new("RGB", (64, 64)))
        with self.assertRaisesRegex(ValueError, "at least 5"):
            preflight_wan_vae_route_state(
                flf_values["route"],
                binding=flf_values["binding"],
                model_type=WAN_I2V,
                image=flf_values["image"],
                last_image=flf_values["last_image"],
                height=64,
                width=64,
                num_frames=1,
                vae_component=vae,
            )

    def test_palette_source_mutation_and_exact_tensor_mutations_are_rejected(self):
        palette_image = Image.new("P", (64, 64))
        palette_image.putpalette([0, 0, 0] * 256)
        image_values = _issue_image_route(image=palette_image)
        palette_image.putpalette([255, 0, 0] * 256)
        with self.assertRaisesRegex(ValueError, "pixels changed"):
            validate_wan_image_encoder_route_state(
                image_values["route"],
                binding=image_values["binding"],
                model_type=WAN_I2V,
                image=palette_image,
                last_image=None,
                height=64,
                width=64,
                image_embeds=image_values["image_embeds"],
                image_encoder=image_values["image_encoder"],
                image_processor=image_values["image_processor"],
            )

        for field in ("image_embeds", "image_condition_latents"):
            with self.subTest(field=field):
                values = _issue_vae_route()
                values[field].add_(1)
                with self.assertRaisesRegex(ValueError, "mutated or rebound"):
                    _vae_validation(values)

    def test_component_and_processor_config_mutations_are_rejected(self):
        mutations = (
            ("image_processor", lambda values: setattr(values["image_processor"], "do_normalize", False)),
            ("image_encoder", lambda values: setattr(values["image_encoder"].config, "image_size", 336)),
            ("video_processor", lambda values: values["video_processor"].config.__setitem__("do_normalize", False)),
            ("vae_mean", lambda values: values["vae"].config.latents_mean.__setitem__(0, 0.5)),
            ("vae_std", lambda values: values["vae"].config.latents_std.__setitem__(0, 0.5)),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                values = _issue_vae_route()
                mutate(values)
                with self.assertRaises(ValueError):
                    _vae_validation(values)

    def test_transformer_contract_and_image_dimension_are_exact(self):
        for field, value in (
            ("patch_size", (1, 2, 1)),
            ("in_channels", 16),
            ("out_channels", 15),
        ):
            transformer = _Transformer()
            setattr(transformer.config, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                wan_transformer_contract_from_component(transformer)

        values = _issue_vae_route()
        transformer = _Transformer()
        transformer.config.image_dim = 1024
        with self.assertRaises(ValueError):
            consume_wan_vae_route_state(
                values["route"],
                binding=values["binding"],
                model_type=WAN_I2V,
                seed=values["seed"],
                execution_device="cpu",
                image_embeds=values["image_embeds"],
                image_condition_latents=values["image_condition_latents"],
                height=values["height"],
                width=values["width"],
                num_frames=values["num_frames"],
                vae_component=values["vae"],
                transformer_component=transformer,
            )

    def test_wrong_execution_devices_are_rejected_at_producer_and_consumer(self):
        values = _issue_vae_route()
        transformer = _Transformer()
        with self.assertRaisesRegex(ValueError, "generator device"):
            consume_wan_vae_route_state(
                values["route"],
                binding=values["binding"],
                model_type=WAN_I2V,
                seed=values["seed"],
                execution_device="cuda",
                image_embeds=values["image_embeds"],
                image_condition_latents=values["image_condition_latents"],
                height=values["height"],
                width=values["width"],
                num_frames=values["num_frames"],
                vae_component=values["vae"],
                transformer_component=transformer,
            )
        with self.assertRaisesRegex(ValueError, "Denoise execution device"):
            issue_decode_route_state(
                values["route"],
                binding=values["binding"],
                latents=torch.zeros_like(values["raw_frame_latents"]),
                vae_component=values["vae"],
                transformer_component=transformer,
                execution_device="cuda",
            )

    def test_processor_configs_are_bounded_before_use(self):
        processor = _ImageProcessor()
        wan_image_processor_config_seal(processor)
        for field, value in (
            ("do_resize", "yes"),
            ("do_center_crop", True),
            ("rescale_factor", 1e20),
            ("image_std", (1.0, 0.0, 1.0)),
            ("pad_size", _Size(height=9000, width=1)),
        ):
            invalid = _ImageProcessor()
            setattr(invalid, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                wan_image_processor_config_seal(invalid)
        video_processor = _VideoProcessor()
        wan_video_processor_config_seal(video_processor)
        video_processor.config["vae_latent_channels"] = 16
        with self.assertRaises(ValueError):
            wan_video_processor_config_seal(video_processor)

    def test_image_embeddings_require_exact_pinned_token_geometry(self):
        values = _issue_image_route()
        for wrong_tokens in (1, 256, 258):
            with self.subTest(tokens=wrong_tokens), self.assertRaisesRegex(ValueError, "batch contract"):
                issue_wan_image_encoder_route_state(
                    binding=values["binding"],
                    image=values["image"],
                    last_image=None,
                    height=values["height"],
                    width=values["width"],
                    image_embeds=torch.zeros((1, wrong_tokens, 1280)),
                    image_encoder=values["image_encoder"],
                    image_processor=values["image_processor"],
                    resized_image=Image.new("L", (values["preflight"][4], values["preflight"][3])),
                    resized_last_image=None,
                    execution_device="cpu",
                    source_snapshot=snapshot_wan_source_media(values["image"], None),
                    preflight_geometry=values["preflight"],
                )

    def test_resized_last_image_outputs_are_mandatory_only_for_flf(self):
        last_image = Image.new("RGB", (64, 64))
        token, _outputs = _bound_outputs()
        image = Image.new("RGB", (64, 64))
        processor = _FlfImageProcessor()
        encoder = _ImageEncoder()
        preflight = preflight_wan_image_encoder_inputs(image=image, last_image=last_image, height=64, width=64)
        embeds = torch.zeros((2, 257, 1280))
        base = dict(
            binding=token,
            image=image,
            last_image=last_image,
            height=64,
            width=64,
            image_embeds=embeds,
            image_encoder=encoder,
            image_processor=processor,
            resized_image=Image.new("L", (64, 64)),
            execution_device="cpu",
            source_snapshot=snapshot_wan_source_media(image, last_image),
            preflight_geometry=preflight,
        )
        for bad_last in (None, object(), Image.new("L", (32, 64))):
            with self.subTest(bad_last=type(bad_last).__name__), self.assertRaises(ValueError):
                issue_wan_image_encoder_route_state(**base, resized_last_image=bad_last)

        values = _issue_image_route(image=image, last_image=last_image)
        vae = _WanVae()
        video_processor = _VideoProcessor()
        vae_preflight = preflight_wan_vae_route_state(
            values["route"],
            binding=values["binding"],
            model_type=WAN_I2V,
            image=image,
            last_image=last_image,
            height=64,
            width=64,
            num_frames=5,
            vae_component=vae,
        )
        second_height, second_width = vae_preflight[3:5]
        vae_base = dict(
            route_state=values["route"],
            binding=values["binding"],
            seed=7,
            generator=torch.Generator(device="cpu").manual_seed(7),
            image=image,
            last_image=last_image,
            height=64,
            width=64,
            num_frames=5,
            image_condition_latents=torch.zeros(
                (1, 20, 2, second_height // 8, second_width // 8)
            ),
            raw_frame_latents=torch.zeros((1, 16, 2, second_height // 8, second_width // 8)),
            vae_component=vae,
            video_processor=video_processor,
            resized_image=Image.new("L", (second_width, second_height)),
            execution_device="cpu",
            preflight_geometry=vae_preflight,
        )
        for bad_last in (None, object(), Image.new("L", (32, 64))):
            with self.subTest(vae_bad_last=type(bad_last).__name__), self.assertRaises(ValueError):
                issue_wan_vae_route_state(**vae_base, resized_last_image=bad_last)

    def test_flf_crop_resize_preflight_matches_pinned_torchvision_axis_order(self):
        from torchvision.transforms.functional import center_crop

        image = Image.new("RGB", (1200, 600))
        last_image = Image.new("RGB", (400, 300))
        preflight = preflight_wan_image_encoder_inputs(
            image=image,
            last_image=last_image,
            height=480,
            width=832,
        )
        self.assertEqual(preflight[3:5], (432, 880))
        self.assertEqual(preflight[7], (660, 880))
        # Pinned Wan passes [computed_width, computed_height] to torchvision,
        # whose API interprets the pair as [height, width].
        actual = center_crop(last_image, [880, 660])
        self.assertEqual(actual.size, preflight[7])
        values = _issue_image_route(
            image=image,
            last_image=last_image,
            height=480,
            width=832,
        )
        self.assertEqual(values["preflight"][7], actual.size)
        second = preflight_wan_vae_route_state(
            values["route"],
            binding=values["binding"],
            model_type=WAN_I2V,
            image=image,
            last_image=last_image,
            height=480,
            width=832,
            num_frames=5,
            vae_component=_WanVae(),
        )
        self.assertEqual(second[3:5], (432, 864))
        self.assertEqual(second[10], (648, 864))
        actual_second = center_crop(last_image, [864, 648])
        self.assertEqual(actual_second.size, second[10])


class WanSchemaTruthTests(unittest.TestCase):
    def test_public_i2v_and_internal_flf_share_the_exact_generic_route(self):
        truth = PINNED_MODULAR_WORKFLOW_TRUTH[WAN_I2V]
        self.assertEqual([name for name, _mode in truth.modes], ["image_to_video"])
        self.assertEqual([name for name, _flow in truth.state_flows], ["flf2v"])
        mode = truth.mode("image_to_video")
        flow = truth.state_flow("flf2v")
        self.assertEqual(mode.action_sequence, ("text_encoder", "image_encoder", "vae_encoder", "denoise", "decoder"))
        self.assertEqual(flow.action_sequence, mode.action_sequence)
        edges = {
            (edge.producer_action, edge.producer_output, edge.consumer_action, edge.consumer_input)
            for edge in mode.state_edges
        }
        self.assertEqual(
            edges,
            {
                ("text_encoder", "embeddings", "denoise", "embeddings"),
                ("image_encoder", "image_embeds", "denoise", "image_embeds"),
                ("image_encoder", ROUTE_STATE_OUTPUT, "vae_encoder", ROUTE_STATE_INPUT),
                ("vae_encoder", "image_condition_latents", "denoise", "image_condition_latents"),
                ("vae_encoder", ROUTE_STATE_OUTPUT, "denoise", ROUTE_STATE_INPUT),
                ("denoise", "latents", "decoder", "latents"),
                ("denoise", ROUTE_STATE_OUTPUT, "decoder", ROUTE_STATE_INPUT),
            },
        )
        self.assertEqual(flow.state_edges, mode.state_edges)
        self.assertEqual(len(mode.upstream_block_sequence), 12)
        self.assertEqual(len(flow.upstream_block_sequence), 14)

    def test_schema_exposes_only_typed_condition_and_opaque_route_outputs(self):
        actions = get_model_type_metadata(WAN_I2V)["node_params"]
        self.assertEqual(actions["image_encoder"]["params"]["image_embeds"]["type"], "image_embeds")
        self.assertEqual(
            actions["vae_encoder"]["params"]["image_condition_latents"]["type"],
            "video_condition_latents",
        )
        self.assertNotIn("first_last_frame_latents", actions["vae_encoder"]["output_names"])
        self.assertEqual(actions["denoise"]["params"]["height"]["max"], 8192)
        self.assertEqual(actions["denoise"]["params"]["width"]["max"], 8192)
        self.assertIn("vae", actions["denoise"]["model_input_names"])

    def test_image_embeddings_same_model_signal_resends_authoritative_route_fields(self):
        node = ImageEmbeddings("wan-image-resync")
        node._model_type = WAN_I2V
        node._pipeline_class = diffusers.WanImage2VideoModularPipeline
        node.send_node_definition = Mock()
        with patch.object(node, "get_signal_value", return_value=WAN_I2V):
            node.update_node({}, None)
        params = node.send_node_definition.call_args.args[0]
        self.assertIn("last_image", params)
        self.assertIn("height", params)
        self.assertIn("width", params)
        self.assertIn(ROUTE_STATE_OUTPUT, params)


class WanPreinitResourceTests(unittest.TestCase):
    def _image_blocks_and_config(self):
        blocks = Mock()
        blocks.input_names = ["image", "last_image", "height", "width"]
        config = {
            "params": {
                "image": {"type": "image"},
                "last_image": {"type": "image"},
                "height": {"type": "int", "min": 1, "max": 8192},
                "width": {"type": "int", "min": 1, "max": 8192},
                "image_encoder": {"type": "diffusers_auto_model"},
            },
            "model_input_names": ["image_encoder"],
            "input_names": ["image", "last_image", "height", "width", "image_encoder"],
            "output_names": ["image_embeds", ROUTE_STATE_OUTPUT],
        }
        return blocks, config

    def test_extreme_first_resize_and_flf_crop_fail_before_init(self):
        _token, outputs = _bound_outputs()
        cases = ((Image.new("RGB", (8192, 1)), None, 4096, 4096, "area-budget"),)
        for image, last_image, height, width, message in cases:
            with self.subTest(message=message):
                blocks, config = self._image_blocks_and_config()
                with (
                    patch(
                        "modules.ModularDiffusers.embeddings.pipeline_class_from_runtime_inputs",
                        return_value=diffusers.WanImage2VideoModularPipeline,
                    ),
                    patch(
                        "modules.ModularDiffusers.embeddings.require_modiff_node_contract",
                        return_value=(blocks, config),
                    ),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    ImageEmbeddings().execute(
                        image_encoder=outputs["image_encoder"],
                        image=image,
                        last_image=last_image,
                        height=height,
                        width=width,
                    )
                blocks.init_pipeline.assert_not_called()

        with self.assertRaisesRegex(ValueError, "last-image"):
            preflight_wan_image_encoder_inputs(
                image=Image.new("RGB", (64, 64)),
                last_image=Image.new("RGB", (1, 8192)),
                height=64,
                width=64,
            )

        direct_resize = preflight_wan_image_encoder_inputs(
            image=Image.new("RGB", (8192, 16)),
            last_image=None,
            height=128,
            width=1024,
        )
        self.assertEqual(direct_resize[5:7], (224, 224))

    def test_wrong_workflow_artifact_is_rejected_before_contract_resolution_and_cache_reuse(self):
        _token, outputs = _bound_outputs()
        image = Image.new("RGB", (64, 64))
        last_image = Image.new("RGB", (64, 64))
        structural = _issue_image_route(image=image, last_image=last_image)
        image_values = {
            "image_encoder": outputs["image_encoder"],
            "image": image,
            "last_image": last_image,
            "height": 64,
            "width": 64,
        }
        vae_values = {
            "vae": outputs["vae_out"],
            "image": image,
            "last_image": last_image,
            "height": 64,
            "width": 64,
            "num_frames": 5,
            "seed": 7,
            ROUTE_STATE_INPUT: structural["route"],
        }
        for node, values, module in (
            (ImageEmbeddings(), image_values, "modules.ModularDiffusers.embeddings"),
            (ImageEncode(), vae_values, "modules.ModularDiffusers.latents"),
        ):
            with self.subTest(node=type(node).__name__):
                contract_resolver = Mock(side_effect=AssertionError("contract resolver must not run"))
                component_resolver = Mock(side_effect=AssertionError("component resolver must not run"))
                with (
                    patch(
                        f"{module}.pipeline_class_from_runtime_inputs",
                        return_value=diffusers.WanImage2VideoModularPipeline,
                    ),
                    patch(f"{module}.require_modiff_node_contract", contract_resolver),
                    patch(f"{module}.resolve_managed_component_by_id", component_resolver),
                    self.assertRaisesRegex(ValueError, "reviewed immutable"),
                ):
                    node.execute(**values)
                contract_resolver.assert_not_called()
                component_resolver.assert_not_called()

                node._pipeline_class = diffusers.WanImage2VideoModularPipeline
                with (
                    patch(f"{module}.require_modiff_node_contract", contract_resolver),
                    patch(f"{module}.resolve_managed_component_by_id", component_resolver),
                    self.assertRaisesRegex(ValueError, "reviewed immutable"),
                ):
                    node._cache_params_equal(values, dict(values))
                contract_resolver.assert_not_called()
                component_resolver.assert_not_called()

    def test_oversized_resolved_video_fails_before_vae_pipeline_init(self):
        image_values = _issue_image_route(height=4096, width=4096)
        vae = _WanVae()
        blocks = Mock()
        blocks.input_names = ["image", "height", "width", "num_frames", "generator"]
        config = {
            "params": {
                "image": {"type": "image"},
                "last_image": {"type": "image"},
                "height": {"type": "int", "min": 1, "max": 8192},
                "width": {"type": "int", "min": 1, "max": 8192},
                "num_frames": {"type": "int", "min": 1, "max": 480},
                "seed": {"type": "int", "min": 0, "max": 4294967295},
                "vae": {"type": "diffusers_auto_model"},
                ROUTE_STATE_INPUT: {"type": "modular_route_state"},
            },
            "model_input_names": ["vae"],
            "input_names": [
                "image",
                "last_image",
                "height",
                "width",
                "num_frames",
                "seed",
                "vae",
                ROUTE_STATE_INPUT,
            ],
            "output_names": ["image_condition_latents", ROUTE_STATE_OUTPUT],
        }
        with (
            patch(
                "modules.ModularDiffusers.latents.pipeline_class_from_runtime_inputs",
                return_value=diffusers.WanImage2VideoModularPipeline,
            ),
            patch("modules.ModularDiffusers.latents.require_modiff_node_contract", return_value=(blocks, config)),
            patch("modules.ModularDiffusers.latents.resolve_managed_component_by_id", return_value=vae),
            self.assertRaisesRegex(ValueError, "512-MiB"),
        ):
            ImageEncode().execute(
                vae=image_values["outputs"]["vae_out"],
                image=image_values["image"],
                height=4096,
                width=4096,
                num_frames=5,
                seed=7,
                route_state_in=image_values["route"],
            )
        blocks.init_pipeline.assert_not_called()


class WanActionBoundaryTests(unittest.TestCase):
    def _run_image_embeddings(
        self,
        *,
        init_mutation=None,
        call_mutation=None,
        pipeline_call_mutation=None,
        use_cache=False,
        execution_device="cpu",
        encoder_component=None,
        init_observer=None,
        overrides=None,
        last_image=None,
    ):
        _token, outputs = _bound_outputs(
            repository=WAN_FLF_REPOSITORY if last_image is not None else WAN_I2V_REPOSITORY,
            revision=WAN_FLF_REVISION if last_image is not None else WAN_I2V_REVISION,
        )
        source = Image.new("RGB", (100, 200), "red")
        processor = _FlfImageProcessor() if last_image is not None else _ImageProcessor()
        manager = {"encoder": encoder_component or _ImageEncoder()}
        pipeline_calls = []

        class FakePipeline:
            blocks = SimpleNamespace(doc="image-embeddings")

            def __init__(self):
                self._execution_device = torch.device(execution_device)
                self.image_encoder = None
                self.image_processor = None

            def update_components(self, **values):
                for name, value in values.items():
                    setattr(self, name, value)

            def __call__(self, **kwargs):
                pipeline_calls.append(dict(kwargs))
                if call_mutation is not None:
                    call_mutation(manager, processor)
                if pipeline_call_mutation is not None:
                    pipeline_call_mutation(self, manager, processor)
                preflight = preflight_wan_image_encoder_inputs(
                    image=source,
                    last_image=last_image,
                    height=480,
                    width=832,
                )
                first_height, first_width = preflight[3:5]
                return {
                    "resized_image": Image.new("L", (first_width, first_height)),
                    "resized_last_image": (
                        Image.new("L", preflight[7]) if last_image is not None else None
                    ),
                    "image_embeds": torch.zeros((2 if last_image is not None else 1, 257, 1280)),
                }

        class FakeBlocks:
            component_names = ["image_encoder", "image_processor"]
            input_names = ["image", "last_image", "height", "width"]
            doc = "image-embeddings"

            @staticmethod
            def init_pipeline(*, components_manager):
                if init_observer is not None:
                    init_observer()
                if init_mutation is not None:
                    init_mutation(manager, processor)
                return FakePipeline()

        config = {
            "params": {
                "image": {"type": "image"},
                "last_image": {"type": "image"},
                "height": {"type": "int", "min": 1, "max": 8192},
                "width": {"type": "int", "min": 1, "max": 8192},
                "image_encoder": {"type": "diffusers_auto_model"},
                "image_embeds": {"type": "image_embeds"},
                ROUTE_STATE_OUTPUT: {"type": "modular_route_state"},
            },
            "model_input_names": ["image_encoder"],
            "input_names": ["image", "last_image", "height", "width", "image_encoder"],
            "output_names": ["image_embeds", ROUTE_STATE_OUTPUT, "doc"],
        }

        class FakeSpec:
            def load(self, *, local_files_only):
                self.local_files_only = local_files_only
                return processor

        def managed(*, ids, return_dict_with_names=True):
            return {"image_encoder": manager["encoder"], "image_processor": processor}

        def resolve(_components, _payload, *, label):
            self.assertIn("image encoder", label.lower())
            return manager["encoder"]

        node = ImageEmbeddings("wan-image-action")
        processor_spec = Mock(return_value=FakeSpec())
        kwargs = {
            "image_encoder": outputs["image_encoder"],
            "image": source,
            "last_image": last_image,
            "height": "480",
            "width": "832",
        }
        kwargs.update(overrides or {})
        with (
            patch(
                "modules.ModularDiffusers.embeddings.pipeline_class_from_runtime_inputs",
                return_value=diffusers.WanImage2VideoModularPipeline,
            ),
            patch(
                "modules.ModularDiffusers.embeddings.require_modiff_node_contract",
                return_value=(FakeBlocks(), config),
            ),
            patch("modules.ModularDiffusers.embeddings.resolve_managed_component_by_id", side_effect=resolve),
            patch("modules.ModularDiffusers.embeddings.collect_model_ids", return_value=["wan-image-encoder"]),
            patch("modules.ModularDiffusers.embeddings.ComponentSpec", processor_spec),
            patch("modules.ModularDiffusers.embeddings.components.add", return_value="wan-image-processor"),
            patch("modules.ModularDiffusers.embeddings.components.get_components_by_ids", side_effect=managed),
        ):
            result = node(**kwargs) if use_cache else node.execute(**kwargs)
            if use_cache:
                cached = node(**dict(kwargs))
                self.assertIs(result, cached)
        self.assertEqual(processor_spec.call_args.kwargs["type_hint"].__name__, "CLIPImageProcessor")
        return result, pipeline_calls

    def test_image_embeddings_manager_init_and_call_swaps_publish_no_route(self):
        for stage in ("init", "call"):
            replacement = _ImageEncoder()

            def swap(manager, _processor):
                manager["encoder"] = replacement

            with self.subTest(stage=stage), self.assertRaisesRegex(ValueError, "changed"):
                self._run_image_embeddings(
                    init_mutation=(swap if stage == "init" else None),
                    call_mutation=(swap if stage == "call" else None),
                )

    def test_image_embeddings_processor_and_encoder_config_mutation_publish_no_route(self):
        mutations = (
            lambda _manager, processor: setattr(processor, "do_normalize", False),
            lambda manager, _processor: setattr(manager["encoder"].config, "patch_size", 16),
            lambda manager, _processor: setattr(manager["encoder"].config, "projection_dim", 768),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self._run_image_embeddings(call_mutation=mutation)
        with self.assertRaisesRegex(ValueError, "components changed"):
            self._run_image_embeddings(
                pipeline_call_mutation=lambda pipeline, _manager, _processor: setattr(
                    pipeline,
                    "image_processor",
                    _ImageProcessor(),
                )
            )

    def test_image_embeddings_canonical_string_replay_hits_cache(self):
        result, calls = self._run_image_embeddings(use_cache=True)
        self.assertIsNotNone(result[ROUTE_STATE_OUTPUT])
        self.assertEqual(len(calls), 1)

    def test_exact_flf_artifact_executes_image_and_vae_actions(self):
        last_image = Image.new("RGB", (100, 200), "blue")
        image_result, image_calls = self._run_image_embeddings(last_image=last_image)
        self.assertIsNotNone(image_result[ROUTE_STATE_OUTPUT])
        self.assertIs(image_calls[0]["last_image"], last_image)

        values = _issue_image_route(
            image=Image.new("RGB", (100, 200), "red"),
            last_image=last_image,
            height=480,
            width=832,
        )
        vae_result, vae_calls, _preflight = self._run_image_encode(values=values)
        self.assertIsNotNone(vae_result[ROUTE_STATE_OUTPUT])
        self.assertIs(vae_calls[0]["last_image"], last_image)

    def _run_image_encode(
        self,
        *,
        values=None,
        init_mutation=None,
        call_mutation=None,
        pipeline_call_mutation=None,
        use_cache=False,
        execution_device="cpu",
        vae_component=None,
        init_observer=None,
        overrides=None,
    ):
        values = values or _issue_image_route(image=Image.new("RGB", (100, 200)), height=480, width=832)
        manager = {"vae": vae_component or _WanVae()}
        processor = _VideoProcessor()
        pipeline_calls = []
        preflight_holder = {}

        class FakePipeline:
            blocks = SimpleNamespace(doc="vae")

            def __init__(self):
                self._execution_device = torch.device(execution_device)
                self.vae = None
                self.video_processor = processor

            def update_components(self, **components):
                for name, component in components.items():
                    setattr(self, name, component)

            def __call__(self, **kwargs):
                pipeline_calls.append(dict(kwargs))
                preflight = preflight_wan_vae_route_state(
                    values["route"],
                    binding=values["binding"],
                    model_type=WAN_I2V,
                    image=values["image"],
                    last_image=values["last_image"],
                    height=values["height"],
                    width=values["width"],
                    num_frames=5,
                    vae_component=manager["vae"],
                )
                preflight_holder["value"] = preflight
                second_height, second_width = preflight[3:5]
                if call_mutation is not None:
                    call_mutation(manager, processor)
                if pipeline_call_mutation is not None:
                    pipeline_call_mutation(self, manager, processor)
                temporal_frames = 2
                result = {
                    "resized_image": Image.new("L", (second_width, second_height)),
                    "resized_last_image": (
                        Image.new("L", preflight[10])
                        if values["last_image"] is not None
                        else None
                    ),
                    "image_condition_latents": torch.zeros(
                        (1, 20, temporal_frames, second_height // 8, second_width // 8)
                    ),
                }
                result[
                    "first_last_frame_latents" if values["last_image"] is not None else "first_frame_latents"
                ] = torch.zeros((1, 16, temporal_frames, second_height // 8, second_width // 8))
                return result

        class FakeBlocks:
            component_names = ["vae", "video_processor"]
            input_names = ["image", "last_image", "height", "width", "num_frames", "generator"]
            doc = "vae"

            @staticmethod
            def init_pipeline(*, components_manager):
                if init_observer is not None:
                    init_observer()
                if init_mutation is not None:
                    init_mutation(manager, processor)
                return FakePipeline()

        config = {
            "params": {
                "image": {"type": "image"},
                "last_image": {"type": "image"},
                "height": {"type": "int", "min": 1, "max": 8192},
                "width": {"type": "int", "min": 1, "max": 8192},
                "num_frames": {"type": "int", "min": 1, "max": 480},
                "seed": {"type": "int", "min": 0, "max": 4294967295},
                "vae": {"type": "diffusers_auto_model"},
                "image_condition_latents": {"type": "video_condition_latents"},
                ROUTE_STATE_INPUT: {"type": "modular_route_state"},
                ROUTE_STATE_OUTPUT: {"type": "modular_route_state"},
            },
            "model_input_names": ["vae"],
            "input_names": [
                "image",
                "last_image",
                "height",
                "width",
                "num_frames",
                "seed",
                "vae",
                ROUTE_STATE_INPUT,
            ],
            "output_names": ["image_condition_latents", ROUTE_STATE_OUTPUT, "doc"],
        }

        def managed(*, ids, return_dict_with_names=True):
            return {"vae": manager["vae"]}

        node = ImageEncode("wan-vae-action")
        kwargs = {
            "vae": values["outputs"]["vae_out"],
            "image": values["image"],
            "last_image": values["last_image"],
            "height": str(values["height"]),
            "width": str(values["width"]),
            "num_frames": "5",
            "seed": "7",
            ROUTE_STATE_INPUT: values["route"],
        }
        kwargs.update(overrides or {})
        with (
            patch(
                "modules.ModularDiffusers.latents.pipeline_class_from_runtime_inputs",
                return_value=diffusers.WanImage2VideoModularPipeline,
            ),
            patch(
                "modules.ModularDiffusers.latents.require_modiff_node_contract",
                return_value=(FakeBlocks(), config),
            ),
            patch(
                "modules.ModularDiffusers.latents.resolve_managed_component_by_id",
                side_effect=lambda *_args, **_kwargs: manager["vae"],
            ),
            patch("modules.ModularDiffusers.latents.collect_model_ids", return_value=["wan-vae"]),
            patch("modules.ModularDiffusers.latents.components.get_components_by_ids", side_effect=managed),
            patch(
                "modules.ModularDiffusers.latents.modular_generator_from_seed",
                side_effect=lambda seed, _pipeline: torch.Generator(device="cpu").manual_seed(seed),
            ),
        ):
            result = node(**kwargs) if use_cache else node.execute(**kwargs)
            if use_cache:
                cached = node(**dict(kwargs))
                self.assertIs(result, cached)
        return result, pipeline_calls, preflight_holder

    def test_vae_routes_first_pass_dimensions_and_cache_replays_strings(self):
        result, calls, preflight = self._run_image_encode(use_cache=True)
        self.assertIsNotNone(result[ROUTE_STATE_OUTPUT])
        self.assertEqual(len(calls), 1)
        self.assertEqual((calls[0]["height"], calls[0]["width"]), (880, 432))
        self.assertEqual(preflight["value"][3:5], (864, 432))

    def test_vae_manager_processor_and_normalization_mutations_publish_no_route(self):
        mutations = (
            ("init-manager", lambda manager, _processor: manager.__setitem__("vae", _WanVae()), "init"),
            ("call-manager", lambda manager, _processor: manager.__setitem__("vae", _WanVae()), "call"),
            ("processor", lambda _manager, processor: processor.config.__setitem__("do_normalize", False), "call"),
            ("mean", lambda manager, _processor: manager["vae"].config.latents_mean.__setitem__(0, 0.0), "call"),
        )
        for label, mutation, stage in mutations:
            with self.subTest(label=label), self.assertRaises(ValueError):
                self._run_image_encode(
                    init_mutation=(mutation if stage == "init" else None),
                    call_mutation=(mutation if stage == "call" else None),
                )
        with self.assertRaisesRegex(ValueError, "video processor changed"):
            self._run_image_encode(
                pipeline_call_mutation=lambda pipeline, _manager, _processor: setattr(
                    pipeline,
                    "video_processor",
                    _VideoProcessor(),
                )
            )

    def test_producer_device_mismatch_publishes_no_image_or_vae_route(self):
        with self.assertRaisesRegex(ValueError, "producing Wan execution device"):
            self._run_image_embeddings(execution_device="cuda")
        with self.assertRaisesRegex(ValueError, "producing Wan execution device"):
            self._run_image_encode(execution_device="cuda")

    def test_wrong_pinned_component_geometry_fails_before_action_init(self):
        encoder_cases = (
            ("patch_size", 16),
            ("num_channels", 4),
            ("projection_dim", 768),
        )
        for field, value in encoder_cases:
            encoder = _ImageEncoder()
            setattr(encoder.config, field, value)
            init_observer = Mock()
            with self.subTest(component="image_encoder", field=field), self.assertRaises(ValueError):
                self._run_image_embeddings(
                    encoder_component=encoder,
                    init_observer=init_observer,
                )
            init_observer.assert_not_called()

        vae_cases = (
            ("in_channels", 4, "config"),
            ("out_channels", 4, "config"),
            ("patch_size", 2, "config"),
            ("scale_factor_spatial", 16, "config"),
            ("scale_factor_temporal", 8, "config"),
            ("temperal_downsample", (True, True, True), "component"),
            ("latents_mean", [0.0] * 16, "config"),
            ("latents_std", [0.0] * 16, "config"),
        )
        for field, value, target in vae_cases:
            vae = _WanVae()
            setattr(vae if target == "component" else vae.config, field, value)
            init_observer = Mock()
            with self.subTest(component="vae", field=field), self.assertRaises(ValueError):
                self._run_image_encode(
                    vae_component=vae,
                    init_observer=init_observer,
                )
            init_observer.assert_not_called()

        transformer_cases = (
            ("patch_size", (1, 2, 3)),
            ("out_channels", 15),
            ("image_dim", 1024),
            ("pos_embed_seq_len", 514),
        )
        for field, value in transformer_cases:
            transformer = _Transformer()
            setattr(transformer.config, field, value)
            init_observer = Mock()
            with self.subTest(component="transformer", field=field), self.assertRaises(ValueError):
                self._run_denoise(
                    transformer_component=transformer,
                    init_observer=init_observer,
                )
            init_observer.assert_not_called()

    def test_image_and_vae_route_scalars_reject_noncanonical_forms(self):
        for invalid in (True, "480.0", "0480"):
            with self.subTest(action="image_embeddings", invalid=invalid), self.assertRaisesRegex(
                ValueError,
                "height",
            ):
                self._run_image_embeddings(overrides={"height": invalid})
        for field, canonical in (("height", "480"), ("num_frames", "5"), ("seed", "7")):
            for invalid in (True, f"{canonical}.0", f"0{canonical}"):
                with self.subTest(action="vae", field=field, invalid=invalid), self.assertRaisesRegex(
                    ValueError,
                    field,
                ):
                    self._run_image_encode(overrides={field: invalid})

    def _run_denoise(
        self,
        *,
        call_mutation=None,
        init_mutation=None,
        use_cache=False,
        overrides=None,
        transformer_component=None,
        init_observer=None,
    ):
        values = _issue_vae_route(image=Image.new("RGB", (100, 200)), height=480, width=832)
        manager = {
            "vae": values["vae"],
            "transformer": transformer_component or _Transformer(),
        }
        scheduler = object()
        pipeline_calls = []

        class FakePipeline:
            _execution_device = torch.device("cpu")
            component_names = ["transformer", "scheduler", "guider"]
            blocks = SimpleNamespace(doc="denoise")

            def __init__(self):
                self.transformer = None
                self.scheduler = None
                self.guider = None

            def update_components(self, **components):
                for name, component in components.items():
                    setattr(self, name, component)

            def __call__(self, **kwargs):
                pipeline_calls.append(dict(kwargs))
                if call_mutation is not None:
                    call_mutation(manager, values)
                return {"latents": torch.zeros_like(values["raw_frame_latents"])}

        class FakeBlocks:
            component_names = ["transformer", "scheduler", "guider"]
            input_names = [
                "prompt_embeds",
                "height",
                "width",
                "num_frames",
                "image_embeds",
                "image_condition_latents",
                "generator",
                "num_inference_steps",
            ]

            @staticmethod
            def init_pipeline(*, components_manager):
                if init_observer is not None:
                    init_observer()
                if init_mutation is not None:
                    init_mutation(manager, values)
                return FakePipeline()

        config = {
            "params": {
                "embeddings": {"type": "embeddings"},
                "height": {"type": "int", "min": 1, "max": 8192},
                "width": {"type": "int", "min": 1, "max": 8192},
                "num_frames": {"type": "int", "min": 1, "max": 480},
                "seed": {"type": "int", "min": 0, "max": 4294967295},
                "num_inference_steps": {"type": "int", "min": 1, "max": 1000},
                "image_embeds": {"type": "image_embeds"},
                "image_condition_latents": {"type": "video_condition_latents"},
                "unet": {"type": "diffusers_auto_model"},
                "vae": {"type": "diffusers_auto_model"},
                "scheduler": {"type": "diffusers_scheduler"},
                ROUTE_STATE_INPUT: {"type": "modular_route_state"},
                ROUTE_STATE_OUTPUT: {"type": "modular_route_state"},
            },
            "model_input_names": ["unet", "vae", "guider", "scheduler"],
            "input_names": [
                "embeddings",
                "height",
                "width",
                "num_frames",
                "seed",
                "num_inference_steps",
                "image_embeds",
                "image_condition_latents",
                "unet",
                "vae",
                "scheduler",
                ROUTE_STATE_INPUT,
            ],
            "output_names": ["latents", ROUTE_STATE_OUTPUT, "doc"],
        }

        def managed(*, ids, return_dict_with_names=True):
            return {"transformer": manager["transformer"], "scheduler": scheduler}

        def resolve(_components, _payload, *, label):
            return manager["vae"] if "VAE" in label else manager["transformer"]

        node = Denoise("wan-denoise-action")
        node.progress = Mock()
        kwargs = {
            "unet": values["outputs"]["unet_out"],
            "vae": values["outputs"]["vae_out"],
            "scheduler": values["outputs"]["scheduler"],
            "embeddings": {"prompt_embeds": torch.zeros((1, 4, 8))},
            "height": "480",
            "width": "832",
            "num_frames": "5",
            "seed": "7",
            "num_inference_steps": "2",
            "image_embeds": values["image_embeds"],
            "image_condition_latents": values["image_condition_latents"],
            ROUTE_STATE_INPUT: values["route"],
        }
        kwargs.update(overrides or {})
        with (
            patch(
                "modules.ModularDiffusers.denoise.pipeline_class_from_runtime_inputs",
                return_value=diffusers.WanImage2VideoModularPipeline,
            ),
            patch(
                "modules.ModularDiffusers.denoise.require_modiff_node_contract",
                return_value=(FakeBlocks(), config),
            ),
            patch("modules.ModularDiffusers.denoise.resolve_managed_component_by_id", side_effect=resolve),
            patch("modules.ModularDiffusers.denoise.collect_model_ids", return_value=["wan-transformer", "wan-scheduler"]),
            patch("modules.ModularDiffusers.denoise.components.get_components_by_ids", side_effect=managed),
            patch("modules.ModularDiffusers.denoise.insert_preview_block"),
        ):
            result = node(**kwargs) if use_cache else node.execute(**kwargs)
            if use_cache:
                cached = node(**dict(kwargs))
                self.assertIs(result, cached)
        return result, pipeline_calls, values

    def test_denoise_uses_second_pass_dimensions_and_replays_canonical_strings(self):
        result, calls, values = self._run_denoise(use_cache=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual((calls[0]["height"], calls[0]["width"]), (864, 432))
        self.assertEqual(calls[0]["num_frames"], 5)
        self.assertIs(calls[0]["image_embeds"], values["image_embeds"])
        self.assertIs(calls[0]["image_condition_latents"], values["image_condition_latents"])
        self.assertIsNotNone(result[ROUTE_STATE_OUTPUT])

    def test_denoise_init_call_manager_config_and_tensor_mutations_publish_no_route(self):
        mutations = (
            ("init-vae", lambda manager, _values: manager.__setitem__("vae", _WanVae()), "init"),
            ("call-vae", lambda manager, _values: manager.__setitem__("vae", _WanVae()), "call"),
            ("transformer", lambda manager, _values: setattr(manager["transformer"].config, "out_channels", 15), "call"),
            ("condition", lambda _manager, values: values["image_condition_latents"].add_(1), "call"),
            ("embeds", lambda _manager, values: values["image_embeds"].add_(1), "call"),
        )
        for label, mutation, stage in mutations:
            with self.subTest(label=label), self.assertRaises(ValueError):
                self._run_denoise(
                    init_mutation=(mutation if stage == "init" else None),
                    call_mutation=(mutation if stage == "call" else None),
                )

    def _run_decode(
        self,
        *,
        init_mutation=None,
        call_mutation=None,
        use_cache=False,
        cache_processor_swap=False,
        precall_processor_swap=False,
        execution_device="cpu",
    ):
        values = _issue_decode_values()
        manager = {"vae": values["vae"]}
        processor = _VideoProcessor()
        pipeline_calls = []

        class FakePipeline:
            blocks = SimpleNamespace(doc="decode")

            def __init__(self):
                self._execution_device = torch.device(execution_device)
                self.vae = None
                self.video_processor = processor

            def update_components(self, **components):
                for name, component in components.items():
                    setattr(self, name, component)

            def __call__(self, **kwargs):
                pipeline_calls.append(dict(kwargs))
                if call_mutation is not None:
                    call_mutation(self, manager, processor)
                return {"videos": [[Image.new("RGB", (8, 8))]]}

        class FakeBlocks:
            component_names = ["vae", "video_processor"]
            input_names = ["latents", "output_type"]
            doc = "decode"

            @staticmethod
            def init_pipeline(*, components_manager):
                if init_mutation is not None:
                    init_mutation(manager, processor)
                return FakePipeline()

        config = {
            "params": {
                "latents": {"type": "latents"},
                "output_type": {"type": "dropdown", "default": "pil", "options": ["np", "pil"]},
                "vae": {"type": "diffusers_auto_model"},
                ROUTE_STATE_INPUT: {"type": "modular_route_state"},
                "videos": {"type": "video"},
            },
            "model_input_names": ["vae"],
            "input_names": ["latents", ROUTE_STATE_INPUT, "output_type", "vae"],
            "output_names": ["videos", "doc"],
        }

        def managed(*, ids, return_dict_with_names=True):
            return {"vae": manager["vae"]}

        node = DecodeLatents("wan-decode-action")
        kwargs = {
            "vae": values["outputs"]["vae_out"],
            "latents": values["latents"],
            "output_type": "pil",
            ROUTE_STATE_INPUT: values["route"],
        }
        cache_equal = None
        real_consume_decode = consume_decode_route_state
        consume_calls = 0

        def consume_with_optional_swap(*args, **kwargs):
            nonlocal consume_calls
            result = real_consume_decode(*args, **kwargs)
            consume_calls += 1
            if precall_processor_swap and consume_calls == 2:
                node._pipeline.video_processor = _VideoProcessor()
            return result

        with (
            patch(
                "modules.ModularDiffusers.latents.pipeline_class_from_runtime_inputs",
                return_value=diffusers.WanImage2VideoModularPipeline,
            ),
            patch(
                "modules.ModularDiffusers.latents.require_modiff_node_contract",
                return_value=(FakeBlocks(), config),
            ),
            patch(
                "modules.ModularDiffusers.latents.resolve_managed_component_by_id",
                side_effect=lambda *_args, **_kwargs: manager["vae"],
            ),
            patch("modules.ModularDiffusers.latents.collect_model_ids", return_value=["wan-vae"]),
            patch("modules.ModularDiffusers.latents.components.get_components_by_ids", side_effect=managed),
            patch(
                "modules.ModularDiffusers.latents.consume_decode_route_state",
                side_effect=consume_with_optional_swap,
            ),
        ):
            result = node(**kwargs) if use_cache else node.execute(**kwargs)
            if use_cache:
                if cache_processor_swap:
                    node._pipeline.video_processor = _VideoProcessor()
                    cache_equal = node._cache_params_equal(kwargs, dict(kwargs))
                else:
                    cached = node(**dict(kwargs))
                    self.assertIs(result, cached)
        return result, pipeline_calls, node, cache_equal

    def test_decode_cache_replay_binds_exact_output_processor_identity(self):
        result, calls, _node, _cache_equal = self._run_decode(use_cache=True)
        self.assertIsNotNone(result["videos"])
        self.assertEqual(len(calls), 1)

        _result, calls, _node, cache_equal = self._run_decode(
            use_cache=True,
            cache_processor_swap=True,
        )
        self.assertFalse(cache_equal)
        self.assertEqual(len(calls), 1)

    def test_decode_init_call_processor_vae_and_device_swaps_publish_no_output(self):
        mutations = (
            ("init-vae", lambda manager, _processor: manager.__setitem__("vae", _WanVae()), "init"),
            (
                "call-vae",
                lambda _pipeline, manager, _processor: manager.__setitem__("vae", _WanVae()),
                "call",
            ),
            (
                "processor-identity",
                lambda pipeline, _manager, _processor: setattr(pipeline, "video_processor", _VideoProcessor()),
                "call",
            ),
            (
                "processor-config",
                lambda _pipeline, _manager, processor: processor.config.__setitem__("do_normalize", False),
                "call",
            ),
            (
                "vae-config",
                lambda _pipeline, manager, _processor: manager["vae"].config.latents_std.__setitem__(0, 1.0),
                "call",
            ),
        )
        for label, mutation, stage in mutations:
            with self.subTest(label=label), self.assertRaises(ValueError):
                self._run_decode(
                    init_mutation=(mutation if stage == "init" else None),
                    call_mutation=(mutation if stage == "call" else None),
                )
        with self.assertRaisesRegex(ValueError, "changed before upstream execution"):
            self._run_decode(precall_processor_swap=True)
        with self.assertRaisesRegex(ValueError, "Decode execution device"):
            self._run_decode(execution_device="cuda")

    def test_route_scalars_reject_bool_fraction_and_leading_zero_before_denoise_call(self):
        for invalid in (True, "480.0", "0480"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "height"):
                self._run_denoise(overrides={"height": invalid})


if __name__ == "__main__":
    unittest.main()
