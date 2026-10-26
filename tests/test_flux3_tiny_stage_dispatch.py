"""Real tiny CPU Flux3 components through native NodeBase stage dispatch.

Components are explicitly unit-installed in the genuine ComponentsManager;
ModelsLoader acquisition, gated index/factory files, app admission, trained
quality, HTTP and hardware fit are not established. Only tokenizer text IDs are
a fixed fixture. Every model, scheduler, guider and executable block is official.
"""

from types import SimpleNamespace
from importlib.util import find_spec

import numpy as np
from PIL import Image
import pytest
import torch


class FixedTextProcessor:
    """Bounded deterministic text IDs; deliberately not an AutoProcessor proof."""

    def apply_chat_template(self, messages, **_kwargs):
        return messages[0]["content"]

    def tokenizer(self, text, **kwargs):
        length = kwargs["max_length"] if kwargs.get("padding") else min(3, kwargs["max_length"])
        ids = [1 + sum(map(ord, text)) % 8, 9, 10][:length]
        count = len(ids)
        return {"input_ids": torch.tensor([ids + [0] * (length - count)]),
                "attention_mask": torch.tensor([[1] * count + [0] * (length - count)])}


@pytest.fixture
def tiny_stage_owner(monkeypatch, tmp_path):
    if find_spec("diffusers.modular_pipelines.flux3") is None:
        pytest.skip("Flux3 tiny stages require the separately reviewed pinned Diffusers source.", allow_module_level=True)
    from diffusers import AutoencoderKLFlux2, ComponentsManager, Flux3Transformer2DModel
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    import modules.ModularDiffusers as modular
    from modules.ModularDiffusers import denoise, embeddings, latents
    from modiff.config import CONFIG
    from modules.ModularDiffusers.route_state import bind_loader_outputs, issue_pipeline_instance_token

    for field in ("work_dir", "data", "images", "videos", "audio", "models", "upscalers", "temp"):
        monkeypatch.setitem(CONFIG.paths, field, str(tmp_path / field))
    manager = ComponentsManager()
    for module in (modular, denoise, embeddings, latents):
        monkeypatch.setattr(module, "components", manager)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(71)
        encoder = Qwen3VLForConditionalGeneration(Qwen3VLConfig(
            text_config={"vocab_size": 16, "hidden_size": 8, "intermediate_size": 16,
                         "num_hidden_layers": 32, "num_attention_heads": 1,
                         "num_key_value_heads": 1, "head_dim": 8, "use_cache": False,
                         "rope_parameters": {"rope_type": "default", "mrope_section": [1, 1, 2]}},
            vision_config={"depth": 1, "hidden_size": 8, "intermediate_size": 16,
                           "num_heads": 1, "out_hidden_size": 8, "patch_size": 2,
                           "spatial_merge_size": 1, "temporal_patch_size": 1},
            image_token_id=12, video_token_id=13, vision_start_token_id=14, vision_end_token_id=15,
        )).to(dtype=torch.bfloat16).eval()
        vae = AutoencoderKLFlux2(
            block_out_channels=(4, 4, 4, 4),
            down_block_types=("DownEncoderBlock2D",) * 4,
            up_block_types=("UpDecoderBlock2D",) * 4,
            layers_per_block=1, latent_channels=2, norm_num_groups=1,
            mid_block_add_attention=False,
        ).float().eval()
        vae.bn.running_mean.fill_(0.1)
        vae.bn.running_var.fill_(0.8)
        processor = FixedTextProcessor()

        def install(pipeline_name):
            torch.manual_seed(73)
            transformer = Flux3Transformer2DModel(
                in_channels=8, cond_in_channels=8, num_layers=1, num_single_layers=1,
                attention_head_dim=16, num_attention_heads=1, joint_attention_dim=64,
                timestep_channels=8, axes_dims_rope=(4, 4, 4, 4),
                flow_map=pipeline_name == "Flux3DistilledModularPipeline",
            ).to(dtype=torch.bfloat16).eval()
            # Official from_pretrained keeps these numerically sensitive
            # modules in FP32. This constructor-only fixture must preserve that
            # policy too; it does not prove acquisition/from_pretrained.
            for name, module in transformer.named_modules():
                if name.rsplit(".", 1)[-1] in transformer._keep_in_fp32_modules:
                    module.float()
            ids = {name: manager.add(name, component, collection="unit-tiny-owner") for name, component in
                   {"text_encoder": encoder, "tokenizer": processor, "vae": vae, "transformer": transformer}.items()}
            common = {"model_type": pipeline_name, "repo_id": "owned-local-tiny-components",
                      "repo_source": "local", "revision": None}
            token = issue_pipeline_instance_token(model_type=pipeline_name, repo_id=common["repo_id"],
                                                  repo_source="local", revision=None)
            payloads = bind_loader_outputs({
                "text_encoders": {**common, "text_encoder": {"model_id": ids["text_encoder"]},
                                  "tokenizer": {"model_id": ids["tokenizer"]}},
                "unet_out": {**common, "model_id": ids["transformer"]},
                "vae_out": {**common, "model_id": ids["vae"]},
            }, token)
            return SimpleNamespace(
                manager=manager, encoder=encoder, processor=processor, vae=vae, transformer=transformer,
                text=payloads["text_encoders"], unet=payloads["unet_out"], vae_input=payloads["vae_out"], token=token,
            )

        yield install
    torch.set_num_threads(previous_threads)


def references():
    return [Image.new("RGB", (64, 48), (220, 40, 20)), Image.new("RGB", (32, 32), (10, 80, 210))]


def assert_pixels_equal(actual, expected):
    actual = [actual] if isinstance(actual, Image.Image) else actual
    expected = [expected] if isinstance(expected, Image.Image) else expected
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        assert isinstance(left, Image.Image) and left.size == (64, 64)
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_actual_reference_receipt_binds_prompt_and_denoise_before_cache(tiny_stage_owner):
    import copy
    from diffusers import FlowMatchEulerDiscreteScheduler
    from modules.ModularDiffusers.embeddings import EncodePrompt
    from modules.ModularDiffusers.denoise import Denoise
    from modules.ModularDiffusers.latents import ImageEncode
    from modules.ModularDiffusers.reference_conditioning import require_reference_binding

    owner = tiny_stage_owner("Flux3DistilledModularPipeline")
    images = [Image.new("RGB", (32 + index % 2 * 16, 32), (index * 20, 30, 80)) for index in range(10)]
    image_encoder = ImageEncode("bound-reference")
    material = image_encoder(vae=owner.vae_input, image=images)
    assert len(material["image_cond_latents"]) == 10
    binding = material["reference_binding"]
    identities = require_reference_binding(binding, component=owner.text, role="text_encoders", manager=owner.manager)
    assert len(identities) == 10 and len(set(identities)) == 10
    layout = {"schemaVersion": 2, "coordinateSpace": "normalized_xyxy", "regions": [{
        "id": "subject", "description": "Remove exact tenth source", "action": "remove", "targetBox": None,
        "source": {"referenceIndex": 9, "expectedReferenceId": identities[9], "box": [0, 0, 1, 1]},
    }]}
    encoder = EncodePrompt("bound-prompt")
    encode_values = {"text_encoders": owner.text, "prompt": "Remove <subject> only.",
                     "layout_conditioning": layout, "reference_binding": binding,
                     "max_sequence_length": 8, "text_encoder_out_layers": (4, 8, 12, 16, 20, 24, 28, 32)}
    prompt = encoder(**encode_values)
    consumed = encoder._execution_input_record["fields"]["prompt"]["value"]
    assert '"from":"ref_image_9"' in consumed and '"tgt_bbox":null' in consumed
    assert encoder.params["prompt"] == "Remove <subject> only."
    encoder(**encode_values)
    assert not encoder._has_changed
    scheduler = FlowMatchEulerDiscreteScheduler(shift=6.93)
    scheduler_id = owner.manager.add("scheduler", scheduler, collection="unit-tiny-owner")
    sampler = Denoise("bound-denoise")
    values = {"unet": owner.unet, "scheduler": {"model_type": "Flux3DistilledModularPipeline",
              "repo_id": "owned-local-tiny-components", "model_id": scheduler_id}, "embeddings": prompt["embeddings"],
              "reference_binding": binding, "image_cond_latents": material["image_cond_latents"],
              "seed": 91, "width": 64, "height": 64, "num_inference_steps": 2, "num_images_per_prompt": 1}
    output = sampler(**values)
    assert torch.isfinite(output["latents"]).all()
    sampler(**values)
    assert not sampler._has_changed
    cloned = copy.deepcopy(prompt["embeddings"])
    sampler(**{**values, "embeddings": cloned})
    assert torch.equal(output["latents"], sampler.output["latents"])
    for change in ({"reference_binding": None}, {"image_cond_latents": None},
                   {"image_cond_latents": [tensor.clone() for tensor in material["image_cond_latents"]]},
                   {"image_cond_latents": list(reversed(material["image_cond_latents"]))},
                   {"reference_binding": {"referenceCount": 10}}):
        with pytest.raises((RuntimeError, ValueError), match="binding|reference|Reference|latent"):
            sampler(**{**values, **change})
    images[9].putpixel((0, 0), (255, 255, 255))
    with pytest.raises((RuntimeError, ValueError), match="content changed"):
        encoder(**encode_values)
    fresh = image_encoder(vae=owner.vae_input, image=images)
    assert image_encoder._has_changed and fresh["reference_binding"] is not binding
    with pytest.raises((RuntimeError, ValueError), match="rebind"):
        encoder(**{**encode_values, "reference_binding": fresh["reference_binding"]})


@pytest.mark.parametrize("count", [1, 2, 8, 9, 10])
def test_real_reference_count_and_non_square_geometry_are_preserved(tiny_stage_owner, count):
    from modules.ModularDiffusers.latents import ImageEncode
    from modules.ModularDiffusers.reference_conditioning import require_reference_binding

    owner = tiny_stage_owner("Flux3DistilledModularPipeline")
    images = [Image.new("RGB", (48, 32), (index, 10, 20)) for index in range(count)]
    node = ImageEncode(f"reference-count-{count}")
    output = node(vae=owner.vae_input, image=images)
    assert [tuple(value.shape[-2:]) for value in output["image_cond_latents"]] == [(2, 3)] * count
    ids = require_reference_binding(output["reference_binding"], component=owner.text,
                                    role="text_encoders", manager=owner.manager)
    assert len(ids) == count
    with pytest.raises((RuntimeError, ValueError), match="1 and 10"):
        node(vae=owner.vae_input, image=images * 11)


def test_reference_binding_rejects_foreign_owner_and_mutated_latent(tiny_stage_owner):
    from modules.ModularDiffusers.latents import ImageEncode
    from modules.ModularDiffusers.reference_conditioning import require_reference_binding
    from modules.ModularDiffusers.route_state import bind_loader_outputs, issue_pipeline_instance_token

    owner = tiny_stage_owner("Flux3DistilledModularPipeline")
    images = references()
    output = ImageEncode("reference-seals")(vae=owner.vae_input, image=images)
    binding = output["reference_binding"]
    foreign = dict(owner.text)
    token = issue_pipeline_instance_token(model_type="Flux3DistilledModularPipeline", repo_id="owned-local-tiny-components",
                                          repo_source="local", revision=None)
    bind_loader_outputs({"text_encoders": foreign}, token)
    with pytest.raises(ValueError, match="different Models Loader"):
        require_reference_binding(binding, component=foreign, role="text_encoders", manager=owner.manager)
    output["image_cond_latents"][0].add_(1)
    with pytest.raises(ValueError, match="mutated"):
        require_reference_binding(binding, component=owner.text, role="text_encoders", manager=owner.manager)


def test_reference_summary_uses_normal_static_cache_route_and_actual_task_context(tiny_stage_owner):
    import copy
    import json
    import pickle
    from modiff.NodeBase import _module_map, node_message_context
    from modiff.server import WebServer
    from modules.ModularDiffusers.embeddings import EncodePrompt
    from modules.ModularDiffusers.latents import ImageEncode

    owner = tiny_stage_owner("Flux3DistilledModularPipeline")
    images = references()
    material = ImageEncode("cache-reference")(vae=owner.vae_input, image=images)
    encoder = EncodePrompt("cache-prompt")
    with node_message_context({"task_id": "unit_reference_task", "attempt_index": 0, "run_input_hash": "run_abcd1234"}):
        encoded = encoder(text_encoders=owner.text, prompt="Unchanged raw prompt", reference_binding=material["reference_binding"],
                          max_sequence_length=8, text_encoder_out_layers=(4, 8, 12, 16, 20, 24, 28, 32))
    instance = object.__new__(WebServer)
    instance.node_cache = {"cache-prompt": encoder}
    instance.modules = _module_map()
    request = SimpleNamespace(match_info={"node": "cache-prompt", "field": "layout_reference_summary"}, query={}, headers={})
    # The string response branch does not await. Drive that actual coroutine
    # without creating Windows' asyncio self-pipe beneath the no-network fence.
    coroutine = instance.cache(request)
    with pytest.raises(StopIteration) as completed:
        coroutine.send(None)
    response = completed.value.value
    assert response.status == 200 and response.content_type == "text/plain"
    assert len(response.body) < 16384
    summary = json.loads(response.body)
    assert summary["taskId"] == "unit_reference_task" and summary["attemptIndex"] == 0
    assert summary["runInputHash"] == "run_abcd1234"
    assert [entry["sourceSize"] for entry in summary["references"]] == [[64, 48], [32, 32]]
    assert [entry["processedSize"] for entry in summary["references"]] == [[64, 48], [32, 32]]
    assert summary["bindingFingerprint"].startswith("sha256:") and summary["ownerFingerprint"].startswith("sha256:")
    assert copy.deepcopy(material["reference_binding"]) is material["reference_binding"]
    with pytest.raises(TypeError, match="process-local"):
        pickle.dumps(material["reference_binding"])
    assert encoder._execution_input_record["fields"]["prompt"]["value"] == "Unchanged raw prompt"
    assert encoded["layout_reference_summary"].encode() == response.body
    # An unchanged later task reuses the actual cached encoding. Its summary
    # remains attributed to the original producer rather than inventing a new
    # encoding/task identity; clients join that producer's persisted snapshot.
    with node_message_context({"task_id": "later_cached_task", "attempt_index": 1, "run_input_hash": "run_eeee1234"}):
        cached = encoder(text_encoders=owner.text, prompt="Unchanged raw prompt", reference_binding=material["reference_binding"],
                         max_sequence_length=8, text_encoder_out_layers=(4, 8, 12, 16, 20, 24, 28, 32))
    assert not encoder._has_changed
    assert cached["layout_reference_summary"] == encoded["layout_reference_summary"]
    coroutine = instance.cache(request)
    with pytest.raises(StopIteration) as cached_completed:
        coroutine.send(None)
    assert cached_completed.value.value.body == response.body


def test_image_reference_summary_recovers_changed_sources_before_stale_prompt_can_encode(tiny_stage_owner):
    import copy
    import json
    from modiff.NodeBase import _module_map, node_message_context
    from modiff.server import WebServer
    from modules.ModularDiffusers.embeddings import EncodePrompt
    from modules.ModularDiffusers.latents import ImageEncode
    from modules.ModularDiffusers.reference_conditioning import require_reference_binding

    owner = tiny_stage_owner("Flux3DistilledModularPipeline")
    images = references()
    source = ImageEncode("recovery-reference")
    encoder = EncodePrompt("recovery-prompt")
    with node_message_context({"task_id": "original_source_task", "attempt_index": 0, "run_input_hash": "run_1234abcd"}):
        old = source(vae=owner.vae_input, image=images)
    old_ids = require_reference_binding(old["reference_binding"], component=owner.text,
                                        role="text_encoders", manager=owner.manager)
    layout = {"schemaVersion": 2, "coordinateSpace": "normalized_xyxy", "regions": [{
        "id": "subject", "description": "Remove the selected source only", "action": "remove", "targetBox": None,
        "source": {"referenceIndex": 1, "expectedReferenceId": old_ids[1], "box": [0, 0, 1, 1]},
    }]}
    values = {"text_encoders": owner.text, "prompt": "Remove <subject>", "layout_conditioning": layout,
              "max_sequence_length": 8, "text_encoder_out_layers": (4, 8, 12, 16, 20, 24, 28, 32)}
    encoder(**values, reference_binding=old["reference_binding"])
    images[1].putpixel((0, 0), (255, 0, 0))
    with node_message_context({"task_id": "reencoded_source_task", "attempt_index": 1, "run_input_hash": "run_abcd1234"}):
        current = source(vae=owner.vae_input, image=images)
    assert source._has_changed
    with pytest.raises(RuntimeError, match="rebind"):
        encoder(**values, reference_binding=current["reference_binding"])

    # This is the ordinary cache route on the successful supplier, independent
    # of the failed Prompt. It exposes a summary, never the opaque authority.
    instance = object.__new__(WebServer)
    instance.node_cache = {source.node_id: source}
    instance.modules = _module_map()
    request = SimpleNamespace(match_info={"node": source.node_id, "field": "layout_reference_summary"}, query={}, headers={})
    coroutine = instance.cache(request)
    with pytest.raises(StopIteration) as completed:
        coroutine.send(None)
    response = completed.value.value
    assert response.status == 200 and response.content_type == "text/plain"
    summary = json.loads(response.body)
    assert summary["taskId"] == "reencoded_source_task" and summary["attemptIndex"] == 1
    assert summary["runInputHash"] == "run_abcd1234"
    assert summary["references"][1]["referenceId"] != old_ids[1]
    assert len(response.body) < 16384
    with node_message_context({"task_id": "later_source_cache_task", "attempt_index": 2, "run_input_hash": "run_eeee1234"}):
        reused = source(vae=owner.vae_input, image=images)
    assert not source._has_changed
    assert reused["layout_reference_summary"].encode() == response.body
    assert json.loads(reused["layout_reference_summary"])["taskId"] == "reencoded_source_task"
    assert layout["regions"][0]["source"]["expectedReferenceId"] == old_ids[1]
    rebound = copy.deepcopy(layout)
    rebound["regions"][0]["source"]["expectedReferenceId"] = summary["references"][1]["referenceId"]
    encoded = encoder(**{**values, "layout_conditioning": rebound}, reference_binding=current["reference_binding"])
    assert encoder._has_changed and "embeddings" in encoded
    assert values["prompt"] == "Remove <subject>"
    caption, projected = encoder._execution_input_record["fields"]["prompt"]["value"].split("\n", 1)
    assert caption == values["prompt"]
    assert json.loads(projected) == [{"id": "subject", "from": "ref_image_1", "src_bbox": [0, 0, 1000, 1000],
                                      "tgt_bbox": None, "desc": "Remove the selected source only"}]


@pytest.mark.parametrize("pipeline_name,reference_count,count", [
    ("Flux3DistilledModularPipeline", 0, 1),
    ("Flux3DistilledModularPipeline", 2, 1),
    ("Flux3DistilledModularPipeline", 2, 2),
    ("Flux3ModularPipeline", 0, 1),
    ("Flux3ModularPipeline", 2, 1),
])
def test_actual_native_stages_match_official_sequence_and_recompute_rng(
    tiny_stage_owner, pipeline_name, reference_count, count,
):
    import diffusers
    from diffusers import FlowMatchEulerDiscreteScheduler
    from modiff.NodeBase import NodeBase
    from modules.ModularDiffusers.embeddings import EncodePrompt
    from modules.ModularDiffusers.denoise import Denoise
    from modules.ModularDiffusers.latents import DecodeLatents, ImageEncode

    owner = tiny_stage_owner(pipeline_name)
    scheduler = FlowMatchEulerDiscreteScheduler(shift=6.93)
    scheduler_id = owner.manager.add("scheduler", scheduler, collection="unit-tiny-owner")
    scheduler_input = {"model_type": pipeline_name, "repo_id": "owned-local-tiny-components",
                       "model_id": scheduler_id}
    encode, image_encode = EncodePrompt("tiny-encode"), ImageEncode("tiny-reference")
    sampler, decode = Denoise("tiny-denoise"), DecodeLatents("tiny-decode")
    assert all(type(node).__call__ is NodeBase.__call__ for node in (encode, image_encode, sampler, decode))
    progress = []
    sampler.progress = lambda percentage, **fields: progress.append((percentage, fields))
    image_inputs = references()[:reference_count]
    vae_calls = []
    hook = owner.vae.decoder.register_forward_pre_hook(lambda _module, args: vae_calls.append(args[0].dtype))
    before_bn = (owner.vae.bn.running_mean.clone(), owner.vae.bn.running_var.clone())
    encode_values = {"text_encoders": owner.text, "prompt": "tiny red subject", "max_sequence_length": 8,
                     "text_encoder_out_layers": (4, 8, 12, 16, 20, 24, 28, 32), "layout_conditioning": None}
    if pipeline_name == "Flux3ModularPipeline":
        encode_values["negative_prompt"] = ""
    try:
        encoded = encode(**encode_values)
        assert encode._has_changed and encode._cache_valid
        assert encoded["embeddings"]["prompt_embeds"].shape == (1, 8, 64)
        if pipeline_name == "Flux3ModularPipeline":
            assert encoded["embeddings"]["negative_prompt_embeds"] is not None
        else:
            assert "negative_prompt_embeds" not in encoded["embeddings"]
        image_values = None
        if image_inputs:
            image_values = image_encode(vae=owner.vae_input, image=image_inputs)
            assert image_encode._has_changed and image_encode._cache_valid
            assert isinstance(image_values["image_cond_latents"], list)
            assert len(image_values["image_cond_latents"]) == reference_count
            assert all(tensor.dtype == torch.float32 for tensor in image_values["image_cond_latents"])
        sampler_values = {"unet": owner.unet, "scheduler": scheduler_input,
                          "embeddings": encoded["embeddings"], "seed": 91,
                          "width": 64, "height": 64, "num_inference_steps": 2,
                          "num_images_per_prompt": count}
        if image_values:
            sampler_values["image_cond_latents"] = image_values["image_cond_latents"]
        denoised = sampler(**sampler_values)
        assert sampler._has_changed and sampler._cache_valid
        staged_latents = denoised["latents"].clone()
        assert staged_latents.shape == (count, 8, 4, 4)
        assert staged_latents.dtype == torch.bfloat16 and torch.isfinite(staged_latents).all()
        decoded = decode(vae=owner.vae_input, latents=denoised["latents"])
        assert decode._has_changed and decode._cache_valid
        decoded_images = decoded["images"]
        decoded_images = [decoded_images] if isinstance(decoded_images, Image.Image) else decoded_images
        assert len(decoded_images) == count
        assert owner.vae.dtype == torch.float32
        assert vae_calls == [torch.float32]
        assert [fields["current_step"] for _percent, fields in progress if fields.get("phase") == "denoising"] == [0, 1, 2]

        # Ordinary NodeBase cache reuse is distinct from fresh inference.
        encode(**encode_values)
        assert not encode._has_changed
        sampler(**sampler_values)
        assert not sampler._has_changed
        sampler.invalidate_cache()
        regenerated = sampler(**sampler_values)
        assert sampler._has_changed
        torch.testing.assert_close(regenerated["latents"], staged_latents, rtol=0, atol=0)
        changed_seed = sampler(**{**sampler_values, "seed": 92})
        assert sampler._has_changed and not torch.equal(changed_seed["latents"], staged_latents)

        workflow = "image_conditioned" if image_inputs else "text2image"
        whole = getattr(diffusers, pipeline_name)(workflow=workflow)
        whole.update_components(text_encoder=owner.encoder, tokenizer=owner.processor,
                                transformer=owner.transformer, vae=owner.vae,
                                scheduler=FlowMatchEulerDiscreteScheduler(shift=6.93))
        values = {"prompt": encode_values["prompt"], "max_sequence_length": 8,
                  "text_encoder_out_layers": encode_values["text_encoder_out_layers"],
                  "height": 64, "width": 64, "num_inference_steps": 2,
                  "num_images_per_prompt": count, "generator": torch.Generator(device="cpu").manual_seed(91)}
        if pipeline_name == "Flux3ModularPipeline":
            values["negative_prompt"] = ""
        if image_inputs:
            values["image"] = image_inputs
        direct = whole(**values, output=["latents", "images"])
        torch.testing.assert_close(direct["latents"], staged_latents, rtol=0, atol=0)
        assert_pixels_equal(decoded["images"], direct["images"])
        assert torch.equal(owner.vae.bn.running_mean, before_bn[0])
        assert torch.equal(owner.vae.bn.running_var, before_bn[1])
        assert encode._pipeline.text_encoder is owner.encoder
        assert sampler._pipeline.transformer is owner.transformer
        assert decode._pipeline.vae is owner.vae
        if pipeline_name == "Flux3ModularPipeline":
            assert sampler._pipeline.guider.config.guidance_scale == 3.5
        else:
            assert "guider" not in sampler._pipeline.component_names
    finally:
        hook.remove()
