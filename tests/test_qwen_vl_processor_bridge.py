"""No-download CPU Qwen processor/encoder bridge and exact adapter boundaries."""

from copy import deepcopy
from types import SimpleNamespace
import inspect
import json

import pytest
import torch
from PIL import Image
from tokenizers.pre_tokenizers import ByteLevel
from transformers import Qwen2Tokenizer, Qwen2VLImageProcessor, Qwen2VLProcessor, Qwen2VLVideoProcessor
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_module
from diffusers import (
    ComponentsManager, QwenImageModularPipeline, QwenImageEditModularPipeline, QwenImageEditPlusModularPipeline,
    QwenImageLayeredModularPipeline,
)
from diffusers.modular_pipelines import PipelineState
from diffusers.modular_pipelines.qwenimage.encoders import (
    QwenImageEditTextEncoderStep as SDKEditStep,
    QwenImageEditPlusTextEncoderStep as SDKPlusStep,
    _extract_masked_hidden,
)
from diffusers.modular_pipelines.qwenimage.modular_blocks_qwenimage_edit import QwenImageEditAutoBlocks
from diffusers.modular_pipelines.qwenimage.modular_blocks_qwenimage_edit_plus import QwenImageEditPlusAutoBlocks
from modules.DiffusersRuntime.qwen_vl import qwen_vl_encoder_outputs
from modules.ModularDiffusers.native_blocks import prepare_native_pipeline_blocks
from modules.ModularDiffusers.qwen_vl import (
    QwenImageEditTextEncoderStep, QwenImageEditPlusTextEncoderStep, prepare_qwen_edit_vl_blocks,
)
from modules.DiffusersImage.qwen_vl import QWEN_EDIT_PIPELINE_CLASSES
from modules.DiffusersImage.main import pipeline_class_from_name, _image_pipeline_adapter, _tag_image_pipeline, get_image_pipeline_adapter


def tiny_fixture():
    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>",
               "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"]
    vocab = {token: i for i, token in enumerate(special)}
    for token in sorted(ByteLevel.alphabet()):
        vocab.setdefault(token, len(vocab))
    tokenizer = Qwen2Tokenizer(vocab=vocab, merges=[], additional_special_tokens=special[1:])
    image_processor = Qwen2VLImageProcessor(patch_size=14, temporal_patch_size=2, merge_size=2,
                                           min_pixels=56 * 56, max_pixels=56 * 56)
    processor = Qwen2VLProcessor(image_processor=image_processor, tokenizer=tokenizer,
                                 video_processor=Qwen2VLVideoProcessor())
    ids = {name: tokenizer.convert_tokens_to_ids(token) for name, token in {
        "image_token_id": "<|image_pad|>", "video_token_id": "<|video_pad|>",
        "vision_start_token_id": "<|vision_start|>", "vision_end_token_id": "<|vision_end|>"}.items()}
    config = Qwen2_5_VLConfig(
        text_config={"vocab_size": len(tokenizer), "hidden_size": 16, "intermediate_size": 32,
                     "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2,
                     "max_position_embeddings": 512, "use_cache": False,
                     "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
                     "rope_parameters": {"rope_type": "default", "mrope_section": [1, 1, 2]}},
        vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2,
                       "out_hidden_size": 16, "patch_size": 14, "temporal_patch_size": 2,
                       "spatial_merge_size": 2, "window_size": 56, "fullatt_block_indexes": [0]},
        **ids,
    )
    torch.manual_seed(314)
    encoder = Qwen2_5_VLForConditionalGeneration(config).eval()
    encoder.set_attn_implementation("eager")
    return processor, encoder

@pytest.fixture(scope="module")
def tiny():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        processor, encoder = tiny_fixture()
    yield processor, encoder
    torch.set_num_threads(previous)


def batch(processor, images, prompt="Edit this object."):
    text = "<|im_start|>user\n" + " ".join(
        "<|vision_start|><|image_pad|><|vision_end|>" for _ in images
    ) + prompt + "<|im_end|>"
    return processor(text=[text], images=images, padding=True, return_tensors="pt").to("cpu")


def original_fields(value):
    return {"input_ids": value["input_ids"], "attention_mask": value["attention_mask"],
            "pixel_values": value.get("pixel_values"), "image_grid_thw": value.get("image_grid_thw"),
            "output_hidden_states": True}


@pytest.mark.parametrize("count,prompt", [(1, "Describe the blue object."), (1, "Do not add rain."),
                                           (2, "Edit the first using the second.")])
def test_genuine_mm_positions_hidden_and_shared_accelerate_owner(tiny, count, prompt):
    processor, encoder = tiny
    images = [Image.new("RGB", (56, 56), color) for color in [(24, 60, 120), (170, 60, 10)]][:count]
    inputs = batch(processor, images, prompt)
    assert inputs["mm_token_type_ids"].ne(0).any()
    assert inputs["image_grid_thw"].tolist() == [[1, 4, 4]] * count
    manager = ComponentsManager()
    source_id = manager.add("text_encoder", encoder, collection="source-owner")
    consumer_id = manager.add("text_encoder", encoder, collection="consumer-owner")
    assert source_id == consumer_id
    source_owner = {"text_encoder": manager.get_one(component_id=source_id)}
    consumer_owner = {"text_encoder": manager.get_one(component_id=consumer_id)}
    manager_collections = deepcopy(manager.collections)
    config = deepcopy(encoder.config.to_dict())
    load_id = getattr(encoder, "_diffusers_load_id", None)
    align = AlignDevicesHook(execution_device="cpu", offload=False)
    add_hook_to_module(encoder, align)
    forward, old_forward = encoder.forward, encoder._old_forward
    positions, entered = [], []
    observer = encoder.model.language_model.rotary_emb.register_forward_pre_hook(
        lambda _module, args: positions.append(args[1].detach().clone()),
    )
    entries = encoder.register_forward_pre_hook(lambda _m, _a, kwargs: entered.append(kwargs), with_kwargs=True)
    try:
        arms = []
        for arm in ("omitted", "direct", "helper"):
            encoder.model.rope_deltas = None
            with torch.no_grad():
                if arm == "helper":
                    output = qwen_vl_encoder_outputs(consumer_owner["text_encoder"], inputs, image_conditioned=True)
                else:
                    fields = original_fields(inputs)
                    if arm == "direct":
                        fields["mm_token_type_ids"] = inputs["mm_token_type_ids"]
                    output = encoder(**fields)
            arms.append((positions[-1], output.hidden_states[-1].detach().clone()))
        assert torch.equal(arms[1][0], arms[2][0]) and torch.equal(arms[1][1], arms[2][1])
        assert not torch.equal(arms[0][0], arms[1][0]) and not torch.equal(arms[0][1], arms[1][1])
        assert entered[-1]["mm_token_type_ids"] is inputs["mm_token_type_ids"]
        assert all(entered[-1][key] is value for key, value in original_fields(inputs).items())
        assert source_owner["text_encoder"] is consumer_owner["text_encoder"] is encoder
        assert manager.get_one(component_id=source_id) is encoder and manager.collections == manager_collections
        assert len(manager.components) == 1
        assert encoder.forward is forward and encoder._old_forward is old_forward and encoder._hf_hook is align
        assert encoder.config.to_dict() == config and getattr(encoder, "_diffusers_load_id", None) == load_id
    finally:
        observer.remove()
        entries.remove()
        remove_hook_from_module(encoder)


class RecordingEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, image_grid_thw=None,
                output_hidden_states=None, mm_token_type_ids=None):
        self.calls.append({"input_ids": input_ids, "attention_mask": attention_mask, "pixel_values": pixel_values,
                           "image_grid_thw": image_grid_thw, "output_hidden_states": output_hidden_states,
                           "mm_token_type_ids": mm_token_type_ids})
        hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 3)
        return SimpleNamespace(hidden_states=[hidden])


class FakeBatch(dict):
    def to(self, device):
        self.device = device
        return self


class RecordingProcessor:
    def __init__(self, include_mm):
        self.include_mm = include_mm
        self.calls = []
        self.batches = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        rows = len(kwargs["text"])
        ids = torch.arange(100).repeat(rows, 1)
        mask = torch.ones_like(ids)
        if rows > 1:
            mask[-1, -10:] = 0
        value = FakeBatch(input_ids=ids, attention_mask=mask, pixel_values=torch.ones(4, 2),
                          image_grid_thw=torch.tensor([[1, 2, 2]]))
        if self.include_mm:
            value["mm_token_type_ids"] = torch.ones_like(ids)
        self.batches.append(value)
        return value


@pytest.mark.parametrize("leaf,sdk,image_key", [(QwenImageEditTextEncoderStep, SDKEditStep, "resized_image"),
                                               (QwenImageEditPlusTextEncoderStep, SDKPlusStep, "resized_cond_image")])
@pytest.mark.parametrize("negative,requires", [("", True), (["first", "second"], True), (None, False)])
def test_native_exact_prompts_images_padding_negative_scope(leaf, sdk, image_key, negative, requires):
    image = [Image.new("RGB", (32, 32)), Image.new("RGB", (32, 48))] if "Plus" in leaf.__name__ else Image.new("RGB", (32, 32))
    results = []
    for cls in (sdk, leaf):
        encoder, processor = RecordingEncoder(), RecordingProcessor(include_mm=True)
        values = PipelineState()
        for key, value in {"prompt": ["edit first", "edit second"], "negative_prompt": negative, image_key: image}.items():
            values.set(key, value)
        components = SimpleNamespace(text_encoder=encoder, processor=processor, requires_unconditional_embeds=requires,
                                     _execution_device="cpu")
        returned = cls()(components, values)
        assert returned == (components, values)
        results.append((processor, encoder, values))
    expected, actual = results
    assert actual[0].calls == expected[0].calls
    assert all(call["images"] is image for call in actual[0].calls)
    assert len(actual[1].calls) == (2 if requires else 1)
    for call, inputs in zip(actual[1].calls, actual[0].batches):
        assert call["mm_token_type_ids"] is inputs["mm_token_type_ids"]
    for name in ("prompt_embeds", "prompt_embeds_mask", "negative_prompt_embeds", "negative_prompt_embeds_mask"):
        first, second = expected[2].get(name), actual[2].get(name)
        assert first is second is None if first is None else torch.equal(first, second)


@pytest.mark.parametrize("name", list(QWEN_EDIT_PIPELINE_CLASSES))
@pytest.mark.parametrize("include_mm", [False, True])
def test_whole_exact_pinned_prompt_method_and_canonical_identity(name, include_mm):
    import diffusers

    sdk, cls = getattr(diffusers, name), pipeline_class_from_name(name)
    assert cls is QWEN_EDIT_PIPELINE_CLASSES[name] and cls is not sdk and issubclass(cls, sdk)
    assert cls.__name__ == sdk.__name__
    assert cls.__init__ is sdk.__init__ and cls.__call__ is sdk.__call__ and cls.encode_prompt is sdk.encode_prompt
    assert cls.load_lora_weights is sdk.load_lora_weights
    assert cls.enable_model_cpu_offload is sdk.enable_model_cpu_offload
    assert cls.model_cpu_offload_seq == sdk.model_cpu_offload_seq
    assert inspect.signature(cls.__init__) == inspect.signature(sdk.__init__)
    assert inspect.signature(cls.__call__) == inspect.signature(sdk.__call__)
    adapter = get_image_pipeline_adapter(name)
    owner = cls(scheduler=None, vae=None, text_encoder=None, tokenizer=None, processor=None, transformer=None)
    assert json.loads(owner.to_json_string())["_class_name"] == name
    _tag_image_pipeline(owner, adapter=adapter, mode=adapter.mode_options[0], repo="private-local", source="local", revision=None)
    assert _image_pipeline_adapter(owner) is adapter
    prompt = ["edit first", "edit second"]
    image = [Image.new("RGB", (32, 32)), Image.new("RGB", (32, 48))] if "Plus" in name else Image.new("RGB", (32, 32))
    outputs = []
    for method in (sdk._get_qwen_prompt_embeds, cls._get_qwen_prompt_embeds):
        processor, encoder = RecordingProcessor(include_mm), RecordingEncoder()
        encoder.dtype = torch.float32
        proxy = SimpleNamespace(text_encoder=encoder, processor=processor, _execution_device="cpu",
                                prompt_template_encode=owner.prompt_template_encode,
                                prompt_template_encode_start_idx=owner.prompt_template_encode_start_idx,
                                _extract_masked_hidden=owner._extract_masked_hidden,
                                _modiff_image_template=cls._modiff_image_template)
        outputs.append((method(proxy, prompt=prompt, image=image, dtype=torch.bfloat16), processor, encoder))
    assert outputs[0][1].calls == outputs[1][1].calls
    assert all(torch.equal(first, second) for first, second in zip(outputs[0][0], outputs[1][0]))
    assert outputs[1][0][0].dtype == torch.bfloat16
    if include_mm:
        assert outputs[1][2].calls[0]["mm_token_type_ids"] is outputs[1][1].batches[0]["mm_token_type_ids"]


@pytest.mark.parametrize("pipeline,root,leaf", [(QwenImageEditModularPipeline, QwenImageEditAutoBlocks, QwenImageEditTextEncoderStep),
                                             (QwenImageEditPlusModularPipeline, QwenImageEditPlusAutoBlocks, QwenImageEditPlusTextEncoderStep)])
def test_fresh_native_tree_only_encoder_schema_and_siblings_preserved(pipeline, root, leaf):
    first, sibling = pipeline().blocks, pipeline().blocks
    encoder = first.sub_blocks["text_encoder"]
    original = encoder.sub_blocks["encode"]
    resize = encoder.sub_blocks["resize"]
    schema = (original.inputs, original.expected_components, original.intermediate_outputs)
    other = {key: value for key, value in first.sub_blocks.items() if key != "text_encoder"}
    adapted_tree = prepare_qwen_edit_vl_blocks(pipeline, first)
    assert adapted_tree is not first
    encoder = adapted_tree.sub_blocks["text_encoder"]
    assert type(encoder.sub_blocks["encode"]) is leaf
    assert encoder.block_classes[1] is encoder.sub_blocks["encode"]
    assert type(encoder.sub_blocks["resize"]) is type(resize)
    assert all(first.sub_blocks[key] is value for key, value in other.items())
    assert first.sub_blocks["text_encoder"].sub_blocks["encode"] is original
    adapted = encoder.sub_blocks["encode"]
    assert (adapted.inputs, adapted.expected_components, adapted.intermediate_outputs) == schema
    assert type(sibling.sub_blocks["text_encoder"].sub_blocks["encode"]) is type(original)
    second = pipeline().blocks
    second = prepare_native_pipeline_blocks(pipeline, second)
    assert type(second.sub_blocks["text_encoder"].sub_blocks["encode"]) is leaf
    if pipeline is QwenImageEditModularPipeline:
        assert type(second.sub_blocks["vae_encoder"].sub_blocks["edit_inpaint"].sub_blocks["preprocess"]).__name__ == "QwenWholeInpaintPreprocessStep"


def test_blueprint_drift_fails_closed_and_exclusions_are_exact():
    import diffusers

    root = QwenImageEditModularPipeline().blocks
    root.sub_blocks["text_encoder"].sub_blocks["encode"] = SDKPlusStep()
    with pytest.raises(ValueError, match="blueprint"):
        prepare_qwen_edit_vl_blocks(QwenImageEditModularPipeline, root)
    for pipeline in (QwenImageModularPipeline, QwenImageLayeredModularPipeline):
        marker = SimpleNamespace()
        assert prepare_qwen_edit_vl_blocks(pipeline, marker) is marker
    for name in ("QwenImagePipeline", "QwenImageImg2ImgPipeline", "QwenImageInpaintPipeline", "QwenImageLayeredPipeline", "QwenImage21Pipeline"):
        assert pipeline_class_from_name(name) is getattr(diffusers, name)


@pytest.mark.parametrize("shape", ["old", "opaque", "positional", "uninspectable", "modern"])
@pytest.mark.parametrize("has_mm,image_conditioned", [(True, True), (False, True), (True, False)])
def test_old_missing_opaque_signature_compatibility_and_exact_field_identity(shape, has_mm, image_conditioned):
    class Old:
        def forward(self, input_ids=None, attention_mask=None, pixel_values=None, image_grid_thw=None, output_hidden_states=None):
            pass
        def __call__(self, **kwargs):
            return kwargs
    class Opaque(Old):
        def forward(self, **kwargs):
            pass
    class Positional(Old):
        def forward(self, mm_token_type_ids=None, /, **kwargs):
            pass
    class Modern(Old):
        def forward(self, *, mm_token_type_ids=None, **kwargs):
            pass
    encoder = {"old": Old, "opaque": Opaque, "positional": Positional, "uninspectable": Old, "modern": Modern}[shape]()
    if shape == "uninspectable":
        encoder.forward = object()
    inputs = {"input_ids": object(), "attention_mask": object(), "pixel_values": object(), "image_grid_thw": object()}
    if has_mm:
        inputs["mm_token_type_ids"] = object()
    result = qwen_vl_encoder_outputs(encoder, inputs, image_conditioned=image_conditioned)
    should_add = shape == "modern" and has_mm and image_conditioned
    assert ("mm_token_type_ids" in result) is should_add
    assert all(result[key] is value for key, value in original_fields(inputs).items())
    if should_add:
        assert result["mm_token_type_ids"] is inputs["mm_token_type_ids"]


def test_encoder_error_propagates_once_without_mutating_owner_or_retry():
    class Failed(RecordingEncoder):
        def forward(self, **kwargs):
            self.calls.append(kwargs)
            raise RuntimeError("real encoder failure")
    encoder = Failed()
    forward = encoder.forward
    inputs = {"input_ids": object(), "attention_mask": object(), "mm_token_type_ids": object()}
    with pytest.raises(RuntimeError, match="real encoder failure"):
        qwen_vl_encoder_outputs(encoder, inputs, image_conditioned=True)
    assert len(encoder.calls) == 1 and encoder.forward == forward


@pytest.mark.parametrize("name", list(QWEN_EDIT_PIPELINE_CLASSES))
def test_whole_saved_model_index_reload_keeps_canonical_factory_and_all_component_aliases(tmp_path, name):
    import diffusers

    sdk, cls = getattr(diffusers, name), pipeline_class_from_name(name)
    owner = cls(scheduler=None, vae=None, text_encoder=None, tokenizer=None, processor=None, transformer=None)
    owner.save_pretrained(tmp_path, safe_serialization=True)
    document = json.loads((tmp_path / "model_index.json").read_text())
    assert document["_class_name"] == name and "_module" not in document
    reloaded = cls.from_pretrained(tmp_path, local_files_only=True, scheduler=None, vae=None,
                                  text_encoder=None, tokenizer=None, processor=None, transformer=None)
    assert type(reloaded) is cls
    assert cls.from_pretrained.__func__ is sdk.from_pretrained.__func__
    assert cls.save_pretrained is sdk.save_pretrained
    assert set(reloaded.components) == set(owner.components)
    assert all(reloaded.components[key] is owner.components[key] for key in owner.components)


@pytest.mark.parametrize("route", ["native-edit", "native-plus", *QWEN_EDIT_PIPELINE_CLASSES])
def test_actual_factory_genuine_processor_field_reaches_encoder_positions_and_packed_hidden(tiny, route):
    processor, encoder = tiny
    images = [Image.new("RGB", (56, 56), color) for color in [(24, 60, 120), (170, 60, 10)]]
    prompt = "Add a small blue mark."
    positions, entered = [], []
    observer = encoder.model.language_model.rotary_emb.register_forward_pre_hook(
        lambda _module, args: positions.append(args[1].detach().clone()),
    )
    entry = encoder.register_forward_pre_hook(lambda _m, _a, kwargs: entered.append(kwargs), with_kwargs=True)
    try:
        encoder.model.rope_deltas = None
        if route.startswith("native"):
            pipeline = QwenImageEditPlusModularPipeline if route == "native-plus" else QwenImageEditModularPipeline
            leaf = prepare_native_pipeline_blocks(pipeline, pipeline().blocks).sub_blocks["text_encoder"].sub_blocks["encode"]
            image = images if route == "native-plus" else images[0]
            values = PipelineState()
            for key, value in {"prompt": prompt, "negative_prompt": None,
                               ("resized_cond_image" if route == "native-plus" else "resized_image"): image}.items():
                values.set(key, value)
            components = SimpleNamespace(text_encoder=encoder, processor=processor, requires_unconditional_embeds=False,
                                         _execution_device="cpu")
            leaf(components, values)
            actual = values.get("prompt_embeds"), values.get("prompt_embeds_mask")
            template, drop_idx = leaf.prompt_template_encode, leaf.prompt_template_encode_start_idx
            image_template = getattr(leaf, "img_template_encode", None)
        else:
            cls = pipeline_class_from_name(route)
            owner = cls(scheduler=None, vae=None, text_encoder=encoder, tokenizer=processor.tokenizer,
                        processor=processor, transformer=None)
            image = images if "Plus" in route else images[0]
            actual = owner._get_qwen_prompt_embeds(prompt, image=image, device="cpu", dtype=torch.float32)
            template, drop_idx = owner.prompt_template_encode, owner.prompt_template_encode_start_idx
            image_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>" if "Plus" in route else None
        actual_positions = positions[-1]
        assert isinstance(entered[-1].get("mm_token_type_ids"), torch.Tensor)
        assert entered[-1]["mm_token_type_ids"].ne(0).any()
        prefix = "" if image_template is None else "".join(image_template.format(i + 1) for i in range(len(images)))
        expected_inputs = processor(text=[template.format(prefix + prompt)], images=image,
                                    padding=True, return_tensors="pt").to("cpu")
        encoder.model.rope_deltas = None
        with torch.no_grad():
            expected = encoder(**original_fields(expected_inputs), mm_token_type_ids=expected_inputs["mm_token_type_ids"])
        expected_hidden = _extract_masked_hidden(expected.hidden_states[-1], expected_inputs["attention_mask"])[0][drop_idx:]
        assert torch.equal(actual_positions, positions[-1])
        assert torch.equal(actual[0][0], expected_hidden)
        assert torch.equal(actual[1], torch.ones(expected_hidden.shape[0], dtype=torch.long).unsqueeze(0))
    finally:
        observer.remove()
        entry.remove()


@pytest.mark.parametrize("name", list(QWEN_EDIT_PIPELINE_CLASSES))
def test_whole_owned_class_cannot_change_existing_runtime_tag_admission(name):
    cls = pipeline_class_from_name(name)
    owner = cls(scheduler=None, vae=None, text_encoder=None, tokenizer=None, processor=None, transformer=None)
    adapter = get_image_pipeline_adapter(name)
    _tag_image_pipeline(owner, adapter=adapter, mode=adapter.mode_options[0], repo="private-local", source="local", revision=None)
    other = next(value for value in QWEN_EDIT_PIPELINE_CLASSES if value != name)
    owner._modiff_image_pipeline_class = other
    owner._modiff_image_adapter = get_image_pipeline_adapter(other)
    with pytest.raises(ValueError, match="runtime class"):
        _image_pipeline_adapter(owner)
    owner._modiff_image_pipeline_class = name
    owner._modiff_image_adapter = adapter
    owner._modiff_image_source = "other"
    with pytest.raises(ValueError, match="canonical hub or local"):
        _image_pipeline_adapter(owner)
    owner._modiff_image_source = "hub"
    owner._modiff_image_repo = "Qwen/Qwen-Image-Edit-2511"
    owner._modiff_image_revision = "main"
    with pytest.raises(ValueError, match="immutable|revision|pinned"):
        _image_pipeline_adapter(owner)
