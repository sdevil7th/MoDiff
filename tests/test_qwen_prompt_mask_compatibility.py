"""Whole/native Qwen T2I masks using pinned SDK encoding as the CPU oracle."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from diffusers import (
    ClassifierFreeGuidance,
    ComponentsManager,
    FlowMatchEulerDiscreteScheduler,
    QwenImageEditModularPipeline,
    QwenImageEditPlusModularPipeline,
    QwenImageLayeredModularPipeline,
    QwenImageModularPipeline,
    QwenImagePipeline,
)
from diffusers.modular_pipelines import PipelineState
from diffusers.modular_pipelines.qwenimage.before_denoise import QwenImageRoPEInputsStep
from diffusers.modular_pipelines.qwenimage.denoise import QwenImageDenoiseStep, QwenImageLoopDenoiser
from diffusers.modular_pipelines.qwenimage.encoders import QwenImageTextEncoderStep, get_qwen_prompt_embeds
from diffusers.modular_pipelines.qwenimage.inputs import QwenImageTextInputsStep
from diffusers.modular_pipelines.qwenimage.prompt_templates import QWENIMAGE_PROMPT_TEMPLATE_START_IDX
from transformers import BatchEncoding

from modules.ModularDiffusers import native_blocks


INPUT_PATHS = (
    ("denoise", "text2image", "input"),
    ("denoise", "inpaint", "input", "text_inputs"),
    ("denoise", "img2img", "input", "text_inputs"),
    ("denoise", "controlnet_text2image", "input"),
    ("denoise", "controlnet_inpaint", "input", "text_inputs"),
    ("denoise", "controlnet_img2img", "input", "text_inputs"),
)
ROPE_PATH = ("denoise", "text2image", "prepare_rope_inputs")


def block_at(blocks, path):
    for name in path:
        blocks = blocks.sub_blocks[name]
    return blocks


class TinyTokenizer:
    """Produce real padded CPU tokens, without a pretrained tokenizer/model."""

    lengths = {"pos6": 6, "pos3": 3, "neg5": 5, "neg2": 2}

    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append((list(texts), dict(kwargs)))
        tags = [next((tag for tag in self.lengths if tag in text), None) for text in texts]
        lengths = [QWENIMAGE_PROMPT_TEMPLATE_START_IDX + self.lengths.get(tag, 1) for tag in tags]
        maximum = min(max(lengths), kwargs["max_length"])
        ids = torch.zeros((len(texts), maximum), dtype=torch.int64)
        masks = torch.zeros_like(ids)
        for index, (tag, length) in enumerate(zip(tags, lengths)):
            length = min(length, maximum)
            base = list(self.lengths).index(tag) * 100 if tag else 400
            ids[index, :length] = base + torch.arange(length)
            masks[index, :length] = 1
        return BatchEncoding({"input_ids": ids, "attention_mask": masks})


class TinyEncoder:
    def __init__(self):
        self.calls = []

    def __call__(self, *, input_ids, attention_mask, output_hidden_states):
        assert input_ids.device.type == attention_mask.device.type == "cpu"
        assert output_hidden_states is True
        self.calls.append((input_ids.clone(), attention_mask.clone()))
        hidden = input_ids.unsqueeze(-1).to(torch.bfloat16) + torch.arange(4, dtype=torch.bfloat16)
        return SimpleNamespace(hidden_states=(hidden,))


class CapturingTransformer(torch.nn.Module):
    """Record the real loop's model inputs without any pretrained weights."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(
        self, hidden_states, timestep, encoder_hidden_states, encoder_hidden_states_mask,
        img_shapes, attention_kwargs, return_dict,
    ):
        assert hidden_states.device.type == encoder_hidden_states.device.type == "cpu"
        assert hidden_states.dtype == encoder_hidden_states.dtype == torch.bfloat16
        assert return_dict is False
        self.calls.append({
            "embeddings": encoder_hidden_states,
            "mask": encoder_hidden_states_mask,
            "img_shapes": img_shapes,
            "timestep": timestep.clone(),
            "attention_kwargs": attention_kwargs,
        })
        return (torch.ones_like(hidden_states),)


def components(enabled):
    guider = ClassifierFreeGuidance(guidance_scale=4.0, enabled=enabled)
    return SimpleNamespace(
        text_encoder=TinyEncoder(), tokenizer=TinyTokenizer(), guider=guider,
        requires_unconditional_embeds=guider.num_conditions > 1, _execution_device=torch.device("cpu"),
    )


def whole_encode(owner, prompt, maximum, copies):
    # The actual pinned whole API performs truncation, batch expansion and its
    # all-valid-mask normalization. Only the tiny tokenizer/model are fixtures.
    reference = SimpleNamespace(
        _execution_device=torch.device("cpu"),
        _get_qwen_prompt_embeds=lambda value, device: get_qwen_prompt_embeds(
            owner.text_encoder, owner.tokenizer, prompt=value, device=device,
        ),
    )
    return QwenImagePipeline.encode_prompt(
        reference, prompt=prompt, device=torch.device("cpu"),
        num_images_per_prompt=copies, max_sequence_length=maximum,
    )


def run_native(prompt, negative, *, enabled=True, maximum=8, copies=1, path=INPUT_PATHS[0]):
    tree = QwenImageModularPipeline().blocks
    tree = native_blocks.prepare_native_pipeline_blocks(QwenImageModularPipeline, tree)
    encoder = block_at(tree, ("text_encoder", "text_encoder"))
    assert type(encoder) is QwenImageTextEncoderStep
    owner = components(enabled)
    state = PipelineState(values={"prompt": prompt, "negative_prompt": negative, "max_sequence_length": maximum})
    returned, state = encoder(owner, state)
    assert returned is owner
    encoded = {name: state.get(name) for name in (
        "prompt_embeds", "prompt_embeds_mask", "negative_prompt_embeds", "negative_prompt_embeds_mask",
    )}
    snapshots = {key: value.clone() for key, value in encoded.items() if isinstance(value, torch.Tensor)}
    state.set("num_images_per_prompt", copies)
    before_normalization = {}
    upstream_call = QwenImageTextInputsStep.__call__

    def capture_prepared(block, owner, state):
        returned, state = upstream_call(block, owner, state)
        before_normalization.update({name: state.get(name) for name in encoded})
        return returned, state

    with patch.object(QwenImageTextInputsStep, "__call__", capture_prepared):
        returned, state = block_at(tree, path)(owner, state)
    assert returned is owner
    if path == INPUT_PATHS[0]:
        # The actual text-input and RoPE stages both consume the tensor masks.
        # Normalization must happen only after these real upstream calls.
        owner.vae_scale_factor = 8
        state.set("height", 64)
        state.set("width", 80)
        returned, state = block_at(tree, ROPE_PATH)(owner, state)
        assert returned is owner
    for name, snapshot in snapshots.items():
        torch.testing.assert_close(encoded[name], snapshot, rtol=0, atol=0)
    return owner, state, encoded, before_normalization, tree


def assert_same_condition(actual_state, expected, negative=False):
    prefix = "negative_" if negative else ""
    embeddings, mask = expected
    torch.testing.assert_close(actual_state.get(prefix + "prompt_embeds"), embeddings, rtol=0, atol=0)
    actual_mask = actual_state.get(prefix + "prompt_embeds_mask")
    if mask is None:
        assert actual_mask is None
    else:
        assert actual_mask.dtype == mask.dtype and actual_mask.device == mask.device
        torch.testing.assert_close(actual_mask, mask, rtol=0, atol=0)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("maximum", [2, 4, 8])
@pytest.mark.parametrize("copies", [1, 3])
def test_actual_encoder_inputs_and_rope_match_whole_mask_truncation_and_batch_order(batch, maximum, copies):
    prompt = "pos6" if batch == 1 else ["pos6", "pos3"]
    negative = "neg5" if batch == 1 else ["neg5", "neg2"]
    owner, state, encoded, prepared, _ = run_native(prompt, negative, maximum=maximum, copies=copies)
    assert isinstance(encoded["prompt_embeds_mask"], torch.Tensor)
    assert isinstance(encoded["negative_prompt_embeds_mask"], torch.Tensor)
    assert_same_condition(state, whole_encode(owner, prompt, maximum, copies))
    assert_same_condition(state, whole_encode(owner, negative, maximum, copies), negative=True)
    assert state.get("batch_size") == batch
    assert state.get("dtype") == torch.bfloat16
    for name in ("prompt_embeds", "negative_prompt_embeds"):
        assert state.get(name) is prepared[name]
    for name in ("prompt_embeds_mask", "negative_prompt_embeds_mask"):
        if state.get(name) is not None:
            # A padded mask remains the exact parent-prepared object, retaining
            # its storage/view/dtype semantics instead of rebuilding it.
            assert state.get(name) is prepared[name]


@pytest.mark.parametrize("positive_padded,negative_padded", [(False, True), (True, False), (True, True)])
def test_positive_and_negative_masks_are_normalized_independently(positive_padded, negative_padded):
    prompt = ["pos6", "pos3"] if positive_padded else ["pos6", "pos6"]
    negative = ["neg5", "neg2"] if negative_padded else ["neg5", "neg5"]
    owner, state, _, prepared, _ = run_native(prompt, negative)
    assert_same_condition(state, whole_encode(owner, prompt, 8, 1))
    assert_same_condition(state, whole_encode(owner, negative, 8, 1), negative=True)
    assert (state.get("prompt_embeds_mask") is not None) is positive_padded
    assert (state.get("negative_prompt_embeds_mask") is not None) is negative_padded
    if positive_padded:
        assert state.get("prompt_embeds_mask") is prepared["prompt_embeds_mask"]
    if negative_padded:
        assert state.get("negative_prompt_embeds_mask") is prepared["negative_prompt_embeds_mask"]


def test_disabled_cfg_has_no_fabricated_negative_embeddings_or_mask():
    owner, state, encoded, _, _ = run_native("pos6", "neg5", enabled=False, copies=2)
    assert len(owner.text_encoder.calls) == 1
    assert encoded["negative_prompt_embeds"] is encoded["negative_prompt_embeds_mask"] is None
    assert state.get("negative_prompt_embeds") is state.get("negative_prompt_embeds_mask") is None
    assert_same_condition(state, whole_encode(owner, "pos6", 8, 2))


@pytest.mark.parametrize("padded", [False, True])
def test_actual_rope_prepares_geometry_before_normalizing_only_all_valid_masks(padded):
    prompt = ["pos6", "pos3"] if padded else "pos6"
    negative = ["neg5", "neg2"] if padded else "neg5"
    _, state, encoded, prepared, tree = run_native(prompt, negative)
    assert type(block_at(tree, INPUT_PATHS[0])) is QwenImageTextInputsStep
    assert type(block_at(tree, ROPE_PATH)) is getattr(native_blocks, "QwenImageWholePromptMaskRoPEInputsStep")
    assert state.get("img_shapes") == [[(1, 4, 5)]] * (2 if padded else 1)
    for prefix in ("", "negative_"):
        assert isinstance(encoded[prefix + "prompt_embeds_mask"], torch.Tensor)
        assert isinstance(prepared[prefix + "prompt_embeds_mask"], torch.Tensor)
        assert state.get(prefix + "prompt_embeds") is prepared[prefix + "prompt_embeds"]
        if padded:
            assert state.get(prefix + "prompt_embeds_mask") is prepared[prefix + "prompt_embeds_mask"]
        else:
            assert state.get(prefix + "prompt_embeds_mask") is None


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("padded", [False, True])
def test_real_denoise_loop_receives_whole_oracle_masks_after_encoder_inputs_and_rope(enabled, padded):
    prompt = ["pos6", "pos3"] if padded else "pos6"
    negative = ["neg5", "neg2"] if padded else "neg5"
    owner, state, encoded, prepared, tree = run_native(prompt, negative, enabled=enabled, copies=2)
    owner.transformer = CapturingTransformer()
    owner.scheduler = FlowMatchEulerDiscreteScheduler()
    owner.scheduler.set_timesteps(1, device="cpu")
    state.set("timesteps", owner.scheduler.timesteps)
    state.set("num_inference_steps", 1)
    state.set("latents", torch.ones((4 if padded else 2, 20, 4), dtype=torch.bfloat16))
    loop = block_at(tree, ("denoise", "text2image", "denoise"))
    assert type(loop) is QwenImageDenoiseStep
    assert type(loop.sub_blocks["denoiser"]) is QwenImageLoopDenoiser
    returned, state = loop(owner, state)
    assert returned is owner
    assert torch.isfinite(state.get("latents")).all()
    assert len(owner.transformer.calls) == (2 if enabled else 1)
    for call, prefix, condition in zip(
        owner.transformer.calls, ("", "negative_"), (prompt, negative),
    ):
        expected_embeddings, expected_mask = whole_encode(owner, condition, 8, 2)
        assert call["embeddings"] is prepared[prefix + "prompt_embeds"]
        torch.testing.assert_close(call["embeddings"], expected_embeddings, rtol=0, atol=0)
        assert call["img_shapes"] == [[(1, 4, 5)]] * (2 if padded else 1)
        assert call["attention_kwargs"] is None
        if expected_mask is None:
            assert call["mask"] is None
        else:
            assert call["mask"] is prepared[prefix + "prompt_embeds_mask"]
            torch.testing.assert_close(call["mask"], expected_mask, rtol=0, atol=0)
        # The encoder's original tensors remain available and unmodified;
        # only the denoiser handoff's all-valid mask convention changes.
        assert isinstance(encoded[prefix + "prompt_embeds_mask"], torch.Tensor)
    if not enabled:
        assert state.get("negative_prompt_embeds") is state.get("negative_prompt_embeds_mask") is None


def test_only_fresh_exact_qwen_t2i_rope_changes_and_blueprint_metadata_agrees():
    original = QwenImageModularPipeline().blocks
    tree = deepcopy(original)

    def inventory(block, path=()):
        entries = {path: block}
        for name, child in block.sub_blocks.items():
            entries.update(inventory(child, path + (name,)))
        return entries

    before = inventory(tree)
    assert native_blocks.prepare_native_pipeline_blocks(QwenImageModularPipeline, tree) is tree
    after = inventory(tree)
    assert set(before) == set(after)
    for path, block in before.items():
        if path != ROPE_PATH:
            assert after[path] is block
    assert after[ROPE_PATH] is not before[ROPE_PATH]
    adapted_type = getattr(native_blocks, "QwenImageWholePromptMaskRoPEInputsStep")
    assert type(block_at(tree, ROPE_PATH)) is adapted_type
    assert type(block_at(original, ROPE_PATH)) is QwenImageRoPEInputsStep
    assert type(block_at(tree, INPUT_PATHS[0])) is QwenImageTextInputsStep
    assert type(block_at(original, INPUT_PATHS[0])) is QwenImageTextInputsStep
    assert type(block_at(tree, ("text_encoder", "text_encoder"))) is QwenImageTextEncoderStep
    assert tree.input_names == original.input_names and tree.component_names == original.component_names
    for path in INPUT_PATHS[1:]:
        assert type(block_at(tree, path)) is type(block_at(original, path)) is QwenImageTextInputsStep
    assert type(block_at(deepcopy(tree), ROPE_PATH)) is adapted_type
    branch = block_at(tree, ("denoise", "text2image"))
    assert branch.block_names == block_at(original, ("denoise", "text2image")).block_names
    index = list(branch.block_names).index("prepare_rope_inputs")
    declaration = list(branch.block_classes)[index]
    assert declaration is adapted_type or type(declaration) is adapted_type


@pytest.mark.parametrize("path", INPUT_PATHS[1:])
def test_existing_native_sibling_input_masks_keep_their_exact_upstream_semantics(path):
    _, state, _, prepared, tree = run_native("pos6", "neg5", path=path)
    assert type(block_at(tree, path)) is QwenImageTextInputsStep
    assert state.get("prompt_embeds_mask") is prepared["prompt_embeds_mask"]
    assert state.get("negative_prompt_embeds_mask") is prepared["negative_prompt_embeds_mask"]
    assert bool(state.get("prompt_embeds_mask").all())
    assert bool(state.get("negative_prompt_embeds_mask").all())


@pytest.mark.parametrize("pipeline_class", [
    QwenImageEditPlusModularPipeline, QwenImageLayeredModularPipeline,
])
def test_edit_plus_and_layered_blueprints_are_not_traversed_or_mutated(pipeline_class):
    tree = pipeline_class().blocks
    original = deepcopy(tree)
    leaves = []

    def visit(block, path=()):
        leaves.append((path, block))
        for name, child in block.sub_blocks.items():
            visit(child, path + (name,))

    visit(tree)
    assert native_blocks.prepare_native_pipeline_blocks(pipeline_class, tree) is tree
    assert all(block_at(tree, path) is block for path, block in leaves)
    assert tree.input_names == original.input_names and tree.component_names == original.component_names


def test_edit_adapter_changes_only_opt_in_inpaint_media_leaves_not_prompt_masks_or_denoising():
    tree = QwenImageEditModularPipeline().blocks
    adapted = native_blocks.prepare_native_pipeline_blocks(QwenImageEditModularPipeline, tree)
    changed = set()

    def compare(left, right, path=()):
        assert left.sub_blocks.keys() == right.sub_blocks.keys()
        if type(left) is not type(right):
            changed.add(path)
        for name in left.sub_blocks:
            compare(left.sub_blocks[name], right.sub_blocks[name], path + (name,))

    compare(tree, adapted)
    assert changed == {
        ("vae_encoder", "edit_inpaint", "preprocess"),
        ("vae_encoder", "edit_inpaint", "encode"),
        ("decode", "inpaint_decode", "postprocess"),
        ("denoise", "edit_inpaint", "prepare_rope_inputs"),
    }
    # The T2I all-valid-mask normalization still never reaches Edit encoding,
    # input preparation or the ordinary Edit denoising branch. The separately
    # selected whole-inpaint convention changes only its own RoPE leaf.


@pytest.mark.parametrize("changed", ["outer", "selector", "branch", "input", "rope"])
def test_qwen_blueprint_drift_is_rejected_before_any_leaf_or_metadata_mutation(changed):
    tree = QwenImageModularPipeline().blocks
    denoise = tree.sub_blocks["denoise"]
    branch = denoise.sub_blocks["text2image"]
    original_input = branch.sub_blocks["input"]
    original_rope = branch.sub_blocks["prepare_rope_inputs"]
    original_classes = list(branch.block_classes)
    if changed == "outer":
        tree = SimpleNamespace(sub_blocks=tree.sub_blocks)
    elif changed == "selector":
        tree.sub_blocks["denoise"] = SimpleNamespace(sub_blocks=denoise.sub_blocks)
    elif changed == "branch":
        denoise.sub_blocks["text2image"] = SimpleNamespace(sub_blocks=branch.sub_blocks)
    elif changed == "input":
        class UnreviewedInput(QwenImageTextInputsStep):
            pass

        branch.sub_blocks["input"] = UnreviewedInput()
    else:
        class UnreviewedRoPE(QwenImageRoPEInputsStep):
            pass

        branch.sub_blocks["prepare_rope_inputs"] = UnreviewedRoPE()
    current_input = branch.sub_blocks["input"]
    current_rope = branch.sub_blocks["prepare_rope_inputs"]
    with pytest.raises(ValueError, match="reviewed"):
        native_blocks.prepare_native_pipeline_blocks(QwenImageModularPipeline, tree)
    assert branch.sub_blocks["input"] is current_input
    assert branch.sub_blocks["prepare_rope_inputs"] is current_rope
    assert list(branch.block_classes) == original_classes
    for path in INPUT_PATHS[1:]:
        assert type(block_at(denoise, path[1:])) is QwenImageTextInputsStep
    if changed != "input":
        assert current_input is original_input
    if changed != "rope":
        assert current_rope is original_rope


@pytest.mark.parametrize("changed", ["names", "declaration_count", "declared_input", "declared_rope"])
def test_qwen_t2i_declaration_drift_rejects_before_replacing_any_real_stage(changed):
    tree = QwenImageModularPipeline().blocks
    branch = block_at(tree, ("denoise", "text2image"))
    original_input = branch.sub_blocks["input"]
    original_rope = branch.sub_blocks["prepare_rope_inputs"]
    if changed == "names":
        branch.block_names = list(branch.block_names)[1:]
    elif changed == "declaration_count":
        branch.block_classes = list(branch.block_classes) + [QwenImageTextInputsStep()]
    elif changed == "declared_input":
        branch.block_classes = [SimpleNamespace(), *list(branch.block_classes)[1:]]
    else:
        declarations = list(branch.block_classes)
        declarations[3] = SimpleNamespace()
        branch.block_classes = declarations
    prior_names = list(branch.block_names)
    prior_declarations = list(branch.block_classes)
    with pytest.raises(ValueError, match="reviewed"):
        native_blocks.prepare_native_pipeline_blocks(QwenImageModularPipeline, tree)
    assert branch.sub_blocks["input"] is original_input
    assert branch.sub_blocks["prepare_rope_inputs"] is original_rope
    assert list(branch.block_names) == prior_names
    assert list(branch.block_classes) == prior_declarations


def test_a_same_named_unreviewed_pipeline_cannot_trigger_the_compatibility_adapter():
    tree = QwenImageModularPipeline().blocks
    original_input = block_at(tree, INPUT_PATHS[0])
    original_rope = block_at(tree, ROPE_PATH)
    unreviewed = type("QwenImageModularPipeline", (), {})
    assert native_blocks.prepare_native_pipeline_blocks(unreviewed, tree) is tree
    assert block_at(tree, INPUT_PATHS[0]) is original_input
    assert block_at(tree, ROPE_PATH) is original_rope


def test_actual_node_contract_prepares_t2i_rope_without_replacing_encoder_or_text_input():
    from modules.ModularDiffusers.modular_utils import require_modiff_node_contract

    denoise, _ = require_modiff_node_contract(QwenImageModularPipeline, "denoise")
    encoder, _ = require_modiff_node_contract(QwenImageModularPipeline, "text_encoder")
    adapted_type = getattr(native_blocks, "QwenImageWholePromptMaskRoPEInputsStep")
    assert type(block_at(denoise, ("text2image", "prepare_rope_inputs"))) is adapted_type
    assert type(block_at(denoise, ("text2image", "input"))) is QwenImageTextInputsStep
    assert type(block_at(encoder, ("text_encoder",))) is QwenImageTextEncoderStep
    for path in INPUT_PATHS[1:]:
        assert type(block_at(denoise, path[1:])) is QwenImageTextInputsStep


def test_real_reviewed_loader_keeps_the_adapter_after_t2i_workflow_pruning_without_weights():
    from modules.ModularDiffusers.loaders import _instantiate_reviewed_builtin_pipeline

    document = {
        "_class_name": "QwenImagePipeline",
        "transformer": ["diffusers", "QwenImageTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKLQwenImage"],
        "text_encoder": ["transformers", "Qwen2_5_VLForConditionalGeneration"],
        "tokenizer": ["transformers", "Qwen2Tokenizer"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    }
    with patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("No downloads allowed")):
        pipeline = _instantiate_reviewed_builtin_pipeline(
            "QwenImageModularPipeline", "Qwen/Qwen-Image-2512",
            index_filename="model_index.json", index_document=document,
            components_manager=ComponentsManager(), collection="qwen-mask-cpu-regression", workflow_id="text2image",
        )
    assert all(getattr(pipeline, name) is None for name in ("transformer", "vae", "text_encoder", "tokenizer"))
    inputs = []
    ropes = []

    def visit(block):
        if isinstance(block, QwenImageTextInputsStep):
            inputs.append(block)
        if isinstance(block, QwenImageRoPEInputsStep):
            ropes.append(block)
        for child in block.sub_blocks.values():
            visit(child)

    visit(pipeline.blocks)
    assert len(inputs) == 1
    assert type(inputs[0]) is QwenImageTextInputsStep
    assert len(ropes) == 1
    assert type(ropes[0]) is getattr(native_blocks, "QwenImageWholePromptMaskRoPEInputsStep")
