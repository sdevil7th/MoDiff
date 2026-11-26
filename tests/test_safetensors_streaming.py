"""Exercise original shard reads and native Accelerate hooks without downloads."""

from hashlib import sha256

import pytest
import torch
from diffusers import QwenImageTransformer2DModel
from transformers import LlamaConfig, LlamaForCausalLM

from modiff.safetensors_streaming import load_streamed_component, streaming_weight_budget, supports_streaming_profile


@pytest.fixture
def checkpoint(tmp_path):
    repo = tmp_path / "models--fixture--image"
    snapshot = repo / "snapshots" / ("1" * 40)
    component = snapshot / "transformer"
    reference = QwenImageTransformer2DModel(
        num_layers=2, attention_head_dim=16, num_attention_heads=2, joint_attention_dim=32,
        axes_dims_rope=(4, 6, 6),
    ).to(dtype=torch.bfloat16).eval()
    reference.save_pretrained(component, max_shard_size="10KB")
    return snapshot, reference


def test_real_sharded_diffusers_stream_matches_resident_on_repeat(checkpoint):
    snapshot, reference = checkpoint
    model = load_streamed_component(
        snapshot, "transformer", QwenImageTransformer2DModel, "diffusers", torch.bfloat16, "cpu",
    )
    assert all(param.is_meta for param in model.parameters())
    generator = torch.Generator().manual_seed(48)
    args = dict(
        hidden_states=torch.randn((1, 4, 64), generator=generator, dtype=torch.bfloat16),
        encoder_hidden_states=torch.randn((1, 3, 32), generator=generator, dtype=torch.bfloat16),
        timestep=torch.tensor([0.5], dtype=torch.bfloat16), img_shapes=[[(1, 2, 2)]],
    )
    with torch.inference_mode():
        expected = reference(**args).sample
        for _ in range(2):
            torch.testing.assert_close(model(**args).sample, expected, rtol=0, atol=0)
            assert all(param.is_meta for param in model.parameters())


def test_streaming_rejects_missing_shard(checkpoint):
    snapshot, _ = checkpoint
    next((snapshot / "transformer").glob("*.safetensors")).unlink()
    with pytest.raises(OSError):
        load_streamed_component(
            snapshot, "transformer", QwenImageTransformer2DModel, "diffusers", torch.bfloat16, "cpu",
        )


def test_streaming_rejects_shards_that_require_a_precision_conversion(checkpoint):
    snapshot, reference = checkpoint
    reference.to(dtype=torch.float32).save_pretrained(snapshot / "transformer", max_shard_size="10KB")
    with pytest.raises(ValueError, match="tensor shape or precision"):
        load_streamed_component(
            snapshot, "transformer", QwenImageTransformer2DModel, "diffusers", torch.bfloat16, "cpu",
        )
    assert not supports_streaming_profile(
        "ZImageModularPipeline", "Tongyi-MAI/Z-Image-Turbo", "text2image", "bfloat16",
    )


def test_hub_aliases_and_tied_transformers_parameters_survive_repeat(tmp_path):
    repo = tmp_path / "models--fixture--text"
    snapshot = repo / "snapshots" / ("2" * 40)
    blobs = repo / "blobs"
    blobs.mkdir(parents=True)
    reference = LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, tie_word_embeddings=True,
    )).to(dtype=torch.bfloat16).eval()
    reference.generation_config.max_new_tokens = 5
    reference.save_pretrained(snapshot, max_shard_size="2KB")
    reference = LlamaForCausalLM.from_pretrained(snapshot, dtype=torch.bfloat16, local_files_only=True).eval()
    for alias in snapshot.glob("*.safetensors"):
        blob = blobs / sha256(alias.read_bytes()).hexdigest()
        alias.rename(blob)
        try:
            alias.symlink_to(blob)
        except OSError as error:
            pytest.skip(f"Cannot create a Hub-style symlink here: {error}")
    model = load_streamed_component(snapshot, None, LlamaForCausalLM, "transformers", torch.bfloat16, "cpu")
    assert model.generation_config.max_new_tokens == 5
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        expected = reference(tokens, use_cache=False).logits
        for _ in range(2):
            torch.testing.assert_close(model(tokens, use_cache=False).logits, expected, rtol=0, atol=0)
            assert all(param.is_meta for param in model.parameters())


def test_manager_keeps_streamed_weights_until_final_collection_release(checkpoint):
    from modules.ModularDiffusers import _MoDiffComponentsManager

    snapshot, _ = checkpoint
    model = load_streamed_component(snapshot, "transformer", QwenImageTransformer2DModel,
                                    "diffusers", torch.bfloat16, "cpu")
    manager = _MoDiffComponentsManager()
    identity = manager.add("transformer", model, collection="first")
    assert manager.add("transformer", model, collection="second") == identity
    manager.remove_from_collection(identity, "first")
    assert manager.components[identity] is model
    assert next(model.parameters()).is_meta
    manager.remove_from_collection(identity, "second")
    assert identity not in manager.components
    assert not list(model.parameters())
    assert not any(hasattr(module, "_hf_hook") for module in model.modules())


@pytest.mark.parametrize("name,host_gib,device_gib,local_gib,shared", [
    ("Integrated Graphics", 32, 24, None, True),
    ("AMD Instinct MI300X VF", 240, 192, 192, False),
    ("AMD Radeon 8060S Graphics", 128, 48, 2, True),
])
def test_rocm_streaming_reserves_both_working_sets_only_for_shared_memory(
        monkeypatch, name, host_gib, device_gib, local_gib, shared):
    from modiff import hardware
    from modiff.safetensors_streaming import require_streaming_capacity

    gib = 1024**3
    memory = {"source": "/proc/meminfo", "available_bytes": 12 * gib, "total_bytes": host_gib * gib}
    monkeypatch.setattr(hardware, "system_memory_snapshot", lambda: memory)
    regions = [{"vram_total": local_gib * gib}] if local_gib is not None else []
    monkeypatch.setattr(hardware, "_linux_amd_memory_regions", lambda: regions)
    monkeypatch.setattr(torch.version, "hip", "reviewed-shared-pool")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: name)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (20 * gib, device_gib * gib))
    monkeypatch.setattr("sys.platform", "linux")
    reserve = {"systemRamBytes": 8 * gib, "vramBytes": 7 * gib, "systemCommitBytes": 12 * gib}
    if shared:
        with pytest.raises(ValueError, match=f"{15 * gib} bytes of systemRamBytes"):
            require_streaming_capacity(reserve, "cuda:0")
        memory["available_bytes"] = 16 * gib
    require_streaming_capacity(reserve, "cuda:0")


def test_budget_separates_windows_commit_and_rejects_casts():
    component = dict(loaderLibrary="diffusers", loaderClass="QwenImageTransformer2DModel",
                     requestedPayloadBytes=20 * 1024**3, largestSourceShardBytes=4 * 1024**3,
                     convertedTargetBytes=0)
    census = dict(dtype="bfloat16", components={"transformer": component})
    budget = streaming_weight_budget(census)
    assert budget["systemCommitBytes"] > budget["systemRamBytes"]
    component["convertedTargetBytes"] = 1
    assert streaming_weight_budget(census) is None
