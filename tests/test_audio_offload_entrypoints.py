"""Real upstream hooks must cover direct embeddings and tensor reads (no weights downloaded)."""
from types import SimpleNamespace

import pytest
import torch

from modiff.diffusers_offload import apply_pipeline_offload


class TextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4)])

    def forward(self, ids):
        return self.layers[0](self.embedding(ids))

    def get_input_embeddings(self):
        return self.embedding


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for actual transfer hooks")
@pytest.mark.parametrize("mode", ["group_cpu", "group_disk"])
def test_direct_embedding_and_condition_tensors_survive_repeated_offload(mode, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    encoder = TextEncoder()
    expected = encoder.embedding.weight.detach().clone()
    condition = torch.nn.Linear(4, 4)
    condition.register_buffer("silence_latent", torch.ones(1, 3, 4))
    components = {"text_encoder": encoder, "condition_encoder": condition}
    pipeline = SimpleNamespace(**components, components=components)
    result = apply_pipeline_offload(
        pipeline, mode=mode, device="cuda:0", node_id="audio",
        component_names=tuple(components), prefer_pipeline_group=False,
        leaf_level_components=("text_encoder",), resident_components=("condition_encoder",),
    )
    assert result.components == ["text_encoder"]
    ids = torch.tensor([[1, 2]], device="cuda:0")
    for _ in range(2):
        encoder(ids)
        lyrics = encoder.get_input_embeddings()(ids)
        torch.testing.assert_close(lyrics.cpu(), expected[torch.tensor([[1, 2]])])
        assert condition.silence_latent.device.type == "cuda"
        assert torch.all(condition.silence_latent == 1)


def test_sequential_preserves_existing_exclusions_without_changing_the_class():
    class Pipeline:
        _exclude_from_cpu_offload = ["existing"]

        def enable_sequential_cpu_offload(self, **kwargs):
            assert self._exclude_from_cpu_offload == ["existing", "condition_encoder"]

    pipeline = Pipeline()
    apply_pipeline_offload(pipeline, mode="sequential_cpu", device="cuda:0", node_id="audio",
                           resident_components=("condition_encoder",))
    assert Pipeline._exclude_from_cpu_offload == ["existing"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for actual transfer hooks")
@pytest.mark.parametrize("mode", ["group_cpu", "group_disk"])
def test_resident_weight_normalized_audio_vae_preserves_direct_encode(mode, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Oobleck uses legacy weight_norm forward pre-hooks. They execute before
    # Diffusers' wrapped leaf forward, so leaf offload rebuilds weight on CPU.
    vae = torch.nn.Sequential(torch.nn.utils.weight_norm(torch.nn.Conv1d(2, 2, 3)))
    signal = torch.randn(1, 2, 12)
    expected = vae(signal).detach()
    encoder = TextEncoder()
    components = {"text_encoder": encoder, "vae": vae}
    pipeline = SimpleNamespace(**components, components=components)
    result = apply_pipeline_offload(
        pipeline, mode=mode, device="cuda:0", node_id="audio-vae",
        component_names=tuple(components), prefer_pipeline_group=False,
        leaf_level_components=("text_encoder",), resident_components=("vae",),
    )
    assert result.components == ["text_encoder"]
    for _ in range(2):
        torch.testing.assert_close(vae(signal.cuda()).cpu(), expected)
