import pytest
import torch

from modules.ModularDiffusers.latents import LatentsPreview


@pytest.mark.parametrize("with_batch", (False, True))
def test_preview_preserves_single_four_channel_spatial_latents(with_batch):
    latents = torch.tensor([1.0, -1.0, 0.5, 0.25]).reshape(4, 1, 1).expand(4, 2, 3)
    if with_batch:
        latents = latents.unsqueeze(0)

    image = LatentsPreview().execute(latents)["image"]

    assert image.mode == "RGB"
    assert image.size == (6, 4)
    assert image.tobytes() == bytes([204, 184, 165]) * (6 * 4)


@pytest.mark.parametrize("shape", ((2, 4, 8, 8), (1, 64, 16), (1, 16, 8, 8), (4, 0, 8)))
def test_preview_explains_incompatible_latent_layouts(shape):
    latents = torch.zeros(shape)

    with pytest.raises(ValueError, match="single four-channel spatial latent.*Decode Latents"):
        LatentsPreview().execute(latents)


def test_preview_preserves_an_empty_input():
    assert LatentsPreview().execute(None) == {"image": None}
