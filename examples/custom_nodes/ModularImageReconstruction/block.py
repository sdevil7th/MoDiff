"""Reuse an AutoencoderKL supplied by any compatible model loader."""

import torch
from PIL import Image
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from diffusers.modular_pipelines import ComponentSpec, InputParam, ModularPipelineBlocks, OutputParam


class ImageReconstruction(ModularPipelineBlocks):
    @property
    def expected_components(self):
        return [ComponentSpec(name="vae", type_hint=AutoencoderKL)]

    @property
    def inputs(self):
        return [InputParam(name="image", type_hint=Image.Image), InputParam(name="amount", type_hint=float, default=1.0)]

    @property
    def intermediate_outputs(self):
        return [OutputParam(name="images", type_hint=list)]

    @property
    def description(self):
        return "Reconstruct an image with a connected VAE and blend it with the source."

    @torch.no_grad()
    def __call__(self, pipeline, state):
        amount = state.get("amount")
        if not 0 <= amount <= 1:
            raise ValueError("Reconstruction amount must be between 0 and 1.")
        vae = pipeline.vae
        processor = VaeImageProcessor(vae_scale_factor=2 ** (len(vae.config.block_out_channels) - 1))
        original_dtype = vae.dtype
        needs_upcast = original_dtype == torch.float16 and vae.config.force_upcast
        # Calling forward preserves the loader's existing device/offload hooks.
        # Posterior mode makes unchanged inputs deterministic; no generator is
        # consumed and no new model is loaded or independently placed.
        try:
            # AutoencoderKL delegates this precision policy to its caller.
            # Respect its declared setting without selecting a model family,
            # and restore the shared component even when forward fails.
            if needs_upcast:
                vae.to(dtype=torch.float32)
            sample = processor.preprocess(state.get("image")).to(device=pipeline._execution_device, dtype=vae.dtype)
            reconstructed = vae(sample, sample_posterior=False, return_dict=False)[0]
        finally:
            if needs_upcast:
                vae.to(dtype=original_dtype)
        if not torch.isfinite(reconstructed).all():
            raise ValueError("The connected VAE produced non-finite values; review its precision settings.")
        images = processor.postprocess(sample.lerp(reconstructed, amount), output_type="pil")
        state.set("images", images)
        return pipeline, state
