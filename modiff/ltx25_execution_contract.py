"""Immutable source contract for the official LTX-2.5 distilled recipe.

This module records upstream behavior only.  Importing it never contacts the
Hub, loads weights, installs kernels, or advertises an executable capability.
Artifact access, optional-runtime, resource, and live-output qualification are
separate admission gates.
"""

from dataclasses import dataclass
from typing import Any, Callable


LTX25_DIFFUSERS_REVISION = "2f7e0154a9db246e95c9ede43edba7db5b130805"
LTX25_REPOSITORY = "Lightricks/LTX-2.5-Diffusers"
LTX25_REPOSITORY_REVISION = "a6de4b5354f078db24d9cf4778c14846788aea3d"
LTX25_DISTILLED_SIGMAS = (
    1.0,
    0.99375,
    0.9875,
    0.98125,
    0.975,
    0.909375,
    0.725,
    0.421875,
)


@dataclass(frozen=True)
class LTX25DistilledExecutionContract:
    pipeline_class: str = "LTX25ModularPipeline"
    blocks_class: str = "LTX25AutoBlocks"
    repository: str = LTX25_REPOSITORY
    repository_revision: str = LTX25_REPOSITORY_REVISION
    diffusers_revision: str = LTX25_DIFFUSERS_REVISION
    transformer_subfolder: str = "transformer"
    dtype: str = "bfloat16"
    sigmas: tuple[float, ...] = LTX25_DISTILLED_SIGMAS
    guidance_scale: float = 1.0
    stg_scale: float = 0.0
    modality_scale: float = 1.0
    guidance_rescale: float = 0.0
    audio_guidance_scale: float = 1.0
    audio_stg_scale: float = 0.0
    audio_modality_scale: float = 1.0
    audio_guidance_rescale: float = 0.0
    prompt_enhancement: bool = False
    decoder_attention_processor: str = "LTX2VideoVaeNeighborhoodNattenProcessor"
    decoder_tiling: bool = True
    output_type: str = "pil"
    quantization: str = "none"
    memory_reserve_margin: str = "20GB"
    width: int = 960
    height: int = 544
    num_frames: int = 121
    frame_rate: float = 24.0
    seed: int = 42


LTX25_DISTILLED_EXECUTION = LTX25DistilledExecutionContract()


def ltx25_distilled_denoise_kwargs(*, sigmas=None) -> dict[str, Any]:
    """Return the exact publisher schedule, rejecting step-count substitutes."""

    selected = LTX25_DISTILLED_SIGMAS if sigmas is None else tuple(sigmas)
    if selected != LTX25_DISTILLED_SIGMAS:
        raise ValueError("LTX-2.5 distilled execution requires the exact reviewed eight-sigma schedule.")
    return {"sigmas": list(LTX25_DISTILLED_SIGMAS)}


def configure_ltx25_distilled_denoise_components(
    pipeline,
    *,
    guider_factory: Callable[..., Any] | None = None,
):
    """Apply the source-required distilled video/audio guiders.

    The split executor initializes a pipeline for one upstream top-level block,
    so denoise and decode setup must also be independently applicable.
    """

    if guider_factory is None:
        from diffusers.modular_pipelines.ltx2.guider import LTX2Guidance

        guider_factory = LTX2Guidance
    update_components = getattr(pipeline, "update_components", None)
    if not callable(update_components):
        raise TypeError("The LTX-2.5 Modular pipeline must support update_components().")

    update_components(
        guider=guider_factory(
            guidance_scale=LTX25_DISTILLED_EXECUTION.guidance_scale,
            stg_scale=LTX25_DISTILLED_EXECUTION.stg_scale,
            modality_scale=LTX25_DISTILLED_EXECUTION.modality_scale,
            guidance_rescale=LTX25_DISTILLED_EXECUTION.guidance_rescale,
        ),
        audio_guider=guider_factory(
            guidance_scale=LTX25_DISTILLED_EXECUTION.audio_guidance_scale,
            stg_scale=LTX25_DISTILLED_EXECUTION.audio_stg_scale,
            modality_scale=LTX25_DISTILLED_EXECUTION.audio_modality_scale,
            guidance_rescale=LTX25_DISTILLED_EXECUTION.audio_guidance_rescale,
        ),
    )
    return pipeline


def configure_ltx25_diffusion_decoder(
    pipeline,
    *,
    decoder_processor_factory: Callable[[], Any] | None = None,
):
    """Verify/install the explicit NATTEN processor and enable tiling.

    Production execution accepts only an already provisioned exact processor;
    constructing it can resolve executable kernel code from the Hub. Tests and
    a future artifact-locked setup action may pass an explicit factory.
    """

    decoder = getattr(pipeline, "diffusion_decoder", None)
    set_processor = getattr(decoder, "set_attn_processor", None)
    enable_tiling = getattr(decoder, "enable_tiling", None)
    if not callable(set_processor) or not callable(enable_tiling):
        raise TypeError("The LTX-2.5 pipeline must expose a configurable tiled diffusion decoder.")

    if decoder_processor_factory is None:
        from diffusers.models.autoencoders.ltx2_diffusion_decoder import (
            LTX2VideoVaeNeighborhoodNattenProcessor,
        )

        processors = getattr(decoder, "attn_processors", None)
        if not isinstance(processors, dict) or not processors or any(
            type(processor) is not LTX2VideoVaeNeighborhoodNattenProcessor
            for processor in processors.values()
        ):
            raise RuntimeError(
                "LTX-2.5 exact diffusion decode requires the reviewed NATTEN processor to be provisioned "
                "through an explicit artifact-locked runtime setup before graph execution. The decoder node "
                "will not fetch Hub kernel code implicitly."
            )
    else:
        set_processor(decoder_processor_factory())
    enable_tiling()
    return pipeline


def configure_ltx25_distilled_components(
    pipeline,
    *,
    guider_factory: Callable[..., Any] | None = None,
    decoder_processor_factory: Callable[[], Any] | None = None,
):
    """Configure both stages when called on a complete Modular pipeline."""

    configure_ltx25_distilled_denoise_components(pipeline, guider_factory=guider_factory)
    configure_ltx25_diffusion_decoder(
        pipeline,
        decoder_processor_factory=decoder_processor_factory,
    )
    return pipeline
