"""Exact, pinned image-variant decisions, separate from live qualification.

Source paths below are relative to the reviewed Diffusers package unless they
start with ``modules/``. No base-family output proves an exact variant works.
"""
from copy import deepcopy

from modiff.modular_workflow_contracts import PINNED_DIFFUSERS_REVISION

_REVIEWED_REVISION = "fbf49e7f35857f76bc57b177e26f12b03687c668"

def _native(profile, task, source, limitations=()):
    return {"decision": "native_recipe", "nativeProfileId": profile, "nativeTask": task,
            "source": source, "limitations": list(limitations)}


_PAG_LIMITS = (
    "Native PAG uses the pinned guider's exclusive start boundary: step zero is unperturbed.",
    "Fixed PAG scale only; adaptive PAG scale remains a standard-pipeline feature.",
    "Explicit mid-block preset targets all ten SDXL transformer layers; not bitwise equivalence.",
)
_REVIEWS = {
    f"{profile}/{task}": _native("sdxl-pag:modular", native_task,
        "guiders/perturbed_attention_guidance.py", _PAG_LIMITS)
    for profile, task, native_task in (
        ("sdxl-pag:direct", "text_to_image", "text_to_image"),
        ("sdxl-pag:img2img-direct", "edit_image", "image_to_image"),
        ("sdxl-pag:inpaint-direct", "inpaint", "inpaint"),
        ("sdxl-pag-controlnet-canny:direct", "control_image", "control_image"),
        ("sdxl-pag-controlnet-canny:img2img-direct", "control_edit_image", "control_edit_image"),
    )
}
_REVIEWS.update({
    "flux-dev:img2img-direct/edit_image": _native(
        "flux-dev:modular", "image_to_image", "modular_pipelines/flux/modular_blocks_flux.py",
        ("The standard edit_image task names the native image2image workflow, not Kontext editing.",)),
    "flux-krea:direct/text_to_image": _native(
        "flux-krea:modular", "text_to_image", "modular_pipelines/flux/before_denoise.py",
        ("Separate Krea artifact; 28 steps and guidance 3.5; no inherited dev output proof.",)),
    "flux-schnell:direct/text_to_image": _native(
        "flux-schnell:modular", "text_to_image", "modular_pipelines/flux/before_denoise.py",
        ("Four steps, guidance zero, 256-token encoder limit; guidance_embeds comes from the transformer config.",)),
    "sdxl-turbo:direct/text_to_image": _native(
        "sdxl-turbo:modular", "text_to_image", "modular_pipelines/stable_diffusion_xl/denoise.py",
        ("512px, one step, CFG disabled, repository scheduler and exact fp16 weight variant.",)),
    "flux2-dev:direct/multi_image_reference_edit": _native(
        "flux2:modular", "multi_image_reference_edit", "modular_pipelines/flux2/encoders.py",
        ("One to eight ordered independent references, not an image batch or stitched canvas.",)),
    "flux2-klein:direct/multi_image_reference_edit": _native(
        "flux2-klein:modular", "multi_image_reference_edit", "modular_pipelines/flux2/encoders.py",
        ("Distilled 4B weights; one to eight ordered independent references; no Klein KV equivalence.",)),
    "flux2-klein-kv:direct/text_to_image": _native(
        "flux2-klein-kv:t2i-modular", "text_to_image", "pipelines/flux2/pipeline_flux2_klein_kv.py",
        ("Text-only takes the standard-forward branch, with no reference K/V cache. Exact 9B KV index declares Flux2KleinPipeline, is_distilled=true, and the normal Flux2 transformer/VAE; this does not admit its cached-reference tasks or inherit 4B memory qualification.",)),
    "flux-kontext:direct/multi_image_reference_edit": _native(
        "flux-kontext:modular", "multi_image_reference_edit", "modules/ImageOperations/main.py:stitch_reference_images",
        ("Visible bounded equal-height horizontal composition precedes the native VAE encoder. Native lists alone represent a batch, not multiple references for one result.",)),
})
for _route in (
    "flux-dev:img2img-direct/edit_image@black-forest-labs/FLUX.1-dev-FP8",
    "flux-kontext:direct/multi_image_reference_edit@black-forest-labs/FLUX.1-Kontext-dev-NVFP4",
):
    _REVIEWS[_route] = {
        "decision": "artifact_blocked", "reason": "root_checkpoint_has_no_reviewed_modular_component_layout",
        "source": "modules/ModularDiffusers/loaders.py:_validate_reviewed_pipeline_index",
        "limitations": ["A root FP8/NVFP4 checkpoint is not a Diffusers component repository. No reviewed native conversion or dequantization recipe is declared; do not substitute base weights."],
    }
for _task in ("edit_image", "multi_image_reference_edit"):
    _REVIEWS[f"flux2-klein-kv:direct/{_task}"] = {
        "decision": "upstream_exception", "reason": "pinned_modular_klein_has_no_kv_extract_cached_path",
        "source": "pipelines/flux2/pipeline_flux2_klein_kv.py",
        "limitations": ["The standard KV pipeline extracts reference K/V once and reuses it on later steps. Pinned modular Klein does not implement kv_cache_mode=extract/cached; keep the exact standard KV executor for reference tasks."],
    }


def image_native_variant_review(route_id):
    if PINNED_DIFFUSERS_REVISION != _REVIEWED_REVISION:
        raise ValueError("Image variant decisions must be re-reviewed for the new Diffusers pin.")
    review = _REVIEWS.get(route_id)
    return {"status": "reviewed", "diffusersRevision": PINNED_DIFFUSERS_REVISION,
            "liveQualification": "not_established_by_review", **deepcopy(review)} if review else None
