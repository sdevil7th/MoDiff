"""Deterministic no-weight coverage inventory for reviewed upstream surfaces.

This module deliberately inventories names before deciding whether they are useful
or safe.  An exported Diffusers class is not evidence that MoDiff can execute it;
unmatched names remain visibly ``unreviewed`` until a separate admission change
adds stronger evidence.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import re
import shutil
import subprocess
import tomllib
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from modiff.diffusers_profiles import EXPERIMENTAL_DIFFUSERS_PIPELINES
from modiff.modular_contract_only_registry import CURRENT_PIN_CONTRACT_ONLY_MODULAR_BY_NAME
from modiff.modular_workflow_contracts import PINNED_DIFFUSERS_REVISION
from modiff.optional_runtimes import OPTIONAL_RUNTIME_PROFILES, TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID
from modiff.studio_execution_specs import STUDIO_EXECUTION_SPEC_DEFINITIONS


UPSTREAM_COVERAGE_SCHEMA_VERSION = 1
UPSTREAM_COVERAGE_PATH = Path(__file__).resolve().parents[1] / "data" / "upstream-coverage.v1.json"
UPSTREAM_COVERAGE_STATUSES = (
    "executable",
    "equivalent",
    "contract-only",
    "research-blocked",
    "intentionally-excluded",
    "unreviewed",
)
TRANSFORMERS_REVIEWED_MAIN_REVISION = "c1ff11866b3e2c473f92460ee0bf68d739921609"
TRANSFORMERS_REVIEWED_MAIN_VERSION = "5.16.0.dev0"

_PIPELINE_SYMBOL = re.compile(r"[A-Za-z][A-Za-z0-9_]*Pipeline")
_PINNED_DIFFUSERS_DEPENDENCY = re.compile(
    r"^diffusers\s*@\s*git\+https://github\.com/huggingface/diffusers\.git@([0-9a-f]{40})$"
)

# These entries record a reviewed routing decision, not a claim that the two
# upstream implementations are byte-identical.  The targets are exact MoDiff
# adapter classes with registered execution specifications for the same public
# task surface.
_EQUIVALENT_PIPELINE_TARGETS = {
    "Flux2KleinModularPipeline": ("Flux2KleinPipeline",),
    "FluxKontextModularPipeline": ("FluxKontextPipeline",),
    "FluxModularPipeline": ("FluxPipeline", "FluxImg2ImgPipeline"),
    "FluxPriorReduxPipeline": ("FluxReduxPipeline",),
    # Upstream keeps this compatibility subclass solely to direct callers to
    # Lumina2Pipeline; it delegates initialization to that exact executable
    # class and emits a removal deprecation.
    "Lumina2Text2ImgPipeline": ("Lumina2Pipeline",),
    # The exact admitted condition adapters already expose these task surfaces
    # for the same LTX families. This is task equivalence only: optional direct
    # pipeline features such as prompt enhancement are not claimed.
    "LTX2ImageToVideoPipeline": ("LTX2ConditionPipeline",),
    "LTXImageToVideoPipeline": ("LTXConditionPipeline",),
    "LTXPipeline": ("LTXConditionPipeline",),
    "WanModularPipeline": ("WanPipeline",),
    "ZImageModularPipeline": ("ZImagePipeline", "ZImageImg2ImgPipeline"),
}

_INTENTIONALLY_EXCLUDED_PIPELINES = {
    "DiffusionPipeline": "Generic factory/base class; it is not an exact task adapter.",
    "ModularPipeline": "Generic composition base class; reviewed subclasses are inventoried separately.",
    "OnnxStableDiffusionImg2ImgPipeline": "ONNX is outside MoDiff's reviewed local execution dependency boundary.",
    "OnnxStableDiffusionInpaintPipeline": "ONNX is outside MoDiff's reviewed local execution dependency boundary.",
    "OnnxStableDiffusionPipeline": "ONNX is outside MoDiff's reviewed local execution dependency boundary.",
    "OnnxStableDiffusionUpscalePipeline": "ONNX is outside MoDiff's reviewed local execution dependency boundary.",
    "StableDiffusionOnnxPipeline": "ONNX is outside MoDiff's reviewed local execution dependency boundary.",
}

# This is a reviewed product decision layer over the static export inventory.
# A symbol listed here was inspected in the exact pinned Diffusers checkout;
# it is not promoted merely because it is importable. Keep the names explicit
# so an upstream rename/addition returns to ``unreviewed`` instead of silently
# inheriting a family-level claim.
_REVIEWED_NON_VIDEO_DEPRECATED_PIPELINES = frozenset(
    {
        "AltDiffusionImg2ImgPipeline",
        "AltDiffusionPipeline",
        "AmusedImg2ImgPipeline",
        "AmusedInpaintPipeline",
        "AmusedPipeline",
        "AudioDiffusionPipeline",
        "AudioLDMPipeline",
        "BlipDiffusionPipeline",
        "BlipDiffusionControlNetPipeline",
        "CycleDiffusionPipeline",
        "DanceDiffusionPipeline",
        "KarrasVePipeline",
        "LDMPipeline",
        "MusicLDMPipeline",
        "PNDMPipeline",
        "PaintByExamplePipeline",
        "RePaintPipeline",
        "ScoreSdeVePipeline",
        "SemanticStableDiffusionPipeline",
        "SpectrogramDiffusionPipeline",
        "StableDiffusionAttendAndExcitePipeline",
        "StableDiffusionControlNetXSPipeline",
        "StableDiffusionDiffEditPipeline",
        "StableDiffusionGLIGENPipeline",
        "StableDiffusionGLIGENTextImagePipeline",
        "StableDiffusionLDM3DPipeline",
        "StableDiffusionModelEditingPipeline",
        "StableDiffusionPanoramaPipeline",
        "StableDiffusionParadigmsPipeline",
        "StableDiffusionPix2PixZeroPipeline",
        "StableDiffusionSAGPipeline",
        "StableDiffusionXLControlNetXSPipeline",
        "UnCLIPImageVariationPipeline",
        "UnCLIPPipeline",
        "UniDiffuserPipeline",
        "VQDiffusionPipeline",
        "VersatileDiffusionDualGuidedPipeline",
        "VersatileDiffusionImageVariationPipeline",
        "VersatileDiffusionPipeline",
        "VersatileDiffusionTextToImagePipeline",
        "WuerstchenCombinedPipeline",
        "WuerstchenDecoderPipeline",
        "WuerstchenPriorPipeline",
    }
)

_REVIEWED_NON_VIDEO_RESEARCH_BLOCKED_PIPELINES = frozenset(
    {
        "BriaFiboEditPipeline",
        "BriaFiboPipeline",
        "BriaPipeline",
        "ChromaInpaintPipeline",
        "CogView4ControlPipeline",
        "Flux2KleinKVPipeline",
        "Flux2Pipeline",
        "FluxControlImg2ImgPipeline",
        "FluxControlInpaintPipeline",
        "FluxControlNetImg2ImgPipeline",
        "FluxControlNetInpaintPipeline",
        "FluxControlNetPipeline",
        "HunyuanDiTPAGPipeline",
        "Ideogram4Pipeline",
        "Kandinsky5I2IPipeline",
        "Kandinsky5T2IPipeline",
        "KandinskyV22Img2ImgPipeline",
        "KolorsPAGPipeline",
        "LDMSuperResolutionPipeline",
        "LLaDA2Pipeline",
        "LatentConsistencyModelImg2ImgPipeline",
        "MarigoldIntrinsicsPipeline",
        "MarigoldNormalsPipeline",
        "PRXPixelPipeline",
        "PixArtAlphaPipeline",
        "PixArtSigmaPAGPipeline",
        "QwenImageControlNetInpaintPipeline",
        "QwenImageControlNetPipeline",
        "QwenImageLayeredPipeline",
        "SanaControlNetPipeline",
        "SanaPAGPipeline",
        "ShapEImg2ImgPipeline",
        "StableDiffusion3PAGImg2ImgPipeline",
        "StableDiffusion3PAGPipeline",
        "StableDiffusionAdapterPipeline",
        "StableDiffusionControlNetImg2ImgPipeline",
        "StableDiffusionControlNetInpaintPipeline",
        "StableDiffusionControlNetPAGInpaintPipeline",
        "StableDiffusionControlNetPAGPipeline",
        "StableDiffusionDepth2ImgPipeline",
        "StableDiffusionImageVariationPipeline",
        "StableDiffusionInstructPix2PixPipeline",
        "StableDiffusionLatentUpscalePipeline",
        "StableDiffusionPAGImg2ImgPipeline",
        "StableDiffusionPAGInpaintPipeline",
        "StableDiffusionXLControlNetImg2ImgPipeline",
        "StableDiffusionXLControlNetInpaintPipeline",
        "StableDiffusionXLControlNetPAGImg2ImgPipeline",
        "StableDiffusionXLControlNetPAGPipeline",
        "StableDiffusionXLControlNetUnionImg2ImgPipeline",
        "StableDiffusionXLControlNetUnionInpaintPipeline",
        "StableDiffusionXLControlNetUnionPipeline",
        "StableUnCLIPImg2ImgPipeline",
        "StableUnCLIPPipeline",
        "ZImageControlNetInpaintPipeline",
        "ZImageControlNetPipeline",
        "ZImageOmniPipeline",
    }
)

_REVIEWED_VIDEO_DEPRECATED_PIPELINES = frozenset(
    {
        "I2VGenXLPipeline",
        "PIAPipeline",
        "TextToVideoSDPipeline",
        "TextToVideoZeroPipeline",
        "TextToVideoZeroSDXLPipeline",
        "VideoToVideoSDPipeline",
    }
)

# Each reason is intentionally class-specific. A related executable model or a
# reusable base checkpoint is not enough to admit a distinct conditioning,
# scheduler, component, or output contract.
_REVIEWED_VIDEO_RESEARCH_BLOCKED_PIPELINES = {
    "AnimateDiffControlNetPipeline": (
        "The class requires conditioning frames and a ControlNet component, while the admitted AnimateDiff "
        "workflow is prompt-only and has no exact combined artifact assembly or action contract."
    ),
    "AnimateDiffPAGPipeline": (
        "The class adds PAG layer selection and PAG scale semantics that the admitted AnimateDiff adapter does "
        "not expose; reuse of its base and motion weights is not feature equivalence."
    ),
    "AnimateDiffSDXLPipeline": (
        "The class requires an SDXL dual-text-encoder and compatible motion-adapter assembly, not the admitted "
        "SD1.5 AnimateDiff artifact contract."
    ),
    "AnimateDiffSparseControlNetPipeline": (
        "The class requires a SparseControlNetModel, sparse frame indices, and an immutable sparse-control "
        "artifact selection that MoDiff has not admitted."
    ),
    "AnimateDiffVideoToVideoControlNetPipeline": (
        "The class combines source-video denoising with ControlNet conditioning and independent strengths; "
        "MoDiff has no exact combined handler, field contract, or workflow."
    ),
    "AnimateDiffVideoToVideoPipeline": (
        "The prompt-only AnimateDiff action deliberately rejects source video; this class still needs an exact "
        "video-to-video handler, artifact assembly receipt, and canonical workflow before admission."
    ),
    "CogVideoXFunControlPipeline": (
        "The class requires a CogVideoX-Fun control-video artifact and control contract; the admitted CogVideoX-2B "
        "selection and action are text-to-video only."
    ),
    "CogVideoXImageToVideoPipeline": (
        "The admitted CogVideoX-2B artifact is text-to-video only and does not establish an image-conditioned "
        "transformer selection or image-to-video action contract."
    ),
    "CogVideoXVideoToVideoPipeline": (
        "The admitted CogVideoX-2B artifact is text-to-video only and does not establish a video-conditioned "
        "transformer selection or video-to-video action contract."
    ),
    "HunyuanSkyreelsImageToVideoPipeline": (
        "This SkyReels-specific Hunyuan image-to-video path has no exact immutable artifact, component recipe, "
        "or MoDiff action/workflow admission."
    ),
    "HunyuanVideoImageToVideoPipeline": (
        "The classic HunyuanVideo image-to-video class is distinct from the admitted FramePack assembly and the "
        "separately reviewed HunyuanVideo-1.5 family; no exact artifact/action contract exists."
    ),
    "HunyuanVideoPipeline": (
        "The classic HunyuanVideo text-to-video class is distinct from the admitted FramePack assembly and the "
        "separately reviewed HunyuanVideo-1.5 family; no exact artifact/action contract exists."
    ),
    "LTX2HDRPipeline": (
        "The class consumes an SDR reference video and an HDR IC-LoRA, then emits linear HDR video; MoDiff lacks "
        "that immutable adapter selection, HDR output contract, and exact action/workflow."
    ),
    "LTX2InContextPipeline": (
        "The pinned Modular inventory records IC-LoRA block truth, but this standard class still requires reference "
        "video conditions, an immutable IC-LoRA, and an exact runnable adapter/workflow."
    ),
    "LTX2LatentUpsamplePipeline": (
        "The latent upsampler is a separate second-stage component; MoDiff has no immutable upsampler selection or "
        "bounded latent-input action/workflow for this class."
    ),
    "LTXLatentUpsamplePipeline": (
        "The latent upsampler is a separate second-stage component; MoDiff has no immutable upsampler selection or "
        "bounded latent-input action/workflow for this class."
    ),
    "MotifVideoImage2VideoPipeline": (
        "The Motif-Video image-to-video family, guider, and model artifacts have no immutable MoDiff admission, "
        "resource envelope, or exact action/workflow."
    ),
    "MotifVideoPipeline": (
        "The Motif-Video text-to-video family, guider, and model artifacts have no immutable MoDiff admission, "
        "resource envelope, or exact action/workflow."
    ),
}

_REVIEWED_PIPELINE_DECISIONS = {
    **{
        name: {
            "status": "intentionally-excluded",
            "reason": (
                "The exact reviewed pin exports this implementation from Diffusers' deprecated namespace; "
                "MoDiff does not admit new workflows against upstream-deprecated pipeline surfaces."
            ),
            "review": "pinned-diffusers-non-video-source-triage",
        }
        for name in _REVIEWED_NON_VIDEO_DEPRECATED_PIPELINES
    },
    **{
        name: {
            "status": "research-blocked",
            "reason": (
                "The exact reviewed source was triaged, but this class/mode has no exact MoDiff execution "
                "specification backed by a reviewed immutable artifact selection; support for a related family "
                "is not equivalence."
            ),
            "review": "pinned-diffusers-non-video-source-triage",
        }
        for name in _REVIEWED_NON_VIDEO_RESEARCH_BLOCKED_PIPELINES
    },
    **{
        name: {
            "status": "intentionally-excluded",
            "reason": (
                "The exact reviewed pin marks this video implementation with DeprecatedPipelineMixin, whose "
                "contract receives no further fixes or feature updates; MoDiff does not add new workflows for it."
            ),
            "review": "pinned-diffusers-video-source-triage",
        }
        for name in _REVIEWED_VIDEO_DEPRECATED_PIPELINES
    },
    **{
        name: {
            "status": "research-blocked",
            "reason": reason,
            "review": "pinned-diffusers-video-source-triage",
        }
        for name, reason in _REVIEWED_VIDEO_RESEARCH_BLOCKED_PIPELINES.items()
    },
}

_TRANSFORMERS_SEMANTIC_DEFINITIONS = (
    {
        "id": "speech-recognition",
        "label": "Speech recognition and speech translation",
        "status": "executable",
        "reason": "Generic bounded speech nodes exist; remote output-quality qualification remains separate.",
        "qualification": "source-and-live-smoke-only",
        "modes": ["speech_to_text", "speech_translation"],
        "nodeKeys": [
            "modules.HuggingFaceSpeech.LoadSpeechRecognitionModel",
            "modules.HuggingFaceSpeech.TranscribeAudio",
        ],
        "canonicalWorkflowIds": [
            "HuggingFaceSpeechRecognitionModel:speech_to_text",
            "HuggingFaceSpeechRecognitionModel:speech_translation",
        ],
        "mainEvidence": {
            "models/auto/modeling_auto.py": ["AutoModelForSpeechSeq2Seq"],
            "pipelines/automatic_speech_recognition.py": ["AutomaticSpeechRecognitionPipeline"],
        },
        "productionEvidence": {
            "models/auto/modeling_auto.py": ["AutoModelForSpeechSeq2Seq"],
            "pipelines/automatic_speech_recognition.py": ["AutomaticSpeechRecognitionPipeline"],
        },
    },
    {
        "id": "bounded-causal-text-generation",
        "label": "Bounded causal text generation",
        "status": "executable",
        "reason": "Task-generic exact-revision/local loading and bounded generation nodes exist.",
        "qualification": "source-implemented-mocked-contract-qualified",
        "modes": ["text_generation"],
        "nodeKeys": [
            "modules.HuggingFaceTransformers.LoadTextGenerationModel",
            "modules.HuggingFaceTransformers.GenerateText",
        ],
        "canonicalWorkflowIds": [],
        "mainEvidence": {
            "models/auto/modeling_auto.py": ["AutoModelForCausalLM"],
            "pipelines/text_generation.py": ["TextGenerationPipeline"],
        },
        "productionEvidence": {
            "models/auto/modeling_auto.py": ["AutoModelForCausalLM"],
            "pipelines/text_generation.py": ["TextGenerationPipeline"],
        },
    },
    {
        "id": "bounded-image-video-to-text",
        "label": "Bounded image/video-to-text generation",
        "status": "executable",
        "reason": "Task-generic processor/model nodes accept bounded in-memory image or video inputs.",
        "qualification": "source-implemented-mocked-contract-qualified",
        "modes": ["image_to_text", "video_to_text"],
        "nodeKeys": [
            "modules.HuggingFaceTransformers.LoadImageTextToTextModel",
            "modules.HuggingFaceTransformers.GenerateImageVideoText",
        ],
        "canonicalWorkflowIds": [],
        "mainEvidence": {
            "models/auto/modeling_auto.py": ["AutoModelForImageTextToText"],
            "pipelines/image_text_to_text.py": ["ImageTextToTextPipeline"],
            "video_processing_utils.py": ["BaseVideoProcessor"],
        },
        "productionEvidence": {
            "models/auto/modeling_auto.py": ["AutoModelForImageTextToText"],
            "pipelines/image_text_to_text.py": ["ImageTextToTextPipeline"],
            "video_processing_utils.py": ["BaseVideoProcessor"],
        },
    },
    {
        "id": "any-to-any-generation",
        "label": "Any-to-any generation",
        "status": "executable",
        "reason": (
            "Task-generic nodes use a finite adapter registry; Janus text/image is bounded and executable while "
            "Qwen2.5-Omni remains contract-only because its official loader requires a non-safetensors speaker file."
        ),
        "qualification": "source-implemented-mocked-adapter-qualified",
        "modes": ["any_to_any_text", "text_to_image"],
        "nodeKeys": [
            "modules.HuggingFaceTransformers.LoadAnyToAnyModel",
            "modules.HuggingFaceTransformers.GenerateAnyToAny",
        ],
        "canonicalWorkflowIds": [],
        "mainEvidence": {
            "models/auto/modeling_auto.py": ["AutoModelForMultimodalLM"],
            "pipelines/any_to_any.py": ["AnyToAnyPipeline"],
        },
        "productionEvidence": {
            "models/auto/modeling_auto.py": ["AutoModelForMultimodalLM"],
            "pipelines/any_to_any.py": ["AnyToAnyPipeline"],
        },
    },
    {
        "id": "emu3-native-image-generation",
        "label": "Native Emu3 image generation",
        "status": "research-blocked",
        "reason": (
            "The production wheel has native token-to-image primitives, but MoDiff lacks a bounded decoder action."
        ),
        "qualification": "no-modiff-action-contract",
        "modes": ["text_to_image"],
        "nodeKeys": [],
        "canonicalWorkflowIds": [],
        "mainEvidence": {
            "models/emu3/modeling_emu3.py": ["Emu3ForConditionalGeneration", "decode_image_tokens"],
            "models/emu3/processing_emu3.py": ["return_for_image_generation", "generation_mode"],
        },
        "productionEvidence": {
            "models/emu3/modeling_emu3.py": ["Emu3ForConditionalGeneration", "decode_image_tokens"],
            "models/emu3/processing_emu3.py": ["return_for_image_generation", "generation_mode"],
        },
    },
    {
        "id": "cosmos3-edge-reasoner-orchestration",
        "label": "Cosmos3 Edge reasoner orchestration",
        "status": "research-blocked",
        "reason": "Reviewed main contains the reasoner, but production 5.14.1 and MoDiff orchestration do not.",
        "qualification": "reviewed-main-only-no-production-or-action-contract",
        "modes": ["image_video_reasoning"],
        "nodeKeys": [],
        "canonicalWorkflowIds": [],
        "mainEvidence": {
            "models/cosmos3_edge/modeling_cosmos3_edge.py": ["Cosmos3EdgeForConditionalGeneration"],
            "models/cosmos3_edge/processing_cosmos3_edge.py": ["Cosmos3EdgeProcessor"],
        },
        "productionEvidence": {},
        "productionAbsentPaths": [
            "models/cosmos3_edge/modeling_cosmos3_edge.py",
            "models/cosmos3_edge/processing_cosmos3_edge.py",
        ],
    },
)

_TEMPLATE_EXPORT_SCRIPT = r"""
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';

const modulePath = resolve(process.argv[1]);
const loaded = await import(pathToFileURL(modulePath).href);
const candidates = Object.values(loaded).filter((value) =>
  Array.isArray(value) &&
  value.length > 0 &&
  value.every((item) =>
    item &&
    typeof item === 'object' &&
    typeof item.id === 'string' &&
    typeof item.modelType === 'string' &&
    typeof item.mode === 'string' &&
    typeof item.verificationStatus === 'string'
  )
).sort((left, right) => right.length - left.length);
if (candidates.length === 0 || (candidates[1]?.length ?? -1) === candidates[0].length) {
  throw new Error(`Expected one largest public template export, found ${candidates.length} candidates.`);
}
// The production bundle minifies export names. The complete public catalogue
// is the unique largest export with this record shape; smaller exports are
// task-specific template subsets.
const templates = candidates[0].map((item) => ({
  id: item.id,
  modelType: item.modelType,
  mode: item.mode,
  verificationStatus: item.verificationStatus,
  readinessPolicy: item.readinessPolicy ?? null,
  evidencePolicy: item.evidencePolicy ?? 'generated',
}));
process.stdout.write(JSON.stringify(templates));
"""


class UpstreamCoverageError(RuntimeError):
    """Raised when one of the coverage sources is incomplete or ambiguous."""


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_sha256(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UpstreamCoverageError(f"Invalid canonical JSON: {path}") from error
    return _sha256_bytes(_stable_json(payload).encode("utf-8"))


def _status_counts(items: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(item["status"] for item in items)
    unknown = set(counts) - set(UPSTREAM_COVERAGE_STATUSES)
    if unknown:
        raise UpstreamCoverageError(f"Unknown coverage statuses: {sorted(unknown)}")
    return {status: counts.get(status, 0) for status in UPSTREAM_COVERAGE_STATUSES}


def _pinned_diffusers_revision(root: Path) -> str:
    with (root / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle).get("project", {})
    matches = []
    for dependency in project.get("dependencies", []):
        if not isinstance(dependency, str):
            continue
        match = _PINNED_DIFFUSERS_DEPENDENCY.fullmatch(dependency.strip())
        if match:
            matches.append(match.group(1))
    if matches != [PINNED_DIFFUSERS_REVISION]:
        raise UpstreamCoverageError(
            "The executable Diffusers dependency, workflow truth, and coverage generator must share one pin."
        )
    return matches[0]


def _normalize_diffusers_source(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    if resolved.is_file():
        if resolved.name != "__init__.py":
            raise UpstreamCoverageError("Diffusers source must resolve to its package or __init__.py.")
        resolved = resolved.parent
    if (resolved / "diffusers" / "__init__.py").is_file():
        resolved = resolved / "diffusers"
    if not (resolved / "__init__.py").is_file():
        raise UpstreamCoverageError("Diffusers source does not contain diffusers/__init__.py.")
    return resolved


def installed_diffusers_source() -> Path:
    """Locate package source without importing Diffusers or an optional backend."""

    import importlib.util

    specification = importlib.util.find_spec("diffusers")
    if specification is None or specification.origin is None:
        raise UpstreamCoverageError("The reviewed Diffusers source is not installed.")
    return _normalize_diffusers_source(Path(specification.origin))


def _git_diffusers_revision(source: Path) -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    root_result = subprocess.run(
        [git, "-C", str(source), "rev-parse", "--show-toplevel"],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    if root_result.returncode != 0:
        return None
    checkout_root = Path(root_result.stdout.strip()).resolve()
    remote_result = subprocess.run(
        [git, "-C", str(checkout_root), "config", "--get", "remote.origin.url"],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    remote = remote_result.stdout.strip().removesuffix("/").removesuffix(".git")
    if remote_result.returncode != 0 or not remote.endswith("huggingface/diffusers"):
        # A site-packages directory may be nested inside the MoDiff checkout.
        # Do not mistake that parent repository's HEAD for the upstream pin.
        return None
    head_result = subprocess.run(
        [git, "-C", str(checkout_root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    if head_result.returncode != 0:
        return None
    relative_init = (source / "__init__.py").relative_to(checkout_root)
    diff_result = subprocess.run(
        [git, "-C", str(checkout_root), "diff", "--quiet", "HEAD", "--", str(relative_init)],
        timeout=10,
        check=False,
    )
    if diff_result.returncode != 0:
        raise UpstreamCoverageError("The Diffusers export module has uncommitted source changes.")
    return head_result.stdout.strip()


def _installed_diffusers_revision(source: Path) -> str | None:
    try:
        distribution = importlib.metadata.distribution("diffusers")
        installed_root = Path(distribution.locate_file("")).resolve()
        source.relative_to(installed_root)
        direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    except (importlib.metadata.PackageNotFoundError, ValueError, json.JSONDecodeError, OSError):
        return None
    vcs_info = direct_url.get("vcs_info") if isinstance(direct_url, dict) else None
    revision = vcs_info.get("commit_id") if isinstance(vcs_info, dict) else None
    return revision if isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision) else None


def _verify_diffusers_source_revision(source: Path, expected_revision: str) -> str:
    revision = _git_diffusers_revision(source) or _installed_diffusers_revision(source)
    if revision != expected_revision:
        found = revision or "unverifiable source"
        raise UpstreamCoverageError(
            f"Diffusers source revision is {found}; expected reviewed pin {expected_revision}."
        )
    return revision


def _normalize_transformers_source(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    if resolved.is_file():
        if resolved.name != "__init__.py":
            raise UpstreamCoverageError("Transformers source must resolve to its package or __init__.py.")
        resolved = resolved.parent
    candidates = (resolved, resolved / "transformers", resolved / "src" / "transformers")
    for candidate in candidates:
        if (candidate / "__init__.py").is_file() and (candidate / "models" / "auto" / "modeling_auto.py").is_file():
            return candidate
    raise UpstreamCoverageError("Transformers source does not contain src/transformers.")


def _static_package_version(init_source: str, *, label: str) -> str:
    try:
        tree = ast.parse(init_source, filename=label)
    except SyntaxError as error:
        raise UpstreamCoverageError(f"{label} could not be parsed.") from error
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (
            isinstance(target, ast.Name)
            and target.id == "__version__"
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    raise UpstreamCoverageError(f"{label} has no static __version__ assignment.")


def _verify_transformers_main_source(source: Path) -> str:
    git = shutil.which("git")
    if git is None:
        raise UpstreamCoverageError("Git is required to verify the reviewed Transformers main checkout.")
    root_result = subprocess.run(
        [git, "-C", str(source), "rev-parse", "--show-toplevel"],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    if root_result.returncode != 0:
        raise UpstreamCoverageError("Transformers main source must be an exact Git checkout.")
    checkout_root = Path(root_result.stdout.strip()).resolve()
    remote_result = subprocess.run(
        [git, "-C", str(checkout_root), "config", "--get", "remote.origin.url"],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    remote = remote_result.stdout.strip().removesuffix("/").removesuffix(".git")
    if remote_result.returncode != 0 or not remote.endswith("huggingface/transformers"):
        raise UpstreamCoverageError("Transformers main source has the wrong upstream remote.")
    head_result = subprocess.run(
        [git, "-C", str(checkout_root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    revision = head_result.stdout.strip()
    if head_result.returncode != 0 or revision != TRANSFORMERS_REVIEWED_MAIN_REVISION:
        raise UpstreamCoverageError(
            f"Transformers main revision is {revision or 'unknown'}; expected {TRANSFORMERS_REVIEWED_MAIN_REVISION}."
        )
    relative_source = source.relative_to(checkout_root)
    diff_result = subprocess.run(
        [git, "-C", str(checkout_root), "diff", "--quiet", "HEAD", "--", str(relative_source)],
        timeout=10,
        check=False,
    )
    if diff_result.returncode != 0:
        raise UpstreamCoverageError("The reviewed Transformers main source has uncommitted changes.")
    version = _static_package_version(
        (source / "__init__.py").read_text(encoding="utf-8"), label="Transformers main __init__.py"
    )
    if version != TRANSFORMERS_REVIEWED_MAIN_VERSION:
        raise UpstreamCoverageError("The reviewed Transformers main version changed unexpectedly.")
    return version


def _transformers_production_contract() -> dict[str, Any]:
    profile = OPTIONAL_RUNTIME_PROFILES[TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID]
    packages = [package for package in profile.packages if package.distribution == "transformers"]
    locks = {
        (item["version"], item["filename"], item["sha256"], item["byteSize"])
        for item in profile.artifact_locks
        if item.get("distribution") == "transformers"
    }
    if len(packages) != 1 or len(locks) != 1:
        raise UpstreamCoverageError("The production Transformers runtime contract is ambiguous.")
    version, filename, sha256, byte_size = next(iter(locks))
    package = packages[0]
    if package.version != version:
        raise UpstreamCoverageError("The production Transformers package and wheel locks disagree.")
    return {
        "runtimeProfileId": profile.id,
        "runtimeProfileSpecDigest": profile.spec_digest,
        "version": version,
        "filename": filename,
        "sha256": sha256,
        "byteSize": byte_size,
    }


def _read_evidence(evidence: dict[str, list[str]], *, reader: Any, source_label: str) -> list[dict[str, Any]]:
    results = []
    for relative_path, symbols in evidence.items():
        try:
            content = reader(relative_path)
        except (KeyError, OSError) as error:
            raise UpstreamCoverageError(f"{source_label} lacks {relative_path}.") from error
        try:
            text_content = content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise UpstreamCoverageError(f"{source_label} has non-text evidence at {relative_path}.") from error
        missing = [symbol for symbol in symbols if re.search(rf"\b{re.escape(symbol)}\b", text_content) is None]
        if missing:
            raise UpstreamCoverageError(f"{source_label} evidence {relative_path} lacks symbols {missing}.")
        results.append(
            {
                "path": relative_path,
                "sha256": _sha256_bytes(content),
                "symbols": list(symbols),
            }
        )
    return results


def _node_action_evidence(root: Path, node_keys: list[str]) -> list[dict[str, Any]]:
    actions_by_path: dict[Path, list[str]] = defaultdict(list)
    for node_key in node_keys:
        parts = node_key.split(".")
        if len(parts) != 3 or parts[0] != "modules":
            raise UpstreamCoverageError(f"Invalid node key in Transformers coverage: {node_key}")
        actions_by_path[root / "modules" / parts[1] / "main.py"].append(parts[2])
    evidence = []
    for source_path in sorted(actions_by_path):
        try:
            content = source_path.read_bytes()
            tree = ast.parse(content.decode("utf-8"), filename=str(source_path))
        except (OSError, UnicodeDecodeError, SyntaxError) as error:
            raise UpstreamCoverageError(f"Transformers coverage node source is unavailable: {source_path}") from error
        class_names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        missing_actions = sorted(set(actions_by_path[source_path]) - class_names)
        if missing_actions:
            raise UpstreamCoverageError(f"Transformers coverage node actions are missing: {missing_actions}")
        evidence.append(
            {
                "path": source_path.relative_to(root).as_posix(),
                "sha256": _sha256_bytes(content),
                "actions": sorted(actions_by_path[source_path]),
            }
        )
    return evidence


def _transformers_semantic_coverage(
    root: Path,
    *,
    transformers_source: Path,
    transformers_wheel: Path,
    canonical_workflow_ids: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = _normalize_transformers_source(transformers_source)
    main_version = _verify_transformers_main_source(source)
    production = _transformers_production_contract()
    wheel = transformers_wheel.resolve(strict=True)
    if wheel.name != production["filename"] or wheel.stat().st_size != production["byteSize"]:
        raise UpstreamCoverageError("The production Transformers wheel does not match the reviewed artifact lock.")
    if _sha256_bytes(wheel.read_bytes()) != production["sha256"]:
        raise UpstreamCoverageError("The production Transformers wheel digest does not match the reviewed lock.")

    with zipfile.ZipFile(wheel) as archive:
        wheel_version = _static_package_version(
            archive.read("transformers/__init__.py").decode("utf-8"),
            label="Transformers production wheel __init__.py",
        )
        if wheel_version != production["version"]:
            raise UpstreamCoverageError("The production Transformers wheel version disagrees with its runtime lock.")

        items = []
        for definition in _TRANSFORMERS_SEMANTIC_DEFINITIONS:
            node_keys = list(definition["nodeKeys"])
            node_evidence = _node_action_evidence(root, node_keys)
            workflow_ids = list(definition["canonicalWorkflowIds"])
            missing_workflows = sorted(set(workflow_ids) - canonical_workflow_ids)
            if missing_workflows:
                raise UpstreamCoverageError(f"Transformers semantics lack canonical workflows: {missing_workflows}")
            main_evidence = _read_evidence(
                definition["mainEvidence"],
                reader=lambda path: (source / path).read_bytes(),
                source_label="Transformers reviewed main",
            )
            production_evidence = _read_evidence(
                definition["productionEvidence"],
                reader=lambda path: archive.read(f"transformers/{path}"),
                source_label="Transformers production wheel",
            )
            production_absent_paths = list(definition.get("productionAbsentPaths", []))
            unexpected_paths = [
                path for path in production_absent_paths if f"transformers/{path}" in archive.namelist()
            ]
            if unexpected_paths:
                raise UpstreamCoverageError(
                    f"Transformers production wheel now contains reviewed-main-only paths: {unexpected_paths}"
                )
            items.append(
                {
                    "id": definition["id"],
                    "label": definition["label"],
                    "status": definition["status"],
                    "reason": definition["reason"],
                    "qualification": definition["qualification"],
                    "modes": list(definition["modes"]),
                    "nodeKeys": node_keys,
                    "nodeEvidence": node_evidence,
                    "canonicalWorkflowIds": workflow_ids,
                    "publicTemplateEligible": False,
                    "reviewedMainEvidence": main_evidence,
                    "productionWheelSupport": bool(production_evidence),
                    "productionWheelEvidence": production_evidence,
                    "productionWheelAbsentPaths": production_absent_paths,
                }
            )
    scope = {
        "reviewedMainRevision": TRANSFORMERS_REVIEWED_MAIN_REVISION,
        "reviewedMainVersion": main_version,
        "reviewedMainDeliveredInProduction": False,
        "semanticInventoryRule": (
            "Finite reviewed inference semantics are classified; "
            "AutoModel aliases are not counted as product features."
        ),
        "productionRuntime": production,
    }
    return scope, items


def _diffusers_version_and_symbols(source: Path) -> tuple[str, list[str]]:
    init_path = source / "__init__.py"
    try:
        tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    except (OSError, SyntaxError) as error:
        raise UpstreamCoverageError("The Diffusers export module could not be parsed.") from error
    version = None
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == "__version__" and isinstance(node.value, ast.Constant):
            version = node.value.value
            break
    if not isinstance(version, str) or not version:
        raise UpstreamCoverageError("The Diffusers source has no static __version__ assignment.")
    symbols = sorted(
        {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and _PIPELINE_SYMBOL.fullmatch(node.value)
        }
    )
    if not symbols:
        raise UpstreamCoverageError("The Diffusers source exports no pipeline symbols.")
    return version, symbols


def _artifact_review_evidence(root: Path) -> dict[str, list[str]]:
    evidence: dict[str, set[str]] = defaultdict(set)

    def visit(value: Any, relative_path: str) -> None:
        if isinstance(value, dict):
            for child in value.values():
                visit(child, relative_path)
        elif isinstance(value, list):
            for child in value:
                visit(child, relative_path)
        elif isinstance(value, str) and _PIPELINE_SYMBOL.fullmatch(value):
            evidence[value].add(relative_path)

    for path in sorted((root / "data").glob("*-artifact-review.json")):
        relative_path = path.relative_to(root).as_posix()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise UpstreamCoverageError(f"Invalid artifact review: {relative_path}") from error
        visit(payload, relative_path)
    return {name: sorted(paths) for name, paths in evidence.items()}


def _load_public_templates(bundle_path: Path) -> list[dict[str, Any]]:
    node = shutil.which("node")
    if node is None:
        raise UpstreamCoverageError("Node.js is required to inspect the checked-in client template mirror.")
    result = subprocess.run(
        [node, "--input-type=module", "-e", _TEMPLATE_EXPORT_SCRIPT, str(bundle_path)],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise UpstreamCoverageError(
            "The checked-in client template mirror could not be inspected: "
            + (result.stderr.strip() or "unknown error")
        )
    try:
        templates = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise UpstreamCoverageError("The template mirror returned invalid JSON.") from error
    if not isinstance(templates, list) or not templates:
        raise UpstreamCoverageError("The template mirror returned no public templates.")
    ids = [item.get("id") for item in templates if isinstance(item, dict)]
    if len(ids) != len(templates) or len(ids) != len(set(ids)) or not all(isinstance(item, str) for item in ids):
        raise UpstreamCoverageError("Public template identifiers must be unique strings.")
    return sorted(templates, key=lambda item: item["id"])


def _pipeline_coverage(root: Path, source: Path) -> tuple[str, list[dict[str, Any]]]:
    version, symbols = _diffusers_version_and_symbols(source)
    symbol_set = set(symbols)
    exact_specs: dict[str, list[dict[str, str]]] = defaultdict(list)
    for spec_id, definition in STUDIO_EXECUTION_SPEC_DEFINITIONS.items():
        pipeline_class = definition["profile"]["pipeline_class"]
        exact_specs[pipeline_class].append(
            {"id": spec_id, "modelType": definition["modelType"], "mode": definition["mode"]}
        )
    for values in exact_specs.values():
        values.sort(key=lambda item: item["id"])

    contract_only = {
        capability["modelType"]
        for capability in EXPERIMENTAL_DIFFUSERS_PIPELINES
        if capability.get("qualificationStatus") == "contract_only"
    } | set(CURRENT_PIN_CONTRACT_ONLY_MODULAR_BY_NAME)

    modular_snapshot_path = root / "data" / "modular-workflow-contracts.json"
    modular_snapshot = json.loads(modular_snapshot_path.read_text(encoding="utf-8"))
    if modular_snapshot.get("diffusersRevision") != PINNED_DIFFUSERS_REVISION:
        raise UpstreamCoverageError("The Modular workflow snapshot has a different Diffusers revision.")
    modular_contracts = {
        item["pipelineClass"]: item for item in modular_snapshot.get("contracts", []) if isinstance(item, dict)
    }

    artifact_evidence = _artifact_review_evidence(root)
    pipeline_items = []
    for name in symbols:
        specifications = exact_specs.get(name, [])
        modular = modular_contracts.get(name)
        artifact_reviews = artifact_evidence.get(name, [])
        equivalent_targets = list(_EQUIVALENT_PIPELINE_TARGETS.get(name, ()))
        reviewed_decision = _REVIEWED_PIPELINE_DECISIONS.get(name)
        if specifications:
            status = "executable"
            reason = "At least one exact backend execution specification selects this upstream class."
        elif name in contract_only:
            status = "contract-only"
            reason = "Reviewed no-weight action/workflow truth exists, but no exact runnable mode is admitted."
        elif equivalent_targets:
            missing_targets = [target for target in equivalent_targets if target not in exact_specs]
            if missing_targets:
                raise UpstreamCoverageError(f"Equivalent targets for {name} are not executable: {missing_targets}")
            status = "equivalent"
            reason = (
                "The public task surface is routed through the listed exact adapter class or classes; this is not "
                "a claim of byte-identical outputs or parity with class-specific optional features."
            )
        elif name in _INTENTIONALLY_EXCLUDED_PIPELINES:
            status = "intentionally-excluded"
            reason = _INTENTIONALLY_EXCLUDED_PIPELINES[name]
        elif reviewed_decision:
            status = reviewed_decision["status"]
            reason = reviewed_decision["reason"]
        elif artifact_reviews:
            status = "research-blocked"
            reason = "Immutable source/artifact research exists, but executable admission remains blocked or pending."
        else:
            status = "unreviewed"
            reason = (
                "No exact adapter, equivalence decision, contract-only record, exclusion, "
                "or artifact review is recorded."
            )
        pipeline_items.append(
            {
                "name": name,
                "status": status,
                "reason": reason,
                "exactExecutionSpecs": specifications,
                "equivalentTo": equivalent_targets,
                "artifactReviews": artifact_reviews,
                "reviewDecision": reviewed_decision["review"] if reviewed_decision else None,
                "modularWorkflowIds": (
                    sorted(item["taskId"] for item in modular.get("workflows", [])) if modular else []
                ),
            }
        )
    if set(_EQUIVALENT_PIPELINE_TARGETS) - symbol_set:
        raise UpstreamCoverageError("An equivalent-source decision names a pipeline absent from the reviewed pin.")
    if set(_INTENTIONALLY_EXCLUDED_PIPELINES) - symbol_set:
        raise UpstreamCoverageError("An exclusion decision names a pipeline absent from the reviewed pin.")
    if set(_REVIEWED_PIPELINE_DECISIONS) - symbol_set:
        raise UpstreamCoverageError("A reviewed pipeline decision names a pipeline absent from the reviewed pin.")
    decision_conflicts = set(_REVIEWED_PIPELINE_DECISIONS) & (
        set(exact_specs) | contract_only | set(_EQUIVALENT_PIPELINE_TARGETS) | set(_INTENTIONALLY_EXCLUDED_PIPELINES)
    )
    if decision_conflicts:
        raise UpstreamCoverageError(
            f"Reviewed pipeline decisions conflict with stronger coverage evidence: {sorted(decision_conflicts)}"
        )
    return version, pipeline_items


def _workflow_and_template_coverage(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    manifest_path = root / "data" / "workflow-library-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    supported = manifest.get("workflows")
    experimental = manifest.get("experimentalWorkflows")
    if not isinstance(supported, list) or not isinstance(experimental, list):
        raise UpstreamCoverageError("The canonical workflow manifest is malformed.")
    workflow_records: dict[str, tuple[dict[str, Any], str]] = {}
    for status, records in (("executable", supported), ("contract-only", experimental)):
        for record in records:
            workflow_id = record.get("id") if isinstance(record, dict) else None
            if not isinstance(workflow_id, str) or workflow_id in workflow_records:
                raise UpstreamCoverageError("Canonical workflow identifiers must be unique strings.")
            graph_path = record.get("graphPath")
            graph_hash = record.get("graphHash")
            resolved_graph_path = root / "data" / "graphs" / str(graph_path)
            if not isinstance(graph_path, str) or not resolved_graph_path.is_file():
                raise UpstreamCoverageError(f"Canonical workflow {workflow_id} has no checked-in graph.")
            if not isinstance(graph_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", graph_hash):
                raise UpstreamCoverageError(f"Canonical workflow {workflow_id} has no exact graph hash.")
            if _canonical_json_sha256(resolved_graph_path) != graph_hash:
                raise UpstreamCoverageError(f"Canonical workflow {workflow_id} has a stale graph hash.")
            workflow_records[workflow_id] = (record, status)

    bundle_path = root / "web" / "assets" / "studio-templates.js"
    bundle_hash = _sha256_bytes(bundle_path.read_bytes())
    public_templates = _load_public_templates(bundle_path)
    gallery = json.loads((root / "web" / "template-gallery" / "manifest.json").read_text(encoding="utf-8"))
    gallery_records: dict[str, dict[str, Any]] = {}
    for item in gallery.get("examples", []):
        template_id = item.get("templateId") if isinstance(item, dict) else None
        if not isinstance(template_id, str) or template_id in gallery_records:
            raise UpstreamCoverageError("Gallery template identifiers must be unique strings.")
        gallery_records[template_id] = item
    public_template_ids = {item["id"] for item in public_templates}
    unknown_gallery_ids = set(gallery_records) - public_template_ids
    if unknown_gallery_ids:
        raise UpstreamCoverageError(f"Gallery examples name unknown public templates: {sorted(unknown_gallery_ids)}")
    template_items = []
    template_ids_by_workflow: dict[str, list[str]] = defaultdict(list)
    for template in public_templates:
        base_id = f"{template['modelType']}:{template['mode']}"
        variant_id = f"{base_id}:{template['id']}"
        workflow_id = variant_id if variant_id in workflow_records else base_id
        if workflow_id not in workflow_records:
            raise UpstreamCoverageError(
                f"Public template {template['id']} has no exact canonical workflow for {base_id}."
            )
        _, workflow_status = workflow_records[workflow_id]
        if workflow_status != "executable":
            raise UpstreamCoverageError(f"Public template {template['id']} resolves only to an experimental workflow.")
        template_ids_by_workflow[workflow_id].append(template["id"])
        gallery_record = gallery_records.get(template["id"])
        quality_review_status = gallery_record.get("qualityReviewStatus") if gallery_record else None
        template_items.append(
            {
                "id": template["id"],
                "status": "executable",
                "reason": "The public template resolves to an exact checked-in canonical workflow.",
                "modelType": template["modelType"],
                "mode": template["mode"],
                "canonicalWorkflowId": workflow_id,
                "verificationStatus": template["verificationStatus"],
                "readinessPolicy": template["readinessPolicy"],
                "evidencePolicy": template["evidencePolicy"],
                "galleryExamplePresent": gallery_record is not None,
                "galleryVerificationStatus": gallery_record.get("verificationStatus") if gallery_record else None,
                "galleryQualityReviewStatus": quality_review_status,
                "reviewedGalleryExample": (
                    isinstance(quality_review_status, str) and quality_review_status.startswith("approved_")
                ),
            }
        )

    workflow_items = []
    for workflow_id in sorted(workflow_records):
        record, status = workflow_records[workflow_id]
        workflow_items.append(
            {
                "id": workflow_id,
                "status": status,
                "reason": (
                    "The supported manifest contains an exact checked-in graph and backend execution pair."
                    if status == "executable"
                    else "The graph is explicitly experimental and is not a supported execution claim."
                ),
                "modelType": record["modelType"],
                "mode": record["mode"],
                "graphPath": record["graphPath"],
                "graphHash": record["graphHash"],
                "qualificationStatus": record.get("qualificationStatus"),
                "publicTemplateIds": sorted(template_ids_by_workflow.get(workflow_id, [])),
            }
        )
    return workflow_items, sorted(template_items, key=lambda item: item["id"]), bundle_hash


def build_upstream_coverage(
    root: Path,
    *,
    diffusers_source: Path | None = None,
    transformers_source: Path | None = None,
    transformers_wheel: Path | None = None,
) -> dict[str, Any]:
    """Build the complete deterministic coverage ledger without model weights."""

    root = root.resolve(strict=True)
    revision = _pinned_diffusers_revision(root)
    source = _normalize_diffusers_source(diffusers_source) if diffusers_source else installed_diffusers_source()
    verified_source_revision = _verify_diffusers_source_revision(source, revision)
    version, pipelines = _pipeline_coverage(root, source)
    workflows, templates, template_bundle_hash = _workflow_and_template_coverage(root)
    if transformers_source is None or transformers_wheel is None:
        raise UpstreamCoverageError(
            "Reviewed Transformers main source and the locked production wheel are required for coverage generation."
        )
    transformers_scope, transformers_semantics = _transformers_semantic_coverage(
        root,
        transformers_source=transformers_source,
        transformers_wheel=transformers_wheel,
        canonical_workflow_ids={item["id"] for item in workflows},
    )
    ledger = {
        "schemaVersion": UPSTREAM_COVERAGE_SCHEMA_VERSION,
        "allowedStatuses": list(UPSTREAM_COVERAGE_STATUSES),
        "scope": {
            "diffusers": {
                "revision": revision,
                "verifiedSourceRevision": verified_source_revision,
                "version": version,
                "exportModuleSha256": _sha256_bytes((source / "__init__.py").read_bytes()),
                "inventoryRule": "Every static top-level Diffusers export ending in Pipeline is classified once.",
            },
            "transformers": transformers_scope,
            "qualificationBoundary": (
                "Executable means source/graph admission only; it does not imply live output, Auto, Gallery, "
                "hardware, license, or release qualification."
            ),
            "unreviewedBoundary": (
                "An unreviewed export is an inventory gap, not a claim that the symbol is useful, safe, or eligible."
            ),
            "templateBundle": "web/assets/studio-templates.js",
            "templateBundleSha256": template_bundle_hash,
            "workflowManifest": "data/workflow-library-manifest.json",
        },
        "summary": {
            "diffusersPipelineSymbolCount": len(pipelines),
            "transformersSemanticCount": len(transformers_semantics),
            "canonicalWorkflowCount": len(workflows),
            "publicTemplateCount": len(templates),
            "pipelineStatusCounts": _status_counts(pipelines),
            "transformersSemanticStatusCounts": _status_counts(transformers_semantics),
            "transformersProductionSupportedSemanticCount": sum(
                item["productionWheelSupport"] for item in transformers_semantics
            ),
            "workflowStatusCounts": _status_counts(workflows),
            "templateStatusCounts": _status_counts(templates),
            "canonicalWorkflowsWithPublicTemplates": sum(bool(item["publicTemplateIds"]) for item in workflows),
            "canonicalWorkflowsWithoutPublicTemplates": sum(not item["publicTemplateIds"] for item in workflows),
            "reviewedGalleryTemplateCount": sum(item["reviewedGalleryExample"] for item in templates),
        },
        "diffusersPipelines": pipelines,
        "transformersSemantics": transformers_semantics,
        "canonicalWorkflows": workflows,
        "publicTemplates": templates,
    }
    semantic = dict(ledger)
    ledger["contentHash"] = "sha256:" + _sha256_bytes(_stable_json(semantic).encode("utf-8"))
    return ledger


def render_upstream_coverage(ledger: dict[str, Any]) -> str:
    return json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def load_upstream_coverage(path: Path = UPSTREAM_COVERAGE_PATH) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UpstreamCoverageError("The checked-in upstream coverage ledger is unavailable or invalid.") from error
    if payload.get("schemaVersion") != UPSTREAM_COVERAGE_SCHEMA_VERSION:
        raise UpstreamCoverageError("The upstream coverage ledger schema is unsupported.")
    expected_hash = payload.get("contentHash")
    semantic = dict(payload)
    semantic.pop("contentHash", None)
    actual_hash = "sha256:" + _sha256_bytes(_stable_json(semantic).encode("utf-8"))
    if expected_hash != actual_hash:
        raise UpstreamCoverageError("The upstream coverage ledger content hash is invalid.")
    return payload
