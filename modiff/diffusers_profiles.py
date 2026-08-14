from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import re

from modiff.diffusers_offload_modes import (
    OFFLOAD_MODE_GROUP_CPU,
    OFFLOAD_MODE_GROUP_DISK,
    OFFLOAD_MODE_MODEL_CPU,
    OFFLOAD_MODE_NONE,
    OFFLOAD_MODE_SEQUENTIAL_CPU,
)
from modiff.modular_contract_only_registry import CURRENT_PIN_CONTRACT_ONLY_MODULAR_PIPELINES
from modiff.modular_workflow_discovery import reviewed_modular_workflow_contract
from modiff.modular_workflow_contracts import (
    FLUX_MODULAR_CONTROL_UNSUPPORTED,
)
from modiff.model_artifact_catalog import require_catalog_revision
from modiff.optional_runtimes import (
    TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID,
    optional_runtime_target,
    public_optional_runtime_profiles,
)
from modiff.studio_execution_specs import (
    FLUX_CANNY_REPO as FLUX_CANNY_REPO,
    FLUX_CANNY_VERIFIED_REPAIR_REPO as FLUX_CANNY_VERIFIED_REPAIR_REPO,
    FLUX_DEPTH_REPO as FLUX_DEPTH_REPO,
    FLUX_DEV_FP8_REPO as FLUX_DEV_FP8_REPO,
    FLUX_KREA_REPO as FLUX_KREA_REPO,
    FLUX_KONTEXT_NVFP4_REPO as FLUX_KONTEXT_NVFP4_REPO,
    FLUX_KONTEXT_REPO as FLUX_KONTEXT_REPO,
    FLUX_FILL_REPO as FLUX_FILL_REPO,
    FLUX2_KLEIN_REPO as FLUX2_KLEIN_REPO,
    FLUX_REDUX_REPO as FLUX_REDUX_REPO,
    FLUX_SCHNELL_REPO as FLUX_SCHNELL_REPO,
    ACE_STEP_LORA_BASE_REPO as ACE_STEP_LORA_BASE_REPO,
    ACE_STEP_REPO as ACE_STEP_REPO,
    LTX_VIDEO_FALLBACK_REPO as LTX_VIDEO_FALLBACK_REPO,
    LTX_VIDEO_REPO as LTX_VIDEO_REPO,
    WAN_22_I2V_A14B_REPO as WAN_22_I2V_A14B_REPO,
    WAN_22_TI2V_5B_REPO as WAN_22_TI2V_5B_REPO,
    WAN_T2V_1_3B_REPO as WAN_T2V_1_3B_REPO,
    studio_execution_profile_definitions,
)


QWEN_IMAGE_2512_REPO = "Qwen/Qwen-Image-2512"
QWEN_IMAGE_2512_PREQUANTIZED_REPO = "unsloth/Qwen-Image-2512-unsloth-bnb-4bit"

VERIFIED_REPAIR_SOURCES = {
    FLUX_CANNY_REPO: FLUX_CANNY_VERIFIED_REPAIR_REPO,
}

OPTIONAL_RUNTIME_REQUIREMENT_SCHEMA_VERSION = 1
OPTIONAL_RUNTIME_DELIVERY_BASE = "base"
OPTIONAL_RUNTIME_DELIVERY_OVERLAY = "optional_overlay"
OPTIONAL_RUNTIME_DELIVERIES = frozenset(
    {OPTIONAL_RUNTIME_DELIVERY_BASE, OPTIONAL_RUNTIME_DELIVERY_OVERLAY}
)
OPTIONAL_RUNTIME_PLATFORM_DELIVERIES = (
    ("linux", "x86_64", OPTIONAL_RUNTIME_DELIVERY_OVERLAY),
    ("linux", "arm64", OPTIONAL_RUNTIME_DELIVERY_BASE),
    ("macos", "x86_64", OPTIONAL_RUNTIME_DELIVERY_BASE),
    ("macos", "arm64", OPTIONAL_RUNTIME_DELIVERY_BASE),
    ("windows", "x86_64", OPTIONAL_RUNTIME_DELIVERY_OVERLAY),
    ("windows", "arm64", OPTIONAL_RUNTIME_DELIVERY_BASE),
)
_OPTIONAL_RUNTIME_PROFILE_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_EXECUTION_PROFILE_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._:-]{0,127}")
_GIB = 1024**3


@dataclass(frozen=True)
class ExpertCudaPolicy:
    schema_version: int
    blocked_dtypes: tuple[str, ...]
    recommended_dtype: str
    offloaded_vram_bytes: int
    resident_vram_bytes: int
    quantized_resident_vram_bytes: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        dtypes = {"float32", "float16", "bfloat16"}
        quantization_modes = {"bnb_4bit", "bnb_8bit", "quanto_float8", "torchao_float8"}
        quantized_modes = tuple(mode for mode, _ in self.quantized_resident_vram_bytes)
        byte_values = (
            self.offloaded_vram_bytes,
            self.resident_vram_bytes,
            *(value for _, value in self.quantized_resident_vram_bytes),
        )
        if (
            self.schema_version != 1
            or not self.blocked_dtypes
            or len(set(self.blocked_dtypes)) != len(self.blocked_dtypes)
            or not set(self.blocked_dtypes).issubset(dtypes)
            or self.recommended_dtype not in dtypes
            or self.recommended_dtype in self.blocked_dtypes
            or len(set(quantized_modes)) != len(quantized_modes)
            or not set(quantized_modes).issubset(quantization_modes)
            or any(not isinstance(value, int) or value <= 0 or value > 1024 * _GIB for value in byte_values)
        ):
            raise ValueError("Invalid reviewed Expert CUDA policy.")


QWEN_EXPERT_CUDA_POLICY = ExpertCudaPolicy(
    schema_version=1,
    blocked_dtypes=("float32",),
    recommended_dtype="bfloat16",
    offloaded_vram_bytes=10 * _GIB,
    resident_vram_bytes=80 * _GIB,
    quantized_resident_vram_bytes=(("bnb_4bit", 24 * _GIB),),
)


@dataclass(frozen=True)
class ExpertQuantizationPolicy:
    schema_version: int
    quantization_mode: str
    offload_mode: str
    modular_node: str
    subfolder: str
    component: str
    four_bit_quant_type: str
    compute_dtype: str
    double_quant: bool

    def __post_init__(self) -> None:
        token = re.compile(r"[a-z][a-z0-9_]{0,63}")
        if (
            self.schema_version != 1
            or self.quantization_mode != "bnb_4bit"
            or self.offload_mode
            not in {
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            }
            or re.fullmatch(r"modules\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+", self.modular_node) is None
            or token.fullmatch(self.subfolder) is None
            or token.fullmatch(self.component) is None
            or self.four_bit_quant_type not in {"nf4", "fp4"}
            or self.compute_dtype not in {"float32", "float16", "bfloat16"}
            or not isinstance(self.double_quant, bool)
        ):
            raise ValueError("Invalid reviewed Expert quantization policy.")


QWEN_EXPERT_QUANTIZATION_POLICY = ExpertQuantizationPolicy(
    schema_version=1,
    quantization_mode="bnb_4bit",
    offload_mode=OFFLOAD_MODE_MODEL_CPU,
    modular_node="modules.ModularDiffusers.QuantizationConfigNode",
    subfolder="transformer",
    component="qwen_low_vram",
    four_bit_quant_type="nf4",
    compute_dtype="bfloat16",
    double_quant=True,
)


@dataclass(frozen=True)
class ExpertMpsPolicy:
    schema_version: int
    qualification: str
    fallback_action: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.qualification not in {"unqualified", "experimental"}
            or self.fallback_action not in {"open_setup", "switch_to_z_image"}
        ):
            raise ValueError("Invalid reviewed Expert MPS policy.")


MPS_UNQUALIFIED_POLICY = ExpertMpsPolicy(1, "unqualified", "open_setup")
MPS_UNQUALIFIED_WITH_Z_IMAGE_FALLBACK_POLICY = ExpertMpsPolicy(
    1, "unqualified", "switch_to_z_image"
)
MPS_EXPERIMENTAL_POLICY = ExpertMpsPolicy(1, "experimental", "open_setup")


@dataclass(frozen=True)
class DiffusersExecutionProfile:
    id: str
    model_type: str
    modes: tuple[str, ...]
    loader_module: str
    loader_action: str
    execution_path: str
    pipeline_class: str
    default_repo: str
    fallback_repo: str | None
    quantizable_components: tuple[str, ...]
    default_quantized_components: tuple[str, ...]
    supported_offload_modes: tuple[str, ...]
    retry_offload_modes: tuple[str, ...]
    max_low_memory_side: int | None
    max_low_memory_steps: int | None
    live_proof: bool
    # All current profiles load Transformers-backed components and use the PEFT
    # integration surface.  A future pure-Diffusers profile must opt out with
    # ``optional_runtime_profiles=()`` rather than inheriting this composite.
    optional_runtime_profiles: tuple[str, ...] = (TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID,)
    # Keep optional-runtime discovery metadata separate from executable
    # delivery. Linux/Windows x86-64 use the qualified overlay; unqualified
    # architectures remain explicitly base-delivered.
    optional_runtime_delivery: str = OPTIONAL_RUNTIME_DELIVERY_OVERLAY
    optional_runtime_platform_deliveries: tuple[tuple[str, str, str], ...] = (
        OPTIONAL_RUNTIME_PLATFORM_DELIVERIES
    )
    compatible_repos: tuple[str, ...] = ()
    expert_quantization_modes: tuple[str, ...] = ()
    expert_cuda_policy: ExpertCudaPolicy | None = None
    expert_quantization_policy: ExpertQuantizationPolicy | None = None
    expert_mps_policy: ExpertMpsPolicy | None = None
    public: bool = True

    def __post_init__(self) -> None:
        expected_loader = {
            "modular-diffusers": ("modules.ModularDiffusers", "ModelsLoader"),
            "dynamic-modular": ("modules.ModularDiffusers", "DynamicBlockNode"),
            "direct-diffusers-image": ("modules.DiffusersImage", "LoadPipeline"),
            "direct-diffusers-video": ("modules.DiffusersVideo", "LoadPipeline"),
            "direct-wan-vace": ("modules.DiffusersVideo", "LoadPipeline"),
            "direct-diffusers-audio": ("modules.DiffusersAudio", "LoadPipeline"),
            "direct-diffusers-three-d": ("modules.DiffusersThreeD", "LoadPipeline"),
            "direct-huggingface-speech": (
                "modules.HuggingFaceSpeech",
                "LoadSpeechRecognitionModel",
            ),
            "direct-huggingface-transformers-text": (
                "modules.HuggingFaceTransformers",
                "LoadTextGenerationModel",
            ),
            "direct-huggingface-transformers-image-text": (
                "modules.HuggingFaceTransformers",
                "LoadImageTextToTextModel",
            ),
        }.get(self.execution_path)
        if expected_loader is None:
            raise ValueError(
                f"Diffusers execution profile {self.id!r} has unsupported execution path "
                f"{self.execution_path!r}."
            )
        if (self.loader_module, self.loader_action) != expected_loader:
            raise ValueError(
                f"Diffusers execution profile {self.id!r} loader "
                f"{self.loader_module}.{self.loader_action} does not match execution path "
                f"{self.execution_path!r}."
            )
        reviewed_quantization_modes = {
            "bnb_4bit",
            "bnb_8bit",
            "quanto_float8",
            "torchao_float8",
        }
        if (
            len(set(self.expert_quantization_modes)) != len(self.expert_quantization_modes)
            or not set(self.expert_quantization_modes).issubset(reviewed_quantization_modes)
        ):
            raise ValueError(f"Diffusers execution profile {self.id!r} has invalid Expert quantization modes.")
        targets = tuple(
            (platform_name, machine)
            for platform_name, machine, delivery in self.optional_runtime_platform_deliveries
            if delivery in OPTIONAL_RUNTIME_DELIVERIES
        )
        if (
            self.optional_runtime_delivery not in OPTIONAL_RUNTIME_DELIVERIES
            or len(targets) != len(self.optional_runtime_platform_deliveries)
            or len(set(targets)) != len(targets)
        ):
            raise ValueError(
                f"Diffusers execution profile {self.id!r} has invalid optional-runtime delivery targets."
            )

    @property
    def backend_path(self) -> str:
        """Return the legacy combined loader key from the explicit target."""

        return f"{self.loader_module}.{self.loader_action}"

    def optional_runtime_delivery_for_target(
        self,
        *,
        platform_name: str | None = None,
        machine: str | None = None,
    ) -> str:
        """Resolve reviewed delivery for one explicit OS/architecture target."""

        if not self.optional_runtime_platform_deliveries:
            return self.optional_runtime_delivery
        selected_platform, selected_machine = optional_runtime_target(
            platform_name=platform_name,
            machine=machine,
        )
        matches = tuple(
            delivery
            for target_platform, target_machine, delivery in self.optional_runtime_platform_deliveries
            if target_platform == selected_platform and target_machine == selected_machine
        )
        return matches[0] if len(matches) == 1 else "invalid"

    def to_public_dict(
        self,
        *,
        observe_optional_runtime: bool = False,
        optional_runtime_catalog_resolver=None,
    ) -> dict:
        data = asdict(self)
        public = {key: list(value) if isinstance(value, tuple) else value for key, value in data.items()}
        public["optional_runtime_delivery"] = self.optional_runtime_delivery_for_target()
        public["optional_runtime_platform_deliveries"] = [
            {"platform": platform_name, "machine": machine, "delivery": delivery}
            for platform_name, machine, delivery in self.optional_runtime_platform_deliveries
        ]
        if self.expert_cuda_policy:
            public["expert_cuda_policy"] = {
                **public["expert_cuda_policy"],
                "blocked_dtypes": list(self.expert_cuda_policy.blocked_dtypes),
                "quantized_resident_vram_bytes": [
                    list(item) for item in self.expert_cuda_policy.quantized_resident_vram_bytes
                ],
            }
        else:
            public.pop("expert_cuda_policy")
        if not self.expert_quantization_policy:
            public.pop("expert_quantization_policy")
        if not self.expert_quantization_modes:
            public.pop("expert_quantization_modes")
        if not self.expert_mps_policy:
            public.pop("expert_mps_policy")
        public["backend_path"] = self.backend_path
        if observe_optional_runtime:
            # Lazy to keep the declarative profile module independent of
            # overlay storage during registry import.
            from modiff.optional_runtime_execution import (
                optional_runtime_requirement_for_profiles as observed_requirement,
            )

            requirement = observed_requirement(
                (self,),
                catalog_resolver=optional_runtime_catalog_resolver,
            )
        else:
            requirement = optional_runtime_requirement_for_profiles((self,))
        public["optionalRuntimeRequirement"] = requirement
        return public


DIFFUSERS_EXECUTION_PROFILES: dict[str, DiffusersExecutionProfile] = {
    "custom-modular:reviewed-loader": DiffusersExecutionProfile(
        id="custom-modular:reviewed-loader",
        model_type="DummyCustomPipeline",
        modes=("reviewed_repository",),
        loader_module="modules.ModularDiffusers",
        loader_action="ModelsLoader",
        execution_path="modular-diffusers",
        pipeline_class="DummyCustomPipeline",
        default_repo="",
        fallback_repo=None,
        quantizable_components=(),
        default_quantized_components=(),
        supported_offload_modes=(OFFLOAD_MODE_NONE, OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK),
        retry_offload_modes=(OFFLOAD_MODE_GROUP_DISK,),
        max_low_memory_side=None,
        max_low_memory_steps=None,
        live_proof=False,
        public=False,
    ),
    "custom-modular:reviewed-block": DiffusersExecutionProfile(
        id="custom-modular:reviewed-block",
        model_type="DummyCustomPipeline",
        modes=("reviewed_repository",),
        loader_module="modules.ModularDiffusers",
        loader_action="DynamicBlockNode",
        execution_path="dynamic-modular",
        pipeline_class="DummyCustomPipeline",
        default_repo="",
        fallback_repo=None,
        quantizable_components=(),
        default_quantized_components=(),
        supported_offload_modes=(OFFLOAD_MODE_NONE, OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK),
        retry_offload_modes=(OFFLOAD_MODE_GROUP_DISK,),
        max_low_memory_side=None,
        max_low_memory_steps=None,
        live_proof=False,
        public=False,
    ),
    "z-image:auto": DiffusersExecutionProfile(
        id="z-image:auto",
        model_type="ZImageModularPipeline",
        modes=("text_to_image",),
        loader_module="modules.DiffusersImage",
        loader_action="LoadPipeline",
        execution_path="direct-diffusers-image",
        pipeline_class="ZImagePipeline",
        default_repo="Tongyi-MAI/Z-Image-Turbo",
        fallback_repo=None,
        quantizable_components=(),
        default_quantized_components=(),
        supported_offload_modes=(OFFLOAD_MODE_NONE, OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK),
        retry_offload_modes=(OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_DISK),
        max_low_memory_side=1024,
        max_low_memory_steps=8,
        live_proof=False,
    ),
    "qwen-image:t2i-direct": DiffusersExecutionProfile(
        id="qwen-image:t2i-direct",
        model_type="QwenImageModularPipeline",
        modes=("text_to_image",),
        loader_module="modules.DiffusersImage",
        loader_action="LoadPipeline",
        execution_path="direct-diffusers-image",
        pipeline_class="QwenImagePipeline",
        default_repo=QWEN_IMAGE_2512_REPO,
        fallback_repo=QWEN_IMAGE_2512_PREQUANTIZED_REPO,
        quantizable_components=("transformer", "text_encoder"),
        default_quantized_components=(),
        supported_offload_modes=(
            OFFLOAD_MODE_NONE,
            OFFLOAD_MODE_MODEL_CPU,
            OFFLOAD_MODE_SEQUENTIAL_CPU,
            OFFLOAD_MODE_GROUP_CPU,
            OFFLOAD_MODE_GROUP_DISK,
        ),
        retry_offload_modes=(OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK),
        max_low_memory_side=1328,
        max_low_memory_steps=50,
        live_proof=False,
    ),
    "qwen-image:modular": DiffusersExecutionProfile(
        id="qwen-image:modular",
        model_type="QwenImageModularPipeline",
        modes=("control_image",),
        loader_module="modules.ModularDiffusers",
        loader_action="ModelsLoader",
        execution_path="modular-diffusers",
        pipeline_class="QwenImageModularPipeline",
        default_repo=QWEN_IMAGE_2512_REPO,
        fallback_repo=None,
        quantizable_components=("transformer", "text_encoder"),
        default_quantized_components=("transformer", "text_encoder"),
        supported_offload_modes=(OFFLOAD_MODE_NONE, OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK),
        retry_offload_modes=(OFFLOAD_MODE_GROUP_DISK,),
        max_low_memory_side=768,
        max_low_memory_steps=28,
        live_proof=False,
    ),
    "qwen-edit:direct-inpaint": DiffusersExecutionProfile(
        id="qwen-edit:direct-inpaint",
        model_type="QwenImageEditModularPipeline",
        modes=("inpaint", "outpaint"),
        loader_module="modules.DiffusersImage",
        loader_action="LoadPipeline",
        execution_path="direct-diffusers-image",
        pipeline_class="QwenImageEditInpaintPipeline",
        default_repo="Qwen/Qwen-Image-Edit",
        fallback_repo=None,
        quantizable_components=("transformer", "text_encoder"),
        default_quantized_components=("transformer", "text_encoder"),
        supported_offload_modes=(
            OFFLOAD_MODE_NONE,
            OFFLOAD_MODE_MODEL_CPU,
            OFFLOAD_MODE_SEQUENTIAL_CPU,
            OFFLOAD_MODE_GROUP_CPU,
            OFFLOAD_MODE_GROUP_DISK,
        ),
        retry_offload_modes=(OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK),
        max_low_memory_side=768,
        max_low_memory_steps=24,
        live_proof=False,
    ),
    "qwen-edit:modular": DiffusersExecutionProfile(
        id="qwen-edit:modular",
        model_type="QwenImageEditModularPipeline",
        modes=("edit_image",),
        loader_module="modules.ModularDiffusers",
        loader_action="ModelsLoader",
        execution_path="modular-diffusers",
        pipeline_class="QwenImageEditModularPipeline",
        default_repo="Qwen/Qwen-Image-Edit",
        fallback_repo=None,
        quantizable_components=("transformer", "text_encoder"),
        default_quantized_components=("transformer", "text_encoder"),
        supported_offload_modes=(OFFLOAD_MODE_NONE, OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK),
        retry_offload_modes=(OFFLOAD_MODE_GROUP_DISK,),
        max_low_memory_side=768,
        max_low_memory_steps=24,
        live_proof=False,
    ),
    "qwen-edit-plus:modular": DiffusersExecutionProfile(
        id="qwen-edit-plus:modular",
        model_type="QwenImageEditPlusModularPipeline",
        modes=("edit_image", "multi_image_reference_edit"),
        loader_module="modules.ModularDiffusers",
        loader_action="ModelsLoader",
        execution_path="modular-diffusers",
        pipeline_class="QwenImageEditPlusModularPipeline",
        default_repo="Qwen/Qwen-Image-Edit-2511",
        fallback_repo=None,
        quantizable_components=("transformer", "text_encoder"),
        default_quantized_components=("transformer", "text_encoder"),
        supported_offload_modes=(OFFLOAD_MODE_NONE, OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK),
        retry_offload_modes=(OFFLOAD_MODE_GROUP_DISK,),
        max_low_memory_side=768,
        max_low_memory_steps=24,
        live_proof=False,
    ),
    "qwen-layered:modular": DiffusersExecutionProfile(
        id="qwen-layered:modular",
        model_type="QwenImageLayeredModularPipeline",
        modes=("layer_decomposition",),
        loader_module="modules.ModularDiffusers",
        loader_action="ModelsLoader",
        execution_path="modular-diffusers",
        pipeline_class="QwenImageLayeredModularPipeline",
        default_repo="Qwen/Qwen-Image-Layered",
        fallback_repo=None,
        quantizable_components=("transformer", "text_encoder"),
        default_quantized_components=("transformer", "text_encoder"),
        supported_offload_modes=(OFFLOAD_MODE_NONE, OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK),
        retry_offload_modes=(OFFLOAD_MODE_GROUP_DISK,),
        max_low_memory_side=768,
        max_low_memory_steps=30,
        live_proof=False,
    ),
    "wan-vace:direct": DiffusersExecutionProfile(
        id="wan-vace:direct",
        model_type="WanVACEPipeline",
        modes=(
            "text_to_video",
            "video_inpaint",
            "video_outpaint",
            "control_to_video",
        ),
        loader_module="modules.DiffusersVideo",
        loader_action="LoadPipeline",
        execution_path="direct-wan-vace",
        pipeline_class="WanVACEPipeline",
        default_repo="Wan-AI/Wan2.1-VACE-1.3B-diffusers",
        fallback_repo=None,
        quantizable_components=(),
        default_quantized_components=(),
        supported_offload_modes=(
            OFFLOAD_MODE_NONE,
            OFFLOAD_MODE_MODEL_CPU,
            OFFLOAD_MODE_SEQUENTIAL_CPU,
            OFFLOAD_MODE_GROUP_CPU,
            OFFLOAD_MODE_GROUP_DISK,
        ),
        retry_offload_modes=(OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK),
        max_low_memory_side=832,
        max_low_memory_steps=24,
        live_proof=False,
    ),
}

DIFFUSERS_EXECUTION_PROFILES.update(
    {
        profile_id: DiffusersExecutionProfile(**definition)
        for profile_id, definition in studio_execution_profile_definitions().items()
    }
)

for profile_id in (
    "qwen-image:t2i-direct",
    "qwen-image:img2img-direct",
    "qwen-image:inpaint-direct",
    "qwen-image:modular",
    "qwen-edit:direct-inpaint",
    "qwen-edit:modular",
    "qwen-edit-plus:modular",
    "qwen-layered:modular",
):
    DIFFUSERS_EXECUTION_PROFILES[profile_id] = replace(
        DIFFUSERS_EXECUTION_PROFILES[profile_id],
        expert_cuda_policy=QWEN_EXPERT_CUDA_POLICY,
        expert_quantization_policy=QWEN_EXPERT_QUANTIZATION_POLICY,
        expert_quantization_modes=("bnb_4bit",),
        expert_mps_policy=(
            MPS_UNQUALIFIED_WITH_Z_IMAGE_FALLBACK_POLICY
            if profile_id == "qwen-image:t2i-direct"
            else MPS_UNQUALIFIED_POLICY
        ),
    )

for profile_id in (
    "wan-vace:direct",
    "wan-22-image-to-video:direct",
    "wan-22-ti2v-5b:direct",
    "wan-text-to-video:direct",
    "wan-video-to-video:direct",
    "ltx-video:direct",
):
    DIFFUSERS_EXECUTION_PROFILES[profile_id] = replace(
        DIFFUSERS_EXECUTION_PROFILES[profile_id],
        expert_mps_policy=MPS_UNQUALIFIED_POLICY,
    )

DIFFUSERS_EXECUTION_PROFILES["z-image:auto"] = replace(
    DIFFUSERS_EXECUTION_PROFILES["z-image:auto"],
    expert_mps_policy=MPS_EXPERIMENTAL_POLICY,
)

SDXL_BASE_REPO = "stabilityai/stable-diffusion-xl-base-1.0"

EXPERIMENTAL_DIFFUSERS_PIPELINES = [
    {
        "modelType": "StableDiffusionXLModularPipeline",
        "label": "Stable Diffusion XL (Modular)",
        "mediaKind": "image",
        "defaultRepo": SDXL_BASE_REPO,
        "pipelineClasses": ["StableDiffusionXLModularPipeline"],
        "backendPath": "modules.ModularDiffusers.ModelsLoader",
        "executionKind": "modular",
        "runnableModes": ["text_to_image", "image_to_image", "control_image", "inpaint"],
        "inputContracts": {
            "image_to_image": {"requiredImages": ["referenceImages"]},
            "control_image": {"requiredImages": ["controlImage"]},
            "inpaint": {"requiredImages": ["referenceImages", "maskImage"]},
        },
        "qualificationStatus": "contract_only",
        "revisionCandidates": [require_catalog_revision(SDXL_BASE_REPO)],
        "autoEligible": False,
        "templateEligible": False,
        "galleryEligible": False,
    },
    {
        "modelType": "FluxModularPipeline",
        "label": "FLUX (Modular)",
        "mediaKind": "image",
        "pipelineClasses": ["FluxModularPipeline"],
        "backendPath": "modules.ModularDiffusers.ModelsLoader",
        "executionKind": "modular",
        "runnableModes": ["text_to_image", "image_to_image"],
        "unsupportedModes": {"control_image": FLUX_MODULAR_CONTROL_UNSUPPORTED},
    },
    {
        # Auto uses the established direct DiffusersImage facade for this
        # model/task pair. Keep the separately supported Modular workflow
        # visible as an Expert capability without creating a second Auto
        # execution profile for the same exact pair.
        "modelType": "ZImageModularPipeline",
        "label": "Z-Image (Modular)",
        "mediaKind": "image",
        "defaultRepo": "Tongyi-MAI/Z-Image-Turbo",
        "pipelineClasses": ["ZImageModularPipeline"],
        "backendPath": "modules.ModularDiffusers.ModelsLoader",
        "executionKind": "modular",
        "runnableModes": ["text_to_image"],
    },
    {
        "modelType": "Flux2KleinModularPipeline",
        "label": "FLUX.2 Klein (Standard Diffusers)",
        "mediaKind": "image",
        "defaultRepo": "black-forest-labs/FLUX.2-klein-4B",
        "pipelineClasses": ["Flux2KleinPipeline"],
        "backendPath": "modules.DiffusersImage.LoadPipeline",
        "executionKind": "standard",
        "executionModelType": "Flux2KleinPipeline",
        "executionProfileIds": ["flux2-klein:direct"],
        "runnableModes": ["text_to_image", "edit_image", "multi_image_reference_edit"],
    },
    {
        "modelType": "WanModularPipeline",
        "label": "Wan Text to Video (Modular)",
        "mediaKind": "video",
        "pipelineClasses": ["WanModularPipeline"],
        "backendPath": "modules.ModularDiffusers.ModelsLoader",
        "executionKind": "modular",
        "runnableModes": ["text_to_video"],
    },
]


# These adapters have a reviewed generic loader/action contract, but no Auto
# execution profile or public template.  Keep that distinction explicit: a
# contract-only capability lets Expert clients discover the exact backend
# class, modes, repository, and input shape without making the pair eligible
# for Auto selection or presenting static/mock evidence as live qualification.
#
# The registry tests compare this table with the task-module adapter maps and
# the immutable artifact catalog.  A new class therefore cannot be published
# here by copying a Diffusers name alone.
CONTRACT_ONLY_DIFFUSERS_PIPELINES = (
    # Standard image adapters with an immutable source and generic action
    # contract, but no Auto execution profile or public template.
    ("QwenImageEditPipeline", "image", "Qwen/Qwen-Image-Edit", ("edit_image",)),
    (
        "QwenImageEditPlusPipeline",
        "image",
        "Qwen/Qwen-Image-Edit-2511",
        ("edit_image", "multi_image_reference_edit"),
    ),
    (
        "LatentConsistencyModelImg2ImgPipeline",
        "image",
        "SimianLuo/LCM_Dreamshaper_v7",
        ("edit_image",),
    ),
    (
        "StableDiffusionPAGImg2ImgPipeline",
        "image",
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        ("edit_image",),
    ),
    (
        "StableDiffusionPAGInpaintPipeline",
        "image",
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        ("inpaint",),
    ),
    (
        "HunyuanDiTPAGPipeline",
        "image",
        "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled",
        ("text_to_image",),
    ),
    (
        "PixArtSigmaPAGPipeline",
        "image",
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        ("text_to_image",),
    ),
    (
        "SanaPAGPipeline",
        "image",
        "Efficient-Large-Model/Sana_600M_1024px_diffusers",
        ("text_to_image",),
    ),
    ("ZImageInpaintPipeline", "image", "Tongyi-MAI/Z-Image-Turbo", ("inpaint", "outpaint")),
    (
        "FluxKontextInpaintPipeline",
        "image",
        FLUX_KONTEXT_REPO,
        ("inpaint", "outpaint"),
    ),
    ("Flux2KleinInpaintPipeline", "image", FLUX2_KLEIN_REPO, ("inpaint", "outpaint")),
)


_CONTRACT_ONLY_BACKEND_PATHS = {
    "image": "modules.DiffusersImage.LoadPipeline",
    "video": "modules.DiffusersVideo.LoadPipeline",
    "audio": "modules.DiffusersAudio.LoadPipeline",
}

_CONTRACT_ONLY_INPUT_CONTRACTS = {
    "edit_image": {"requiredImages": ["referenceImages"]},
    "multi_image_reference_edit": {"requiredImages": ["referenceImages"]},
    "inpaint": {"requiredImages": ["referenceImages", "maskImage"]},
    "outpaint": {"requiredImages": ["referenceImages"]},
    "image_to_video": {"requiredImages": ["referenceImages"]},
    "video_to_video": {"requiredVideos": ["sourceVideo"]},
    "reference_to_video": {"requiredImages": ["referenceImages"]},
    "character_animate": {
        "requiredImages": ["referenceImages"],
        "requiredVideos": ["poseVideo", "faceVideo"],
    },
    "character_replace": {
        "requiredImages": ["referenceImages"],
        "requiredVideos": ["poseVideo", "faceVideo", "backgroundVideo", "maskVideo"],
    },
}


def _contract_only_pipeline_capability(
    pipeline_class: str,
    media_kind: str,
    default_repo: str,
    modes: tuple[str, ...],
) -> dict:
    revision = require_catalog_revision(default_repo)
    return {
        "modelType": pipeline_class,
        "label": pipeline_class,
        "mediaKind": media_kind,
        "defaultRepo": default_repo,
        "pipelineClasses": [pipeline_class],
        "backendPath": _CONTRACT_ONLY_BACKEND_PATHS[media_kind],
        "executionKind": "standard",
        "runnableModes": list(modes),
        "inputContracts": {
            mode: dict(_CONTRACT_ONLY_INPUT_CONTRACTS[mode])
            for mode in modes
            if mode in _CONTRACT_ONLY_INPUT_CONTRACTS
        },
        "qualificationStatus": "contract_only",
        "revisionCandidates": [revision],
        "autoEligible": False,
        "templateEligible": False,
        "galleryEligible": False,
    }


EXPERIMENTAL_DIFFUSERS_PIPELINES.extend(
    _contract_only_pipeline_capability(*contract)
    for contract in CONTRACT_ONLY_DIFFUSERS_PIPELINES
)


def _contract_only_modular_capability(specification) -> dict:
    workflow_contract = reviewed_modular_workflow_contract(specification.class_name)
    return {
        "modelType": specification.class_name,
        "label": specification.label,
        "mediaKind": specification.batch,
        "pipelineClasses": [specification.class_name],
        "backendPath": "modules.ModularDiffusers.ModelsLoader",
        "executionKind": "modular",
        "runnableModes": [],
        "upstreamWorkflows": [workflow["taskId"] for workflow in workflow_contract["workflows"]],
        "workflowContractSchemaVersion": workflow_contract["schemaVersion"],
        "contractBatch": specification.batch,
        "qualificationStatus": "contract_only",
        "expertVisible": True,
        "autoEligible": False,
        "templateEligible": False,
        "galleryEligible": False,
        "optionalRuntimeProfileIds": [],
    }


EXPERIMENTAL_DIFFUSERS_PIPELINES.extend(
    _contract_only_modular_capability(specification)
    for specification in CURRENT_PIN_CONTRACT_ONLY_MODULAR_PIPELINES
)


def optional_runtime_requirement_for_profiles(
    profiles: tuple[DiffusersExecutionProfile, ...] | list[DiffusersExecutionProfile],
    *,
    platform_name: str | None = None,
    machine: str | None = None,
) -> dict:
    """Describe whether selected execution profiles require an overlay now.

    This helper is declarative and deliberately does not inspect installed
    distributions or managed overlay state.  Runtime observations are added by
    :mod:`modiff.optional_runtime_execution` only for ``optional_overlay``
    delivery.
    """

    selected = tuple(profiles)
    profile_ids: list[str] = []
    invalid_profile_ids = False
    deliveries: set[str] = set()
    execution_profile_ids: list[str] = []
    base_profile_with_optional_ids = False
    for profile in selected:
        if (
            isinstance(profile.id, str)
            and _EXECUTION_PROFILE_ID_PATTERN.fullmatch(profile.id)
        ):
            if profile.id in execution_profile_ids:
                invalid_profile_ids = True
            else:
                execution_profile_ids.append(profile.id)
        else:
            invalid_profile_ids = True
        delivery = profile.optional_runtime_delivery_for_target(
            platform_name=platform_name,
            machine=machine,
        )
        if delivery not in OPTIONAL_RUNTIME_DELIVERIES:
            deliveries.add("invalid")
        else:
            deliveries.add(delivery)
        valid_ids_for_profile = 0
        profile_seen_ids: set[str] = set()
        for raw_profile_id in profile.optional_runtime_profiles:
            if (
                not isinstance(raw_profile_id, str)
                or not _OPTIONAL_RUNTIME_PROFILE_ID_PATTERN.fullmatch(raw_profile_id)
                or raw_profile_id in profile_seen_ids
            ):
                invalid_profile_ids = True
                continue
            profile_id = raw_profile_id
            profile_seen_ids.add(profile_id)
            valid_ids_for_profile += 1
            if profile_id not in profile_ids:
                profile_ids.append(profile_id)
        if delivery == OPTIONAL_RUNTIME_DELIVERY_BASE and valid_ids_for_profile:
            base_profile_with_optional_ids = True

    requires_overlay = OPTIONAL_RUNTIME_DELIVERY_OVERLAY in deliveries
    contract_invalid = bool(
        invalid_profile_ids
        or len(profile_ids) > 32
        or len(execution_profile_ids) > 32
        or "invalid" in deliveries
        or (requires_overlay and base_profile_with_optional_ids)
        or (requires_overlay and not profile_ids)
    )
    if contract_invalid:
        delivery = OPTIONAL_RUNTIME_DELIVERY_OVERLAY
        required_now = True
        state = "unavailable"
        reason = "execution_profile_contract_invalid"
    elif requires_overlay:
        delivery = OPTIONAL_RUNTIME_DELIVERY_OVERLAY
        required_now = True
        state = "unavailable"
        reason = "optional_runtime_status_required"
    else:
        delivery = OPTIONAL_RUNTIME_DELIVERY_BASE
        required_now = False
        state = "base_satisfied"
        reason = "base_runtime_contract" if profile_ids else "no_optional_runtime_required"

    return {
        "schemaVersion": OPTIONAL_RUNTIME_REQUIREMENT_SCHEMA_VERSION,
        "delivery": delivery,
        "requiredNow": required_now,
        "profileIds": profile_ids[:32],
        "executionProfileIds": execution_profile_ids[:32],
        "state": state,
        "reason": reason,
    }


def execution_profiles_for_execution(
    model_type: str,
    mode: str | None = None,
) -> tuple[DiffusersExecutionProfile, ...]:
    """Resolve declared profiles for one backend-owned model/mode pair."""

    normalized_model_type = str(model_type or "").strip()
    normalized_mode = str(mode or "").strip()
    return tuple(
        profile
        for profile in DIFFUSERS_EXECUTION_PROFILES.values()
        if profile.model_type == normalized_model_type
        and (not normalized_mode or normalized_mode in profile.modes)
    )


def optional_runtime_requirement_for_execution(
    model_type: str,
    mode: str | None = None,
) -> dict:
    """Return declarative optional-runtime delivery for one exact pair."""

    return optional_runtime_requirement_for_profiles(
        execution_profiles_for_execution(model_type, mode)
    )


def resolve_execution_profiles_for_loader(
    module: str,
    action: str,
    values: dict,
) -> tuple[tuple[DiffusersExecutionProfile, ...], str | None]:
    """Resolve an executable loader from authoritative node parameters.

    The returned reason is non-``None`` when the loader belongs to a declared
    Diffusers backend path but its exact execution profile cannot be proven.
    Base delivery intentionally tolerates that legacy/Expert ambiguity.  Once a
    relevant profile selects ``optional_overlay``, the execution guard treats
    the same reason as a fail-closed contract blocker.
    """

    backend_path = f"{str(module or '').strip()}.{str(action or '').strip()}"
    backend_profiles = tuple(
        profile
        for profile in DIFFUSERS_EXECUTION_PROFILES.values()
        if profile.backend_path == backend_path
    )
    if not backend_profiles:
        return (), None
    if not isinstance(values, dict):
        return backend_profiles, "loader_parameters_invalid"

    identity_key = "model_type" if action == "ModelsLoader" else "pipeline_class"
    raw_identity = "DummyCustomPipeline" if action == "DynamicBlockNode" else values.get(identity_key)
    identity = raw_identity.strip() if isinstance(raw_identity, str) else ""
    if not identity:
        return backend_profiles, "loader_identity_missing"

    matching = tuple(
        profile
        for profile in backend_profiles
        if (
            profile.model_type == identity
            if action == "ModelsLoader"
            else profile.pipeline_class == identity
        )
    )
    if not matching:
        return backend_profiles, "loader_selection_unregistered"
    if len(matching) == 1:
        return matching, None

    raw_repository = values.get("model_id") or values.get("repo_id")
    if isinstance(raw_repository, str):
        repository = raw_repository.strip()
        repository_source = "hub"
    elif (
        isinstance(raw_repository, dict)
        and set(raw_repository).issubset({"source", "value"})
        and raw_repository.get("source") in {"hub", "local"}
        and isinstance(raw_repository.get("value"), str)
    ):
        repository = raw_repository["value"].strip()
        repository_source = raw_repository["source"]
    else:
        repository = ""
        repository_source = ""
    if repository and repository_source == "hub":
        repository_matches = tuple(
            profile
            for profile in matching
            if repository
            in {profile.default_repo, profile.fallback_repo, *profile.compatible_repos}
        )
        if len(repository_matches) == 1:
            return repository_matches, None

    return matching, "loader_profile_ambiguous"


def public_experimental_pipelines(
    *,
    observe_optional_runtime: bool = False,
    optional_runtime_catalog_resolver=None,
) -> list[dict]:
    parameter_aliases = {
        "modelRepository": ["model_id", "model", "repo"],
        "guidanceScale": ["guidance_scale", "true_cfg_scale", "guidance"],
        "steps": ["num_inference_steps", "steps"],
        "sourceImage": ["image", "reference_images"],
        "maskImage": ["mask_image", "mask"],
        "controlImage": ["control_image", "conditioning_image"],
    }
    public_pipelines = []
    for pipeline in EXPERIMENTAL_DIFFUSERS_PIPELINES:
        profile_ids = pipeline.get("executionProfileIds", [])
        execution_profiles = [
            DIFFUSERS_EXECUTION_PROFILES[profile_id].to_public_dict(
                observe_optional_runtime=observe_optional_runtime,
                optional_runtime_catalog_resolver=optional_runtime_catalog_resolver,
            )
            for profile_id in profile_ids
            if profile_id in DIFFUSERS_EXECUTION_PROFILES
        ]
        if len(execution_profiles) != len(profile_ids):
            # A dangling profile reference must never leave a mode looking
            # runnable. Tests make this branch a permanent registry invariant.
            execution_profiles = []
            runnable_modes = []
            pipeline_classes = []
            backend_path = None
            qualification_status = "invalid_contract"
        elif execution_profiles:
            runnable_modes = list(
                dict.fromkeys(mode for profile in execution_profiles for mode in profile["modes"])
            )
            pipeline_classes = list(
                dict.fromkeys(profile["pipeline_class"] for profile in execution_profiles)
            )
            backend_paths = {profile["backend_path"] for profile in execution_profiles}
            backend_path = next(iter(backend_paths)) if len(backend_paths) == 1 else None
            qualification_status = pipeline.get("qualificationStatus", "unqualified")
        else:
            runnable_modes = list(pipeline["runnableModes"])
            pipeline_classes = list(pipeline["pipelineClasses"])
            backend_path = pipeline.get("backendPath")
            qualification_status = pipeline.get("qualificationStatus", "unqualified")

        optional_runtime_profile_ids = list(
            dict.fromkeys(
                profile_id
                for profile in execution_profiles
                for profile_id in profile.get("optional_runtime_profiles", [])
            )
        )
        if not optional_runtime_profile_ids:
            optional_runtime_profile_ids = list(
                pipeline["optionalRuntimeProfileIds"]
                if "optionalRuntimeProfileIds" in pipeline
                else (TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID,)
            )
        selected_profile_contracts = tuple(
            DIFFUSERS_EXECUTION_PROFILES[profile_id]
            for profile_id in profile_ids
            if profile_id in DIFFUSERS_EXECUTION_PROFILES
        )
        if observe_optional_runtime and selected_profile_contracts:
            from modiff.optional_runtime_execution import (
                optional_runtime_requirement_for_profiles as observed_requirement,
            )

            optional_runtime_requirement = observed_requirement(
                selected_profile_contracts,
                catalog_resolver=optional_runtime_catalog_resolver,
            )
        else:
            optional_runtime_requirement = optional_runtime_requirement_for_profiles(
                selected_profile_contracts
            )

        public_pipelines.append(
            {
                **pipeline,
                "pipelineClasses": pipeline_classes,
                "backendPath": backend_path,
                "runnableModes": runnable_modes,
                "schemaVersion": 2,
                "supportTier": "experimental",
                "executionProfiles": execution_profiles,
                "inputContracts": dict(pipeline.get("inputContracts", {})),
                "unsupportedModes": {
                    mode: dict(contract) for mode, contract in pipeline.get("unsupportedModes", {}).items()
                },
                "parameterAliases": parameter_aliases,
                "defaults": dict(pipeline.get("defaults", {})),
                "artifactCandidates": [pipeline["defaultRepo"]] if pipeline.get("defaultRepo") else [],
                "revisionCandidates": list(pipeline.get("revisionCandidates", [])),
                "quantizationSupport": dict(
                    pipeline.get(
                        "quantizationSupport",
                        {"defaultMode": "none", "components": [], "offloadModes": []},
                    )
                ),
                "qualificationStatus": qualification_status,
                "optionalRuntimeProfileIds": optional_runtime_profile_ids,
                "optionalRuntimeProfiles": public_optional_runtime_profiles(
                    optional_runtime_profile_ids
                ),
                **(
                    {
                        "optionalRuntimeRequirement": optional_runtime_requirement
                    }
                    if execution_profiles
                    else {}
                ),
            }
        )
    return public_pipelines


def public_execution_profiles(
    *,
    observe_optional_runtime: bool = False,
    optional_runtime_catalog_resolver=None,
) -> list[dict]:
    return [
        profile.to_public_dict(
            observe_optional_runtime=observe_optional_runtime,
            optional_runtime_catalog_resolver=optional_runtime_catalog_resolver,
        )
        for profile in DIFFUSERS_EXECUTION_PROFILES.values()
        if profile.public
    ]


def optional_runtime_profile_ids_for_execution(
    model_type: str,
    mode: str | None = None,
) -> tuple[str, ...]:
    """Resolve optional runtime IDs for one declared model/mode pair."""

    profile_ids: list[str] = []
    for profile in execution_profiles_for_execution(model_type, mode):
        for profile_id in profile.optional_runtime_profiles:
            if profile_id not in profile_ids:
                profile_ids.append(profile_id)
    return tuple(profile_ids)
