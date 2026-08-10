"""Declarative contracts for workflow-specific optional model runtimes.

This module is intentionally Python-standard-library only.  Registry discovery,
template browsing, and Auto planning may inspect these contracts and local
distribution metadata, but must never import, install, or activate the declared
packages.  Installation and activation remain a later, explicit P0.5 slice.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
import sys
from types import MappingProxyType
from typing import Callable, Iterable, Mapping


OPTIONAL_RUNTIME_SCHEMA_VERSION = 1
TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID = "huggingface-transformers-peft-5.14.1-0.20.0"
_MAX_OBSERVED_VERSION_LENGTH = 128


@dataclass(frozen=True)
class OptionalRuntimePackageContract:
    """One exact distribution in a reviewed optional-runtime specification."""

    distribution: str
    import_name: str
    version: str
    publisher: str
    project_url: str
    distribution_url: str
    license: str
    role: str
    required_symbols: tuple[str, ...]
    required_class_symbols: tuple[str, ...] = ()

    @property
    def requirement(self) -> str:
        return f"{self.distribution}=={self.version}"

    def to_spec_dict(self) -> dict:
        return {
            "distribution": self.distribution,
            "importName": self.import_name,
            "requiredVersion": self.version,
            "requirement": self.requirement,
            # Provenance and license are reviewed contract declarations, not
            # claims derived from mutable installed-distribution metadata.
            "publisher": self.publisher,
            "projectUrl": self.project_url,
            "distributionUrl": self.distribution_url,
            "license": self.license,
            "role": self.role,
            "requiredSymbols": list(self.required_symbols),
            "requiredClassSymbols": list(self.required_class_symbols),
        }


@dataclass(frozen=True)
class OptionalRuntimeBaseContract:
    """One dependency deliberately supplied by the verified base profile."""

    distribution: str
    import_name: str
    specifier: str = ""
    platforms: tuple[str, ...] = ()

    def to_spec_dict(self) -> dict:
        value = {
            "distribution": self.distribution,
            "importName": self.import_name,
            "specifier": self.specifier,
        }
        if self.platforms:
            value["platforms"] = list(self.platforms)
        return value


@dataclass(frozen=True)
class OptionalRuntimeProfile:
    """An exact composite runtime contract that is not yet executable."""

    id: str
    label: str
    packages: tuple[OptionalRuntimePackageContract, ...]
    base_packages: tuple[OptionalRuntimeBaseContract, ...]
    required_diffusers_symbols: tuple[str, ...] = ()
    require_peft_backend: bool = False
    artifact_locks: tuple[dict, ...] = ()
    pipeline_adapter_symbols: tuple[str, ...] = ()
    excluded_qualification_symbols: tuple[str, ...] = ()
    pipeline_adapter_methods: tuple[tuple[str, tuple[str, ...]], ...] = ()
    contract_state: str = "candidate_unqualified"
    install_policy: str = "explicit_first_use"
    cutover_ready: bool = False
    install_action_available: bool = False
    activation_available: bool = False

    def to_spec_dict(self) -> dict:
        """Return immutable fields used to identify the exact reviewed spec."""

        return {
            "schemaVersion": OPTIONAL_RUNTIME_SCHEMA_VERSION,
            "id": self.id,
            "label": self.label,
            "contractState": self.contract_state,
            "cutoverReady": self.cutover_ready,
            "installActionAvailable": self.install_action_available,
            "activationAvailable": self.activation_available,
            "installPolicy": self.install_policy,
            "packages": [
                package.to_spec_dict() for package in self.packages if package.role == "runtime_root"
            ],
            "stagedPackages": [package.to_spec_dict() for package in self.packages],
            "baseRequirements": [package.to_spec_dict() for package in self.base_packages],
            "requiredDiffusersSymbols": list(self.required_diffusers_symbols),
            "requirePeftBackend": self.require_peft_backend,
            "artifactLocks": [dict(artifact) for artifact in self.artifact_locks],
            "pipelineAdapterSymbols": list(self.pipeline_adapter_symbols),
            "excludedQualificationSymbols": list(self.excluded_qualification_symbols),
            "pipelineAdapterMethods": [
                {"method": method, "requiredParameters": list(parameters)}
                for method, parameters in self.pipeline_adapter_methods
            ],
        }

    @property
    def spec_digest(self) -> str:
        canonical = json.dumps(
            self.to_spec_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


_TRANSFORMERS_PEFT_PROFILE = OptionalRuntimeProfile(
    id=TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID,
    label="Hugging Face Transformers + PEFT",
    packages=(
        OptionalRuntimePackageContract(
            distribution="transformers",
            import_name="transformers",
            version="5.14.1",
            publisher="Hugging Face",
            project_url="https://github.com/huggingface/transformers",
            distribution_url="https://pypi.org/project/transformers/",
            license="Apache-2.0",
            role="runtime_root",
            required_symbols=(
                "Qwen2_5_VLForConditionalGeneration",
                "Qwen2Tokenizer",
                "Qwen2VLProcessor",
                "AutoTokenizer",
                "CLIPImageProcessor",
                "CLIPVisionModel",
                "UMT5EncoderModel",
                "T5EncoderModel",
                "T5TokenizerFast",
                "PreTrainedModel",
                "PreTrainedTokenizerFast",
                "CLIPTextModel",
                "CLIPTextModelWithProjection",
                "CLIPTokenizer",
                "CLIPVisionModelWithProjection",
                "SiglipImageProcessor",
                "SiglipVisionModel",
                "Qwen2TokenizerFast",
                "Qwen3ForCausalLM",
                "AutoProcessor",
                "Mistral3ForConditionalGeneration",
                "BitsAndBytesConfig",
                "transformers.core_model_loading:tqdm",
                "transformers.utils.logging:tqdm",
                "transformers.utils.logging:get_verbosity",
                "transformers.utils.logging:set_verbosity",
            ),
            required_class_symbols=(
                "Qwen2_5_VLForConditionalGeneration",
                "Qwen2Tokenizer",
                "Qwen2VLProcessor",
                "AutoTokenizer",
                "CLIPImageProcessor",
                "CLIPVisionModel",
                "UMT5EncoderModel",
                "T5EncoderModel",
                "T5TokenizerFast",
                "PreTrainedModel",
                "PreTrainedTokenizerFast",
                "CLIPTextModel",
                "CLIPTextModelWithProjection",
                "CLIPTokenizer",
                "CLIPVisionModelWithProjection",
                "SiglipImageProcessor",
                "SiglipVisionModel",
                "Qwen2TokenizerFast",
                "Qwen3ForCausalLM",
                "AutoProcessor",
                "Mistral3ForConditionalGeneration",
                "BitsAndBytesConfig",
            ),
        ),
        OptionalRuntimePackageContract(
            distribution="peft",
            import_name="peft",
            version="0.20.0",
            publisher="Hugging Face",
            project_url="https://github.com/huggingface/peft",
            distribution_url="https://pypi.org/project/peft/",
            license="Apache-2.0",
            role="runtime_root",
            required_symbols=(
                "LoraConfig",
                "PeftConfig",
                "PeftModel",
                "inject_adapter_in_model",
                "set_peft_model_state_dict",
                "tuners.tuners_utils.BaseTunerLayer",
                "tuners.lora.layer.LoraLayer",
                "utils.other._get_submodules",
                "utils.save_and_load.get_peft_model_state_dict",
                "peft.utils.hotswap:check_hotswap_configs_compatible",
                "peft.utils.hotswap:hotswap_adapter_from_state_dict",
                "peft.utils.hotswap:prepare_model_for_compiled_hotswap",
            ),
            required_class_symbols=(
                "LoraConfig",
                "PeftConfig",
                "PeftModel",
                "tuners.tuners_utils.BaseTunerLayer",
                "tuners.lora.layer.LoraLayer",
            ),
        ),
        OptionalRuntimePackageContract(
            distribution="tokenizers",
            import_name="tokenizers",
            version="0.22.2",
            publisher="Hugging Face",
            project_url="https://github.com/huggingface/tokenizers",
            distribution_url="https://pypi.org/project/tokenizers/",
            license="Apache-2.0",
            role="runtime_dependency",
            required_symbols=("Tokenizer", "tokenizers.tokenizers:Tokenizer"),
            required_class_symbols=("Tokenizer", "tokenizers.tokenizers:Tokenizer"),
        ),
        OptionalRuntimePackageContract(
            distribution="typer",
            import_name="typer",
            version="0.27.1",
            publisher="FastAPI",
            project_url="https://github.com/fastapi/typer",
            distribution_url="https://pypi.org/project/typer/",
            license="MIT",
            role="runtime_dependency",
            required_symbols=("Typer",),
            required_class_symbols=("Typer",),
        ),
        OptionalRuntimePackageContract(
            distribution="annotated-doc",
            import_name="annotated_doc",
            version="0.0.5",
            publisher="FastAPI",
            project_url="https://github.com/fastapi/annotated-doc",
            distribution_url="https://pypi.org/project/annotated-doc/",
            license="MIT",
            role="runtime_dependency",
            required_symbols=("Doc",),
            required_class_symbols=("Doc",),
        ),
        OptionalRuntimePackageContract(
            distribution="rich",
            import_name="rich",
            version="15.0.0",
            publisher="Textualize",
            project_url="https://github.com/Textualize/rich",
            distribution_url="https://pypi.org/project/rich/",
            license="MIT",
            role="runtime_dependency",
            required_symbols=("print",),
        ),
        OptionalRuntimePackageContract(
            distribution="markdown-it-py",
            import_name="markdown_it",
            version="4.2.0",
            publisher="Executable Books",
            project_url="https://github.com/executablebooks/markdown-it-py",
            distribution_url="https://pypi.org/project/markdown-it-py/",
            license="MIT",
            role="runtime_dependency",
            required_symbols=("MarkdownIt",),
            required_class_symbols=("MarkdownIt",),
        ),
        OptionalRuntimePackageContract(
            distribution="mdurl",
            import_name="mdurl",
            version="0.1.2",
            publisher="Executable Books",
            project_url="https://github.com/executablebooks/mdurl",
            distribution_url="https://pypi.org/project/mdurl/",
            license="MIT",
            role="runtime_dependency",
            required_symbols=("encode",),
        ),
        OptionalRuntimePackageContract(
            distribution="pygments",
            import_name="pygments",
            version="2.20.0",
            publisher="Pygments",
            project_url="https://github.com/pygments/pygments",
            distribution_url="https://pypi.org/project/Pygments/",
            license="BSD-2-Clause",
            role="runtime_dependency",
            required_symbols=("highlight",),
        ),
        OptionalRuntimePackageContract(
            distribution="shellingham",
            import_name="shellingham",
            version="1.5.4",
            publisher="Sarugaku",
            project_url="https://github.com/sarugaku/shellingham",
            distribution_url="https://pypi.org/project/shellingham/",
            license="ISC",
            role="runtime_dependency",
            required_symbols=("detect_shell",),
        ),
    ),
    # These transitive dependencies remain owned by the reviewed base profile.
    # Their exact observed versions and metadata origins are frozen into each
    # validation binding; the accelerator lock digest alone is not sufficient.
    base_packages=tuple(
        OptionalRuntimeBaseContract(
            distribution=distribution,
            import_name=import_name,
            specifier=specifier,
        )
        for distribution, import_name, specifier in (
            ("accelerate", "accelerate", ">=0.21.0"),
            ("anyio", "anyio", ""),
            ("certifi", "certifi", ""),
            ("click", "click", ""),
            ("filelock", "filelock", ""),
            ("fsspec", "fsspec", ""),
            ("h11", "h11", ""),
            ("hf-xet", "hf_xet", ""),
            ("httpcore", "httpcore", ""),
            ("httpx", "httpx", ""),
            ("huggingface-hub", "huggingface_hub", ">=1.5.0,<2.0"),
            ("idna", "idna", ""),
            ("jinja2", "jinja2", ""),
            ("markupsafe", "markupsafe", ""),
            ("mpmath", "mpmath", ""),
            ("networkx", "networkx", ""),
            ("numpy", "numpy", ">=1.17"),
            ("packaging", "packaging", ">=20.0"),
            ("pillow", "PIL", ">=10.0.1,<=15.0"),
            ("protobuf", "google.protobuf", ">=6.31.1"),
            ("psutil", "psutil", ""),
            ("pyyaml", "yaml", ">=5.1"),
            ("regex", "regex", ">=2025.10.22"),
            ("safetensors", "safetensors", ">=0.8.0"),
            ("sentencepiece", "sentencepiece", ">=0.2.0"),
            ("setuptools", "setuptools", ""),
            ("sympy", "sympy", ""),
            ("torch", "torch", ">=2.6.0"),
            ("torchvision", "torchvision", ">=0.21.0"),
            ("tqdm", "tqdm", ">=4.60"),
            ("typing-extensions", "typing_extensions", ""),
        )
    )
    + (
        OptionalRuntimeBaseContract(
            distribution="colorama",
            import_name="colorama",
            specifier="",
            platforms=("windows",),
        ),
    ),
    required_diffusers_symbols=(
        "ComponentSpec",
        "ComponentsManager",
        "ModularPipeline",
        "AutoencoderKLWan",
        "BitsAndBytesConfig",
        "QwenImagePipeline",
        "QwenImageEditInpaintPipeline",
        "QwenImageModularPipeline",
        "QwenImageEditModularPipeline",
        "QwenImageEditPlusModularPipeline",
        "QwenImageLayeredModularPipeline",
        "WanVACEPipeline",
        "WanVideoToVideoPipeline",
        "WanPipeline",
        "WanImageToVideoPipeline",
        "WanModularPipeline",
        "WanImage2VideoModularPipeline",
        "LTXConditionPipeline",
        "AceStepPipeline",
        "StableDiffusionXLModularPipeline",
        "StableDiffusionXLPipeline",
        "FluxPipeline",
        "Flux2KleinPipeline",
        "FluxKontextPipeline",
        "FluxFillPipeline",
        "FluxControlPipeline",
        "FluxPriorReduxPipeline",
        "FluxModularPipeline",
        "FluxKontextModularPipeline",
        "Flux2KleinModularPipeline",
    ),
    require_peft_backend=True,
    # Candidate metadata deliberately has no executable wheel/hash lock yet.
    # Even an accidental flag flip must not make version-only acquisition run.
    artifact_locks=(),
    pipeline_adapter_symbols=(
        "AceStepPipeline",
        "FluxPipeline",
        "Flux2KleinPipeline",
        "LTXConditionPipeline",
        "QwenImagePipeline",
        "StableDiffusionXLPipeline",
        "WanPipeline",
    ),
    excluded_qualification_symbols=("add_weighted_adapter",),
    pipeline_adapter_methods=(
        ("load_lora_weights", ("pretrained_model_name_or_path_or_dict", "adapter_name", "hotswap")),
        ("set_adapters", ("adapter_names", "adapter_weights")),
        ("delete_adapters", ("adapter_names",)),
        ("fuse_lora", ("components", "lora_scale", "safe_fusing", "adapter_names")),
        ("unfuse_lora", ("components",)),
        ("unload_lora_weights", ()),
        ("get_list_adapters", ()),
        ("enable_lora_hotswap", ("kwargs",)),
    ),
)

OPTIONAL_RUNTIME_PROFILES: Mapping[str, OptionalRuntimeProfile] = MappingProxyType(
    {_TRANSFORMERS_PEFT_PROFILE.id: _TRANSFORMERS_PEFT_PROFILE}
)


def _selected_profiles(profile_ids: Iterable[str] | None) -> tuple[OptionalRuntimeProfile, ...]:
    requested_ids = tuple(OPTIONAL_RUNTIME_PROFILES) if profile_ids is None else tuple(profile_ids)
    selected: list[OptionalRuntimeProfile] = []
    seen: set[str] = set()
    for raw_profile_id in requested_ids:
        profile_id = str(raw_profile_id or "").strip()
        if profile_id in seen:
            continue
        try:
            profile = OPTIONAL_RUNTIME_PROFILES[profile_id]
        except KeyError as exc:
            raise ValueError(f"Unknown optional runtime profile {profile_id!r}.") from exc
        seen.add(profile_id)
        selected.append(profile)
    return tuple(selected)


def _bounded_observed_version(value: object) -> str:
    """Return bounded printable metadata text suitable for a public response."""

    rendered = str(value).strip()
    rendered = "".join(character if character.isprintable() else "?" for character in rendered)
    if len(rendered) > _MAX_OBSERVED_VERSION_LENGTH:
        rendered = rendered[: _MAX_OBSERVED_VERSION_LENGTH - 3] + "..."
    return rendered


def _package_status(
    package: OptionalRuntimePackageContract,
    *,
    version_resolver: Callable[[str], str],
) -> dict:
    public = package.to_spec_dict()
    try:
        installed_version = _bounded_observed_version(version_resolver(package.distribution))
    except metadata.PackageNotFoundError:
        return {**public, "status": "missing"}
    except (ValueError, TypeError, OSError):
        # Corrupt or unreadable local distribution metadata cannot establish an
        # exact match.  Classify it with the same fail-closed readiness as a
        # version mismatch without reflecting untrusted exception text.
        return {
            **public,
            "status": "wrong_version",
            "metadataState": "unreadable",
        }

    return {
        **public,
        "status": "present_unqualified" if installed_version == package.version else "wrong_version",
        "installedVersion": installed_version,
    }


def public_optional_runtime_profiles(
    profile_ids: Iterable[str] | None = None,
    *,
    version_resolver: Callable[[str], str] | None = None,
) -> list[dict]:
    """Publish exact contracts plus local metadata-only presence observations.

    Exact package presence is intentionally reported as ``present_unqualified``:
    this slice neither validates an overlay origin nor makes the runtime ready
    for optional-runtime cutover.
    """

    resolve_version = version_resolver or metadata.version
    public_profiles = []
    for profile in _selected_profiles(profile_ids):
        root_packages = [package for package in profile.packages if package.role == "runtime_root"]
        package_statuses = [
            _package_status(package, version_resolver=resolve_version)
            for package in root_packages
        ]
        package_states = {package["status"] for package in package_statuses}
        if "missing" in package_states:
            status = "missing"
        elif "wrong_version" in package_states:
            status = "wrong_version"
        else:
            status = "present_unqualified"

        public_profiles.append(
            {
                **profile.to_spec_dict(),
                "specDigest": profile.spec_digest,
                "status": status,
                "requirements": [package.requirement for package in root_packages],
                "stagedRequirements": [package.requirement for package in profile.packages],
                "packages": package_statuses,
            }
        )
    return public_profiles


def optional_runtime_requirements(profile_ids: Iterable[str]) -> tuple[str, ...]:
    """Resolve exact requirements without consulting the host environment."""

    requirements: list[str] = []
    for profile in _selected_profiles(profile_ids):
        for package in profile.packages:
            if package.requirement not in requirements:
                requirements.append(package.requirement)
    return tuple(requirements)


def optional_runtime_base_distributions(
    profile_ids: Iterable[str],
    *,
    platform_name: str | None = None,
) -> tuple[str, ...]:
    """Resolve base-owned dependency names for the current platform."""

    selected_platform = platform_name or (
        "windows" if sys.platform.startswith("win") else "macos" if sys.platform == "darwin" else "linux"
    )
    distributions: list[str] = []
    for profile in _selected_profiles(profile_ids):
        for package in profile.base_packages:
            if package.platforms and selected_platform not in package.platforms:
                continue
            if package.distribution not in distributions:
                distributions.append(package.distribution)
    return tuple(distributions)


def optional_runtime_base_contracts(
    profile_ids: Iterable[str],
    *,
    platform_name: str | None = None,
) -> tuple[dict, ...]:
    """Resolve executable base-owned contracts for fresh-process validation."""

    selected_platform = platform_name or (
        "windows" if sys.platform.startswith("win") else "macos" if sys.platform == "darwin" else "linux"
    )
    contracts: list[dict] = []
    seen: set[str] = set()
    for profile in _selected_profiles(profile_ids):
        for package in profile.base_packages:
            if package.platforms and selected_platform not in package.platforms:
                continue
            if package.distribution in seen:
                continue
            seen.add(package.distribution)
            contracts.append(package.to_spec_dict())
    return tuple(contracts)
