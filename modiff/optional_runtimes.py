"""Declarative contracts for workflow-specific optional model runtimes.

This module is intentionally Python-standard-library only.  Registry discovery,
template browsing, and Auto planning may inspect these contracts and local
distribution metadata, but must never import, install, or activate the declared
packages.  Installation and activation remain a later, explicit P0.5 slice.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
from importlib import metadata
import json
import platform
import sys
from types import MappingProxyType
from typing import Callable, Iterable, Mapping


OPTIONAL_RUNTIME_SCHEMA_VERSION = 1
TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID = "huggingface-transformers-peft-5.14.1-0.20.0"
TRANSFORMERS_MAIN_PEFT_RUNTIME_PROFILE_ID = (
    "huggingface-transformers-main-96fe6dce-peft-0.20.0"
)
TRANSFORMERS_MAIN_COMMIT = "96fe6dce36cc929a5ffd3e34296554c4cb6b669e"
TRANSFORMERS_MAIN_REVIEW_BASE_COMMIT = "a597f974857b3d92939971296bc0deb93d33d780"
TRANSFORMERS_MAIN_REVIEWED_DELTA_PATHS = (
    "tests/models/axk1/test_modeling_axk1.py",
    "tests/models/gemma/test_modeling_gemma.py",
)
_MAX_OBSERVED_VERSION_LENGTH = 128

_OPTIONAL_RUNTIME_TARGETS = (
    ("linux", "cp312", "x86_64"),
    ("linux", "cp312", "arm64"),
    ("macos", "cp312", "x86_64"),
    ("macos", "cp312", "arm64"),
    ("windows", "cp312", "x86_64"),
    ("windows", "cp312", "arm64"),
)


def optional_runtime_target(
    *,
    platform_name: str | None = None,
    machine: str | None = None,
) -> tuple[str, str]:
    """Return one normalized runtime target without importing optional packages."""

    selected_platform = str(platform_name or "").strip().lower()
    if not selected_platform:
        selected_platform = (
            "windows"
            if sys.platform.startswith("win")
            else "macos"
            if sys.platform == "darwin"
            else "linux"
        )
    selected_machine = str(machine or platform.machine()).strip().lower()
    normalized_machine = {
        "amd64": "x86_64",
        "x64": "x86_64",
        "aarch64": "arm64",
    }.get(selected_machine, selected_machine)
    return selected_platform, normalized_machine
_PURE_RUNTIME_WHEELS = (
    ("transformers", "5.14.1", "transformers-5.14.1-py3-none-any.whl", "https://files.pythonhosted.org/packages/6f/67/8d85ca2323233ae3c0365a659c4e52ee1f587b440e4bc577e7d8e4416d0f/transformers-5.14.1-py3-none-any.whl", "9db974c4079ede2d1a3ea7ca5a240df33f2cc26fc2b36ba64c5f2a4f43b6e725", 11625234),
    ("peft", "0.20.0", "peft-0.20.0-py3-none-any.whl", "https://files.pythonhosted.org/packages/28/79/13bcabb8048126422d5c4b880575d40886c726f354db88cfeed4325525bb/peft-0.20.0-py3-none-any.whl", "0fbba16ffebfad3de96e06f2da6860fd860292324b85b6141909fa1e26ea9233", 775777),
    ("typer", "0.27.1", "typer-0.27.1-py3-none-any.whl", "https://files.pythonhosted.org/packages/43/89/9518bc0c3929bee36b3a4a8e3daddd6e03f92f9961c66d4983b837160543/typer-0.27.1-py3-none-any.whl", "53150287edd11baeb4e4722c8e394fcdf8181c0ae89485cba8d25c778d5edd56", 122874),
    ("annotated-doc", "0.0.5", "annotated_doc-0.0.5-py3-none-any.whl", "https://files.pythonhosted.org/packages/3e/30/e900b21425a860e195f32e37657aa1f7c7f2b1bfb26f03ca209b90933c06/annotated_doc-0.0.5-py3-none-any.whl", "117bac03a25ede5df5440e855b32d556049ca169ead221505badf432fed4b101", 5302),
    ("rich", "15.0.0", "rich-15.0.0-py3-none-any.whl", "https://files.pythonhosted.org/packages/82/3b/64d4899d73f91ba49a8c18a8ff3f0ea8f1c1d75481760df8c68ef5235bf5/rich-15.0.0-py3-none-any.whl", "33bd4ef74232fb73fe9279a257718407f169c09b78a87ad3d296f548e27de0bb", 310654),
    ("markdown-it-py", "4.2.0", "markdown_it_py-4.2.0-py3-none-any.whl", "https://files.pythonhosted.org/packages/b3/81/4da04ced5a082363ecfa159c010d200ecbd959ae410c10c0264a38cac0f5/markdown_it_py-4.2.0-py3-none-any.whl", "9f7ebbcd14fe59494226453aed97c1070d83f8d24b6fc3a3bcf9a38092641c4a", 91687),
    ("mdurl", "0.1.2", "mdurl-0.1.2-py3-none-any.whl", "https://files.pythonhosted.org/packages/b3/38/89ba8ad64ae25be8de66a6d463314cf1eb366222074cfda9ee839c56a4b4/mdurl-0.1.2-py3-none-any.whl", "84008a41e51615a49fc9966191ff91509e3c40b939176e643fd50a5c2196b8f8", 9979),
    ("pygments", "2.20.0", "pygments-2.20.0-py3-none-any.whl", "https://files.pythonhosted.org/packages/f4/7e/a72dd26f3b0f4f2bf1dd8923c85f7ceb43172af56d63c7383eb62b332364/pygments-2.20.0-py3-none-any.whl", "81a9e26dd42fd28a23a2d169d86d7ac03b46e2f8b59ed4698fb4785f946d0176", 1231151),
    ("shellingham", "1.5.4", "shellingham-1.5.4-py2.py3-none-any.whl", "https://files.pythonhosted.org/packages/e0/f9/0595336914c5619e5f28a1fb793285925a8cd4b432c9da0a987836c7f822/shellingham-1.5.4-py2.py3-none-any.whl", "7ecfff8f2fd72616f7481040475a65b2bf8af90a56c89140852d1120324e8686", 9755),
)
_TOKENIZERS_RUNTIME_WHEELS = {
    ("linux", "x86_64"): ("tokenizers-0.22.2-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl", "https://files.pythonhosted.org/packages/2e/76/932be4b50ef6ccedf9d3c6639b056a967a86258c6d9200643f01269211ca/tokenizers-0.22.2-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl", "369cc9fc8cc10cb24143873a0d95438bb8ee257bb80c71989e3ee290e8d72c67", 3274982),
    ("linux", "arm64"): ("tokenizers-0.22.2-cp39-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl", "https://files.pythonhosted.org/packages/d6/84/7990e799f1309a8b87af6b948f31edaa12a3ed22d11b352eaf4f4b2e5753/tokenizers-0.22.2-cp39-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl", "2249487018adec45d6e3554c71d46eb39fa8ea67156c640f7513eb26f318cec7", 3290736),
    ("macos", "x86_64"): ("tokenizers-0.22.2-cp39-abi3-macosx_10_12_x86_64.whl", "https://files.pythonhosted.org/packages/92/97/5dbfabf04c7e348e655e907ed27913e03db0923abb5dfdd120d7b25630e1/tokenizers-0.22.2-cp39-abi3-macosx_10_12_x86_64.whl", "544dd704ae7238755d790de45ba8da072e9af3eea688f698b137915ae959281c", 3100275),
    ("macos", "arm64"): ("tokenizers-0.22.2-cp39-abi3-macosx_11_0_arm64.whl", "https://files.pythonhosted.org/packages/2e/47/174dca0502ef88b28f1c9e06b73ce33500eedfac7a7692108aec220464e7/tokenizers-0.22.2-cp39-abi3-macosx_11_0_arm64.whl", "1e418a55456beedca4621dbab65a318981467a2b188e982a23e117f115ce5001", 2981472),
    ("windows", "x86_64"): ("tokenizers-0.22.2-cp39-abi3-win_amd64.whl", "https://files.pythonhosted.org/packages/65/71/0670843133a43d43070abeb1949abfdef12a86d490bea9cd9e18e37c5ff7/tokenizers-0.22.2-cp39-abi3-win_amd64.whl", "c9ea31edff2968b44a88f97d784c2f16dc0729b8b143ed004699ebca91f05c48", 2747786),
    ("windows", "arm64"): ("tokenizers-0.22.2-cp39-abi3-win_arm64.whl", "https://files.pythonhosted.org/packages/72/f4/0de46cfa12cdcbcd464cc59fde36912af405696f687e53a091fb432f694c/tokenizers-0.22.2-cp39-abi3-win_arm64.whl", "9ce725d22864a1e965217204946f830c37876eee3b2ba6fc6255e8e903d5fcbc", 2612133),
}


def _transformers_peft_artifact_locks() -> tuple[dict, ...]:
    locks = []
    for platform_name, python_tag, machine in _OPTIONAL_RUNTIME_TARGETS:
        tokenizers = _TOKENIZERS_RUNTIME_WHEELS[(platform_name, machine)]
        wheels = (
            _PURE_RUNTIME_WHEELS[0],
            _PURE_RUNTIME_WHEELS[1],
            ("tokenizers", "0.22.2", *tokenizers),
            *_PURE_RUNTIME_WHEELS[2:],
        )
        for distribution, version, filename, url, sha256, byte_size in wheels:
            locks.append(
                {
                    "distribution": distribution,
                    "version": version,
                    "filename": filename,
                    "url": url,
                    "sha256": sha256,
                    "byteSize": byte_size,
                    "platform": platform_name,
                    "pythonTag": python_tag,
                    "machine": machine,
                }
            )
    return tuple(locks)


def _transformers_main_source_build() -> dict:
    return {
        "schemaVersion": 1,
        "distribution": "transformers",
        "version": "5.16.0.dev0",
        "sourceArtifact": {
            "kind": "github_commit_tarball",
            "repository": "huggingface/transformers",
            "commit": TRANSFORMERS_MAIN_COMMIT,
            "archiveRoot": f"transformers-{TRANSFORMERS_MAIN_COMMIT}",
            "filename": f"transformers-{TRANSFORMERS_MAIN_COMMIT}.tar.gz",
            "url": (
                "https://codeload.github.com/huggingface/transformers/tar.gz/"
                f"{TRANSFORMERS_MAIN_COMMIT}"
            ),
            "sha256": "e9903aec337657fd8ae1fd1e7812efed159c2cf4444e83e7fc877e252127e1b3",
            "byteSize": 20_532_315,
        },
        "recipe": "modiff_pure_python_wheel_v1",
        "pythonTag": "py3",
        "sourceDateEpoch": 315_532_800,
        "sourceFiles": ["LICENSE", "README.md", "pyproject.toml", "setup.py"],
        "sourceTrees": ["src/transformers"],
        "buildDependencies": [],
        "wheelMetadata": {
            "summary": "Transformers: the model-definition framework for state-of-the-art machine learning models in text, vision, audio, and multimodal models, for both inference and training.",
            "license": "Apache 2.0 License",
            "requiresPython": ">=3.10.0",
            "requiresDist": [
                "huggingface-hub>=1.5.0,<2.0",
                "numpy>=1.17",
                "packaging>=20.0",
                "pyyaml>=5.1",
                "regex>=2025.10.22",
                "tokenizers>=0.22.0,<=0.23.0",
                "typer",
                "safetensors>=0.8.0",
                "tqdm>=4.60",
            ],
            "consoleScripts": {
                "transformers": "transformers.cli.transformers:main",
            },
            "packageRoot": "src",
            "licenseFile": "LICENSE",
        },
        "outputWheel": {
            "distribution": "transformers",
            "version": "5.16.0.dev0",
            "filename": "transformers-5.16.0.dev0-py3-none-any.whl",
            "sha256": "8a439d25595c6dde486cfbd5a6ed8158e0fe7554ec236491668425e11952898f",
            "byteSize": 52_395_984,
            "platform": "any",
            "pythonTag": "py3",
            "machine": "any",
        },
    }


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
class OptionalRuntimeTargetContract:
    """Qualification and action policy for one exact OS/architecture target."""

    platform: str
    machine: str
    contract_state: str
    cutover_ready: bool
    install_action_available: bool
    activation_available: bool

    def __post_init__(self) -> None:
        if (
            (self.platform, "cp312", self.machine) not in _OPTIONAL_RUNTIME_TARGETS
            or self.contract_state not in {"qualified", "candidate_unqualified"}
            or any(
                type(value) is not bool
                for value in (
                    self.cutover_ready,
                    self.install_action_available,
                    self.activation_available,
                )
            )
            or len(
                {
                    self.cutover_ready,
                    self.install_action_available,
                    self.activation_available,
                }
            )
            != 1
            or (self.contract_state == "qualified") != self.cutover_ready
        ):
            raise ValueError("Invalid optional-runtime target contract.")

    def to_spec_dict(self) -> dict:
        return {
            "platform": self.platform,
            "machine": self.machine,
            "contractState": self.contract_state,
            "cutoverReady": self.cutover_ready,
            "installActionAvailable": self.install_action_available,
            "activationAvailable": self.activation_available,
        }


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
    source_builds: tuple[dict, ...] = ()
    pipeline_adapter_symbols: tuple[str, ...] = ()
    excluded_qualification_symbols: tuple[str, ...] = ()
    pipeline_adapter_methods: tuple[tuple[str, tuple[str, ...]], ...] = ()
    contract_state: str = "candidate_unqualified"
    install_policy: str = "explicit_first_use"
    cutover_ready: bool = False
    install_action_available: bool = False
    activation_available: bool = False
    target_contracts: tuple[OptionalRuntimeTargetContract, ...] = ()

    def __post_init__(self) -> None:
        targets = tuple((contract.platform, contract.machine) for contract in self.target_contracts)
        if len(set(targets)) != len(targets):
            raise ValueError("Optional-runtime target contracts must be unique.")

    def contract_for_target(
        self,
        *,
        platform_name: str | None = None,
        machine: str | None = None,
    ) -> OptionalRuntimeTargetContract:
        """Resolve effective actions for one explicit target, failing closed."""

        selected_platform, selected_machine = optional_runtime_target(
            platform_name=platform_name,
            machine=machine,
        )
        if not self.target_contracts:
            return OptionalRuntimeTargetContract(
                platform=selected_platform,
                machine=selected_machine,
                contract_state=self.contract_state,
                cutover_ready=self.cutover_ready,
                install_action_available=self.install_action_available,
                activation_available=self.activation_available,
            )
        matches = tuple(
            contract
            for contract in self.target_contracts
            if contract.platform == selected_platform and contract.machine == selected_machine
        )
        if len(matches) == 1:
            return matches[0]
        return OptionalRuntimeTargetContract(
            platform=selected_platform,
            machine=selected_machine,
            contract_state="unsupported_target",
            cutover_ready=False,
            install_action_available=False,
            activation_available=False,
        )

    def to_spec_dict(self) -> dict:
        """Return immutable fields used to identify the exact reviewed spec."""

        spec = {
            "schemaVersion": OPTIONAL_RUNTIME_SCHEMA_VERSION,
            "id": self.id,
            "label": self.label,
            "contractState": self.contract_state,
            "cutoverReady": self.cutover_ready,
            "installActionAvailable": self.install_action_available,
            "activationAvailable": self.activation_available,
            "installPolicy": self.install_policy,
            "targetContracts": [contract.to_spec_dict() for contract in self.target_contracts],
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
        if self.source_builds:
            spec["sourceBuilds"] = [
                deepcopy(source_build) for source_build in self.source_builds
            ]
        return spec

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
                "AutoConfig",
                "AutoTokenizer",
                "AutoModelForCausalLM",
                "LlamaForCausalLM",
                "CLIPImageProcessor",
                "CLIPVisionModel",
                "UMT5EncoderModel",
                "T5EncoderModel",
                "T5Tokenizer",
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
                "Qwen3Model",
                "Qwen3VLForConditionalGeneration",
                "Qwen3VLProcessor",
                "AutoProcessor",
                "AutoModelForImageTextToText",
                "Idefics3ForConditionalGeneration",
                "Idefics3Processor",
                "AutoModelForMultimodalLM",
                "JanusForConditionalGeneration",
                "JanusProcessor",
                "JanusImageProcessor",
                "AutoModelForSpeechSeq2Seq",
                "AnyToAnyPipeline",
                "pipeline",
                "Mistral3Model",
                "Mistral3ForConditionalGeneration",
                "Ministral3ForCausalLM",
                "GlmImageForConditionalGeneration",
                "GlmImageProcessor",
                "ByT5Tokenizer",
                "BitsAndBytesConfig",
                "GemmaModel",
                "Gemma2Model",
                "GemmaPreTrainedModel",
                "Gemma2PreTrainedModel",
                "GemmaTokenizer",
                "GemmaTokenizerFast",
                "transformers.models.t5gemma.modeling_t5gemma:T5GemmaEncoder",
                "LlamaTokenizer",
                "LlamaTokenizerFast",
                "transformers.core_model_loading:tqdm",
                "transformers.utils.logging:tqdm",
                "transformers.utils.logging:get_verbosity",
                "transformers.utils.logging:set_verbosity",
            ),
            required_class_symbols=(
                "Qwen2_5_VLForConditionalGeneration",
                "Qwen2Tokenizer",
                "Qwen2VLProcessor",
                "AutoConfig",
                "AutoTokenizer",
                "AutoModelForCausalLM",
                "LlamaForCausalLM",
                "CLIPImageProcessor",
                "CLIPVisionModel",
                "UMT5EncoderModel",
                "T5EncoderModel",
                "T5Tokenizer",
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
                "Qwen3Model",
                "Qwen3VLForConditionalGeneration",
                "Qwen3VLProcessor",
                "AutoProcessor",
                "AutoModelForImageTextToText",
                "Idefics3ForConditionalGeneration",
                "Idefics3Processor",
                "AutoModelForMultimodalLM",
                "JanusForConditionalGeneration",
                "JanusProcessor",
                "JanusImageProcessor",
                "AutoModelForSpeechSeq2Seq",
                "AnyToAnyPipeline",
                "Mistral3Model",
                "Mistral3ForConditionalGeneration",
                "Ministral3ForCausalLM",
                "GlmImageForConditionalGeneration",
                "GlmImageProcessor",
                "ByT5Tokenizer",
                "BitsAndBytesConfig",
                "GemmaModel",
                "Gemma2Model",
                "GemmaPreTrainedModel",
                "Gemma2PreTrainedModel",
                "GemmaTokenizer",
                "GemmaTokenizerFast",
                "transformers.models.t5gemma.modeling_t5gemma:T5GemmaEncoder",
                "LlamaTokenizer",
                "LlamaTokenizerFast",
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
        "QwenImageControlNetModel",
        "QwenImageControlNetPipeline",
        "QwenImageLayeredPipeline",
        "QwenImageEditPipeline",
        "QwenImageEditPlusPipeline",
        "QwenImageImg2ImgPipeline",
        "QwenImageInpaintPipeline",
        "QwenImageEditInpaintPipeline",
        "ChromaImg2ImgPipeline",
        "ChromaInpaintPipeline",
        "LTX2Pipeline",
        "QwenImageModularPipeline",
        "QwenImageEditModularPipeline",
        "QwenImageEditPlusModularPipeline",
        "QwenImageLayeredModularPipeline",
        "AnimateDiffPAGPipeline",
        "AnimateDiffVideoToVideoPipeline",
        "AnimateDiffControlNetPipeline",
        "AnimateDiffVideoToVideoControlNetPipeline",
        "CogVideoXVideoToVideoPipeline",
        "WanVACEPipeline",
        "WanVideoToVideoPipeline",
        "WanPipeline",
        "WanImageToVideoPipeline",
        "WanModularPipeline",
        "WanImage2VideoModularPipeline",
        "LTXConditionPipeline",
        "AceStepPipeline",
        "StableAudioPipeline",
        "LongCatAudioDiTPipeline",
        "AudioLDM2Pipeline",
        "ShapEPipeline",
        "StableDiffusionXLModularPipeline",
        "StableDiffusionXLPipeline",
        "StableDiffusionXLImg2ImgPipeline",
        "StableDiffusionXLInpaintPipeline",
        "StableDiffusionControlNetPipeline",
        "StableDiffusionControlNetImg2ImgPipeline",
        "StableDiffusionControlNetInpaintPipeline",
        "StableDiffusionControlNetPAGPipeline",
        "StableDiffusionControlNetPAGInpaintPipeline",
        "StableDiffusionXLControlNetPipeline",
        "StableDiffusionXLControlNetImg2ImgPipeline",
        "StableDiffusionXLControlNetInpaintPipeline",
        "StableDiffusionXLControlNetPAGPipeline",
        "StableDiffusionXLControlNetPAGImg2ImgPipeline",
        "ZImageImg2ImgPipeline",
        "ZImageInpaintPipeline",
        "FluxImg2ImgPipeline",
        "FluxInpaintPipeline",
        "FluxPipeline",
        "Flux2KleinPipeline",
        "Flux2KleinInpaintPipeline",
        "FluxKontextPipeline",
        "FluxKontextInpaintPipeline",
        "FluxFillPipeline",
        "FluxControlPipeline",
        "FluxControlImg2ImgPipeline",
        "FluxControlInpaintPipeline",
        "FluxPriorReduxPipeline",
        "FluxModularPipeline",
        "FluxKontextModularPipeline",
        "Flux2KleinModularPipeline",
        "ErnieImagePipeline",
        "ErnieImageTransformer2DModel",
        "AutoencoderKLFlux2",
        "GlmImagePipeline",
        "GlmImageTransformer2DModel",
        "HunyuanDiTPipeline",
        "HunyuanDiTControlNetPipeline",
        "HunyuanDiT2DModel",
        "HunyuanDiT2DControlNetModel",
        "JoyImageEditPipeline",
        "JoyImageEditPlusPipeline",
        "JoyImageEditTransformer3DModel",
        "JoyImageEditPlusTransformer3DModel",
        "Kandinsky3Pipeline",
        "Kandinsky3Img2ImgPipeline",
        "Kandinsky3UNet",
        "LongCatImagePipeline",
        "LongCatImageEditPipeline",
        "LongCatImageTransformer2DModel",
        "LuminaPipeline",
        "LuminaNextDiT2DModel",
        "Lumina2Pipeline",
        "Lumina2Transformer2DModel",
        "OmniGenPipeline",
        "OmniGenTransformer2DModel",
        "OvisImagePipeline",
        "OvisImageTransformer2DModel",
        "PRXPipeline",
        "PRXTransformer2DModel",
        "NucleusMoEImagePipeline",
        "NucleusMoEImageTransformer2DModel",
        "AutoencoderKLQwenImage",
    ),
    require_peft_backend=True,
    # Exact locks do not enable installation by themselves. Qualification,
    # action availability, activation, and cutover remain separate gates.
    artifact_locks=_transformers_peft_artifact_locks(),
    pipeline_adapter_symbols=(
        "AceStepPipeline",
        "FluxPipeline",
        "Flux2KleinPipeline",
        "LTXConditionPipeline",
        "QwenImagePipeline",
        "QwenImageControlNetPipeline",
        "QwenImageLayeredPipeline",
        "QwenImageEditPipeline",
        "QwenImageEditPlusPipeline",
        "QwenImageImg2ImgPipeline",
        "QwenImageInpaintPipeline",
        "ChromaImg2ImgPipeline",
        "ChromaInpaintPipeline",
        "LTX2Pipeline",
        "StableDiffusionXLPipeline",
        "StableDiffusionXLImg2ImgPipeline",
        "StableDiffusionXLInpaintPipeline",
        "ZImageImg2ImgPipeline",
        "ZImageInpaintPipeline",
        "FluxImg2ImgPipeline",
        "FluxInpaintPipeline",
        "FluxKontextInpaintPipeline",
        "Flux2KleinInpaintPipeline",
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
    # Windows and Linux x86-64 have executable qualification evidence. Other
    # architectures stay base-delivered until the same qualifier runs there.
    contract_state="qualified_platform_scoped",
    cutover_ready=True,
    install_action_available=True,
    activation_available=True,
    target_contracts=tuple(
        OptionalRuntimeTargetContract(
            platform=platform_name,
            machine=machine,
            contract_state=(
                "qualified"
                if (platform_name, machine) in {("linux", "x86_64"), ("windows", "x86_64")}
                else "candidate_unqualified"
            ),
            cutover_ready=(platform_name, machine)
            in {("linux", "x86_64"), ("windows", "x86_64")},
            install_action_available=(platform_name, machine)
            in {("linux", "x86_64"), ("windows", "x86_64")},
            activation_available=(platform_name, machine)
            in {("linux", "x86_64"), ("windows", "x86_64")},
        )
        for platform_name, _python_tag, machine in _OPTIONAL_RUNTIME_TARGETS
    ),
)


_TRANSFORMERS_MAIN_PEFT_PROFILE = replace(
    _TRANSFORMERS_PEFT_PROFILE,
    id=TRANSFORMERS_MAIN_PEFT_RUNTIME_PROFILE_ID,
    label="Hugging Face Transformers main + PEFT (Linux x86-64 qualified)",
    packages=(
        replace(
            _TRANSFORMERS_PEFT_PROFILE.packages[0],
            version="5.16.0.dev0",
            distribution_url=(
                "https://github.com/huggingface/transformers/commit/"
                f"{TRANSFORMERS_MAIN_COMMIT}"
            ),
        ),
        *_TRANSFORMERS_PEFT_PROFILE.packages[1:],
    ),
    artifact_locks=tuple(
        dict(artifact)
        for artifact in _TRANSFORMERS_PEFT_PROFILE.artifact_locks
        if artifact["distribution"] != "transformers"
    ),
    source_builds=(_transformers_main_source_build(),),
    # Actions are admitted by exact target contracts below. Keep the profile
    # defaults closed so no unlisted or newly detected platform inherits the
    # Linux qualification.
    contract_state="qualified_platform_scoped",
    cutover_ready=False,
    install_action_available=False,
    activation_available=False,
    target_contracts=tuple(
        OptionalRuntimeTargetContract(
            platform=platform_name,
            machine=machine,
            contract_state=(
                "qualified"
                if (platform_name, machine) == ("linux", "x86_64")
                else "candidate_unqualified"
            ),
            cutover_ready=(platform_name, machine) == ("linux", "x86_64"),
            install_action_available=(platform_name, machine)
            == ("linux", "x86_64"),
            activation_available=(platform_name, machine) == ("linux", "x86_64"),
        )
        for platform_name, _python_tag, machine in _OPTIONAL_RUNTIME_TARGETS
    ),
)

OPTIONAL_RUNTIME_PROFILES: Mapping[str, OptionalRuntimeProfile] = MappingProxyType(
    {
        _TRANSFORMERS_PEFT_PROFILE.id: _TRANSFORMERS_PEFT_PROFILE,
        _TRANSFORMERS_MAIN_PEFT_PROFILE.id: _TRANSFORMERS_MAIN_PEFT_PROFILE,
    }
)


def project_optional_runtime_qualification(
    profile: OptionalRuntimeProfile,
    *,
    platform_name: str | None = None,
    machine: str | None = None,
) -> OptionalRuntimeProfile:
    """Project one pending target for the isolated qualification harness."""

    target = profile.contract_for_target(
        platform_name=platform_name,
        machine=machine,
    )
    target_flags = (
        target.cutover_ready,
        target.install_action_available,
        target.activation_available,
    )
    if target.contract_state == "qualified" and target_flags == (True, True, True):
        return profile
    if target.contract_state != "candidate_unqualified" or target_flags != (False, False, False):
        raise RuntimeError("qualification requires a coherent qualified or pending target contract")
    qualified_target = replace(
        target,
        contract_state="qualified",
        cutover_ready=True,
        install_action_available=True,
        activation_available=True,
    )
    return replace(
        profile,
        target_contracts=tuple(
            qualified_target
            if (contract.platform, contract.machine) == (target.platform, target.machine)
            else contract
            for contract in profile.target_contracts
        ),
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
    platform_name: str | None = None,
    machine: str | None = None,
) -> list[dict]:
    """Publish exact contracts plus local metadata-only presence observations.

    Exact package presence is intentionally reported as ``present_unqualified``:
    this slice neither validates an overlay origin nor makes the runtime ready
    for optional-runtime cutover.
    """

    resolve_version = version_resolver or metadata.version
    public_profiles = []
    for profile in _selected_profiles(profile_ids):
        target_contract = profile.contract_for_target(
            platform_name=platform_name,
            machine=machine,
        )
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
                **target_contract.to_spec_dict(),
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
