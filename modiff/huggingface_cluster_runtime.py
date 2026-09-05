"""Live runtime qualification for graph-qualified Hugging Face Cluster Nodes.

Static node-library admission deliberately does not consult installed models,
optional environments, or hardware.  This module joins those volatile facts
for an explicit Expert recipe when the backend has no declared Auto pair.  It
does not publish executability and it never imports Diffusers, Transformers,
Torch, or a model module.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import re
import time
from typing import Any, Mapping

from modiff.auto_resource import artifact_revision_cache_status, build_auto_resource_plan
from modiff.block_definition_v2 import (
    block_definition_canonical_sha256_v2,
    block_execution_parameter_hash_v2,
    validate_block_instance_v2,
)
from modiff.diffusers_profiles import execution_profiles_for_execution
from modiff.modular_contract_only_registry import equivalent_modular_targets
from modiff.model_artifact_catalog import catalog_repository_pin
from modiff.optional_runtime_execution import (
    optional_runtime_requirement_blocks_execution,
    optional_runtime_requirement_for_execution,
)
from modiff.registered_block_v2_catalog import registered_block_v2_definition_pins
from modiff.studio_execution_specs import (
    studio_execution_spec_for_pair,
    studio_expert_resource_requirements,
)


CLUSTER_RUNTIME_QUALIFICATION_SCHEMA_VERSION = 1
_RECIPE_KEYS = {"device", "dtype", "quantizationMode", "autoOffload", "offloadMode"}
_DTYPES = {"float32", "float16", "bfloat16"}
_QUANTIZATION_MODES = {"none", "bnb_4bit", "bnb_8bit", "quanto_float8", "torchao_float8"}
_DEVICE = re.compile(r"(?:cpu|mps|cuda:[0-9]{1,3}|xpu:[0-9]{1,3})")
_EQUIVALENT_STANDARD_EXECUTION_CLASSES = {
    "Flux2ModularPipeline": "Flux2Pipeline",
    "ErnieImageModularPipeline": "ErnieImagePipeline",
    "LTXModularPipeline": "LTXConditionPipeline",
    "LTX2ModularPipeline": "LTX2ConditionPipeline",
    "Wan22ModularPipeline": "Wan22Pipeline",
    "Wan22Image2VideoModularPipeline": "WanImageToVideoPipeline",
}
_EQUIVALENT_STANDARD_WORKFLOW_EXECUTION_CLASSES = {
    ("LTX2ModularPipeline", "in_context"): "LTX2InContextPipeline",
}


def _equivalent_standard_execution_class(model_type: str, workflow_id: str) -> str | None:
    return _EQUIVALENT_STANDARD_WORKFLOW_EXECUTION_CLASSES.get(
        (model_type, workflow_id),
        _EQUIVALENT_STANDARD_EXECUTION_CLASSES.get(model_type),
    )


def _effective_reviewed_artifact(
    admission: Mapping[str, Any],
    profile: Any,
    selected_repository: Any,
) -> dict[str, str]:
    """Resolve an optional same-pipeline instance choice to one exact pin."""

    artifact = admission.get("artifact")
    if not isinstance(artifact, Mapping):
        raise _invalid("the admission has no exact artifact.")
    baseline_repo = str(artifact.get("repo") or "")
    baseline_revision = str(artifact.get("revision") or "")
    selected = str(selected_repository or baseline_repo).strip()
    if selected == baseline_repo:
        return {"repo": baseline_repo, "revision": baseline_revision}
    if "modelVariant" not in admission.get("executionParameterSources", ()):
        raise _invalid("this registered Block does not expose a reviewed model-variant control.")
    admitted = {
        str(repository)
        for repository in (
            profile.default_repo,
            profile.fallback_repo,
            *profile.compatible_repos,
        )
        if isinstance(repository, str) and repository
    }
    if selected not in admitted:
        raise _invalid(f"model repository {selected!r} is not an admitted same-pipeline variant.")
    pin = catalog_repository_pin(selected, model_type=str(profile.model_type))
    revision = str((pin or {}).get("revision") or "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise _invalid("the selected model variant has no immutable catalog revision.")
    return {"repo": selected, "revision": revision}
_DIRECT_LOADER_CONTRACTS = {
    "direct-diffusers-image": ("modules.DiffusersImage", "LoadPipeline"),
    "direct-diffusers-video": ("modules.DiffusersVideo", "LoadPipeline"),
}
_GIB = 1024**3
_AUTO_AUTHORITY_KEYS = {
    "schemaVersion",
    "instance",
    "form",
}
_BLOCK_RECEIPT_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{1,511}$")
_AUTO_RESOURCE_FORM_FIELDS = {
    "width",
    "height",
    "steps",
    "numFrames",
    "device",
    "dtype",
    "quantizationMode",
    "autoOffload",
    "offloadMode",
}


# Historical literal retained during the V1 recovery compatibility window. It
# is not execution authority; current pins come from the independently rebuilt,
# cross-runtime-validated compiled catalog below.
_LEGACY_REGISTERED_BLOCK_V2_DEFINITION_PINS: Mapping[str, tuple[str, str]] = {
    "diffusers.cluster-admission:AnimaModularPipeline:img2img:workflow:official_top_level_blocks": (
        "block-definition-v2-3e0ff301",
        "sha256:da899f67a8dd4f61dbcdc39fae37e4fc76b1f9014d950be5d291512abbd82fe1",
    ),
    "diffusers.cluster-admission:AnimaModularPipeline:text2image:workflow:official_top_level_blocks": (
        "block-definition-v2-0700db91",
        "sha256:41dfbad54e74eb70607deed1d418ec5163c774727fee2d78c2005529a1b00127",
    ),
    "diffusers.cluster-admission:Cosmos3DistilledModularPipeline:image2video:workflow:official_top_level_blocks": (
        "block-definition-v2-f6daa6b4",
        "sha256:71f3951a436b166e4a86b45c6f0f7b454f16b5f504cefd44b7e6185ddc45dda6",
    ),
    "diffusers.cluster-admission:Cosmos3DistilledModularPipeline:text2image:workflow:official_top_level_blocks": (
        "block-definition-v2-e98c80c0",
        "sha256:77002c798e90eeeb2e840f118268e1eba32d93538eb2d51c73a17e38f6e07f4d",
    ),
    "diffusers.cluster-admission:Cosmos3OmniModularPipeline:image2video_with_sound:workflow:official_top_level_blocks": (
        "block-definition-v2-eaef6477",
        "sha256:4601fa24cafbbe8b860d7467b38339e8186a4cdce773ff94e3cda590ec91a6eb",
    ),
    "diffusers.cluster-admission:Cosmos3OmniModularPipeline:image2video:workflow:official_top_level_blocks": (
        "block-definition-v2-53f100ac",
        "sha256:0676cc739451af78a431ffb8f378291cb3918772665e7c7b84c02ceb9545453e",
    ),
    "diffusers.cluster-admission:Cosmos3OmniModularPipeline:text2image:workflow:official_top_level_blocks": (
        "block-definition-v2-9e49b1ad",
        "sha256:575fa8ff246f58f67ff534c30b6c456ae665baa0dd7fdf003276ff923507a5e4",
    ),
    "diffusers.cluster-admission:Cosmos3OmniModularPipeline:text2video_with_sound:workflow:official_top_level_blocks": (
        "block-definition-v2-014577d6",
        "sha256:fcc987f4684aa0b682a489253888570cc662bb3b38c3e5e12ab42916adf5da32",
    ),
    "diffusers.cluster-admission:Cosmos3OmniModularPipeline:text2video:workflow:official_top_level_blocks": (
        "block-definition-v2-52677066",
        "sha256:409af71f2ad30e898cfe4be8ba728cddc5a65b774347e5443f4796f1720d04b4",
    ),
    "diffusers.cluster-admission:Cosmos3OmniModularPipeline:video2video_with_sound:workflow:official_top_level_blocks": (
        "block-definition-v2-86030b62",
        "sha256:7383e14ffdb807e7cf5016d6deb44713d4dcaedecfcb24f97c6f265ee094907e",
    ),
    "diffusers.cluster-admission:Cosmos3OmniModularPipeline:video2video:workflow:official_top_level_blocks": (
        "block-definition-v2-8c3a003b",
        "sha256:c8de506963cfb8c7aaf8e102d7aec7689033a32c10e5105e155f7b52c9ddc731",
    ),
    "diffusers.cluster-admission:ErnieImageModularPipeline:text2image:mode:equivalent_standard_route": (
        "block-definition-v2-778d2320",
        "sha256:ef45c8d18cf88c5f23caca37dcf04b848bc687a2e57fdb1fe5374f07448b4ac7",
    ),
    "diffusers.cluster-admission:Flux2KleinBaseModularPipeline:image_conditioned:mode:edit_image": (
        "block-definition-v2-6e4f2796",
        "sha256:ad50e37a85108aa731473214a519f2b8b782a947509f3d5334c4657ba52198de",
    ),
    "diffusers.cluster-admission:Flux2KleinBaseModularPipeline:text2image:mode:text_to_image": (
        "block-definition-v2-cde6daf8",
        "sha256:3394ccb262d53254dda4aecd8718dadadb1cb16c60548df4b93ca0295c65f970",
    ),
    "diffusers.cluster-admission:Flux2KleinModularPipeline:image_conditioned:mode:edit_image": (
        "block-definition-v2-88e893de",
        "sha256:a8d983ee07ef4fd4cdfdac0b15d1de8ef42ccf2ea5cbbdb03263250aaab0e292",
    ),
    "diffusers.cluster-admission:Flux2KleinModularPipeline:text2image:mode:text_to_image": (
        "block-definition-v2-4e6ba70c",
        "sha256:ab57fb06f35d74b7f14909d9da9a46eadb0ce00300500e89ec668b16ed1bb507",
    ),
    "diffusers.cluster-admission:Flux2ModularPipeline:image_conditioned:mode:equivalent_standard_route": (
        "block-definition-v2-3f971efd",
        "sha256:fb77b70e97e1daefbfcfe4a636ceb72fc1d6e462604d3bf963860a93e53a5286",
    ),
    "diffusers.cluster-admission:Flux2ModularPipeline:text2image:mode:equivalent_standard_route": (
        "block-definition-v2-626b28a6",
        "sha256:d9a6a962cff5d1830a10123034d6d8745cf8f15b64d5fa761ba4a972faacf5c7",
    ),
    "diffusers.cluster-admission:FluxKontextModularPipeline:image_conditioned:mode:edit_image": (
        "block-definition-v2-5b90216a",
        "sha256:04b28618ec09a534518195404fd385a86a6be191c9d3ddb441dfabb00aaf2490",
    ),
    "diffusers.cluster-admission:FluxKontextModularPipeline:text2image:mode:text_to_image": (
        "block-definition-v2-ab0e4ede",
        "sha256:117a766587daae7470297d0cb875fd8f8c7f8935c4942730f7806f1a334f0e2c",
    ),
    "diffusers.cluster-admission:FluxModularPipeline:image2image:mode:image_to_image": (
        "block-definition-v2-4389ae9b",
        "sha256:dbf0a348ecc98ff4e08b543136a59feac170a36c31673de5cd1f8a36372fcd98",
    ),
    "diffusers.cluster-admission:FluxModularPipeline:text2image:mode:text_to_image": (
        "block-definition-v2-bf198a70",
        "sha256:91c25694f017b8d1303ddebe38f06e0921d60ab8e99d7e6f52348c34220a1621",
    ),
    "diffusers.cluster-admission:HeliosModularPipeline:image2video:workflow:official_top_level_blocks": (
        "block-definition-v2-4e7b95ac",
        "sha256:4158b58a9a1cb7c68690438828a1279a3b7720df722a09ec4a37074d5e34b076",
    ),
    "diffusers.cluster-admission:HeliosModularPipeline:text2video:workflow:official_top_level_blocks": (
        "block-definition-v2-ae615655",
        "sha256:98992cdb006806199cd7174bf6e36dd5d0c305a0f1cfc21844d18752717da5b2",
    ),
    "diffusers.cluster-admission:HeliosModularPipeline:video2video:workflow:official_top_level_blocks": (
        "block-definition-v2-e65a6699",
        "sha256:7562e3034bec6a244781227e9cded23bea0d64fe93efe8cdda4cb50fc4c7a49b",
    ),
    "diffusers.cluster-admission:HeliosPyramidDistilledModularPipeline:image2video:workflow:official_top_level_blocks": (
        "block-definition-v2-1bc2d348",
        "sha256:b65e9137e9c32e6267e7df067777e0b54684e454cb177dfcd0fc79ed1061ab52",
    ),
    "diffusers.cluster-admission:HeliosPyramidDistilledModularPipeline:text2video:workflow:official_top_level_blocks": (
        "block-definition-v2-5edb6824",
        "sha256:f3b2a3162d25d7edef61ef5f7450356287ac41b5c15a0c28cfde2b204ba419b9",
    ),
    "diffusers.cluster-admission:HeliosPyramidDistilledModularPipeline:video2video:workflow:official_top_level_blocks": (
        "block-definition-v2-93ef8d8c",
        "sha256:3f627de8c5f957b05e9c4a7df179d0447632f7dec1c8d35928443d5a0b3619f2",
    ),
    "diffusers.cluster-admission:HeliosPyramidModularPipeline:image2video:workflow:official_top_level_blocks": (
        "block-definition-v2-68c27e73",
        "sha256:75353757712c7f26e918a8d88a4856bd29958b23716208fda89050eff016d7c8",
    ),
    "diffusers.cluster-admission:HeliosPyramidModularPipeline:text2video:workflow:official_top_level_blocks": (
        "block-definition-v2-57590dd5",
        "sha256:2cf79d03ce9ab081a311e3f56dd7dab4a95ba956e29e1e0ddc701dae4b08e038",
    ),
    "diffusers.cluster-admission:HeliosPyramidModularPipeline:video2video:workflow:official_top_level_blocks": (
        "block-definition-v2-65e96a3a",
        "sha256:4989a2b204d7275c90e5c0abadc042ff48aa5812622d7d518ba35208df4260eb",
    ),
    "diffusers.cluster-admission:HunyuanVideo15ModularPipeline:image2video:workflow:official_top_level_blocks": (
        "block-definition-v2-5e2d8724",
        "sha256:5cbac0236e1c24f0b0ae59b4b24dc68bd964e2226f80d9550d2a4b35c22250d0",
    ),
    "diffusers.cluster-admission:HunyuanVideo15ModularPipeline:text2video:workflow:official_top_level_blocks": (
        "block-definition-v2-6320339d",
        "sha256:a48416b2c496c6ccfc221a6c4c11ef6c7c053a39a335a82ff955b52ba05f7ece",
    ),
    "diffusers.cluster-admission:LTX2ModularPipeline:condition:mode:equivalent_standard_route": (
        "block-definition-v2-a50af000",
        "sha256:95870bbf3ef8bb1641190ec30996ec9e2c9136e563543b4c47c0d33cefb5aeb2",
    ),
    "diffusers.cluster-admission:LTX2ModularPipeline:image2video:mode:equivalent_standard_route": (
        "block-definition-v2-4aa366eb",
        "sha256:63d01a6231acfb764c956a3c10586f18d2894fc11ddeb1698796ac0d48b7398c",
    ),
    "diffusers.cluster-admission:LTX2ModularPipeline:in_context:mode:equivalent_standard_route": (
        "block-definition-v2-1643638f",
        "sha256:99cd61fc5ff8c64cc8a88768710d9706f4adc8d958d2a45438e54b822badb48b",
    ),
    "diffusers.cluster-admission:LTX2ModularPipeline:text2video:mode:equivalent_standard_route": (
        "block-definition-v2-ddc149ff",
        "sha256:87d7c60a574bd09ea25553cdb41b63e70706718900a5cf21c9433bfa384e8e53",
    ),
    "diffusers.cluster-admission:LTXModularPipeline:image2video:mode:equivalent_standard_route": (
        "block-definition-v2-1467ba5e",
        "sha256:a6157cd68f88e7becbf2186f80eb1a9ee32d99d7b57f7befd96a5ab6ac888dc4",
    ),
    "diffusers.cluster-admission:LTXModularPipeline:text2video:mode:equivalent_standard_route": (
        "block-definition-v2-9081de68",
        "sha256:8b4f828427be046755a17ff0b985636f50ed5dbd9f251520ddb93d1933e280cf",
    ),
    "diffusers.cluster-admission:MiniMaxH3ModularPipeline:fl2va:workflow:official_top_level_blocks": (
        "block-definition-v2-5bb1b2cc",
        "sha256:bd6b14e56de85ae79edc09fd2f85f4c1667d19fbdd4e242a9b2ae179a2a57d09",
    ),
    "diffusers.cluster-admission:MiniMaxH3ModularPipeline:ref2va:workflow:official_top_level_blocks": (
        "block-definition-v2-c9a45340",
        "sha256:19a1d2d1bfd87dea68f9cce4e8ca157f17ad1a572f09cd5fbac0e514cbfd13d8",
    ),
    "diffusers.cluster-admission:MiniMaxH3ModularPipeline:t2va:workflow:official_top_level_blocks": (
        "block-definition-v2-b84c4cea",
        "sha256:77e301b2b63402627fa6f666c4dae6e1850813bc0f0ee4d6dc8bd80940276bdb",
    ),
    "diffusers.cluster-admission:MiniMaxMusic3ModularPipeline:default:workflow:official_top_level_blocks": (
        "block-definition-v2-5f610dae",
        "sha256:b5f8b4618df9680126056c82203e6f867529f572e114ee874761f5ca689d2e7d",
    ),
    "diffusers.cluster-admission:QwenImageEditModularPipeline:image_conditioned_inpainting:state_flow:image_conditioned_inpainting": (
        "block-definition-v2-6db80189",
        "sha256:77833b94588e02f1b31f0553c0a9e002465f4504a518daeffb852838335e7c64",
    ),
    "diffusers.cluster-admission:QwenImageEditModularPipeline:image_conditioned:mode:edit_image": (
        "block-definition-v2-9e1d3446",
        "sha256:ab4ac8ccd84357b8dfdea998ef4deba0422c7e4e7c50ea7648cf129b98a5c4e5",
    ),
    "diffusers.cluster-admission:QwenImageEditPlusModularPipeline:default:mode:edit_image": (
        "block-definition-v2-eea02ef9",
        "sha256:e7ca6f828728952d31cf139d0137e03f06c08aee7d6657375f0ed233a6caff2d",
    ),
    "diffusers.cluster-admission:QwenImageEditPlusModularPipeline:default:mode:multi_image_reference_edit": (
        "block-definition-v2-68230891",
        "sha256:a64ca4e8ba5a8bdbe8e3b273359e7e9c6311c6e924625f7cf61b2beeac11db66",
    ),
    "diffusers.cluster-admission:QwenImageLayeredModularPipeline:default:mode:layer_decomposition": (
        "block-definition-v2-672fd128",
        "sha256:e0b128133a292e95f8c26f1f4af7c2be38072ba2a0b15292c2a47edb334c2aa5",
    ),
    "diffusers.cluster-admission:QwenImageModularPipeline:controlnet_image2image:state_flow:controlnet_image2image": (
        "block-definition-v2-b4d7b3d9",
        "sha256:b04fe8002ec90fded4e6afc44f9a372beeb1318e05a1c3a9b12710e6eab8de82",
    ),
    "diffusers.cluster-admission:QwenImageModularPipeline:controlnet_inpainting:state_flow:controlnet_inpainting": (
        "block-definition-v2-56a0f783",
        "sha256:a1126edca088101a6c5a3656d7c58f062298747dac0ccf7a457cd1766951cdd2",
    ),
    "diffusers.cluster-admission:QwenImageModularPipeline:controlnet_text2image:mode:control_image": (
        "block-definition-v2-963f08e9",
        "sha256:3005bc16be252fb045135f7b566416ef2ec327724e279ee038571cd664df8f92",
    ),
    "diffusers.cluster-admission:QwenImageModularPipeline:image2image:state_flow:image2image": (
        "block-definition-v2-70c164ae",
        "sha256:2185b2603637f71b387c1e222c7d6f715cd9cf715db4d63548092f94dff72b0d",
    ),
    "diffusers.cluster-admission:QwenImageModularPipeline:inpainting:state_flow:inpainting": (
        "block-definition-v2-ed1824b4",
        "sha256:d84f73112e17932b5cbe7a327ee05480319cf1b9be8c91cc7e9a9a14645ac92a",
    ),
    "diffusers.cluster-admission:QwenImageModularPipeline:text2image:mode:text_to_image": (
        "block-definition-v2-d1dfd4fa",
        "sha256:2510249b605872d833b890f299827fd94a1b96e513f54470d4653e431973f719",
    ),
    "diffusers.cluster-admission:StableDiffusionXLModularPipeline:controlnet_image2image:mode:control_edit_image": (
        "block-definition-v2-1b2bba09",
        "sha256:38640cdceacb3604ea9421110e8f9e3bf9a1b899e52409e27198e277229f57be",
    ),
    "diffusers.cluster-admission:StableDiffusionXLModularPipeline:controlnet_inpainting:mode:control_inpaint": (
        "block-definition-v2-487704c3",
        "sha256:d8208eceb8cb65e0dfa43adb652b58e7818954b56c5d6a728d9a1d15c447ea61",
    ),
    "diffusers.cluster-admission:StableDiffusionXLModularPipeline:controlnet_text2image:mode:control_image": (
        "block-definition-v2-e66ac4a9",
        "sha256:0fb62ed62c8830e5878f47821473be3fe4e04ffd5337b99ac3b60d453a964429",
    ),
    "diffusers.cluster-admission:StableDiffusionXLModularPipeline:controlnet_union_image2image:state_flow:controlnet_union_image2image": (
        "block-definition-v2-5a909a73",
        "sha256:ab307cb54b5a26c6a96114190ab105d4882a0d1b572736f13d33ddc9b574a397",
    ),
    "diffusers.cluster-admission:StableDiffusionXLModularPipeline:controlnet_union_inpainting:state_flow:controlnet_union_inpainting": (
        "block-definition-v2-e2fbd563",
        "sha256:89c5fc14dd26293c976a90d88a79aa97d802d3ba005899fda4d520cc2b8430d9",
    ),
    "diffusers.cluster-admission:StableDiffusionXLModularPipeline:controlnet_union_text2image:state_flow:controlnet_union_text2image": (
        "block-definition-v2-664d129f",
        "sha256:6571ee635731d55ecc64c3941ee4da9d841474407362545df4a23702e5f9651c",
    ),
    "diffusers.cluster-admission:StableDiffusionXLModularPipeline:image2image:mode:image_to_image": (
        "block-definition-v2-01cd1b45",
        "sha256:b79366160139708869af2b57688cc9ead3d32f910f67bddaf1d1807d392f2487",
    ),
    "diffusers.cluster-admission:StableDiffusionXLModularPipeline:inpainting:mode:inpaint": (
        "block-definition-v2-650c893c",
        "sha256:2534edf697047a5e6ab12f8e1194ae86aa420987700c34e58b6060e3c1f2fd3c",
    ),
    "diffusers.cluster-admission:StableDiffusionXLModularPipeline:ip_adapter_controlnet_image2image:state_flow:ip_adapter_controlnet_image2image": (
        "block-definition-v2-62688431",
        "sha256:7f642075eca36295c93824b53dda6c2902d22a2abb162b71d3366c18cfef959c",
    ),
    "diffusers.cluster-admission:StableDiffusionXLModularPipeline:ip_adapter_controlnet_inpainting:state_flow:ip_adapter_controlnet_inpainting": (
        "block-definition-v2-61b07bf0",
        "sha256:eb9d28fec50fe03e34a0586047a1dccf813db61718a26b1c12345a01d4ce8ab1",
    ),
    "diffusers.cluster-admission:StableDiffusionXLModularPipeline:ip_adapter_controlnet_text2image:state_flow:ip_adapter_controlnet_text2image": (
        "block-definition-v2-9464562b",
        "sha256:b509f549e233929b2932429d3421d74259057aa8f5e2446fbc1210b88167372e",
    ),
    "diffusers.cluster-admission:StableDiffusionXLModularPipeline:ip_adapter_controlnet_union_image2image:state_flow:ip_adapter_controlnet_union_image2image": (
        "block-definition-v2-8b6ce381",
        "sha256:5e0b171c594a1c907cba667cc79251ba9a7dbc9f05555023bbd550574c9b1033",
    ),
    "diffusers.cluster-admission:StableDiffusionXLModularPipeline:ip_adapter_controlnet_union_inpainting:state_flow:ip_adapter_controlnet_union_inpainting": (
        "block-definition-v2-4f43d942",
        "sha256:f71b0bfcf47de4faa98c60d00025389c41af95fd66314cb76fb2b5e9f5a3ac2f",
    ),
    "diffusers.cluster-admission:StableDiffusionXLModularPipeline:ip_adapter_controlnet_union_text2image:state_flow:ip_adapter_controlnet_union_text2image": (
        "block-definition-v2-62ef7529",
        "sha256:2bddf2400eb3be0b6687613dd7b02bd4121b551a14fe596b471579cba6462870",
    ),
    "diffusers.cluster-admission:StableDiffusionXLModularPipeline:ip_adapter_image2image:state_flow:ip_adapter_image2image": (
        "block-definition-v2-e9543f79",
        "sha256:975ad44f91e3722cc2806f39635c6ef9bd5c0aec0a5c3e1e6766b0c85aef10b4",
    ),
    "diffusers.cluster-admission:StableDiffusionXLModularPipeline:ip_adapter_inpainting:state_flow:ip_adapter_inpainting": (
        "block-definition-v2-300f240f",
        "sha256:8a75da8aeaac37622f0163aee959ec7b236ce25842ab30440e559119c6ddc4a4",
    ),
    "diffusers.cluster-admission:StableDiffusionXLModularPipeline:ip_adapter_text2image:state_flow:ip_adapter_text2image": (
        "block-definition-v2-a6c7beb9",
        "sha256:f0cb554553df1e8935836c687b3ac4ff30c201ce49b8cedbcadcb271f3d4ee58",
    ),
    "diffusers.cluster-admission:StableDiffusionXLModularPipeline:text2image:mode:text_to_image": (
        "block-definition-v2-7f5d2a37",
        "sha256:d8c08d4f7b422508c2b014496de0ec3cbcfa32cf242b5ed5be845e68747039fd",
    ),
    "diffusers.cluster-admission:Wan22Image2VideoModularPipeline:default:mode:equivalent_standard_route": (
        "block-definition-v2-ed108bcb",
        "sha256:b29d38899c199f026ed7cef5b99cd8b21964dbea75a94043b794dd305fbae4b9",
    ),
    "diffusers.cluster-admission:Wan22ModularPipeline:default:mode:equivalent_standard_route": (
        "block-definition-v2-3cf916af",
        "sha256:c98310784c5b3147e9ab6f3c8dbfcd3521be1e1a598e8345613857704da5d367",
    ),
    "diffusers.cluster-admission:WanAnimate2DistilledModularPipeline:default:workflow:official_top_level_blocks": (
        "block-definition-v2-7083375c",
        "sha256:642302467a965e5e132595f3395cad54ca9aab0d3b720f387e4b54aff0b9a0eb",
    ),
    "diffusers.cluster-admission:WanAnimate2ModularPipeline:default:workflow:official_top_level_blocks": (
        "block-definition-v2-5c1f9c74",
        "sha256:05cb1e42c3199d97cf8d7bd469c1b35d7392b6dee43d6198d793f1a9ed224d60",
    ),
    "diffusers.cluster-admission:WanImage2VideoModularPipeline:flf2v:state_flow:flf2v": (
        "block-definition-v2-749846b2",
        "sha256:fb712af0585ee5f2db59fb6b4781cf71c4721cc40def4cf5852d3ba81524f603",
    ),
    "diffusers.cluster-admission:WanImage2VideoModularPipeline:image2video:mode:image_to_video": (
        "block-definition-v2-0ccaba01",
        "sha256:2b58a622bf5f4f1954568791fb09edbc38e477b495d3877d29091a0caf139cdc",
    ),
    "diffusers.cluster-admission:WanModularPipeline:default:mode:text_to_video": (
        "block-definition-v2-e061a995",
        "sha256:9809cf51a871c4b85b0a027532619c24ca82c26e436ffcbb83c3c0cbdd5495a2",
    ),
    "diffusers.cluster-admission:WanTI2VPipeline:text_to_video:mode:text_to_video": (
        "block-definition-v2-5a2c04d3",
        "sha256:ad86ef2f9b3dd1701e700343449634596c8b2a327bcc9f8d08d6e9c2e55d544c",
    ),
    "diffusers.cluster-admission:ZImageModularPipeline:image2image:mode:image_to_image": (
        "block-definition-v2-0c071b30",
        "sha256:5f5a0b3a467b5e5bb5012ba6d50d8bd9ee502bdb3e76f5680c44b02fde149810",
    ),
    "diffusers.cluster-admission:ZImageModularPipeline:text2image:mode:text_to_image": (
        "block-definition-v2-461b6b49",
        "sha256:a7d02d78c38552b9dfc7e3ab0ce51b1c03f658311a6a25c966980715efc35fce",
    ),
    "transformers.cluster-admission:HuggingFaceAnyToAnyModel:image_to_text": (
        "block-definition-v2-0afedc2e",
        "sha256:ad4632f5f78ba0c1daf9c7f7135ffdb8d37ee6b8b1b7147f84ee395f468b3b51",
    ),
    "transformers.cluster-admission:HuggingFaceAnyToAnyModel:text_generation": (
        "block-definition-v2-e1b6efdd",
        "sha256:7c1f79f4ddf0999d91740bb2fc3eec814ac71be39817aaf2a2bd235239290591",
    ),
    "transformers.cluster-admission:HuggingFaceAnyToAnyModel:text_to_image": (
        "block-definition-v2-ff70bec5",
        "sha256:62f98ac93547e764029737f5eb2d6079d08d118db99a9ee7f4ad6fddf0579f29",
    ),
    "transformers.cluster-admission:HuggingFaceCTCSpeechRecognitionModel:speech_to_text": (
        "block-definition-v2-0538acfd",
        "sha256:93e0da40072e8dc625e17572072eb85ac10dbf2b4a658357e351098b1ce3220a",
    ),
    "transformers.cluster-admission:HuggingFaceImageTextToTextModel:image_to_text": (
        "block-definition-v2-911b1c05",
        "sha256:1d2d81a9f5de42cba994a09593c4f7c6530bdc6910a470162d1dc28054b46edd",
    ),
    "transformers.cluster-admission:HuggingFaceSpeechRecognitionModel:speech_to_text": (
        "block-definition-v2-9572427b",
        "sha256:afc2e974a19b5082ce0dc1529d734a32439f61cc8795a56ad59db39285ac379f",
    ),
    "transformers.cluster-admission:HuggingFaceSpeechRecognitionModel:speech_translation": (
        "block-definition-v2-d24f88d0",
        "sha256:a29af182025a23d62f6bb7f619783d771d52775870dfcdd8eeb2176ab778bfd4",
    ),
    "transformers.cluster-admission:HuggingFaceTextGenerationModel:text_generation": (
        "block-definition-v2-d433f9be",
        "sha256:898427024ec85e94bcc6c0dc89e43a6c11285a8f2d975ca25f35b49982f74888",
    ),
}

# Backend-pinned public content hash and collision-resistant canonical SHA-256
# for every exact registered BlockDefinitionV2. Auto fails closed on all other
# definitions. Expert/manual execution deliberately does not depend on these
# pins. This mapping and fresh graph insertion now share one generated source,
# so schema changes cannot update the UI while leaving Auto on stale literals.
REGISTERED_BLOCK_V2_DEFINITION_PINS: Mapping[str, tuple[str, str]] = (
    registered_block_v2_definition_pins()
)


def _invalid(message: str) -> ValueError:
    return ValueError(f"Cannot qualify Hugging Face Cluster Node Expert execution: {message}")


def _auto_invalid(message: str) -> ValueError:
    return ValueError(f"Cannot qualify Hugging Face Cluster Node Auto execution: {message}")


def _single(items: list[Mapping[str, Any]], message: str) -> Mapping[str, Any]:
    if len(items) != 1:
        raise _invalid(message)
    return items[0]


def _public_artifact_status(status: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "repo": status.get("repo"),
        "revision": status.get("revision"),
        "installed": status.get("installed") is True,
        "complete": status.get("complete") is True,
        "exactRevisionComplete": status.get("exactRevisionComplete") is True,
        "repairRequired": status.get("repairRequired") is True,
        "reason": str(status.get("reason") or ""),
        "missingFiles": [str(value) for value in (status.get("missingFiles") or [])[:25]],
        "corruptFiles": [str(value) for value in (status.get("corruptFiles") or [])[:25]],
        "activeFiles": [str(value) for value in (status.get("activeFiles") or [])[:25]],
    }


def _runtime_identity(runtime_fingerprint: Mapping[str, Any]) -> tuple[str, str]:
    execution = str(runtime_fingerprint.get("fingerprint") or "")
    resource = str(runtime_fingerprint.get("resourceFingerprint") or "")
    fingerprint = re.compile(r"sha256:[0-9a-f]{64}")
    if fingerprint.fullmatch(execution) is None or fingerprint.fullmatch(resource) is None:
        raise _invalid("the backend runtime fingerprint is unavailable.")
    return execution, resource


def _assert_device(recipe: Mapping[str, Any], runtime_fingerprint: Mapping[str, Any]) -> None:
    device = str(recipe["device"])
    if _DEVICE.fullmatch(device) is None:
        raise _invalid("the requested device is invalid.")
    torch_state = runtime_fingerprint.get("torch")
    torch_state = torch_state if isinstance(torch_state, Mapping) else {}
    if device.startswith("cuda:"):
        index = int(device.split(":", 1)[1])
        if not torch_state.get("cuda_available") or index >= int(torch_state.get("cuda_device_count") or 0):
            raise _invalid("the requested CUDA/ROCm device is unavailable.")
    elif device.startswith("xpu:"):
        index = int(device.split(":", 1)[1])
        if not torch_state.get("xpu_available") or index >= int(torch_state.get("xpu_device_count") or 0):
            raise _invalid("the requested XPU device is unavailable.")
    elif device == "mps" and not torch_state.get("mps_available"):
        raise _invalid("the requested MPS device is unavailable.")


def _assert_recipe(recipe: Mapping[str, Any], profile: Any, runtime_fingerprint: Mapping[str, Any]) -> None:
    if set(recipe) != _RECIPE_KEYS:
        raise _invalid("the Expert recipe fields are incomplete or unknown.")
    if recipe.get("dtype") not in _DTYPES:
        raise _invalid("the requested dtype is unsupported.")
    if recipe.get("quantizationMode") not in _QUANTIZATION_MODES:
        raise _invalid("the requested quantization mode is unsupported.")
    if recipe.get("offloadMode") not in set(profile.supported_offload_modes):
        raise _invalid("the requested offload mode is outside the reviewed execution profile.")
    allowed_quantization = {"none", *profile.expert_quantization_modes}
    if recipe.get("quantizationMode") not in allowed_quantization:
        raise _invalid("the requested quantization mode is outside the reviewed Expert profile.")
    cuda_policy = profile.expert_cuda_policy
    if cuda_policy is not None and recipe.get("dtype") in set(cuda_policy.blocked_dtypes):
        raise _invalid("the requested dtype is blocked by the reviewed Expert CUDA policy.")
    auto_offload = recipe.get("autoOffload")
    if not isinstance(auto_offload, bool) or auto_offload != (recipe.get("offloadMode") != "none"):
        raise _invalid("auto-offload and offload mode disagree.")
    if recipe.get("offloadMode") != "none" and not str(recipe.get("device")).startswith("cuda:"):
        raise _invalid("the selected CPU/disk offload mode requires a CUDA-device API.")
    _assert_device(recipe, runtime_fingerprint)


def _assert_expert_resources(
    model_type: str,
    mode: str,
    recipe: Mapping[str, Any],
    runtime_fingerprint: Mapping[str, Any],
) -> None:
    requirement = studio_expert_resource_requirements(
        model_type,
        mode,
        str(recipe.get("offloadMode") or "none"),
    )
    if requirement is None:
        return
    hardware = runtime_fingerprint.get("hardware")
    if not isinstance(hardware, Mapping):
        raise _invalid("the reviewed large-model resource snapshot is unavailable.")
    accelerator = hardware.get("accelerator")
    accelerator = accelerator if isinstance(accelerator, Mapping) else None
    if accelerator is None:
        accelerator = next(
            (
                device
                for device in hardware.get("devices", ())
                if isinstance(device, Mapping) and device.get("type") in {"cuda", "mps", "xpu"}
            ),
            {},
        )
    system_memory = hardware.get("systemMemory")
    system_memory = system_memory if isinstance(system_memory, Mapping) else {}
    offload_disk = hardware.get("offloadDisk")
    offload_disk = offload_disk if isinstance(offload_disk, Mapping) else {}
    raw_system = hardware.get("system")
    raw_system = raw_system if isinstance(raw_system, Mapping) else {}
    raw_disk = hardware.get("disk")
    raw_disk = raw_disk if isinstance(raw_disk, Mapping) else {}

    required_vram = int(requirement.get("vramBytes") or 0)
    actual_vram = int(accelerator.get("totalBytes") or accelerator.get("planning_memory_total") or 0)
    memory_kind = str(accelerator.get("memoryKind") or accelerator.get("memory_kind") or "dedicated").lower()
    if memory_kind in {"shared", "unified"} and recipe.get("offloadMode") != "none":
        actual_vram = max(
            actual_vram,
            int(accelerator.get("accessibleTotalBytes") or 0),
            int(accelerator.get("sharedTotalBytes") or 0),
            int(accelerator.get("shared_memory_total") or 0),
            int(accelerator.get("torch_vram_total") or 0),
        )
    required_ram = int(requirement.get("systemRamBytes") or 0)
    actual_ram = int(system_memory.get("totalBytes") or raw_system.get("ram_total") or 0)
    required_disk = int(requirement.get("diskFreeBytes") or 0)
    actual_disk = int(offload_disk.get("freeBytes") or raw_disk.get("free_bytes") or 0)
    missing = []
    if required_vram and actual_vram < required_vram * 0.99:
        missing.append(f"at least {required_vram // _GIB} GiB accelerator-accessible memory")
    if required_ram and actual_ram < required_ram * 0.99:
        missing.append(f"at least {required_ram // _GIB} GiB system RAM")
    if required_disk and actual_disk < required_disk:
        missing.append(f"at least {required_disk // _GIB} GiB free offload disk")
    if missing:
        raise _invalid(
            f"the reviewed {recipe.get('offloadMode')} large-model recipe requires "
            f"{', '.join(missing)}. Select a feasible offload mode before loading the model."
        )


def qualify_huggingface_cluster_expert_runtime(
    payload: Mapping[str, Any],
    *,
    library: Mapping[str, Any],
    runtime_fingerprint: Mapping[str, Any],
    local_models: list[dict[str, Any]] | None,
    optional_runtime_requirement: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a bounded volatile receipt for one exact Expert Cluster recipe."""

    if set(payload) not in (
        {"schemaVersion", "definitionId", "admissionId", "resourceMode", "recipe"},
        {"schemaVersion", "definitionId", "admissionId", "resourceMode", "recipe", "artifactRepo"},
    ):
        raise _invalid("the qualification request shape is invalid.")
    if payload.get("schemaVersion") != CLUSTER_RUNTIME_QUALIFICATION_SCHEMA_VERSION:
        raise _invalid("the qualification request schema is unsupported.")
    if payload.get("resourceMode") != "expert":
        raise _invalid("this endpoint qualifies only explicit Expert recipes.")
    definition_id = str(payload.get("definitionId") or "")
    admission_id = str(payload.get("admissionId") or "")
    if len(definition_id) > 512 or len(admission_id) > 512:
        raise _invalid("the definition or admission identifier is invalid.")
    definitions = [
        definition
        for definition in library.get("definitions", ())
        if isinstance(definition, Mapping) and definition.get("id") == definition_id
    ]
    definition = _single(definitions, "the exact library definition is unavailable.")
    admissions = [
        admission
        for admission in definition.get("executionAdmissions", ())
        if isinstance(admission, Mapping) and admission.get("id") == admission_id
    ]
    admission = _single(admissions, "the exact static execution admission is unavailable.")
    publication = admission.get("publication")
    if (
        admission.get("status") != "admitted"
        or admission.get("claim") != "static_graph_contract_compatible"
        or not isinstance(publication, Mapping)
        or publication.get("readiness") != "graph_qualified"
        or publication.get("insertable") is not True
    ):
        raise _invalid("the selected definition is not graph-qualified.")

    model_type = str(definition.get("pipelineClass") or "")
    mode = str(admission.get("studioMode") or "")
    provider = str(definition.get("provider") or "")
    profiles = list(execution_profiles_for_execution(model_type, mode))
    profile = _single(profiles, "the exact backend execution profile is unavailable.")
    spec = admission.get("studioExecutionSpec")
    public_spec = studio_execution_spec_for_pair(model_type, mode)
    if (
        not isinstance(spec, Mapping)
        or spec.get("executionProfileId") != profile.id
        or public_spec is None
        or public_spec.get("id") != spec.get("id")
        or public_spec.get("contentHash") != spec.get("contentHash")
        or public_spec.get("executionProfileId") != profile.id
        or public_spec.get("loaderModule") != profile.loader_module
        or public_spec.get("loaderAction") != profile.loader_action
        or public_spec.get("executionPath") != profile.execution_path
        or public_spec.get("pipelineClass") != profile.pipeline_class
    ):
        raise _invalid("the execution profile does not match the sealed graph admission.")
    if provider == "diffusers":
        if definition.get("integrationStatus") == "reviewed_diffusers_composite":
            if (
                definition.get("definitionKind") != "studio_execution_composite"
                or profile.pipeline_class != model_type
                or (profile.loader_module, profile.loader_action)
                != _DIRECT_LOADER_CONTRACTS.get(profile.execution_path)
                or admission.get("sealedBindingValues", {}).get("pipelineClass") != profile.pipeline_class
                or admission.get("sealedBindingValues", {}).get("mode") != mode
            ):
                raise _invalid("the execution profile does not match the sealed standard Diffusers composite.")
        elif definition.get("integrationStatus") == "equivalent_standard_route":
            expected_execution_class = _equivalent_standard_execution_class(
                model_type,
                str(definition.get("workflowId") or ""),
            )
            resolved_upstream_class = (
                "WanPipeline" if expected_execution_class == "Wan22Pipeline" else expected_execution_class
            )
            if (
                profile.pipeline_class != expected_execution_class
                or resolved_upstream_class
                not in equivalent_modular_targets(model_type, str(definition.get("workflowId") or ""))
                or (profile.loader_module, profile.loader_action)
                != _DIRECT_LOADER_CONTRACTS.get(profile.execution_path)
                or admission.get("sealedBindingValues", {}).get("pipelineClass") != profile.pipeline_class
                or admission.get("sealedBindingValues", {}).get("mode") != mode
            ):
                raise _invalid("the execution profile does not match the sealed equivalent Diffusers admission.")
        elif (
            definition.get("integrationStatus")
            not in {"reviewed_modiff_contract", "reviewed_modular_workflow_route"}
            or profile.pipeline_class != model_type
            or profile.loader_module != "modules.ModularDiffusers"
            or profile.loader_action != "ModelsLoader"
            or profile.execution_path != "modular-diffusers"
        ):
            raise _invalid("the execution profile does not match the sealed Modular Diffusers admission.")
    elif provider == "transformers":
        if (
            profile.pipeline_class != definition.get("blocksClass")
            or not profile.execution_path.startswith("direct-huggingface-")
            or profile.loader_module not in {"modules.HuggingFaceTransformers", "modules.HuggingFaceSpeech"}
        ):
            raise _invalid("the execution profile does not match the sealed Transformers composite admission.")
    else:
        raise _invalid("the Cluster provider is unsupported.")

    recipe = payload.get("recipe")
    if not isinstance(recipe, Mapping):
        raise _invalid("the Expert recipe is missing.")
    _assert_recipe(recipe, profile, runtime_fingerprint)
    _assert_expert_resources(model_type, mode, recipe, runtime_fingerprint)
    execution_fingerprint, resource_fingerprint = _runtime_identity(runtime_fingerprint)

    selected_optional_requirement = (
        optional_runtime_requirement_for_execution(model_type, mode)
        if optional_runtime_requirement is None
        else dict(optional_runtime_requirement)
    )
    if optional_runtime_requirement_blocks_execution(dict(selected_optional_requirement)):
        state = str(selected_optional_requirement.get("state") or "unavailable").replace("_", " ")
        raise _invalid(f"the reviewed optional runtime is {state}.")
    requirement_profiles = selected_optional_requirement.get("executionProfileIds") or []
    if selected_optional_requirement.get("requiredNow") is True and profile.id not in requirement_profiles:
        raise _invalid("the optional-runtime receipt does not name the exact execution profile.")

    artifact = _effective_reviewed_artifact(admission, profile, payload.get("artifactRepo"))
    artifact_status = artifact_revision_cache_status(
        str(artifact.get("repo") or ""),
        str(artifact.get("revision") or ""),
        local_models,
    )
    if not artifact_status.get("exactRevisionComplete") or artifact_status.get("repairRequired"):
        raise _invalid(str(artifact_status.get("reason") or "the exact artifact is unavailable."))

    dependencies = []
    for dependency in admission.get("modelDependencies", ()):
        if not isinstance(dependency, Mapping):
            raise _invalid("a sealed model dependency is malformed.")
        status = artifact_revision_cache_status(
            str(dependency.get("repo") or ""),
            str(dependency.get("revision") or ""),
            local_models,
        )
        if not status.get("exactRevisionComplete") or status.get("repairRequired"):
            raise _invalid(str(status.get("reason") or "an exact dependency is unavailable."))
        dependencies.append(
            {
                "id": dependency.get("id"),
                "kind": dependency.get("kind"),
                "artifactStatus": _public_artifact_status(status),
            }
        )

    checked_at = int(time.time() * 1000)
    receipt_core = {
        "definitionId": definition_id,
        "admissionId": admission_id,
        "executionProfileId": profile.id,
        "modelType": model_type,
        "mode": mode,
        "pipelineClass": profile.pipeline_class,
        "loaderModule": profile.loader_module,
        "loaderAction": profile.loader_action,
        "executionPath": profile.execution_path,
        "runtimeFingerprint": execution_fingerprint,
        "resourceFingerprint": resource_fingerprint,
        "artifactStatus": _public_artifact_status(artifact_status),
        "dependencies": dependencies,
        "optionalRuntimeRequirement": deepcopy(dict(selected_optional_requirement)),
        "recipe": deepcopy(dict(recipe)),
    }
    qualification_fingerprint = hashlib.sha256(
        json.dumps(receipt_core, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return {
        "schemaVersion": CLUSTER_RUNTIME_QUALIFICATION_SCHEMA_VERSION,
        "claim": "expert_cluster_runtime_qualified",
        "publicationExecutable": False,
        "checkedAt": checked_at,
        "qualificationFingerprint": f"sha256:{qualification_fingerprint}",
        **receipt_core,
    }


def _candidate_artifact(candidate: Mapping[str, Any]) -> str | None:
    resolution = candidate.get("artifactResolution")
    resolution = resolution if isinstance(resolution, Mapping) else {}
    resolved = resolution.get("resolved")
    resolved = resolved if isinstance(resolved, Mapping) else {}
    install_target = candidate.get("installTarget")
    install_target = install_target if isinstance(install_target, Mapping) else {}
    values = {
        str(value)
        for value in (
            candidate.get("resolvedArtifact"),
            candidate.get("artifact"),
            candidate.get("modelRepo"),
            resolved.get("repo"),
            install_target.get("repo"),
        )
        if value
    }
    return next(iter(values)) if len(values) == 1 else None


def _auto_recipe(candidate: Mapping[str, Any], form: Mapping[str, Any]) -> dict[str, Any]:
    offload_mode = str(candidate.get("offloadMode") or form.get("offloadMode") or "none")
    return {
        "device": str(form.get("device") or "cpu"),
        "dtype": str(candidate.get("dtype") or form.get("dtype") or "float32"),
        "quantizationMode": str(
            candidate.get("quantizationMode") or form.get("quantizationMode") or "none"
        ),
        "autoOffload": bool(
            candidate.get("autoOffload")
            if isinstance(candidate.get("autoOffload"), bool)
            else offload_mode != "none"
        ),
        "offloadMode": offload_mode,
    }


def _project_instance_value_to_form(value: Any, current: Any) -> Any:
    """Mirror the client's narrow scalar projection into a Studio form."""

    if isinstance(current, (int, float)) and not isinstance(current, bool) and isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return value
        if not math.isfinite(number):
            return value
        return int(number) if isinstance(current, int) and number.is_integer() else number
    if isinstance(current, bool) and isinstance(value, str):
        if value == "true":
            return True
        if value == "false":
            return False
    return value


def _same_json_value(left: Any, right: Any) -> bool:
    try:
        return json.dumps(left, sort_keys=True, separators=(",", ":"), ensure_ascii=False) == json.dumps(
            right,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    except (TypeError, ValueError):
        return False


def qualify_huggingface_cluster_auto_authority(
    payload: Mapping[str, Any],
    *,
    library: Mapping[str, Any],
    runtime_fingerprint: Mapping[str, Any],
    local_models: list[dict[str, Any]] | None,
    data_dir: str,
    registered_definition_pins: Mapping[str, tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Issue one short-lived Auto receipt for an exact registered V2 instance.

    The backend strictly validates the complete instance and recomputes every
    receipt hash. It also pins the compiled definition to an exact admitted
    compiler output and rejects structural/interface customization. The
    browser therefore cannot bind a valid resource plan to substituted graph,
    interface, values, admission, or artifact material.
    """

    if set(payload) != _AUTO_AUTHORITY_KEYS or payload.get("schemaVersion") != 1:
        raise _auto_invalid("the authority request shape is invalid.")
    try:
        instance = validate_block_instance_v2(payload.get("instance"))
    except ValueError as error:
        raise _auto_invalid(f"the V2 Block instance is invalid: {error}") from error
    snapshot = instance["definitionSnapshot"]
    source = snapshot["source"]
    if source.get("kind") not in {"diffusers_catalog", "transformers_catalog"}:
        raise _auto_invalid("authority is available only for an exact registered catalog Block.")
    catalog_definition_id = str(source.get("manifestDefinitionId") or "")
    admission_id = str(source.get("executionAdmissionId") or "")
    repository = str(source.get("repository") or "")
    repository_revision = str(source.get("repositoryRevision") or "")
    if any(
        _BLOCK_RECEIPT_TOKEN.fullmatch(value) is None
        for value in (catalog_definition_id, admission_id, snapshot["definitionId"], snapshot["contentHash"])
    ):
        raise _auto_invalid("the registered V2 Block provenance is incomplete or malformed.")
    if snapshot["definitionId"] != admission_id:
        raise _auto_invalid("the V2 Block definition is not bound to the exact execution admission.")
    pinned_definitions = (
        REGISTERED_BLOCK_V2_DEFINITION_PINS
        if registered_definition_pins is None
        else registered_definition_pins
    )
    expected_definition_pin = pinned_definitions.get(admission_id)
    if expected_definition_pin is None or snapshot["contentHash"] != expected_definition_pin[0]:
        raise _auto_invalid("the V2 Block definition does not match the backend-pinned compiler output.")
    if block_definition_canonical_sha256_v2(snapshot) != expected_definition_pin[1]:
        raise _auto_invalid(
            "the V2 Block definition canonical material does not match the backend-pinned SHA-256."
        )
    if (
        instance["customization"]["state"] == "structure_changed"
        or instance["effectiveGraph"]["graphHash"] != snapshot["graph"]["graphHash"]
        or instance["effectiveInterface"]["effectiveInterfaceHash"]
        != instance["effectiveInterface"]["baseInterfaceHash"]
    ):
        raise _auto_invalid(
            "Auto does not authorize structurally or publicly customized registered Blocks; "
            "use Expert/manual execution for that exact concrete graph."
        )
    form = payload.get("form")
    if not isinstance(form, Mapping):
        raise _auto_invalid("the authority form is missing.")
    missing_resource_fields = _AUTO_RESOURCE_FORM_FIELDS - set(form)
    if missing_resource_fields:
        raise _auto_invalid(
            "the authority form omits resource-impacting field(s): "
            f"{', '.join(sorted(missing_resource_fields))}."
        )

    definitions = [
        definition
        for definition in library.get("definitions", ())
        if isinstance(definition, Mapping) and definition.get("id") == catalog_definition_id
    ]
    if len(definitions) != 1:
        raise _auto_invalid("the exact library definition is unavailable.")
    definition = definitions[0]
    if definition.get("contentHash") != source.get("manifestContentHash"):
        raise _auto_invalid("the registered catalog definition hash is stale.")
    if (
        definition.get("libraryRevision") != source.get("libraryRevision")
        or definition.get("pipelineClass") != source.get("pipelineClass")
        or definition.get("workflowId") != source.get("workflow")
        or definition.get("provider") != source.get("library")
        or definition.get("publisher") != source.get("provider")
        or definition.get("blocksClass") != source.get("blocksClass")
    ):
        raise _auto_invalid(
            "the embedded V2 Block source does not match the exact registered catalog definition."
        )
    admissions = [
        admission
        for admission in definition.get("executionAdmissions", ())
        if isinstance(admission, Mapping) and admission.get("id") == admission_id
    ]
    if len(admissions) != 1:
        raise _auto_invalid("the exact static execution admission is unavailable.")
    admission = admissions[0]
    artifact = admission.get("artifact")
    if (
        not isinstance(artifact, Mapping)
        or artifact.get("repo") != repository
        or artifact.get("revision") != repository_revision
    ):
        raise _auto_invalid("the requested artifact does not match the immutable execution admission.")
    profiles = list(
        execution_profiles_for_execution(
            str(definition.get("pipelineClass") or ""),
            str(admission.get("studioMode") or ""),
        )
    )
    profile = _single(profiles, "the exact backend execution profile is unavailable.")
    selected_artifact = _effective_reviewed_artifact(
        admission,
        profile,
        instance["values"].get("modelVariant"),
    )
    repository = selected_artifact["repo"]
    repository_revision = selected_artifact["revision"]
    # Older registered Blocks did not project an explicit modelRepo. They
    # remain valid only for their sealed baseline artifact; current Blocks
    # always send the selected same-family variant explicitly.
    if str(form.get("modelRepo") or repository) != repository:
        raise _auto_invalid("the planner form does not match the selected model variant.")
    if form.get("modelType") != definition.get("pipelineClass") or form.get("mode") != admission.get("studioMode"):
        raise _auto_invalid("the planner form targets another model or workflow mode.")
    source_by_input = {
        str(binding.get("input")): str(binding.get("bindingSource"))
        for binding in admission.get("instanceInputBindings", ())
        if isinstance(binding, Mapping) and binding.get("input") and binding.get("bindingSource")
    }
    values = instance["values"]
    for control in instance["effectiveInterface"]["controls"]:
        control_id = control["controlId"]
        if control_id in values:
            value = values[control_id]
        elif "defaultValue" in control:
            value = control["defaultValue"]
        else:
            continue
        form_key = source_by_input.get(control_id, control_id)
        if form_key not in form:
            continue
        projected = _project_instance_value_to_form(value, form[form_key])
        if not _same_json_value(projected, form[form_key]):
            raise _auto_invalid(
                f"the planner form field {form_key} does not match the validated V2 instance value."
            )

    try:
        plan = build_auto_resource_plan(
            {"form": dict(form)},
            runtime_fingerprint=dict(runtime_fingerprint),
            local_models=local_models,
            data_dir=data_dir,
        )
    except (TypeError, ValueError) as error:
        raise _auto_invalid(f"the backend planner rejected the request: {error}") from error
    candidate = plan.get("selectedCandidate")
    if plan.get("canAutoRun") is not True or not isinstance(candidate, Mapping):
        raise _auto_invalid(str(plan.get("blockingReason") or "the backend planner has no ready exact candidate."))
    if _candidate_artifact(candidate) != repository:
        raise _auto_invalid("the selected candidate substitutes another artifact.")
    resolution = candidate.get("artifactResolution")
    resolution = resolution if isinstance(resolution, Mapping) else {}
    resolved = resolution.get("resolved")
    resolved = resolved if isinstance(resolved, Mapping) else {}
    candidate_revision = candidate.get("artifactRevision") or resolved.get("revision")
    if candidate_revision and candidate_revision != repository_revision:
        raise _auto_invalid("the selected candidate resolves another artifact revision.")
    requested_recipe = {
        "device": str(form.get("device") or "cpu"),
        "dtype": str(form.get("dtype") or "float32"),
        "quantizationMode": str(form.get("quantizationMode") or "none"),
        "autoOffload": form.get("autoOffload"),
        "offloadMode": str(form.get("offloadMode") or "none"),
    }
    candidate_recipe = _auto_recipe(candidate, form)
    if not isinstance(requested_recipe["autoOffload"], bool) or candidate_recipe != requested_recipe:
        raise _auto_invalid(
            "the selected candidate requires a different execution recipe; "
            "apply that plan to the Block instance before requesting authority."
        )

    try:
        qualification = qualify_huggingface_cluster_expert_runtime(
            {
                "schemaVersion": 1,
                "definitionId": catalog_definition_id,
                "admissionId": admission_id,
                "artifactRepo": repository,
                "resourceMode": "expert",
                "recipe": candidate_recipe,
            },
            library=library,
            runtime_fingerprint=runtime_fingerprint,
            local_models=local_models,
            optional_runtime_requirement=(
                candidate.get("optionalRuntimeRequirement")
                if isinstance(candidate.get("optionalRuntimeRequirement"), Mapping)
                else None
            ),
        )
    except ValueError as error:
        detail = str(error)
        prefix = "Cannot qualify Hugging Face Cluster Node Expert execution: "
        if detail.startswith(prefix):
            detail = detail[len(prefix) :]
        raise _auto_invalid(f"the exact runtime qualification failed: {detail}") from error
    candidate_contract = candidate.get("studioExecutionSpecContract")
    candidate_contract = candidate_contract if isinstance(candidate_contract, Mapping) else {}
    expected_contract = admission.get("studioExecutionSpec")
    expected_contract = expected_contract if isinstance(expected_contract, Mapping) else {}
    candidate_dependencies = candidate.get("modelDependencies")
    candidate_dependencies = candidate_dependencies if isinstance(candidate_dependencies, list) else []
    qualified_dependencies = [
        {
            "id": dependency.get("id"),
            "kind": dependency.get("kind"),
            "repo": (dependency.get("artifactStatus") or {}).get("repo"),
            "revision": (dependency.get("artifactStatus") or {}).get("revision"),
        }
        for dependency in qualification["dependencies"]
    ]
    if (
        candidate.get("modelType") != qualification["modelType"]
        or candidate.get("mode") != qualification["mode"]
        or candidate.get("executionProfileId") != qualification["executionProfileId"]
        or candidate.get("loaderModule") != qualification["loaderModule"]
        or candidate.get("loaderAction") != qualification["loaderAction"]
        or candidate.get("executionPath") != qualification["executionPath"]
        or candidate.get("pipelineClass") != qualification["pipelineClass"]
        or candidate_contract.get("id") != expected_contract.get("id")
        or candidate_contract.get("contentHash") != expected_contract.get("contentHash")
        or candidate_contract.get("executionProfileId") != expected_contract.get("executionProfileId")
        or candidate_dependencies != qualified_dependencies
    ):
        raise _auto_invalid("the selected candidate does not match the sealed execution admission.")

    issued_at = datetime.now(timezone.utc)
    artifact_revisions = {repository: repository_revision}
    artifact_revisions.update(
        {
            str(dependency["repo"]): str(dependency["revision"])
            for dependency in qualified_dependencies
            if dependency.get("repo") and dependency.get("revision")
        }
    )
    return {
        "kind": "auto",
        "definitionId": admission_id,
        "definitionContentHash": snapshot["contentHash"],
        "effectiveGraphHash": instance["effectiveGraph"]["graphHash"],
        "executionParameterHash": block_execution_parameter_hash_v2(instance),
        "artifactRevisions": artifact_revisions,
        "admissionId": admission_id,
        # BlockInstanceV2 deliberately uses one cross-runtime UTC timestamp
        # representation: either whole seconds or exactly milliseconds.  The
        # default Python isoformat() emits six fractional digits and therefore
        # produced receipts that neither the TypeScript nor Python V2 schema
        # validator could attach to an instance.  Emit the schema's canonical
        # millisecond precision instead of asking either consumer to weaken
        # the persisted contract.
        "issuedAt": issued_at.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "expiresAt": (issued_at + timedelta(minutes=5))
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z"),
    }
