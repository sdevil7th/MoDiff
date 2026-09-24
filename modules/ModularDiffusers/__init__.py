# Derived from cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f; modified by MoDiff.
from diffusers import ComponentsManager

from modiff.diffusers_offload import (  # noqa: F401 - preloaded for AST registry evaluation
    OFFLOAD_MODE_GROUP_CPU,
    OFFLOAD_MODE_GROUP_DISK,
    OFFLOAD_MODE_MODEL_CPU,
    OFFLOAD_MODE_NONE,
    offload_mode_param,
)

from .modular_utils import (
    FLUX_LAYER_BLOCK_OPTIONS,
    QWEN_IMAGE_LAYER_BLOCK_OPTIONS,
    SDXL_LAYER_BLOCK_OPTIONS,
    ModiffPipelineRegistry,
    get_modular_guider_options,
    get_modular_layer_block_options,
    get_modular_node_action_options,
    get_modular_scheduler_options,
    get_all_model_types,
)


MESSAGE_DURATION = 5000
components = ComponentsManager()


# Global singleton registry instance for the package
MODULAR_REGISTRY = ModiffPipelineRegistry()

MODULE_PARSE = [
    "adapters",
    "controlnet",
    "denoise",
    "embeddings",
    "guiders",
    "ip_adapter",
    "latents",
    "loaders",
    "schedulers",
    "dynamic_node",
    "workflow_blocks",
    "reviewed_blocks",
]

SDXL_BLOCKS = list(SDXL_LAYER_BLOCK_OPTIONS)
QWEN_IMAGE_BLOCKS = list(QWEN_IMAGE_LAYER_BLOCK_OPTIONS)
FLUX_BLOCKS = list(FLUX_LAYER_BLOCK_OPTIONS)
MODULAR_LAYER_BLOCK_OPTIONS = get_modular_layer_block_options()
MODULAR_GUIDER_OPTIONS = get_modular_guider_options()
MODULAR_SCHEDULER_OPTIONS = get_modular_scheduler_options()
MODULAR_NODE_ACTION_OPTIONS = get_modular_node_action_options()
MODULAR_TEXT_ENCODER_OPTIONS = MODULAR_NODE_ACTION_OPTIONS.get("text_encoder", {})
MODULAR_IMAGE_ENCODER_OPTIONS = MODULAR_NODE_ACTION_OPTIONS.get("image_encoder", {})
MODULAR_DENOISE_OPTIONS = MODULAR_NODE_ACTION_OPTIONS.get("denoise", {})
MODULAR_IP_ADAPTER_OPTIONS = MODULAR_NODE_ACTION_OPTIONS.get("ip_adapter", {})
MODULAR_DECODER_OPTIONS = MODULAR_NODE_ACTION_OPTIONS.get("decoder", {})
MODULAR_VAE_ENCODER_OPTIONS = MODULAR_NODE_ACTION_OPTIONS.get("vae_encoder", {})
MODULAR_CONTROLNET_OPTIONS = MODULAR_NODE_ACTION_OPTIONS.get("controlnet", {})
MODULAR_MODEL_TYPE_OPTIONS = get_all_model_types(include_contract_only=True)

# The static node-registry parser resolves schema constants against this
# package object. Export reviewed dynamic options so the public /nodes
# contract contains mappings instead of unresolved identifier strings.
from .guiders import GUIDER_OPTIONS as GUIDER_OPTIONS  # noqa: E402,F401
from .reviewed_blocks import (  # noqa: E402,F401
    _REVIEWED_BLOCK_INPUT_PARAMS as _REVIEWED_BLOCK_INPUT_PARAMS,
    _REVIEWED_BLOCK_INPUT_ALIASES as _REVIEWED_BLOCK_INPUT_ALIASES,
    _REVIEWED_BLOCK_OUTPUT_PARAMS as _REVIEWED_BLOCK_OUTPUT_PARAMS,
    _REVIEWED_LOOP_PORT_PARAMS as _REVIEWED_LOOP_PORT_PARAMS,
)
