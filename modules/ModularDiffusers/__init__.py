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
    get_modular_scheduler_options,
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
]

SDXL_BLOCKS = list(SDXL_LAYER_BLOCK_OPTIONS)
QWEN_IMAGE_BLOCKS = list(QWEN_IMAGE_LAYER_BLOCK_OPTIONS)
FLUX_BLOCKS = list(FLUX_LAYER_BLOCK_OPTIONS)
MODULAR_LAYER_BLOCK_OPTIONS = get_modular_layer_block_options()
MODULAR_GUIDER_OPTIONS = get_modular_guider_options()
MODULAR_SCHEDULER_OPTIONS = get_modular_scheduler_options()

# The static node-registry parser resolves schema constants against this
# package object. Export reviewed dynamic options so the public /nodes
# contract contains mappings instead of unresolved identifier strings.
from .guiders import GUIDER_OPTIONS as GUIDER_OPTIONS  # noqa: E402,F401
