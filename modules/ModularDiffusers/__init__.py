# Derived from cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f; modified by MoDiff.
from diffusers import ComponentsManager

from modiff.diffusers_offload import (  # noqa: F401 - preloaded for AST registry evaluation
    OFFLOAD_MODE_GROUP_CPU,
    OFFLOAD_MODE_GROUP_DISK,
    OFFLOAD_MODE_MODEL_CPU,
    OFFLOAD_MODE_NONE,
    offload_mode_param,
)

from .modular_utils import ModiffPipelineRegistry


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
    "latents",
    "loaders",
    "schedulers",
    "dynamic_node",
]

SDXL_BLOCKS = [
    "down_blocks.1.attentions.0.transformer_blocks",
    "down_blocks.1.attentions.1.transformer_blocks",
    "down_blocks.2.attentions.0.transformer_blocks",
    "down_blocks.2.attentions.1.transformer_blocks",
    "mid_block.attentions.0.transformer_blocks",
    "up_blocks.0.attentions.0.transformer_blocks",
    "up_blocks.0.attentions.1.transformer_blocks",
    "up_blocks.0.attentions.2.transformer_blocks",
    "up_blocks.1.attentions.0.transformer_blocks",
    "up_blocks.1.attentions.1.transformer_blocks",
    "up_blocks.1.attentions.2.transformer_blocks",
]

QWEN_IMAGE_BLOCKS = ["transformer_blocks"]

FLUX_BLOCKS = ["transformer_blocks", "single_transformer_blocks"]

# The static node-registry parser resolves schema constants against this
# package object. Export the Guider options so the public /nodes contract
# contains the actual mapping instead of the unresolved identifier string.
from .guiders import GUIDER_OPTIONS as GUIDER_OPTIONS  # noqa: E402,F401
