from .main import *  # noqa: F403


def _registry_entry(node_class):
    """Build the runtime registry shape for inherited Diffusers image nodes."""

    return {
        "type": "custom",
        "label": getattr(node_class, "label", None),
        "category": getattr(node_class, "category", "default"),
        "description": node_class.__doc__ or "",
        "resizable": getattr(node_class, "resizable", False),
        "skipParamsCheck": getattr(node_class, "skipParamsCheck", False),
        "style": getattr(node_class, "style", {}),
        "params": node_class.params,
    }


# The global AST scanner intentionally recognizes direct NodeBase subclasses only.
# These thin facades inherit their execution behavior but expose distinct graph
# contracts, so register exactly these existing subclasses through the supported
# precomputed-map path.
MODULE_MAP = {
    node_class.__name__: _registry_entry(node_class)
    for node_class in (Edit, ControlEdit, Inpaint, ControlInpaint, ControlGenerate, OutpaintCanvas)  # noqa: F405
}
