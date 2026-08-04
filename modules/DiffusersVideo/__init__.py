from .main import *  # noqa: F403


def _registry_entry(node_class, *, hidden=False):
    return {
        "type": "custom",
        "label": node_class.label,
        "category": node_class.category,
        "description": node_class.__doc__ or "",
        "resizable": getattr(node_class, "resizable", False),
        "skipParamsCheck": getattr(node_class, "skipParamsCheck", False),
        "style": getattr(node_class, "style", {}),
        "params": node_class.params,
        "hidden": hidden,
    }


MODULE_MAP = {
    node_class.__name__: _registry_entry(node_class)
    for node_class in (
        LoadPipeline,  # noqa: F405
        Generate,  # noqa: F405
        GenerateVideoAudio,  # noqa: F405
        BuildShotJobs,  # noqa: F405
        GenerateShotJob,  # noqa: F405
        GenerateSequence,  # noqa: F405
        PlanLongVideo,  # noqa: F405
    )
}

# Keep the old action executable for imported graphs, but do not advertise a
# model-specific node in the node library.
MODULE_MAP["GenerateLTX2"] = _registry_entry(GenerateLTX2, hidden=True)  # noqa: F405
