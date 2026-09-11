"""Runtime helpers shared by split official Modular Diffusers workflows.

The upstream ``ModularPipeline`` receives one ``torch.Generator`` object for a
whole call.  A visual split workflow must therefore continue that same object
from its sealed process-local block state.  Recreating a generator from the
same seed at every stage resets the random stream and is not equivalent to the
upstream pipeline.
"""

from collections.abc import Mapping

import torch

from .modular_utils import modular_generator_from_seed, normalize_modular_seed


def _workflow_state_value(state, name):
    if state is None:
        return None
    if isinstance(state, Mapping):
        return state.get(name)
    getter = getattr(state, "get", None)
    if callable(getter):
        return getter(name)
    raise TypeError("A Modular workflow continuation state must provide keyed values.")


def continuation_generator_from_seed(seed, pipeline, state=None):
    """Create the workflow generator once, then reuse it from block state.

    ``state`` is issued by MoDiff's process-local workflow-state authority, so
    accepting its generator does not expose a graph-supplied Python object.
    The seed and execution device are nevertheless checked so a downstream
    edited node cannot silently continue a different random stream.
    """

    normalized_seed = normalize_modular_seed(seed)
    existing = _workflow_state_value(state, "generator")
    if existing is None:
        return modular_generator_from_seed(normalized_seed, pipeline)
    if not isinstance(existing, torch.Generator):
        raise TypeError("The Modular workflow continuation contains an invalid generator.")
    if existing.initial_seed() != normalized_seed:
        raise ValueError(
            "The Modular workflow seed changed after sampling began. Start a new run from the first "
            "generator-using block."
        )

    execution_device = getattr(pipeline, "_execution_device", None)
    if execution_device is None:
        raise RuntimeError(
            "The Modular workflow could not resolve its execution device for deterministic sampling."
        )
    if torch.device(existing.device) != torch.device(execution_device):
        raise ValueError("The Modular workflow generator belongs to a different execution device.")
    return existing
