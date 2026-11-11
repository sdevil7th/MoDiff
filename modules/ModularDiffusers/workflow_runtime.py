"""Runtime helpers shared by split official Modular Diffusers workflows.

The upstream ``ModularPipeline`` receives one ``torch.Generator`` object for a
whole call. A visual split workflow must continue that random stream from its
sealed process-local block state. Recreating a generator from the
same seed at every stage resets the random stream and is not equivalent to the
upstream pipeline.
"""

from collections.abc import Mapping
from contextlib import contextmanager

import torch

from .modular_utils import modular_generator_from_seed, normalize_modular_seed


@contextmanager
def workflow_stage_progress(node, blocks):
    """Bridge official nested loops to the existing node progress/Stop boundary."""
    from diffusers.modular_pipelines import LoopSequentialPipelineBlocks
    from .reviewed_blocks import _ReviewedNodeProgressBar

    pending, seen, patched = [blocks], set(), []
    try:
        while pending:
            block = pending.pop()
            if id(block) in seen:
                continue
            seen.add(id(block))
            if len(seen) > 256:
                raise ValueError('The selected workflow stage exceeds its bounded block tree.')
            if isinstance(block, LoopSequentialPipelineBlocks):
                patched.append((block, 'progress_bar' in vars(block), vars(block).get('progress_bar')))
                block.progress_bar = lambda iterable=None, total=None: _ReviewedNodeProgressBar(
                    node, iterable=iterable, total=total)
            children = getattr(block, 'sub_blocks', None)
            if isinstance(children, Mapping):
                pending.extend(children.values())
        yield
    finally:
        for block, had_override, original in reversed(patched):
            if had_override:
                block.progress_bar = original
            else:
                del block.progress_bar


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
    """Continue from the preceding stage without advancing its cached snapshot.

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
    # ModularPipeline deep-copies ``state``, but then applies keyword arguments
    # without copying them. Passing the cached Generator here would advance the
    # preceding stage's snapshot, changing a retry or a downstream-only edit.
    # Clone the *current* stream position, not just its initial seed.
    return existing.clone_state()
