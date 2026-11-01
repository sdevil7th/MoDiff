"""Temporary progress binding for fresh official Modular workflow blocks."""
from collections.abc import Mapping
from contextlib import contextmanager


@contextmanager
def modular_loop_progress(node, pipeline):
    # Called only after the builtin workflow's existing source/owner checks.
    # Discovery and Auto inspection never construct or walk executable blocks.
    from diffusers.modular_pipelines.modular_pipeline import LoopSequentialPipelineBlocks
    from .reviewed_blocks import _ReviewedNodeProgressBar

    pending = [getattr(pipeline, 'blocks', None)]
    seen, loops = set(), []
    while pending:
        block = pending.pop()
        if id(block) in seen:
            continue
        seen.add(id(block))
        if len(seen) > 256:
            raise ValueError('Official workflow progress exceeds the finite block bound.')
        if isinstance(block, LoopSequentialPipelineBlocks):
            loops.append(block)
        children = getattr(block, 'sub_blocks', None)
        if isinstance(children, Mapping):
            pending.extend(children.values())

    bindings = []
    try:
        if getattr(node, '_interrupt', False):
            raise InterruptedError('Execution interrupted before the Modular Diffusers loop.')
        for block in loops:
            bindings.append((block, 'progress_bar' in vars(block), vars(block).get('progress_bar')))
            block.progress_bar = lambda iterable=None, total=None: _ReviewedNodeProgressBar(
                node, iterable=iterable, total=total)
        yield
    finally:
        for block, had_original, original in reversed(bindings):
            if had_original:
                block.progress_bar = original
            else:
                del block.progress_bar
