"""Attach exact block/shape context without dumping prompts, tensors or credentials."""

from contextlib import contextmanager


@contextmanager
def modular_execution_diagnostics(block, state, *, path):
    try:
        yield
    except Exception as error:
        label = f"Modular block {'/'.join(path)} ({type(block).__name__})"
        if isinstance(error, MemoryError) or any(cls.__name__ == "OutOfMemoryError" for cls in type(error).__mro__):
            error.add_note(label)
            raise
        shapes = []
        for spec in getattr(block, "inputs", ()):
            if not spec.name:
                continue
            value = state.get(spec.name)
            shape = getattr(value, "shape", None)
            if shape is not None:
                shapes.append(f"{spec.name}={tuple(shape)}")
            if len(shapes) == 8:
                break
        details = f" Input shapes: {', '.join(shapes)}." if shapes else ""
        raise ValueError(
            f"{label}: {error}.{details} Inspect this block's connected inputs and component requirements; the saved graph was not repaired or changed."
        ) from error
