from types import SimpleNamespace

import pytest

from modiff.modular_runtime_diagnostics import modular_execution_diagnostics


def test_shape_failure_names_block_and_actual_shapes_without_dumping_values():
    block = SimpleNamespace(inputs=[SimpleNamespace(name="latents"), SimpleNamespace(name="prompt")])
    state = {"latents": SimpleNamespace(shape=(1, 32, 16)), "prompt": "PRIVATE PROMPT"}
    with pytest.raises(ValueError, match=r"denoise/transformer.*mat1 and mat2.*latents=\(1, 32, 16\)") as result:
        with modular_execution_diagnostics(block, state, path=["denoise", "transformer"]):
            raise RuntimeError("mat1 and mat2 shapes cannot be multiplied")
    assert "PRIVATE PROMPT" not in str(result.value)
    assert isinstance(result.value.__cause__, RuntimeError)


def test_resource_failure_keeps_original_category_and_success_keeps_state_unchanged():
    block = SimpleNamespace(inputs=[])
    state = {"value": 42}
    with modular_execution_diagnostics(block, state, path=["decode"]):
        pass
    assert state == {"value": 42}
    original = MemoryError("allocation failed")
    with pytest.raises(MemoryError) as result:
        with modular_execution_diagnostics(block, state, path=["decode"]):
            raise original
    assert result.value is original
    assert "decode" in original.__notes__[0]
