"""Read actual captured loader policy; never import a model to plan memory."""

import sys
from types import ModuleType

import pytest

from modiff.runtime_environment import diffusers_parallel_loading_enabled


def loaded_flag(monkeypatch, name, value):
    module = ModuleType(name)
    if value is not None:
        module.HF_ENABLE_PARALLEL_LOADING = value
    monkeypatch.setitem(sys.modules, name, module)


@pytest.mark.parametrize("env,expected", [("0", False), ("YES", True), ("true", True),
                                         (" FALSE ", False), ("y", False), ("", False)])
def test_not_yet_imported_loader_uses_exact_upstream_boolean_tokens(monkeypatch, env, expected):
    monkeypatch.delitem(sys.modules, "diffusers.models.modeling_utils", raising=False)
    monkeypatch.delitem(sys.modules, "diffusers.utils.constants", raising=False)
    monkeypatch.setenv("HF_ENABLE_PARALLEL_LOADING", env)
    assert diffusers_parallel_loading_enabled() is expected


@pytest.mark.parametrize("captured,constant,env", [(True, False, "0"), (False, True, "YES")])
def test_actual_model_mixin_copy_takes_precedence_over_later_constant_and_env_edits(monkeypatch, captured, constant, env):
    loaded_flag(monkeypatch, "diffusers.models.modeling_utils", captured)
    loaded_flag(monkeypatch, "diffusers.utils.constants", constant)
    monkeypatch.setenv("HF_ENABLE_PARALLEL_LOADING", env)
    assert diffusers_parallel_loading_enabled() is captured


def test_loaded_constants_bind_the_next_model_mixin_import(monkeypatch):
    monkeypatch.delitem(sys.modules, "diffusers.models.modeling_utils", raising=False)
    loaded_flag(monkeypatch, "diffusers.utils.constants", True)
    monkeypatch.setenv("HF_ENABLE_PARALLEL_LOADING", "0")
    assert diffusers_parallel_loading_enabled() is True


@pytest.mark.parametrize("value", [None, "0", 0])
def test_unknown_loaded_contract_cannot_grant_serial_narrowing(monkeypatch, value):
    loaded_flag(monkeypatch, "diffusers.models.modeling_utils", value)
    monkeypatch.setenv("HF_ENABLE_PARALLEL_LOADING", "0")
    assert diffusers_parallel_loading_enabled() is True


def test_policy_read_imports_no_model_runtime(monkeypatch):
    import builtins

    def forbidden_import(*args, **kwargs):
        raise AssertionError("Planning imported a runtime")

    monkeypatch.setattr(builtins, "__import__", forbidden_import)
    assert type(diffusers_parallel_loading_enabled()) is bool
