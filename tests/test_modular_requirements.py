"""Declared compatibility does not imply live tensor or model qualification."""

import pytest
import importlib.util
from types import SimpleNamespace

from modiff.modular_requirements import assess_component_requirements, validate_runtime_component_requirements


@pytest.mark.parametrize("available,creation,status,code", [
    ("Scheduler", "from_pretrained", "compatible", None),
    ("OtherScheduler", "from_pretrained", "incompatible", "component_type_conflict"),
    (None, "from_pretrained", "unresolved", "component_missing"),
    (None, "from_config", "compatible", None),
])
def test_component_requirement_classification(available, creation, status, code):
    source = {"components": [{"name": "scheduler", "type": "Scheduler", "creationMethod": creation}]}
    destination = {"components": [{"name": "scheduler", "type": available}] if available else []}
    result = assess_component_requirements(source, destination)
    assert result["status"] == status
    assert [issue["code"] for issue in result["issues"]] == ([code] if code else [])
    if code:
        assert result["issues"][0] == {"code": code, "component": "scheduler", "requiredType": "Scheduler", "actualType": available}


def test_actual_components_accept_subclasses_and_name_missing_or_wrong_types():
    class Scheduler:
        pass

    class CompatibleScheduler(Scheduler):
        pass

    block = SimpleNamespace(expected_components=[SimpleNamespace(name="scheduler",type_hint=Scheduler)])
    pipeline = SimpleNamespace(scheduler=CompatibleScheduler())
    validate_runtime_component_requirements(block,pipeline,path=["denoise","timesteps"])
    pipeline.scheduler = object()
    with pytest.raises(ValueError,match="denoise/timesteps.*component 'scheduler'.*expected.*Scheduler.*received builtins.object"):
        validate_runtime_component_requirements(block,pipeline,path=["denoise","timesteps"])
    pipeline.scheduler = None
    with pytest.raises(ValueError,match="component 'scheduler': required component is missing"):
        validate_runtime_component_requirements(block,pipeline,path=["denoise","timesteps"])


@pytest.mark.skipif(importlib.util.find_spec('transformers') is None, reason='Optional runtime not active')
def test_upstream_auto_factories_accept_their_concrete_component_instances_only():
    from transformers import (AutoProcessor, AutoTokenizer, AutoImageProcessor,
                              AutoFeatureExtractor, PixtralProcessor, LlamaTokenizer,
                              CLIPImageProcessor, WhisperFeatureExtractor)
    for factory, concrete in ((AutoProcessor, PixtralProcessor), (AutoTokenizer, LlamaTokenizer),
                              (AutoImageProcessor, CLIPImageProcessor),
                              (AutoFeatureExtractor, WhisperFeatureExtractor)):
        block = SimpleNamespace(expected_components=[SimpleNamespace(name='tokenizer', type_hint=factory)])
        # No weights/downloads or model initialization: genuine upstream classes.
        value = concrete.__new__(concrete)
        validate_runtime_component_requirements(block, SimpleNamespace(tokenizer=value), path=['encoder'])
        with pytest.raises(ValueError, match='tokenizer.*expected.*received'):
            validate_runtime_component_requirements(block, SimpleNamespace(tokenizer=object()), path=['encoder'])
    wrong_factory = type('AutoProcessor', (), {})
    block = SimpleNamespace(expected_components=[SimpleNamespace(name='tokenizer', type_hint=wrong_factory)])
    with pytest.raises(ValueError, match='expected.*AutoProcessor'):
        validate_runtime_component_requirements(block, SimpleNamespace(tokenizer=PixtralProcessor.__new__(PixtralProcessor)), path=['encoder'])
