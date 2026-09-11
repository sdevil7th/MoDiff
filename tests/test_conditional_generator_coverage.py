"""Companion snapshots keep coverage when a model leaves a routing registry."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None,
                    reason="requires the staged optional Transformers runtime")
def test_conditional_generator_covers_reviewed_classes_and_constructor_config():
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    with patch.object(sys, "path", [scripts, *sys.path]):
        import generate_modular_conditional_contracts as generator
        from modiff.modular_workflow_discovery import load_reviewed_modular_workflow_snapshot

        expected = {item["pipelineClass"] for item in load_reviewed_modular_workflow_snapshot()["contracts"]}
        assert {"Cosmos3DistilledModularPipeline", "MiniMaxH3ModularPipeline"} <= expected
        constructed = []

        class Constructors:
            def __getattr__(self, name):
                def construct(**kwargs):
                    constructed.append((name, kwargs))
                    return SimpleNamespace(name=name)
                return construct

        with patch.object(generator, "diffusers", Constructors()), patch.object(
            generator, "PINNED_MODULAR_WORKFLOW_TRUTH",
            {"Cosmos3DistilledModularPipeline": SimpleNamespace(constructor_config=(("fixture", 7),))},
        ), patch.object(generator, "build_modular_conditional_contract", lambda pipeline: pipeline.name), patch.object(
            generator, "merge_modular_conditional_contracts", lambda parts: parts
        ):
            result = generator.generate()
        assert len(result) == len(expected)
        assert set(result) == expected
        assert dict(constructed)["Cosmos3DistilledModularPipeline"] == {"config_dict": {"fixture": 7}}
        assert dict(constructed)["MiniMaxH3ModularPipeline"] == {}
