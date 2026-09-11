"""Weightless component requirements for exact reviewed Modular block contracts.

Static assessment compares declarations without loading models. Runtime checks
validate actual upstream component types (including factory return types), not
pipeline labels. Tensor shape/operation semantics remain upstream-owned. A
missing pretrained component is unresolved, not an identity mismatch.
"""

from collections.abc import Mapping
from typing import Any


def assess_component_requirements(source: Mapping[str, Any], destination: Mapping[str, Any]) -> dict[str, Any]:
    available = {component["name"]: component for component in destination.get("components", ())}
    issues = []
    for component in source.get("components", ()):
        name = component["name"]
        existing = available.get(name)
        required_type = component.get("type")
        if existing is None:
            if component.get("creationMethod") == "from_config":
                # The exact pinned upstream ComponentSpec constructs this; no
                # model download, guessed component or repository code is used.
                continue
            issues.append({"code": "component_missing", "component": name,
                           "requiredType": required_type, "actualType": None})
        elif existing.get("type") != required_type:
            issues.append({"code": "component_type_conflict", "component": name,
                           "requiredType": required_type, "actualType": existing.get("type")})
    status = ("incompatible" if any(item["code"] == "component_type_conflict" for item in issues)
              else "unresolved" if issues else "compatible")
    return {"status": status, "issues": issues}


def _matches_component_type(value, expected):
    if isinstance(value, expected):
        return True
    # Auto* declarations are factories, not bases of their returned objects.
    # Resolve official class identities lazily: metadata discovery must not
    # import/install the optional runtime, and a same-named user class is not
    # evidence of factory compatibility. Artifact identity stays loader-owned.
    if not expected.__module__.startswith('transformers.models.auto.'):
        return False
    from transformers import AutoProcessor, AutoTokenizer, AutoImageProcessor, AutoFeatureExtractor
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from transformers.processing_utils import ProcessorMixin
    from transformers.image_processing_base import ImageProcessingMixin
    from transformers.feature_extraction_utils import FeatureExtractionMixin
    types = {
        AutoTokenizer: (PreTrainedTokenizerBase,),
        AutoImageProcessor: (ImageProcessingMixin,),
        AutoFeatureExtractor: (FeatureExtractionMixin,),
        # The reviewed AutoProcessor implementation falls back to tokenizer,
        # image processor and feature extractor for single-modality artifacts.
        AutoProcessor: (ProcessorMixin, PreTrainedTokenizerBase,
                        ImageProcessingMixin, FeatureExtractionMixin),
    }
    return isinstance(value, types.get(expected, ()))


def validate_runtime_component_requirements(block, pipeline, *, path):
    """Use actual upstream Python types; valid subclasses are not label mismatches."""
    for spec in getattr(block, "expected_components", ()):
        name = spec.name
        expected = getattr(spec, "type_hint", None)
        value = getattr(pipeline, name, None)
        label = f"Modular block {'/'.join(path)} ({type(block).__name__}), component {name!r}"
        if value is None:
            raise ValueError(f"{label}: required component is missing. Connect/load it through the component loader before running this block.")
        if isinstance(expected, type) and not _matches_component_type(value, expected):
            required = f"{expected.__module__}.{expected.__name__}"
            actual = f"{type(value).__module__}.{type(value).__name__}"
            raise ValueError(f"{label}: expected {required}, received {actual}. Connect a compatible component or replace the block; prompts and parameters were not changed.")
