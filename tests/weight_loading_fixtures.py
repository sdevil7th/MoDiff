"""Controlled loading metadata for existing requested-dtype planner fixtures.

These records deliberately model already-converted source weights. They are
neither inspected artifact headers nor a measured allocation/fit qualification.
Tests of source conversion use real tiny safetensors header fixtures instead.
"""

import hashlib
import json


def same_dtype_conversion_reserve(lower_bound):
    if not lower_bound or not lower_bound.get("components"):
        return None  # Unknown-parameter fixtures also have unknown source headers.
    components = {}
    for name, size in lower_bound["components"].items():
        layout = json.dumps(["controlled-same-requested-dtype", name, size,
                             lower_bound["revision"], lower_bound["dtype"]])
        components[name] = {
            "loaderLibrary": "diffusers",
            "loaderClass": "ControlledPlannerFixtureModel",
            "layoutSha256": hashlib.sha256(layout.encode()).hexdigest(),
            "sourcePayloadBytes": size,
            "requestedPayloadBytes": size,
            "convertedTargetBytes": 0,
            "largestSourceShardBytes": size,
            "largestConvertedTensorBytes": 0,
            "serialCpuOverlapBytes": size,
            "conservativeCpuOverlapBytes": size,
            "serialCudaOverlapBytes": size,
            "conservativeCudaOverlapBytes": size,
        }
    return {
        "revision": lower_bound["revision"], "dtype": lower_bound["dtype"],
        "components": components,
        "proofScope": "source_requested_loading_overlap_only", "complete": False,
    }
