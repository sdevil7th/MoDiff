"""Machine admission tiers and measured/declared working demand are distinct."""

import pytest

from modiff.auto_resource import AUTO_MODEL_REQUIREMENTS, _candidate
from modiff.studio_execution_specs import studio_auto_model_requirements
from modiff.workflow_auto_lifecycle import assert_next_owner_capacity


GIB = 1024 ** 3


def candidate(requirements):
    return _candidate(candidate_id="test", rank=1, model_type="TestPipeline", mode="text_to_image",
                      execution_path="direct-diffusers-image", loader_module="modules.DiffusersImage",
                      loader_action="LoadPipeline", artifact="test/model", dtype="bfloat16",
                      quantization_mode="none", quantized_components=[], offload_mode="model_cpu",
                      quality_tier="native", reason="Existing static declaration", generation={}, installed=True,
                      requirements=requirements)


def hardware(free_ram=5 * GIB):
    return {"accelerator": {"kind": "cuda", "totalBytes": 16 * GIB, "freeBytes": 4 * GIB},
            "systemMemory": {"totalBytes": 32 * GIB, "availableBytes": free_ram},
            "offloadDisk": {"freeBytes": 100 * GIB}}


def owner():
    return {"ownerId": "loader", "device": "cuda:0", "offloadMode": "model_cpu",
            "requirements": {"systemRamBytes": 0, "vramBytes": 0},
            "capacityRequirements": {"systemRamBytes": 8 * GIB, "vramBytes": 8 * GIB},
            "workingMemoryPolicy": "runtime_headroom_policy"}


def test_all_source_static_tiers_declare_their_total_capacity_semantics():
    assert AUTO_MODEL_REQUIREMENTS
    assert all(value["memorySemantics"] == "machine_capacity" for value in AUTO_MODEL_REQUIREMENTS.values())
    assert all(value["memorySemantics"] == "machine_capacity" for value in studio_auto_model_requirements().values())


def test_static_candidate_preserves_capacity_and_reports_unmeasured_working_demand():
    requirements = {"minimum": {"systemRamBytes": 8 * GIB, "vramBytes": 8 * GIB}}
    result = candidate(requirements)
    assert result["requirements"]["memorySemantics"] == result["memorySemantics"] == "machine_capacity"
    assert result["requirements"]["minimum"] == requirements["minimum"]
    assert result["workingMemoryRequirements"] is None
    assert result["workingMemoryPolicy"] == "runtime_headroom_policy"
    assert result["proof"]["status"] == "declared_safe"  # No promotion to measured execution proof.
    assert "memorySemantics" not in requirements  # Source caller input is preserved.


def test_candidate_keeps_an_explicit_working_contract_separate_from_machine_tier():
    working = {"systemRamBytes": 6 * GIB, "vramBytes": 4 * GIB}
    result = candidate({"minimum": {"systemRamBytes": 32 * GIB, "vramBytes": 16 * GIB},
                        "workingMemoryRequirements": working})
    assert result["workingMemoryRequirements"] == working
    assert result["workingMemoryPolicy"] == "explicit_working_demand"


def test_lifecycle_rechecks_totals_and_actual_headroom_after_release():
    assert_next_owner_capacity(None, owner(), hardware())
    with pytest.raises(ValueError, match="actual free memory"):
        assert_next_owner_capacity(None, owner(), hardware(4279840768))
    smaller = hardware()
    smaller["systemMemory"]["totalBytes"] = 6 * GIB
    with pytest.raises(ValueError, match="8 GiB total"):
        assert_next_owner_capacity(None, owner(), smaller)


def test_lifecycle_does_not_replace_explicit_working_demand_with_headroom_policy():
    required = owner()
    required["requirements"]["systemRamBytes"] = 7 * GIB
    required["workingMemoryPolicy"] = "explicit_working_demand"
    with pytest.raises(ValueError, match="actual free memory"):
        assert_next_owner_capacity(None, required, hardware())


def test_shared_offload_capacity_is_rechecked_using_the_same_offload_recipe():
    shared = hardware(24 * GIB)
    shared["accelerator"].update(totalBytes=2 * GIB, freeBytes=2 * GIB, memoryKind="shared",
                                 sharedTotalBytes=32 * GIB, accessibleTotalBytes=32 * GIB)
    assert_next_owner_capacity(None, owner(), shared)


def test_legacy_cleanup_does_not_misread_capacity_tiers_as_free_working_memory():
    from modiff.server import WebServer

    requirements = {"memorySemantics": "machine_capacity", "minimum": {"systemRamBytes": 8 * GIB, "vramBytes": 8 * GIB}}
    hints = {"autoResourcePlan": {"requirements": requirements, "offloadMode": "model_cpu"}}
    assert WebServer._auto_candidate_minimums(hints) == {}
    hints["autoResourcePlan"]["workingMemoryRequirements"] = {"systemRamBytes": 6 * GIB, "vramBytes": 4 * GIB}
    assert WebServer._auto_candidate_minimums(hints) == {"systemRamBytes": 6 * GIB, "vramBytes": 4 * GIB}
