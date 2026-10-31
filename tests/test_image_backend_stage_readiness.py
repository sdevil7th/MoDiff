"""Declared backend image stages do not imply upstream or runtime qualification."""
import builtins
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRIMARY_ROUTES = (
    "flux-fill:direct/inpaint", "flux-fill:direct/outpaint",
    "flux-canny:direct/control_image", "flux-depth:direct/control_image",
    "flux-redux:direct/edit_image", "flux-redux:direct/multi_image_reference_edit",
)


def guarded_build():
    from modiff.image_prototyping_readiness import build_image_prototyping_readiness
    original = builtins.__import__

    def no_heavy_or_registry(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"modules", "torch", "diffusers", "transformers", "peft"}:
            raise AssertionError("Readiness imported a registry or model library")
        return original(name, *args, **kwargs)

    with patch("builtins.__import__", no_heavy_or_registry), patch("socket.socket.connect", side_effect=AssertionError("network")):
        return build_image_prototyping_readiness(ROOT)


@pytest.fixture(scope="module")
def ledger():
    return guarded_build()


@pytest.mark.parametrize("identity", PRIMARY_ROUTES)
def test_current_six_primaries_are_backend_stages_without_upstream_or_live_promotion(ledger, identity):
    row = next(row for row in ledger["routes"] if row["routeId"] == identity)
    assert row["disposition"] == "backend_stages_integrated"
    assert row["dispositionReason"] == "complete_backend_image_stage_chain_declared"
    assert row["exception"] is None
    assert row["evidence"]["structural"] == "implemented_awaiting_qualification"
    assert row["upstream"]["fullStageChain"] is False
    assert row["upstream"]["nativeWorkflowId"] is None
    assert row["evidence"]["nativeOutput"] == "pending"
    assert row["evidence"]["platform"] == "pending"
    assert row["evidence"]["browser"] == "pending_final_candidate"


@pytest.mark.parametrize("corruption", [
    "missing_stage", "missing_geometry", "foreign_profile", "immutable_pair", "foreign_primary",
    "raw_whole", "no_spec", "alias_output", "duplicate_supplier", "missing_preview",
])
def test_incomplete_foreign_or_whole_declarations_do_not_become_backend_stages(ledger, corruption):
    from modiff import image_prototyping_readiness as readiness
    from modiff.diffusers_profiles import public_execution_profiles
    from modiff.studio_execution_specs import studio_execution_spec_for_pair
    profile = next(row for row in public_execution_profiles(observe_optional_runtime=False, platform_name="linux", machine="x86_64")
                   if row["id"] == "flux-fill:direct")
    artifact = deepcopy(next(row["artifact"] for row in ledger["routes"] if row["routeId"] == PRIMARY_ROUTES[0]))
    spec = studio_execution_spec_for_pair(profile["model_type"], "inpaint")
    if corruption == "missing_stage":
        spec["roles"] = [row for row in spec["roles"] if row[1] != "modules.DiffusersImage.Denoise"]
    elif corruption == "missing_geometry":
        spec["edges"] = [row for row in spec["edges"] if row[1] != "width_out"]
    elif corruption == "foreign_profile":
        profile["id"] = "foreign:direct"
    elif corruption == "immutable_pair":
        artifact["revision"] = "a" * 40
    elif corruption == "foreign_primary":
        artifact["repository"] = "foreign/model"
    elif corruption == "raw_whole":
        spec["roles"] = [row for row in spec["roles"] if row[1] not in {
            "modules.DiffusersImage.EncodeInputs", "modules.DiffusersImage.Denoise", "modules.DiffusersImage.DecodeLatents"}]
        spec["roles"].append(("whole", "modules.DiffusersImage.Inpaint", 0, 0))
    elif corruption == "no_spec":
        spec = None
    elif corruption == "alias_output":
        spec["edges"] = [(source, "images" if output == "latents_out" else output, target, field)
                         for source, output, target, field in spec["edges"]]
    elif corruption == "duplicate_supplier":
        spec["edges"] = [*spec["edges"], ("imageEncode", "encoded_inputs", "decode", "width")]
    else:
        spec["edges"] = [row for row in spec["edges"] if not (row[0] == "decode" and row[1] == "images")]
    assert readiness._declared_backend_image_stages(profile, "inpaint", artifact, spec) is False


def test_all_other_routes_and_upstream_pending_evidence_are_exactly_preserved(monkeypatch, ledger):
    from modiff import image_prototyping_readiness as readiness
    monkeypatch.setattr(readiness, "_declared_backend_image_stages", lambda *_args: False)
    original = guarded_build()
    assert ledger["sources"] == original["sources"]
    assert len(ledger["routes"]) == len(original["routes"]) == 220
    rows = {row["routeId"]: row for row in ledger["routes"]}
    baseline = {row["routeId"]: row for row in original["routes"]}
    changed = []
    for identity, row in rows.items():
        if identity not in PRIMARY_ROUTES:
            assert row == baseline[identity]
            continue
        assert row["upstream"] == baseline[identity]["upstream"]
        assert {key: value for key, value in row["evidence"].items() if key != "structural"} == {
            key: value for key, value in baseline[identity]["evidence"].items() if key != "structural"}
        left, right = deepcopy(row), deepcopy(baseline[identity])
        for item in (left, right):
            for key in ("disposition", "dispositionReason", "exception"):
                item.pop(key)
            item["evidence"].pop("structural")
        assert left == right
        changed.append(identity)
    assert set(changed) == set(PRIMARY_ROUTES)
    assert len(rows) - len(changed) == 214
    # The declared compatible Canny repair and unconverted sibling modes are
    # among these unchanged whole routes, not generalized from their family.
    assert rows["flux-canny:direct/control_image@fuliucansheng/FLUX.1-Canny-dev-diffusers"]["disposition"] == "documented_whole_pipeline_exception"
