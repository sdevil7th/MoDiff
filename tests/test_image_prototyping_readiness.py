import json
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_image_readiness_ledger_reproduces_without_registry_or_network(monkeypatch):
    from modiff.image_prototyping_readiness import (
        build_image_prototyping_readiness,
        load_image_prototyping_readiness,
    )

    monkeypatch.setattr("socket.socket.connect", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()))
    generated = build_image_prototyping_readiness(ROOT)
    assert generated == load_image_prototyping_readiness()
    assert generated["summary"]["routeCount"] == len(generated["routes"])
    assert generated["summary"]["denominatorFrozen"] is False
    assert generated["coverageBoundary"]["clientTemplatesAndTaskChoosers"] == "requires_paired_client_gate"


@pytest.mark.parametrize("host", [("linux", "x86_64"), ("windows", "x86_64"), ("macos", "arm64")])
def test_image_readiness_ledger_is_independent_of_host_runtime_target(monkeypatch, host):
    from modiff import diffusers_profiles
    from modiff.image_prototyping_readiness import (
        build_image_prototyping_readiness,
        load_image_prototyping_readiness,
    )

    resolve_target = diffusers_profiles.optional_runtime_target
    monkeypatch.setattr(
        diffusers_profiles,
        "optional_runtime_target",
        lambda *, platform_name=None, machine=None: resolve_target(
            platform_name=platform_name or host[0], machine=machine or host[1],
        ),
    )
    # Live publication must still follow the host, unlike the static inventory.
    profile = diffusers_profiles.DIFFUSERS_EXECUTION_PROFILES["z-image:modular"]
    assert profile.to_public_dict()["optional_runtime_profiles"] == list(
        profile.optional_runtime_profile_ids_for_target(platform_name=host[0], machine=host[1])
    )
    generated = build_image_prototyping_readiness(ROOT)
    expected = load_image_prototyping_readiness()
    assert generated["sources"] == expected["sources"]
    assert generated == expected


@pytest.mark.parametrize("target", [{"platform_name": "windows"}, {"machine": "arm64"}])
def test_explicit_inventory_target_cannot_observe_installed_runtime(target):
    from modiff.diffusers_profiles import public_execution_profiles

    with pytest.raises(ValueError, match="Runtime observations cannot use an explicit target"):
        public_execution_profiles(observe_optional_runtime=True, **target)


def test_image_readiness_distinguishes_native_stages_exceptions_and_non_diffusion_tasks():
    from modiff.image_prototyping_readiness import load_image_prototyping_readiness

    ledger = load_image_prototyping_readiness()
    by_id = {route["routeId"]: route for route in ledger["routes"]}

    qwen = by_id["qwen-image:modular/modular_text_to_image"]
    assert qwen["disposition"] == "native_integrated"
    assert qwen["upstream"]["fullStageChain"] is True
    assert {"encode_prompt", "denoise", "decode_latents"}.issubset(qwen["upstream"]["stageRoles"])

    glm = by_id["glm-image:direct/text_to_image"]
    assert glm["disposition"] == "documented_whole_pipeline_exception"
    assert glm["exception"]["checkedDiffusersRevision"] == ledger["diffusersRevision"]

    ernie = by_id["ernie-image-turbo:official-modular-workflow/text_to_image"]
    assert ernie["disposition"] == "native_integrated"
    assert ernie["upstream"]["fullStageChain"] is True
    assert {"rewrite_prompt", "encode_prompt", "denoise", "decode_latents"}.issubset(
        ernie["upstream"]["stageRoles"]
    )

    depth = next(route for route in ledger["routes"] if route["canonicalTask"] == "depth_estimation")
    assert depth["disposition"] == "non_diffusion_image_task"


def test_image_readiness_rejects_tampering_even_with_valid_json(tmp_path):
    from modiff.image_prototyping_readiness import (
        ImagePrototypingReadinessError,
        load_image_prototyping_readiness,
    )

    value = deepcopy(load_image_prototyping_readiness())
    value["routes"][0]["disposition"] = "runnable"
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ImagePrototypingReadinessError):
        load_image_prototyping_readiness(path)


def test_generic_stage_routes_do_not_require_whole_workflow_adapters():
    from modiff.image_prototyping_readiness import load_image_prototyping_readiness

    routes = load_image_prototyping_readiness()["routes"]
    for pipeline in ("FluxModularPipeline", "StableDiffusionXLModularPipeline"):
        selected = [row for row in routes if row["implementation"]["pipelineClass"] == pipeline]
        assert selected
        assert all(row["disposition"] == "native_integrated" for row in selected)


def test_preserved_standard_alternative_is_not_an_upstream_exception_or_missing_native_route():
    from modiff.image_prototyping_readiness import load_image_prototyping_readiness

    routes = load_image_prototyping_readiness()["routes"]
    direct = next(row for row in routes if row["routeId"] == "ernie-image-turbo:direct/text_to_image")
    assert direct["disposition"] == "preserved_standard_alternative"
    assert direct["integratedNativeAlternativeProfileIds"] == ["ernie-image-turbo:official-modular-workflow"]
    assert direct["exception"] is None
    assert direct["evidence"]["nativeOutput"] != "passed"


def test_missing_action_binding_cannot_be_called_integrated(monkeypatch):
    from modiff import image_prototyping_readiness as readiness

    monkeypatch.delitem(readiness.MODULAR_ACTION_BINDINGS, "denoise")
    rows = readiness.build_image_prototyping_readiness(ROOT)["routes"]
    flux = [row for row in rows if row["implementation"]["pipelineClass"] == "FluxModularPipeline"]
    assert flux and all(row["disposition"] == "native_integration_missing" for row in flux)


def test_exact_native_variant_recipes_do_not_inherit_family_output_qualification():
    from modiff.image_prototyping_readiness import load_image_prototyping_readiness
    rows = {row["routeId"]: row for row in load_image_prototyping_readiness()["routes"]}
    for identity in ("flux-schnell:direct/text_to_image", "flux-krea:direct/text_to_image"):
        assert rows[identity]["disposition"] == "preserved_standard_alternative"
        assert rows[identity]["upstream"]["candidateNativePipelineClass"] == "FluxModularPipeline"
        assert rows[identity]["exception"] is None
        assert rows[identity]["exactVariantReview"]["status"] == "reviewed"
        assert rows[identity]["exactVariantReview"]["liveQualification"] == "not_established_by_review"


def test_exact_reviews_retain_quantized_artifact_blockers_and_true_kv_exception():
    from modiff.image_prototyping_readiness import load_image_prototyping_readiness
    rows = {row["routeId"]: row for row in load_image_prototyping_readiness()["routes"]}
    reviews = [row for row in rows.values() if row.get("exactVariantReview")]
    assert len(reviews) == 17  # five PAG integrations plus twelve exact-variant reviews
    assert not any(row["dispositionReason"] == "native_family_requires_exact_artifact_variant_task_review"
                   for row in rows.values())
    for row in reviews:
        review = row["exactVariantReview"]
        assert review["source"] and review["limitations"] and review["status"] == "reviewed"
        if review["decision"] == "artifact_blocked":
            assert row["disposition"] == "blocked" and row["exception"] is None
            assert not row["integratedNativeAlternativeProfileIds"]
        elif review["decision"] == "upstream_exception":
            assert row["disposition"] == "documented_whole_pipeline_exception"
            assert not row["integratedNativeAlternativeProfileIds"]
        else:
            assert row["disposition"] == "preserved_standard_alternative"
            assert row["integratedNativeAlternativeProfileIds"] == [review["nativeProfileId"]]


def test_advertised_artifact_variants_are_distinct_required_rows():
    from modiff.image_prototyping_readiness import build_image_prototyping_readiness
    rows = {row["routeId"]: row for row in build_image_prototyping_readiness(ROOT)["routes"]}
    base = "flux-dev:modular/text_to_image"
    alternate = rows[f"{base}@black-forest-labs/FLUX.1-dev-FP8"]
    assert alternate["artifactVariantOf"] == base
    assert alternate["artifact"]["repository"] != rows[base]["artifact"]["repository"]
    assert alternate["artifact"]["revision"]
    assert alternate["evidence"]["nativeOutput"] != "passed"


def test_workflow_scoped_native_variants_are_not_omitted_or_leaked_to_other_tasks():
    from modiff.image_prototyping_readiness import build_image_prototyping_readiness
    rows = {row["routeId"]: row for row in build_image_prototyping_readiness(ROOT)["routes"]}
    repository = "unsloth/Qwen-Image-2512-unsloth-bnb-4bit"
    alternate = rows[f"qwen-image:modular/modular_text_to_image@{repository}"]
    assert alternate["disposition"] == "native_integrated"
    assert f"qwen-image:modular/modular_image_to_image@{repository}" not in rows
    assert rows[f"qwen-image:t2i-direct/text_to_image@{repository}"]["disposition"] == "preserved_standard_alternative"
