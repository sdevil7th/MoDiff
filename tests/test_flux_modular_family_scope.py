"""FLUX family scope must distinguish executable hierarchy from a fallback."""

from modiff.modular_workflow_contracts import PINNED_MODULAR_WORKFLOW_TRUTH
from modiff.huggingface_node_library import reviewed_huggingface_node_library


def test_all_five_installed_flux_modular_pipelines_have_reviewed_native_truth():
    for name in ("FluxModularPipeline", "FluxKontextModularPipeline",
                 "Flux2ModularPipeline", "Flux2KleinModularPipeline",
                 "Flux2KleinBaseModularPipeline"):
        assert name in PINNED_MODULAR_WORKFLOW_TRUTH, f"{name} lacks reviewed native hierarchy"


def test_loop_and_decode_matrices_cover_all_native_routes_not_ordinary_composites():
    from collections import Counter
    from tests.test_flux_modular_loop_matrix import admissions as loop_admissions
    from tests.test_flux_modular_decode_matrix import admissions as decode_admissions
    native = loop_admissions()
    assert decode_admissions() == native
    assert Counter(item['pipelineClass'] for item in native) == {
        'FluxModularPipeline': 2, 'FluxKontextModularPipeline': 2,
        'Flux2ModularPipeline': 2, 'Flux2KleinModularPipeline': 2,
        'Flux2KleinBaseModularPipeline': 2,
    }
    ordinary = [item for item in reviewed_huggingface_node_library()['definitions']
                if item.get('pipelineClass', '').startswith('Flux')
                and item.get('definitionKind') == 'studio_execution_composite']
    assert len(ordinary) == 26
    assert not {item['id'] for item in native} & {item['id'] for item in ordinary}


def test_full_flux2_admissions_seal_native_loader_and_keep_standard_pipeline_separate():
    from modiff.studio_execution_specs import studio_execution_spec_for_pair

    definitions = [item for item in reviewed_huggingface_node_library()["definitions"]
                   if item.get("pipelineClass") == "Flux2ModularPipeline"]
    assert len(definitions) == 2
    for definition in definitions:
        assert definition["integrationStatus"] == "reviewed_modiff_contract"
        admission, = definition["executionAdmissions"]
        assert admission["status"] == "admitted"
        spec = studio_execution_spec_for_pair(definition["pipelineClass"], admission["studioMode"])
        assert spec["id"] == admission["studioExecutionSpec"]["id"]
        assert spec["loaderModule"] == "modules.ModularDiffusers"
        assert spec["loaderAction"] == "ModelsLoader"
        assert spec["pipelineClass"] == "Flux2ModularPipeline"
    # Ordinary authored/legacy direct nodes retain their genuine standard route.
    direct = studio_execution_spec_for_pair("Flux2Pipeline", "multi_image_reference_edit")
    assert direct["loaderModule"] == "modules.DiffusersImage"


def test_saved_flux2_fallback_loader_retains_its_explicit_profile_without_reappearing_in_catalog():
    from modiff.diffusers_profiles import resolve_execution_profiles_for_loader, public_execution_profiles
    from modiff.studio_execution_specs import studio_execution_spec_for_pair

    legacy_id = "flux2-modular:equivalent-standard"
    for mode in ("text_to_image", "multi_image_reference_edit"):
        profiles, reason = resolve_execution_profiles_for_loader(
            "modules.DiffusersImage", "LoadPipeline",
            {"pipeline_class": "Flux2Pipeline", "mode": mode, "execution_profile_id": legacy_id},
        )
        assert reason is None
        assert len(profiles) == 1
        assert profiles[0].id == legacy_id
        assert profiles[0].pipeline_class == "Flux2Pipeline"
    assert legacy_id not in {profile["id"] for profile in public_execution_profiles()}
    assert studio_execution_spec_for_pair("Flux2ModularPipeline", "text_to_image")["loaderAction"] == "ModelsLoader"


def test_future_flux3_discovery_is_separate_from_published_native_matrix():
    import gzip
    import json
    from pathlib import Path

    from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES, public_execution_profiles

    from tests.test_flux_modular_loop_matrix import admissions as loop_admissions

    future_names = {"Flux3ModularPipeline", "Flux3DistilledModularPipeline"}
    library = reviewed_huggingface_node_library()
    future = [item for item in library["definitions"] if item["pipelineClass"] in future_names]
    assert {(item["pipelineClass"], item["workflowId"]) for item in future} == {
        (name, workflow) for name in future_names for workflow in ("text2image", "image_conditioned")
    }
    assert all(not item["executionAdmissions"] and item["executionClaim"] == "discovery_only" for item in future)
    assert not future_names & {item["pipelineClass"] for item in loop_admissions()}
    public = public_execution_profiles()
    assert not future_names & {item["model_type"] for item in public}
    profiles = [item for item in DIFFUSERS_EXECUTION_PROFILES.values() if item.model_type in future_names]
    assert len(profiles) == 2
    assert all(item.public is False and item.live_proof is False and item.expert_quantization_modes == () for item in profiles)
    catalog = json.loads(gzip.decompress((Path(__file__).resolve().parents[1] /
                                       "modiff/registered_block_v2_catalog.v1.json.gz").read_bytes()))
    assert not {item["id"] for item in future} & {
        item["definition"]["source"]["manifestDefinitionId"] for item in catalog["entries"]
    }
