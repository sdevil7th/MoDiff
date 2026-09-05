import copy
import hashlib
import json
import unittest

from modiff.huggingface_cluster_publication_audit import (
    ClusterPublicationAuditError,
    _RESOURCE_REPORT_HASH_PREFIX,
    _RESOURCE_ROUTE_BINDING_HASH_PREFIX,
    _canonical_hash,
    audit_cluster_publication_route,
    checked_in_publication_evidence,
    current_resource_route_manifest,
    validate_resource_recipe_coverage,
)
from modiff.huggingface_node_library import reviewed_huggingface_node_library
from modiff.huggingface_cluster_runtime import REGISTERED_BLOCK_V2_DEFINITION_PINS


QWEN = "diffusers.cluster-admission:QwenImageModularPipeline:text2image:mode:text_to_image"
MINIMAX = (
    "diffusers.cluster-admission:MiniMaxMusic3ModularPipeline:default:"
    "workflow:official_top_level_blocks"
)
WAN = "diffusers.cluster-admission:WanTI2VPipeline:text_to_video:mode:text_to_video"


def _route_binding(admission, *, definition_pin=None):
    admission_id = admission["id"]
    pin = definition_pin or REGISTERED_BLOCK_V2_DEFINITION_PINS[admission_id]
    return {
        "schemaVersion": 1,
        "admissionId": admission_id,
        "blockDefinition": {
            "definitionId": admission_id,
            "contentHash": pin[0],
            "canonicalSha256": pin[1],
        },
        "studioExecutionSpec": admission["studioExecutionSpec"],
        "artifact": {
            "repository": admission["artifact"]["repo"],
            "revision": admission["artifact"]["revision"],
        },
        "modelDependencies": sorted(
            [
                {
                    "id": dependency["id"],
                    "kind": dependency["kind"],
                    "repository": dependency["repo"],
                    "revision": dependency["revision"],
                }
                for dependency in admission["modelDependencies"]
            ],
            key=lambda dependency: (
                dependency["id"],
                dependency["repository"],
                dependency["revision"],
            ),
        ),
    }


def _binding_hash(binding):
    return _RESOURCE_ROUTE_BINDING_HASH_PREFIX + hashlib.sha256(
        json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _exact_recipe(recipe):
    value = {
        "modelType": recipe["modelType"],
        "dtype": recipe["dtype"],
        "offloadMode": recipe["offloadMode"],
        "quantizationMode": recipe["quantizationMode"],
        "quantizedComponents": [],
        "autoOffload": recipe["offloadMode"] != "none",
        "device": "cuda:0",
        "deviceMap": None,
        "pipelineClass": recipe["modelType"],
        "executionPath": "diffusers_pipeline",
        "attentionBackend": "auto",
        "regionalCompile": False,
        "denoiserCache": "none",
        "channelsLast": False,
        "layerwiseCasting": False,
    }
    value["recipeHash"] = "sha256:resource-recipe-v2:" + hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return value


def _proof(task_id, captured_at, suffix):
    return {
        "taskId": task_id,
        "capturedAt": captured_at,
        "graphHash": "sha256:canonical-graph-v1:test",
        "outputHash": f"sha256:decoded-output-v1:{suffix}",
        "outputCollectionHash": f"sha256:decoded-output-collection-v1:{suffix}",
        "executionDurationSeconds": 1.0,
        "peakMemoryBytes": 1024,
        "peakReservedBytes": 2048,
        "processRssBytes": 4096,
        "backend": "rocm",
        "device": "cuda:0",
        "proofPath": f"data/qualification/release/route-resource-provenance/{task_id}.json",
        "proofSha256": f"sha256:{suffix * 64}",
    }


def _exact_evidence():
    proofs = [
        _proof("task-exact-route-a", "2026-09-02T00:00:00.000Z", "2"),
        _proof("task-exact-route-b", "2026-09-02T00:05:00.000Z", "3"),
    ]
    return {
        "bindingStatus": "exact_route_bound_two_proof",
        "evidenceSource": "route_resource_two_live_proofs",
        "proofCount": 2,
        "taskIds": [proof["taskId"] for proof in proofs],
        "lastSuccessAt": proofs[-1]["capturedAt"],
        "proofs": proofs,
    }


def _exact_qualification(binding, binding_hash, recipe):
    return {
        "routeBinding": binding,
        "routeBindingHash": binding_hash,
        "workloadHash": f"sha256:resource-workload-v1:{'4' * 64}",
        "modelSetHash": f"sha256:model-set-v1:{'1' * 64}",
        "recipe": _exact_recipe(recipe),
        "runtimeContract": {
            "runtimeFingerprint": "sha256:stable-runtime-v1:test",
            "runtimeLockFingerprint": "sha256:stable-runtime-v1:test",
            "backendSourceFingerprint": "sha256:backend-source-v1:test",
            "backendContractFingerprint": "sha256:backend-contract-v1:test",
            "deterministicFingerprint": "sha256:deterministic-v1:test",
        },
        "evidence": _exact_evidence(),
    }


class HuggingFaceClusterPublicationAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = reviewed_huggingface_node_library()
        resource, cls.release = checked_in_publication_evidence()
        # Local qualification workspaces may contain stale ignored receipts.
        # Unit tests start from the checked family evidence with an explicitly
        # empty exact-route lane and add current route fixtures per scenario.
        cls.resource = copy.deepcopy(resource)
        cls.resource["routeQualifications"] = []
        cls.resource["reportHash"] = _canonical_hash(
            cls.resource,
            field="reportHash",
            prefix=_RESOURCE_REPORT_HASH_PREFIX,
        )

    def audit(self, admission_id, *, resource=None):
        return audit_cluster_publication_route(
            admission_id,
            library=self.library,
            resource_coverage=resource or self.resource,
            release_report=self.release,
        )

    def test_current_routes_publish_only_evidence_they_actually_have(self):
        routes = {admission_id: self.audit(admission_id) for admission_id in (QWEN, MINIMAX, WAN)}
        self.assertEqual(set(routes), {QWEN, MINIMAX, WAN})

        # The preserved schema-v1 Qwen and MiniMax reviews did not bind the
        # exact BlockDefinitionV2 identity.  They are historical evidence,
        # not current publication authority.
        self.assertTrue(all(not route["flags"]["liveProof"] for route in routes.values()))
        for route in routes.values():
            self.assertTrue(route["flags"]["insertable"])
            self.assertFalse(route["flags"]["executable"])
            self.assertFalse(route["flags"]["autoEligible"])
            self.assertFalse(route["flags"]["galleryEligible"])
            self.assertFalse(route["flags"]["releaseEligible"])

    def test_routes_remain_unqualified_without_current_exact_receipts(self):
        qwen = self.audit(QWEN)
        minimax = self.audit(MINIMAX)
        wan = self.audit(WAN)

        for route in (qwen, minimax, wan):
            self.assertEqual(route["evidence"]["resourceRecipes"]["exactRouteQualifiedCount"], 0)
            blocker_codes = {reason["code"] for reason in route["blockers"]}
            self.assertIn("runtime_resource_admission_required", blocker_codes)
            self.assertFalse(route["flags"]["executable"])
            self.assertFalse(route["flags"]["autoEligible"])
            self.assertFalse(route["flags"]["galleryEligible"])
            self.assertFalse(route["flags"]["releaseEligible"])
        self.assertIn(
            "live_output_review_pending",
            {reason["code"] for reason in wan["blockers"]},
        )
        self.assertIn(
            "live_output_review_reapproval_required",
            {reason["code"] for reason in qwen["blockers"]},
        )
        self.assertIn(
            "live_output_review_reapproval_required",
            {reason["code"] for reason in minimax["blockers"]},
        )
        self.assertEqual(qwen["evidence"]["outputReview"]["status"], "historical_non_authorizing")
        self.assertEqual(minimax["evidence"]["outputReview"]["status"], "historical_non_authorizing")

    def test_current_route_manifest_projects_each_pinned_v2_route_exactly_once(self):
        manifest = current_resource_route_manifest(self.library)
        self.assertEqual(manifest["schemaVersion"], 1)
        self.assertEqual(manifest["format"], "modiff.current-resource-routes.v1")
        self.assertTrue(manifest["manifestHash"].startswith("sha256:current-resource-routes-v1:"))
        admission_ids = [route["admissionId"] for route in manifest["routes"]]
        self.assertEqual(admission_ids, sorted(set(admission_ids)))
        qwen = next(route for route in manifest["routes"] if route["admissionId"] == QWEN)
        self.assertEqual(qwen["routeBinding"], _route_binding(next(
            admission
            for definition in self.library["definitions"]
            for admission in definition.get("executionAdmissions", ())
            if admission["id"] == QWEN
        )))

        ambiguous = copy.deepcopy(self.library)
        owner = next(
            definition
            for definition in ambiguous["definitions"]
            if any(admission.get("id") == QWEN for admission in definition.get("executionAdmissions", ()))
        )
        owner["executionAdmissions"].append(copy.deepcopy(next(
            admission for admission in owner["executionAdmissions"] if admission["id"] == QWEN
        )))
        with self.assertRaisesRegex(ClusterPublicationAuditError, "ambiguous admission"):
            current_resource_route_manifest(ambiguous)

    def test_even_exact_resource_evidence_requires_separate_auto_authority(self):
        resource = copy.deepcopy(self.resource)
        admission = next(
            admission
            for definition in self.library["definitions"]
            for admission in definition.get("executionAdmissions", ())
            if admission["id"] == QWEN
        )
        recipe = next(
            recipe
            for recipe in resource["recipes"]
            if recipe["modelType"] == "QwenImageModularPipeline" and recipe["status"] == "qualified"
        )
        route_binding = _route_binding(admission)
        route_binding_hash = _binding_hash(route_binding)
        resource["routeQualifications"].append(
            _exact_qualification(route_binding, route_binding_hash, recipe)
        )
        resource["reportHash"] = _canonical_hash(
            resource,
            field="reportHash",
            prefix=_RESOURCE_REPORT_HASH_PREFIX,
        )

        result = self.audit(QWEN, resource=resource)
        self.assertEqual(result["evidence"]["resourceRecipes"]["status"], "exact_route_qualified")
        self.assertFalse(result["flags"]["autoEligible"])
        self.assertIn(
            "auto_publication_authority_missing",
            {reason["code"] for reason in result["blockers"]},
        )

    def test_exact_qwen_route_receipt_cannot_qualify_a_sibling_qwen_workflow(self):
        resource = copy.deepcopy(self.resource)
        qwen_admission = next(
            admission
            for definition in self.library["definitions"]
            for admission in definition.get("executionAdmissions", ())
            if admission["id"] == QWEN
        )
        sibling = next(
            admission
            for definition in self.library["definitions"]
            for admission in definition.get("executionAdmissions", ())
            if admission["id"] != QWEN
            and admission.get("artifact") == qwen_admission["artifact"]
            and admission["id"] in REGISTERED_BLOCK_V2_DEFINITION_PINS
        )
        recipe = next(
            recipe
            for recipe in resource["recipes"]
            if recipe["modelType"] == "QwenImageModularPipeline" and recipe["status"] == "qualified"
        )
        binding = _route_binding(qwen_admission)
        binding_hash = _binding_hash(binding)
        resource["routeQualifications"].append(
            _exact_qualification(binding, binding_hash, recipe)
        )
        resource["reportHash"] = _canonical_hash(
            resource,
            field="reportHash",
            prefix=_RESOURCE_REPORT_HASH_PREFIX,
        )

        result = self.audit(sibling["id"], resource=resource)
        self.assertEqual(result["evidence"]["resourceRecipes"]["exactRouteQualifiedCount"], 0)
        self.assertNotEqual(result["evidence"]["resourceRecipes"]["status"], "exact_route_qualified")

    def test_legacy_v1_resource_report_without_route_lane_normalizes_to_unbound(self):
        legacy = copy.deepcopy(self.resource)
        legacy.pop("routeQualifications")
        legacy["reportHash"] = _canonical_hash(
            legacy,
            field="reportHash",
            prefix=_RESOURCE_REPORT_HASH_PREFIX,
        )
        normalized = validate_resource_recipe_coverage(legacy)
        self.assertEqual(normalized["routeQualifications"], [])

    def test_family_recipe_cannot_impersonate_route_by_copying_only_admission_and_spec(self):
        resource = copy.deepcopy(self.resource)
        admission = next(
            admission
            for definition in self.library["definitions"]
            for admission in definition.get("executionAdmissions", ())
            if admission["id"] == QWEN
        )
        route_binding = _route_binding(
            admission,
            definition_pin=("block-definition-v2-deadbeef", f"sha256:{'0' * 64}"),
        )
        binding_hash = _binding_hash(route_binding)
        family_recipe = next(
            recipe
            for recipe in resource["recipes"]
            if recipe["modelType"] == "QwenImageModularPipeline" and recipe["status"] == "qualified"
        )
        resource["routeQualifications"].append(
            _exact_qualification(route_binding, binding_hash, family_recipe)
        )
        resource["reportHash"] = _canonical_hash(
            resource,
            field="reportHash",
            prefix=_RESOURCE_REPORT_HASH_PREFIX,
        )

        with self.assertRaisesRegex(ClusterPublicationAuditError, "current pinned BlockDefinitionV2"):
            self.audit(QWEN, resource=resource)

    def test_forged_resource_report_hash_fails_closed(self):
        resource = copy.deepcopy(self.resource)
        resource["recipes"][0]["modelType"] = "ForgedPipeline"
        with self.assertRaisesRegex(ClusterPublicationAuditError, "hash"):
            validate_resource_recipe_coverage(resource)

    def test_route_qualification_requires_complete_v2_and_collision_resistant_identity(self):
        resource = copy.deepcopy(self.resource)
        resource["routeQualifications"].append(
            {
                "routeBinding": {
                    "schemaVersion": 1,
                    "admissionId": QWEN,
                    "blockDefinition": {
                        "definitionId": QWEN,
                        "contentHash": "block-definition-v2-1234abcd",
                        # A short public V2 content hash alone is not enough to
                        # bind publication evidence collision-resistently.
                        "canonicalSha256": "block-definition-v2-1234abcd",
                    },
                    "studioExecutionSpec": {
                        "id": "qwen:test:v1",
                        "contentHash": "studio-spec-v1-test",
                        "executionProfileId": "qwen:test",
                    },
                    "artifact": {
                        "repository": "Qwen/Qwen-Image-2512",
                        "revision": "a" * 40,
                    },
                    "modelDependencies": [],
                },
                "routeBindingHash": f"{_RESOURCE_ROUTE_BINDING_HASH_PREFIX}{'0' * 64}",
                "recipe": {
                    "modelType": "QwenImageModularPipeline",
                    "recipeHash": "sha256:resource-recipe-v2:test",
                },
                "evidence": {
                    "bindingStatus": "exact_route_bound",
                    "routeBindingHash": f"{_RESOURCE_ROUTE_BINDING_HASH_PREFIX}{'0' * 64}",
                },
            }
        )
        resource["reportHash"] = _canonical_hash(
            resource,
            field="reportHash",
            prefix=_RESOURCE_REPORT_HASH_PREFIX,
        )

        with self.assertRaisesRegex(ClusterPublicationAuditError, "malformed|identity"):
            validate_resource_recipe_coverage(resource)

    def test_exact_route_resource_lane_requires_two_distinct_complete_proofs(self):
        admission = next(
            admission
            for definition in self.library["definitions"]
            for admission in definition.get("executionAdmissions", ())
            if admission["id"] == QWEN
        )
        family_recipe = next(
            recipe
            for recipe in self.resource["recipes"]
            if recipe["modelType"] == "QwenImageModularPipeline" and recipe["status"] == "qualified"
        )
        binding = _route_binding(admission)
        qualification = _exact_qualification(binding, _binding_hash(binding), family_recipe)

        for mutate in (
            lambda value: value["evidence"]["proofs"].pop(),
            lambda value: value["evidence"]["taskIds"].__setitem__(1, value["evidence"]["taskIds"][0]),
            lambda value: value["runtimeContract"].__setitem__(
                "runtimeLockFingerprint", "sha256:stable-runtime-v1:other"
            ),
            lambda value: value.__setitem__("workloadHash", "sha256:resource-workload-v1:short"),
            lambda value: value["evidence"]["proofs"][0].__setitem__("peakMemoryBytes", 0),
        ):
            resource = copy.deepcopy(self.resource)
            candidate = copy.deepcopy(qualification)
            mutate(candidate)
            resource["routeQualifications"].append(candidate)
            resource["reportHash"] = _canonical_hash(
                resource,
                field="reportHash",
                prefix=_RESOURCE_REPORT_HASH_PREFIX,
            )
            with self.assertRaisesRegex(ClusterPublicationAuditError, "identity"):
                validate_resource_recipe_coverage(resource)

    def test_parallel_live_proof_flag_cannot_bypass_exact_receipt(self):
        library = copy.deepcopy(self.library)
        admission = next(
            admission
            for definition in library["definitions"]
            for admission in definition.get("executionAdmissions", ())
            if admission["id"] == WAN
        )
        admission["publication"]["liveProof"] = True
        with self.assertRaisesRegex(ClusterPublicationAuditError, "liveProof"):
            audit_cluster_publication_route(
                WAN,
                library=library,
                resource_coverage=self.resource,
                release_report=self.release,
            )


if __name__ == "__main__":
    unittest.main()
