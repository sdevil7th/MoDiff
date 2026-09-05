import copy
import re
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from modiff.auto_resource import artifact_revision_cache_status
from modiff.block_definition_v2 import (
    block_definition_canonical_sha256_v2,
    block_definition_content_hash_v2,
    block_execution_parameter_hash_v2,
    block_graph_hash_v2,
    block_interface_hash_v2,
    validate_block_instance_v2,
)
from modiff.huggingface_cluster_runtime import (
    REGISTERED_BLOCK_V2_DEFINITION_PINS,
    qualify_huggingface_cluster_auto_authority,
    qualify_huggingface_cluster_expert_runtime,
)
from modiff.huggingface_node_library import build_huggingface_node_library


WAN_REPO = "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers"
WAN_REVISION = "17c30769b1e0b5dcaa1799b117bf20a9c31f59d7"
SMOLLM_REPO = "HuggingFaceTB/SmolLM2-135M-Instruct"
SMOLLM_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
ERNIE_REPO = "baidu/ERNIE-Image-Turbo"
ERNIE_REVISION = "bc68c81e2a1730a394d5fc9fae70713dee940140"
LTX2_REPO = "Lightricks/LTX-2"
LTX2_REVISION = "47da56e2ad66ce4125a9922b4a8826bf407f9d0a"
MINIMAX_MUSIC3_REPO = "MiniMaxAI/MiniMax-Music3"
MINIMAX_MUSIC3_REVISION = "fbdf52fbaaca799592917417eb05f1899f1255ec"
WAN_22_I2V_REPO = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
WAN_22_I2V_REVISION = "596658fd9ca6b7b71d5057529bbf319ecbc61d74"
WAN_22_TI2V_REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
WAN_22_TI2V_REVISION = "b8fff7315c768468a5333511427288870b2e9635"
QWEN_IMAGE_REPO = "Qwen/Qwen-Image"
QWEN_IMAGE_REVISION = "75e0b4be04f60ec59a75f475837eced720f823b6"
GIB = 1024**3


def registered_v2_instance(definition, admission):
    graph = {
        "nodes": [
            {
                "nodeId": "generate",
                "nodeType": "HuggingFaceTransformers.TextGeneration",
                "data": {"params": {}},
                "semanticRole": "generate",
            }
        ],
        "edges": [],
        "executionOrder": ["generate"],
        "graphHash": "pending",
    }
    graph["graphHash"] = block_graph_hash_v2(graph)
    controls = [
        {
            "controlId": control_id,
            "label": control_id,
            "binding": {"nodeId": "generate", "fieldId": control_id},
            "valueType": value_type,
            "defaultValue": default,
            "order": order,
        }
        for order, (control_id, value_type, default) in enumerate(
            (
                ("device", "string", "cpu"),
                ("dtype", "string", "float32"),
                ("quantizationMode", "string", "none"),
                ("autoOffload", "bool", False),
                ("offloadMode", "string", "none"),
            )
        )
    ]
    boundary = {
        "mode": "explicit",
        "inputs": [],
        "outputs": [
            {
                "portId": "text",
                "label": "Text",
                "valueType": "string",
                "required": False,
                "binding": {"nodeId": "generate", "fieldOrPortId": "text"},
            }
        ],
    }
    preview = {"nodeId": "generate", "outputPortId": "text", "mediaType": "text", "primary": True}
    source = {
        "kind": "transformers_catalog",
        "catalogCategory": "transformers",
        "provider": definition["publisher"],
        "library": "transformers",
        "libraryRevision": definition["libraryRevision"],
        "pipelineClass": definition["pipelineClass"],
        "blocksClass": definition["blocksClass"],
        "workflow": definition["workflowId"],
        "manifestDefinitionId": definition["id"],
        "manifestContentHash": definition["contentHash"],
        "executionAdmissionId": admission["id"],
        "repository": admission["artifact"]["repo"],
        "repositoryRevision": admission["artifact"]["revision"],
    }
    snapshot = {
        "schemaVersion": 2,
        "definitionId": admission["id"],
        "displayName": definition["label"],
        "contentHash": "pending",
        "source": source,
        "graph": graph,
        "boundary": boundary,
        "controls": controls,
        "previews": [preview],
        "ownership": {"kind": "registered", "definitionMutable": False},
    }
    snapshot["contentHash"] = block_definition_content_hash_v2(snapshot)
    interface_hash = block_interface_hash_v2(snapshot)
    return {
        "schemaVersion": 2,
        "instanceId": "registered-auto-instance",
        "definitionRef": {"definitionId": admission["id"], "contentHash": snapshot["contentHash"]},
        "definitionSnapshot": snapshot,
        "effectiveGraph": copy.deepcopy(graph),
        "effectiveInterface": {
            "boundary": copy.deepcopy(boundary),
            "controls": copy.deepcopy(controls),
            "baseInterfaceHash": interface_hash,
            "effectiveInterfaceHash": interface_hash,
        },
        "values": {
            "device": "cpu",
            "dtype": "float32",
            "quantizationMode": "none",
            "autoOffload": False,
            "offloadMode": "none",
        },
        "customization": {
            "state": "parameters_changed",
            "baseGraphHash": graph["graphHash"],
            "effectiveGraphHash": graph["graphHash"],
        },
        "presentation": {
            "expanded": False,
            "position": {"x": 0, "y": 0},
            "size": {"width": 360, "height": 320},
            "internalLayout": {"generate": {"x": 0, "y": 0}},
        },
        "previewStates": [{"binding": preview, "status": "idle"}],
        "authorities": [],
    }


def runtime_fingerprint(*, system_ram_gib=None, disk_free_gib=None):
    output = {
        "fingerprint": f"sha256:{'a' * 64}",
        "resourceFingerprint": f"sha256:{'b' * 64}",
        "torch": {
            "cuda_available": True,
            "cuda_device_count": 1,
            "xpu_available": False,
            "xpu_device_count": 0,
            "mps_available": False,
        },
    }
    if system_ram_gib is not None and disk_free_gib is not None:
        output["hardware"] = {
            "schema_version": 2,
            "system": {"ram_total": system_ram_gib * GIB},
            "devices": [
                {
                    "type": "cuda",
                    "memory_kind": "shared",
                    "planning_memory_total": 2 * GIB,
                    "shared_memory_total": 96 * GIB,
                    "torch_vram_total": 96 * GIB,
                }
            ],
            "disk": {"free_bytes": disk_free_gib * GIB},
        }
    return output


def active_optional_runtime():
    return {
        "schemaVersion": 1,
        "delivery": "optional_overlay",
        "requiredNow": True,
        "profileIds": ["huggingface-transformers-main-96fe6dce-peft-0.20.0"],
        "executionProfileIds": ["wan-flf:modular"],
        "state": "active",
        "reason": "optional_runtime_active",
    }


def expert_request():
    return {
        "schemaVersion": 1,
        "definitionId": "diffusers.modular:WanImage2VideoModularPipeline:flf2v",
        "admissionId": (
            "diffusers.cluster-admission:WanImage2VideoModularPipeline:flf2v:state_flow:flf2v"
        ),
        "resourceMode": "expert",
        "recipe": {
            "device": "cuda:0",
            "dtype": "bfloat16",
            "quantizationMode": "none",
            "autoOffload": True,
            "offloadMode": "model_cpu",
        },
    }


class HuggingFaceClusterRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.cache_dir = Path(self.temporary.name)
        snapshot = (
            self.cache_dir
            / "models--Wan-AI--Wan2.1-FLF2V-14B-720P-diffusers"
            / "snapshots"
            / WAN_REVISION
        )
        snapshot.mkdir(parents=True)
        (snapshot / "diffusion_pytorch_model.safetensors").write_bytes(b"reviewed-weight-fixture")
        self.local_models = [
            {
                "id": WAN_REPO,
                "cache_dirs": [str(self.cache_dir)],
                "revisions": [{"hash": WAN_REVISION}],
            }
        ]

    def test_auto_definition_pins_match_every_client_registered_v2_route(self):
        client_studio = Path(__file__).resolve().parents[2] / "MoDiff-client" / "src" / "studio"
        route_files = [
            client_studio / "registeredBlockV2Routes.ts",
            client_studio / "registeredBlockV2FanOutRoutes.ts",
        ]
        if any(not path.is_file() for path in route_files):
            self.skipTest("Sibling MoDiff-client checkout is required for the registered V2 pin parity gate.")
        source = "\n".join(path.read_text(encoding="utf-8") for path in route_files)
        patterns = (
            re.compile(
                r"admissionId:\s*'([^']+)'.*?"
                r"compiledDefinitionContentHash:\s*'([^']+)'.*?"
                r"compiledDefinitionCanonicalSha256:\s*'([^']+)'",
                re.DOTALL,
            ),
            re.compile(
                r'"admissionId":"([^"]+)".*?'
                r'"compiledDefinitionContentHash":"([^"]+)".*?'
                r'"compiledDefinitionCanonicalSha256":"([^"]+)"',
                re.DOTALL,
            ),
        )
        client_pins = {
            admission_id: (content_hash, canonical_sha256)
            for pattern in patterns
            for admission_id, content_hash, canonical_sha256 in pattern.findall(source)
        }
        self.assertEqual(len(client_pins), 90)
        self.assertEqual(dict(REGISTERED_BLOCK_V2_DEFINITION_PINS), client_pins)

    def test_transformers_composite_receipt_joins_direct_profile_and_optional_runtime(self):
        snapshot = (
            self.cache_dir
            / "models--HuggingFaceTB--SmolLM2-135M-Instruct"
            / "snapshots"
            / SMOLLM_REVISION
        )
        snapshot.mkdir(parents=True)
        (snapshot / "model.safetensors").write_bytes(b"reviewed-transformers-weight-fixture")
        local_models = [
            {
                "id": SMOLLM_REPO,
                "cache_dirs": [str(self.cache_dir)],
                "revisions": [{"hash": SMOLLM_REVISION}],
            }
        ]
        receipt = qualify_huggingface_cluster_expert_runtime(
            {
                "schemaVersion": 1,
                "definitionId": "transformers.composite:HuggingFaceTextGenerationModel:text_generation",
                "admissionId": "transformers.cluster-admission:HuggingFaceTextGenerationModel:text_generation",
                "resourceMode": "expert",
                "recipe": {
                    "device": "cpu",
                    "dtype": "float32",
                    "quantizationMode": "none",
                    "autoOffload": False,
                    "offloadMode": "none",
                },
            },
            library=build_huggingface_node_library(),
            runtime_fingerprint=runtime_fingerprint(),
            local_models=local_models,
            optional_runtime_requirement={
                "schemaVersion": 1,
                "delivery": "optional_overlay",
                "requiredNow": True,
                "profileIds": ["huggingface-transformers-peft-5.14.1-0.20.0"],
                "executionProfileIds": ["smollm2-135m-instruct:direct"],
                "state": "active",
                "reason": "optional_runtime_active",
            },
        )

        self.assertEqual(receipt["executionProfileId"], "smollm2-135m-instruct:direct")
        self.assertEqual(receipt["pipelineClass"], "AutoModelForCausalLM")
        self.assertEqual(receipt["loaderModule"], "modules.HuggingFaceTransformers")
        self.assertEqual(receipt["executionPath"], "direct-huggingface-transformers-text")
        self.assertTrue(receipt["artifactStatus"]["exactRevisionComplete"])

    def test_auto_planner_issues_short_lived_receipt_for_exact_v2_instance(self):
        snapshot = (
            self.cache_dir
            / "models--HuggingFaceTB--SmolLM2-135M-Instruct"
            / "snapshots"
            / SMOLLM_REVISION
        )
        snapshot.mkdir(parents=True)
        (snapshot / "model.safetensors").write_bytes(b"reviewed-transformers-weight-fixture")
        local_models = [
            {
                "id": SMOLLM_REPO,
                "cache_dirs": [str(self.cache_dir)],
                "revisions": [{"hash": SMOLLM_REVISION}],
            }
        ]
        library = build_huggingface_node_library()
        definition = next(
            item
            for item in library["definitions"]
            if item["id"] == "transformers.composite:HuggingFaceTextGenerationModel:text_generation"
        )
        admission = next(
            item
            for item in definition["executionAdmissions"]
            if item["id"] == "transformers.cluster-admission:HuggingFaceTextGenerationModel:text_generation"
        )
        spec = admission["studioExecutionSpec"]
        candidate = {
            "id": "smollm2-135m-instruct-cpu",
            "modelType": definition["pipelineClass"],
            "mode": admission["studioMode"],
            "executionProfileId": spec["executionProfileId"],
            "loaderModule": "modules.HuggingFaceTransformers",
            "loaderAction": "LoadTextGenerationModel",
            "executionPath": "direct-huggingface-transformers-text",
            "pipelineClass": definition["blocksClass"],
            "resolvedArtifact": SMOLLM_REPO,
            "artifactRevision": SMOLLM_REVISION,
            "artifactResolution": {
                "resolved": {"repo": SMOLLM_REPO, "revision": SMOLLM_REVISION}
            },
            "studioExecutionSpecContract": {
                "id": spec["id"],
                "contentHash": spec["contentHash"],
                "executionProfileId": spec["executionProfileId"],
            },
            "modelDependencies": admission["modelDependencies"],
            "dtype": "float32",
            "quantizationMode": "none",
            "autoOffload": False,
            "offloadMode": "none",
            "optionalRuntimeRequirement": {
                "schemaVersion": 1,
                "delivery": "optional_overlay",
                "requiredNow": True,
                "profileIds": ["huggingface-transformers-peft-5.14.1-0.20.0"],
                "executionProfileIds": [spec["executionProfileId"]],
                "state": "active",
                "reason": "optional_runtime_active",
            },
        }
        instance = registered_v2_instance(definition, admission)
        pinned_definitions = {
            admission["id"]: (
                instance["definitionSnapshot"]["contentHash"],
                block_definition_canonical_sha256_v2(instance["definitionSnapshot"]),
            )
        }
        payload = {
            "schemaVersion": 1,
            "instance": instance,
            "form": {
                "modelType": definition["pipelineClass"],
                "mode": admission["studioMode"],
                "device": "cpu",
                "dtype": "float32",
                "quantizationMode": "none",
                "autoOffload": False,
                "offloadMode": "none",
                "width": 512,
                "height": 512,
                "steps": 1,
                "numFrames": 1,
            },
        }
        with patch(
            "modiff.huggingface_cluster_runtime.build_auto_resource_plan",
            return_value={"canAutoRun": True, "selectedCandidate": candidate},
        ):
            receipt = qualify_huggingface_cluster_auto_authority(
                payload,
                library=library,
                runtime_fingerprint=runtime_fingerprint(),
                local_models=local_models,
                data_dir=str(self.cache_dir),
                registered_definition_pins=pinned_definitions,
            )

        self.assertEqual(receipt["kind"], "auto")
        self.assertEqual(receipt["definitionId"], admission["id"])
        self.assertEqual(receipt["admissionId"], admission["id"])
        self.assertEqual(receipt["definitionContentHash"], instance["definitionSnapshot"]["contentHash"])
        self.assertEqual(receipt["effectiveGraphHash"], instance["effectiveGraph"]["graphHash"])
        self.assertEqual(receipt["executionParameterHash"], block_execution_parameter_hash_v2(instance))
        self.assertEqual(receipt["artifactRevisions"], {SMOLLM_REPO: SMOLLM_REVISION})
        self.assertLess(receipt["issuedAt"], receipt["expiresAt"])
        self.assertRegex(receipt["issuedAt"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")
        self.assertRegex(receipt["expiresAt"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")
        # The authority endpoint must return a receipt that round-trips
        # through the exact persisted BlockInstanceV2 validator.  This is the
        # same attachment the browser performs before submitting /graph.
        validated = validate_block_instance_v2({**instance, "authorities": [receipt]})
        self.assertEqual(validated["authorities"], [receipt])

        collision_pin = {
            admission["id"]: (
                instance["definitionSnapshot"]["contentHash"],
                f"sha256:{'0' * 64}",
            )
        }
        with self.assertRaisesRegex(ValueError, "backend-pinned SHA-256"):
            qualify_huggingface_cluster_auto_authority(
                payload,
                library=library,
                runtime_fingerprint=runtime_fingerprint(),
                local_models=local_models,
                data_dir=str(self.cache_dir),
                registered_definition_pins=collision_pin,
            )
        with self.assertRaisesRegex(ValueError, "backend-pinned compiler output"):
            qualify_huggingface_cluster_auto_authority(
                payload,
                library=library,
                runtime_fingerprint=runtime_fingerprint(),
                local_models=local_models,
                data_dir=str(self.cache_dir),
                registered_definition_pins={},
            )

        substituted = {**candidate, "resolvedArtifact": "someone/unsafe-substitute"}
        with (
            patch(
                "modiff.huggingface_cluster_runtime.build_auto_resource_plan",
                return_value={"canAutoRun": True, "selectedCandidate": substituted},
            ),
            self.assertRaisesRegex(ValueError, "substitutes another artifact"),
        ):
            qualify_huggingface_cluster_auto_authority(
                payload,
                library=library,
                runtime_fingerprint=runtime_fingerprint(),
                local_models=local_models,
                data_dir=str(self.cache_dir),
                registered_definition_pins=pinned_definitions,
            )

        opaque_hash_substitution = {**payload, "effectiveGraphHash": "block-graph-v2-deadbeef"}
        with self.assertRaisesRegex(
            ValueError,
            r"^Cannot qualify Hugging Face Cluster Node Auto execution: the authority request shape is invalid\.$",
        ):
            qualify_huggingface_cluster_auto_authority(
                opaque_hash_substitution,
                library=library,
                runtime_fingerprint=runtime_fingerprint(),
                local_models=local_models,
                data_dir=str(self.cache_dir),
                registered_definition_pins=pinned_definitions,
            )

        omitted_resource_field = copy.deepcopy(payload)
        omitted_resource_field["form"].pop("width")
        with self.assertRaisesRegex(ValueError, "omits resource-impacting field.*width"):
            qualify_huggingface_cluster_auto_authority(
                omitted_resource_field,
                library=library,
                runtime_fingerprint=runtime_fingerprint(),
                local_models=local_models,
                data_dir=str(self.cache_dir),
                registered_definition_pins=pinned_definitions,
            )

        graph_substitution = copy.deepcopy(payload)
        graph = graph_substitution["instance"]["effectiveGraph"]
        graph["nodes"][0]["data"]["params"]["unsafe"] = {"value": True}
        graph["graphHash"] = block_graph_hash_v2(graph)
        graph_substitution["instance"]["customization"].update(
            {"state": "structure_changed", "effectiveGraphHash": graph["graphHash"]}
        )
        with self.assertRaisesRegex(ValueError, "structurally or publicly customized"):
            qualify_huggingface_cluster_auto_authority(
                graph_substitution,
                library=library,
                runtime_fingerprint=runtime_fingerprint(),
                local_models=local_models,
                data_dir=str(self.cache_dir),
                registered_definition_pins=pinned_definitions,
            )

        interface_substitution = copy.deepcopy(payload)
        interface = interface_substitution["instance"]["effectiveInterface"]
        interface["boundary"]["outputs"][0]["label"] = "Substituted output"
        interface["effectiveInterfaceHash"] = block_interface_hash_v2(interface)
        interface_substitution["instance"]["customization"]["state"] = "structure_changed"
        with self.assertRaisesRegex(ValueError, "structurally or publicly customized"):
            qualify_huggingface_cluster_auto_authority(
                interface_substitution,
                library=library,
                runtime_fingerprint=runtime_fingerprint(),
                local_models=local_models,
                data_dir=str(self.cache_dir),
                registered_definition_pins=pinned_definitions,
            )

        value_substitution = copy.deepcopy(payload)
        value_substitution["instance"]["values"]["dtype"] = "bfloat16"
        with self.assertRaisesRegex(ValueError, "form field dtype"):
            qualify_huggingface_cluster_auto_authority(
                value_substitution,
                library=library,
                runtime_fingerprint=runtime_fingerprint(),
                local_models=local_models,
                data_dir=str(self.cache_dir),
                registered_definition_pins=pinned_definitions,
            )

        definition_substitution = copy.deepcopy(payload)
        snapshot = definition_substitution["instance"]["definitionSnapshot"]
        snapshot["graph"]["nodes"][0]["data"]["params"]["unsafe"] = {"value": True}
        snapshot["graph"]["graphHash"] = block_graph_hash_v2(snapshot["graph"])
        snapshot["contentHash"] = block_definition_content_hash_v2(snapshot)
        definition_substitution["instance"]["definitionRef"]["contentHash"] = snapshot["contentHash"]
        definition_substitution["instance"]["effectiveGraph"] = copy.deepcopy(snapshot["graph"])
        definition_substitution["instance"]["customization"].update(
            {
                "baseGraphHash": snapshot["graph"]["graphHash"],
                "effectiveGraphHash": snapshot["graph"]["graphHash"],
            }
        )
        with self.assertRaisesRegex(ValueError, "backend-pinned compiler output"):
            qualify_huggingface_cluster_auto_authority(
                definition_substitution,
                library=library,
                runtime_fingerprint=runtime_fingerprint(),
                local_models=local_models,
                data_dir=str(self.cache_dir),
                registered_definition_pins=pinned_definitions,
            )

    def test_standard_diffusers_composite_receipt_keeps_the_direct_pipeline_identity(self):
        snapshot = (
            self.cache_dir
            / "models--Wan-AI--Wan2.2-TI2V-5B-Diffusers"
            / "snapshots"
            / WAN_22_TI2V_REVISION
        )
        snapshot.mkdir(parents=True)
        (snapshot / "model.safetensors").write_bytes(b"reviewed-wan-5b-weight-fixture")
        receipt = qualify_huggingface_cluster_expert_runtime(
            {
                "schemaVersion": 1,
                "definitionId": "diffusers.composite:WanTI2VPipeline:text_to_video",
                "admissionId": "diffusers.cluster-admission:WanTI2VPipeline:text_to_video:mode:text_to_video",
                "resourceMode": "expert",
                "recipe": {
                    "device": "cuda:0",
                    "dtype": "bfloat16",
                    "quantizationMode": "none",
                    "autoOffload": True,
                    "offloadMode": "group_disk",
                },
            },
            library=build_huggingface_node_library(),
            runtime_fingerprint=runtime_fingerprint(),
            local_models=[
                {
                    "id": WAN_22_TI2V_REPO,
                    "cache_dirs": [str(self.cache_dir)],
                    "revisions": [{"hash": WAN_22_TI2V_REVISION}],
                }
            ],
            optional_runtime_requirement={
                "schemaVersion": 1,
                "delivery": "main_runtime",
                "requiredNow": False,
                "profileIds": [],
                "executionProfileIds": [],
                "state": "not_required",
                "reason": "main_runtime_route",
            },
        )

        self.assertEqual(receipt["modelType"], "WanTI2VPipeline")
        self.assertEqual(receipt["pipelineClass"], "WanTI2VPipeline")
        self.assertEqual(receipt["loaderModule"], "modules.DiffusersVideo")
        self.assertEqual(receipt["executionPath"], "direct-diffusers-video")
        self.assertTrue(receipt["artifactStatus"]["exactRevisionComplete"])

    def test_qwen_same_family_expert_receipt_uses_the_selected_exact_artifact(self):
        snapshot = (
            self.cache_dir
            / "models--Qwen--Qwen-Image"
            / "snapshots"
            / QWEN_IMAGE_REVISION
        )
        snapshot.mkdir(parents=True)
        (snapshot / "model_index.json").write_text("{}", encoding="utf-8")
        (snapshot / "diffusion_pytorch_model.safetensors").write_bytes(b"reviewed-qwen-base-weight-fixture")
        local_models = [
            {
                "id": QWEN_IMAGE_REPO,
                "cache_dirs": [str(self.cache_dir)],
                "revisions": [{"hash": QWEN_IMAGE_REVISION}],
            }
        ]
        request = {
            "schemaVersion": 1,
            "definitionId": "diffusers.modular:QwenImageModularPipeline:text2image",
            "admissionId": (
                "diffusers.cluster-admission:QwenImageModularPipeline:text2image:mode:text_to_image"
            ),
            "artifactRepo": QWEN_IMAGE_REPO,
            "resourceMode": "expert",
            "recipe": {
                "device": "cuda:0",
                "dtype": "bfloat16",
                "quantizationMode": "none",
                "autoOffload": True,
                "offloadMode": "model_cpu",
            },
        }
        receipt = qualify_huggingface_cluster_expert_runtime(
            request,
            library=build_huggingface_node_library(),
            runtime_fingerprint=runtime_fingerprint(),
            local_models=local_models,
            optional_runtime_requirement={
                "schemaVersion": 1,
                "delivery": "base",
                "requiredNow": False,
                "profileIds": [],
                "executionProfileIds": [],
                "state": "base_satisfied",
                "reason": "base_runtime_contract",
            },
        )

        self.assertEqual(receipt["artifactStatus"]["repo"], QWEN_IMAGE_REPO)
        self.assertEqual(receipt["artifactStatus"]["revision"], QWEN_IMAGE_REVISION)
        self.assertEqual(receipt["pipelineClass"], "QwenImageModularPipeline")

        request["artifactRepo"] = "Qwen/Qwen-Image-Edit-2511"
        with self.assertRaisesRegex(ValueError, "not an admitted same-pipeline variant"):
            qualify_huggingface_cluster_expert_runtime(
                request,
                library=build_huggingface_node_library(),
                runtime_fingerprint=runtime_fingerprint(),
                local_models=local_models,
                optional_runtime_requirement={
                    "schemaVersion": 1,
                    "delivery": "base",
                    "requiredNow": False,
                    "profileIds": [],
                    "executionProfileIds": [],
                    "state": "base_satisfied",
                    "reason": "base_runtime_contract",
                },
            )

    def test_equivalent_standard_diffusers_receipt_keeps_modular_identity_and_direct_executor(self):
        snapshot = self.cache_dir / "models--baidu--ERNIE-Image-Turbo" / "snapshots" / ERNIE_REVISION
        snapshot.mkdir(parents=True)
        (snapshot / "model.safetensors").write_bytes(b"reviewed-ernie-weight-fixture")
        receipt = qualify_huggingface_cluster_expert_runtime(
            {
                "schemaVersion": 1,
                "definitionId": "diffusers.modular:ErnieImageModularPipeline:text2image",
                "admissionId": (
                    "diffusers.cluster-admission:ErnieImageModularPipeline:text2image:"
                    "mode:equivalent_standard_route"
                ),
                "resourceMode": "expert",
                "recipe": {
                    "device": "cpu",
                    "dtype": "float32",
                    "quantizationMode": "none",
                    "autoOffload": False,
                    "offloadMode": "none",
                },
            },
            library=build_huggingface_node_library(),
            runtime_fingerprint=runtime_fingerprint(),
            local_models=[
                {
                    "id": ERNIE_REPO,
                    "cache_dirs": [str(self.cache_dir)],
                    "revisions": [{"hash": ERNIE_REVISION}],
                }
            ],
            optional_runtime_requirement={
                "schemaVersion": 1,
                "delivery": "base",
                "requiredNow": False,
                "profileIds": [],
                "executionProfileIds": [],
                "state": "base_satisfied",
                "reason": "base_runtime_contract",
            },
        )

        self.assertEqual(receipt["modelType"], "ErnieImageModularPipeline")
        self.assertEqual(receipt["pipelineClass"], "ErnieImagePipeline")
        self.assertEqual(receipt["loaderModule"], "modules.DiffusersImage")
        self.assertEqual(receipt["loaderAction"], "LoadPipeline")
        self.assertEqual(receipt["executionPath"], "direct-diffusers-image")
        self.assertEqual(receipt["artifactStatus"]["revision"], ERNIE_REVISION)
        self.assertTrue(receipt["artifactStatus"]["exactRevisionComplete"])
        self.assertFalse(receipt["publicationExecutable"])

    def test_ltx2_equivalent_standard_receipt_accepts_the_sealed_condition_pipeline(self):
        snapshot = self.cache_dir / "models--Lightricks--LTX-2" / "snapshots" / LTX2_REVISION
        snapshot.mkdir(parents=True)
        (snapshot / "model_index.json").write_text("{}", encoding="utf-8")
        (snapshot / "diffusion_pytorch_model.safetensors").write_bytes(b"reviewed-ltx2-weight-fixture")
        receipt = qualify_huggingface_cluster_expert_runtime(
            {
                "schemaVersion": 1,
                "definitionId": "diffusers.modular:LTX2ModularPipeline:text2video",
                "admissionId": (
                    "diffusers.cluster-admission:LTX2ModularPipeline:text2video:"
                    "mode:equivalent_standard_route"
                ),
                "resourceMode": "expert",
                "recipe": {
                    "device": "cuda:0",
                    "dtype": "bfloat16",
                    "quantizationMode": "none",
                    "autoOffload": True,
                    "offloadMode": "model_cpu",
                },
            },
            library=build_huggingface_node_library(),
            runtime_fingerprint=runtime_fingerprint(),
            local_models=[
                {
                    "id": LTX2_REPO,
                    "cache_dirs": [str(self.cache_dir)],
                    "revisions": [{"hash": LTX2_REVISION}],
                }
            ],
            optional_runtime_requirement={
                "schemaVersion": 1,
                "delivery": "optional_overlay",
                "requiredNow": True,
                "profileIds": ["huggingface-transformers-main-96fe6dce-peft-0.20.0"],
                "executionProfileIds": ["ltx2-modular:equivalent-standard"],
                "state": "active",
                "reason": "optional_runtime_active",
            },
        )

        self.assertEqual(receipt["modelType"], "LTX2ModularPipeline")
        self.assertEqual(receipt["pipelineClass"], "LTX2ConditionPipeline")
        self.assertEqual(receipt["loaderModule"], "modules.DiffusersVideo")
        self.assertEqual(receipt["loaderAction"], "LoadPipeline")
        self.assertEqual(receipt["executionPath"], "direct-diffusers-video")
        self.assertEqual(receipt["artifactStatus"]["revision"], LTX2_REVISION)
        self.assertTrue(receipt["artifactStatus"]["exactRevisionComplete"])

    def test_reviewed_whole_modular_workflow_receipt_accepts_the_sealed_minimax_profile(self):
        snapshot = (
            self.cache_dir
            / "models--MiniMaxAI--MiniMax-Music3"
            / "snapshots"
            / MINIMAX_MUSIC3_REVISION
        )
        snapshot.mkdir(parents=True)
        (snapshot / "modular_model_index.json").write_text("{}", encoding="utf-8")
        (snapshot / "diffusion_pytorch_model.safetensors").write_bytes(b"reviewed-minimax-weight-fixture")
        receipt = qualify_huggingface_cluster_expert_runtime(
            {
                "schemaVersion": 1,
                "definitionId": "diffusers.modular:MiniMaxMusic3ModularPipeline:default",
                "admissionId": (
                    "diffusers.cluster-admission:MiniMaxMusic3ModularPipeline:default:"
                    "workflow:official_top_level_blocks"
                ),
                "resourceMode": "expert",
                "recipe": {
                    "device": "cuda:0",
                    "dtype": "bfloat16",
                    "quantizationMode": "none",
                    "autoOffload": True,
                    "offloadMode": "model_cpu",
                },
            },
            library=build_huggingface_node_library(),
            runtime_fingerprint=runtime_fingerprint(),
            local_models=[
                {
                    "id": MINIMAX_MUSIC3_REPO,
                    "cache_dirs": [str(self.cache_dir)],
                    "revisions": [{"hash": MINIMAX_MUSIC3_REVISION}],
                }
            ],
            optional_runtime_requirement={
                "schemaVersion": 1,
                "delivery": "base",
                "requiredNow": False,
                "profileIds": [],
                "executionProfileIds": [],
                "state": "base_satisfied",
                "reason": "base_runtime_contract",
            },
        )

        self.assertEqual(receipt["modelType"], "MiniMaxMusic3ModularPipeline")
        self.assertEqual(receipt["pipelineClass"], "MiniMaxMusic3ModularPipeline")
        self.assertEqual(receipt["executionProfileId"], "minimax-music3:official-modular-workflow")
        self.assertEqual(receipt["loaderModule"], "modules.ModularDiffusers")
        self.assertEqual(receipt["executionPath"], "modular-diffusers")
        self.assertTrue(receipt["artifactStatus"]["exactRevisionComplete"])

    def test_exact_revision_status_does_not_accept_another_repo_snapshot(self):
        accepted = artifact_revision_cache_status(WAN_REPO, WAN_REVISION, self.local_models)
        rejected = artifact_revision_cache_status(WAN_REPO, "c" * 40, self.local_models)

        self.assertTrue(accepted["exactRevisionComplete"])
        self.assertFalse(accepted["repairRequired"])
        self.assertFalse(rejected["exactRevisionComplete"])
        self.assertTrue(rejected["repairRequired"])

    def test_wan_expert_receipt_joins_graph_runtime_artifact_and_recipe(self):
        receipt = qualify_huggingface_cluster_expert_runtime(
            expert_request(),
            library=build_huggingface_node_library(),
            runtime_fingerprint=runtime_fingerprint(),
            local_models=self.local_models,
            optional_runtime_requirement=active_optional_runtime(),
        )

        self.assertEqual(receipt["claim"], "expert_cluster_runtime_qualified")
        self.assertFalse(receipt["publicationExecutable"])
        self.assertEqual(receipt["executionProfileId"], "wan-flf:modular")
        self.assertEqual(receipt["artifactStatus"]["revision"], WAN_REVISION)
        self.assertTrue(receipt["artifactStatus"]["exactRevisionComplete"])
        self.assertEqual(receipt["recipe"]["offloadMode"], "model_cpu")
        self.assertEqual(receipt["runtimeFingerprint"], f"sha256:{'a' * 64}")
        self.assertEqual(receipt["resourceFingerprint"], f"sha256:{'b' * 64}")

    def test_wan_22_a14b_rejects_unsafe_cpu_residency_and_accepts_disk_groups(self):
        snapshot = (
            self.cache_dir
            / "models--Wan-AI--Wan2.2-I2V-A14B-Diffusers"
            / "snapshots"
            / WAN_22_I2V_REVISION
        )
        snapshot.mkdir(parents=True)
        (snapshot / "model_index.json").write_text("{}", encoding="utf-8")
        (snapshot / "diffusion_pytorch_model.safetensors").write_bytes(b"reviewed-wan22-weight-fixture")
        local_models = [
            {
                "id": WAN_22_I2V_REPO,
                "cache_dirs": [str(self.cache_dir)],
                "revisions": [{"hash": WAN_22_I2V_REVISION}],
            }
        ]
        request = {
            "schemaVersion": 1,
            "definitionId": "diffusers.modular:Wan22Image2VideoModularPipeline:default",
            "admissionId": (
                "diffusers.cluster-admission:Wan22Image2VideoModularPipeline:default:"
                "mode:equivalent_standard_route"
            ),
            "resourceMode": "expert",
            "recipe": {
                "device": "cuda:0",
                "dtype": "bfloat16",
                "quantizationMode": "none",
                "autoOffload": True,
                "offloadMode": "model_cpu",
            },
        }
        optional_runtime = {
            "schemaVersion": 1,
            "delivery": "optional_overlay",
            "requiredNow": True,
            "profileIds": ["huggingface-transformers-main-96fe6dce-peft-0.20.0"],
            "executionProfileIds": ["wan22-i2v:equivalent-standard"],
            "state": "active",
            "reason": "optional_runtime_active",
        }

        with self.assertRaisesRegex(ValueError, "at least 160 GiB system RAM"):
            qualify_huggingface_cluster_expert_runtime(
                request,
                library=build_huggingface_node_library(),
                runtime_fingerprint=runtime_fingerprint(system_ram_gib=121, disk_free_gib=148),
                local_models=local_models,
                optional_runtime_requirement=optional_runtime,
            )

        request["recipe"]["offloadMode"] = "group_disk"
        receipt = qualify_huggingface_cluster_expert_runtime(
            request,
            library=build_huggingface_node_library(),
            runtime_fingerprint=runtime_fingerprint(system_ram_gib=121, disk_free_gib=148),
            local_models=local_models,
            optional_runtime_requirement=optional_runtime,
        )
        self.assertEqual(receipt["recipe"]["offloadMode"], "group_disk")

    def test_expert_receipt_rejects_stale_runtime_or_unreviewed_recipe(self):
        missing_runtime = {**active_optional_runtime(), "state": "missing"}
        with self.assertRaisesRegex(ValueError, "optional runtime is missing"):
            qualify_huggingface_cluster_expert_runtime(
                expert_request(),
                library=build_huggingface_node_library(),
                runtime_fingerprint=runtime_fingerprint(),
                local_models=self.local_models,
                optional_runtime_requirement=missing_runtime,
            )

        quantized = expert_request()
        quantized["recipe"]["quantizationMode"] = "bnb_4bit"
        with self.assertRaisesRegex(ValueError, "outside the reviewed Expert profile"):
            qualify_huggingface_cluster_expert_runtime(
                quantized,
                library=build_huggingface_node_library(),
                runtime_fingerprint=runtime_fingerprint(),
                local_models=self.local_models,
                optional_runtime_requirement=active_optional_runtime(),
            )


if __name__ == "__main__":
    unittest.main()
