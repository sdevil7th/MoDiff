import hashlib
import json
import unittest
from collections import defaultdict
from pathlib import Path

from modiff.model_artifact_catalog import catalog_repository_pin, catalog_revision


GRAPH_ROOT = Path(__file__).resolve().parents[1] / "data" / "graphs"
WORKFLOW_MANIFEST = GRAPH_ROOT.parent / "workflow-library-manifest.json"
OUTPUT_NODE_KEYS = {
    ("modules.Audio", "Export"),
    ("modules.Image", "Preview"),
    ("modules.Primitive", "DataViewer"),
    ("modules.Primitive", "ExportData"),
    ("modules.Video", "Export"),
    ("modules.Video", "ExportWithAudio"),
}
DIFFUSERS_PIPELINE_MODULES = {
    "modules.DiffusersAudio",
    "modules.DiffusersImage",
    "modules.DiffusersVideo",
}


def _field_value(field):
    value = (field or {}).get("value")
    if isinstance(value, dict):
        return str(value.get("value") or "")
    return str(value or "")


def _canonical_graph_digest(graph_path):
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    canonical = json.dumps(graph, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _loader_repository(node):
    data = node.get("data", {})
    params = data.get("params", {})
    if data.get("action") == "DynamicBlockNode":
        return _field_value(params.get("repo_id"))
    if data.get("action") == "ModelsLoader":
        return _field_value(params.get("repo_id"))
    return _field_value(params.get("model_id"))


def _is_curated_loader(node):
    data = node.get("data", {})
    return (data.get("module") in DIFFUSERS_PIPELINE_MODULES and data.get("action") == "LoadPipeline") or (
        data.get("module") == "modules.ModularDiffusers"
        and data.get("action") in {"ModelsLoader", "AutoModelLoader", "DynamicBlockNode"}
    )


def _node_key(node):
    data = (node or {}).get("data", {})
    return data.get("module"), data.get("action")


def _intentional_disabled_managed_fallback_node_ids(graph):
    """Return only the reversible base exporter replaced by soundtrack.v1.

    Studio keeps this one managed node so removing the controlled soundtrack
    block can restore the base video-export route.  The exception is valid only
    while the complete, enabled soundtrack route terminates at an active muxed
    exporter and the dormant base exporter is owned, disabled, and isolated.
    """

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    role_nodes = {}
    duplicate_roles = set()
    for node in nodes:
        role = node.get("data", {}).get("studioRole")
        if not isinstance(role, str):
            continue
        if role in role_nodes:
            duplicate_roles.add(role)
        role_nodes[role] = node

    soundtrack_roles = {
        "soundtrackQuantization": ("modules.DiffusersRuntime", "PipelineQuantizationConfigV2"),
        "soundtrackRecipe": ("modules.DiffusersRuntime", "DiffusersExecutionRecipe"),
        "soundtrackPipeline": ("modules.DiffusersAudio", "LoadPipeline"),
        "soundtrackGenerate": ("modules.DiffusersAudio", "Generate"),
        "soundtrackAudioFit": ("modules.Audio", "FitDuration"),
        "exportWithAudio": ("modules.Video", "ExportWithAudio"),
    }
    for role, expected_key in soundtrack_roles.items():
        node = role_nodes.get(role)
        if (
            role in duplicate_roles
            or _node_key(node) != expected_key
            or node.get("data", {}).get("studioOwned") is not True
            or node.get("data", {}).get("uiState", {}).get("disabled") is True
        ):
            return set()

    fallback = role_nodes.get("videoExport")
    if (
        "videoExport" in duplicate_roles
        or _node_key(fallback) != ("modules.Video", "Export")
        or fallback.get("data", {}).get("studioOwned") is not True
        or fallback.get("data", {}).get("uiState", {}).get("disabled") is not True
        or any(edge.get("source") == fallback.get("id") or edge.get("target") == fallback.get("id") for edge in edges)
    ):
        return set()

    role_ids = {role: role_nodes[role]["id"] for role in soundtrack_roles}

    def has_edge(source_role, source_handle, target_role, target_handle):
        return any(
            edge.get("source") == role_ids[source_role]
            and edge.get("sourceHandle") == source_handle
            and edge.get("target") == role_ids[target_role]
            and edge.get("targetHandle") == target_handle
            for edge in edges
        )

    exact_soundtrack_route = (
        has_edge("soundtrackQuantization", "quantization_config", "soundtrackRecipe", "quantization_config")
        and has_edge("soundtrackRecipe", "execution_recipe", "soundtrackPipeline", "execution_recipe")
        and has_edge("soundtrackPipeline", "pipeline", "soundtrackGenerate", "pipeline")
        and has_edge("soundtrackGenerate", "audio", "soundtrackAudioFit", "audio")
        and has_edge("soundtrackAudioFit", "output", "exportWithAudio", "audio")
    )
    exporter_edges = [edge for edge in edges if edge.get("target") == role_ids["exportWithAudio"]]
    video_edge = next((edge for edge in exporter_edges if edge.get("targetHandle") == "video"), None)
    video_source = next((node for node in nodes if node.get("id") == (video_edge or {}).get("source")), None)
    allowed_video_sources = {
        "wanGenerate": (("modules.DiffusersVideo", "Generate"), {"video_out"}),
        "videoCompose": (("modules.Video", "Compose"), {"video"}),
        "upscaler": (("modules.Spandrel", "Upscaler"), {"output", "image"}),
    }
    video_source_contract = allowed_video_sources.get((video_source or {}).get("data", {}).get("studioRole"))
    if (
        not exact_soundtrack_route
        or len(exporter_edges) != 2
        or video_source is None
        or video_source_contract is None
        or _node_key(video_source) != video_source_contract[0]
        or video_edge.get("sourceHandle") not in video_source_contract[1]
        or video_source.get("data", {}).get("studioOwned") is not True
        or video_source.get("data", {}).get("uiState", {}).get("disabled") is True
        or any(edge.get("source") == role_ids["exportWithAudio"] for edge in edges)
    ):
        return set()
    return {fallback["id"]}


class GraphCatalogIntegrityTests(unittest.TestCase):
    def test_ace_step_graphs_keep_the_positive_shift_contract(self):
        graph_dir = GRAPH_ROOT / "studio" / "ace-step-audio-pipeline"
        checked = 0
        for graph_path in sorted(graph_dir.glob("*.json")):
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
            for node in graph.get("nodes", []):
                data = node.get("data", {})
                if data.get("module") != "modules.DiffusersAudio" or data.get("action") != "Generate":
                    continue
                shift = data.get("params", {}).get("shift")
                if not isinstance(shift, dict):
                    continue
                checked += 1
                self.assertEqual(shift.get("min"), 0.1, graph_path.name)
                self.assertGreater(float(shift.get("value")), 0, graph_path.name)
        self.assertEqual(checked, 6)

    def test_curated_graphs_exclude_runtime_and_machine_state(self):
        node_runtime_fields = {"measured", "selected", "dragging"}
        measured_runtime_fields = {"memoryUsage", "executionTime"}
        for graph_path in sorted(GRAPH_ROOT.rglob("*.json")):
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
            graph_label = str(graph_path.relative_to(GRAPH_ROOT))
            self.assertNotIn("/cache/", json.dumps(graph), f"{graph_label} contains a runtime cache URL")
            for node in graph.get("nodes", []):
                self.assertEqual(
                    node_runtime_fields.intersection(node),
                    set(),
                    f"{graph_label}:{node.get('id')} contains editor runtime state",
                )
                data = node.get("data", {})
                self.assertEqual(measured_runtime_fields.intersection(data), set(), graph_label)
                self.assertFalse(data.get("isCached") is True, graph_label)
                self.assertFalse(data.get("cache") is True, graph_label)
                self.assertFalse(any(value != 0 for value in data.get("time", [])), graph_label)
                self.assertFalse(any(value != 0 for value in data.get("memory", [])), graph_label)
                device = data.get("params", {}).get("device", {})
                device_options = json.dumps(device.get("options", {}))
                for machine_field in ("arch", "name", "total_memory"):
                    self.assertNotIn(
                        f'"{machine_field}"',
                        device_options,
                        f"{graph_label}:{node.get('id')} contains host-specific device inventory",
                    )

    def test_hugging_face_graph_inputs_use_immutable_resolve_revisions(self):
        mutable_resolve_markers = ("/resolve/main/", "/resolve/master/")
        for graph_path in sorted(GRAPH_ROOT.rglob("*.json")):
            graph_text = graph_path.read_text(encoding="utf-8")
            for marker in mutable_resolve_markers:
                self.assertNotIn(
                    marker,
                    graph_text,
                    f"{graph_path.relative_to(GRAPH_ROOT)} contains a mutable Hugging Face URL",
                )

    def test_hub_diffusers_adapter_graphs_pin_revision_weight_and_digest(self):
        checked = defaultdict(int)
        for graph_path in sorted(GRAPH_ROOT.rglob("*.json")):
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
            for node in graph.get("nodes", []):
                data = node.get("data", {})
                module = data.get("module")
                if (
                    module not in {"modules.DiffusersAudio", "modules.DiffusersImage"}
                    or data.get("action") != "LoadAdapter"
                ):
                    continue
                params = data.get("params", {})
                selection = (params.get("adapter_path") or {}).get("value")
                if not isinstance(selection, dict) or str(selection.get("source") or "").casefold() != "hub":
                    continue
                checked[module] += 1
                revision = str((params.get("revision") or {}).get("value") or "")
                digest = str((params.get("expected_sha256") or {}).get("value") or "")
                weight_name = str((params.get("weight_name") or {}).get("value") or "")
                self.assertRegex(revision, r"^[0-9a-f]{40}$", str(graph_path.relative_to(GRAPH_ROOT)))
                self.assertRegex(digest, r"^[0-9a-f]{64}$", str(graph_path.relative_to(GRAPH_ROOT)))
                self.assertTrue(
                    weight_name.endswith(".safetensors"),
                    f"{graph_path.relative_to(GRAPH_ROOT)}: Hub adapter must use safetensors",
                )
        self.assertGreater(checked["modules.DiffusersAudio"], 0)
        self.assertEqual(checked["modules.DiffusersImage"], 11)

    def test_hub_modular_lora_graphs_pin_the_generic_auxiliary_identity(self):
        expected_revisions = {
            "lightx2v/Qwen-Image-Edit-2511-Lightning": "d74eba145674fd7e31b949324e148e21e7118abd",
        }
        manifest = json.loads(WORKFLOW_MANIFEST.read_text(encoding="utf-8"))
        manifest_workflows = {
            workflow["graphPath"]: workflow
            for workflow in [
                *manifest.get("workflows", []),
                *manifest.get("experimentalWorkflows", []),
            ]
        }
        checked = 0
        for graph_path in sorted(GRAPH_ROOT.rglob("*.json")):
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
            for node in graph.get("nodes", []):
                data = node.get("data", {})
                if data.get("module") != "modules.ModularDiffusers" or data.get("action") != "Lora":
                    continue
                params = data.get("params", {})
                selection = (params.get("model") or {}).get("value")
                if not isinstance(selection, dict) or selection.get("source") != "hub":
                    continue
                checked += 1
                label = str(graph_path.relative_to(GRAPH_ROOT))
                manifest_label = graph_path.relative_to(GRAPH_ROOT).as_posix()
                repository = str(selection.get("value") or "")
                revision = str((params.get("revision") or {}).get("value") or "")
                digest = str((params.get("expected_sha256") or {}).get("value") or "")
                weight_name = str((params.get("weight_name") or {}).get("value") or "")
                self.assertEqual(revision, expected_revisions[repository], label)
                self.assertRegex(digest, r"^[0-9a-f]{64}$", label)
                self.assertTrue(weight_name.endswith(".safetensors"), label)
                self.assertNotIn("filter", (params.get("model") or {}).get("fieldOptions", {}), label)
                self.assertIn(manifest_label, manifest_workflows, label)
                self.assertIn(
                    repository,
                    manifest_workflows[manifest_label].get("requiredArtifacts", []),
                    f"{label}: workflow manifest omits its mandatory Hub LoRA",
                )
        self.assertEqual(checked, 2)

    def test_hub_modular_component_graphs_pin_and_manifest_their_artifact(self):
        manifest = json.loads(WORKFLOW_MANIFEST.read_text(encoding="utf-8"))
        manifest_workflows = {
            workflow["graphPath"]: workflow
            for workflow in [
                *manifest.get("workflows", []),
                *manifest.get("experimentalWorkflows", []),
            ]
        }
        checked = 0
        for graph_path in sorted(GRAPH_ROOT.rglob("*.json")):
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
            for node in graph.get("nodes", []):
                data = node.get("data", {})
                if data.get("module") != "modules.ModularDiffusers" or data.get("action") != "AutoModelLoader":
                    continue
                params = data.get("params", {})
                selection = (params.get("model_id") or {}).get("value")
                if not isinstance(selection, dict) or selection.get("source") != "hub":
                    continue
                checked += 1
                label = str(graph_path.relative_to(GRAPH_ROOT))
                manifest_label = graph_path.relative_to(GRAPH_ROOT).as_posix()
                repository = str(selection.get("value") or "")
                revision = str((params.get("revision") or {}).get("value") or "")
                self.assertEqual(revision, catalog_revision(repository), label)
                self.assertIn(manifest_label, manifest_workflows, label)
                self.assertIn(
                    repository,
                    manifest_workflows[manifest_label].get("requiredArtifacts", []),
                    f"{label}: workflow manifest omits its mandatory Hub component",
                )
        self.assertEqual(checked, 1)

    def test_workflow_manifest_hashes_match_the_canonical_graphs(self):
        manifest = json.loads(WORKFLOW_MANIFEST.read_text(encoding="utf-8"))
        workflows = [
            *manifest.get("workflows", []),
            *manifest.get("experimentalWorkflows", []),
        ]
        for workflow in workflows:
            graph_path = GRAPH_ROOT / workflow["graphPath"]
            digest = _canonical_graph_digest(graph_path)
            self.assertEqual(digest, workflow["graphHash"], workflow["graphPath"])

    def test_curated_hub_loaders_store_the_exact_catalog_revision(self):
        checked = 0
        for graph_path in sorted(GRAPH_ROOT.rglob("*.json")):
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
            for node in graph.get("nodes", []):
                if not _is_curated_loader(node):
                    continue
                repo = _loader_repository(node)
                expected = catalog_revision(repo)
                if expected is None:
                    continue
                checked += 1
                params = node.get("data", {}).get("params", {})
                actual = str((params.get("revision") or {}).get("value") or "")
                self.assertEqual(actual, expected, f"{graph_path.relative_to(GRAPH_ROOT)}: {repo}")

                if node.get("data", {}).get("action") == "DynamicBlockNode":
                    self.assertIs(
                        (params.get("trust_remote_code") or {}).get("value"),
                        False,
                        f"{graph_path.relative_to(GRAPH_ROOT)} must not silently trust remote code",
                    )

                pin = catalog_repository_pin(repo) or {}
                if node.get("data", {}).get("action") == "LoadPipeline":
                    self.assertNotEqual(
                        pin.get("format"),
                        "gguf",
                        f"{graph_path.relative_to(GRAPH_ROOT)} passes a GGUF component repo to a full pipeline loader",
                    )
        self.assertGreater(checked, 0)

    def test_regenerated_graphs_keep_the_reviewed_effective_control_values(self):
        manifest = json.loads(WORKFLOW_MANIFEST.read_text(encoding="utf-8"))
        workflow_paths = {
            workflow["id"]: GRAPH_ROOT / workflow["graphPath"]
            for workflow in [
                *manifest.get("workflows", []),
                *manifest.get("experimentalWorkflows", []),
            ]
        }

        def graph_for(workflow_id):
            return json.loads(workflow_paths[workflow_id].read_text(encoding="utf-8"))

        def node_for_role(graph, role):
            matches = [node for node in graph.get("nodes", []) if node.get("data", {}).get("studioRole") == role]
            self.assertEqual(len(matches), 1, f"{role} must be unique")
            return matches[0]

        def params_for(graph, role):
            return node_for_role(graph, role).get("data", {}).get("params", {})

        # These families use guidance_scale. true_cfg_scale is a shared-node
        # field consumed only by the separately reviewed FramePack adapter.
        guidance_by_workflow = {
            "LTXVideoPipeline:image_to_video": 1,
            "LTXVideoPipeline:reference_to_video": 1,
            "LTXVideoPipeline:text_to_video": 1,
            "LTXVideoPipeline:video_to_video": 1,
            "WanImageToVideoPipeline:image_to_video": 3.5,
            "WanTI2VPipeline:text_to_video": 5,
            "WanVACEPipeline:control_to_video": 5,
            "WanVACEPipeline:text_to_video": 5,
            "WanVACEPipeline:video_inpaint": 5,
            "WanVACEPipeline:video_outpaint": 5,
            "WanVideoPipeline:text_to_video": 4.5,
            "WanVideoPipeline:video_color_edit": 5,
            "WanVideoPipeline:video_to_video": 5,
        }
        for workflow_id, expected_guidance in guidance_by_workflow.items():
            with self.subTest(workflow=workflow_id, control="true_cfg_scale"):
                generate = params_for(graph_for(workflow_id), "wanGenerate")
                self.assertNotIn("value", generate["true_cfg_scale"])
                self.assertEqual(generate["guidance_scale"].get("value"), expected_guidance)

        # LTX and Wan I2V do not accept a scheduler override. Wan TI2V does and
        # retains the official flow-shift value of eight.
        no_flow_shift = {
            "LTXVideoPipeline:image_to_video",
            "LTXVideoPipeline:reference_to_video",
            "LTXVideoPipeline:text_to_video",
            "LTXVideoPipeline:video_to_video",
            "WanImageToVideoPipeline:image_to_video",
        }
        for workflow_id in no_flow_shift:
            with self.subTest(workflow=workflow_id, control="scheduler_flow_shift"):
                self.assertNotIn("value", params_for(graph_for(workflow_id), "wanGenerate")["scheduler_flow_shift"])
        self.assertEqual(
            params_for(graph_for("WanTI2VPipeline:text_to_video"), "wanGenerate")["scheduler_flow_shift"].get("value"),
            8,
        )

        for workflow_id in ("AnimateDiffPipeline:text_to_video", "AnimateLCMPipeline:text_to_video"):
            with self.subTest(workflow=workflow_id, control="max_sequence_length"):
                self.assertEqual(
                    params_for(graph_for(workflow_id), "wanGenerate")["max_sequence_length"].get("value"),
                    77,
                )

        native_flash_migrations = {
            "WanVACEPipeline:control_to_video",
            "WanVACEPipeline:text_to_video",
            "WanVACEPipeline:video_inpaint",
            "WanVACEPipeline:video_outpaint",
            "WanVideoPipeline:video_color_edit",
            "WanVideoPipeline:video_to_video",
        }
        for workflow_id in native_flash_migrations:
            with self.subTest(workflow=workflow_id, control="attention_backend"):
                recipe = params_for(graph_for(workflow_id), "diffusersRecipe")
                self.assertEqual(recipe["device"].get("value"), "cuda:0")
                self.assertEqual(recipe["attention_backend"].get("value"), "_native_flash")
                self.assertEqual(recipe["attention_components"].get("value"), "transformer")

        model_cpu_migrations = {
            "AceStepAudioPipeline:audio_continuation",
            "AceStepAudioPipeline:audio_repaint",
            "AceStepAudioPipeline:audio_variation",
            "Flux2KleinPipeline:edit_image",
            "Flux2KleinPipeline:multi_image_reference_edit",
            "Flux2KleinPipeline:text_to_image",
            "FluxSchnellPipeline:text_to_image",
            "LTXVideoPipeline:image_to_video",
            "LTXVideoPipeline:reference_to_video",
            "LTXVideoPipeline:text_to_video",
            "LTXVideoPipeline:video_to_video",
            "QwenImageEditModularPipeline:edit_image",
            "QwenImageEditModularPipeline:inpaint",
            "QwenImageEditModularPipeline:outpaint",
            "QwenImageLayeredModularPipeline:layer_decomposition",
            "QwenImageModularPipeline:text_to_image",
            "StableDiffusionXLPAGPipeline:text_to_image",
            "WanVACEPipeline:control_to_video",
            "WanVACEPipeline:video_inpaint",
            "WanVACEPipeline:video_outpaint",
            "WanVideoPipeline:text_to_video",
            "WanVideoPipeline:video_color_edit",
            "WanVideoPipeline:video_to_video",
        }
        loader_roles = ("audioPipeline", "diffusersImagePipeline", "models", "wanPipeline")
        for workflow_id in model_cpu_migrations:
            with self.subTest(workflow=workflow_id, control="offload_mode"):
                graph = graph_for(workflow_id)
                loaders = [
                    node for node in graph.get("nodes", []) if node.get("data", {}).get("studioRole") in loader_roles
                ]
                self.assertEqual(len(loaders), 1)
                loader_params = loaders[0].get("data", {}).get("params", {})
                self.assertIs(loader_params["auto_offload"].get("value"), True)
                self.assertEqual(loader_params["offload_mode"].get("value"), "model_cpu")
                recipes = [
                    node
                    for node in graph.get("nodes", [])
                    if node.get("data", {}).get("studioRole") == "diffusersRecipe"
                ]
                if recipes:
                    self.assertEqual(
                        recipes[0].get("data", {}).get("params", {})["offload_mode"].get("value"),
                        "model_cpu",
                    )

        inactive_quantization_migrations = {
            "AceStepAudioPipeline:audio_continuation",
            "AceStepAudioPipeline:audio_repaint",
            "AceStepAudioPipeline:audio_variation",
            "Flux2KleinPipeline:edit_image",
            "Flux2KleinPipeline:multi_image_reference_edit",
            "Flux2KleinPipeline:text_to_image",
            "FluxSchnellPipeline:text_to_image",
            "QwenImageEditModularPipeline:inpaint",
            "QwenImageEditModularPipeline:outpaint",
            "QwenImageModularPipeline:text_to_image",
        }
        for workflow_id in inactive_quantization_migrations:
            with self.subTest(workflow=workflow_id, control="quantization_components"):
                quantization = params_for(graph_for(workflow_id), "diffusersQuantization")
                self.assertEqual(quantization["backend"].get("value"), "none")
                self.assertEqual(quantization["components"].get("value"), ["transformer"])

        layered = params_for(
            graph_for("QwenImageLayeredModularPipeline:layer_decomposition"),
            "loadImage",
        )
        self.assertEqual(layered["alpha_channel"].get("value"), "add alpha")

    def test_soundtrack_fallback_exception_requires_the_exact_controlled_route(self):
        graph_path = GRAPH_ROOT / "studio" / "wan-ti2-v-pipeline" / "text-to-video.json"
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        fallback_ids = _intentional_disabled_managed_fallback_node_ids(graph)
        self.assertEqual(fallback_ids, {"node-09"})

        def tampered(mutator):
            candidate = json.loads(json.dumps(graph))
            mutator(candidate)
            return _intentional_disabled_managed_fallback_node_ids(candidate)

        def node_for_role(candidate, role):
            return next(node for node in candidate["nodes"] if node.get("data", {}).get("studioRole") == role)

        self.assertEqual(
            tampered(lambda candidate: node_for_role(candidate, "videoExport")["data"].update(studioOwned=False)),
            set(),
        )
        self.assertEqual(
            tampered(
                lambda candidate: (
                    node_for_role(candidate, "soundtrackGenerate")["data"]
                    .setdefault("uiState", {})
                    .update(disabled=True)
                )
            ),
            set(),
        )
        self.assertEqual(
            tampered(
                lambda candidate: candidate["edges"].__setitem__(
                    slice(None),
                    [edge for edge in candidate["edges"] if edge.get("targetHandle") != "audio"],
                )
            ),
            set(),
        )
        self.assertEqual(
            tampered(
                lambda candidate: candidate["edges"].append(
                    {
                        "source": node_for_role(candidate, "wanGenerate")["id"],
                        "sourceHandle": "video_out",
                        "target": node_for_role(candidate, "videoExport")["id"],
                        "targetHandle": "video",
                    }
                )
            ),
            set(),
        )

    def test_every_catalog_node_contributes_to_a_visible_or_exported_output(self):
        graph_paths = sorted(GRAPH_ROOT.rglob("*.json"))
        self.assertGreater(len(graph_paths), 0)

        for graph_path in graph_paths:
            with self.subTest(graph=graph_path.relative_to(GRAPH_ROOT)):
                graph = json.loads(graph_path.read_text(encoding="utf-8"))
                nodes = {node["id"]: node for node in graph.get("nodes", [])}
                intentional_fallbacks = _intentional_disabled_managed_fallback_node_ids(graph)
                incoming = defaultdict(list)
                incident = set()
                for edge in graph.get("edges", []):
                    if edge.get("source") not in nodes or edge.get("target") not in nodes:
                        continue
                    incoming[edge["target"]].append(edge["source"])
                    incident.update((edge["source"], edge["target"]))

                outputs = [
                    node_id
                    for node_id, node in nodes.items()
                    if (
                        node.get("data", {}).get("module"),
                        node.get("data", {}).get("action"),
                    )
                    in OUTPUT_NODE_KEYS
                    and node_id not in intentional_fallbacks
                ]
                self.assertTrue(outputs, "graph has no preview, export, or data output")

                used = set(outputs)
                pending = list(outputs)
                while pending:
                    node_id = pending.pop()
                    for source_id in incoming[node_id]:
                        if source_id in used:
                            continue
                        used.add(source_id)
                        pending.append(source_id)

                disabled = [
                    node_id
                    for node_id, node in nodes.items()
                    if node.get("data", {}).get("uiState", {}).get("disabled") is True
                    and node_id not in intentional_fallbacks
                ]
                isolated = [
                    node_id for node_id in nodes if node_id not in intentional_fallbacks and node_id not in incident
                ]
                unreachable = [
                    node_id for node_id in nodes if node_id not in intentional_fallbacks and node_id not in used
                ]
                self.assertEqual(disabled, [], f"disabled execution nodes: {disabled}")
                self.assertEqual(isolated, [], f"isolated nodes: {isolated}")
                self.assertEqual(unreachable, [], f"nodes outside every output path: {unreachable}")


if __name__ == "__main__":
    unittest.main()
