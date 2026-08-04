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
    return (
        data.get("module") in DIFFUSERS_PIPELINE_MODULES
        and data.get("action") == "LoadPipeline"
    ) or (
        data.get("module") == "modules.ModularDiffusers"
        and data.get("action") in {"ModelsLoader", "DynamicBlockNode"}
    )


class GraphCatalogIntegrityTests(unittest.TestCase):
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

    def test_every_catalog_node_contributes_to_a_visible_or_exported_output(self):
        graph_paths = sorted(GRAPH_ROOT.rglob("*.json"))
        self.assertGreater(len(graph_paths), 0)

        for graph_path in graph_paths:
            with self.subTest(graph=graph_path.relative_to(GRAPH_ROOT)):
                graph = json.loads(graph_path.read_text(encoding="utf-8"))
                nodes = {node["id"]: node for node in graph.get("nodes", [])}
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
                ]
                self.assertTrue(outputs, "graph has no preview, export, or data-viewer output")

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
                ]
                isolated = [node_id for node_id in nodes if node_id not in incident]
                unreachable = [node_id for node_id in nodes if node_id not in used]
                self.assertEqual(disabled, [], f"disabled execution nodes: {disabled}")
                self.assertEqual(isolated, [], f"isolated nodes: {isolated}")
                self.assertEqual(unreachable, [], f"nodes outside every output path: {unreachable}")


if __name__ == "__main__":
    unittest.main()
