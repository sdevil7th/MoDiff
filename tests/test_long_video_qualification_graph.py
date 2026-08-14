import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from modiff.model_artifact_catalog import catalog_revision
from modiff.server import WebServer
from modules import MODULE_MAP
from modules.DiffusersVideo.main import PlanLongVideo
from scripts.materialize_long_video_qualification import (
    MODEL_REPO,
    MODEL_REVISION,
    TEMPLATE_PATH,
    _loopback_base_url,
    assert_model_ready,
    materialize_graph,
    read_template,
)


class LongVideoQualificationGraphTests(unittest.TestCase):
    def setUp(self):
        self.graph = read_template()

    def test_reviewed_template_is_a_complete_durable_api_graph(self):
        self.assertTrue(TEMPLATE_PATH.is_file())
        self.assertEqual(catalog_revision(MODEL_REPO), MODEL_REVISION)
        self.assertEqual(
            self.graph["runtimeHints"]["resolvedArtifact"],
            f"{MODEL_REPO}@{MODEL_REVISION}",
        )
        self.assertEqual(self.graph["runtimeHints"]["maxRuntimeSeconds"], 21600)
        self.assertEqual(
            self.graph["paths"],
            [[
                "quantization",
                "recipe",
                "opening-image",
                "pipeline",
                "plan",
                "carry",
                "items",
                "generate",
                "retain",
                "result",
                "join",
            ]],
        )
        loop = self.graph["loops"][0]
        self.assertEqual(loop["iterationMode"], "collection")
        self.assertEqual(loop["iterations"], 374)
        self.assertEqual(loop["maxIterations"], 512)
        self.assertTrue(loop["carry"])
        self.assertTrue(loop["collect"])
        self.assertTrue(loop["durable"])
        self.assertEqual(
            loop["bodyNodeIds"],
            ["carry", "items", "generate", "retain", "result"],
        )

        server = WebServer.__new__(WebServer)
        prepared = server._prepare_graph_loops(self.graph)
        self.assertEqual(prepared["loops"][0]["body"], loop["bodyNodeIds"])
        self.assertTrue(prepared["loops"][0]["durable"])

    def test_every_connection_targets_registered_nodes_and_declared_outputs(self):
        nodes = self.graph["nodes"]
        for node_id, node in nodes.items():
            self.assertIn(node["module"], MODULE_MAP, node_id)
            self.assertIn(node["action"], MODULE_MAP[node["module"]], node_id)
            target_params = MODULE_MAP[node["module"]][node["action"]]["params"]
            for param_name, param in node["params"].items():
                self.assertIn(param_name, target_params, f"{node_id}.{param_name}")
                source_id = param.get("sourceId") if isinstance(param, dict) else None
                if not source_id:
                    continue
                self.assertIn(source_id, nodes)
                source = nodes[source_id]
                source_params = MODULE_MAP[source["module"]][source["action"]]["params"]
                source_key = param["sourceKey"]
                self.assertIn(source_key, source_params, f"{source_id}.{source_key}")
                self.assertEqual(source_params[source_key].get("display"), "output")

    def test_template_recipe_plans_the_declared_30_minute_workload(self):
        params = {
            key: value.get("value")
            for key, value in self.graph["nodes"]["plan"]["params"].items()
            if key != "opening_image"
        }
        result = PlanLongVideo().execute(
            **params,
            opening_image=Image.new("RGB", (8, 6), "black"),
        )
        expected = self.graph["qualification"]["expected"]
        self.assertEqual(result["job_count"], expected["jobCount"])
        self.assertEqual(result["planned_frames"], expected["plannedFrames"])
        self.assertEqual(result["planned_seconds"], expected["plannedSeconds"])
        self.assertEqual(result["overlap_frames"], 4)
        self.assertTrue(all(job["uses_previous_last_frame"] for job in result["jobs"][1:]))

    def test_materializer_binds_image_bytes_and_exact_recovery_identity(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            opening = root / "opening.webp"
            opening.write_bytes(b"first-opening-image")
            with patch(
                "scripts.materialize_long_video_qualification.resolve_runtime_input_path",
                return_value=opening,
            ):
                first, first_receipt = materialize_graph(
                    opening,
                    "images/qualification/opening.webp",
                    sid="remote-session",
                )
                opening.write_bytes(b"second-opening-image")
                second, second_receipt = materialize_graph(
                    opening,
                    "images/qualification/opening.webp",
                    sid="remote-session",
                )

        self.assertNotIn("__OPENING_IMAGE__", json.dumps(first))
        self.assertNotIn("__RUN_INPUT_HASH__", json.dumps(first))
        self.assertNotIn("__SID__", json.dumps(first))
        self.assertEqual(first["nodes"]["opening-image"]["params"]["file"]["value"], [
            "images/qualification/opening.webp"
        ])
        self.assertEqual(first["runtimeHints"]["runInputHash"], first_receipt["runInputHash"])
        self.assertRegex(first_receipt["runInputHash"], r"^[0-9a-f]{64}$")
        self.assertNotEqual(first_receipt["runInputHash"], second_receipt["runInputHash"])
        self.assertNotEqual(first_receipt["openingImageSha256"], second_receipt["openingImageSha256"])
        self.assertEqual(second["sid"], "remote-session")

    def test_submit_helpers_require_loopback_and_the_exact_complete_cache_revision(self):
        self.assertEqual(_loopback_base_url("http://127.0.0.1:8088/"), "http://127.0.0.1:8088")
        self.assertEqual(_loopback_base_url("http://localhost:8088"), "http://localhost:8088")
        for value in ("https://example.com", "http://127.0.0.1:8088/api", "file:///tmp/app"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _loopback_base_url(value)

        ready = {
            "id": MODEL_REPO,
            "complete": True,
            "repair_required": False,
            "size": 123,
            "revisions": [{"hash": MODEL_REVISION}],
        }
        with patch("scripts.materialize_long_video_qualification._request_json", return_value=[ready]):
            self.assertEqual(assert_model_ready("http://127.0.0.1:8088")["revision"], MODEL_REVISION)
        with patch(
            "scripts.materialize_long_video_qualification._request_json",
            return_value=[{**ready, "revisions": [{"hash": "0" * 40}]}],
        ):
            with self.assertRaisesRegex(RuntimeError, "not complete in the app cache"):
                assert_model_ready("http://127.0.0.1:8088")


if __name__ == "__main__":
    unittest.main()
