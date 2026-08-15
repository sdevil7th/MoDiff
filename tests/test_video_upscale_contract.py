import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

import modules as module_registry
from modiff.controlled_artifacts import controlled_artifact_receipts_from_graph
from modiff.diffusers_profiles import execution_profiles_for_execution
from modiff.server import WebServer
from modiff.studio_execution_specs import studio_execution_spec_for_pair, validate_studio_execution_specs
from modiff.task_template_contracts import contracts_by_pair
from modules.Video.main import (
    MAX_VIDEO_UPSCALE_FRAMES,
    VIDEO_UPSCALE_MODE,
    VIDEO_UPSCALE_MODEL_SELECTION,
    VIDEO_UPSCALE_PIPELINE_CLASS,
    UpscaleVideo,
)


class VideoUpscaleContractTests(unittest.TestCase):
    def test_node_publishes_the_exact_reviewed_generic_contract(self):
        self.assertEqual(UpscaleVideo.params["operation"]["options"], [VIDEO_UPSCALE_MODE])
        self.assertEqual(UpscaleVideo.params["pipeline_class"]["default"], VIDEO_UPSCALE_PIPELINE_CLASS)
        self.assertEqual(UpscaleVideo.params["model_id"]["default"], VIDEO_UPSCALE_MODEL_SELECTION)
        self.assertEqual(VIDEO_UPSCALE_MODEL_SELECTION["revision"], "42efb9c3eeed1f5c0c8a626cf5f7f4481dfbb094")
        self.assertEqual(VIDEO_UPSCALE_MODEL_SELECTION["byteSize"], 67_061_725)

    def test_contract_identity_and_source_bounds_fail_before_model_load(self):
        node = UpscaleVideo("bounded-upscale")
        with patch("modules.Video.main._bounded_video_operation_assets") as assets:
            with self.assertRaisesRegex(ValueError, "pipeline contract"):
                node.execute(video="source.mp4", pipeline_class="OtherUpscaleV1")
            assets.assert_not_called()

        oversized = {
            "path": "/managed/source.mp4",
            "width": 640,
            "height": 360,
            "fps": 24,
            "frame_count": MAX_VIDEO_UPSCALE_FRAMES + 1,
        }
        with patch("modules.Video.main._bounded_video_operation_assets", return_value=[oversized]):
            with patch("modules.Spandrel.main.Upscaler") as upscaler:
                with self.assertRaisesRegex(ValueError, "at most"):
                    node.execute(video="source.mp4")
                upscaler.assert_not_called()

    def test_streaming_execution_keeps_one_frame_in_the_model_boundary(self):
        source = {
            "path": "/managed/source.mp4",
            "width": 2,
            "height": 2,
            "fps": 24,
            "frame_count": 2,
        }
        raw_frames = [np.zeros((2, 2, 3), dtype=np.uint8), np.full((2, 2, 3), 64, dtype=np.uint8)]
        reader = MagicMock()
        reader.__iter__.return_value = iter(raw_frames)
        writer = MagicMock()
        model_node = MagicMock()
        model_node.execute.side_effect = [
            {"output": [Image.new("RGB", (4, 4), color="black")]},
            {"output": [Image.new("RGB", (4, 4), color="gray")]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "upscaled.mp4"
            node = UpscaleVideo("streaming-upscale")
            node.progress = MagicMock()
            with (
                patch("modules.Video.main._bounded_video_operation_assets", return_value=[source]),
                patch("modiff.media_assets.current_task_id", return_value="task-upscale"),
                patch("modiff.media_assets.allocate_video_path", return_value=("asset-upscale", destination)),
                patch("imageio.get_reader", return_value=reader),
                patch("imageio.get_writer", return_value=writer) as get_writer,
                patch("modules.Spandrel.main.Upscaler", return_value=model_node),
                patch(
                    "modules.Video.main._derived_asset_result",
                    return_value={"file": str(destination), "frames": 2, "duration_seconds": 2 / 24},
                ) as derived,
            ):
                result = node.execute(video="source.mp4", device="cpu", fps=24, tile_size=256, tile_overlap=32)

        self.assertEqual(result["video_out"], str(destination))
        self.assertEqual((result["width"], result["height"], result["frames"], result["fps"]), (4, 4, 2, 24.0))
        self.assertEqual(model_node.execute.call_count, 2)
        for call in model_node.execute.call_args_list:
            self.assertIsInstance(call.kwargs["image"], Image.Image)
            self.assertEqual(call.kwargs["model_id"], VIDEO_UPSCALE_MODEL_SELECTION)
        get_writer.assert_called_once_with(destination, fps=24.0, quality=8, codec="libx264")
        self.assertEqual(writer.append_data.call_count, 2)
        reader.close.assert_called_once_with()
        writer.close.assert_called_once_with()
        derived.assert_called_once_with(
            destination,
            "asset-upscale",
            [source],
            "spandrel-video-upscale",
        )

    def test_combined_node_is_part_of_controlled_upscaler_receipts(self):
        selection = dict(VIDEO_UPSCALE_MODEL_SELECTION)
        graph = {
            "nodes": {
                "upscale": {
                    "module": "modules.Video",
                    "action": "UpscaleVideo",
                    "params": {"model_id": {"value": selection}},
                }
            },
            "paths": [["upscale"]],
        }
        receipt = {"kind": "spandrel_upscaler", "module": "modules.Video", "action": "UpscaleVideo"}
        with patch(
            "modiff.controlled_artifacts.resolve_upscaler_artifact",
            return_value=SimpleNamespace(receipt=receipt),
        ) as resolve:
            self.assertEqual(controlled_artifact_receipts_from_graph(graph), [receipt])
        resolve.assert_called_once_with(
            selection,
            contract=("modules.Video", "UpscaleVideo"),
        )


class FakeRequest:
    query = {}


class VideoUpscaleStudioContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_profile_capability_and_task_contract_are_exact_and_non_gallery(self):
        specs = validate_studio_execution_specs(module_registry.MODULE_MAP)
        specification = studio_execution_spec_for_pair("SpandrelVideoUpscale", VIDEO_UPSCALE_MODE)
        self.assertIn(specification, specs)
        self.assertEqual(specification["loaderModule"], "modules.Video")
        self.assertEqual(specification["loaderAction"], "UpscaleVideo")
        self.assertEqual(specification["pipelineClass"], VIDEO_UPSCALE_PIPELINE_CLASS)
        self.assertEqual(specification["contentHash"], "studio-spec-v1-231f621f")

        profiles = execution_profiles_for_execution("SpandrelVideoUpscale", VIDEO_UPSCALE_MODE)
        self.assertEqual(len(profiles), 1)
        self.assertEqual(profiles[0].id, "real-esrgan-x2-video-upscale:direct")
        self.assertEqual(profiles[0].optional_runtime_profiles, ())

        response = await WebServer(module_registry.MODULE_MAP).model_capabilities(FakeRequest())
        self.assertEqual(response.status, 200)
        payload = json.loads(response.text)
        capability = next(item for item in payload["capabilities"] if item["modelType"] == "SpandrelVideoUpscale")
        self.assertEqual(capability["runnableModes"], [VIDEO_UPSCALE_MODE])
        self.assertEqual(capability["executionStatus"], "expert_only")
        self.assertEqual(capability["defaultRepo"], "nateraw/real-esrgan")
        self.assertEqual(capability["downloadFiles"], ["RealESRGAN_x2plus.pth"])
        self.assertEqual(capability["revisionCandidates"], [VIDEO_UPSCALE_MODEL_SELECTION["revision"]])
        self.assertFalse(capability["autoEligible"])
        self.assertTrue(capability["templateEligible"])
        self.assertFalse(capability["galleryEligible"])

        contract = contracts_by_pair(payload["taskTemplateContracts"])[("SpandrelVideoUpscale", VIDEO_UPSCALE_MODE)]
        self.assertEqual(
            contract["requiredMedia"],
            [{"kind": "video", "field": "sourceVideo", "minimumCount": 1}],
        )
        self.assertEqual(contract["output"]["nodeKey"], "modules.Video.Export")
        self.assertEqual(contract["loaderRole"], "videoUpscaler")


if __name__ == "__main__":
    unittest.main()
