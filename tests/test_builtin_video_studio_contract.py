import json
import unittest
from unittest.mock import patch

from PIL import Image

import modules as module_registry
from modiff.diffusers_profiles import (
    execution_profiles_for_execution,
    optional_runtime_profile_ids_for_execution,
)
from modiff.server import WebServer
from modiff.studio_execution_specs import (
    studio_execution_spec_for_pair,
    validate_studio_execution_specs,
)
from modiff.task_template_contracts import contracts_by_pair
from modules.Video.main import (
    MAX_EXTRACTED_FRAMES,
    MAX_VIDEO_OPERATION_FRAMES_PER_INPUT,
    MAX_VIDEO_OPERATION_INPUTS,
    VIDEO_OPERATION_MODES,
    VIDEO_OPERATION_PIPELINE_CLASS,
    FrameExtract,
    ProcessVideo,
)


class FakeRequest:
    query = {}


def _asset(*, frames=24, width=16, height=16, fps=24.0, path="source.mp4"):
    return {
        "path": path,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frames,
        "duration_seconds": frames / fps if fps else 0,
    }


class BuiltinVideoOperationTests(unittest.TestCase):
    def test_frame_extract_delegates_with_one_bounded_file_asset(self):
        expected = {
            "frames": [Image.new("RGB", (2, 2), "red")],
            "selected_indices": [0],
            "timestamps": [0.0],
        }
        with (
            patch("modules.Video.main._file_asset_collection", return_value=[_asset()]),
            patch("modules.Video.main.FrameExtract.execute", return_value=expected) as execute,
        ):
            result = ProcessVideo().execute(
                videos="source.mp4",
                pipeline_class=VIDEO_OPERATION_PIPELINE_CLASS,
                operation="video_frame_extract",
                selection_mode="first",
            )
        self.assertEqual(result["images"], expected["frames"])
        self.assertIsNone(result["video"])
        self.assertEqual(result["selected_indices"], [0])
        self.assertEqual(execute.call_args.kwargs["video"]["path"], "source.mp4")
        self.assertEqual(execute.call_args.kwargs["mode"], "first")

    def test_stitch_delegates_to_file_backed_concatenation(self):
        assets = [_asset(path="one.mp4"), _asset(path="two.mp4")]
        with (
            patch("modules.Video.main._file_asset_collection", return_value=assets),
            patch(
                "modules.Video.main.ConcatenateAssets.execute",
                return_value={"file": "joined.mp4"},
            ) as execute,
        ):
            result = ProcessVideo().execute(
                videos=["one.mp4", "two.mp4"],
                operation="video_stitch",
                transition_seconds=0.5,
            )
        self.assertEqual(result["video"], "joined.mp4")
        self.assertIsNone(result["images"])
        self.assertEqual(execute.call_args.kwargs["clips"], assets)
        self.assertEqual(execute.call_args.kwargs["transition_seconds"], 0.5)

    def test_facade_fails_closed_on_identity_operation_count_and_media_bounds(self):
        with self.assertRaisesRegex(ValueError, "contract identity"):
            ProcessVideo().execute(videos="source.mp4", pipeline_class="plugin", operation="video_frame_extract")
        with self.assertRaisesRegex(ValueError, "Unsupported built-in video operation"):
            ProcessVideo().execute(videos="source.mp4", operation="video_interpolate")
        with (
            patch("modules.Video.main._file_asset_collection", return_value=[_asset(), _asset()]),
            self.assertRaisesRegex(ValueError, "exactly one"),
        ):
            ProcessVideo().execute(videos=["one.mp4", "two.mp4"], operation="video_frame_extract")
        with (
            patch("modules.Video.main._file_asset_collection", return_value=[_asset()]),
            self.assertRaisesRegex(ValueError, "between 2"),
        ):
            ProcessVideo().execute(videos="one.mp4", operation="video_stitch")
        invalid_assets = (
            [_asset(frames=MAX_VIDEO_OPERATION_FRAMES_PER_INPUT + 1)],
            [_asset(width=8192, height=8192)],
            [_asset(fps=0)],
            [_asset() for _ in range(MAX_VIDEO_OPERATION_INPUTS + 1)],
        )
        for assets in invalid_assets:
            with (
                self.subTest(assets=len(assets)),
                patch("modules.Video.main._file_asset_collection", return_value=assets),
                self.assertRaises(ValueError),
            ):
                ProcessVideo().execute(videos="source.mp4", operation="video_frame_extract")
        with (
            patch("modules.Video.main._file_asset_collection", return_value=[_asset()]),
            self.assertRaisesRegex(ValueError, "finite values"),
        ):
            ProcessVideo().execute(
                videos="source.mp4",
                operation="video_frame_extract",
                selection_mode="timecodes",
                timecodes="nan",
            )
        with (
            patch("modules.Video.main._file_asset_collection", return_value=[_asset(), _asset()]),
            self.assertRaisesRegex(ValueError, "crossfade"),
        ):
            ProcessVideo().execute(
                videos=["one.mp4", "two.mp4"],
                operation="video_stitch",
                transition_seconds=float("inf"),
            )

    def test_underlying_frame_extractor_caps_outputs_before_materializing_them(self):
        frames = [Image.new("RGB", (1, 1), "black") for _ in range(MAX_EXTRACTED_FRAMES + 1)]
        with self.assertRaisesRegex(ValueError, f"at most {MAX_EXTRACTED_FRAMES}"):
            FrameExtract().execute(video=frames, mode="every_n", every_n=1, fps=24)


class BuiltinVideoStudioContractTests(unittest.IsolatedAsyncioTestCase):
    def test_specs_use_one_model_neutral_base_runtime_profile(self):
        specs = validate_studio_execution_specs(module_registry.MODULE_MAP)
        pairs = {(item["modelType"], item["mode"]): item for item in specs}
        for mode in VIDEO_OPERATION_MODES:
            with self.subTest(mode=mode):
                specification = pairs[("BuiltinVideoOperation", mode)]
                self.assertEqual(specification, studio_execution_spec_for_pair("BuiltinVideoOperation", mode))
                self.assertEqual(specification["loaderModule"], "modules.Video")
                self.assertEqual(specification["loaderAction"], "ProcessVideo")
                self.assertEqual(specification["executionPath"], "builtin-video-operation")
                self.assertEqual(specification["pipelineClass"], VIDEO_OPERATION_PIPELINE_CLASS)
                profiles = execution_profiles_for_execution("BuiltinVideoOperation", mode)
                self.assertEqual(len(profiles), 1)
                self.assertEqual(profiles[0].id, "builtin-video-operations:direct")
                self.assertEqual(profiles[0].optional_runtime_profiles, ())
                self.assertEqual(optional_runtime_profile_ids_for_execution("BuiltinVideoOperation", mode), ())

    async def test_capability_and_task_contracts_are_install_free_and_media_exact(self):
        response = await WebServer(module_registry.MODULE_MAP).model_capabilities(FakeRequest())
        self.assertEqual(response.status, 200)
        payload = json.loads(response.text)
        capability = next(item for item in payload["capabilities"] if item["modelType"] == "BuiltinVideoOperation")
        self.assertEqual(capability["modes"], list(VIDEO_OPERATION_MODES))
        self.assertEqual(capability["executionStatus"], "supported")
        self.assertEqual(capability["artifactKind"], "builtin")
        self.assertFalse(capability["artifactInstallRequired"])
        self.assertEqual(capability["modeOutputKinds"], {"video_frame_extract": "image", "video_stitch": "video"})
        self.assertFalse(capability["autoEligible"])
        self.assertTrue(capability["templateEligible"])
        self.assertFalse(capability["galleryEligible"])

        contracts = contracts_by_pair(payload["taskTemplateContracts"])
        extract = contracts[("BuiltinVideoOperation", "video_frame_extract")]
        stitch = contracts[("BuiltinVideoOperation", "video_stitch")]
        self.assertEqual(extract["requiredMedia"], [{"kind": "video", "field": "sourceVideo", "minimumCount": 1}])
        self.assertEqual(extract["mediaKind"], "image")
        self.assertEqual(extract["output"]["nodeKey"], "modules.Image.Preview")
        self.assertEqual(stitch["requiredMedia"], [{"kind": "video", "field": "referenceVideos", "minimumCount": 2}])
        self.assertEqual(stitch["mediaKind"], "video")
        self.assertEqual(stitch["output"]["nodeKey"], "modules.Video.Export")


if __name__ == "__main__":
    unittest.main()
