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
    MAX_VIDEO_OPERATION_PIXELS,
    MAX_VIDEO_INTERPOLATION_FPS,
    MAX_VIDEO_INTERPOLATION_PIXEL_FRAMES,
    MAX_VIDEO_REVERSE_PIXEL_FRAMES,
    VIDEO_OPERATION_MODES,
    VIDEO_OPERATION_PIPELINE_CLASS,
    FrameExtract,
    FrameInterpolateAsset,
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

    def test_trim_reverse_and_tile_delegate_to_retained_video_nodes(self):
        first = _asset(frames=48, path="one.mp4")
        second = _asset(frames=36, path="two.mp4")
        with (
            patch("modules.Video.main._file_asset_collection", return_value=[first]),
            patch("modules.Video.main.TrimAsset.execute", return_value={"file": "trimmed.mp4"}) as trim,
        ):
            result = ProcessVideo().execute(
                videos="one.mp4",
                operation="video_trim",
                start_seconds=0.25,
                end_seconds=1.5,
            )
        self.assertEqual(result["video"], "trimmed.mp4")
        self.assertEqual(trim.call_args.kwargs["video"], first)
        self.assertEqual(trim.call_args.kwargs["start_seconds"], 0.25)
        self.assertEqual(trim.call_args.kwargs["end_seconds"], 1.5)

        with (
            patch("modules.Video.main._file_asset_collection", return_value=[first]),
            patch("modules.Video.main.ReverseAsset.execute", return_value={"file": "reversed.mp4"}) as reverse,
        ):
            result = ProcessVideo().execute(videos="one.mp4", operation="video_reverse")
        self.assertEqual(result["video"], "reversed.mp4")
        self.assertEqual(reverse.call_args.kwargs["video"], first)

        with (
            patch("modules.Video.main._file_asset_collection", return_value=[first, second]),
            patch("modules.Video.main.StackTileAssets.execute", return_value={"file": "tiled.mp4"}) as tile,
        ):
            result = ProcessVideo().execute(
                videos=["one.mp4", "two.mp4"],
                operation="video_tile",
                columns=2,
                sync="shortest",
                gap=8,
                background="gray",
            )
        self.assertEqual(result["video"], "tiled.mp4")
        self.assertEqual(tile.call_args.kwargs["videos"], [first, second])
        self.assertEqual(tile.call_args.kwargs["columns"], 2)
        self.assertEqual(tile.call_args.kwargs["sync"], "shortest")
        self.assertEqual(tile.call_args.kwargs["gap"], 8)
        self.assertEqual(tile.call_args.kwargs["background"], "gray")

    def test_frame_interpolation_delegates_with_one_bounded_source_and_exact_target_rate(self):
        source = _asset(frames=48, fps=24, path="source.mp4")
        with (
            patch("modules.Video.main._file_asset_collection", return_value=[source]),
            patch(
                "modules.Video.main.FrameInterpolateAsset.execute",
                return_value={"file": "interpolated.mp4"},
            ) as execute,
        ):
            result = ProcessVideo().execute(
                videos="source.mp4",
                operation="frame_interpolation",
                interpolation_fps=60,
            )
        self.assertEqual(result["video"], "interpolated.mp4")
        self.assertEqual(execute.call_args.kwargs["video"], source)
        self.assertEqual(execute.call_args.kwargs["target_fps"], 60)

    def test_frame_interpolation_uses_literal_bounded_ffmpeg_blend_filter(self):
        source = _asset(frames=48, fps=24, path="source.mp4")
        result = {"file": "interpolated.mp4", "asset": {"fps": 60}}
        with (
            patch("modules.Video.main._file_asset_collection", return_value=[source]),
            patch("modiff.media_assets.allocate_video_path", return_value=("asset", "output.mp4")),
            patch("modiff.media_assets.run_ffmpeg") as run_ffmpeg,
            patch("modules.Video.main._derived_asset_result", return_value=result),
        ):
            self.assertIs(FrameInterpolateAsset().execute(video=source, target_fps=60), result)
        command = run_ffmpeg.call_args.args[0]
        self.assertEqual(
            command[command.index("-vf") + 1],
            (
                "tpad=stop_mode=clone:stop_duration=1,"
                "minterpolate=fps=60:mi_mode=blend,"
                "trim=duration=2,setpts=PTS-STARTPTS"
            ),
        )
        self.assertEqual(command[command.index("-r") + 1], "60")
        self.assertIn("-an", command)

    def test_new_operations_fail_closed_on_workload_and_field_bounds(self):
        source = _asset(frames=48)
        trim_cases = (
            {"start_seconds": float("nan")},
            {"start_seconds": -1},
            {"start_seconds": 2},
            {"start_seconds": 1, "end_seconds": 0.5},
            {"end_seconds": 3},
        )
        for values in trim_cases:
            with (
                self.subTest(trim=values),
                patch("modules.Video.main._file_asset_collection", return_value=[source]),
                self.assertRaisesRegex(ValueError, "trim"),
            ):
                ProcessVideo().execute(videos="source.mp4", operation="video_trim", **values)
        for operation in ("frame_interpolation", "video_trim", "video_reverse"):
            with (
                self.subTest(operation=operation),
                patch("modules.Video.main._file_asset_collection", return_value=[source, source]),
                self.assertRaisesRegex(ValueError, "exactly one"),
            ):
                ProcessVideo().execute(videos=["one.mp4", "two.mp4"], operation=operation)
        reverse_width = 1024
        reverse_frames = MAX_VIDEO_REVERSE_PIXEL_FRAMES // (reverse_width * reverse_width) + 1
        with (
            patch(
                "modules.Video.main._file_asset_collection",
                return_value=[_asset(width=reverse_width, height=reverse_width, frames=reverse_frames)],
            ),
            self.assertRaisesRegex(ValueError, "pixel-frames"),
        ):
            ProcessVideo().execute(videos="source.mp4", operation="video_reverse")

        interpolation_cases = (24, 12, float("nan"), MAX_VIDEO_INTERPOLATION_FPS + 1)
        for target_fps in interpolation_cases:
            with (
                self.subTest(interpolation_fps=target_fps),
                patch("modules.Video.main._file_asset_collection", return_value=[source]),
                self.assertRaisesRegex(ValueError, "interpolation FPS"),
            ):
                ProcessVideo().execute(
                    videos="source.mp4",
                    operation="frame_interpolation",
                    interpolation_fps=target_fps,
                )
        oversized_interpolation = _asset(
            frames=MAX_VIDEO_OPERATION_FRAMES_PER_INPUT,
            fps=1,
        )
        with (
            patch("modules.Video.main._file_asset_collection", return_value=[oversized_interpolation]),
            self.assertRaisesRegex(ValueError, "frame output limit"),
        ):
            ProcessVideo().execute(
                videos="source.mp4",
                operation="frame_interpolation",
                interpolation_fps=MAX_VIDEO_INTERPOLATION_FPS,
            )
        pixel_frame_limited = _asset(width=4096, height=4096, frames=24, fps=24)
        self.assertLessEqual(4096 * 4096, MAX_VIDEO_OPERATION_PIXELS)
        with (
            patch("modules.Video.main._file_asset_collection", return_value=[pixel_frame_limited]),
            self.assertRaisesRegex(ValueError, str(MAX_VIDEO_INTERPOLATION_PIXEL_FRAMES)),
        ):
            ProcessVideo().execute(
                videos="source.mp4",
                operation="frame_interpolation",
                interpolation_fps=60,
            )

        tile_assets = [_asset(width=1920, height=1080), _asset(width=1920, height=1080)]
        invalid_tile_fields = (
            {"columns": 0},
            {"columns": 5},
            {"gap": -1},
            {"gap": 257},
            {"sync": "stretch"},
            {"background": "black;movie=payload"},
        )
        for values in invalid_tile_fields:
            with (
                self.subTest(tile=values),
                patch("modules.Video.main._file_asset_collection", return_value=tile_assets),
                self.assertRaisesRegex(ValueError, "tile"),
            ):
                ProcessVideo().execute(videos=["one.mp4", "two.mp4"], operation="video_tile", **values)
        oversized = [_asset(width=4096, height=2048), _asset(width=4096, height=2048)]
        self.assertLessEqual(4096 * 2048, MAX_VIDEO_OPERATION_PIXELS)
        with (
            patch("modules.Video.main._file_asset_collection", return_value=oversized),
            self.assertRaisesRegex(ValueError, "tile output"),
        ):
            ProcessVideo().execute(videos=["one.mp4", "two.mp4"], operation="video_tile", columns=2, gap=1)

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
        self.assertEqual(
            capability["modeOutputKinds"],
            {
                "video_frame_extract": "image",
                "frame_interpolation": "video",
                "video_stitch": "video",
                "video_trim": "video",
                "video_reverse": "video",
                "video_tile": "video",
            },
        )
        self.assertFalse(capability["autoEligible"])
        self.assertTrue(capability["templateEligible"])
        self.assertFalse(capability["galleryEligible"])

        contracts = contracts_by_pair(payload["taskTemplateContracts"])
        extract = contracts[("BuiltinVideoOperation", "video_frame_extract")]
        interpolate = contracts[("BuiltinVideoOperation", "frame_interpolation")]
        stitch = contracts[("BuiltinVideoOperation", "video_stitch")]
        trim = contracts[("BuiltinVideoOperation", "video_trim")]
        reverse = contracts[("BuiltinVideoOperation", "video_reverse")]
        tile = contracts[("BuiltinVideoOperation", "video_tile")]
        self.assertEqual(extract["requiredMedia"], [{"kind": "video", "field": "sourceVideo", "minimumCount": 1}])
        self.assertEqual(extract["mediaKind"], "image")
        self.assertEqual(extract["output"]["nodeKey"], "modules.Image.Preview")
        self.assertEqual(stitch["requiredMedia"], [{"kind": "video", "field": "referenceVideos", "minimumCount": 2}])
        self.assertEqual(stitch["mediaKind"], "video")
        self.assertEqual(stitch["output"]["nodeKey"], "modules.Video.Export")
        for contract in (interpolate, trim, reverse):
            self.assertEqual(contract["requiredMedia"], [{"kind": "video", "field": "sourceVideo", "minimumCount": 1}])
            self.assertEqual(contract["mediaKind"], "video")
            self.assertEqual(contract["output"]["nodeKey"], "modules.Video.Export")
        self.assertEqual(tile["requiredMedia"], [{"kind": "video", "field": "referenceVideos", "minimumCount": 2}])
        self.assertEqual(tile["mediaKind"], "video")
        self.assertEqual(tile["output"]["nodeKey"], "modules.Video.Export")


if __name__ == "__main__":
    unittest.main()
