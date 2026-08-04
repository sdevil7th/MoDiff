import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from modules.Video.main import (
    Concatenate,
    Crossfade,
    ConcatenateAssets,
    FirstLastSegmentBuilder,
    FrameExtract,
    KeyframeChain,
    MaskedComposite,
    MuxAudioAsset,
    Reverse,
    StackTile,
    TemporalCleanPlate,
    ExtendCleanPlate,
    Trim,
    TrimAsset,
)


def solid(color, count=1, size=(8, 6)):
    return [Image.new("RGB", size, color) for _ in range(count)]


class VideoOperationTests(unittest.TestCase):
    def test_file_native_trim_and_join_delegate_to_ffmpeg_without_decoding_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.mp4"
            second_path = Path(directory) / "second.mp4"
            first_path.write_bytes(b"first")
            second_path.write_bytes(b"second")
            first = {
                "asset_id": "first",
                "path": str(first_path),
                "width": 16,
                "height": 12,
                "fps": 8,
                "frame_count": 16,
                "duration_seconds": 2,
            }
            second = {**first, "asset_id": "second", "path": str(second_path)}
            third_path = Path(directory) / "third.mp4"
            third_path.write_bytes(b"third")
            third = {**first, "asset_id": "third", "path": str(third_path)}
            commands = []

            def capture(arguments, destination):
                commands.append(arguments)
                return destination

            result = {"asset": {}, "file": "result.mp4", "duration_seconds": 3.5, "frames": 28}
            with (
                patch("modiff.media_assets.run_ffmpeg", side_effect=capture),
                patch("modules.Video.main._derived_asset_result", return_value=result),
            ):
                self.assertIs(TrimAsset().execute(video=first, start_seconds=0.5, end_seconds=1.5), result)
                self.assertIs(
                    ConcatenateAssets().execute(clips=[first, second, third], transition_seconds=0.5),
                    result,
                )

            self.assertIn("-ss", commands[0])
            self.assertIn("-filter_complex", commands[1])
            filter_graph = commands[1][commands[1].index("-filter_complex") + 1]
            self.assertEqual(filter_graph.count("xfade=transition=fade"), 2)
            self.assertIn("settb=expr=1/8", filter_graph)
            self.assertIn("[raw_x1]settb=expr=1/8,setpts=PTS-STARTPTS,fps=8[x1]", filter_graph)
            self.assertEqual(commands[1][commands[1].index("-r") + 1], "8.0")

    def test_file_native_audio_mux_preserves_video_stream_and_matches_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            video_path = Path(directory) / "video.mp4"
            audio_path = Path(directory) / "audio.wav"
            video_path.write_bytes(b"video")
            audio_path.write_bytes(b"audio")
            source = {
                "asset_id": "video",
                "path": str(video_path),
                "width": 16,
                "height": 12,
                "fps": 8,
                "frame_count": 16,
                "duration_seconds": 2,
            }
            commands = []
            result = {"asset": {}, "file": "muxed.mp4", "duration_seconds": 2, "frames": 16}
            with (
                patch("modiff.media_assets.run_ffmpeg", side_effect=lambda args, output: commands.append(args)),
                patch("modules.Video.main._derived_asset_result", return_value=result),
            ):
                self.assertIs(MuxAudioAsset().execute(video=source, audio=str(audio_path), fit="match_video"), result)

            self.assertIn("copy", commands[0])
            self.assertIn("apad", commands[0])
            self.assertIn("2", commands[0])

    def test_frame_extract_supports_negative_indices_and_timecodes(self):
        clip = solid("red", 5)
        indexed = FrameExtract().execute(video=clip, mode="indices", indices="0, 2, -1", fps=2)
        timed = FrameExtract().execute(video=clip, mode="timecodes", timecodes="0.5, 1.5", fps=2)

        self.assertEqual(indexed["selected_indices"], [0, 2, 4])
        self.assertEqual(indexed["timestamps"], [0, 1, 2])
        self.assertEqual(timed["selected_indices"], [1, 3])

    def test_frame_extract_reads_only_requested_boundaries_from_a_file_asset(self):
        import imageio.v2 as imageio
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "boundaries.mp4"
            writer = imageio.get_writer(path, fps=2, codec="libx264", macro_block_size=1)
            try:
                for value in (0, 80, 160):
                    writer.append_data(np.full((16, 16, 3), value, dtype=np.uint8))
            finally:
                writer.close()

            result = FrameExtract().execute(video=str(path), mode="first_last")

        self.assertEqual(result["selected_indices"], [0, 2])
        self.assertEqual(len(result["frames"]), 2)

    def test_trim_concatenate_reverse_and_crossfade_have_exact_frame_counts(self):
        red = solid("red", 4)
        blue = solid("blue", 4)
        trimmed = Trim().execute(video=red, range_mode="frames", start=1, end=3, fps=2)["output"]
        joined = Concatenate().execute(clips=[trimmed, blue], transition_seconds=0.5, fps=2)
        faded = Crossfade().execute(first=red, second=blue, duration_seconds=1, fps=2)

        self.assertEqual(len(trimmed), 2)
        self.assertEqual(joined["frames"], 5)
        self.assertEqual(faded["frames"], 6)
        self.assertEqual(Reverse().execute(video=trimmed)["output"][0].getpixel((0, 0)), (255, 0, 0))

    def test_video_tile_holds_short_clips_and_builds_requested_grid(self):
        result = StackTile().execute(
            videos=[solid("red", 1), solid("blue", 3)],
            columns=2,
            sync="longest_hold",
            gap=2,
            background="black",
        )

        self.assertEqual(result["frames"], 3)
        self.assertEqual(result["output"][0].size, (18, 6))
        self.assertEqual(result["output"][-1].getpixel((10, 0)), (0, 0, 255))

    def test_masked_composite_preserves_source_outside_white_region(self):
        source = solid("red", 2, size=(4, 2))
        generated = solid("blue", 2, size=(4, 2))
        mask = Image.new("L", (4, 2), 0)
        for x in (2, 3):
            for y in (0, 1):
                mask.putpixel((x, y), 255)

        result = MaskedComposite().execute(source=source, generated=generated, mask=mask)

        self.assertEqual(result["frames"], 2)
        self.assertEqual(result["output"][0].getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(result["output"][0].getpixel((3, 1)), (0, 0, 255))

    def test_masked_composite_rejects_frame_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "frame mismatch"):
            MaskedComposite().execute(
                source=solid("red", 2),
                generated=solid("blue", 1),
                mask=Image.new("L", (8, 6), 255),
            )

    def test_temporal_clean_plate_preserves_endpoints_and_frame_count(self):
        result = TemporalCleanPlate().execute(
            video=[
                Image.new("RGB", (2, 2), (0, 0, 0)),
                Image.new("RGB", (2, 2), (255, 0, 0)),
                Image.new("RGB", (2, 2), (255, 255, 255)),
            ],
            start_index=0,
            end_index=-1,
            easing="linear",
        )

        self.assertEqual(result["frames"], 3)
        self.assertEqual(result["output"][0].getpixel((0, 0)), (0, 0, 0))
        # PIL blends 8-bit channel values with floor rounding.
        self.assertEqual(result["output"][1].getpixel((0, 0)), (127, 127, 127))
        self.assertEqual(result["output"][2].getpixel((0, 0)), (255, 255, 255))

    def test_extend_clean_plate_mirrors_clean_strip_to_the_left(self):
        frame = Image.new("RGB", (6, 2), "black")
        frame.putpixel((3, 0), (10, 0, 0))
        frame.putpixel((4, 0), (20, 0, 0))

        result = ExtendCleanPlate().execute(
            video=[frame], boundary_x=3, extend_left=2, mode="mirror", top=0, bottom=1
        )

        self.assertEqual(result["frames"], 1)
        self.assertEqual(result["output"][0].getpixel((1, 0)), (20, 0, 0))
        self.assertEqual(result["output"][0].getpixel((2, 0)), (10, 0, 0))
        self.assertEqual(result["output"][0].getpixel((3, 0)), (10, 0, 0))

    def test_keyframe_jobs_pair_neighbors_and_chain_loop_results(self):
        keyframes = [solid("red")[0], solid("green")[0], solid("blue")[0]]
        jobs = FirstLastSegmentBuilder().execute(
            keyframes=keyframes,
            prompts='["move right", "move closer"]',
            settings='{"steps": 20}',
        )
        chain = KeyframeChain().execute(
            clips=[solid("red", 3), solid("blue", 3)],
            boundary="drop_duplicate",
            fps=2,
        )

        self.assertEqual(jobs["count"], 2)
        self.assertIs(jobs["jobs"][0]["last_frame"], keyframes[1])
        self.assertEqual(jobs["jobs"][1]["prompt"], "move closer")
        self.assertEqual(chain["frames"], 5)


if __name__ == "__main__":
    unittest.main()
