import tempfile
import unittest
from pathlib import Path

from modiff.media_assets import cleanup_media_assets, coerce_video_asset, list_media_assets, register_video_asset


class MediaAssetTests(unittest.TestCase):
    def test_registered_asset_keeps_operation_provenance_and_can_be_coerced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "derived.mp4"
            path.write_bytes(b"video")
            record = register_video_asset(
                path,
                asset_id="derived",
                width=8,
                height=6,
                fps=2,
                frame_count=4,
                source_asset_ids=["source-a"],
                operation="trim",
                root=root,
            )

            coerced = coerce_video_asset(record)

            self.assertEqual(coerced["path"], str(path.resolve()))
            self.assertEqual(coerced["source_asset_ids"], ["source-a"])
            self.assertEqual(coerced["operation"], "trim")

    def test_cleanup_is_scoped_and_never_removes_pinned_or_external_files(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as external_directory:
            root = Path(directory)
            run_a = root / "run-a"
            run_a.mkdir()
            ordinary = run_a / "ordinary.mp4"
            pinned = run_a / "pinned.mp4"
            external = Path(external_directory) / "external.mp4"
            for path in (ordinary, pinned, external):
                path.write_bytes(b"video")
            register_video_asset(ordinary, asset_id="ordinary", task_id="a", width=8, height=6, fps=2, frame_count=4, root=root)
            register_video_asset(pinned, asset_id="pinned", task_id="a", width=8, height=6, fps=2, frame_count=4, pinned=True, root=root)
            register_video_asset(external, asset_id="external", task_id="a", width=8, height=6, fps=2, frame_count=4, root=root)

            report = cleanup_media_assets(task_id="a", root=root)

            self.assertFalse(ordinary.exists())
            self.assertTrue(pinned.exists())
            self.assertTrue(external.exists())
            self.assertEqual([item["asset_id"] for item in report["removed"]], ["ordinary"])
            self.assertEqual({item["asset_id"] for item in list_media_assets(root=root)}, {"pinned", "external"})


if __name__ == "__main__":
    unittest.main()
