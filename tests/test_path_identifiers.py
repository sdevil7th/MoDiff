import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from modiff.config import CONFIG
from modiff.media_assets import coerce_video_asset
from modiff.path_identifiers import (
    data_path_identifier,
    resolve_data_path_identifier,
    resolve_managed_path_identifier,
    resolve_runtime_input_path,
)


class PathIdentifierTests(unittest.TestCase):
    def test_separate_data_root_uses_portable_identifier_without_relating_to_work_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work-volume"
            data = root / "data-volume"
            media = data / "images" / "source.png"
            work.mkdir()
            media.parent.mkdir(parents=True)
            media.write_bytes(b"image")

            identifier = data_path_identifier(media, data)

            self.assertEqual(identifier, "@data/images/source.png")
            self.assertNotIn(str(root), identifier)
            self.assertEqual(resolve_data_path_identifier(identifier, data), media.resolve())
            self.assertEqual(
                resolve_managed_path_identifier(identifier, work_root=work, data_root=data),
                media.resolve(),
            )
            self.assertEqual(
                resolve_runtime_input_path(identifier, work_root=work, data_root=data),
                media.resolve(),
            )

    def test_legacy_work_relative_data_identifier_remains_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            data = work / "data"
            media = data / "images" / "legacy.png"
            media.parent.mkdir(parents=True)
            media.write_bytes(b"image")

            resolved = resolve_managed_path_identifier(
                "data/images/legacy.png",
                work_root=work,
                data_root=data,
            )

            self.assertEqual(resolved, media.resolve())

    def test_data_identifier_rejects_traversal_and_malformed_namespace_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary) / "data"
            work = Path(temporary) / "work"
            data.mkdir()
            work.mkdir()

            for identifier in (
                "@data/../secret.png",
                "@data/images/../../secret.png",
                "@data//secret.png",
                "@data\\..\\secret.png",
                "@data/C:/secret.png",
                "@database/secret.png",
            ):
                with self.subTest(identifier=identifier):
                    self.assertIsNone(
                        resolve_managed_path_identifier(identifier, work_root=work, data_root=data)
                    )

    def test_data_identifier_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            external = root / "external"
            data.mkdir()
            external.mkdir()
            (external / "secret.png").write_bytes(b"secret")
            try:
                (data / "linked").symlink_to(external, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"Symlinks are unavailable: {exc}")

            with self.assertRaisesRegex(ValueError, "escapes"):
                resolve_data_path_identifier("@data/linked/secret.png", data)

    def test_media_loader_consumers_resolve_server_issued_identifier(self):
        from modules.Audio.main import _resolve_file
        from modules.Image.main import Load as LoadImage
        from modules.MediaSource.main import LocalMedia

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            data = root / "data"
            image_path = data / "images" / "source.png"
            audio_path = data / "audio" / "source.wav"
            video_path = data / "videos" / "source.mp4"
            work.mkdir()
            image_path.parent.mkdir(parents=True)
            audio_path.parent.mkdir(parents=True)
            video_path.parent.mkdir(parents=True)
            Image.new("RGB", (2, 3), "orange").save(image_path)
            audio_path.write_bytes(b"wav")
            video_path.write_bytes(b"video")

            with patch.dict(
                CONFIG.paths,
                {"work_dir": str(work), "data": str(data)},
            ):
                image = LoadImage().execute(file="@data/images/source.png")
                local = LocalMedia().execute(file="@data/images/source.png")
                audio = _resolve_file("@data/audio/source.wav")
                video = coerce_video_asset(
                    {
                        "path": "@data/videos/source.mp4",
                        "width": 2,
                        "height": 2,
                        "fps": 1,
                        "frame_count": 1,
                        "duration_seconds": 1,
                    }
                )

            self.assertEqual(image["image"].size, (2, 3))
            self.assertEqual(local["path"], str(image_path.resolve()))
            self.assertEqual(audio, audio_path.resolve())
            self.assertEqual(video["path"], str(video_path.resolve()))


if __name__ == "__main__":
    unittest.main()
