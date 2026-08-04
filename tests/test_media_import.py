import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from modiff.config import CONFIG
from modiff.media_import import (
    _PinnedHTTPConnection,
    _public_http_url,
    audio_as_wav,
    import_root,
    import_web_media,
    import_youtube_media,
)
from modules.MediaSource.main import LocalMedia, YouTubeMedia


class MediaImportTests(unittest.TestCase):
    def test_default_import_root_uses_configured_data_directory(self):
        self.assertEqual(import_root(), (Path(CONFIG.paths["data"]) / "imports").resolve())

    @patch("modiff.media_import.socket.getaddrinfo", return_value=[(None, None, None, None, ("127.0.0.1", 0))])
    def test_private_web_hosts_are_rejected(self, _resolve):
        with self.assertRaisesRegex(ValueError, "private"):
            _public_http_url("http://internal.example/image.png")

    @patch("modiff.media_import.socket.socket")
    @patch(
        "modiff.media_import.socket.getaddrinfo",
        return_value=[(2, 1, 6, "", ("93.184.216.34", 80))],
    )
    def test_http_connection_uses_the_validated_numeric_address(self, _resolve, socket_mock):
        connection = _PinnedHTTPConnection("example.com", 80, timeout=3)

        connection.connect()

        socket_mock.assert_called_once_with(2, 1)
        socket_mock.return_value.connect.assert_called_once_with(("93.184.216.34", 80))

    @patch("modiff.media_import.socket.socket")
    @patch(
        "modiff.media_import.socket.getaddrinfo",
        return_value=[(2, 1, 6, "", ("127.0.0.1", 80))],
    )
    def test_connection_time_dns_rebinding_to_private_address_is_rejected(self, _resolve, socket_mock):
        connection = _PinnedHTTPConnection("rebound.example", 80, timeout=3)

        with self.assertRaisesRegex(ValueError, "private"):
            connection.connect()

        socket_mock.assert_not_called()

    @patch("modiff.media_import.socket.getaddrinfo", return_value=[(None, None, None, None, ("93.184.216.34", 0))])
    def test_credentials_are_rejected(self, _resolve):
        with self.assertRaisesRegex(ValueError, "public HTTP"):
            _public_http_url("https://user:pass@example.com/file.png")

    @patch("modiff.media_import.socket.getaddrinfo", return_value=[(None, None, None, None, ("142.250.0.1", 0))])
    def test_youtube_host_allowlist(self, _resolve):
        self.assertIn("youtube.com", _public_http_url("https://www.youtube.com/watch?v=test", youtube_only=True))
        with self.assertRaisesRegex(ValueError, "YouTube"):
            _public_http_url("https://example.com/watch?v=test", youtube_only=True)

    def test_local_media_returns_existing_absolute_path(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "input.png"
            path.write_bytes(b"test")
            self.assertEqual(LocalMedia("local").execute(file=str(path))["path"], str(path.resolve()))

    def test_youtube_node_requires_rights_confirmation(self):
        with self.assertRaisesRegex(ValueError, "permission"):
            YouTubeMedia("youtube").execute(url="https://youtu.be/test", rights_confirmed=False)

    @patch("modiff.media_import.build_opener")
    @patch("modiff.media_import.socket.getaddrinfo", return_value=[(None, None, None, None, ("93.184.216.34", 0))])
    def test_web_media_is_content_addressed(self, _resolve, build_opener_mock):
        response = MagicMock()
        response.headers.get_content_type.return_value = "image/png"
        response.headers.get.return_value = str(len(b"png-data"))
        response.geturl.return_value = "https://example.com/image.png"
        response.read = MagicMock(side_effect=[b"png-data", b""])
        response.__enter__.return_value = response
        build_opener_mock.return_value.open.return_value = response
        with TemporaryDirectory() as directory:
            first = import_web_media("https://example.com/image.png", root=directory)
            response.read = MagicMock(side_effect=[b"png-data", b""])
            second = import_web_media("https://example.com/image.png", root=directory)
            self.assertEqual(first, second)
            self.assertEqual(first.read_bytes(), b"png-data")

    @patch("modiff.media_import._public_http_url", return_value="https://youtu.be/test")
    def test_youtube_import_uses_single_item_and_duration_limit(self, _safe_url):
        fake_module = MagicMock()
        captured = {}

        class FakeDownloader:
            def __init__(self, options):
                captured.update(options)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def extract_info(self, _url, download=True):
                output = Path(captured["paths"]["home"]) / "test.mp4"
                output.write_bytes(b"video")
                return {"id": "test", "duration": 12}

        fake_module.YoutubeDL = FakeDownloader
        with patch.dict("sys.modules", {"yt_dlp": fake_module}), TemporaryDirectory() as directory:
            result = import_youtube_media("https://youtu.be/test", root=directory)
        self.assertTrue(result.name.startswith("test-"))
        self.assertTrue(captured["noplaylist"])
        self.assertEqual(captured["playlist_items"], "1")

    @patch("subprocess.run")
    def test_audio_transcode_is_cached(self, run_mock):
        def create_output(command, **_kwargs):
            Path(command[-1]).write_bytes(b"wav")
            return MagicMock(returncode=0, stderr="", stdout="")

        run_mock.side_effect = create_output
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp3"
            source.write_bytes(b"mp3")
            result = audio_as_wav(source, root=directory)
            self.assertEqual(result.suffix, ".wav")
            self.assertEqual(result.read_bytes(), b"wav")

    @patch("subprocess.run")
    def test_concurrent_audio_transcodes_share_the_atomic_cached_result(self, run_mock):
        temporary_paths = []

        def create_output(command, **_kwargs):
            temporary = Path(command[-1])
            temporary_paths.append(temporary)
            temporary.write_bytes(b"wav")
            return MagicMock(returncode=0, stderr="", stdout="")

        run_mock.side_effect = create_output
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp3"
            source.write_bytes(b"mp3")
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(lambda _index: audio_as_wav(source, root=directory), range(4)))
            leftovers = list((Path(directory) / "audio").glob(".*.wav"))

        self.assertEqual(len(set(results)), 1)
        self.assertEqual(run_mock.call_count, 1)
        self.assertEqual(len(temporary_paths), 1)
        self.assertNotEqual(temporary_paths[0], results[0])
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
