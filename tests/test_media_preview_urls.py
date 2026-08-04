import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.io import wavfile
from unittest.mock import patch

from modiff import media_io, server as server_module
from modiff.media_io import export_media_file
from modiff.server import (
    WebServer,
    audio_download_filename,
    byte_range_response,
    file_backed_media_preview,
    parse_audio_download_sample_rate,
    resample_wav_bytes,
)


class MediaPreviewUrlTests(unittest.TestCase):
    def test_file_backed_media_uses_file_route(self):
        url = file_backed_media_preview("audio/source track.wav")

        self.assertTrue(url.startswith("/file?file=audio%2Fsource%20track.wav&t="))

    def test_existing_browser_url_is_preserved(self):
        self.assertEqual(
            file_backed_media_preview("https://example.test/source.wav"),
            "https://example.test/source.wav",
        )

    def test_non_path_media_uses_cache_fallback(self):
        self.assertIsNone(file_backed_media_preview({"samples": []}))

    def test_cached_media_supports_browser_byte_ranges(self):
        response = byte_range_response(
            SimpleNamespace(headers={"Range": "bytes=2-5"}),
            b"0123456789",
            content_type="audio/wav",
            filename="output.wav",
        )

        self.assertEqual(response.status, 206)
        self.assertEqual(response.body, b"2345")
        self.assertEqual(response.headers["Accept-Ranges"], "bytes")
        self.assertEqual(response.headers["Content-Range"], "bytes 2-5/10")
        self.assertEqual(response.headers["Content-Length"], "4")

    def test_cached_media_rejects_unsatisfiable_range(self):
        response = byte_range_response(
            SimpleNamespace(headers={"Range": "bytes=20-30"}),
            b"0123456789",
            content_type="audio/wav",
        )

        self.assertEqual(response.status, 416)
        self.assertEqual(response.headers["Content-Range"], "bytes */10")

    def test_download_sample_rate_rewrites_the_wav_header_and_duration(self):
        source = BytesIO()
        wavfile.write(source, 48000, np.arange(48000, dtype=np.int16))

        converted = resample_wav_bytes(source.getvalue(), 44100)
        converted_rate, converted_samples = wavfile.read(BytesIO(converted))

        self.assertEqual(converted_rate, 44100)
        self.assertEqual(converted_samples.shape[0], 44100)

    def test_download_sample_rate_contract_rejects_unsupported_rates(self):
        self.assertEqual(parse_audio_download_sample_rate("44100"), 44100)
        self.assertIsNone(parse_audio_download_sample_rate(None))
        with self.assertRaisesRegex(ValueError, "must be one of"):
            parse_audio_download_sample_rate("22050")

    def test_download_filename_records_the_actual_export_rate(self):
        self.assertEqual(audio_download_filename("mix.wav", 44100), "mix-44.1kHz.wav")

    def test_concurrent_media_exports_create_one_atomic_cached_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            Image.new("RGB", (4, 3), "navy").save(source)
            export_root = root / "exports"

            def fake_export(_source, destination, _format_id, _options):
                destination.write_bytes(b"encoded")

            with patch.object(media_io, "_image_export", side_effect=fake_export) as encode:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    results = list(
                        executor.map(
                            lambda _index: export_media_file(
                                source,
                                kind="image",
                                format_id="png",
                                cache_root=export_root,
                            ),
                            range(4),
                        )
                    )
            leftovers = list(export_root.glob(".*"))

        self.assertEqual(len({result[0] for result in results}), 1)
        self.assertEqual(encode.call_count, 1)
        self.assertEqual(leftovers, [])


class WorkspaceFileRouteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "work"
        self.workspace.mkdir()
        self.sibling = self.root / "work-secret"
        self.sibling.mkdir()
        self.secret = self.sibling / "secret.png"
        Image.new("RGB", (2, 2), "red").save(self.secret)
        self.server = WebServer(modules={}, work_dir=str(self.workspace), data_dir=str(self.workspace))

    def tearDown(self):
        self.temporary.cleanup()

    async def test_file_routes_reject_traversal_and_sibling_prefixes(self):
        list_response = await self.server.listdir(
            SimpleNamespace(query={"path": "../work-secret"})
        )
        preview_response = await self.server.preview(
            SimpleNamespace(query={"file": str(self.secret)})
        )
        stream_response = await self.server.stream(
            SimpleNamespace(query={"file": "../work-secret/secret.png"})
        )

        self.assertEqual(list_response.status, 403)
        self.assertEqual(preview_response.status, 403)
        self.assertEqual(stream_response.status, 403)
        self.assertIn("outside", json.loads(preview_response.text)["error"])

    async def test_file_routes_reject_resolved_link_escapes(self):
        escaped_candidate = self.workspace / "linked-secret" / "secret.png"
        original_resolve = Path.resolve

        def resolve_with_escape(path, strict=False):
            if path == escaped_candidate:
                return self.secret
            return original_resolve(path, strict=strict)

        with patch.object(Path, "resolve", resolve_with_escape):
            response = await self.server.stream(
                SimpleNamespace(query={"file": "linked-secret/secret.png"})
            )

        self.assertEqual(response.status, 403)

    async def test_valid_workspace_files_remain_available(self):
        image_path = self.workspace / "image.png"
        Image.new("RGB", (3, 2), "blue").save(image_path)

        listing = json.loads((await self.server.listdir(SimpleNamespace(query={"path": "."}))).text)
        preview = await self.server.preview(SimpleNamespace(query={"file": "image.png"}))
        stream = await self.server.stream(SimpleNamespace(query={"file": "image.png"}))

        self.assertEqual([item["name"] for item in listing["files"]], ["image.png"])
        self.assertEqual(preview.status, 200)
        self.assertEqual(stream.status, 200)

    async def test_preview_decoding_uses_a_worker_without_disabling_pillow_limits(self):
        image_path = self.workspace / "threaded.png"
        Image.new("RGB", (3, 2), "blue").save(image_path)
        original_limit = Image.MAX_IMAGE_PIXELS
        calls = []

        async def run_in_worker(function, *args, **kwargs):
            calls.append(function)
            return function(*args, **kwargs)

        with patch.object(server_module.asyncio, "to_thread", side_effect=run_in_worker):
            response = await self.server.preview(SimpleNamespace(query={"file": "threaded.png"}))

        self.assertEqual(response.status, 200)
        self.assertEqual(calls, [server_module.render_image_preview])
        self.assertEqual(Image.MAX_IMAGE_PIXELS, original_limit)

    async def test_preview_rejects_malformed_dimensions_and_oversized_sources(self):
        image_path = self.workspace / "bounded.png"
        Image.new("RGB", (3, 2), "blue").save(image_path)

        malformed = await self.server.preview(
            SimpleNamespace(query={"file": "bounded.png", "width": "not-a-number"})
        )
        with patch.object(server_module, "MAX_PREVIEW_IMAGE_PIXELS", 4):
            oversized = await self.server.preview(SimpleNamespace(query={"file": "bounded.png"}))
        with patch("PIL.Image.open", side_effect=Image.DecompressionBombError("bomb")):
            pillow_bomb = await self.server.preview(SimpleNamespace(query={"file": "bounded.png"}))

        self.assertEqual(malformed.status, 400)
        self.assertEqual(oversized.status, 400)
        self.assertEqual(pillow_bomb.status, 400)


class SeparateDataRootUploadTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "work-volume"
        self.data = self.root / "data-volume"
        self.workspace.mkdir()
        self.data.mkdir()
        self.server = WebServer(modules={}, work_dir=str(self.workspace), data_dir=str(self.data))

    def tearDown(self):
        self.temporary.cleanup()

    async def test_upload_returns_data_root_identifier_and_every_file_route_resolves_it(self):
        encoded = BytesIO()
        Image.new("RGB", (3, 2), "green").save(encoded, format="PNG")
        upload = SimpleNamespace(filename="source.png", file=BytesIO(encoded.getvalue()))

        class UploadRequest:
            async def post(self):
                return {"file": upload, "type": "images"}

        upload_response = await self.server.filePost(UploadRequest())
        payload = json.loads(upload_response.text)
        identifier = payload["path"]

        self.assertEqual(upload_response.status, 200)
        self.assertEqual(identifier, "@data/images/source.png")
        self.assertNotIn(str(self.root), upload_response.text)
        self.assertTrue((self.data / "images" / "source.png").is_file())

        file_response = await self.server.fileGet(SimpleNamespace(query={"file": identifier}))
        preview_response = await self.server.preview(SimpleNamespace(query={"file": identifier}))
        stream_response = await self.server.stream(SimpleNamespace(query={"file": identifier}))
        probe_response = await self.server.media_probe(
            SimpleNamespace(query={"file": identifier, "media_kind": "image"})
        )

        self.assertEqual(file_response.status, 200)
        self.assertEqual(preview_response.status, 200)
        self.assertEqual(stream_response.status, 200)
        self.assertEqual(probe_response.status, 200)

    async def test_data_listing_uses_identifiers_and_traversal_is_rejected(self):
        image_path = self.data / "images" / "listed.png"
        image_path.parent.mkdir(parents=True)
        Image.new("RGB", (2, 2), "purple").save(image_path)

        listing_response = await self.server.listdir(
            SimpleNamespace(query={"path": "@data/images"})
        )
        listing = json.loads(listing_response.text)

        self.assertEqual(listing_response.status, 200)
        self.assertEqual(listing["path"], "@data/images")
        self.assertEqual(listing["abs_path"], "@data/images")
        self.assertEqual(listing["files"][0]["path"], "@data/images/listed.png")
        self.assertNotIn(str(self.root), listing_response.text)

        traversal = "@data/images/../../outside.png"
        responses = [
            await self.server.fileGet(SimpleNamespace(query={"file": traversal})),
            await self.server.preview(SimpleNamespace(query={"file": traversal})),
            await self.server.stream(SimpleNamespace(query={"file": traversal})),
            await self.server.listdir(SimpleNamespace(query={"path": traversal})),
        ]
        self.assertTrue(all(response.status == 403 for response in responses))


if __name__ == "__main__":
    unittest.main()
