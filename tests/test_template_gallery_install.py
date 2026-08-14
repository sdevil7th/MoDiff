import hashlib
import json
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from modiff.template_gallery import (
    TemplateGalleryError,
    _path_is_link_or_reparse,
    install_template_gallery,
    plan_template_gallery_install,
    validate_template_gallery_manifest,
    verify_template_gallery_tree,
)


def canonical_json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def fixture_contract(files):
    records = []
    for path, content_type, purposes, data in files:
        records.append(
            {
                "path": path,
                "size": len(data),
                "sha256": "sha256:bytes:" + hashlib.sha256(data).hexdigest(),
                "contentType": content_type,
                "purposes": purposes,
            }
        )
    records.sort(key=lambda record: record["path"])
    identity = "sha256:canonical-json:" + hashlib.sha256(
        canonical_json_bytes({"schemaVersion": 1, "assetVersion": "v1", "assets": records})
    ).hexdigest()
    source = {
        "schemaVersion": 1,
        "mode": "huggingface",
        "localBasePath": "/template-gallery",
        "repoType": "dataset",
        "repoId": "unit/template-gallery",
        "revision": "a" * 40,
        "pathPrefix": "template-gallery",
        "assetManifestPath": "_modiff/template-assets.v1.json",
        "assetSetId": identity,
        "completeAssetSetId": identity,
        "unavailableAssets": [],
    }
    manifest = {
        "schemaVersion": 1,
        "assetVersion": "v1",
        "status": "ready",
        "assetSetId": identity,
        "assetCount": len(records),
        "totalBytes": sum(record["size"] for record in records),
        "missingAssets": [],
        "assets": records,
    }
    return source, manifest


class TemplateGalleryInstallTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.files = [
            ("template-gallery/manifest.json", "application/json", ["runtime"], b"{\"examples\":[]}"),
            (
                "template-gallery/runtime-inputs/assets/input.png",
                "image/png",
                ["runtime"],
                b"exact image bytes",
            ),
        ]
        self.source, self.manifest = fixture_contract(self.files)

    def tearDown(self):
        self.temporary.cleanup()

    def _write_contract(self):
        source_path = self.root / "web/assets/template-asset-source.v1.json"
        remote_manifest = self.root / "remote-template-assets.v1.json"
        source_path.parent.mkdir(parents=True)
        source_path.write_text(json.dumps(self.source), encoding="utf-8")
        remote_manifest.write_text(json.dumps(self.manifest), encoding="utf-8")
        return source_path, remote_manifest

    def _write_snapshot(self):
        snapshot = self.root / "snapshot"
        for path, _content_type, _purposes, data in self.files:
            target = snapshot / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return snapshot

    def test_manifest_identity_and_paths_fail_closed(self):
        validated = validate_template_gallery_manifest(self.source, self.manifest)
        self.assertEqual(validated["assetSetId"], self.source["assetSetId"])

        traversing = json.loads(json.dumps(self.manifest))
        traversing["assets"][0]["path"] = "template-gallery/../secret"
        with self.assertRaisesRegex(TemplateGalleryError, "unsafe"):
            validate_template_gallery_manifest(self.source, traversing)

        mismatched = json.loads(json.dumps(self.manifest))
        mismatched["assets"][0]["size"] += 1
        mismatched["totalBytes"] += 1
        with self.assertRaisesRegex(TemplateGalleryError, "immutable source identity"):
            validate_template_gallery_manifest(self.source, mismatched)

    def test_windows_reparse_points_are_treated_as_links(self):
        attributes = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        with patch.object(
            Path,
            "lstat",
            return_value=SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=attributes),
        ):
            self.assertTrue(_path_is_link_or_reparse(self.root / "junction"))

    def test_plan_reserves_download_staging_active_queue_and_safety_space(self):
        source_path, remote_manifest = self._write_contract()
        cache = self.root / "cache"
        gallery = self.root / "web/template-gallery"
        free_bytes = 10_000
        reserve_bytes = 1_000
        queued_bytes = 2_000
        with patch(
            "modiff.template_gallery.shutil.disk_usage",
            return_value=SimpleNamespace(total=20_000, used=10_000, free=free_bytes),
        ):
            _source, _manifest, plan = plan_template_gallery_install(
                source_path,
                gallery,
                cache_root=cache,
                queued_reservation_bytes=queued_bytes,
                reserve_bytes=reserve_bytes,
                download_file=lambda **_kwargs: str(remote_manifest),
            )

        self.assertTrue(plan["sizeKnown"])
        self.assertEqual(plan["queuedReservationBytes"], queued_bytes)
        self.assertEqual(
            plan["reservationBytes"],
            self.manifest["totalBytes"] * 2 + 8 * 1024**2,
        )
        self.assertFalse(plan["fitsWithQueue"])
        self.assertFalse(plan["installed"])
        self.assertFalse(cache.exists(), "A read-only install plan must not create the cache directory.")

    def test_install_stages_and_verifies_before_atomic_promotion(self):
        snapshot = self._write_snapshot()
        gallery = self.root / "web/template-gallery"
        receipt = self.root / "data/template-gallery-install.v1.json"
        result = install_template_gallery(
            self.source,
            self.manifest,
            gallery,
            cache_root=self.root / "cache",
            receipt_path=receipt,
            download_snapshot=lambda **_kwargs: str(snapshot),
        )

        self.assertTrue(result["complete"])
        self.assertTrue(result["restartRequired"])
        self.assertEqual(result["assetCount"], 2)
        self.assertEqual(json.loads(receipt.read_text())["assetSetId"], self.source["assetSetId"])
        self.assertTrue(verify_template_gallery_tree(gallery, self.manifest)["complete"])
        self.assertEqual((gallery / "runtime-inputs/assets/input.png").read_bytes(), b"exact image bytes")

    def test_failed_staging_preserves_existing_gallery(self):
        snapshot = self._write_snapshot()
        (snapshot / "template-gallery/runtime-inputs/assets/input.png").write_bytes(b"tampered image bytes")
        gallery = self.root / "web/template-gallery"
        gallery.mkdir(parents=True)
        marker = gallery / "existing.txt"
        marker.write_text("preserve me", encoding="utf-8")

        with self.assertRaisesRegex(TemplateGalleryError, "differs"):
            install_template_gallery(
                self.source,
                self.manifest,
                gallery,
                cache_root=self.root / "cache",
                download_snapshot=lambda **_kwargs: str(snapshot),
            )
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve me")
        self.assertFalse(any(gallery.parent.glob(".template-gallery-stage-*")))


if __name__ == "__main__":
    unittest.main()
