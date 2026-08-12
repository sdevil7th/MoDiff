import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from huggingface_hub.utils import LocalEntryNotFoundError

from modiff.auxiliary_ip_adapter import resolve_reviewed_sdxl_ip_adapter
from modiff.model_artifact_catalog import catalog_repository_pin


REPOSITORY = "h94/IP-Adapter"
REVISION = "018e402774aeeddd60609b4ecdb7e298259dc729"
WEIGHT = "sdxl_models/ip-adapter_sdxl.safetensors"


def _pin(payload=b"reviewed-ip-adapter"):
    return {
        "repo": REPOSITORY,
        "revision": REVISION,
        "purpose": "sdxl-ip-adapter",
        "weightName": WEIGHT,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "byteSize": len(payload),
        "imageEncoderSubfolder": "models/image_encoder",
        "imageEncoderClass": "CLIPVisionModelWithProjection",
    }


class AuxiliaryIPAdapterContractTests(unittest.TestCase):
    def test_catalog_declares_the_exact_reviewed_single_adapter(self):
        pin = catalog_repository_pin(REPOSITORY)
        self.assertEqual(pin["kind"], "auxiliary")
        self.assertEqual(pin["revision"], REVISION)
        self.assertEqual(pin["purpose"], "sdxl-ip-adapter")
        self.assertEqual(pin["weightName"], WEIGHT)
        self.assertEqual(pin["sha256"], "ba1002529e783604c5f326d49f0122025392d1d20ac8d573b3eeb3e6dea4ebb6")
        self.assertEqual(pin["byteSize"], 702585376)
        self.assertEqual(pin["imageEncoderSubfolder"], "models/image_encoder")
        self.assertEqual(pin["imageEncoderClass"], "CLIPVisionModelWithProjection")

    def test_resolution_is_local_only_and_content_addressed(self):
        payload = b"reviewed-ip-adapter"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ip-adapter_sdxl.safetensors"
            path.write_bytes(payload)
            with (
                patch("modiff.auxiliary_ip_adapter.catalog_repository_pin", return_value=_pin(payload)),
                patch("modiff.auxiliary_ip_adapter.hf_hub_download", return_value=str(path)) as download,
            ):
                resolved = resolve_reviewed_sdxl_ip_adapter(
                    selection={"source": "hub", "value": REPOSITORY},
                    revision=REVISION,
                    weight_name=WEIGHT,
                )

        download.assert_called_once_with(
            repo_id=REPOSITORY,
            revision=REVISION,
            filename=WEIGHT,
            local_files_only=True,
        )
        self.assertEqual(resolved.repository, REPOSITORY)
        self.assertEqual(resolved.revision, REVISION)
        self.assertEqual(resolved.weight_name, "ip-adapter_sdxl.safetensors")
        self.assertEqual(resolved.content_sha256, hashlib.sha256(payload).hexdigest())
        self.assertEqual(resolved.byte_size, len(payload))

    def test_unreviewed_selectors_and_mutable_identity_fail_before_cache_access(self):
        cases = (
            (None, REVISION, WEIGHT, _pin(), "model-selector object"),
            ({"source": "local", "value": REPOSITORY}, REVISION, WEIGHT, _pin(), "only reviewed immutable Hub"),
            ({"source": "hub", "value": "other/repo"}, REVISION, WEIGHT, None, "not a reviewed"),
            ({"source": "hub", "value": REPOSITORY}, "main", WEIGHT, _pin(), "immutable repository revision"),
            ({"source": "hub", "value": REPOSITORY}, REVISION, "../adapter.bin", _pin(), "traversal-free"),
        )
        for selection, revision, weight, pin, message in cases:
            with (
                self.subTest(message=message),
                patch("modiff.auxiliary_ip_adapter.catalog_repository_pin", return_value=pin),
                patch("modiff.auxiliary_ip_adapter.hf_hub_download") as download,
                self.assertRaisesRegex((TypeError, ValueError), message),
            ):
                resolve_reviewed_sdxl_ip_adapter(
                    selection=selection,
                    revision=revision,
                    weight_name=weight,
                )
            download.assert_not_called()

    def test_missing_size_or_digest_mismatch_never_falls_back_to_network(self):
        with (
            patch("modiff.auxiliary_ip_adapter.catalog_repository_pin", return_value=_pin()),
            patch(
                "modiff.auxiliary_ip_adapter.hf_hub_download",
                side_effect=LocalEntryNotFoundError("private-cache-marker"),
            ) as download,
            self.assertRaisesRegex(FileNotFoundError, "never downloads"),
        ):
            resolve_reviewed_sdxl_ip_adapter(
                selection={"source": "hub", "value": REPOSITORY},
                revision=REVISION,
                weight_name=WEIGHT,
            )
        self.assertTrue(download.call_args.kwargs["local_files_only"])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ip-adapter_sdxl.safetensors"
            path.write_bytes(b"tampered")
            for pin, message in (
                ({**_pin(b"tampered"), "byteSize": 99}, "byte size"),
                ({**_pin(b"tampered"), "sha256": "0" * 64}, "SHA-256"),
            ):
                with (
                    self.subTest(message=message),
                    patch("modiff.auxiliary_ip_adapter.catalog_repository_pin", return_value=pin),
                    patch("modiff.auxiliary_ip_adapter.hf_hub_download", return_value=str(path)),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    resolve_reviewed_sdxl_ip_adapter(
                        selection={"source": "hub", "value": REPOSITORY},
                        revision=REVISION,
                        weight_name=WEIGHT,
                    )


if __name__ == "__main__":
    unittest.main()
