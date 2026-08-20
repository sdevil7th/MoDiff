import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from modiff.original_input_fixtures import (
    OriginalInputFixtureError,
    build_original_input_fixture_ledger,
    write_fixture_kit,
    write_original_input_fixtures,
)


ROOT = Path(__file__).resolve().parents[1]
try:
    import imageio.v2  # noqa: F401
    import numpy  # noqa: F401
except ImportError:
    _HAS_VIDEO_WRITER = False
else:
    _HAS_VIDEO_WRITER = True


def _asset(name, kind):
    return {
        "fileName": name,
        "relativePath": f"kit/{name}",
        "mediaKind": kind,
        "byteSize": 12,
        "sha256": "a" * 64,
        "provenance": "original_modiff_procedural_v1",
    }


def _authoring(*items, workflow="AnimateDiffControlNetPipeline:control_to_video"):
    return {
        "specifications": [
            {
                "id": f"template-authoring:{workflow}",
                "canonicalWorkflowId": workflow,
                "authoringState": "draft_complete_input_selection_pending",
                "mode": "control_to_video",
                "mediaKind": "video",
                "inputPlan": {"status": "selection_required", "items": list(items)},
                "claims": {"inputSelected": False},
            }
        ]
    }


class OriginalInputFixtureTests(unittest.TestCase):
    def test_maps_pending_fields_without_selecting_the_authoring_ledger(self):
        assets = {
            "control_motion.mp4": _asset("control_motion.mp4", "video"),
            "still_life.png": _asset("still_life.png", "image"),
            "tram_stop.png": _asset("tram_stop.png", "image"),
        }
        ledger = build_original_input_fixture_ledger(
            Path("."),
            authoring=_authoring(
                {
                    "field": "controlVideo",
                    "mediaKind": "video",
                    "minimumCount": 1,
                    "technicalRequirements": ["decodable_video"],
                },
                {
                    "field": "referenceImages",
                    "mediaKind": "image",
                    "minimumCount": 2,
                    "technicalRequirements": ["decodable_image"],
                },
            ),
            assets=assets,
        )
        self.assertEqual(ledger["kind"], "original_input_fixture_ledger")
        self.assertFalse(ledger["boundary"]["selectsInputAssets"])
        self.assertFalse(ledger["boundary"]["authoringSpecMutated"])
        self.assertFalse(ledger["boundary"]["thirdPartyMedia"])
        mapping = ledger["workflows"][0]
        self.assertFalse(mapping["claims"]["inputSelected"])
        self.assertEqual(mapping["items"][0]["fixtures"][0]["fileName"], "control_motion.mp4")
        self.assertEqual(
            [item["fileName"] for item in mapping["items"][1]["fixtures"]],
            ["still_life.png", "tram_stop.png"],
        )
        self.assertEqual(mapping["items"][0]["rightsState"], "review_required")
        self.assertFalse(mapping["items"][0]["authoringLedgerSelected"])

    def test_unknown_field_is_fail_closed(self):
        with self.assertRaisesRegex(OriginalInputFixtureError, "No original fixture mapping"):
            build_original_input_fixture_ledger(
                Path("."),
                authoring=_authoring({"field": "unknownField", "mediaKind": "image", "minimumCount": 1}),
                assets={},
            )

    @unittest.skipUnless(_HAS_VIDEO_WRITER, "imageio/numpy are required to write original video fixtures")
    def test_kit_writes_original_png_and_wav_bytes(self):
        with TemporaryDirectory() as directory:
            destination = Path(directory)
            assets = write_fixture_kit(destination)
            still = destination / "kit" / "still_life.png"
            tone = destination / "kit" / "tone_a.wav"
            self.assertTrue(still.is_file())
            self.assertTrue(tone.is_file())
            with Image.open(still) as image:
                self.assertEqual(image.size, (512, 512))
            self.assertEqual(assets["still_life.png"]["provenance"], "original_modiff_procedural_v1")
            self.assertEqual(len(assets["still_life.png"]["sha256"]), 64)


class OriginalInputFixtureLiveLedgerTests(unittest.TestCase):
    def test_pending_authoring_specs_map_to_original_fixtures_without_mutating_the_ledger(self):
        path = ROOT / "data" / "template-authoring-specs.v1.json"
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        if not _HAS_VIDEO_WRITER:
            self.skipTest("imageio/numpy are required to write original video fixtures.")
        with TemporaryDirectory() as directory:
            ledger = write_original_input_fixtures(ROOT, destination=Path(directory))
        after = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(before, after)
        self.assertEqual(ledger["summary"]["workflowCount"], 92)
        self.assertTrue(ledger["policy"]["doesNotMutateAuthoringLedger"])
        self.assertTrue(ledger["policy"]["doesNotSelectInputs"])
        mapped = {item["canonicalWorkflowId"] for item in ledger["workflows"]}
        self.assertIn("AnimateDiffControlNetPipeline:control_to_video", mapped)
        self.assertEqual(ledger["boundary"]["authoringSpecMutated"], False)


if __name__ == "__main__":
    unittest.main()
