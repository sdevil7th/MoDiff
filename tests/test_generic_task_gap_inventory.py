from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from modiff.generic_task_gap_inventory import (
    GenericTaskGapInventoryError,
    build_generic_task_gap_inventory,
    write_generic_task_gap_inventory,
)


ROOT = Path(__file__).resolve().parents[1]


def _resolution(contract_id, *, mode, state, catalog_id=None, title="Example"):
    return {
        "contractId": contract_id,
        "catalogId": catalog_id or contract_id.rsplit(":", 1)[-1],
        "title": title,
        "selectedCandidateMode": mode,
        "resolutionState": state,
        "mappingMeaning": "test",
        "blockers": ["new_bounded_modiff_task_contract_required"] if "new_task" in state else ["parity"],
    }


class GenericTaskGapInventoryTests(unittest.TestCase):
    def test_no_remaining_weight_free_generic_task_is_authorable(self):
        ledger = build_generic_task_gap_inventory(
            Path("."),
            resolution_ledger={
                "resolutions": [
                    _resolution(
                        "comfy-research:template:utility_image_stitch",
                        mode="image_stitch",
                        state="existing_task_boundary_builtin_operation_review_required",
                    ),
                    _resolution(
                        "comfy-research:template:utility_interpolation_image_upscale",
                        mode="image_upscale",
                        state="existing_task_boundary_builtin_operation_review_required",
                    ),
                    _resolution(
                        "comfy-research:template:3d_hunyuan3d-v2.1",
                        mode="image_to_3d",
                        state="new_task_boundary_required",
                    ),
                ]
            },
        )
        self.assertEqual(ledger["summary"]["authorableWithoutWeightsCount"], 0)
        self.assertEqual(ledger["summary"]["implementedWithoutWeights"], ["comfy-research:template:utility_image_stitch"])
        self.assertEqual(
            ledger["summary"]["remainingBuiltinParityReviews"],
            ["comfy-research:template:utility_interpolation_image_upscale"],
        )
        self.assertEqual(ledger["summary"]["newTaskBoundaryCount"], 1)
        self.assertFalse(ledger["boundary"]["addsModelNamedNodes"])
        self.assertEqual(ledger["diffusersSideContractsStillNeedingGenericTasks"][0]["candidateTask"], "diffusion_text")
        self.assertEqual(ledger["summary"]["nextAuthorableWithoutWeights"], [])
        self.assertFalse(next(item for item in ledger["gaps"] if item["selectedCandidateMode"] == "image_stitch")["needsWeights"])
        self.assertTrue(next(item for item in ledger["gaps"] if item["selectedCandidateMode"] == "image_to_3d")["needsWeights"])

    def test_empty_resolutions_are_rejected(self):
        with self.assertRaisesRegex(GenericTaskGapInventoryError, "no resolutions"):
            build_generic_task_gap_inventory(Path("."), resolution_ledger={"resolutions": []})


class GenericTaskGapInventoryLiveLedgerTests(unittest.TestCase):
    def test_checked_in_resolution_ledger_has_28_new_tasks_and_zero_weight_free_work(self):
        path = ROOT / "data" / "research" / "comfy-contract-resolution.v1.json"
        if not path.is_file():
            self.skipTest("Comfy contract-resolution ledger is not present.")
        with TemporaryDirectory() as directory:
            destination = Path(directory)
            ledger = write_generic_task_gap_inventory(ROOT, destination=destination)
            self.assertTrue((destination / "inventory.v1.json").is_file())
            self.assertTrue((destination / "index.html").is_file())
        self.assertEqual(ledger["summary"]["newTaskBoundaryCount"], 28)
        self.assertEqual(ledger["summary"]["authorableWithoutWeightsCount"], 0)
        self.assertEqual(ledger["summary"]["implementedWithoutWeightsCount"], 1)
        self.assertEqual(ledger["summary"]["builtinParityReviewCount"], 1)
        self.assertEqual(ledger["summary"]["newTaskModeCounts"]["image_to_3d"], 7)
        self.assertEqual(ledger["summary"]["newTaskModeCounts"]["camera_to_video"], 3)
        self.assertIn(
            "diffusion_text",
            [row["candidateTask"] for row in ledger["diffusersSideContractsStillNeedingGenericTasks"]],
        )


if __name__ == "__main__":
    unittest.main()
