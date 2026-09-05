import os
from pathlib import Path
import tempfile
import unittest

from modiff.hf_incomplete_cleanup import (
    HfIncompleteCleanupError,
    build_hf_incomplete_cleanup_plan,
    cleanup_hf_incomplete_files,
)


class HfIncompleteCleanupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.blobs = self.root / "models--org--model" / "blobs"
        self.blobs.mkdir(parents=True)
        self.models = [{"id": "org/model", "cache_dir": str(self.root)}]

    def tearDown(self):
        self.temporary.cleanup()

    def write_incomplete(self, name="weight.incomplete", *, age_seconds=7200, contents=b"partial"):
        path = self.blobs / name
        path.write_bytes(contents)
        timestamp = 10_000 - age_seconds
        os.utime(path, (timestamp, timestamp))
        return path

    def plan(self, **overrides):
        arguments = {
            "models": self.models,
            "current_task": None,
            "queued_tasks": {},
            "download_repos": {},
            "template_gallery_active": False,
            "now": 10_000,
        }
        arguments.update(overrides)
        return build_hf_incomplete_cleanup_plan(**arguments)

    def test_stale_regular_blob_has_hash_bound_plan_and_exact_cleanup(self):
        path = self.write_incomplete()
        plan = self.plan()

        self.assertTrue(plan["canCleanup"])
        self.assertEqual(plan["eligibleFileCount"], 1)
        self.assertEqual(plan["eligibleBytes"], 7)
        self.assertRegex(plan["planHash"], r"^sha256:[0-9a-f]{64}$")

        result = cleanup_hf_incomplete_files(plan)
        self.assertEqual(result["removedFileCount"], 1)
        self.assertEqual(result["removedBytes"], 7)
        self.assertFalse(path.exists())

    def test_recent_blob_and_active_work_block_cleanup(self):
        self.write_incomplete(age_seconds=30)
        plan = self.plan(current_task={"task_id": "active"})

        self.assertFalse(plan["canCleanup"])
        self.assertEqual(
            [blocker["code"] for blocker in plan["blockers"]],
            ["graph_execution_active", "recent_incomplete_download"],
        )

    def test_plan_hash_does_not_tick_with_display_age(self):
        self.write_incomplete()

        first = self.plan(now=10_000)
        second = self.plan(now=10_030)

        self.assertNotEqual(first["files"][0]["ageSeconds"], second["files"][0]["ageSeconds"])
        self.assertEqual(first["planHash"], second["planHash"])

    def test_changed_blob_fails_closed(self):
        path = self.write_incomplete()
        plan = self.plan()
        path.write_bytes(b"changed")

        with self.assertRaisesRegex(HfIncompleteCleanupError, "changed"):
            cleanup_hf_incomplete_files(plan)
        self.assertTrue(path.exists())

    def test_symlink_and_non_blob_incomplete_files_are_ignored(self):
        target = self.root / "target"
        target.write_bytes(b"partial")
        (self.blobs / "linked.incomplete").symlink_to(target)
        outside = self.root / "models--org--model" / "snapshots" / "revision"
        outside.mkdir(parents=True)
        (outside / "weight.incomplete").write_bytes(b"partial")

        plan = self.plan()
        self.assertEqual(plan["files"], [])
        self.assertFalse(plan["canCleanup"])


if __name__ == "__main__":
    unittest.main()
