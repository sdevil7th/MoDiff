import unittest
from pathlib import Path
from sys import path


path.insert(0, str(Path(__file__).resolve().parents[1]))

from modiff.disk_activity import DiskActivityCounters, DiskActivitySampler  # noqa: E402


class DiskActivitySamplerTests(unittest.TestCase):
    def test_reports_interval_active_time_instead_of_capacity_used(self):
        samples = iter(
            (
                DiskActivityCounters("disk:0", 400.0, 1_000.0, "unit-test"),
                DiskActivityCounters("disk:0", 480.0, 2_000.0, "unit-test"),
            )
        )
        sampler = DiskActivitySampler(lambda _path: next(samples))

        self.assertEqual(sampler.sample("data"), (None, "unit-test"))
        active_percent, source = sampler.sample("data")

        self.assertEqual(active_percent, 8.0)
        self.assertEqual(source, "unit-test")

    def test_clamps_parallel_or_inconsistent_counters_to_one_hundred_percent(self):
        samples = iter(
            (
                DiskActivityCounters("disk:0", 0.0, 0.0, "unit-test"),
                DiskActivityCounters("disk:0", 1_500.0, 1_000.0, "unit-test"),
            )
        )
        sampler = DiskActivitySampler(lambda _path: next(samples))

        sampler.sample("data")

        self.assertEqual(sampler.sample("data"), (100.0, "unit-test"))

    def test_counter_failure_is_reported_as_unavailable(self):
        def fail(_path):
            raise OSError("unavailable")

        self.assertEqual(DiskActivitySampler(fail).sample("data"), (None, None))


if __name__ == "__main__":
    unittest.main()
