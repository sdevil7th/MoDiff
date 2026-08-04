import unittest
from unittest.mock import patch

from utils.memory_menager import MemoryManager


class OffloadedPipelineStub:
    def __init__(self):
        self.to_calls = []

    def to(self, device):
        self.to_calls.append(device)
        raise AssertionError("clear must not materialize a discarded offloaded pipeline on CPU")


class MemoryManagerCleanupTests(unittest.TestCase):
    def test_clear_drops_offloaded_models_without_moving_them_to_cpu(self):
        manager = MemoryManager()
        pipeline = OffloadedPipelineStub()
        manager.add(pipeline)

        with patch("utils.memory_menager.memory_flush") as flush:
            cleared = manager.clear()

        self.assertEqual(cleared, 1)
        self.assertEqual(manager.cache, {})
        self.assertEqual(pipeline.to_calls, [])
        flush.assert_called_once_with()
