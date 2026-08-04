import os
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modiff.server import WebServer


class DeterministicModeTests(unittest.TestCase):
    def setUp(self):
        self.server = object.__new__(WebServer)
        self.graph = {
            "deterministicMode": {"enabled": True, "seed": 8201, "strict": False},
            "nodes": {},
        }

    def test_non_strict_mode_locks_rng_without_forcing_deterministic_kernels(self):
        with (
            patch("torch.manual_seed") as manual_seed,
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.manual_seed_all") as manual_seed_all,
            patch("torch.use_deterministic_algorithms") as deterministic_algorithms,
        ):
            applied = WebServer._apply_deterministic_mode(self.server, self.graph)

        manual_seed.assert_called_once_with(8201)
        manual_seed_all.assert_called_once_with(8201)
        deterministic_algorithms.assert_not_called()
        self.assertFalse(applied["strict"])
        self.assertTrue(applied["settings"]["torch_manual_seed"])
        self.assertFalse(applied["settings"]["torch_deterministic_algorithms"])

    def test_strict_mode_still_enables_deterministic_kernels(self):
        self.graph["deterministicMode"]["strict"] = True
        with (
            patch("torch.manual_seed"),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.use_deterministic_algorithms") as deterministic_algorithms,
        ):
            applied = WebServer._apply_deterministic_mode(self.server, self.graph)

        deterministic_algorithms.assert_called_once_with(True, warn_only=False)
        self.assertTrue(applied["strict"])
        self.assertTrue(applied["settings"]["torch_deterministic_algorithms"])

    def test_strict_mode_fails_closed_when_torch_cannot_enforce_it(self):
        self.graph["deterministicMode"]["strict"] = True
        with (
            patch("torch.manual_seed"),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.use_deterministic_algorithms", side_effect=RuntimeError("unsupported kernel")),
        ):
            with self.assertRaisesRegex(RuntimeError, "Strict deterministic Torch settings"):
                WebServer._apply_deterministic_mode(self.server, self.graph)

    def test_strict_mode_requires_a_fixed_seed(self):
        self.graph["deterministicMode"] = {"enabled": True, "strict": True}

        with self.assertRaisesRegex(ValueError, "requires a fixed seed"):
            WebServer._apply_deterministic_mode(self.server, self.graph)

    def test_process_wide_execution_state_is_restored_after_a_run(self):
        restored = {}
        cuda = SimpleNamespace(
            is_initialized=lambda: False,
            set_rng_state_all=lambda value: restored.update(cuda_rng=value),
            set_per_process_memory_fraction=lambda fraction, index: restored.update(memory_fraction=(fraction, index)),
        )
        cudnn = SimpleNamespace(benchmark=True, deterministic=False, allow_tf32=True)
        matmul = SimpleNamespace(allow_tf32=True)
        fake_torch = SimpleNamespace(
            get_rng_state=lambda: "torch-before",
            set_rng_state=lambda value: restored.update(torch_rng=value),
            are_deterministic_algorithms_enabled=lambda: False,
            is_deterministic_algorithms_warn_only_enabled=lambda: False,
            use_deterministic_algorithms=lambda enabled, warn_only=False: restored.update(
                deterministic=(enabled, warn_only)
            ),
            cuda=cuda,
            backends=SimpleNamespace(cudnn=cudnn, cuda=SimpleNamespace(matmul=matmul)),
        )

        before_random = random.getstate()
        before_numpy = np.random.get_state()
        prior_hash_seed = os.environ.pop("PYTHONHASHSEED", None)
        try:
            with patch(
                "modiff.server.import_module",
                side_effect=lambda name: fake_torch if name == "torch" else np,
            ):
                state = WebServer._capture_execution_process_state(self.server)
                random.seed(999)
                np.random.seed(999)
                os.environ["PYTHONHASHSEED"] = "999"
                cudnn.benchmark = False
                cudnn.deterministic = True
                cudnn.allow_tf32 = False
                matmul.allow_tf32 = False
                WebServer._restore_execution_process_state(
                    self.server,
                    state,
                    {"runtimeHints": {"device": "cuda:0"}},
                )
        finally:
            if prior_hash_seed is not None:
                os.environ["PYTHONHASHSEED"] = prior_hash_seed
            else:
                os.environ.pop("PYTHONHASHSEED", None)

        self.assertEqual(random.getstate(), before_random)
        self.assertTrue(np.array_equal(np.random.get_state()[1], before_numpy[1]))
        self.assertEqual(restored["torch_rng"], "torch-before")
        self.assertEqual(restored["deterministic"], (False, False))
        self.assertEqual(restored["memory_fraction"], (1.0, 0))
        self.assertTrue(cudnn.benchmark)
        self.assertFalse(cudnn.deterministic)
        self.assertTrue(cudnn.allow_tf32)
        self.assertTrue(matmul.allow_tf32)


if __name__ == "__main__":
    unittest.main()
