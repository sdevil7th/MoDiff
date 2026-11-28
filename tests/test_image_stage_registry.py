"""Actual facade through maintained AST registry, no node construction."""

import modules
from modules.DiffusersImage import main

from image_stage_test_support import stage_offline_runtime as stage_offline_runtime
import pytest

pytestmark = pytest.mark.usefixtures("stage_offline_runtime")


def test_actual_registry_preserves_live_stage_params():
    for action in ("EncodeInputs", "Denoise"):
        published = modules.MODULE_MAP["modules.DiffusersImage"][action]["params"]
        assert set(published) == set(getattr(main, action).params)
        assert published == getattr(main, action).params
