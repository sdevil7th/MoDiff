"""Keep test registry discovery away from the operator's installed extensions."""

from tempfile import TemporaryDirectory

import pytest


def pytest_configure(config):
    # This runs before collection: several test modules import the node registry
    # at module scope, before any autouse fixture could isolate discovery.
    from modiff.custom_extensions import ExtensionStore

    directory = TemporaryDirectory(prefix="modiff-test-extensions-")
    original_init = ExtensionStore.__init__

    def isolated_init(self, root=None):
        original_init(self, root if root is not None else directory.name)

    patch = pytest.MonkeyPatch()
    patch.setattr(ExtensionStore, "__init__", isolated_init)
    config.add_cleanup(directory.cleanup)
    config.add_cleanup(patch.undo)
