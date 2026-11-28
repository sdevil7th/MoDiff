"""Select provider-bound source audits without confusing SDK symbol presence.

This early skip does not grant readiness. On the selected version, the normal
executor still checks the exact provider source, receipt and runtime binding.
"""

from importlib import metadata

import pytest


def require_diffusers_provider_version(profile_id):
    from modiff.optional_runtimes import OPTIONAL_RUNTIME_PROFILES

    profile = OPTIONAL_RUNTIME_PROFILES[profile_id]
    (package,) = [item for item in profile.packages if item.distribution == "diffusers"]
    actual = metadata.version("diffusers")
    if actual != package.version:
        pytest.skip(
            f"Provider-bound source audit requires {profile_id} / Diffusers {package.version}; "
            f"installed stable {actual} is a separate runtime binding."
        )
