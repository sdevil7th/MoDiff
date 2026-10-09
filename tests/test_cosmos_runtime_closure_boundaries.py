"""Bidirectional cv2 exclusions and immutable publisher caption provenance."""
import hashlib
from importlib import metadata
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from modiff import optimization_packages as packages
from modiff import optional_runtimes as runtimes
from modiff.cosmos_safety_contract import COSMOS_SAFETY_RUNTIME_PROFILE_ID
from modiff.optional_runtimes import (
    GALLERY_MEDIA_RUNTIME_PROFILE_ID, OPTIONAL_RUNTIME_PROFILES,
    project_optional_runtime_qualification,
)


@pytest.mark.parametrize("selected,conflict", [
    (GALLERY_MEDIA_RUNTIME_PROFILE_ID, "opencv-python"),
    (GALLERY_MEDIA_RUNTIME_PROFILE_ID, "opencv-contrib-python"),
    (GALLERY_MEDIA_RUNTIME_PROFILE_ID, "opencv-contrib-python-headless"),
    (COSMOS_SAFETY_RUNTIME_PROFILE_ID, "opencv-python-headless"),
])
@pytest.mark.parametrize("action", ["install", "activate"])
def test_opposite_cv2_provider_rejected_before_lease_or_activation(selected, conflict, action):
    profile = project_optional_runtime_qualification(
        OPTIONAL_RUNTIME_PROFILES[selected], platform_name="linux", machine="x86_64",
    )

    def installed(name):
        if name == conflict:
            return "5.0.0.93"
        raise metadata.PackageNotFoundError(name)

    with patch("modiff.optional_runtimes.optional_runtime_target", return_value=("linux", "x86_64")), \
         patch.object(packages, "OPTIONAL_RUNTIME_PROFILES", {selected: profile}), \
         patch.object(runtimes, "OPTIONAL_RUNTIME_PROFILES", {selected: profile}), \
         patch("modiff.optional_runtimes.metadata.version", side_effect=installed), \
         patch.object(packages, "current_base_binding", side_effect=AssertionError("No base mutation")), \
         patch.object(packages, "reserve_install", side_effect=AssertionError("No lease")) as lease, \
         patch.object(packages, "_activate_environment_transaction", side_effect=AssertionError("No activation")) as activation:
        with pytest.raises(RuntimeError, match="single provider"):
            if action == "install":
                packages.install_optional_runtime(selected, profile.spec_digest, consent=True)
            else:
                packages.activate_optional_runtime_environment("unexamined", selected, profile.spec_digest, consent=True)
        lease.assert_not_called()
        activation.assert_not_called()


def test_publisher_caption_descriptor_binds_unmodified_original_and_notice():
    root = Path(__file__).resolve().parents[1]
    descriptor = json.loads((root / "data/cosmos3-super-t2i-publisher-caption-provenance.v1.json").read_text())
    assert descriptor == {
        "schemaVersion": 1,
        "publisher": "NVIDIA",
        "repository": "nvidia/Cosmos3-Super-Text2Image",
        "revision": "daf3d374804be4c512c2135568a7cb95d4341d79",
        "sourcePath": "assets/example_caption.json",
        "sourceUrl": "https://huggingface.co/nvidia/Cosmos3-Super-Text2Image/resolve/daf3d374804be4c512c2135568a7cb95d4341d79/assets/example_caption.json",
        "sourceGitBlob": "01db693d45c8d2f17e0d4190978bfc857a0d104e",
        "localPath": "data/cosmos3-super-t2i-publisher-caption.v1.json",
        "localSha256": "c068a8d430c87bc752c775567113463c8fa3c9370ac7047318852bdc124bd5e3",
        "byteSize": 7382,
        "publisherDeclaredLicense": "OpenMDW-1.1",
        "publisherLicenseMetadataName": "openmdw1.1-license",
        "publisherLicenseUrl": "https://openmdw.ai/license/1-1/",
        "licenseMetadataSourceUrl": "https://huggingface.co/nvidia/Cosmos3-Super-Text2Image/blob/daf3d374804be4c512c2135568a7cb95d4341d79/README.md",
        "licenseFile": "licenses/OpenMDW-1.1.txt",
        "licenseFileSha256": "be21b29b1c69af5b4a4e60a6bd7bc933eec374e52be29d1ebb9e6a17a6588491",
        "licenseFileSourceUrl": "https://openmdw.ai/license/1-1/",
        "modified": False,
    }
    body = (root / descriptor["localPath"]).read_bytes()
    assert len(body) == descriptor["byteSize"]
    assert hashlib.sha256(body).hexdigest() == descriptor["localSha256"]
    assert hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest() == descriptor["sourceGitBlob"]
    notice = (root / "THIRD_PARTY_NOTICES.md").read_text()
    for value in (descriptor["localPath"], descriptor["revision"], descriptor["publisherDeclaredLicense"], descriptor["sourceUrl"]):
        assert value in notice
    agreement = (root / descriptor["licenseFile"]).read_bytes()
    assert hashlib.sha256(agreement).hexdigest() == descriptor["licenseFileSha256"]
    assert agreement.startswith(b"OpenMDW License Agreement, version 1.1 (OpenMDW-1.1)")
    assert b"IN NO EVENT SHALL THE PROVIDERS" in agreement


def test_cosmos_profile_preserves_existing_weighted_adapter_exclusion():
    spec = OPTIONAL_RUNTIME_PROFILES[COSMOS_SAFETY_RUNTIME_PROFILE_ID].to_spec_dict()
    assert spec["excludedQualificationSymbols"] == ["add_weighted_adapter"]
