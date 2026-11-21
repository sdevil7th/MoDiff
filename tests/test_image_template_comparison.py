from copy import deepcopy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin
import pytest

from modiff.image_template_comparison import CONTRACT_FORMAT, compare_image_template_outputs


@pytest.fixture
def contract():
    return {"format": CONTRACT_FORMAT, "templateId": "example", "task": "text_to_image",
            "models": [{"role": "base", "repoId": "test/model", "revision": "a" * 40}],
            "settings": {"seed": 42, "prompt": "original prompt", "steps": 4, "dtype": "bfloat16"},
            "runtime": {"torch": "reviewed", "diffusersCommit": "b" * 40, "transformers": "reviewed",
                        "peft": "reviewed", "platform": "linux", "device": "cuda:0"},
            "inputs": [], "expectedOutputs": [{"width": 8, "height": 8}]}


@pytest.fixture
def image():
    return Image.fromarray(np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3))


def compare(image, contract, **kwargs):
    return compare_image_template_outputs([image], [image.copy()], baseline_contract=contract,
                                          candidate_contract=kwargs.pop("candidate_contract", contract), **kwargs)


def test_exact_decoded_match_is_not_aesthetic_or_resource_approval(image, contract):
    result = compare(image, contract)
    assert result["passed"] and result["pixelIdentity"]
    assert result["outputs"][0]["maxPixelError"] == 0
    assert result["outputs"][0]["psnrDb"] is None
    assert result["aestheticApproval"] == "human_review_required"
    assert result["qualifiesAuto"] is result["publishesGalleryAssets"] is False


@pytest.mark.parametrize("field,value", [("seed", 43), ("prompt", "different"), ("dtype", "float32")])
def test_identical_pixels_cannot_hide_changed_generation_settings(image, contract, field, value):
    changed = deepcopy(contract)
    changed["settings"][field] = value
    result = compare(image, contract, candidate_contract=changed)
    assert not result["passed"]
    assert result["contractMismatchPaths"] == [f"settings.{field}"]


def test_revision_runtime_and_ordered_input_changes_are_not_comparable(image, contract):
    contract["inputs"] = [{"role": "reference", "sha256": "c" * 64}, {"role": "reference", "sha256": "d" * 64}]
    changed = deepcopy(contract)
    changed["models"][0]["revision"] = "e" * 40
    changed["runtime"]["torch"] = "another"
    changed["inputs"].reverse()
    result = compare(image, contract, candidate_contract=changed)
    assert not result["passed"]
    assert "models[0].revision" in result["contractMismatchPaths"]
    assert "runtime.torch" in result["contractMismatchPaths"]
    assert "inputs[0].sha256" in result["contractMismatchPaths"]


def test_output_collection_and_dimensions_must_match(image, contract):
    for outputs in ([], [image.resize((9, 8))], [image, image]):
        result = compare_image_template_outputs([image], outputs, baseline_contract=contract, candidate_contract=contract)
        assert not result["passed"]


def test_small_numeric_drift_requires_an_explicit_tolerance(image, contract):
    changed = np.asarray(image).copy()
    changed[0, 0, 0] += 1
    candidate = Image.fromarray(changed)
    strict = compare_image_template_outputs([image], [candidate], baseline_contract=contract, candidate_contract=contract)
    assert not strict["passed"] and strict["outputs"][0]["maxPixelError"] == 1
    allowed = compare_image_template_outputs([image], [candidate], baseline_contract=contract,
                                             candidate_contract=contract, max_pixel_error=1)
    assert allowed["passed"] and not allowed["pixelIdentity"]


def test_alpha_changes_are_not_lost_by_rgb_conversion(image, contract):
    left, right = image.convert("RGBA"), image.convert("RGBA")
    right.putalpha(0)
    result = compare_image_template_outputs([left], [right], baseline_contract=contract, candidate_contract=contract)
    assert not result["passed"]
    assert result["outputs"][0]["maxPixelError"] == 255


@pytest.mark.parametrize("change", ["random_seed", "mutable_revision", "nan", "missing_runtime", "missing_inputs"])
def test_incomplete_or_unbound_contracts_are_rejected(image, contract, change):
    if change == "random_seed":
        contract["settings"]["seed"] = -1
    elif change == "mutable_revision":
        contract["models"][0]["revision"] = "main"
    elif change == "nan":
        contract["settings"]["guidance"] = float("nan")
    elif change == "missing_runtime":
        del contract["runtime"]
    else:
        del contract["inputs"]
    with pytest.raises(ValueError):
        compare(image, contract)


def test_cli_compares_pixels_despite_different_png_metadata_and_records_raw_hashes(tmp_path, image, contract):
    old, new = tmp_path / "old.png", tmp_path / "new.png"
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("producer", "old nodes")
    image.save(old, pnginfo=metadata)
    image.save(new)
    recipe = tmp_path / "contract.json"
    recipe.write_text(json.dumps(contract), encoding="utf-8")
    report = tmp_path / "report.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "compare_image_template_outputs.py"
    args = [sys.executable, str(script), "--baseline", str(old), "--candidate", str(new),
            "--baseline-contract", str(recipe), "--candidate-contract", str(recipe)]
    process = subprocess.run([*args, "--report", str(report)], text=True, capture_output=True, check=False)
    assert process.returncode == 0, process.stderr
    result = json.loads(report.read_text())
    assert result["pixelIdentity"]
    assert result["files"]["baseline"][0]["sha256"] != result["files"]["candidate"][0]["sha256"]
    blocked = subprocess.run([*args, "--report", str(old)], text=True, capture_output=True, check=False)
    assert blocked.returncode != 0
    assert "must not overwrite" in blocked.stderr


@pytest.fixture
def wheel_contract(contract):
    del contract["runtime"]["diffusersCommit"]
    contract["runtime"].update({
        "diffusersVersion": "0.41.0",
        "diffusersWheelSha256": "sha256:ea8918b7dfd92ce793b6db689a551b8cd25d52c6c0fcd3ade0706a5fd2a25990",
    })
    return contract


def test_published_wheel_contract_preserves_actual_identity_without_a_git_receipt(image, wheel_contract):
    result = compare(image, wheel_contract)
    assert result["passed"] and result["pixelIdentity"]
    assert "diffusersCommit" not in wheel_contract["runtime"]
    assert result["qualifiesAuto"] is False


@pytest.mark.parametrize("field,value", [("diffusersVersion", "0.41.1"), ("diffusersWheelSha256", "d" * 64)])
def test_published_wheel_identity_changes_are_not_comparable(image, wheel_contract, field, value):
    changed = deepcopy(wheel_contract)
    changed["runtime"][field] = value
    result = compare(image, wheel_contract, candidate_contract=changed)
    assert not result["passed"]
    assert result["contractMismatchPaths"] == ["runtime." + field]


@pytest.mark.parametrize("change", ["version_missing", "hash_missing", "invalid_version", "invalid_hash", "both_sources"])
def test_published_wheel_contract_rejects_incomplete_or_ambiguous_identity(image, wheel_contract, change):
    runtime = wheel_contract["runtime"]
    if change == "version_missing":
        del runtime["diffusersVersion"]
    elif change == "hash_missing":
        del runtime["diffusersWheelSha256"]
    elif change == "invalid_version":
        runtime["diffusersVersion"] = "latest"
    elif change == "invalid_hash":
        runtime["diffusersWheelSha256"] = "not-a-wheel-digest"
    else:
        runtime["diffusersCommit"] = "b" * 40
    with pytest.raises(ValueError):
        compare(image, wheel_contract)


def test_cli_accepts_exact_published_wheel_recipe_and_rejects_changed_artifact(tmp_path, image, wheel_contract):
    old = tmp_path / "old.png"
    new = tmp_path / "new.png"
    image.save(old)
    image.save(new)
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(json.dumps(wheel_contract))
    after.write_text(json.dumps(wheel_contract))
    script = Path(__file__).resolve().parents[1] / "scripts/compare_image_template_outputs.py"
    args = [sys.executable, str(script), "--baseline", str(old), "--candidate", str(new),
            "--baseline-contract", str(before), "--candidate-contract", str(after)]
    report = tmp_path / "same-wheel.json"
    accepted = subprocess.run([*args, "--report", str(report)], capture_output=True, text=True)
    assert accepted.returncode == 0, accepted.stderr
    assert json.loads(report.read_text())["pixelIdentity"]
    wheel_contract["runtime"]["diffusersWheelSha256"] = "e" * 64
    after.write_text(json.dumps(wheel_contract))
    mismatch = tmp_path / "changed-wheel.json"
    rejected = subprocess.run([*args, "--report", str(mismatch)], capture_output=True, text=True)
    assert rejected.returncode == 1
    assert json.loads(mismatch.read_text())["contractMismatchPaths"] == ["runtime.diffusersWheelSha256"]


def test_legacy_git_contract_retains_optional_actual_version_metadata(image, contract):
    contract["runtime"]["diffusersVersion"] = "0.41.0.dev0"
    assert compare(image, contract)["passed"]
