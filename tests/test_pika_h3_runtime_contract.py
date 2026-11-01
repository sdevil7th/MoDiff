"""Exact source-provider boundaries; no model downloads or hardware claims."""
import json
from pathlib import Path

import pytest

from modiff import optional_runtimes
from modiff.runtime_source_builds import validate_source_build_contract


PROFILE_ID = "huggingface-diffusers-video-d0b9b1b"
REVISION = "d0b9b1b5a540e5f9df2f8ec236347bf9c5aa7d1f"
ROOT = Path(__file__).resolve().parents[1]


def profile():
    return optional_runtimes.OPTIONAL_RUNTIME_PROFILES[PROFILE_ID]


def test_video_source_profile_is_exact_and_does_not_replace_the_stable_base():
    selected = profile()
    assert selected.provides_diffusers
    assert {package.distribution for package in selected.packages} == {"diffusers"}
    package = selected.packages[0]
    assert package.version == "0.42.0.dev0"
    assert package.publisher == "Hugging Face"
    assert package.project_url == "https://github.com/huggingface/diffusers"
    build = validate_source_build_contract(selected.source_builds[0])
    assert build['sourceArtifact']['commit'] == REVISION
    assert build['sourceArtifact']['sha256'] == '77252509be5b2faba7caa033fca558c055d963232278c1cca9d64f6980e1476a'
    assert build['outputWheel']['sha256'] == 'fb48ff55fcd326858542a99d2f726b5ac69f87de7056332037a4cb1eb72ca705'
    assert build['outputWheel']['byteSize'] == 28324103
    assert build['buildDependencies'] == []
    assert build['sourceTrees'] == ['src/diffusers']
    assert '"diffusers==0.41.0"' in (ROOT / 'pyproject.toml').read_text(encoding='utf-8')
    assert selected.satisfies_profiles == (), 'New source is not qualification of older image providers.'


def test_candidate_preserves_required_base_libraries_and_constrains_apis():
    selected = profile()
    base = {item.distribution: item for item in selected.base_packages}
    assert 'diffusers' not in base
    assert base['transformers'].specifier == '>=5.18.0'
    assert base['peft'].specifier == '>=0.21.2'
    assert base['huggingface-hub'].specifier == '>=1.32.0,<2.0'
    assert {'torch', 'torchvision', 'safetensors', 'tokenizers'}.issubset(base)
    assert {'MiniMaxH3ModularPipeline', 'PikaH3FlexModularPipeline', 'MiniMaxH3Scheduler',
            'MiniMaxH3Transformer3DModel', 'AutoencoderKLMiniMaxH3',
            'AutoencoderKLMiniMaxH3Audio'}.issubset(selected.required_diffusers_symbols)
    assert selected.require_peft_backend
    assert selected.excluded_qualification_symbols == ('add_weighted_adapter',)


@pytest.mark.parametrize('platform,machine', [
    ('linux', 'x86_64'), ('windows', 'x86_64'), ('macos', 'arm64'), ('linux', 'aarch64'),
])
def test_only_the_observed_linux_package_target_is_enabled(platform, machine):
    target = profile().contract_for_target(platform_name=platform, machine=machine)
    qualified = (platform, machine) == ('linux', 'x86_64')
    assert target.contract_state == ('qualified' if qualified else 'candidate_unqualified')
    assert target.cutover_ready is qualified
    assert target.install_action_available is qualified
    assert target.activation_available is qualified


def test_linux_projection_matches_the_exact_qualified_package_spec():
    assert profile().spec_digest == 'sha256:6b49afb6e8a2e69c86e1d93bb5173d69a3864d3ef5834c118afadababe9cdb20'


def test_old_evaluation_and_provider_identities_remain_unchanged():
    document = json.loads((ROOT / 'data/diffusers-evaluation-runtime.v1.json').read_bytes())
    assert set(document['profiles']) == {'flux3-7ecb1e1'}
    assert document['profiles']['flux3-7ecb1e1']['wheelSha256'] == 'c19c4721a952fd373d0cab705f582eb81675cad0fb844b88fb449c40de8c74a3'
    old = optional_runtimes.OPTIONAL_RUNTIME_PROFILES[optional_runtimes.DIFFUSERS_FLUX3_RUNTIME_PROFILE_ID]
    assert old.spec_digest == 'sha256:82017c1615d807c1808c13929f0ee10b180058b2f976e51209a80e41939a40fe'
