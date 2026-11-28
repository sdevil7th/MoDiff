from copy import deepcopy

import pytest

from modiff import model_artifact_catalog as catalog
from modiff.studio_execution_specs import reviewed_repository_download_files, studio_capability_definitions


def test_klein_kv_uses_its_exact_9b_sharded_layout_not_the_4b_selection():
    capability = studio_capability_definitions()['Flux2KleinKVPipeline']
    files = capability['downloadFiles']
    assert 'LICENSE' in files and 'LICENSE.md' not in files
    assert 'transformer/diffusion_pytorch_model.safetensors' not in files
    assert 'transformer/diffusion_pytorch_model.safetensors.index.json' in files
    assert {name for name in files if name.startswith('text_encoder/model-')} == {
        f'text_encoder/model-{index:05d}-of-00004.safetensors' for index in range(1, 5)
    }
    assert {name for name in files if name.startswith('transformer/diffusion_pytorch_model-')} == {
        f'transformer/diffusion_pytorch_model-{index:05d}-of-00002.safetensors' for index in range(1, 3)
    }
    repo = capability['defaultRepo']
    inventory = catalog.catalog_download_inventory(repo, catalog.catalog_revision(repo), files)
    assert inventory and inventory['exactBytes'] > 30_000_000_000


def test_alternate_qwen_download_layouts_remain_distinct_and_revision_bound():
    selections = {repo: {'repo': repo, 'revision': catalog.catalog_revision(repo),
                         'downloadFiles': reviewed_repository_download_files(repo)}
                  for repo in ('Qwen/Qwen-Image', 'unsloth/Qwen-Image-2512-unsloth-bnb-4bit')}
    original = selections['Qwen/Qwen-Image']
    bnb = selections['unsloth/Qwen-Image-2512-unsloth-bnb-4bit']
    assert 'tokenizer/chat_template.jinja' not in original['downloadFiles']
    assert 'LICENSE' in original['downloadFiles']
    assert 'text_encoder/model-00001-of-00002.safetensors' in bnb['downloadFiles']
    assert 'transformer/diffusion_pytorch_model-00003-of-00003.safetensors' in bnb['downloadFiles']
    for selection in (original, bnb):
        assert catalog.catalog_download_inventory(selection['repo'], selection['revision'], selection['downloadFiles'])
    # Single root checkpoints are not falsely offered as full Diffusers layouts.
    for repo in ('black-forest-labs/FLUX.1-dev-FP8', 'black-forest-labs/FLUX.1-Kontext-dev-NVFP4'):
        assert reviewed_repository_download_files(repo) == []


def test_qwen_image_21_turbo_uses_its_exact_reviewed_download_selection():
    model_type = 'QwenImage21TurboPipeline'
    repo = 'Qwen/Qwen-Image-2.1-Turbo'
    revision = 'fc13ba1986fcb905854dc8de2f52eee395f49b5b'
    capability = studio_capability_definitions()[model_type]
    files = capability['downloadFiles']
    assert capability['defaultRepo'] == repo
    assert catalog.catalog_repository_pin(repo, model_type=model_type) == {
        'repo': repo,
        'revision': revision,
        'license': 'qwen-research',
        'kind': 'base',
        'modelType': model_type,
    }
    inventory = catalog.catalog_download_inventory(repo, revision, files)
    assert inventory['exactBytes'] == 32_451_624_677
    assert len(files) == 18
    entries = {entry['path']: entry for entry in inventory['files']}
    assert entries['LICENSE']['gitBlobId'] == '13ae08d5a5828f508cf0e852b1b316c0c0bbb9b3'
    assert {path: entry['byteSize'] for path, entry in entries.items() if path.endswith('.safetensors')} == {
        'text_encoder/model.safetensors': 17_534_340_584,
        'transformer/diffusion_pytorch_model-00001-of-00002.safetensors': 9_968_332_504,
        'transformer/diffusion_pytorch_model-00002-of-00002.safetensors': 4_261_951_904,
        'vae/diffusion_pytorch_model.safetensors': 675_508_656,
    }
    assert entries['vae/diffusion_pytorch_model.safetensors']['sha256'] == (
        '71879ffd5321e6d10c3c87513e2b474b1252efa7f3dec2969214a9bf06a6dd5c'
    )
    base_revision = catalog.catalog_revision('Qwen/Qwen-Image-2.1')
    assert catalog.catalog_download_inventory(repo, base_revision, files) is None
    assert catalog.catalog_download_inventory(repo, revision, files + ['assets/architecture.jpg']) is None


def test_download_byte_inventory_invalidates_on_changed_selection_or_revision(monkeypatch):
    row = deepcopy(catalog.read_model_artifact_catalog()['downloadInventories'][0])
    files = [item['path'] for item in row['files']]
    monkeypatch.setattr(catalog, 'read_model_artifact_catalog', lambda: {'downloadInventories': [row]})
    assert catalog.catalog_download_inventory(row['repo'], row['revision'], files)['exactBytes'] > 0
    assert catalog.catalog_download_inventory(row['repo'], '0' * 40, files) is None
    assert catalog.catalog_download_inventory(row['repo'], row['revision'], files + ['extra.bin']) is None
    row['files'][0]['byteSize'] = True
    with pytest.raises(ValueError, match='Invalid download file'):
        catalog.catalog_download_inventory(row['repo'], row['revision'], files)
    row['files'][0]['byteSize'] = 1
    row['files'].append(deepcopy(row['files'][0]))
    with pytest.raises(ValueError, match='Invalid download file'):
        catalog.catalog_download_inventory(row['repo'], row['revision'], files)
