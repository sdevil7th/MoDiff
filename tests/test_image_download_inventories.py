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
