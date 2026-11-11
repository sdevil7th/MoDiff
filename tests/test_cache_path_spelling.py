"""Cache inspection preserves the filesystem spelling of model directories."""

from modiff.auto_resource import _cache_dirs_for_record, artifact_cache_status
from utils.huggingface import _dedupe_locations


def test_cache_snapshot_checks_preserve_mixed_case_paths(tmp_path):
    cache = tmp_path / 'MixedCase Models' / 'Hub'
    revision = 'a' * 40
    repo = 'owner/model'
    snapshot = cache / 'models--owner--model' / 'snapshots' / revision
    snapshot.mkdir(parents=True)
    (snapshot / 'model.safetensors').write_bytes(b'path-resolution-fixture')
    record = {'id': repo, 'cache_dir': str(cache), 'cache_dirs': [str(cache)],
              'revisions': [{'hash': revision}]}
    assert _cache_dirs_for_record(record) == [str(cache)]
    assert _dedupe_locations([('configured', str(cache)), ('duplicate', str(cache))]) == [('configured', str(cache))]
    status = artifact_cache_status(repo, [record])
    assert status['complete']
    assert status['snapshots'][0]['path'] == str(snapshot)
