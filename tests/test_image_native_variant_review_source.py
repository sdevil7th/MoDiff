"""Re-reviewed source identity preserves every legacy variant decision."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from modiff import image_native_variant_reviews as variants
from modiff.modular_workflow_contracts import PINNED_DIFFUSERS_REVISION


BASELINE = json.loads((Path(__file__).parent / 'fixtures/image-native-variant-reviews.v1.json').read_text())
REVIEWED = '7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941'


@pytest.mark.parametrize('route', sorted(BASELINE['reviews']))
def test_all_seventeen_decisions_change_source_identity_only(route):
    assert PINNED_DIFFUSERS_REVISION == REVIEWED
    expected = deepcopy(BASELINE['reviews'][route])
    expected['diffusersRevision'] = REVIEWED
    actual = variants.image_native_variant_review(route)
    assert actual == expected
    assert actual['liveQualification'] == 'not_established_by_review'


def test_unknown_source_pin_is_not_implicitly_reviewed(monkeypatch):
    monkeypatch.setattr(variants, 'PINNED_DIFFUSERS_REVISION', '0' * 40)
    with pytest.raises(ValueError, match='re-reviewed'):
        variants.image_native_variant_review(next(iter(BASELINE['reviews'])))


def test_blocked_roots_and_kv_exceptions_do_not_inherit_new_flux3_quantization():
    reviews = {route: variants.image_native_variant_review(route) for route in BASELINE['reviews']}
    blocked = [value for value in reviews.values() if value['decision'] == 'artifact_blocked']
    exceptions = [value for value in reviews.values() if value['decision'] == 'upstream_exception']
    assert len(blocked) == len(exceptions) == 2
    assert {value['reason'] for value in blocked} == {'root_checkpoint_has_no_reviewed_modular_component_layout'}
    assert {value['reason'] for value in exceptions} == {'pinned_modular_klein_has_no_kv_extract_cached_path'}
    assert all('nativeProfileId' not in value for value in blocked + exceptions)


def test_variant_reviews_remain_independent_copies_and_unknown_routes_absent():
    route = next(iter(BASELINE['reviews']))
    value = variants.image_native_variant_review(route)
    value['limitations'].append('caller mutation')
    assert variants.image_native_variant_review(route)['limitations'] == BASELINE['reviews'][route]['limitations']
    assert variants.image_native_variant_review('unknown:profile/unknown_task') is None
