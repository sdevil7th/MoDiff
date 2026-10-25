"""Synthetic tokenizer/official-API boundaries; no model weights or fit proof."""
from copy import deepcopy
from dataclasses import asdict
import json
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest

from modiff import prompt_enhancement_constraints as adapter
from modiff.structured_prompt_constraints import ByteVocabulary, ConstraintError, ConstraintLimits


LAYOUT = {'schemaVersion': 1, 'coordinateSpace': 'normalized_xyxy', 'regions': [
    {'id': 'fox', 'description': 'fox', 'targetBox': [.2, .2, .8, .8]}]}
T2I = 'A red fox <fox>.\n[{"id":"fox","bbox":[200,200,800,800],"desc":"red fox"}]'
EDIT = ('Move <fox>.\n[{"id":"fox","from":"ref_image_0","src_bbox":[100,100,400,400],'
        '"tgt_bbox":[200,200,800,800],"desc":"red fox"}]')


@pytest.fixture
def synthetic(monkeypatch):
    """Deliberately synthetic 256-byte vocabulary, not an actual tokenizer test."""
    contract = adapter.constraint_contract()
    special = contract['reservedAddedTokenIds']
    added = [{'id': identity, 'content': 'eos' if identity == 248046 else f'control{identity}',
        'single_word': False, 'lstrip': False, 'rstrip': False, 'normalized': False,
        'special': identity in contract['trueSpecialTokenIds']} for identity in special]
    document = {'model': {'type': 'BPE', 'byte_fallback': False,
        'vocab': {chr(i): i for i in range(256)}, 'merges': []},
        'decoder': {'type': 'ByteLevel'}, 'normalizer': None,
        'pre_tokenizer': {'type': 'Split', 'pattern': {'Regex': r'\p{L}+'}},
        'post_processor': None, 'added_tokens': added}
    effective_document = deepcopy(document)
    effective_document['pre_tokenizer']['pattern']['Regex'] = r'[\p{L}\p{M}]+'
    tokenizers = ModuleType('tokenizers')
    class Tokenizer:
        def __init__(self):
            self.document = deepcopy(effective_document)
            self.decoder = SimpleNamespace(decode=lambda spellings:
                bytes(ord(c) for s in spellings for c in s).decode('utf-8', errors='replace'))
        def to_str(self):
            return json.dumps(self.document)
        def get_vocab(self, *, with_added_tokens):
            assert with_added_tokens is False
            return dict(self.document['model']['vocab'])
        def get_added_tokens_decoder(self):
            return {row['id']: SimpleNamespace(**row) for row in self.document['added_tokens']}
    tokenizers.Tokenizer, tokenizers.__version__ = Tokenizer, contract['tokenizersVersion']
    transformers = ModuleType('transformers')
    class Qwen3_5Tokenizer:
        eos_token_id = 248046
        def __init__(self):
            self.backend_tokenizer = Tokenizer()
        def decode(self, ids, **kwargs):
            assert kwargs == {'skip_special_tokens': False, 'clean_up_tokenization_spaces': False}
            return bytes(ids).decode('utf-8', errors='strict')
    transformers.Qwen3_5Tokenizer, transformers.__version__ = Qwen3_5Tokenizer, contract['transformersVersion']
    convert = ModuleType('transformers.convert_slow_tokenizer')
    convert.bytes_to_unicode = lambda: {i: chr(i) for i in range(256)}
    for name, module in [('transformers', transformers), ('tokenizers', tokenizers),
                         ('transformers.convert_slow_tokenizer', convert)]:
        monkeypatch.setitem(sys.modules, name, module)
    contract['ordinaryVocabularyTokens'] = 256
    vocabulary = ByteVocabulary({i: bytes([i]) for i in range(256)}, special_token_ids=special,
        eos_token_ids=[248046], limits=ConstraintLimits(max_new_tokens=3584, max_request_seconds=1800))
    contract['ordinaryVocabularySha256'] = 'sha256:' + vocabulary.fingerprint
    contract['tokenizerSemanticSha256'] = adapter._digest(adapter._canonical(effective_document))
    contract['sourcePreTokenizerSha256'] = adapter._digest(adapter._canonical(document['pre_tokenizer']))
    contract['effectivePreTokenizerSha256'] = adapter._digest(adapter._canonical(effective_document['pre_tokenizer']))
    monkeypatch.setattr(adapter, 'constraint_contract', lambda: deepcopy(contract))
    documents = {'tokenizer.json': document, 'tokenizer_config.json': {'eos_token': 'eos'},
        'generation_config.json': {'eos_token_id': [248046, 248044], 'pad_token_id': 248044}}
    monkeypatch.setattr(adapter, '_read_artifacts', lambda *_args: deepcopy(documents))
    processor = SimpleNamespace(tokenizer=Qwen3_5Tokenizer())
    observer = SimpleNamespace(check=lambda: None)
    return SimpleNamespace(contract=contract, document=document, documents=documents,
        processor=processor, observer=observer, tokenizer_class=Qwen3_5Tokenizer,
        transformers=transformers, tokenizers=tokenizers)


def make(fixture, *, layout=None, mode='t2i_bbox', count=0, limit=3584, deadline=None):
    return adapter.PreparedConstraint(fixture.processor, layout or LAYOUT, mode, count,
        [9000, 9001], 'sha256:' + '1' * 64, limit,
        deadline or time.monotonic() + 30, fixture.observer)


def emit(prepared, raw):
    tokens = list(raw.encode('utf-8'))
    # The exact same prompt prefix and every model-emitted byte are checked.
    for index, identity in enumerate(tokens):
        assert identity in prepared.callback(0, [9000, 9001, *tokens[:index]])
    assert prepared.callback(0, [9000, 9001, *tokens]) == [248046]
    return [*tokens, 248046]


def test_source_descriptor_is_separate_bounded_and_contract_only():
    contract = adapter.constraint_contract()
    assert contract['tokenizerClass'] == 'Qwen3_5Tokenizer'
    assert contract['sourcePreTokenizerSha256'] == (
        'sha256:d730489d91319a99960af82e8fed44be8533ca92615198807b41c29dda5556fc')
    assert contract['effectivePreTokenizerSha256'] == (
        'sha256:4cba7e5c03fddf4af78a44f283740fd19687522c60cc36ee271b655f634a7af3')
    assert contract['tokenizerSemanticSha256'] == (
        'sha256:40ebf1d8462c42c3b781b870aaa28884b24f86234bc0b49d54d2bbcf5ddc404a')
    assert contract['status'] == 'contract_only'
    assert contract['configuredEosTokenIds'] == [248046, 248044]
    assert contract['effectiveEosTokenIds'] == [248046]
    assert len(contract['reservedAddedTokenIds']) == 33 and len(contract['trueSpecialTokenIds']) == 21
    assert contract['preparationRamBytes'] == 256 * 1024**2 and contract['deadlineSeconds'] == 1800
    assert contract['limits'] == asdict(ConstraintLimits(max_new_tokens=3584, max_request_seconds=1800))
    assert all(type(value) is int for value in contract['limits'].values())
    assert not any('path' in key.lower() for key in contract if key != 'artifacts')


@pytest.mark.parametrize('editing', [False, True])
def test_literal_model_emitted_output_uses_full_decode_and_unchanged_strict_parser(synthetic, editing):
    from modiff.prompt_enhancement_output import validate_target_layout_output
    layout = deepcopy(LAYOUT)
    if editing:
        layout['regions'][0]['source'] = {'referenceIndex': 0, 'box': [.1, .1, .4, .4]}
    mode, count, raw = ('i2i_bbox', 1, EDIT) if editing else ('t2i_bbox', 0, T2I)
    before = deepcopy(layout)
    prepared = make(synthetic, layout=layout, mode=mode, count=count)
    ids = emit(prepared, raw)
    text, completion = prepared.finish(ids)
    assert text == raw and completion['terminalTokenId'] == 248046
    expected = deepcopy(layout)
    expected['regions'][0]['description'] = 'red fox'
    assert validate_target_layout_output(text, layout, mode=mode, image_count=count)['layoutConditioning'] == expected
    adapter.validate_consumed(prepared.metadata, layout=layout, mode=mode, image_count=count,
        controls={'min_new_tokens': 0, 'max_new_tokens': 3584}, input_hash='sha256:' + '1' * 64)
    adapter.validate_completion(completion, prepared.metadata, text=text,
        generated_tokens=len(ids), input_tokens=2)
    assert layout == before
    held = prepared.callback
    prepared.close()
    assert held.vocabulary is None and not held._cache and prepared.tokenizer is None
    with pytest.raises(ConstraintError, match='refused'):
        held(0, [9000, 9001])


def test_exact_declared_source_to_effective_pre_tokenizer_transition_is_allowed(synthetic):
    native = synthetic.processor.tokenizer.backend_tokenizer
    assert synthetic.document['pre_tokenizer'] != native.document['pre_tokenizer']
    prepared = make(synthetic)
    assert prepared.metadata['tokenizerSemanticSha256'] == synthetic.contract['tokenizerSemanticSha256']
    prepared.close()


@pytest.mark.parametrize('field', ['sourcePreTokenizerSha256', 'effectivePreTokenizerSha256'])
@pytest.mark.parametrize('fault', ['missing', 'invalid'])
def test_source_descriptor_requires_both_exact_pre_tokenizer_digest_fields(tmp_path, monkeypatch, field, fault):
    document = json.loads(adapter.DESCRIPTOR.read_text(encoding='utf-8'))
    if fault == 'missing':
        document.pop(field)
    else:
        document[field] = 'unreviewed'
    descriptor = tmp_path / 'constraints.json'
    descriptor.write_text(json.dumps(document), encoding='utf-8')
    monkeypatch.setattr(adapter, 'DESCRIPTOR', descriptor)
    with pytest.raises(ValueError, match='exact reviewed scope'):
        adapter.constraint_contract()


@pytest.mark.parametrize('change', ['class', 'version', 'decoder', 'normalizer', 'post_processor',
    'vocab', 'added', 'eos', 'semantic', 'source_pre_tokenizer', 'effective_pre_tokenizer',
    'source_pre_pin', 'effective_pre_pin'])
def test_actual_tokenizer_semantic_or_api_drift_refuses(synthetic, change):
    native = synthetic.processor.tokenizer.backend_tokenizer
    if change == 'class':
        synthetic.processor.tokenizer = object()
    elif change == 'version':
        synthetic.tokenizers.__version__ = '0.0.0'
    elif change == 'decoder':
        native.document['decoder']['type'] = 'unreviewed'
    elif change in ('normalizer', 'post_processor'):
        native.document[change] = {'type': 'unreviewed'}
    elif change == 'source_pre_tokenizer':
        synthetic.documents['tokenizer.json']['pre_tokenizer']['pattern']['Regex'] = 'unreviewed'
    elif change == 'effective_pre_tokenizer':
        native.document['pre_tokenizer']['pattern']['Regex'] = 'unreviewed'
    elif change in ('source_pre_pin', 'effective_pre_pin'):
        field = 'sourcePreTokenizerSha256' if change == 'source_pre_pin' else 'effectivePreTokenizerSha256'
        synthetic.contract[field] = 'sha256:' + '0' * 64
    elif change == 'vocab':
        native.document['model']['vocab']['x'] = 900
    elif change == 'added':
        native.document['added_tokens'][0]['special'] = False
    elif change == 'eos':
        synthetic.processor.tokenizer.eos_token_id = 248044
    else:
        synthetic.contract['tokenizerSemanticSha256'] = 'sha256:' + '0' * 64
    with pytest.raises(ValueError):
        make(synthetic)


@pytest.mark.parametrize('fault', ['missing-eos', 'padding', 'prefix', 'geometry', 'decode', 'cancel', 'deadline'])
def test_continuation_errors_refuse_without_repair(synthetic, fault):
    prepared = make(synthetic)
    if fault == 'prefix':
        with pytest.raises(ConstraintError, match='prefix'):
            prepared.callback(0, [8000, 9001])
        return
    if fault == 'geometry':
        raw = T2I.replace('[200,200,800,800]', '[0,0,800,800]')
        with pytest.raises(ConstraintError):
            prepared.finish([*raw.encode(), 248046])
        return
    ids = [*T2I.encode(), 248046]
    if fault == 'missing-eos':
        ids.pop()
    elif fault == 'padding':
        ids[-1] = 248044
    elif fault == 'decode':
        synthetic.processor.tokenizer.decode = lambda *_args, **_kwargs: 'changed text'
    elif fault == 'cancel':
        synthetic.observer.check = lambda: (_ for _ in ()).throw(InterruptedError('owned stop'))
    else:
        prepared.callback._deadline = time.monotonic() - 1
    with pytest.raises((ValueError, InterruptedError)):
        prepared.finish(ids)


@pytest.mark.parametrize('fault', ['dtype', 'device', 'quantization', 'native-eos', 'minimum', 'callback', 'auto-control'])
def test_controls_and_precision_are_explicit_without_substitution(synthetic, fault):
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[248046, 248044], pad_token_id=248044))
    receipt = {'runtime': {'dtype': 'float32', 'device': 'cuda:0', 'quantization': {'mode': 'none'}}}
    controls = {'min_new_tokens': 0, 'do_sample': False, 'num_beams': 1, 'repetition_penalty': 1.0}
    kwargs = {}
    adapter.validate_generation(model, receipt, controls, kwargs)
    if fault in ('dtype', 'device', 'quantization'):
        receipt['runtime'][fault] = {'dtype': 'bfloat16', 'device': 'cpu', 'quantization': {'mode': 'bnb_4bit'}}[fault]
    elif fault == 'native-eos':
        model.generation_config.eos_token_id = 248046
    elif fault == 'minimum':
        controls['min_new_tokens'] = 1
    elif fault == 'callback':
        kwargs['prefix_allowed_tokens_fn'] = lambda *_args: [1]
    else:
        kwargs['custom_generate'] = 'unreviewed'
    before = deepcopy(receipt)
    with pytest.raises(ValueError, match='exact FP32'):
        adapter.validate_generation(model, receipt, controls, kwargs)
    assert receipt == before


@pytest.mark.parametrize('field', ['contract', 'authoritativeRowsSha256', 'vocabularySha256',
    'tokenizerSemanticSha256', 'renderedInputIdsSha256', 'promptTokenCount', 'limits'])
def test_consumed_fingerprints_need_independent_expected_values(synthetic, field):
    prepared = make(synthetic)
    metadata = deepcopy(prepared.metadata)
    metadata[field] = None
    with pytest.raises(ValueError):
        adapter.validate_consumed(metadata, layout=LAYOUT, mode='t2i_bbox', image_count=0,
            controls={'min_new_tokens': 0, 'max_new_tokens': 3584}, input_hash='sha256:' + '1' * 64)
    prepared.close()


def test_verified_eos_at_exact_token_bound_is_a_stop_not_truncation(synthetic):
    prepared = make(synthetic, limit=len(T2I.encode()) + 1)
    ids = emit(prepared, T2I)
    text, completion = prepared.finish(ids)
    assert len(ids) == prepared.metadata['limits']['max_new_tokens']
    adapter.validate_completion(completion, prepared.metadata, text=text,
        generated_tokens=len(ids), input_tokens=2)
    prepared.close()


def test_existing_finite_processor_accepts_prefix_negative_inf_but_not_invalid_scores(synthetic):
    torch = pytest.importorskip('torch')
    from modiff import prompt_enhancement as engine
    synthetic.transformers.LogitsProcessor = type('LogitsProcessor', (), {})
    engine._finite_score_processor_type.cache_clear()
    try:
        processor = engine._finite_score_processor(synthetic.observer)
        # This is the ordinary official prefix-mask result: forbidden tokens
        # become -inf while at least one permitted model score remains finite.
        masked = torch.tensor([[float('-inf'), 1.0, float('-inf')]])
        assert processor(torch.tensor([[1]]), masked) is masked
        for scores in ([float('nan'), 1.0], [float('inf'), 1.0], [float('-inf'), float('-inf')]):
            with pytest.raises(RuntimeError, match='invalid model scores'):
                processor(torch.tensor([[1]]), torch.tensor([scores]))
    finally:
        engine._finite_score_processor_type.cache_clear()


@pytest.mark.parametrize('fault', [None, 'missing-eos', 'cancel'])
def test_same_official_generate_call_receives_owned_prefix_stop_and_finite_processor(synthetic, monkeypatch, fault):
    """Stub model API/CPU capture only; production adapter controls/finalizer run."""
    torch = pytest.importorskip('torch')
    from modiff import prompt_enhancement as engine
    from modiff import prompt_enhancement_summary as transport
    synthetic.transformers.LogitsProcessor = type('LogitsProcessor', (), {})
    synthetic.transformers.LogitsProcessorList = type('LogitsProcessorList', (list,), {})
    synthetic.transformers.StoppingCriteria = type('StoppingCriteria', (), {})
    synthetic.transformers.StoppingCriteriaList = type('StoppingCriteriaList', (list,), {})
    engine._finite_score_processor_type.cache_clear()
    stopped, captured, input_checks = [False], [], []
    class Observer:
        task = object()
        def __init__(self, node):
            self.node = node
        def check(self):
            if stopped[0]:
                raise InterruptedError('owned cancellation')
        def progress(self, *_args, **_kwargs):
            self.check()
    monkeypatch.setattr(engine, '_OwnedGeneration', Observer)
    contract, _raw = engine._declared_contract(with_bytes=True)
    prompts = {key: 'reviewed synthetic prompt' for key in ('t2i', 'i2i', 't2i_bbox', 'i2i_bbox')}
    monkeypatch.setattr(engine, '_selected_contract', lambda *_args: (contract, prompts, b'declaration'))
    monkeypatch.setattr(engine, 'prepare_request', lambda *_args, **_kwargs: ('t2i_bbox', 'rendered full request', deepcopy(LAYOUT)))
    base_fields = {'rendered_input_ids_sha256': 'sha256:' + '1' * 64}
    monkeypatch.setattr(engine, '_consumed_projection', lambda *_args, **_kwargs:
        (dict(base_fields), {'fingerprints': dict(base_fields)}))
    class Summary:
        def __init__(self, *_args, **_kwargs):
            pass
        def observe_inputs(self, consumed, controls):
            assert 'authored_layout_constraint' in consumed['fingerprints']
        def finish(self, result, controls):
            assert result['layoutOutputValidated'] is True and result['finishReason'] == 'stop'
            return 'synthetic producer transport'
    monkeypatch.setattr(transport, 'BoundSummary', Summary)
    nodes = ModuleType('modules.HuggingFaceTransformers.main')
    nodes._normalized_optional_images = lambda value: ([], 0)
    nodes._bounded_int = lambda value, **kwargs: value or kwargs['default']
    nodes._generation_controls = lambda kwargs, **_rest: {
        'max_new_tokens': kwargs['max_new_tokens'], 'min_new_tokens': 0, 'do_sample': False,
        'num_beams': 1, 'repetition_penalty': 1.0, 'return_dict_in_generate': False}
    nodes._batch_to_device = lambda value, **_kwargs: value
    nodes._shape = lambda value: list(value.shape)
    nodes._output_sequences = lambda value: value
    nodes._token_prefix_matches = lambda output, prefix, count: torch.equal(output[:, :count], prefix)
    package = ModuleType('modules.HuggingFaceTransformers')
    package.main = nodes
    monkeypatch.setitem(sys.modules, 'modules.HuggingFaceTransformers', package)
    monkeypatch.setitem(sys.modules, 'modules.HuggingFaceTransformers.main', nodes)
    resource = ModuleType('modiff.transformers_image_text_resources')
    resource.enforce_image_text_input_bounds = lambda *_args, **_kwargs: {'synthetic': True}
    monkeypatch.setitem(sys.modules, 'modiff.transformers_image_text_resources', resource)
    lifecycle = ModuleType('modiff.ancillary_owner_lifecycle')
    lifecycle.constrained_preallocation_context = lambda *_args: time.monotonic() + 30
    lifecycle.check_preallocation_inputs = lambda node, processor, fields, limits: input_checks.append(deepcopy(fields))
    lifecycle.before_image_text_generation = lambda model, batch, controls: None
    monkeypatch.setitem(sys.modules, 'modiff.ancillary_owner_lifecycle', lifecycle)
    processor_type = type('SyntheticProcessor', (SimpleNamespace,), {
        '__call__': lambda self, **kwargs: {'input_ids': torch.tensor([[9000, 9001]])}})
    processor = processor_type(tokenizer=synthetic.processor.tokenizer, image_processor=SimpleNamespace(patch_size=16))
    receipt = {'runtime': {'dtype': 'float32', 'device': 'cuda:0', 'quantization': {'mode': 'none'}}}
    class Model:
        generation_config = SimpleNamespace(eos_token_id=[248046, 248044], pad_token_id=248044)
        def generate(self, **kwargs):
            captured.append(kwargs)
            callback = kwargs['prefix_allowed_tokens_fn']
            tokens = list(T2I.encode())
            for index, identity in enumerate(tokens):
                prefix = torch.tensor([9000, 9001, *tokens[:index]])
                assert identity in callback(0, prefix)
                kwargs['logits_processor'][0](prefix, torch.tensor([[float('-inf'), 1.0]]))
                assert kwargs['stopping_criteria'][0](prefix[None, :], None) is False
                if fault == 'cancel' and index == 3:
                    stopped[0] = True
            terminal = [] if fault == 'missing-eos' else [248046]
            return torch.tensor([[9000, 9001, *tokens, *terminal]])
    node = SimpleNamespace(record_generation_inputs=lambda fields: None,
        mm_exec=lambda fn, device, models: fn())
    kwargs = {'operation': adapter.OPERATION, 'prompt': 'A fox <fox>.', 'layout_conditioning': deepcopy(LAYOUT),
        'enhancement_max_new_tokens': 3584}
    try:
        if fault:
            with pytest.raises((ValueError, InterruptedError)):
                engine.generate(node, Model(), processor, receipt, kwargs)
        else:
            result = engine.generate(node, Model(), processor, receipt, kwargs)
            assert result['result']['text'] == T2I and result['result']['applicationAllowed'] is True
            assert result['text'] is None and result['proposal_summary'] == 'synthetic producer transport'
        assert len(captured) == 1 and len(input_checks) == 1
        assert captured[0]['prefix_allowed_tokens_fn'].vocabulary is None
        assert kwargs['prompt'] == 'A fox <fox>.' and 'prefix_allowed_tokens_fn' not in kwargs
    finally:
        engine._finite_score_processor_type.cache_clear()


@pytest.mark.parametrize('fault', [None, 'completion-missing', 'completion-hash', 'wrong-input-count',
    'descriptor-change', 'not-validated', 'dtype', 'extra-old-field'])
def test_bound_summary_requires_new_contract_and_actual_eos_at_limit(synthetic, monkeypatch, fault, tmp_path):
    from modiff import prompt_enhancement as engine
    from modiff import prompt_enhancement_summary as transport
    from modiff.prompt_enhancement_output import validate_target_layout_output
    maximum = len(T2I.encode()) + 1
    prepared = make(synthetic, limit=maximum)
    native, completion = prepared.finish([*T2I.encode(), 248046])
    contract, declaration = engine._declared_contract(with_bytes=True)
    monkeypatch.setattr(transport, '_producer', lambda _observer: {'synthetic': 'producer-bound elsewhere'})
    receipt = {'runtime': {'dtype': 'float32', 'device': 'cuda:0', 'quantization': {'mode': 'none'}}}
    monkeypatch.setattr(transport, '_model_receipt', lambda value, _contract: None)
    published = []
    monkeypatch.setattr(transport, 'finish_pending', lambda *_args: published.append(True))
    nodes = ModuleType('modules.HuggingFaceTransformers.main')
    nodes.MAX_MEDIA_SIDE, nodes.MAX_MEDIA_PIXELS, nodes.MAX_TOTAL_MEDIA_PIXELS = 4096, 16777216, 67108864
    monkeypatch.setitem(sys.modules, 'modules.HuggingFaceTransformers.main', nodes)
    rendered = 'complete original rendered prompt'
    fields = {'operation': adapter.OPERATION, 'enhancement_max_new_tokens': maximum,
        'prompt_adapter_id': contract['id'], 'system_prompt_mode': 't2i_bbox', 'system_prompt_sha256': '0' * 64,
        'rendered_prompt_sha256': adapter._digest(rendered.encode()),
        'rendered_prompt_utf8_bytes': len(rendered.encode()),
        'rendered_input_ids_sha256': 'sha256:' + '1' * 64, 'reference_source_sha256': [],
        'authored_layout_constraint': deepcopy(prepared.metadata)}
    consumed = {'schemaVersion': 1, 'renderedPrompt': rendered, 'references': [],
        'processedImageGrid': [], 'fingerprints': fields}
    controls = {'max_new_tokens': maximum, 'min_new_tokens': 0, 'do_sample': False, 'num_beams': 1,
        'repetition_penalty': 1.0, 'return_dict_in_generate': False}
    result = {'consumedInput': consumed, 'systemPromptMode': 't2i_bbox', 'modelReceipt': receipt,
        'inputTokens': 2, 'generatedTokens': maximum, 'finishReason': 'stop', 'applyRequired': True,
        'semanticFidelityVerified': False, 'structuredProposal': validate_target_layout_output(
            native, LAYOUT, mode='t2i_bbox', image_count=0), 'applicationBlockedReason': None,
        'operation': adapter.OPERATION, 'constraintCompletion': completion, 'text': native,
        'layoutOutputValidated': True, 'applicationAllowed': True}
    bound = transport.BoundSummary(synthetic.observer, declaration, contract, 'A fox <fox>.', LAYOUT,
        operation=adapter.OPERATION)
    if fault == 'completion-missing':
        del result['constraintCompletion']
    elif fault == 'completion-hash':
        result['constraintCompletion']['rawOutputSha256'] = 'sha256:' + '0' * 64
    elif fault == 'wrong-input-count':
        result['inputTokens'] = 3
    elif fault == 'descriptor-change':
        # The synthetic fixture substitutes the descriptor reader, so mutate
        # that independent source witness rather than an unused disk path.
        synthetic.contract['declarationSha256'] = 'sha256:' + '0' * 64
    elif fault == 'not-validated':
        result['applicationAllowed'] = False
    elif fault == 'dtype':
        receipt['runtime']['dtype'] = 'bfloat16'
    elif fault == 'extra-old-field':
        fields['unbound-extra'] = True
    if fault:
        with pytest.raises(ValueError):
            bound.finish(result, controls)
        assert not published
    else:
        summary = json.loads(bound.finish(result, controls))
        assert summary['proposal']['finishReason'] == 'stop'
        assert summary['proposal']['generatedTokens'] == maximum
        assert summary['proposal']['constraintCompletion'] == completion and published == [True]
    prepared.close()
