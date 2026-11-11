"""Explicit Expert adapter for the official tokenizer's authored-table prefix.

This module binds an already selected local processor and existing generation
call. It never loads a model, installs a runtime, repairs a table or grants Auto.
The model must emit every byte, and the independent strict parser remains final.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import re
import time

from modiff.structured_prompt_constraints import (
    AuthoredLayoutPrefixConstraint, ByteVocabulary, CONSTRAINT_VERSION, ConstraintLimits,
)

OPERATION = 'enhance_prompt_constrained_authored_v1'
LAYOUT_POLICY = 'constrained-authored-table-v1'
DESCRIPTOR = Path(__file__).resolve().parents[1] / 'data/prompt-enhancement-constraints.v1.json'
PREPARATION_RAM_BYTES = 256 * 1024**2
DEADLINE_SECONDS = 1800


def _digest(value):
    return 'sha256:' + hashlib.sha256(value).hexdigest()


def _canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')


def _wire_limits(limits):
    # JSON.parse cannot preserve Python's integral-float spelling. Wire hashes
    # use integer values; internal dataclass clocks and bounds are unchanged.
    return {key: int(value) if type(value) is float and value.is_integer() else value
            for key, value in asdict(limits).items()}


def constraint_contract():
    """Read bounded source authority; no processor, model or cache observation."""
    from modiff.prompt_enhancement import _json_file, _declared_contract
    raw, value = _json_file(DESCRIPTOR, 16 * 1024)
    adapter = _declared_contract()
    keys = {'schemaVersion', 'version', 'status', 'operation', 'layoutPolicy', 'adapterId',
        'repository', 'revision', 'transformersVersion', 'tokenizersVersion', 'tokenizerClass',
        'ordinaryVocabularyTokens', 'ordinaryVocabularySha256', 'tokenizerSemanticSha256',
        'sourcePreTokenizerSha256', 'effectivePreTokenizerSha256',
        'reservedAddedTokenIds', 'trueSpecialTokenIds',
        'configuredEosTokenIds', 'effectiveEosTokenIds', 'padTokenId', 'preparationRamBytes',
        'deadlineSeconds', 'artifacts'}
    if (type(value) is not dict or set(value) != keys or type(value['schemaVersion']) is not int
            or value['schemaVersion'] != 1 or value['version'] != CONSTRAINT_VERSION
            or value['status'] != 'contract_only' or value['operation'] != OPERATION
            or value['layoutPolicy'] != LAYOUT_POLICY or value['adapterId'] != adapter['id']
            or any(value[key] != adapter[key] for key in ('repository', 'revision', 'transformersVersion'))
            or value['tokenizersVersion'] != '0.23.2' or value['tokenizerClass'] != 'Qwen3_5Tokenizer'
            or type(value['ordinaryVocabularyTokens']) is not int or value['ordinaryVocabularyTokens'] != 248044
            or any(type(value[key]) is not str or not re.fullmatch('sha256:[0-9a-f]{64}', value[key])
                   for key in ('ordinaryVocabularySha256', 'tokenizerSemanticSha256',
                               'sourcePreTokenizerSha256', 'effectivePreTokenizerSha256'))
            or value['reservedAddedTokenIds'] != list(range(248044, 248077))
            or value['trueSpecialTokenIds'] != [*range(248044, 248058), *range(248070, 248077)]
            or value['configuredEosTokenIds'] != [248046, 248044] or value['effectiveEosTokenIds'] != [248046]
            or type(value['padTokenId']) is not int or value['padTokenId'] != 248044
            or type(value['preparationRamBytes']) is not int or value['preparationRamBytes'] != PREPARATION_RAM_BYTES
            or type(value['deadlineSeconds']) is not int or value['deadlineSeconds'] != DEADLINE_SECONDS):
        raise ValueError('The authored-table constraint declaration differs from its exact reviewed scope.')
    for key in ('reservedAddedTokenIds', 'trueSpecialTokenIds', 'configuredEosTokenIds', 'effectiveEosTokenIds'):
        if type(value[key]) is not list or any(type(identity) is not int for identity in value[key]):
            raise ValueError('Constraint token identities require exact integer JSON arrays.')
    artifacts = value['artifacts']
    if type(artifacts) is not list or [row.get('path') for row in artifacts if type(row) is dict] != [
            'tokenizer.json', 'tokenizer_config.json', 'generation_config.json']:
        raise ValueError('The constraint tokenizer artifacts are not the exact logical selection.')
    for row in artifacts:
        if (type(row) is not dict or set(row) != {'path', 'byteSize', 'sha256'}
                or type(row['byteSize']) is not int or not 0 < row['byteSize'] <= 24 * 1024**2
                or type(row['sha256']) is not str or not re.fullmatch('[0-9a-f]{64}', row['sha256'])):
            raise ValueError('The constraint tokenizer artifact is malformed.')
    result = deepcopy(value)
    result['declarationSha256'] = _digest(raw)
    result['limits'] = _wire_limits(ConstraintLimits(max_new_tokens=3584, max_request_seconds=DEADLINE_SECONDS))
    return result


def authoritative_rows(layout, mode, image_count):
    """Same lossless projection as the unchanged preservation request/parser."""
    from modiff.prompt_enhancement_output import require_authored_target_layout
    from modiff.layout_conditioning import _upstream_box
    parsed, _expected = require_authored_target_layout(layout)
    if mode not in ('t2i_bbox', 'i2i_bbox') or (mode == 't2i_bbox') != (image_count == 0):
        raise ValueError('Constrained output requires the exact authored layout/reference mode.')
    rows = []
    for region in parsed['regions']:
        source = region.get('source')
        target = None if region['targetBox'] is None else _upstream_box(region['targetBox'])
        row = {'id': region['id']}
        if image_count:
            if source is not None and source['referenceIndex'] >= image_count:
                raise ValueError('The constrained row names a reference outside the actual ordered context.')
            row.update({'from': 'target' if source is None else f"ref_image_{source['referenceIndex']}",
                'src_bbox': None if source is None else _upstream_box(source['box']), 'tgt_bbox': target})
        else:
            if source is not None:
                raise ValueError('A text-only constrained table cannot acquire reference-source authority.')
            row['bbox'] = target
        row['desc'] = region['description']
        rows.append(row)
    return rows


def begin_preallocation(scope):
    """Start once, at actual owned provider preallocation, never from graph data."""
    from modiff.ancillary_owner_lifecycle import _Scope
    if (type(scope) is not _Scope or not scope.preview or scope.auto
            or scope.server.current_task.get('current_node') != scope.provider
            or hasattr(scope, '_authored_constraint_deadline')):
        raise ValueError('Constrained generation requires a fresh exact Expert preallocation scope.')
    scope.check()
    scope._authored_constraint_deadline = time.monotonic() + DEADLINE_SECONDS
    return scope._authored_constraint_deadline


def _read_artifacts(contract, observer):
    from utils.huggingface import exact_cached_snapshot_path
    from modiff.hf_cache_layout import resolve_snapshot_cache_file
    snapshot = exact_cached_snapshot_path(contract['repository'], contract['revision'], marker_file='config.json')
    cache_root = snapshot.parent.parent.parent
    documents = {}
    for row in contract['artifacts']:
        observer.check()
        path = resolve_snapshot_cache_file(snapshot / row['path'], snapshot=snapshot,
            cache_root=cache_root, repository=contract['repository'])
        before = path.stat()
        if before.st_size != row['byteSize']:
            raise ValueError('Constrained tokenizer artifact size changed.')
        raw = bytearray()
        with path.open('rb') as stream:
            while chunk := stream.read(1024 * 1024):
                observer.check()
                raw.extend(chunk)
                if len(raw) > row['byteSize']:
                    raise ValueError('Constrained tokenizer artifact exceeded its immutable size.')
        after = path.stat()
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) !=
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                or len(raw) != row['byteSize'] or hashlib.sha256(raw).hexdigest() != row['sha256']):
            raise ValueError('Constrained tokenizer artifact changed while being verified.')
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError('Duplicate tokenizer JSON keys are not supported.')
                result[key] = value
            return result
        documents[row['path']] = json.loads(raw, object_pairs_hook=unique,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError('Tokenizer JSON must be finite.')))
    observer.check()
    return documents


def _tokenizer_vocabulary(processor, contract, observer):
    """Bind actual official APIs to the verified immutable ByteLevel semantics."""
    import transformers
    import tokenizers
    from transformers.convert_slow_tokenizer import bytes_to_unicode
    tokenizer = getattr(processor, 'tokenizer', None)
    native = getattr(tokenizer, 'backend_tokenizer', None)
    expected_class = getattr(transformers, contract['tokenizerClass'], None)
    if (str(transformers.__version__) != contract['transformersVersion']
            or str(tokenizers.__version__) != contract['tokenizersVersion']
            or not isinstance(expected_class, type) or type(tokenizer) is not expected_class
            or type(native) is not tokenizers.Tokenizer):
        raise ValueError('Constrained generation requires its exact official tokenizer APIs.')
    documents = _read_artifacts(contract, observer)
    selected = documents['tokenizer.json']
    serialized = native.to_str()
    if type(serialized) is not str or len(serialized.encode('utf-8')) > 32 * 1024**2:
        raise ValueError('The actual tokenizer serialization exceeds its finite bound.')
    actual = json.loads(serialized)
    keys = ('model', 'decoder', 'normalizer', 'pre_tokenizer', 'post_processor', 'added_tokens')
    if (type(selected) is not dict or type(actual) is not dict
            or any(actual.get(key) != selected.get(key) for key in keys if key != 'pre_tokenizer')
            or selected.get('model', {}).get('type') != 'BPE'
            or selected['model'].get('byte_fallback') is not False
            or selected.get('decoder', {}).get('type') != 'ByteLevel'):
        raise ValueError('The actual tokenizer differs from its immutable ByteLevel BPE semantics.')
    # The exact official class reconstructs its preprocessing regex. Bind both
    # reviewed forms; no other semantic difference or arbitrary rewrite is allowed.
    if (_digest(_canonical(selected.get('pre_tokenizer'))) != contract['sourcePreTokenizerSha256']
            or _digest(_canonical(actual.get('pre_tokenizer'))) != contract['effectivePreTokenizerSha256']):
        raise ValueError('The source or effective tokenizer preprocessing fingerprint changed.')
    semantic_digest = hashlib.sha256()
    for index, chunk in enumerate(json.JSONEncoder(ensure_ascii=True, sort_keys=True,
            separators=(',', ':'), allow_nan=False).iterencode({key: actual.get(key) for key in keys})):
        if index % 128 == 0:
            observer.check()
        semantic_digest.update(chunk.encode('ascii'))
    semantic_hash = 'sha256:' + semantic_digest.hexdigest()
    if semantic_hash != contract['tokenizerSemanticSha256']:
        raise ValueError('The reviewed tokenizer semantic fingerprint changed.')
    added = selected['added_tokens']
    reserved = sorted(row['id'] for row in added)
    special = sorted(row['id'] for row in added if row['special'] is True)
    generation = documents['generation_config.json']
    config = documents['tokenizer_config.json']
    if (reserved != contract['reservedAddedTokenIds'] or special != contract['trueSpecialTokenIds']
            or generation.get('eos_token_id') != contract['configuredEosTokenIds']
            or generation.get('pad_token_id') != contract['padTokenId']
            or tokenizer.eos_token_id not in contract['effectiveEosTokenIds']
            or [row['id'] for row in added if row['content'] == config.get('eos_token')] != contract['effectiveEosTokenIds']):
        raise ValueError('The actual tokenizer special/control/EOS identities changed.')
    source = selected['model']['vocab']
    if (type(source) is not dict or len(source) != contract['ordinaryVocabularyTokens']
            or native.get_vocab(with_added_tokens=False) != source
            or set(source.values()) != set(range(contract['ordinaryVocabularyTokens']))):
        raise ValueError('The actual ordinary token vocabulary changed.')
    native_added = native.get_added_tokens_decoder()
    if set(native_added) != set(reserved):
        raise ValueError('The actual added-token inventory changed.')
    for row in added:
        token = native_added[row['id']]
        if any(getattr(token, key, None) != row[key] for key in
                ('content', 'single_word', 'lstrip', 'rstrip', 'normalized', 'special')):
            raise ValueError('The actual added-token flags/content changed.')
    # Retain only ordinary spellings while building the trie. The large merge
    # arrays and both complete JSON documents are not request-lifetime caches.
    del documents, selected, actual, serialized, added, native_added, config, generation
    reverse = {character: byte for byte, character in bytes_to_unicode().items()}
    token_bytes = {}
    for index, (spelling, identity) in enumerate(source.items()):
        if index % 128 == 0:
            observer.check()
        try:
            raw = bytes(reverse[character] for character in spelling)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('The tokenizer has no exact ordinary ByteLevel mapping.') from error
        if native.decoder.decode([spelling]) != raw.decode('utf-8', errors='replace'):
            raise ValueError('The actual official ByteLevel decoder differs from its exact bytes.')
        token_bytes[identity] = raw
    observer.check()
    return tokenizer, token_bytes, semantic_hash


class _DeadlineCheck:
    def __init__(self, observer, deadline):
        if (type(deadline) not in (int, float) or not math.isfinite(deadline)
                or deadline <= time.monotonic()):
            raise ValueError('The actual owned constrained-generation deadline is required.')
        self.observer, self.deadline = observer, deadline

    def check(self):
        self.observer.check()
        if time.monotonic() >= self.deadline:
            raise InterruptedError('The owned constrained-generation cooperative deadline expired.')


class PreparedConstraint:
    """Request-local vocabulary/callback; never attached to the retained model."""
    def __init__(self, processor, layout, mode, image_count, prompt_ids, input_hash,
                 limit, deadline, observer):
        contract = constraint_contract()
        if type(limit) is not int or not 0 < limit <= 3584:
            raise ValueError('Constrained generation requires the bounded selected token limit.')
        checked = _DeadlineCheck(observer, deadline)
        checked.check()
        tokenizer, token_bytes, semantic_hash = _tokenizer_vocabulary(processor, contract, checked)
        limits = ConstraintLimits(**contract['limits'])
        limits = replace(limits, max_new_tokens=limit)
        def check():
            checked.check()
            return False
        check()
        vocabulary = ByteVocabulary(token_bytes, special_token_ids=contract['reservedAddedTokenIds'],
            eos_token_ids=contract['effectiveEosTokenIds'], limits=limits, should_stop=check)
        if 'sha256:' + vocabulary.fingerprint != contract['ordinaryVocabularySha256']:
            raise ValueError('The actual ordinary vocabulary fingerprint differs from source authority.')
        rows = authoritative_rows(layout, mode, image_count)
        self.callback = AuthoredLayoutPrefixConstraint(vocabulary, rows, mode=mode, image_count=image_count,
            prompt_token_ids=prompt_ids, deadline_monotonic=deadline, should_stop=check)
        self.metadata = {'contract': contract, 'authoritativeRowsSha256': _digest(_canonical(rows)),
            'vocabularySha256': 'sha256:' + vocabulary.fingerprint, 'tokenizerSemanticSha256': semantic_hash,
            'renderedInputIdsSha256': input_hash, 'promptTokenCount': len(prompt_ids), 'limits': _wire_limits(limits)}
        self.tokenizer, self.observer, self.vocabulary = tokenizer, observer, vocabulary

    def finish(self, continuation):
        ids = continuation.tolist() if hasattr(continuation, 'tolist') else list(continuation)
        raw = self.callback.finalize(ids)
        decoded = self.tokenizer.decode(ids[:-1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if type(decoded) is not str or decoded.encode('utf-8') != raw.encode('utf-8'):
            raise ValueError('The actual tokenizer full decode differs from the emitted grammar bytes.')
        self.observer.check()
        return raw, {'version': CONSTRAINT_VERSION, 'terminalTokenId': ids[-1], 'grammarComplete': True,
            'rawOutputSha256': _digest(raw.encode('utf-8')), 'generatedTokens': len(ids),
            'constraintFingerprintSha256': _digest(_canonical(self.metadata))}

    def close(self):
        callback = self.callback
        if callback is not None:
            callback._failed = True
            callback._cache.clear()
            callback._cache_bytes = 0
            callback._should_stop = callback.vocabulary = None
        self.callback = self.tokenizer = self.vocabulary = self.observer = None


def validate_generation(model, receipt, controls, kwargs):
    """No caller callback, silent precision substitution or changed native EOS."""
    runtime = receipt.get('runtime', {})
    config = getattr(model, 'generation_config', None)
    contract = constraint_contract()
    if (runtime.get('dtype') != 'float32' or runtime.get('device') != 'cuda:0'
            or runtime.get('quantization') != {'mode': 'none'}
            or getattr(config, 'eos_token_id', None) != contract['configuredEosTokenIds']
            or getattr(config, 'pad_token_id', None) != contract['padTokenId']
            or type(controls.get('min_new_tokens')) is not int or controls['min_new_tokens'] != 0
            or controls.get('do_sample') is not False or type(controls.get('num_beams')) is not int
            or controls['num_beams'] != 1 or type(controls.get('repetition_penalty')) not in (int, float)
            or controls['repetition_penalty'] != 1.0
            or any(key in kwargs for key in ('prefix_allowed_tokens_fn', 'constraints', 'force_words_ids',
                'custom_generate', 'eos_token_id', 'pad_token_id', 'logits_processor', 'stopping_criteria'))):
        raise ValueError('Constrained generation requires its exact FP32/none Expert controls and native EOS.')


def validate_consumed(metadata, *, layout, mode, image_count, controls, input_hash):
    """Independent expected source/layout authority, never proposal-supplied pins."""
    expected = constraint_contract()
    keys = {'contract', 'authoritativeRowsSha256', 'vocabularySha256', 'tokenizerSemanticSha256',
            'renderedInputIdsSha256', 'promptTokenCount', 'limits'}
    if (type(metadata) is not dict or set(metadata) != keys or metadata['contract'] != expected
            or metadata['authoritativeRowsSha256'] != _digest(_canonical(authoritative_rows(layout, mode, image_count)))
            or metadata['vocabularySha256'] != expected['ordinaryVocabularySha256']
            or metadata['tokenizerSemanticSha256'] != expected['tokenizerSemanticSha256']
            or metadata['renderedInputIdsSha256'] != input_hash
            or type(metadata['promptTokenCount']) is not int or not 0 < metadata['promptTokenCount'] <= 32768
            or controls['min_new_tokens'] != 0):
        raise ValueError('Constrained consumed inputs differ from independent source/layout/tokenizer authority.')
    limits = replace(ConstraintLimits(**expected['limits']), max_new_tokens=controls['max_new_tokens'])
    if (type(metadata['limits']) is not dict or any(type(value) is not int for value in metadata['limits'].values())
            or metadata['limits'] != _wire_limits(limits)):
        raise ValueError('Constrained consumed limits differ from the actual finite generation controls.')


def validate_completion(value, metadata, *, text, generated_tokens, input_tokens):
    keys = {'version', 'terminalTokenId', 'grammarComplete', 'rawOutputSha256',
            'generatedTokens', 'constraintFingerprintSha256'}
    if (type(value) is not dict or set(value) != keys or value['version'] != CONSTRAINT_VERSION
            or type(text) is not str
            or type(value['terminalTokenId']) is not int or value['terminalTokenId'] != 248046
            or value['grammarComplete'] is not True or type(value['generatedTokens']) is not int
            or value['generatedTokens'] != generated_tokens or metadata['promptTokenCount'] != input_tokens
            or value['rawOutputSha256'] != _digest(text.encode('utf-8'))
            or value['constraintFingerprintSha256'] != _digest(_canonical(metadata))):
        raise ValueError('Constrained completion requires the actual complete model-emitted bytes and terminal EOS.')


def prepare(processor, batch, layout, mode, image_count, limit, deadline, observer, input_hash):
    ids = batch.get('input_ids')
    shape = getattr(ids, 'shape', ())
    if len(shape) != 2 or shape[0] != 1 or getattr(getattr(ids, 'device', None), 'type', None) != 'cpu':
        raise ValueError('Constrained generation binds one actual CPU prompt token prefix before placement.')
    prompt = ids[0].tolist()
    if type(prompt) is not list or any(type(value) is not int for value in prompt):
        raise ValueError('Constrained generation requires actual integer prompt token IDs.')
    return PreparedConstraint(processor, layout, mode, image_count, prompt, input_hash, limit, deadline, observer)
