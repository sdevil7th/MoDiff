"""Numerical generation failures must not become applicable prompt proposals."""
from unittest.mock import Mock

import pytest
import torch

import test_prompt_enhancement as stage_one
import test_prompt_enhancement_ownership as owner_tests
from modiff.native_component_lora import native_component_graph_scope
from modiff.NodeBase import memory_manager
from modiff.prompt_enhancement import _finite_score_processor
from modules.HuggingFaceTransformers import main as nodes

actual_pup = stage_one.actual_pup
isolated_manager = owner_tests.isolated_manager


def test_nonfinite_native_scores_fail_before_token_or_proposal(actual_pup, monkeypatch):
    handle, _ = actual_pup
    model = handle['model']
    node = nodes.GenerateImageVideoText('pup-invalid-scores')
    graph, _task, _server = stage_one.owned_generation(monkeypatch, node)
    decoder = Mock(wraps=nodes._decode)
    monkeypatch.setattr(nodes, '_decode', decoder)
    calls = []

    def invalid_logits(_module, _inputs, output):
        calls.append(True)
        assert memory_manager._active_ids[model._mm_id] > 0
        return torch.full_like(output, float('nan'))

    hook = model.lm_head.register_forward_hook(invalid_logits)
    try:
        with native_component_graph_scope(graph, stage_one.unit_source_fingerprint()):
            with pytest.raises(RuntimeError, match='invalid model scores'):
                node(model=handle, prompt='A red cube.', operation='enhance_prompt',
                     enhancement_max_new_tokens=2, min_new_tokens=2)
    finally:
        hook.remove()
    assert calls == [True] and not decoder.called
    assert not memory_manager._active_ids
    assert memory_manager.get_model(model._mm_id) is model
    assert not node._cache_valid and all(value is None for value in node.output.values())


@pytest.mark.parametrize('values', [[[0., 1., -2.]], [[float('-inf'), 0., float('-inf')]],
                                  [[0., -1.], [float('-inf'), 0.]]])
def test_valid_and_masked_scores_are_returned_unchanged(values):
    from transformers import LogitsProcessor
    observer = Mock()
    processor = _finite_score_processor(observer)
    assert isinstance(processor, LogitsProcessor)
    scores = torch.tensor(values)
    original = scores.clone()
    assert processor(torch.ones((len(values), 1), dtype=torch.long), scores) is scores
    assert torch.equal(scores, original) and observer.check.called


@pytest.mark.parametrize('values', [[[float('nan'), 0.]], [[0., float('inf')]],
    [[float('-inf'), float('-inf')]], [[float('nan'), float('nan')]],
    [[0., -1.], [float('nan'), 0.]], [[0., -1.], [float('-inf'), float('-inf')]]])
def test_each_row_requires_usable_unsanitized_scores(values):
    processor = _finite_score_processor(Mock())
    scores = torch.tensor(values)
    original = scores.clone()
    with pytest.raises(RuntimeError, match='invalid model scores.*no proposal can be applied'):
        processor(torch.ones((len(values), 1), dtype=torch.long), scores)
    assert torch.allclose(scores, original, equal_nan=True)


def test_score_check_preserves_task_cancellation():
    observer = Mock()
    observer.check.side_effect = InterruptedError('current task stopped')
    with pytest.raises(InterruptedError, match='current task stopped'):
        _finite_score_processor(observer)(torch.ones((1, 1)), torch.tensor([[float('nan')]]))


def test_finite_exclamation_tokens_are_not_filtered(actual_pup, monkeypatch):
    from modiff import ancillary_owner_lifecycle
    from modiff.transformers_image_text_resources import _validate_image_text_t2i_generation_controls
    handle, _ = actual_pup
    model = handle['model']
    node = nodes.GenerateImageVideoText('pup-finite-exclamation')
    graph, _task, _server = stage_one.owned_generation(monkeypatch, node)
    validated = []
    before_generate = ancillary_owner_lifecycle.before_image_text_generation
    official_generate = model.generate

    def checked_controls(actual_model, batch, controls):
        # Isolate the actual controls boundary, not the full native Auto grant.
        _validate_image_text_t2i_generation_controls(controls)
        validated.append(dict(controls))
        before_generate(actual_model, batch, controls)

    def checked_generate(**kwargs):
        assert len(validated) == 1
        assert all(kwargs[key] is value for key, value in validated[0].items())
        return official_generate(**kwargs)

    monkeypatch.setattr(ancillary_owner_lifecycle, 'before_image_text_generation', checked_controls)
    monkeypatch.setattr(model, 'generate', checked_generate)

    def finite_exclamation(_module, _inputs, output):
        scores = torch.full_like(output, float('-inf'))
        scores[..., 0] = 0.
        return scores

    hook = model.lm_head.register_forward_hook(finite_exclamation)
    try:
        with native_component_graph_scope(graph, stage_one.unit_source_fingerprint()):
            node(model=handle, prompt='A red cube.', operation='enhance_prompt',
                 enhancement_max_new_tokens=2, min_new_tokens=2)
    finally:
        hook.remove()
    assert node.output['text'] == node.output['result']['text'] == '!!'
    assert node.output['result']['generatedTokens'] == 2 and node.output['result']['applyRequired']
    assert not memory_manager._active_ids


def owned_auto_controls(monkeypatch):
    from transformers import LogitsProcessorList, StoppingCriteria, StoppingCriteriaList
    from modiff.prompt_enhancement import _OwnedGeneration
    node = nodes.GenerateImageVideoText('pup-auto-score-control')
    graph, task, server = stage_one.owned_generation(monkeypatch, node)
    with native_component_graph_scope(graph, stage_one.unit_source_fingerprint()):
        controls = nodes._generation_controls({'max_new_tokens': 2})
        controls['stopping_criteria'] = StoppingCriteriaList([StoppingCriteria()])
        controls['logits_processor'] = LogitsProcessorList([_finite_score_processor(_OwnedGeneration(node))])
    return graph, task, server, controls


def test_auto_generation_accepts_actual_task_owned_score_guard(monkeypatch):
    from modiff.transformers_image_text_resources import _validate_image_text_t2i_generation_controls
    graph, _task, _server, controls = owned_auto_controls(monkeypatch)
    with native_component_graph_scope(graph, stage_one.unit_source_fingerprint()):
        _validate_image_text_t2i_generation_controls(controls)


@pytest.mark.parametrize('fault', ['missing', 'extra', 'empty', 'plain-list', 'list-subclass', 'foreign',
    'subclass', 'unbound', 'unbound-real', 'other-graph', 'stopping', 'stopping-extra', 'unknown',
    'greedy', 'beam-count', 'max-count', 'min-count', 'return'])
def test_auto_rejects_missing_or_replaced_guard_and_other_control_drift(monkeypatch, fault):
    from copy import deepcopy
    from transformers import LogitsProcessor, LogitsProcessorList
    from modiff.transformers_image_text_resources import _validate_image_text_t2i_generation_controls
    graph, _task, _server, controls = owned_auto_controls(monkeypatch)
    guard = controls['logits_processor'][0]
    if fault == 'missing':
        del controls['logits_processor']
    elif fault == 'extra':
        controls['logits_processor'].append(guard)
    elif fault == 'empty':
        controls['logits_processor'].clear()
    elif fault == 'plain-list':
        controls['logits_processor'] = [guard]
    elif fault == 'list-subclass':
        class ReplacedList(LogitsProcessorList):
            pass
        controls['logits_processor'] = ReplacedList([guard])
    elif fault == 'foreign':
        class NoOp(LogitsProcessor):
            def __call__(self, input_ids, scores):
                return scores
        controls['logits_processor'] = LogitsProcessorList([NoOp()])
    elif fault == 'subclass':
        class Replaced(type(guard)):
            def __call__(self, input_ids, scores):
                return scores
        controls['logits_processor'] = LogitsProcessorList([Replaced(guard.observer)])
    elif fault == 'unbound':
        guard.observer = Mock()
    elif fault == 'unbound-real':
        from modiff.prompt_enhancement import _OwnedGeneration
        guard.observer = _OwnedGeneration(guard.observer.node)
    elif fault == 'other-graph':
        graph = deepcopy(graph)
    elif fault == 'stopping':
        controls['stopping_criteria'].clear()
    elif fault == 'stopping-extra':
        controls['stopping_criteria'].append(controls['stopping_criteria'][0])
    elif fault == 'unknown':
        controls['temperature'] = 1.
    elif fault == 'greedy':
        controls['do_sample'] = True
    elif fault == 'beam-count':
        controls['num_beams'] = 2
    elif fault == 'max-count':
        controls['max_new_tokens'] = 3585
    elif fault == 'min-count':
        controls['min_new_tokens'] = 3
    else:
        controls['return_dict_in_generate'] = True
    with native_component_graph_scope(graph, stage_one.unit_source_fingerprint()):
        with pytest.raises(ValueError, match='undeclared control|task-owned finite-score|greedy|alternate generation controls'):
            _validate_image_text_t2i_generation_controls(controls)


def test_auto_guard_retains_current_task_and_interrupt_checks(monkeypatch):
    from modiff.transformers_image_text_resources import _validate_image_text_t2i_generation_controls
    graph, task, server, controls = owned_auto_controls(monkeypatch)
    with native_component_graph_scope(graph, stage_one.unit_source_fingerprint()):
        task['interrupt_requested'] = True
        with pytest.raises(InterruptedError, match='interrupted by the current task'):
            _validate_image_text_t2i_generation_controls(controls)
        task['interrupt_requested'] = False
        server.current_task = dict(task)
        with pytest.raises(InterruptedError, match='no longer belongs'):
            _validate_image_text_t2i_generation_controls(controls)
