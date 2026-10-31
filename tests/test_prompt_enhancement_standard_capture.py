"""Actual ordinary caption producer vs scoped enhancement capture; synthetic weights."""
from copy import deepcopy
import json
import os
from pathlib import Path

from PIL import Image

import test_prompt_enhancement as stage_one
from modiff.execution_input_provenance import build_resolved_execution_inputs
from modules.HuggingFaceTransformers import main as nodes

actual_pup = stage_one.actual_pup


def save_producer(node, label):
    record = deepcopy(node._execution_input_record)
    graph = {'nodes': {node.node_id: {'module': node.module_name, 'action': node.class_name, 'params': {}}}}
    receipt = build_resolved_execution_inputs(graph, {node.node_id: record},
        task_id='pup-cpu-unit-' + label, attempt_index=0, node_id=node.node_id)
    output = os.environ.get('MODIFF_PUP_CAPTURE_DIR')
    if output:
        path = Path(output)
        path.mkdir(parents=True, exist_ok=True)
        with (path / (label + '.json')).open('x', encoding='utf-8') as stream:
            json.dump({'proofScope': 'Actual NodeBase/official tiny CPU call capture plus real receipt builder over an explicit single-node unit graph; not server/HTTP/trained execution',
                       'record': record, 'resolvedExecutionInputs': receipt}, stream, indent=2)
    return record, receipt


def test_fresh_standard_hidden_defaults_must_not_claim_unused_enhancement_limit(actual_pup):
    handle, _ = actual_pup
    common = dict(model=handle, prompt='Describe the red square.', images=[Image.new('RGB', (256, 256), 'red')],
                  max_new_tokens=2, min_new_tokens=2)
    legacy = nodes.GenerateImageVideoText('standard-legacy-comparison')
    legacy(**common)
    fresh = nodes.GenerateImageVideoText('standard-fresh-defaults')
    fresh(**common, operation='standard', enhancement_max_new_tokens=3584, layout_conditioning=None)
    record, receipt = save_producer(fresh, 'standard-fresh-defaults')
    assert fresh.output == legacy.output
    assert record['fields']['operation']['value'] == 'standard'
    assert 'enhancement_max_new_tokens' not in record['fields']
    assert 'enhancementMaxNewTokens' not in receipt['summary']
    assert not any(key.startswith(('rendered_', 'reference_', 'processed_', 'system_')) for key in record['fields'])


def test_legacy_omitted_standard_operation_is_not_synthesized(actual_pup):
    handle, _ = actual_pup
    node = nodes.GenerateImageVideoText('standard-legacy-omitted')
    node(model=handle, prompt='Describe the blue square.', images=[Image.new('RGB', (256, 256), 'blue')],
         max_new_tokens=2, min_new_tokens=2)
    record, receipt = save_producer(node, 'standard-legacy-omitted')
    assert 'operation' not in record['fields'] and 'textOperation' not in receipt['summary']
    assert 'enhancement_max_new_tokens' not in record['fields']
    assert node.output['result']['task'] == 'image-video-to-text'


def test_enhance_producer_retains_computed_token_reference_and_tensor_evidence(actual_pup):
    handle, _ = actual_pup
    node = nodes.GenerateImageVideoText('enhancement-two-reference-producer')
    node(model=handle, prompt='Combine the red and blue squares.', operation='enhance_prompt',
         images=[Image.new('RGB', (256, 256), color) for color in ['red', 'blue']],
         enhancement_max_new_tokens=2, min_new_tokens=2)
    record, receipt = save_producer(node, 'enhance-two-references')
    summary = receipt['summary']
    assert record['fields']['enhancement_max_new_tokens']['value'] == 2
    assert summary['textOperation'] == 'enhance_prompt' and summary['enhancementMaxNewTokens'] == 2
    assert len(summary['referenceSourceSha256']) == 2
    assert summary['processedGridShape'] == [2, 3]
    assert len(summary['processedPixelShape']) == 2
    assert summary['renderedPromptSha256'].startswith('sha256:')
