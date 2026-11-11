
"""Saved vanilla compatibility; no adapter permission or safety/media proof."""
import json
import modiff
from copy import deepcopy
from pathlib import Path
import pytest
from test_native_image_component_lora import runtime as _runtime, graph, descriptor, MODEL, REPO
from modiff.model_artifact_catalog import require_catalog_revision

runtime = _runtime

def saved_source(runtime, item=None):
    spec=json.loads((Path(modiff.__file__).resolve().parents[1]/'data/studio-execution-spec-compatibility.v1.json').read_text())['specifications'][0]['specification']
    source=graph(item)
    owner=source['nodes']['models']['params']
    owner['repo_id']['value']={'source':'hub','value':'nvidia/Cosmos3-Nano'}
    owner['revision']['value']=require_catalog_revision('nvidia/Cosmos3-Nano',model_type=MODEL)
    owner['reviewed_variant']['value']=REPO
    source['nodes']['afterDecode']={'module':'modules.ModularDiffusers','action':'WorkflowCosmos3OmniAfterDecode','params':{
        'pipeline_class':{'value':MODEL},'workflow_id':{'value':'text2image'},'block_path':{'value':'after_decode'}}}
    for left,port,right,field in spec['edges']:
        source['nodes'][right]['params'][field]={'sourceId':left,'sourceKey':port}
    source['paths']=[['models','prompt','denoise','decode','preview'],['models','prompt','denoise','decode','afterDecode']]
    for role,field,_source in spec['bindings']:
        params=source['nodes'][role]['params']
        if field not in params:
            node=source['nodes'][role]
            definition=runtime.server.modules[node['module']][node['action']]['params'][field]
            params[field]={'value':deepcopy(definition.get('value',definition.get('default')))}
    hints={'modelType':MODEL,'mode':'text_to_image','resourceMode':'expert',
           'studioExecutionSpec':{'schemaVersion':1,'id':spec['id'],'contentHash':spec['contentHash'],
             'nodes':{role[0]:role[0] for role in spec['roles']}}}
    source['runtimeHints']=runtime.server._coerce_runtime_hints(hints)
    assert source['runtimeHints']['studioExecutionSpec']['contentHash']=='studio-spec-v1-91ed105d'
    return source

def test_saved_historical_none_vanilla_stays_ordinary(runtime):
    node=runtime.run(saved_source(runtime))
    assert node.loader.requires_safety_checker is True
    assert node.output['pipeline_components']
    assert node._native_disk_lora_state is None

@pytest.mark.parametrize('field', ['value', 'default'])
def test_saved_historical_nonempty_adapter_remains_refused(runtime,tmp_path,field):
    item=descriptor(tmp_path)
    source=saved_source(runtime,[item])
    source['nodes']['models']['params']['lora_list']={field:[item]}
    with pytest.raises((ValueError,RuntimeError)):runtime.run(source)
    assert not runtime.events
    assert all(not getattr(ref(),'peft_config',{}) for name,ref in runtime.references if name=='transformer' and ref() is not None)
