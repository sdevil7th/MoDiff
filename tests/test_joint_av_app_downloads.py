"""The ordinary app must acquire one finite task partition, never a whole H3 repo."""
import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from modiff.joint_av_artifacts import reviewed_joint_av_artifact_closure
from modiff.server import WebServer
from modiff.studio_execution_specs import studio_capability_definition
import test_hf_download_concurrency as download_tests


PIPELINES = ('MiniMaxH3ModularPipeline', 'PikaH3FlexModularPipeline')


@pytest.mark.parametrize('pipeline', PIPELINES)
def test_app_inventory_publishes_each_exact_task_without_combining_partitions(pipeline):
    capability = studio_capability_definition(pipeline)
    assert capability['downloadFiles'] == []
    selections = capability['artifactSelections']
    assert len(selections) == 3
    for workflow, mode in (
        ('t2va', 'text_to_video_with_audio'),
        ('fl2va', 'first_last_frame_to_video_with_audio'),
        ('ref2va', 'reference_to_video_with_audio'),
    ):
        expected = reviewed_joint_av_artifact_closure(pipeline, workflow)
        selection = next(row for row in selections if row['modes'] == [mode])
        assert selection['repo'] == expected['repository']
        assert selection['revision'] == expected['revision']
        assert selection['downloadFiles'] == [row['path'] for row in expected['files']]
    assert capability['qualifiedModes'] == []
    assert all(capability[field] is False for field in ('autoEligible', 'templateEligible', 'galleryEligible', 'liveProof'))


@pytest.mark.parametrize('pipeline', PIPELINES)
@pytest.mark.parametrize('fault', ('missing', 'partial', 'wildcard', 'mixed', 'revision'))
def test_app_rejects_unreviewed_selections_before_planning_or_scheduling(pipeline, fault):
    asyncio.run(_rejected_selection(pipeline, fault))


async def _rejected_selection(pipeline, fault):
    expected = reviewed_joint_av_artifact_closure(pipeline, 't2va')
    files = [row['path'] for row in expected['files']]
    revision = expected['revision']
    if fault == 'missing':
        files = []
    elif fault == 'partial':
        files = files[:-1]
    elif fault == 'wildcard':
        files = ['*']
    elif fault == 'mixed':
        files += ['transformer_ref/config.json']
    else:
        revision = 'a' * 40
    server = WebServer(modules={})
    server.loop = asyncio.get_running_loop()
    plan = download_tests.HuggingFaceDownloadConcurrencyTests._space_plan()
    with patch('modiff.server.plan_hub_model_download', return_value=plan) as planner:
        response = await server.hf_download_plan(download_tests.FakeRequest(
            repo_id=expected['repository'], revision=revision, file=','.join(files)))
    assert response.status == 400
    assert json.loads(response.text)['code'] == 'invalid_reviewed_model_selection'
    planner.assert_not_called()
    server._run_hf_download_task = AsyncMock(return_value={'complete': True})
    response = await server.hf_download(download_tests.FakeRequest(
        repo_id=expected['repository'], revision=revision, file=','.join(files)))
    assert response.status == 400
    server._run_hf_download_task.assert_not_called()
    assert server.hf_download_tasks == {}


@pytest.mark.parametrize('pipeline', PIPELINES)
@pytest.mark.parametrize('workflow', ('t2va', 'fl2va', 'ref2va'))
def test_app_download_and_plan_use_the_same_immutable_task_files(pipeline, workflow):
    asyncio.run(_accepted_selection(pipeline, workflow))


async def _accepted_selection(pipeline, workflow):
    expected = reviewed_joint_av_artifact_closure(pipeline, workflow)
    files = sorted(row['path'] for row in expected['files'])
    server = WebServer(modules={})
    server.loop = asyncio.get_running_loop()
    request = download_tests.FakeRequest(repo_id=expected['repository'], revision=expected['revision'], file=','.join(files))
    with patch('modiff.server.plan_hub_model_download', return_value=download_tests.HuggingFaceDownloadConcurrencyTests._space_plan()) as planner:
        response = await server.hf_download_plan(request)
    assert response.status == 200
    assert planner.call_args.args == (expected['repository'], files, expected['revision'])
    server._run_hf_download_task = AsyncMock(return_value={'complete': True})
    response = await server.hf_download(request)
    assert response.status == 200
    repo, entry = server._run_hf_download_task.call_args.args
    assert repo == expected['repository']
    assert entry['revision'] == expected['revision'] and entry['requested_files'] == files
