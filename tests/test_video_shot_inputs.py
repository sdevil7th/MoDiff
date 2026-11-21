"""A storyboard supplies ordinary explicit shots, never a model executor."""
import json

import pytest


def select(records, **values):
    from modules.VideoConditioning.main import ShotInputs
    return ShotInputs('shot-inputs-unit')(storyboard=json.dumps(records), **values)


def test_explicit_shots_keep_record_seeds_and_shared_defaults_including_zero():
    records = [{'prompt': 'first', 'seed': 0}, {'prompt': 'second'}, 'third']
    outputs = [select(records, index=index, base_seed=8511, width=832, height=480,
                      num_frames=81, fps=16., steps=50, guidance_scale=4.5) for index in range(3)]
    assert [row['prompt'] for row in outputs] == ['first', 'second', 'third']
    assert [row['seed'] for row in outputs] == [0, 8512, 8513]
    assert all(row['frames'] == 81 and row['steps_out'] == 50 and row['width_out'] == 832 for row in outputs)


def test_quality_shot_values_match_existing_job_preparation_without_requiring_other_images():
    from modules.DiffusersVideo.main import BuildShotJobs
    records = [{'prompt': 'dispatch', 'duration_seconds': 5.4},
               {'prompt': 'arrival', 'duration_seconds': 5.4, 'seed': 0, 'conditioning_strength': .7}]
    settings = dict(base_seed=42017, fps=15, width=832, height=480, steps=50, guidance_scale=6.)
    previous = BuildShotJobs('old-job-reference')(shots=records, mode='text_to_video', **settings)['jobs']
    for index, expected in enumerate(previous):
        actual = select(records, index=index, **settings)
        for output, field in [('prompt', 'prompt'), ('seed', 'seed'), ('frames', 'num_frames'),
                              ('width_out', 'width'), ('height_out', 'height'), ('fps_out', 'fps'),
                              ('steps_out', 'steps'), ('guidance', 'guidance_scale'),
                              ('condition_strength', 'conditioning_strength')]:
            assert actual[output] == expected[field]


def test_nine_scene_selection_is_explicit_and_not_the_legacy_sequence_six_shot_limit():
    records = [{'prompt': f'scene {index}', 'seed': 1201047366} for index in range(9)]
    assert select(records, index=8)['prompt'] == 'scene 8'


def test_storyboard_has_one_shared_seed_and_preserves_exact_authored_json():
    from modules.VideoConditioning.main import Storyboard
    text = '[  "first", {"prompt": "second", "seed": 0} ]'
    result = Storyboard('storyboard-unit')(shots_json=text, base_seed=0)
    assert result == {'storyboard': text, 'seed': 0, 'count': 2}
    assert [select(json.loads(result['storyboard']), base_seed=result['seed'], index=index)['seed']
            for index in range(2)] == [0, 0]


def test_text_selection_and_generation_settings_are_independent_acyclic_suppliers():
    from modules.VideoConditioning.main import ShotPrompt
    text = '[{"prompt":"First", "seed":7},{"prompt":"Second", "seed":8}]'
    assert ShotPrompt('shot-text-unit')(storyboard=text, index=1) == {'prompt': 'Second'}
    assert ShotPrompt('shot-text-override')(storyboard=text, index=1, prompt_override='Changed') == {'prompt': 'Changed'}
    assert select(json.loads(text), index=1)['seed'] == 8


def test_selector_defaults_do_not_impersonate_consumed_model_controls():
    from modules.VideoConditioning.main import ShotInputs
    node = ShotInputs('shot-evidence-unit')
    node(storyboard='["scene"]', width=832, steps=50, guidance_scale=6.)
    assert node._execution_input_record['fields'] == {}


@pytest.mark.parametrize(('records', 'values'), [
    ([], {}), (['one'], {'index': -1}), (['one'], {'index': 1}),
    ([{'prompt': ''}], {}), ([{'prompt': 'one', 'seed': True}], {}),
    ([{'prompt': 'one', 'duration_seconds': float('nan')}], {}),
    (['one'], {'width': 0}), (['one'], {'fps': float('inf')}),
])
def test_invalid_storyboard_or_controls_fail_before_any_model_work(records, values):
    with pytest.raises(RuntimeError):
        select(records, **values)


def test_shot_inputs_are_discovered_with_the_same_typed_outputs_as_dispatch():
    from modules import MODULE_MAP
    declaration = MODULE_MAP['modules.VideoConditioning']['ShotInputs']['params']
    for name, kind in [('prompt', 'text'), ('seed', 'int'), ('frames', 'int'),
                       ('width_out', 'int'), ('height_out', 'int'), ('guidance', 'float')]:
        assert declaration[name]['display'] == 'output' and declaration[name]['type'] == kind
