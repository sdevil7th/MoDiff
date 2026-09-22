from modiff.authoring_examples import seed_operation_example


def draft(task="text_to_image", pipeline="FuturePipeline", **field):
    return {
        "operation": {"task": task, "pipelineClass": pipeline},
        "params": {"prompt": {"type": "text", "default": "", **field}},
        "values": {},
    }


def test_new_text_and_edit_operations_have_different_meaningful_examples():
    text, edit = draft(), draft("edit_image")
    seed_operation_example(text)
    seed_operation_example(edit)
    assert text["values"]["prompt"]
    assert edit["values"]["prompt"] != text["values"]["prompt"]
    assert text["params"]["prompt"]["fieldOptions"]["exampleAttribution"] == "MoDiff task example"


def test_qwen21_example_retains_creator_source_without_changing_numeric_defaults():
    node = draft(pipeline="QwenImage21Pipeline")
    node["params"]["num_inference_steps"] = {"value": 40}
    seed_operation_example(node)
    assert node["params"]["prompt"]["fieldOptions"]["exampleSource"] == "https://github.com/QwenLM/Qwen-Image-2.1"
    assert node["params"]["num_inference_steps"]["value"] == 40


def test_existing_hidden_output_and_unrelated_prompts_are_preserved():
    for fields in ({"value": "Authored prompt"}, {"hidden": True}, {"display": "output"}):
        node = draft(**fields)
        seed_operation_example(node)
        assert not node["values"]
    node = draft("text_generation")
    seed_operation_example(node)
    assert not node["values"]


def test_profile_examples_replace_only_generated_defaults_and_keep_accurate_attribution():
    node = draft()
    seed_operation_example(node)
    generic = node["values"]["prompt"]
    seed_operation_example(node, "black-forest-labs/FLUX.1-schnell")
    assert node["values"]["prompt"] != generic
    assert node["params"]["prompt"]["fieldOptions"]["exampleSource"].endswith("FLUX.1-schnell")
    seed_operation_example(node)
    assert "cat" in node["values"]["prompt"]
    seed_operation_example(node, "example/other-model")
    assert node["values"]["prompt"] == generic
    assert "exampleSource" not in node["params"]["prompt"]["fieldOptions"]
    reference = draft("multi_image_reference_edit", "QwenImage21Pipeline")
    seed_operation_example(reference)
    assert reference["params"]["prompt"]["fieldOptions"]["exampleAttribution"] == "MoDiff task example"
