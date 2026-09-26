from diffusers.modular_pipelines import ModularPipelineBlocks, InputParam, OutputParam


class PromptSuffix(ModularPipelineBlocks):
    @property
    def inputs(self):
        return [InputParam(name="text", type_hint=str, default="hello")]

    @property
    def intermediate_outputs(self):
        return [OutputParam(name="result", type_hint=str)]

    @property
    def description(self):
        return "A small custom Modular block with no model dependencies."

    def __call__(self, pipeline, state):
        state.set("result", state.get("text") + " — modular")
        return pipeline, state
