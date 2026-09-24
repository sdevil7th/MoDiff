from modiff.NodeBase import NodeBase


class PromptPrefix(NodeBase):
    """Add a reusable prefix to a prompt without loading any models."""

    label = "Prompt Prefix"
    category = "Text"
    resizable = True
    params = {
        "text": {
            "label": "Prompt",
            "type": "string",
            "display": "textarea",
            "default": "a quiet observatory",
        },
        "prompt_input": {
            "label": "Prompt Input",
            "type": "string",
            "display": "input",
            "required": False,
            "description": "Optional connected prompt. When connected, this replaces the inline Prompt value.",
        },
        "prefix": {
            "label": "Prefix",
            "type": "string",
            "display": "textarea",
            "default": "Watercolor:",
        },
        "result": {"label": "Prompt", "type": "string", "display": "output"},
    }

    def execute(self, text, prefix, prompt_input=None):
        prompt = prompt_input if prompt_input is not None else text
        return {"result": f"{prefix} {prompt}".strip()}
