from modiff.NodeBase import NodeBase


class PromptPrefix(NodeBase):
    """Add a reusable prefix to a prompt without loading any models."""

    label = "Prompt Prefix"
    category = "Text"
    params = {
        "text": {"label": "Prompt", "type": "string", "isInput": True, "default": "a quiet observatory"},
        "prefix": {"label": "Prefix", "type": "string", "default": "Watercolor:"},
        "result": {"label": "Prompt", "type": "string", "display": "output"},
    }

    def execute(self, text, prefix):
        return {"result": f"{prefix} {text}".strip()}
