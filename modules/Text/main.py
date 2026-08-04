# Derived from cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f; modified by MoDiff.
from modiff.NodeBase import NodeBase

class TextToList(NodeBase):
    """
    Convert a text string into a list of strings
    """

    label = "Text to List"
    category = "text"
    resizable = True
    params = {
        "text": {"type": "string", "default": "", "display": "text"},
        "separator": {
            "type": "string",
            "options": {
                "\n": "\\n",
                ",": ",",
                " ": "[space]",
                "|": "|",
                ";": ";",
                ":": ":",
            },
            "default": ","
        },
        "output": {"type": "string", "display": "output"},
    }

    def execute(self, **kwargs):
        text = kwargs["text"]
        separator = kwargs["separator"]
        return [item.strip() for item in text.split(separator)]
