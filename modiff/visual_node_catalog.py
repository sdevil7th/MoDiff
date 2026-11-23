"""Built-in stage discovery metadata. Search terms are never execution authority."""
from functools import lru_cache
import json
from pathlib import Path


@lru_cache(maxsize=1)
def visual_node_catalog():
    return json.loads(Path(__file__).with_name("visual_node_catalog.v1.json").read_text(encoding="utf-8"))
