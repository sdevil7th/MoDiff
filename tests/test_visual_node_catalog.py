import asyncio
import json
from collections import Counter
from types import SimpleNamespace

from modiff.server import WebServer
from modiff.visual_node_catalog import visual_node_catalog
from modules import MODULE_MAP


def test_catalog_covers_every_builtin_and_all_selectable_presets():
    catalog = visual_node_catalog()
    schemas = {
        f"{module}.{action}": schema
        for module, actions in MODULE_MAP.items() if module.startswith("modules.")
        for action, schema in actions.items()
    }
    assert catalog["version"] == 1
    assert Counter(stage["tier"] for stage in catalog["stages"]) == {"browse": 15, "task": 14, "advanced": 6}
    assert len({s["id"] for s in catalog["stages"]}) == len({s["label"] for s in catalog["stages"]}) == 35
    covered = {key for group in catalog["otherActions"].values() for key in group}
    standalone = set()
    for stage in catalog["stages"]:
        assert stage["summary"] and stage["modes"]
        concrete = {mode["node"] for mode in stage["modes"]}
        assert not standalone.intersection(concrete), "Every standalone action needs one canonical stage"
        standalone.update(concrete)
        guidance = {mode["guidance"]: mode for mode in stage["modes"] if "guidance" in mode}
        if guidance:
            assert set(guidance) == {"plain", "guided"}
            assert "control_image" not in schemas[guidance["plain"]["node"]]["params"]
            assert schemas[guidance["guided"]["node"]]["params"]["control_image"]["required"]
        covered.update(stage.get("members", []))
        for mode in stage["modes"]:
            covered.add(mode["node"])
            schema = schemas[mode["node"]]
            assert mode["label"] and not schema.get("hidden")
            for field, value in mode.get("settings", {}).items():
                options = schema["params"][field]["options"]
                assert value in [o.get("value") if isinstance(o, dict) else o for o in options]
    assert covered == set(schemas)


def test_registry_advertises_discovery_separately_from_executable_node():
    server = WebServer(modules={"modules.Example": {"Example": {"params": {}}}}, work_dir=".", data_dir="data")
    def fetch(key):
        return json.loads(asyncio.run(server.nodes(SimpleNamespace(match_info={"id": key}))).body)
    complete = fetch("")
    individual = fetch("modules.Example/Example")
    assert complete["visualNodeCatalog"] == visual_node_catalog()
    assert complete["nodes"] == individual["nodes"]
    assert "visualNodeCatalog" not in individual
