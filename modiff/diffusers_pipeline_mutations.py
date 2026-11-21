"""Owned mutation policies for existing direct Diffusers pipeline nodes.

Inspection follows declared pipeline ports and literal controls only. These
process-local cache markers grant neither Auto admission nor memory credit.
"""

from copy import deepcopy
import hashlib
import json
import weakref


_PIPELINE_TYPES = {
    "modules.DiffusersImage": "image_diffusion_pipeline",
    "modules.DiffusersAudio": "audio_diffusion_pipeline",
}
_MUTATIONS = {
    ("modules.DiffusersImage", "LoadAdapter"),
    ("modules.DiffusersAudio", "LoadAdapter"),
    ("modules.DiffusersAudio", "SetAdapters"),
    ("modules.DiffusersAudio", "FuseAdapters"),
}
# This existing built-in returns its input pipeline after configuring it. Its
# runtime changes are not a reusable mutation contract: retain execution with a
# cold owner rather than silently adopting or rejecting the ordinary graph.
_PASSTHROUGHS = {("modules.DiffusersRuntime", "ApplyPipelineRuntimeConfig"): "configured_pipeline"}
_OWNER_IGNORED_INPUTS = {
    # Match the declared loaders' NodeBase cache exclusions. These configure the
    # semantic output contract, rather than from_pretrained construction.
    "modules.DiffusersImage": {"mode", "execution_profile_id"},
    "modules.DiffusersAudio": {"mode", "audio_contract"},
}


def _lora_receipt(node_id, node, controls):
    from modiff.auxiliary_lora import controlled_lora_receipts_from_graph

    literal_node = {**node, "params": {field: {"value": value} for field, value in controls.items()}}
    receipts = controlled_lora_receipts_from_graph({"nodes": {node_id: literal_node}, "paths": [[node_id]]})
    return receipts[0] if receipts else None


def _contract(node):
    return node.get("module"), node.get("action")


def _controls(node, schema):
    values, reusable = {}, True
    for key, definition in schema.items():
        if (
            key == "pipeline"
            or definition.get("display", "").startswith("ui_")
            or definition.get("display") == "output"
        ):
            continue
        param = node.get("params", {}).get(key, {})
        if param.get("sourceId"):
            # Do not execute even a pure supplier or borrow its old output.
            values[key] = {"sourceId": param["sourceId"], "sourceKey": param.get("sourceKey")}
            reusable = False
        else:
            value = param.get("value")
            values[key] = definition.get("default") if value is None else value
    return values, reusable


def inspect_direct_pipeline_mutations(graph, registry):
    """Return exact owner/ordered-chain policies; reject conflicting branches.

    Multiple consumers of one terminal policy are valid. A vanilla bypass or
    different mutation chain cannot share the same mutable pipeline owner.
    """
    nodes = graph["nodes"]
    order = list(dict.fromkeys(key for path in graph["paths"] for key in path))
    executable = set(order)
    owners = {
        key: nodes[key]
        for key in order
        if nodes[key].get("module") in _PIPELINE_TYPES and nodes[key].get("action") == "LoadPipeline"
    }
    if not owners:
        return {}
    policies = {}
    for key, node in owners.items():
        schema = registry.get(node["module"], {}).get(node["action"], {}).get("params", {})
        controls, reusable = _controls(node, schema)
        for field in _OWNER_IGNORED_INPUTS[node["module"]]:
            controls.pop(field, None)
        policies[key] = {"chain": [], "reusable": reusable, "receipts": [], "loading": controls}
    traced, visiting, mutations = {}, set(), {}

    def canonical_source(param):
        key, port = param.get("sourceId"), param.get("sourceKey")
        if key not in executable:
            return False
        return (
            (key in owners and port == "pipeline")
            or (_contract(nodes[key]) in _MUTATIONS and port == "output")
            or (_contract(nodes[key]) in _PASSTHROUGHS and _PASSTHROUGHS[_contract(nodes[key])] == port)
        )

    def trace(key, port):
        pair = key, port
        if pair in traced:
            return traced[pair]
        if pair in visiting:
            raise ValueError("A direct pipeline mutation chain must not contain a cycle.")
        if key not in executable:
            raise ValueError("A direct pipeline connection needs an executable upstream node.")
        visiting.add(pair)
        node = nodes[key]
        contract = _contract(node)
        result = None
        if key in owners and port == "pipeline":
            result = key, ()
        elif contract in _MUTATIONS and port == "output":
            param = node.get("params", {}).get("pipeline", {})
            if not param.get("sourceId") or not param.get("sourceKey"):
                raise ValueError("A direct pipeline mutation needs its declared pipeline connection.")
            result = trace(param["sourceId"], param["sourceKey"])
            if result is not None:
                owner, chain = result
                if owners[owner]["module"] != node["module"]:
                    raise ValueError("A direct pipeline mutation must use its matching pipeline owner.")
                schema = registry[node["module"]][node["action"]]["params"]
                controls, reusable = _controls(node, schema)
                receipt = None
                if node["action"] == "LoadAdapter" and reusable:
                    receipt = _lora_receipt(key, node, controls)
                    if receipt is None:  # Existing empty image adapter is a no-op.
                        traced[pair] = result
                        visiting.remove(pair)
                        return result
                mutations[key] = {
                    "nodeId": key,
                    "module": node["module"],
                    "action": node["action"],
                    "source": [param["sourceId"], param["sourceKey"]],
                    "controls": controls,
                    "receipt": receipt,
                }
                policies[owner]["reusable"] &= reusable
                if receipt is not None:
                    policies[owner]["receipts"].append(receipt)
                result = owner, (*chain, key)
        elif contract in _PASSTHROUGHS and port == _PASSTHROUGHS[contract]:
            param = node.get("params", {}).get("pipeline", {})
            if not param.get("sourceId") or not param.get("sourceKey"):
                raise ValueError("A configured direct pipeline needs its declared pipeline connection.")
            result = trace(param["sourceId"], param["sourceKey"])
            if result is not None:
                policies[result[0]]["reusable"] = False
        elif key in owners or contract in _MUTATIONS or contract in _PASSTHROUGHS:
            raise ValueError("A direct pipeline connection must use its declared output port.")
        else:
            for field, param in node.get("params", {}).items():
                if (field == "pipeline" and param.get("sourceId")) or canonical_source(param):
                    if trace(param["sourceId"], param.get("sourceKey")) is not None:
                        raise ValueError(
                            "An undeclared direct pipeline passthrough needs an independent pipeline owner."
                        )
        traced[pair] = result
        visiting.remove(pair)
        return result

    terminal = {owner: set() for owner in owners}
    consumed = set()
    for key in order:
        node = nodes[key]
        schema = registry.get(node.get("module"), {}).get(node.get("action"), {}).get("params", {})
        for field, param in node.get("params", {}).items():
            declared = field == "pipeline" and schema.get(field, {}).get("type") in _PIPELINE_TYPES.values()
            if not param.get("sourceId") or not (declared or canonical_source(param)):
                continue
            result = trace(param["sourceId"], param.get("sourceKey"))
            if result is None:
                # Do not infer ownership from arbitrary custom Python references.
                if _contract(node) in _MUTATIONS:
                    raise ValueError("A direct pipeline mutation needs an independent declared pipeline owner.")
                continue
            owner, chain = result
            consumed.add(param["sourceId"])
            if field == "pipeline" and _contract(node) in _MUTATIONS:
                trace(key, "output")
            elif field == "pipeline" and _contract(node) in _PASSTHROUGHS:
                trace(key, _PASSTHROUGHS[_contract(node)])
            else:
                terminal[owner].add(chain)
            if not declared or node.get("module") not in _PIPELINE_TYPES:
                policies[owner]["reusable"] = False
    # Even an orphan mutation executes and leaves its owner's live state changed.
    leaves = {
        key for key in order if _contract(nodes[key]) in _MUTATIONS or _contract(nodes[key]) in _PASSTHROUGHS
    } - consumed
    for key in leaves:
        result = trace(key, _PASSTHROUGHS.get(_contract(nodes[key]), "output"))
        if result is not None:
            owner, chain = result
            terminal[owner].add(chain)
    for owner, policy in policies.items():
        chains = terminal[owner] or {()}
        if len(chains) != 1:
            raise ValueError(
                "Conflicting direct pipeline policies need independent pipeline loaders; connect shared consumers to one terminal mutation chain."
            )
        policy["chain"] = list(next(iter(chains)))
        if any(set(policy["chain"]).intersection(loop.get("bodyNodeIds", [])) for loop in graph.get("loops", [])):
            raise ValueError(
                "Direct pipeline mutation loops need a separately reviewed iteration policy; keep the mutation chain outside the loop."
            )
        policy["steps"] = [mutations[key] for key in policy["chain"]]
        identity = {
            "ownerId": owner,
            "contract": _contract(owners[owner]),
            "loading": policy["loading"],
            "chain": policy["steps"],
        }
        policy["identity"] = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return policies


def cleanup_hints_without_owned_mutations(hints, policies):
    """Only scoped cleanup comparison changes; admission receipts stay complete."""
    hints = deepcopy(hints)
    if isinstance(hints, dict):
        managed = [receipt for policy in policies.values() for receipt in policy["receipts"]]
        hints["controlledArtifacts"] = [
            receipt for receipt in hints.get("controlledArtifacts", []) if receipt not in managed
        ]
    return hints


def _binding(node, port):
    if not getattr(node, "_cache_valid", False) or getattr(node, "_cache_invalidated", False):
        return None
    output = getattr(node, "output", None)
    if not isinstance(output, dict) or output.get(port) is None:
        return None
    try:
        from utils.memory_menager import memory_manager

        models = tuple((key, id(memory_manager.get_model(key))) for key in node._mm_models)
        if any(memory_manager.get_model(key) is None for key in node._mm_models):
            return None
        return {
            "node": weakref.ref(node),
            "output": id(output),
            "pipeline": id(output[port]),
            "snapshot": node._cache_input_snapshot,
            "implementation": node._cache_implementation,
            "models": models,
        }
    except (AttributeError, TypeError):
        return None


def _matches(binding, node, port):
    if not binding or binding["node"]() is not node or _binding(node, port) != binding:
        return False
    from modiff.node_cache_identity import implementation_identity

    return implementation_identity(node) == binding["implementation"]


def changed_mutation_owners(server, policies):
    records = getattr(server, "_direct_pipeline_mutation_records", {})
    for owner, record in list(records.items()):
        binding = record.get("loader")
        if not binding or binding["node"]() is not server.node_cache.get(owner):
            records.pop(owner, None)
    changed = []
    for owner, policy in policies.items():
        cached = server.node_cache.get(owner)
        if cached is None:
            continue
        record = records.get(owner, {})
        valid = policy["reusable"] and record.get("complete") and record.get("identity") == policy["identity"]
        valid = valid and _matches(record.get("loader"), cached, "pipeline")
        valid = valid and all(
            _matches(record.get("mutations", {}).get(key), server.node_cache.get(key), "output")
            for key in policy["chain"]
        )
        if not valid:
            changed.append(owner)
    return changed


def begin_mutation(server, node_id):
    for owner, policy in getattr(server, "_direct_pipeline_mutation_policies", {}).items():
        if node_id in policy["chain"]:
            record = getattr(server, "_direct_pipeline_mutation_records", {}).get(owner)
            if record is not None:
                record["complete"] = False
                if node_id == policy["chain"][0]:
                    record["mutations"] = {}


def record_mutation_output(server, node_id):
    policies = getattr(server, "_direct_pipeline_mutation_policies", {})
    records = getattr(server, "_direct_pipeline_mutation_records", None)
    if records is None:
        server._direct_pipeline_mutation_records = records = {}
    for owner, policy in policies.items():
        if node_id != owner and node_id not in policy["chain"]:
            continue
        cached = server.node_cache.get(owner)
        binding = _binding(cached, "pipeline")
        record = records.get(owner)
        if (
            record is None
            or record.get("identity") != policy["identity"]
            or not _matches(record.get("loader"), cached, "pipeline")
        ):
            record = records[owner] = {
                "identity": policy["identity"],
                "loader": binding,
                "mutations": {},
                "complete": False,
            }
        if node_id in policy["chain"]:
            mutated = _binding(server.node_cache.get(node_id), "output")
            if binding is None or mutated is None or mutated["pipeline"] != binding["pipeline"]:
                record["complete"] = False
                return
            record["mutations"][node_id] = mutated
        complete = bool(binding) and all(key in record["mutations"] for key in policy["chain"])
        if complete and policy["chain"] and node_id == policy["chain"][-1]:
            # Literal artifact identity must still agree after dispatch. This is
            # not an attestation of adversarial file changes during upstream I/O.
            for step in policy["steps"]:
                if (
                    step["receipt"] is not None
                    and _lora_receipt(step["nodeId"], step, step["controls"]) != step["receipt"]
                ):
                    record["complete"] = False
                    raise ValueError(
                        "A direct pipeline adapter artifact changed during mutation; reselect it and run again."
                    )
        record["complete"] = complete
