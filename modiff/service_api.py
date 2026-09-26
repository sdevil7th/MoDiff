"""Service export/validation only. Submission still uses POST /graph."""

import asyncio

from aiohttp import web
from modiff import service_package as service


load_json = service.load_json


class ServiceAPI:
    def _service_contract(self, graph):
        from modiff.backend_source_identity import backend_source_identity

        if backend_source_identity()["fingerprint"] != self.backend_source_identity["fingerprint"]:
            raise ValueError("Backend source changed since startup. Restart before service export/execution.")
        return service.runtime_contract(graph, self.backend_source_identity, self._extension_store())

    def _validate_service_graph(self, graph):
        envelope = graph.get("servicePackage")
        if "servicePackage" not in graph:
            return
        if not isinstance(envelope, dict) or set(envelope) != {"package", "values"}:
            raise ValueError("Invalid service package execution envelope.")
        expected = service.prepare_package(
            envelope["package"],
            envelope["values"],
            registry=self.modules,
            contract=self._service_contract(graph),
            sid=graph.get("sid"),
        )
        if (
            service.graph_material(expected) != service.graph_material(graph)
            or expected.get("deterministicMode") != graph.get("deterministicMode")
            or self._coerce_runtime_hints(expected.get("runtimeHints"))
            != self._coerce_runtime_hints(graph.get("runtimeHints"))
        ):
            raise ValueError("Prepared service graph was modified. Prepare the package again.")

    async def service_package(self, request):
        try:
            data = bytearray()
            async for chunk in request.content.iter_chunked(65536):
                data.extend(chunk)
                if len(data) > service.LIMIT:
                    raise ValueError("Service request exceeds 8 MiB.")
            body = load_json(data)
            if not isinstance(body, dict):
                raise ValueError("Service request must be an object.")
            operation = body.get("operation")
            keys = {
                "inspect": {"operation", "graph"},
                "build": {"operation", "graph", "interface"},
                "prepare": {"operation", "package", "values", "sid"},
            }
            if not isinstance(operation, str) or operation not in keys or set(body) != keys[operation]:
                raise ValueError("Use inspect, build or prepare with their declared fields.")

            def perform():
                package = body.get("package")
                graph = (
                    body.get("graph")
                    if operation != "prepare"
                    else package.get("graph")
                    if isinstance(package, dict)
                    else None
                )
                if not isinstance(graph, dict):
                    raise ValueError("Service request needs an API graph.")
                candidates = service.inspect_graph(graph, self.modules)
                if operation == "inspect":
                    return {"error": False, **candidates}
                contract = self._service_contract(graph)
                if operation == "build":
                    package = service.build_package(graph, body["interface"], registry=self.modules, contract=contract)
                    return {"error": False, "package": package}
                graph = service.prepare_package(
                    body["package"], body["values"], registry=self.modules, contract=contract, sid=body["sid"]
                )
                return {"error": False, "graph": graph}

            return web.json_response(await asyncio.to_thread(perform))
        except (ValueError, TypeError, KeyError, OSError) as error:
            return web.json_response({"error": True, "message": str(error)[:2048]}, status=400)
