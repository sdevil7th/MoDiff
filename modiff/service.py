"""Local service client. Uses the ordinary graph queue and persisted run outputs."""

from __future__ import annotations

import argparse
import http.client
import ipaddress
import json
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlsplit
import uuid


def request(server, path, body=None):
    from modiff.service_package import load_json
    from modiff.service_package import LIMIT, canonical

    url = urlsplit(server)
    try:
        local = ipaddress.ip_address(url.hostname or "").is_loopback
    except ValueError:
        local = False
    if (
        not local
        or url.scheme != "http"
        or url.path not in {"", "/"}
        or url.query
        or url.fragment
        or url.username
        or url.password
    ):
        raise ValueError("Use a loopback HTTP origin, such as http://127.0.0.1:8088.")
    connection = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=30)
    try:
        connection.request(
            "POST" if body is not None else "GET",
            path,
            body=canonical(body) if body is not None else None,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        raw = response.read(LIMIT + 1)
        value = load_json(raw)
        if response.status != 200 or not isinstance(value, dict) or value.get("error"):
            raise ValueError(
                value.get("message", f"HTTP {response.status}")
                if isinstance(value, dict)
                else "Invalid service response."
            )
        return value
    finally:
        connection.close()


def run(server, package, values, *, timeout=300):
    from modiff.service_package import service_outputs

    if not 0 < timeout <= 86400:
        raise ValueError("Timeout must be between 0 and 86400 seconds.")
    prepared = request(
        server,
        "/service_package",
        {"operation": "prepare", "package": package, "values": values, "sid": "service_" + uuid.uuid4().hex},
    )
    submitted = request(server, "/graph", prepared["graph"])
    task_id = submitted.get("task_id")
    if not isinstance(task_id, str) or not re.fullmatch(r"[\w-]{1,128}", task_id):
        raise ValueError("Graph submission returned no valid task ID; do not automatically resubmit.")
    print(f"Submitted task {task_id}", file=sys.stderr)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        queue = request(server, "/queue")
        task = next((item for item in queue.get("recent", []) if item.get("task_id") == task_id), None)
        if task is not None:
            if task.get("status") != "completed":
                raise ValueError(f"Run {task_id} ended with {task.get('status')}. Inspect /runs/{task_id}.")
            return service_outputs(package, request(server, f"/runs/{task_id}"), task_id)
        time.sleep(0.25)
    raise ValueError(
        f"Timed out waiting for {task_id}. The run remains queued/running; inspect /runs/{task_id} before retrying."
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "build", "run"))
    parser.add_argument("file", type=Path, help="API graph for inspect/build; service package for run")
    parser.add_argument("--interface", type=Path, help="Named bindings JSON for build")
    parser.add_argument("--inputs", type=Path, help="Named input values JSON for run")
    parser.add_argument("--server", default="http://127.0.0.1:8088")
    parser.add_argument("--output", type=Path, help="Write JSON to a new file, without overwriting existing files")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args(argv)
    try:
        from modiff.service_package import load_json
        from modiff.service_package import LIMIT

        def read(path):
            with path.open("rb") as handle:
                return load_json(handle.read(LIMIT + 1))

        document = read(args.file)
        if args.command == "run":
            result = run(args.server, document, read(args.inputs) if args.inputs else {}, timeout=args.timeout)
        else:
            body = {"operation": args.command, "graph": document}
            if args.command == "build":
                if not args.interface:
                    raise ValueError("build requires --interface with named inputs/outputs.")
                body["interface"] = read(args.interface)
            result = request(args.server, "/service_package", body)
            if args.command == "build":
                result = result["package"]
        encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(encoded)
        else:
            print(encoded, end="")
        return 0
    except (ValueError, OSError, KeyError, http.client.HTTPException) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
