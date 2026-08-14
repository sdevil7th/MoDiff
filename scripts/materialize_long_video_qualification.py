#!/usr/bin/env python3
"""Bind and optionally submit the reviewed P6 long-video graph through MoDiff."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import sys
import urllib.error
import urllib.request
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modiff.path_identifiers import resolve_runtime_input_path  # noqa: E402


TEMPLATE_PATH = ROOT / "data" / "graphs" / "qualification" / "ltx-30-minute-continuation.template.json"
MODEL_REPO = "Lightricks/LTX-Video-0.9.8-13B-distilled"
MODEL_REVISION = "7c64400e1861cc0d7b98d570a1926d5408ec60cd"


def read_template(path: Path = TEMPLATE_PATH) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("The long-video qualification template must be a JSON object.")
    return value


def _opening_image_identifier(value: str) -> str:
    identifier = str(value or "").strip()
    if not identifier or identifier.startswith(("http://", "https://")) or "\\" in identifier:
        raise ValueError("Opening image ID must be one portable app-managed path identifier.")
    if identifier.startswith("/") or any(part == ".." for part in identifier.split("/")):
        raise ValueError("Opening image ID must stay relative to an app-managed input root.")
    return identifier


def materialize_graph(
    opening_image: str | os.PathLike[str],
    opening_image_id: str,
    *,
    sid: str,
    template_path: Path = TEMPLATE_PATH,
) -> tuple[dict, dict]:
    source = Path(opening_image).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Opening image does not exist: {source}")
    identifier = _opening_image_identifier(opening_image_id)
    resolved_identifier = resolve_runtime_input_path(identifier).resolve()
    if resolved_identifier != source:
        raise ValueError(
            "Opening image path and app-managed ID must resolve to the same file so the recovery hash "
            "matches the bytes the graph will load."
        )
    session_id = str(sid or "").strip()
    if not session_id or len(session_id) > 512:
        raise ValueError("A bounded app session ID is required.")

    graph = deepcopy(read_template(template_path))
    graph["sid"] = session_id
    graph["nodes"]["opening-image"]["params"]["file"]["value"] = [identifier]
    graph["runtimeHints"]["runInputHash"] = ""
    image_bytes = source.read_bytes()
    image_sha256 = hashlib.sha256(image_bytes).hexdigest()
    graph["qualification"].pop("openingImagePlaceholder", None)
    graph["qualification"].pop("runInputHashPlaceholder", None)
    graph["qualification"]["openingImage"] = {
        "id": identifier,
        "byteSize": len(image_bytes),
        "sha256": image_sha256,
    }
    canonical = json.dumps(graph, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    run_input_hash = hashlib.sha256(canonical + b"\0" + image_bytes).hexdigest()
    graph["runtimeHints"]["runInputHash"] = run_input_hash
    graph["qualification"]["runInputHash"] = run_input_hash
    receipt = {
        "schemaVersion": 1,
        "qualificationId": graph["qualification"]["id"],
        "openingImageId": identifier,
        "openingImageBytes": len(image_bytes),
        "openingImageSha256": image_sha256,
        "runInputHash": run_input_hash,
        "modelRepo": MODEL_REPO,
        "modelRevision": MODEL_REVISION,
        "jobCount": graph["qualification"]["expected"]["jobCount"],
        "targetSeconds": graph["qualification"]["expected"]["targetSeconds"],
    }
    return graph, receipt


def _loopback_base_url(value: str) -> str:
    normalized = str(value or "").strip().rstrip("/")
    parsed = urlparse(normalized)
    host = str(parsed.hostname or "").strip().strip("[]").lower()
    if parsed.scheme not in {"http", "https"} or not host or parsed.path not in {"", "/"}:
        raise ValueError("App URL must be an HTTP(S) loopback origin without a path.")
    if host != "localhost":
        try:
            if not ipaddress.ip_address(host).is_loopback:
                raise ValueError
        except ValueError as exc:
            raise ValueError("Long-video qualification may be submitted only to a loopback MoDiff app.") from exc
    return normalized


def _request_json(url: str, *, payload: dict | None = None):
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method="POST" if body is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            value = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"MoDiff returned HTTP {exc.code}: {detail}") from exc
    if not isinstance(value, (dict, list)):
        raise RuntimeError("MoDiff returned a malformed JSON response.")
    return value


def assert_model_ready(base_url: str) -> dict:
    cache = _request_json(f"{base_url}/hf_cache")
    if not isinstance(cache, list):
        raise RuntimeError("MoDiff returned a malformed model-cache response.")
    model = next((item for item in cache if isinstance(item, dict) and item.get("id") == MODEL_REPO), None)
    revisions = model.get("revisions") if isinstance(model, dict) else None
    revision_ready = any(
        isinstance(item, dict) and item.get("hash") == MODEL_REVISION for item in (revisions or [])
    )
    if (
        not isinstance(model, dict)
        or model.get("complete") is not True
        or model.get("repair_required") is True
        or not revision_ready
    ):
        raise RuntimeError(
            f"The exact {MODEL_REPO}@{MODEL_REVISION} snapshot is not complete in the app cache."
        )
    return {
        "complete": True,
        "repo": MODEL_REPO,
        "revision": MODEL_REVISION,
        "size": model.get("size"),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opening-image", required=True, help="Local file used only to bind the input hash.")
    parser.add_argument(
        "--opening-image-id",
        required=True,
        help="Portable app-managed path, for example images/qualification-opening.webp.",
    )
    parser.add_argument("--sid", required=True, help="Active app WebSocket session ID.")
    parser.add_argument("--output", type=Path, help="Write the materialized API graph to this path.")
    parser.add_argument("--submit", action="store_true", help="Queue the six-hour qualification through POST /graph.")
    parser.add_argument(
        "--consent-long-run",
        action="store_true",
        help="Required with --submit; confirms the remote qualification host is approved.",
    )
    parser.add_argument("--app-url", default="http://127.0.0.1:8088")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.submit and not args.consent_long_run:
        raise SystemExit("--submit requires --consent-long-run")
    if not args.output and not args.submit:
        raise SystemExit("Choose --output and/or --submit.")
    graph, receipt = materialize_graph(
        args.opening_image,
        args.opening_image_id,
        sid=args.sid,
    )
    if args.output:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(graph, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
        receipt["materializedGraph"] = str(destination)
    if args.submit:
        base_url = _loopback_base_url(args.app_url)
        receipt["modelCache"] = assert_model_ready(base_url)
        response = _request_json(f"{base_url}/graph", payload=graph)
        if not isinstance(response, dict) or response.get("error") is not False or not response.get("task_id"):
            raise RuntimeError(f"MoDiff did not accept the long-video graph: {response}")
        receipt["submission"] = {
            "taskId": response["task_id"],
            "queued": True,
            "through": "POST /graph",
        }
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
