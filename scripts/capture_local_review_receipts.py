#!/usr/bin/env python3
"""Capture completed app tasks and local output hashes for human review."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import urllib.error
import urllib.request

from modiff.local_review_receipts import (
    build_local_review_ledger,
    loopback_base_url,
    merge_local_review_ledger,
    parse_record,
    validate_local_review_ledger,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "qualification" / "local-review" / "technical-candidates.v1.json"


def _request_run(base_url: str, task_id: str) -> dict:
    request = urllib.request.Request(f"{base_url}/runs/{task_id}", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            value = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        raise RuntimeError(f"MoDiff returned HTTP {error.code} for task {task_id}: {detail}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"MoDiff returned a malformed run response for task {task_id}.")
    return value


def _output_path(value: str | None) -> Path:
    destination = Path(value).expanduser() if value else DEFAULT_OUTPUT
    destination = destination.resolve(strict=False)
    qualification_root = (ROOT / "data" / "qualification").resolve()
    try:
        destination.relative_to(qualification_root)
    except ValueError as error:
        raise ValueError("Local review receipt output must stay under data/qualification.") from error
    if destination.suffix.lower() != ".json":
        raise ValueError("Local review receipt output must use the .json suffix.")
    return destination


def _write_atomic(destination: Path, document: dict) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8088")
    parser.add_argument(
        "--record",
        action="append",
        default=[],
        metavar="WORKFLOW|TASK|OUTPUTS",
        help="Repeat for each completed candidate; OUTPUTS is a comma-separated app-data path list.",
    )
    parser.add_argument("--output", help="JSON receipt path under data/qualification.")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Validate the existing ledger and append only newly fetched task receipts.",
    )
    parser.add_argument("--validate", action="store_true", help="Validate an existing receipt instead of capturing.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    destination = _output_path(args.output)
    if args.validate:
        if args.record or args.merge:
            raise SystemExit("--validate does not accept --record or --merge.")
        payload = json.loads(destination.read_text(encoding="utf-8"))
        validate_local_review_ledger(payload, root=ROOT)
        print(
            f"Validated {payload['summary']['receiptCount']} local review receipts / "
            f"{payload['summary']['outputCount']} outputs."
        )
        return 0
    if not args.record:
        raise SystemExit("Capture requires at least one --record.")
    server = loopback_base_url(args.server)
    records = [parse_record(value) for value in args.record]

    def fetch_run(task_id):
        return _request_run(server, task_id)

    if args.merge:
        if not destination.is_file() or destination.is_symlink():
            raise SystemExit("--merge requires an existing regular receipt ledger.")
        existing = json.loads(destination.read_text(encoding="utf-8"))
        payload = merge_local_review_ledger(
            root=ROOT,
            existing=existing,
            records=records,
            fetch_run=fetch_run,
        )
    else:
        payload = build_local_review_ledger(root=ROOT, records=records, fetch_run=fetch_run)
    validate_local_review_ledger(payload, root=ROOT)
    _write_atomic(destination, payload)
    print(
        f"{'Merged' if args.merge else 'Wrote'} {destination}: "
        f"{payload['summary']['receiptCount']} receipts / "
        f"{payload['summary']['outputCount']} outputs pending human review."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
