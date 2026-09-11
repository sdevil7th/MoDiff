#!/usr/bin/env python3
"""Run a disposable backend for the historical Cluster browser lifecycle proof.

This helper never serves or mutates the source data directory. It selects one
mapping-ready, single-root saved workflow, copies its exact bytes into an empty
test data directory, verifies that the isolated migration scope is exactly one
blocked historical Cluster, and serves that copy on loopback.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREFERRED_MAPPING_IDS = (
    "legacy-cluster-compiler-mapping:transformers-ctc-stt:2026-09-02",
    "legacy-cluster-compiler-mapping:minimax-music3:2026-09-02",
    "legacy-cluster-compiler-mapping:wan-22-ti2v-5b:2026-09-02",
)


def _sha256(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _required_mapping(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    authority = candidate.get("historicalCompilerMappingAuthority")
    if not isinstance(authority, Mapping):
        raise ValueError("Candidate does not carry reviewed historical compiler-mapping authority.")
    mapping_id = authority.get("mappingId")
    if not isinstance(mapping_id, str) or not mapping_id:
        raise ValueError("Candidate historical compiler mapping is missing its mapping ID.")
    return authority


def _safe_source_file(source_root: Path, source_path: str) -> Path:
    relative = Path(source_path)
    if relative.is_absolute() or relative.parts[:1] != ("user-workflows",) or relative.suffix != ".json":
        raise ValueError(f"Unsafe historical workflow source path: {source_path!r}.")
    target = (source_root / relative).resolve(strict=True)
    try:
        target.relative_to(source_root)
    except ValueError as error:
        raise ValueError(f"Historical workflow source escapes its data directory: {source_path!r}.") from error
    if not target.is_file() or target.is_symlink():
        raise ValueError(f"Historical workflow source is not a regular file: {source_path!r}.")
    return target


def _single_root_workflow(raw: bytes, candidate: Mapping[str, Any]) -> Mapping[str, Any] | None:
    try:
        workflow = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(workflow, Mapping):
        return None
    snapshot = workflow.get("snapshot")
    if not isinstance(snapshot, Mapping):
        return None
    nodes = snapshot.get("nodes")
    edges = snapshot.get("edges")
    if not isinstance(nodes, list) or len(nodes) != 1 or not isinstance(edges, list) or edges:
        return None
    root = nodes[0]
    if not isinstance(root, Mapping) or root.get("id") != candidate.get("id") or root.get("type") != "cluster":
        return None
    data = root.get("data")
    if not isinstance(data, Mapping) or not isinstance(data.get("huggingFaceClusterInstance"), Mapping):
        return None
    workflow_id = workflow.get("id")
    if not isinstance(workflow_id, str) or not workflow_id or Path(candidate["sourcePath"]).stem != workflow_id:
        return None
    return workflow


def _choose_fixture(
    source_root: Path,
    preview: Mapping[str, Any],
) -> tuple[Mapping[str, Any], bytes, Mapping[str, Any]]:
    candidates = preview.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Migration preview did not expose a candidate list.")
    eligible: list[tuple[int, int, str, Mapping[str, Any], bytes, Mapping[str, Any]]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        if candidate.get("kind") != "legacy_registered_cluster_instance" or candidate.get("status") != "blocked":
            continue
        try:
            authority = _required_mapping(candidate)
            source_path = candidate.get("sourcePath")
            if not isinstance(source_path, str):
                continue
            source_file = _safe_source_file(source_root, source_path)
            raw = source_file.read_bytes()
            workflow = _single_root_workflow(raw, candidate)
            if workflow is None or _sha256(raw) != candidate.get("sourceSha256"):
                continue
            mapping_id = str(authority["mappingId"])
            try:
                preference = PREFERRED_MAPPING_IDS.index(mapping_id)
            except ValueError:
                preference = len(PREFERRED_MAPPING_IDS)
            eligible.append((preference, len(raw), source_path, candidate, raw, workflow))
        except (OSError, TypeError, ValueError):
            continue
    if not eligible:
        raise ValueError(
            "No reviewed mapping-ready single-root historical Cluster workflow is available in the source data."
        )
    _, _, _, candidate, raw, workflow = min(eligible, key=lambda item: item[:3])
    return candidate, raw, workflow


def _prepare_fixture(source_data_dir: Path, destination_data_dir: Path, metadata_path: Path) -> dict[str, Any]:
    from modiff.composite_migration import scan_composite_migration_preview

    source_root = source_data_dir.expanduser().resolve(strict=True)
    destination_root = destination_data_dir.expanduser().resolve(strict=False)
    metadata_target = metadata_path.expanduser().resolve(strict=False)
    if source_root == destination_root:
        raise ValueError("The disposable data directory must differ from the source data directory.")
    try:
        destination_root.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise ValueError("The disposable data directory must not be inside the source data directory.")
    if destination_root.exists() and any(destination_root.iterdir()):
        raise ValueError("The disposable data directory must be empty.")
    destination_root.mkdir(parents=True, exist_ok=True)
    try:
        metadata_target.relative_to(destination_root)
    except ValueError as error:
        raise ValueError("Fixture metadata must be written inside the disposable data directory.") from error

    source_preview = scan_composite_migration_preview(source_root)
    candidate, raw, workflow = _choose_fixture(source_root, source_preview)
    source_path = str(candidate["sourcePath"])
    destination_file = destination_root.joinpath(*Path(source_path).parts)
    destination_file.parent.mkdir(parents=True, exist_ok=True)
    destination_file.write_bytes(raw)

    isolated_preview = scan_composite_migration_preview(destination_root)
    summary = isolated_preview.get("summary")
    isolated_candidates = isolated_preview.get("candidates")
    if not isinstance(summary, Mapping) or not isinstance(isolated_candidates, list):
        raise ValueError("The isolated migration preview is malformed.")
    if (
        summary.get("sourceCount") != 1
        or summary.get("targetFileCount") != 0
        or summary.get("convertibleCandidateCount") != 0
        or summary.get("blockedCandidateCount") != 1
        or summary.get("legacyClusterBlockedCount") != 1
        or len(isolated_candidates) != 1
    ):
        raise ValueError("The disposable migration scope is not exactly one blocked historical Cluster workflow.")
    isolated = isolated_candidates[0]
    if not isinstance(isolated, Mapping) or isolated.get("id") != candidate.get("id"):
        raise ValueError("The isolated historical Cluster identity changed while preparing the fixture.")
    authority = _required_mapping(isolated)
    original_sha256 = _sha256(raw)
    if original_sha256 != isolated.get("sourceSha256"):
        raise ValueError("The copied historical workflow bytes do not match the backend preview receipt.")

    metadata = {
        "schemaVersion": 1,
        "kind": "isolated_historical_cluster_migration_fixture",
        "sourcePath": source_path,
        "workflowId": workflow["id"],
        "instanceId": isolated["id"],
        "mappingId": authority["mappingId"],
        "mappingHash": authority["mappingHash"],
        "originalSha256": original_sha256,
        "originalByteLength": len(raw),
    }
    metadata_target.parent.mkdir(parents=True, exist_ok=True)
    metadata_target.write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return metadata


async def _serve(args: argparse.Namespace) -> None:
    from modiff.backend_source_identity import capture_process_backend_source_identity

    capture_process_backend_source_identity()
    from modiff.server import MODULE_MAP, WebServer

    metadata = _prepare_fixture(Path(args.source_data_dir), Path(args.data_dir), Path(args.metadata))
    server = WebServer(
        MODULE_MAP,
        host="127.0.0.1",
        port=args.port,
        secure=False,
        cors=False,
        work_dir=str(PROJECT_ROOT),
        data_dir=str(Path(args.data_dir).resolve()),
    )
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signal_number in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signal_number, stopped.set)
        except NotImplementedError:
            pass
    await server.run()
    print(
        json.dumps(
            {
                "event": "isolated_historical_migration_backend_ready",
                "port": args.port,
                "workflowId": metadata["workflowId"],
                "mappingId": metadata["mappingId"],
                "originalSha256": metadata["originalSha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        await stopped.wait()
    finally:
        await server.cleanup()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data-dir", required=True, help="Read-only MoDiff data directory to select from.")
    parser.add_argument("--data-dir", required=True, help="Empty disposable data directory served by this process.")
    parser.add_argument("--metadata", required=True, help="Metadata JSON path inside the disposable data directory.")
    parser.add_argument("--port", required=True, type=int)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not 1024 <= args.port <= 65535:
        raise SystemExit("--port must be between 1024 and 65535.")
    os.chdir(PROJECT_ROOT)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    asyncio.run(_serve(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
