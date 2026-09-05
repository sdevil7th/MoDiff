from __future__ import annotations

import json
import ipaddress
import logging
import os
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from subprocess import Popen
from typing import Any
from urllib.parse import urlparse


logger = logging.getLogger("modiff")


def _loopback_host(host: str | None) -> bool:
    normalized = str(host or "").strip().strip("[]").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _allowed_browser_origin(origin: str | None) -> bool:
    """The privileged supervisor is callable only by local browser origins."""

    if not origin:
        return True
    try:
        parsed = urlparse(origin)
        host = str(parsed.hostname or "").strip().strip("[]").lower()
    except (TypeError, ValueError):
        return False
    if parsed.scheme not in {"http", "https"} or not host:
        return False
    return _loopback_host(host)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def compact_task_history(tasks: Any) -> list[dict[str, Any]]:
    """Return polling-safe task summaries while retaining lazy detail lookup.

    Full workflow snapshots remain in the worker's task history and are served
    by ``/runs/{task_id}``. Repeating every completed graph in each `/queue`
    poll makes refresh requests unnecessarily large and especially harmful on
    constrained connections.
    """
    if not isinstance(tasks, list):
        return []
    compacted = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        summary = {key: value for key, value in task.items() if key != "workflow_snapshot"}
        runtime_fingerprint = summary.get("runtimeFingerprint")
        if isinstance(runtime_fingerprint, dict):
            summary["runtimeFingerprint"] = runtime_fingerprint.get("resourceFingerprint") or runtime_fingerprint.get(
                "fingerprint"
            )
        summary["has_workflow_snapshot"] = isinstance(task.get("workflow_snapshot"), dict)
        compacted.append(summary)
    return compacted


class SupervisorController:
    """Small process-external control plane for an unresponsive model worker."""

    def __init__(self, queue_state_path: str | os.PathLike[str]):
        self.queue_state_path = Path(queue_state_path)
        self._lock = threading.RLock()
        self._worker: Popen[Any] | None = None
        self._restart_requested = False
        self._shutting_down = False

    def set_worker(self, worker: Popen[Any] | None) -> None:
        with self._lock:
            self._worker = worker
            if worker is not None:
                self._restart_requested = False

    def set_shutting_down(self) -> None:
        with self._lock:
            self._shutting_down = True

    def consume_restart_request(self) -> bool:
        with self._lock:
            requested = self._restart_requested
            self._restart_requested = False
            return requested

    def reconcile_interrupted_worker(
        self,
        *,
        worker_pid: int | None = None,
        return_code: int | None = None,
    ) -> bool:
        """Move work owned by a vanished worker into durable terminal history.

        Graph callables and futures live only in the worker process, so an
        unexpected worker exit cannot safely resume either the active run or
        its queued successors.  Preserve their navigation/workflow snapshots
        and make the interruption explicit before a replacement worker starts.
        """

        with self._lock:
            state = _read_json(self.queue_state_path)
            state_worker_pid = state.get("workerPid")
            if worker_pid is not None and state_worker_pid != worker_pid:
                return False

            current = state.get("current")
            queued = state.get("queued") if isinstance(state.get("queued"), dict) else {}
            if not isinstance(current, dict) and not queued:
                return False

            completed_at = time.time()
            terminal = []
            if isinstance(current, dict):
                exit_detail = f" (exit code {return_code})" if return_code is not None else ""
                terminal.append(
                    {
                        **current,
                        "status": "failed",
                        "completed_at": completed_at,
                        "updated_at": completed_at,
                        "message": f"The backend worker exited unexpectedly during execution{exit_detail}.",
                        "error": "The model runtime stopped before the run completed.",
                        "exception_type": "BackendWorkerExit",
                        "category": "runtime",
                        "error_code": "backend_worker_exited",
                        "recovery_hint": (
                            "The backend restarted and released model memory. Retry with a lower-memory "
                            "resource plan when the interruption occurred while loading or running a model."
                        ),
                        "backend_restart": True,
                    }
                )
            terminal.extend(
                {
                    **task,
                    "status": "cancelled",
                    "completed_at": completed_at,
                    "updated_at": completed_at,
                    "message": "Cancelled because the backend worker restarted after an unexpected exit.",
                    "backend_restart": True,
                }
                for task in queued.values()
                if isinstance(task, dict)
            )
            prior_recent = state.get("recent") if isinstance(state.get("recent"), list) else []
            _write_json_atomic(
                self.queue_state_path,
                {
                    "workerPid": state_worker_pid,
                    "updatedAt": completed_at,
                    "queued": {},
                    "current": None,
                    "recent": [*terminal, *prior_recent][:30],
                },
            )
            return True

    def status(self) -> dict[str, Any]:
        with self._lock:
            worker = self._worker
            running = bool(worker is not None and worker.poll() is None)
            return {
                "error": False,
                "ready": running and not self._shutting_down,
                "workerPid": worker.pid if worker is not None else None,
                "workerRunning": running,
                "restartRequested": self._restart_requested,
            }

    def queue(self) -> dict[str, Any]:
        state = _read_json(self.queue_state_path)
        status = self.status()
        worker_pid = status.get("workerPid")
        recent = compact_task_history(state.get("recent"))
        if not status.get("workerRunning") or state.get("workerPid") != worker_pid:
            return {"queued": {}, "current": None, "recent": recent}
        return {
            "queued": state.get("queued") if isinstance(state.get("queued"), dict) else {},
            "current": state.get("current") if isinstance(state.get("current"), dict) else None,
            "recent": recent,
        }

    def stop(self) -> tuple[int, dict[str, Any]]:
        with self._lock:
            worker = self._worker
            if worker is None or worker.poll() is not None:
                return HTTPStatus.CONFLICT, {
                    "error": True,
                    "message": "No backend worker is currently running.",
                }
            if self._shutting_down:
                return HTTPStatus.CONFLICT, {
                    "error": True,
                    "message": "The backend is already shutting down.",
                }

            raw_state = _read_json(self.queue_state_path)
            snapshot_matches_worker = raw_state.get("workerPid") == worker.pid
            queue = self.queue()
            current = queue.get("current")
            queued = queue.get("queued") if isinstance(queue.get("queued"), dict) else {}
            if snapshot_matches_worker and current is None and not queued:
                return HTTPStatus.CONFLICT, {
                    "error": True,
                    "message": "Nothing to do. No task is currently running or queued.",
                }

            self._restart_requested = True
            worker_pid = worker.pid
            try:
                worker.kill()
            except ProcessLookupError:
                pass

            cancelled = []
            if isinstance(current, dict):
                cancelled.append(
                    {
                        **current,
                        "status": "cancelled",
                        "completed_at": time.time(),
                        "message": "Execution stopped by the supervisor control plane.",
                    }
                )
            cancelled.extend(
                {
                    **task,
                    "status": "cancelled",
                    "completed_at": time.time(),
                    "message": "Cancelled before execution.",
                }
                for task in queued.values()
                if isinstance(task, dict)
            )
            # `queue()` intentionally strips completed workflow snapshots for
            # lightweight polling. Persist from the raw state so Stop never
            # destroys the lazy `/runs/{task_id}` recovery data.
            prior_recent = raw_state.get("recent") if isinstance(raw_state.get("recent"), list) else []
            _write_json_atomic(
                self.queue_state_path,
                {
                    "workerPid": worker_pid,
                    "updatedAt": time.time(),
                    "queued": {},
                    "current": None,
                    "recent": [*cancelled, *prior_recent][:30],
                },
            )
            return HTTPStatus.OK, {
                "error": False,
                "message": "Execution stopped. The backend worker is restarting to release RAM and VRAM.",
                "task_id": current.get("task_id") if isinstance(current, dict) else None,
                "cancelled_queued_task_ids": list(queued),
                "cleanup_pending": True,
                "backend_restart": True,
                "snapshot_recovery": not snapshot_matches_worker,
            }


def _handler(controller: SupervisorController):
    class SupervisorControlHandler(BaseHTTPRequestHandler):
        server_version = "MoDiffSupervisorControl/1"

        def log_message(self, format: str, *args: object) -> None:
            logger.debug("Supervisor control: " + format, *args)

        def _trusted_request_boundary(self) -> bool:
            try:
                request_host = urlparse(f"//{self.headers.get('Host', '')}").hostname
                peer_host = self.client_address[0]
            except (AttributeError, IndexError, TypeError, ValueError):
                return False
            return _loopback_host(request_host) and _loopback_host(peer_host)

        def _reject_untrusted_boundary(self) -> bool:
            if self._trusted_request_boundary():
                return False
            self._send(
                HTTPStatus.FORBIDDEN,
                {"error": True, "message": "Supervisor requests require a literal loopback Host and peer."},
            )
            return True

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            origin = self.headers.get("Origin")
            if origin and _allowed_browser_origin(origin):
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:
            if self._reject_untrusted_boundary():
                return
            if not _allowed_browser_origin(self.headers.get("Origin")):
                self._send(HTTPStatus.FORBIDDEN, {"error": True, "message": "Cross-site requests are not allowed."})
                return
            self._send(HTTPStatus.NO_CONTENT, {})

        def do_GET(self) -> None:
            if self._reject_untrusted_boundary():
                return
            if self.path == "/health":
                self._send(HTTPStatus.OK, controller.status())
                return
            if self.path == "/queue":
                self._send(HTTPStatus.OK, controller.queue())
                return
            self._send(HTTPStatus.NOT_FOUND, {"error": True, "message": "Control route not found."})

        def do_POST(self) -> None:
            if self._reject_untrusted_boundary():
                return
            if not _allowed_browser_origin(self.headers.get("Origin")):
                self._send(HTTPStatus.FORBIDDEN, {"error": True, "message": "Cross-site requests are not allowed."})
                return
            if self.path == "/stop":
                status, payload = controller.stop()
                self._send(status, payload)
                return
            self._send(HTTPStatus.NOT_FOUND, {"error": True, "message": "Control route not found."})

    return SupervisorControlHandler


class SupervisorControlServer:
    def __init__(self, controller: SupervisorController, host: str, port: int):
        self.controller = controller
        self.server = ThreadingHTTPServer((host, port), _handler(controller))
        self.thread = threading.Thread(target=self.server.serve_forever, name="modiff-supervisor-control", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
