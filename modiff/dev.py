"""Legacy managed-installer entry point; ordinary setup uses ``uv sync``.

This module intentionally uses only the standard library until dispatch.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def runtime_command(command: str, arguments: list[str]) -> tuple[list[str], dict[str, str]]:
    from modiff.install import _rocm_environment
    from modiff.runtime_profile import read_state
    from modiff.runtime_environment import configure_allocator

    venv = ROOT / ".venv"
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        raise ValueError("No environment exists. Run uv sync --extra cpu, or choose your accelerator extra.")
    state = read_state(venv) or {}
    environment = os.environ.copy()
    if state.get("profile") == "amd-rocm-linux":
        environment = _rocm_environment()
    configure_allocator(environment)
    target = ["-m", "modiff.preflight"] if command == "check" else [str(ROOT / "main.py")]
    return [str(python), *target, *arguments], environment


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(
            "Usage: python -m modiff.dev {plan|setup|check|run} [arguments]\n"
            "plan/setup: managed installer arguments; setup preserves an existing .venv unless --repair is explicit.\n"
            "check: preflight arguments; run: main.py arguments. See docs/developer-setup.md."
        )
        return 0
    command, *arguments = arguments
    try:
        if command in {"plan", "setup"}:
            from modiff import install

            parsed = install.parser().parse_args(arguments)
            read_only = command == "plan" or parsed.dry_run or parsed.system_check or parsed.guide
            if not read_only and ((ROOT / ".venv").exists() or (ROOT / ".venv").is_symlink()) and not parsed.repair:
                raise ValueError(
                    "Existing .venv preserved. Use modiff.dev check; replacing it requires explicit --repair."
                )
            return install.main([*arguments, *(["--dry-run"] if command == "plan" else [])])
        if command in {"check", "run"}:
            cmd, environment = runtime_command(command, arguments)
            # Inherit the terminal and signals, without a shell or interpreter resolution.
            return subprocess.call(cmd, cwd=ROOT, env=environment)
        raise ValueError(f"Unknown command: {command}. Use --help.")
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
