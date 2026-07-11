"""Legacy compatibility wrapper for ``modiff.preflight``."""

from modiff.preflight import *  # noqa: F401,F403
from modiff.preflight import main


if __name__ == "__main__":
    raise SystemExit(main())
