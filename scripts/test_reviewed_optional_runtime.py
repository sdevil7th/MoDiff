"""Run pytest with the existing validated active runtime; never install packages.

Invoke through scripts/with-runtime-env.sh so the base accelerator is configured
before imports. Tests needing a different overlay require its normal explicit
application activation first. No raw path bypass of the artifact seal is used.
"""
import os
from pathlib import Path
import sys


def main():
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    root = str(Path(__file__).resolve().parents[1])
    sys.path.insert(0, root)
    from modiff.optimization_packages import activate_runtime_overlay
    environment = activate_runtime_overlay()
    if not environment:
        raise SystemExit('No validated active optional runtime. Use the application Setup repair/activation flow first.')
    # Subprocess contract generators must see the same validated overlay and
    # checkout; they inherit the no-bytecode rule as well.
    from modiff.optimization_packages import _environment_inspection
    inspection = _environment_inspection(environment)
    if inspection.get('status') != 'ready':
        raise SystemExit('The optional runtime changed during test setup; execution refused.')
    os.environ['PYTHONPATH'] = os.pathsep.join([root, str(inspection['sitePackages'])])
    import pytest
    return pytest.main(sys.argv[1:] or ['-q', 'tests'])


if __name__ == '__main__':
    raise SystemExit(main())
