"""Check the Pika source supplement using an installed reviewed runtime; no downloads."""
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    if not sys.dont_write_bytecode:
        raise RuntimeError("Launch this source audit with python -B.")
    from modiff.optimization_packages import activate_runtime_overlay
    activate_runtime_overlay()
    from modiff.supplemental_modular_contracts import (
        PIKA_PIPELINE, reproduce_installed_pika_contract, reviewed_supplemental_modular_contract,
    )
    actual = reproduce_installed_pika_contract()
    if actual != reviewed_supplemental_modular_contract(PIKA_PIPELINE):
        raise RuntimeError("The reviewed Pika workflow supplement differs from the installed upstream declarations.")
    print("Pika upstream workflow supplement matches the exact installed source provider; no model weights loaded.")


if __name__ == "__main__":
    main()
