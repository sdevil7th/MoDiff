"""Compare old/new raw template images with matching consumed recipe contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from contextlib import ExitStack

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modiff.image_template_comparison import compare_image_template_outputs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline-contract", type=Path, required=True)
    parser.add_argument("--candidate-contract", type=Path, required=True)
    parser.add_argument("--max-pixel-error", type=int, default=0)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    sources = [*args.baseline, *args.candidate, args.baseline_contract, args.candidate_contract]
    if any(args.report.resolve() == path.resolve() for path in sources):
        parser.error("The report must not overwrite an image or source contract.")
    with ExitStack() as stack:
        baseline = [stack.enter_context(Image.open(path)) for path in args.baseline]
        candidate = [stack.enter_context(Image.open(path)) for path in args.candidate]
        report = compare_image_template_outputs(
            baseline, candidate,
            baseline_contract=json.loads(args.baseline_contract.read_text(encoding="utf-8")),
            candidate_contract=json.loads(args.candidate_contract.read_text(encoding="utf-8")),
            max_pixel_error=args.max_pixel_error,
        )
    report["files"] = {
        role: [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]
        for role, paths in (("baseline", args.baseline), ("candidate", args.candidate))
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "pixelIdentity": report["pixelIdentity"],
                      "contractMismatchPaths": report["contractMismatchPaths"], "report": str(args.report)}, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
