#!/usr/bin/env python3
"""One-click L1 entry point: bare Human walking experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ===== USER CONFIG =====
HUMAN = "healthy100"
DURATION = 20.0
VIDEO = True
EVALUATION = True
RUN_NAME = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orchestration import run_human_only  # noqa: E402


def _options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human", default=HUMAN)
    parser.add_argument("--duration", type=float, default=DURATION)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-evaluation", action="store_true")
    parser.add_argument("--run-name", default=RUN_NAME)
    return parser.parse_args()


def main() -> int:
    options = _options()
    try:
        result = run_human_only(
            human_condition=options.human,
            duration_s=options.duration,
            video=VIDEO and not options.no_video,
            evaluation=EVALUATION and not options.no_evaluation,
            run_name=options.run_name,
        )
    except Exception as exc:
        print(f"ERROR: L1 Human-only experiment failed: {exc}", file=sys.stderr)
        return 1
    output_directory = (PROJECT_ROOT / result["artifacts"]["manifest"]).parent
    print(f"L1 Human-only complete: {output_directory}")
    print(f"Regression: {result.get('regression', {}).get('passed', 'not run')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
