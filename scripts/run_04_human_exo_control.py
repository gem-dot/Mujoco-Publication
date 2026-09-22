#!/usr/bin/env python3
"""One-click unified Human/Patient and Flexible Human-Exo experiment entry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ==========================
# USER CONFIG
# ==========================
HUMAN = "healthy100"
# healthy100 / mild80 / moderate60 / severe40

EXOSKELETON = "flexible"
# none / flexible

ASSISTANCE = "zero"
# zero is READY; basic/pid/impedance/adp are not yet enabled on Flexible V1

DURATION = 20.0
VIDEO = True
EVALUATION = True
RUN_NAME = None

# Patient evaluation uses the latest verified Healthy100 Flexible Zero result.
# Set an explicit result directory only when reproducing against a fixed run.
HEALTHY_REFERENCE_RESULT = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orchestration import run_human_exo  # noqa: E402


def _options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human", default=HUMAN)
    parser.add_argument("--exoskeleton", default=EXOSKELETON)
    parser.add_argument("--assistance", default=ASSISTANCE)
    parser.add_argument("--duration", type=float, default=DURATION)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-evaluation", action="store_true")
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--healthy-reference", default=HEALTHY_REFERENCE_RESULT)
    return parser.parse_args()


def main() -> int:
    options = _options()
    try:
        result = run_human_exo(
            human_condition=options.human,
            exoskeleton=options.exoskeleton,
            assistance=options.assistance,
            duration_s=options.duration,
            video=VIDEO and not options.no_video,
            evaluation=EVALUATION and not options.no_evaluation,
            run_name=options.run_name,
            healthy_reference_result=options.healthy_reference,
        )
    except Exception as exc:
        print(f"ERROR: Unified Human-Exo experiment failed: {exc}", file=sys.stderr)
        return 1

    output_directory = (PROJECT_ROOT / result["artifacts"]["manifest"]).parent
    print(f"Unified experiment complete: {output_directory}")
    print(f"Duration:    {result['actual_duration_s']:.3f} s")
    print(f"Termination: {result['termination_reason']}")
    print(f"Success:     {result['success']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
