#!/usr/bin/env python3
"""One-click three-case ICRA video demonstration.

Open this file in VS Code and click Run Python File. No arguments are needed.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path


# ============================================================
# USER CONFIGURATION
# ============================================================
REUSE_EXISTING_RESULTS = False
DURATION_S = 20.0
VIDEO_FPS = 25
SHOW_VIEWER = False
AUTO_OPEN_RESULT = False


PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_PYTHON = PROJECT_ROOT / ".venv/bin/python"
PROJECT_MJPYTHON = PROJECT_ROOT / ".venv/bin/mjpython"


def _bootstrap_runtime() -> None:
    viewer_ready = os.environ.get("ICRA_DEMO_MJPYTHON") == "1"
    if platform.system() == "Darwin" and SHOW_VIEWER and not viewer_ready:
        if not PROJECT_MJPYTHON.is_file():
            raise FileNotFoundError(f"Project mjpython is missing: {PROJECT_MJPYTHON}")
        environment = os.environ.copy()
        environment["ICRA_DEMO_MJPYTHON"] = "1"
        os.execve(
            str(PROJECT_PYTHON),
            [str(PROJECT_PYTHON), str(PROJECT_MJPYTHON), str(Path(__file__).resolve())],
            environment,
        )
    if not viewer_ready and Path(sys.executable).resolve() != PROJECT_PYTHON.resolve():
        if not PROJECT_PYTHON.is_file():
            raise FileNotFoundError(f"Project Python is missing: {PROJECT_PYTHON}")
        os.execve(
            str(PROJECT_PYTHON),
            [str(PROJECT_PYTHON), str(Path(__file__).resolve())],
            os.environ.copy(),
        )


if __name__ == "__main__":
    _bootstrap_runtime()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.icra_three_case_demo import DemoSettings, run_demo  # noqa: E402


def main() -> int:
    return run_demo(
        DemoSettings(
            reuse_existing_results=REUSE_EXISTING_RESULTS,
            duration_s=DURATION_S,
            video_fps=VIDEO_FPS,
            show_viewer=SHOW_VIEWER,
            auto_open_result=AUTO_OPEN_RESULT,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
