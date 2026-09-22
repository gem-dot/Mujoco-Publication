#!/usr/bin/env python3
"""One-click final benchmark for the frozen ICRA controller."""

from __future__ import annotations

import hashlib
import contextlib
import os
import platform
import subprocess
import sys
from pathlib import Path


# ============================================================
# USER CONFIGURATION
# ============================================================
# NORMAL USE:
# Open this file and click "Run Python File".
#
# Final Event Moderate60:
# RUN_CONDITION = "moderate60"
# RUN_CONTROLLER = "final_event"
#
# Final Event Healthy:
# RUN_CONDITION = "healthy100"
#
# Final Continuous:
# RUN_CONTROLLER = "continuous"
RUN_CONDITION = "moderate60"
RUN_CONTROLLER = "final_event"
DURATION_S = 20.0
SHOW_VIEWER = True
SAVE_VIDEO = True
SAVE_PHYSICS_1KHZ = True
AUTO_OPEN_RESULT = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PYTHON = PROJECT_ROOT / ".venv/bin/python"
PROJECT_MJPYTHON = PROJECT_ROOT / ".venv/bin/mjpython"
ICRA_SOURCE = PROJECT_ROOT / "controllers/ICRA.py"
REFERENCE_RESULT = (
    PROJECT_ROOT / "results/experiments/20260822_214246__HUMAN_EXO_HEALTHY100_ZERO"
)
EXPECTED_ICRA_SHA256 = "971e34d1ad8f6af6355f2ec4ac1e16600355f8f56dbab2582b92fb4a9ddb89ef"
EXPECTED_YAC_HASHES = {
    "controllers/YAC.py": "92a822b86bae56810b40489a3d10b61f7329b3c8dc977cddd753144452254bd2",
    "controllers/YAC_benchmark.py": "ef3987ac50d637350eb28519dcd17c0bda11757eb3fa9177a878380dde769604",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_setting(name: str, configured: str) -> str:
    """Private environment overrides support automated smoke tests without CLI args."""
    return os.environ.get(name, configured).strip().lower()


def _runtime_bool(disable_name: str, configured: bool) -> bool:
    return bool(configured) and os.environ.get(disable_name) != "1"


def _bootstrap_runtime() -> None:
    headless = os.environ.get("ICRA_BENCHMARK_HEADLESS") == "1"
    viewer_ready = os.environ.get("ICRA_BENCHMARK_MJPYTHON") == "1"
    viewer = _runtime_bool("ICRA_BENCHMARK_HEADLESS", SHOW_VIEWER)
    if platform.system() == "Darwin" and viewer and not headless and not viewer_ready:
        if not PROJECT_MJPYTHON.is_file():
            raise FileNotFoundError(f"Project mjpython is missing: {PROJECT_MJPYTHON}")
        environment = os.environ.copy()
        environment["ICRA_BENCHMARK_MJPYTHON"] = "1"
        # Execute the wrapper through Python because this project path contains
        # spaces, which macOS cannot parse in the wrapper's shebang.
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

from orchestration import standard_workflow  # noqa: E402
from scripts.analyze_ICRA_final import finalize_result  # noqa: E402


CONTROLLERS = {
    "continuous": {
        "mode": "intention_target_game_active",
        "label": "Final Continuous ICRA",
        "directory": "ICRA_FINAL_CONTINUOUS",
    },
    "final_event": {
        "mode": "residual_event_triggered_intention_target_game_active",
        "label": "Final Residual-Event ICRA",
        "directory": "ICRA_FINAL_EVENT",
    },
}
CONDITIONS = {
    "healthy100": "Healthy100",
    "moderate60": "Moderate60 bilateral vasti weakness",
}


def _assert_freeze() -> dict[str, str]:
    hashes = {"controllers/ICRA.py": _sha256(ICRA_SOURCE)}
    for relative, expected in EXPECTED_YAC_HASHES.items():
        hashes[relative] = _sha256(PROJECT_ROOT / relative)
        if hashes[relative] != expected:
            raise RuntimeError(f"Frozen source hash mismatch: {relative}")
    if hashes["controllers/ICRA.py"] != EXPECTED_ICRA_SHA256:
        raise RuntimeError("Frozen scientific source hash mismatch: controllers/ICRA.py")
    return hashes


def _result_directory(validation: dict[str, object]) -> Path:
    raw = str(dict(validation["artifacts"])["manifest"])
    path = Path(raw)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve().parent


def _print_header(condition: str, controller: str, duration: float,
                  viewer: bool, video: bool) -> None:
    print("=" * 40)
    print("ICRA FINAL BENCHMARK")
    print("=" * 40)
    print(f"Condition: {CONDITIONS[condition]}")
    print(f"Controller: {CONTROLLERS[controller]['label']}")
    print(f"Duration: {duration:.1f} s")
    print(f"Viewer: {'ON' if viewer else 'OFF'}")
    print(f"Video: {'ON' if video else 'OFF'}")
    print("Physics: 1 kHz")
    print("Control: 100 Hz")
    print("=" * 40)


def _print_result(summary: dict[str, object], result: Path) -> None:
    phase = dict(summary["phase_assistance"])
    torque = dict(summary["motor_torque"])
    event = summary.get("event")
    print("\n" + "=" * 40)
    print("ICRA FINAL RESULT")
    print("=" * 40)
    print(f"Condition: {summary['condition']}")
    print(f"Controller: {summary['controller']}")
    print(f"Survival: {float(summary['survival_time_s']):.3f} s")
    print(f"Termination: {summary['termination_reason']}")
    print(f"Tracking RMS: {float(summary['tracking_RMS_rad']):.6f} rad")
    print(f"S_loading: {float(phase['S_loading_nm']):.6f} Nm")
    print(f"A_midswing: {float(phase['A_midswing_nm']):.6f} Nm")
    print(f"R_phase: {float(phase['R_phase']):.6f}")
    print(f"Ja: {float(summary['Ja_activation_squared_proxy']):.6f}")
    print(f"M5 HRI RMS: {float(summary['M5_physical_HRI_RMS_total_nm']):.6f} Nm")
    print("Torque RMS: "
          f"Hip {float(torque['HipExo_L']['RMS_nm']):.6f} Nm, "
          f"Knee {float(torque['KneeExo_L']['RMS_nm']):.6f} Nm")
    print("Torque peak: "
          f"Hip {float(torque['HipExo_L']['peak_abs_nm']):.6f} Nm, "
          f"Knee {float(torque['KneeExo_L']['peak_abs_nm']):.6f} Nm")
    if isinstance(event, dict):
        print("Learned residual refresh reduction: "
              f"{100.0 * float(event['learned_residual_refresh_reduction']):.2f} %")
        print(f"Accepted events: {int(event['accepted_residual_refreshes'])}")
    print(f"Result directory: {result}")
    for label, filename in (
        ("Video", "video.mp4"), ("Timeseries", "timeseries.csv"),
        ("Summary", "summary.json"), ("Basic figure", "basic_results.png"),
    ):
        path = result / filename
        print(f"{label}: {path if path.is_file() else 'N/A'}")
    print("=" * 40)


def main() -> int:
    condition = _runtime_setting("ICRA_BENCHMARK_CONDITION", RUN_CONDITION)
    controller = _runtime_setting("ICRA_BENCHMARK_CONTROLLER", RUN_CONTROLLER)
    duration = float(os.environ.get("ICRA_BENCHMARK_DURATION_S", DURATION_S))
    viewer = _runtime_bool("ICRA_BENCHMARK_HEADLESS", SHOW_VIEWER)
    video = _runtime_bool("ICRA_BENCHMARK_NO_VIDEO", SAVE_VIDEO)
    if condition not in CONDITIONS:
        raise ValueError("RUN_CONDITION must be 'healthy100' or 'moderate60'")
    if controller not in CONTROLLERS:
        raise ValueError("RUN_CONTROLLER must be 'continuous' or 'final_event'")
    if duration <= 0.0:
        raise ValueError("DURATION_S must be positive")
    hashes_before = _assert_freeze()
    if not REFERENCE_RESULT.is_dir():
        raise FileNotFoundError(f"Frozen Healthy reference is missing: {REFERENCE_RESULT}")
    _print_header(condition, controller, duration, viewer, video)
    print("Running real MuJoCo physics...", flush=True)
    run_name = f"{CONTROLLERS[controller]['directory']}_{condition.upper()}"
    with open(os.devnull, "w", encoding="utf-8") as quiet, contextlib.redirect_stdout(quiet):
        validation = standard_workflow.run_human_exo(
            human_condition=condition,
            exoskeleton="flexible",
            assistance="icra",
            duration_s=duration,
            video=video,
            evaluation=False,
            run_name=run_name,
            healthy_reference_result=REFERENCE_RESULT,
            controller_mode=str(CONTROLLERS[controller]["mode"]),
            output_root=PROJECT_ROOT / "results/experiments",
            enforce_regression=False,
            viewer=viewer,
            viewer_keep_open=False,
            record_git_metadata=True,
        )
    result = _result_directory(validation)
    hashes_after = _assert_freeze()
    if hashes_after != hashes_before:
        raise RuntimeError("Frozen source changed during benchmark execution")
    summary = finalize_result(
        result,
        condition=condition,
        controller=controller,
        requested_duration_s=duration,
        save_video=video,
        save_physics_1khz=SAVE_PHYSICS_1KHZ,
        frozen_hashes=hashes_after,
        benchmark_sha256=_sha256(Path(__file__).resolve()),
    )
    _print_result(summary, result)
    if AUTO_OPEN_RESULT and platform.system() == "Darwin":
        subprocess.run(["open", str(result)], check=False)
    return 0 if bool(summary["all_safety_checks_passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
