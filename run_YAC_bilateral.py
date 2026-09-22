#!/usr/bin/env python3
"""One-click visual runner for the bilateral Moderate60 YAC experiment."""

from __future__ import annotations

import csv
import os
import platform
import sys
from pathlib import Path


# ==========================================
# USER SETTINGS
# ==========================================
MODE = "full_yac"  # zero / nominal / full_yac
DURATION = 20.0
VIEWER = True
# ==========================================


PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_PYTHON = PROJECT_ROOT / ".venv/bin/python"
MJ_PYTHON = PROJECT_ROOT / ".venv/bin/mjpython"


def _ensure_project_runtime() -> None:
    headless = os.environ.get("MUJOCO_HEADLESS") == "1"
    needs_mjpython = VIEWER and not headless and platform.system() == "Darwin"
    viewer_ready = os.environ.get("YAC_BILATERAL_MJPYTHON") == "1"
    if needs_mjpython and not viewer_ready:
        if not PROJECT_PYTHON.is_file() or not MJ_PYTHON.is_file():
            raise FileNotFoundError("Project .venv Python/mjpython is missing")
        environment = os.environ.copy()
        environment["YAC_BILATERAL_MJPYTHON"] = "1"
        os.execve(
            str(PROJECT_PYTHON),
            [
                str(PROJECT_PYTHON),
                str(MJ_PYTHON),
                str(Path(__file__).resolve()),
            ],
            environment,
        )
    if viewer_ready:
        return
    if Path(sys.executable).resolve() != PROJECT_PYTHON.resolve():
        if not PROJECT_PYTHON.is_file():
            raise FileNotFoundError(f"Project Python is missing: {PROJECT_PYTHON}")
        os.execve(
            str(PROJECT_PYTHON),
            [str(PROJECT_PYTHON), str(Path(__file__).resolve())],
            os.environ.copy(),
        )


_ensure_project_runtime()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orchestration import run_human_exo  # noqa: E402


OUTPUT_ROOT = PROJECT_ROOT / "results/runs/YAC_bilateral_visual"
HEALTHY_REFERENCE_RESULT = (
    PROJECT_ROOT
    / "results/experiments/20260822_214246__HUMAN_EXO_HEALTHY100_ZERO"
)
ACTIVE_SCALE = 1.0
ADP_SCALE = 1.0
EVENT_PARAMETERS = {
    "lambda_eta": 5.0,
    "sigma_eta": 0.025,
    "xi": 0.005,
    "rho_delta": 1.0,
    "epsilon_delta": 0.0025,
}


def _mode_settings() -> tuple[str, str | None, bool]:
    mode = MODE.strip().lower()
    if mode == "zero":
        return "zero", None, False
    if mode == "nominal":
        return "yac", "bilateral_nominal", False
    if mode == "full_yac":
        return "yac", "bilateral_full_yac", True
    raise ValueError("MODE must be: zero, nominal, or full_yac")


def _summary(validation: dict[str, object]) -> dict[str, float | str]:
    output_directory = (PROJECT_ROOT / validation["artifacts"]["manifest"]).parent
    with (output_directory / "timeseries.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    speed = [float(row["forward_velocity"]) for row in rows]
    pitch = [abs(float(row["pelvis_pitch"])) for row in rows]
    height = [float(row["pelvis_height"]) for row in rows]
    return {
        "survival_time_s": float(validation["actual_duration_s"]),
        "termination_reason": str(validation["termination_reason"]),
        "average_speed_mps": sum(speed) / len(speed),
        "max_abs_pelvis_pitch_deg": max(pitch),
        "min_pelvis_height_m": min(height),
        "output_directory": str(output_directory),
    }


def main() -> int:
    assistance, controller_mode, event_trigger = _mode_settings()
    smoke = os.environ.get("YAC_BILATERAL_SMOKE") == "1"
    duration = min(DURATION, 0.05) if smoke else DURATION
    viewer = VIEWER and not smoke and os.environ.get("MUJOCO_HEADLESS") != "1"
    mode_label = {
        "zero": "Zero",
        "nominal": "Bilateral Nominal",
        "full_yac": "Bilateral Full YAC",
    }[MODE.strip().lower()]

    print("==========================================")
    print("YAC Bilateral Visual Experiment")
    print("Pathology: vasti L/R = 60%")
    print(f"Mode: {mode_label}")
    print("Motors: HipL KneeL HipR KneeR" if assistance == "yac" else "Motors: Zero")
    print(f"Duration: {duration:g} s")
    print("==========================================", flush=True)

    validation = run_human_exo(
        human_condition="moderate60",
        exoskeleton="flexible",
        assistance=assistance,
        duration_s=duration,
        video=False,
        evaluation=False,
        run_name=f"YAC_BILATERAL_VISUAL_MODERATE60_{MODE.upper()}",
        healthy_reference_result=(
            HEALTHY_REFERENCE_RESULT if assistance == "yac" else None
        ),
        controller_mode=controller_mode,
        active_scale=ACTIVE_SCALE if assistance == "yac" else None,
        adp_scale=ADP_SCALE if assistance == "yac" else None,
        event_trigger=event_trigger,
        event_parameters=dict(EVENT_PARAMETERS) if event_trigger else None,
        output_root=OUTPUT_ROOT,
        enforce_regression=False,
        viewer=viewer,
        viewer_keep_open=viewer,
        record_git_metadata=False,
    )
    damage = validation["damage_verification"]["muscles"]
    for muscle in ("vasti_l", "vasti_r"):
        values = damage[muscle]
        print(
            f"{muscle}: healthy={values['healthy_gain_force_N']:.6f} N, "
            f"impaired={values['applied_gain_force_N']:.6f} N, "
            f"ratio={values['gain_ratio']:.6f}"
        )
    summary = _summary(validation)
    print("==========================================")
    print(f"Survival time: {summary['survival_time_s']:.3f} s")
    print(f"Termination reason: {summary['termination_reason']}")
    print(f"Average speed: {summary['average_speed_mps']:.4f} m/s")
    print(f"Max abs pelvis pitch: {summary['max_abs_pelvis_pitch_deg']:.3f} deg")
    print(f"Min pelvis height: {summary['min_pelvis_height_m']:.4f} m")
    print(f"Results: {summary['output_directory']}")
    print("==========================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
