#!/usr/bin/env python3
"""Flexible-HRI adapted YAC 的唯一一键 MuJoCo Demo 入口。"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PYTHON = PROJECT_ROOT / ".venv/bin/python"

# [PLATFORM] VS Code 的解释器即使未选中项目 .venv，也先切换到唯一正式
# runtime；这发生在导入 orchestration/model helper 之前，避免 legacy helper
# 的 import-time interpreter re-exec 劫持正式入口。
if Path(sys.executable).resolve() != PROJECT_PYTHON.resolve():
    if not PROJECT_PYTHON.is_file():
        raise FileNotFoundError(f"Project Python is missing: {PROJECT_PYTHON}")
    os.execve(
        str(PROJECT_PYTHON),
        [str(PROJECT_PYTHON), str(Path(__file__).resolve())],
        os.environ.copy(),
    )

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orchestration import run_human_exo  # noqa: E402


# ==========================================
# USER SETTINGS
# ==========================================
HUMAN = "mild80"
DURATION = 20.0
VIDEO = True
EVALUATION = False
YAC_MODE = "full_yac"
ADP_SCALE = 1.0
EVENT_TRIGGER = True
# ==========================================

# [NUMERICAL CALIBRATION] Paper 规定 Dynamic Event 结构与正性条件；
# 下列数值是当前 100 Hz MuJoCo implementation 的首版保守校准。
# eta_ss≈(sigma/lambda)||z||^2=0.005||z||^2，与 xi=0.005 合并后
# 给出约 sqrt(0.01)=10% 的 relative sampled-state-error tolerance。
# epsilon_delta=0.0025 在 z,eta 接近零时给出 0.05 rad/s sample-error
# floor，避免 threshold->0 导致不必要的高频触发。它不是 torque
# limit，也不是 controller dead zone。
EVENT_PARAMETERS: dict[str, float] = {
    "lambda_eta": 5.0,
    "sigma_eta": 0.025,
    "xi": 0.005,
    "rho_delta": 1.0,
    "epsilon_delta": 0.0025,
}

ACTIVE_SCALE = 1.0
HEALTHY_REFERENCE_RESULT = (
    PROJECT_ROOT
    / "results/experiments/20260822_214246__HUMAN_EXO_HEALTHY100_ZERO"
)
OUTPUT_ROOT = PROJECT_ROOT / "results/algorithms/YAC"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _values(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray(
        [float(row[key]) if row.get(key, "") != "" else math.nan for row in rows],
        dtype=float,
    )


def _bools(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray(
        [row.get(key, "").strip().lower() == "true" for row in rows],
        dtype=bool,
    )


def _rms(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.sqrt(np.mean(finite * finite)))


def _peak(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.max(np.abs(finite)))


def _video_metadata(path: Path) -> dict[str, Any]:
    reader = imageio.get_reader(path)
    try:
        metadata = reader.get_meta_data()
        first_frame = np.asarray(reader.get_data(0))
        frame_count = int(reader.count_frames())
    finally:
        reader.close()
    fps = float(metadata.get("fps", 0.0))
    return {
        "decoded": bool(first_frame.size and frame_count > 0),
        "fps": fps,
        "width": int(first_frame.shape[1]),
        "height": int(first_frame.shape[0]),
        "frames": frame_count,
        "duration_s": frame_count / fps if fps > 0.0 else None,
    }


def _summarize(output_directory: Path) -> dict[str, Any]:
    validation = json.loads(
        (output_directory / "validation.json").read_text(encoding="utf-8")
    )
    timeseries = _rows(output_directory / "timeseries.csv")
    diagnostics = _rows(output_directory / "controller_diagnostics.csv")
    reference = _bools(diagnostics, "reference_valid")
    errors = np.column_stack(
        (
            _values(diagnostics, "error_hip_rad"),
            _values(diagnostics, "error_knee_rad"),
        )
    )
    tracking_valid = reference & np.isfinite(errors).all(axis=1)
    weights = np.column_stack(
        [_values(diagnostics, f"critic_weight_{index}") for index in range(6)]
    )
    bellman = _values(diagnostics, "bellman_error")
    bellman = bellman[np.isfinite(bellman)]
    event = _bools(diagnostics, "event_triggered")
    # Event fraction 只以 Full YAC 中有效的 ADP control steps 为分母；
    # reference-invalid/inactive rows 不可纳入。
    event_samples = _bools(diagnostics, "event_adp_valid")
    video_path = output_directory / "video.mp4"
    summary = {
        "duration_s": float(validation["actual_duration_s"]),
        "termination": str(validation["termination_reason"]),
        "success": bool(validation["success"]),
        "average_speed_mps": float(
            np.mean(_values(timeseries, "forward_velocity"))
        ),
        "minimum_pelvis_height_m": float(
            np.min(_values(timeseries, "pelvis_height"))
        ),
        "max_abs_pelvis_pitch_deg": _peak(
            _values(timeseries, "pelvis_pitch")
        ),
        "tracking_combined_rmse_rad": _rms(errors[tracking_valid]),
        "hip_actual_rms_nm": _rms(
            _values(timeseries, "exo_actual_torque_HipExo_L_nm")
        ),
        "hip_actual_peak_nm": _peak(
            _values(timeseries, "exo_actual_torque_HipExo_L_nm")
        ),
        "knee_actual_rms_nm": _rms(
            _values(timeseries, "exo_actual_torque_KneeExo_L_nm")
        ),
        "knee_actual_peak_nm": _peak(
            _values(timeseries, "exo_actual_torque_KneeExo_L_nm")
        ),
        "hip_mu_rms_nm": _rms(_values(diagnostics, "mu_applied_hip_nm")),
        "hip_mu_peak_nm": _peak(_values(diagnostics, "mu_applied_hip_nm")),
        "knee_mu_rms_nm": _rms(_values(diagnostics, "mu_applied_knee_nm")),
        "knee_mu_peak_nm": _peak(_values(diagnostics, "mu_applied_knee_nm")),
        "critic_weight_norm_final": float(np.linalg.norm(weights[-1])),
        "critic_weight_norm_peak": float(
            np.max(np.linalg.norm(weights, axis=1))
        ),
        "bellman_rms": _rms(bellman),
        "event_count": int(np.count_nonzero(event)),
        "event_fraction": (
            float(np.count_nonzero(event) / np.count_nonzero(event_samples))
            if np.count_nonzero(event_samples)
            else 0.0
        ),
        "result_directory": str(output_directory),
        "video": str(video_path),
        "diagnostics": str(output_directory / "controller_diagnostics.csv"),
        "video_metadata": _video_metadata(video_path) if VIDEO else None,
    }
    (output_directory / "demo_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _print_startup() -> None:
    print(f"Python             : {sys.executable}", flush=True)
    if "/Mujoco/.venv/bin/python" not in str(Path(sys.executable).resolve()):
        raise RuntimeError("Formal YAC run is not using the project .venv")
    print("============================================================")
    print("                   FULL YAC TEST")
    print("============================================================")
    print("Human             : Mild80")
    print("")
    print(f"Duration          : {DURATION:.1f} s")
    print("")
    print(f"YAC Mode          : {YAC_MODE}")
    print("Continuous Critic : ON")
    print(f"Event Trigger     : {'ON' if EVENT_TRIGGER else 'OFF'}")
    print("")
    print(f"lambda_eta        : {EVENT_PARAMETERS['lambda_eta']:.1f} 1/s")
    print(f"sigma_eta         : {EVENT_PARAMETERS['sigma_eta']:.3f} 1/s")
    print(f"xi                : {EVENT_PARAMETERS['xi']:.3f}")
    print(f"rho_delta         : {EVENT_PARAMETERS['rho_delta']:.1f}")
    print(f"epsilon_delta     : {EVENT_PARAMETERS['epsilon_delta']:.4f}")
    print("")
    print("Target relative")
    print("sample error      : ~10%")
    print("")
    print("Event semantics   :")
    print("sampled z / current a / current critic / current g")
    print("")
    print("Residual SW Clip  : OFF")
    print("Total SW Clip     : OFF")
    print("")
    print(f"Video             : {'ON' if VIDEO else 'OFF'}")
    print("")
    print("============================================================", flush=True)


def main() -> int:
    _print_startup()
    run_name = (
        "MILD80_FULL_YAC_DEMO"
        if YAC_MODE == "full_yac"
        else "MILD80_YAC_UNCLIPPED_CONTINUOUS_ADP"
    )
    validation = run_human_exo(
        human_condition=HUMAN,
        exoskeleton="flexible",
        assistance="yac",
        duration_s=DURATION,
        video=VIDEO,
        evaluation=EVALUATION,
        run_name=run_name,
        healthy_reference_result=HEALTHY_REFERENCE_RESULT,
        controller_mode=YAC_MODE,
        active_scale=ACTIVE_SCALE,
        adp_scale=ADP_SCALE,
        event_trigger=EVENT_TRIGGER,
        event_parameters=EVENT_PARAMETERS,
        output_root=OUTPUT_ROOT,
    )
    output_directory = (PROJECT_ROOT / validation["artifacts"]["manifest"]).parent
    summary = _summarize(output_directory)
    if not summary["success"]:
        completion_title = "YAC RUN STOPPED"
    elif YAC_MODE == "full_yac":
        completion_title = "FULL YAC DEMO COMPLETE"
    else:
        completion_title = "CONTINUOUS ADP RUN COMPLETE"
    print("==============================================")
    print(f"             {completion_title}")
    print("==============================================")
    print(f"Duration: {summary['duration_s']:.3f} s")
    print(f"Termination: {summary['termination']}")
    print(f"Average speed: {summary['average_speed_mps']:.4f} m/s")
    print(f"Max pelvis pitch: {summary['max_abs_pelvis_pitch_deg']:.3f} deg")
    print(
        "Hip actual torque RMS / peak: "
        f"{summary['hip_actual_rms_nm']:.4f} / {summary['hip_actual_peak_nm']:.4f} N*m"
    )
    print(
        "Knee actual torque RMS / peak: "
        f"{summary['knee_actual_rms_nm']:.4f} / {summary['knee_actual_peak_nm']:.4f} N*m"
    )
    print(
        "Hip ADP mu RMS / peak: "
        f"{summary['hip_mu_rms_nm']:.4f} / {summary['hip_mu_peak_nm']:.4f} N*m"
    )
    print(
        "Knee ADP mu RMS / peak: "
        f"{summary['knee_mu_rms_nm']:.4f} / {summary['knee_mu_peak_nm']:.4f} N*m"
    )
    print(
        "Critic weight norm final / peak: "
        f"{summary['critic_weight_norm_final']:.6g} / "
        f"{summary['critic_weight_norm_peak']:.6g}"
    )
    print(f"Bellman RMS: {summary['bellman_rms']:.6g}")
    print(f"Event count: {summary['event_count']}")
    print(f"Event fraction: {summary['event_fraction']:.3%}")
    print(f"Result directory: {summary['result_directory']}")
    print(f"Video: {summary['video']}")
    print(f"Diagnostics: {summary['diagnostics']}")
    print("==============================================")
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
