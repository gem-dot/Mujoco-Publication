#!/usr/bin/env python3
"""One-click final assistance benchmark for the frozen bilateral YAC controller.

This file is orchestration and evaluation only.  The controller implementation,
reference, intention mapping, impedance transformation, critic, Bellman update,
event trigger, and torque law remain sourced exclusively from controllers/YAC.py.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


# ============================================================
# USER SETTINGS
# ============================================================
DEFAULT_DURATION_S = 20.0
DEFAULT_SAVE_VIDEO = True
DEFAULT_FULL_YAC_VIEWER = True
DEFAULT_GENERATE_FIGURES = True
DEFAULT_GENERATE_COMPARISON_VIDEO = True
DEFAULT_STARTUP_CUTOFF_S = 2.0


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PYTHON = PROJECT_ROOT / ".venv/bin/python"
PROJECT_MJPYTHON = PROJECT_ROOT / ".venv/bin/mjpython"

# VS Code "Run Python File" enters the validated project runtime.  On macOS,
# mjpython is used only when the user-facing Full-YAC viewer is enabled.
if __name__ == "__main__":
    _HEADLESS = os.environ.get("YAC_BENCHMARK_HEADLESS") == "1"
    _VIEWER_READY = os.environ.get("YAC_BENCHMARK_MJPYTHON") == "1"
    if (
        platform.system() == "Darwin"
        and DEFAULT_FULL_YAC_VIEWER
        and not _HEADLESS
        and not _VIEWER_READY
    ):
        if not PROJECT_PYTHON.is_file() or not PROJECT_MJPYTHON.is_file():
            raise FileNotFoundError("Project .venv Python/mjpython is missing")
        environment = os.environ.copy()
        environment["YAC_BENCHMARK_MJPYTHON"] = "1"
        os.execve(
            str(PROJECT_PYTHON),
            [str(PROJECT_PYTHON), str(PROJECT_MJPYTHON), str(Path(__file__).resolve())],
            environment,
        )
    if not _VIEWER_READY and Path(sys.executable).resolve() != PROJECT_PYTHON.resolve():
        if not PROJECT_PYTHON.is_file():
            raise FileNotFoundError(f"Project Python is missing: {PROJECT_PYTHON}")
        os.execve(
            str(PROJECT_PYTHON),
            [str(PROJECT_PYTHON), str(Path(__file__).resolve())],
            os.environ.copy(),
        )

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import imageio.v2 as imageio  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import matplotlib  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageEnhance, ImageFont  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from controllers.YAC import (  # noqa: E402
    DEFAULT_EVENT_PARAMETERS,
    DEFAULT_HEALTHY_REFERENCE_RESULT,
    YACController,
)
from evaluation import metrics  # noqa: E402
from orchestration import run_human_exo  # noqa: E402
from orchestration import standard_workflow  # noqa: E402


CONDITIONS = (
    ("NO_EXO", "01_MODERATE60_NO_EXO", "none", "zero"),
    ("EXO_ZERO", "02_MODERATE60_EXO_ZERO", "flexible", "zero"),
    ("FULL_YAC", "03_MODERATE60_FULL_YAC", "flexible", "yac"),
)
HUMAN_JOINT_COLUMNS = {
    "hip_l": ("hip_flexion_l", "left_hip_angle_rad"),
    "knee_l": ("knee_angle_l", "left_knee_angle_rad"),
    "hip_r": ("hip_flexion_r", "right_hip_angle_rad"),
    "knee_r": ("knee_angle_r", "right_knee_angle_rad"),
}
M5_LABELS = ("hip_l", "knee_l", "hip_r", "knee_r")
MOTOR_CHANNELS = ("HipExo_L", "KneeExo_L", "HipExo_R", "KneeExo_R")
INTENTION_FAMILIES = (
    "hamstrings", "bifemsh", "glutmax", "iliopsoas", "rectfem", "vasti"
)
HEALTHY_REFERENCE = DEFAULT_HEALTHY_REFERENCE_RESULT
EXISTING_INTENTION_ABLATION = (
    PROJECT_ROOT
    / "results/runs/YAC_paper_experiment_v1"
    / "20260903_140811__YAC_PAPER_FULL_INTENTION"
    / "yac_paper_summary.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(value), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _artifact_path(validation: dict[str, Any], name: str) -> Path | None:
    raw = validation.get("artifacts", {}).get(name)
    if not raw:
        return None
    path = Path(str(raw))
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _result_directory(validation: dict[str, Any]) -> Path:
    manifest = _artifact_path(validation, "manifest")
    if manifest is None or not manifest.is_file():
        raise FileNotFoundError("Run manifest artifact is unavailable")
    return manifest.parent


def _numeric(table: dict[str, np.ndarray], name: str) -> np.ndarray:
    values, resolved = metrics.numeric_column(table, [name])
    if values is None:
        raise ValueError(f"Required numeric column is missing: {name}")
    if resolved != name:
        raise ValueError(f"Unexpected column resolution for {name}: {resolved}")
    return np.asarray(values, dtype=float)


def _text(table: dict[str, np.ndarray], name: str) -> np.ndarray:
    values, resolved = metrics.text_column(table, [name])
    if values is None:
        raise ValueError(f"Required text column is missing: {name}")
    if resolved != name:
        raise ValueError(f"Unexpected column resolution for {name}: {resolved}")
    return np.asarray(values, dtype=object)


def _finite_integral(time_s: np.ndarray, values: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(time_s) & np.isfinite(values)
    if np.count_nonzero(valid) < 2:
        raise ValueError("Integral requires at least two finite samples")
    return float(np.trapezoid(values[valid], time_s[valid]))


def _percent_change(baseline: float | None, current: float | None) -> float | None:
    if baseline is None or current is None or abs(baseline) <= 1.0e-15:
        return None
    return 100.0 * (current - baseline) / baseline


def _new_benchmark_root() -> Path:
    parent = PROJECT_ROOT / "results/runs/YAC_benchmark"
    stem = datetime.now().strftime("%Y%m%d_%H%M%S") + "__YAC_FINAL_BENCHMARK"
    candidate = parent / stem
    suffix = 1
    while candidate.exists():
        candidate = parent / f"{stem}__{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _validate_human_joint_mapping() -> dict[str, Any]:
    model_path = standard_workflow.HEALTHY_MODEL_XML.resolve()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    mapping: dict[str, Any] = {}
    for label, (joint_name, csv_column) in HUMAN_JOINT_COLUMNS.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Healthy Human joint is missing: {joint_name}")
        mapping[label] = {
            "model_joint_name": joint_name,
            "joint_id": int(joint_id),
            "qpos_address": int(model.jnt_qposadr[joint_id]),
            "saved_timeseries_column": csv_column,
        }
    return {
        "model_xml": str(model_path),
        "model_xml_sha256": _sha256(model_path),
        "mapping": mapping,
        "mapping_method": "MuJoCo joint-name lookup plus runner named observation schema",
    }


def _pathology_ratios(validation: dict[str, Any]) -> dict[str, float]:
    muscles = validation.get("damage_verification", {}).get("muscles", {})
    ratios: dict[str, float] = {}
    for muscle in ("vasti_l", "vasti_r"):
        value = muscles.get(muscle, {}).get("gain_ratio")
        if value is None:
            raise ValueError(f"Runtime pathology audit is missing {muscle}")
        ratios[muscle] = float(value)
        if not math.isclose(ratios[muscle], 0.6, rel_tol=0.0, abs_tol=1.0e-12):
            raise RuntimeError(f"{muscle} runtime Fmax ratio is {ratios[muscle]}, not 0.6")
    return ratios


def _link_condition_artifacts(
    condition_root: Path, result_directory: Path, validation: dict[str, Any]
) -> dict[str, str | None]:
    names = (
        "manifest", "timeseries", "validation", "video",
        "physics_timeseries_1khz", "controller_diagnostics",
    )
    linked: dict[str, str | None] = {}
    for name in names:
        target = _artifact_path(validation, name)
        if target is None or not target.is_file():
            linked[name] = None
            continue
        link = condition_root / target.name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(os.path.relpath(target, condition_root))
        linked[name] = str(link.resolve())
    _write_json(
        condition_root / "result_pointer.json",
        {
            "result_directory": str(result_directory),
            "artifacts": linked,
            "symlink_semantics": "condition-root convenience links; source artifacts remain in workflow result directory",
        },
    )
    return linked


def _run_condition(
    label: str,
    folder: str,
    exoskeleton: str,
    assistance: str,
    benchmark_root: Path,
    headless: bool,
) -> dict[str, Any]:
    condition_root = benchmark_root / folder
    condition_root.mkdir(parents=True, exist_ok=False)
    kwargs: dict[str, Any] = {
        "human_condition": "moderate60",
        "exoskeleton": exoskeleton,
        "assistance": assistance,
        "duration_s": DEFAULT_DURATION_S,
        "video": DEFAULT_SAVE_VIDEO,
        "evaluation": False,
        "run_name": label,
        "output_root": condition_root,
        "enforce_regression": False,
    }
    if assistance == "yac":
        kwargs.update(
            {
                "healthy_reference_result": HEALTHY_REFERENCE,
                "controller_mode": "full_yac",
                "active_scale": 1.0,
                "adp_scale": 1.0,
                "event_trigger": True,
                "event_parameters": dict(DEFAULT_EVENT_PARAMETERS),
                "intention_mode": "live_activation",
                "viewer": DEFAULT_FULL_YAC_VIEWER and not headless,
                "viewer_keep_open": DEFAULT_FULL_YAC_VIEWER and not headless,
                "record_git_metadata": False,
            }
        )

    # The public human-only workflow normally points to the bare-Human tuned
    # parameter file.  This benchmark deliberately binds the same frozen 77D
    # Flexible-Exo Human controller in all three conditions for fairness.
    original_human_only_controller = standard_workflow.HEALTHY_CONTROLLER
    try:
        if label == "NO_EXO":
            standard_workflow.HEALTHY_CONTROLLER = standard_workflow.FLEXIBLE_CONTROLLER
        validation = run_human_exo(**kwargs)
    finally:
        standard_workflow.HEALTHY_CONTROLLER = original_human_only_controller

    result_directory = _result_directory(validation)
    ratios = _pathology_ratios(validation)
    manifest = _read_json(result_directory / "manifest.json")
    linked = _link_condition_artifacts(condition_root, result_directory, validation)
    return {
        "label": label,
        "folder": folder,
        "condition_root": str(condition_root),
        "result_directory": str(result_directory),
        "validation": validation,
        "manifest": manifest,
        "pathology_ratios": ratios,
        "linked_artifacts": linked,
    }


def _activation_columns(table: dict[str, np.ndarray]) -> list[str]:
    columns = sorted(name for name in table if name.startswith("muscle_activation_"))
    if not columns:
        raise ValueError("No Human muscle activation channels were saved")
    return columns


def _window_effort(
    table: dict[str, np.ndarray], start_s: float, end_s: float
) -> tuple[float, dict[str, float], dict[str, float]]:
    time_s = _numeric(table, "time")
    columns = _activation_columns(table)
    matrix = np.column_stack([_numeric(table, name) for name in columns])
    mask = (
        np.isfinite(time_s)
        & (time_s >= start_s)
        & (time_s <= end_s)
        & np.all(np.isfinite(matrix), axis=1)
    )
    if np.count_nonzero(mask) < 2:
        raise ValueError("Human-effort window has fewer than two finite samples")
    selected_time = time_s[mask]
    duration = float(selected_time[-1] - selected_time[0])
    if duration <= 0.0:
        raise ValueError("Human-effort window duration is non-positive")
    per_muscle: dict[str, float] = {}
    for index, column in enumerate(columns):
        muscle = column.removeprefix("muscle_activation_")
        per_muscle[muscle] = float(
            np.trapezoid(np.square(matrix[mask, index]), selected_time) / duration
        )
    bilateral: dict[str, float] = {}
    for muscle in per_muscle:
        if not muscle.endswith("_l"):
            continue
        family = muscle[:-2]
        right = f"{family}_r"
        if right in per_muscle:
            bilateral[family] = per_muscle[muscle] + per_muscle[right]
    return float(sum(per_muscle.values())), per_muscle, bilateral


def _strike_times(
    table: dict[str, np.ndarray], side: str, start_s: float, end_s: float
) -> np.ndarray:
    time_s = _numeric(table, "time")
    strikes = _text(table, "heel_strike_side")
    selected = time_s[
        (strikes == side)
        & np.isfinite(time_s)
        & (time_s >= start_s)
        & (time_s <= end_s)
    ]
    return np.asarray(selected, dtype=float)


def _normalize_cycles(
    sample_time_s: np.ndarray,
    boundaries_s: np.ndarray,
    series: dict[str, np.ndarray],
) -> dict[str, Any]:
    grid = np.linspace(0.0, 1.0, 101)
    cycles: dict[str, list[np.ndarray]] = {name: [] for name in series}
    intervals: list[list[float]] = []
    for start, stop in zip(boundaries_s[:-1], boundaries_s[1:]):
        mask = (
            np.isfinite(sample_time_s)
            & (sample_time_s >= start)
            & (sample_time_s < stop)
        )
        if np.count_nonzero(mask) < 5 or stop <= start:
            continue
        phase = (sample_time_s[mask] - start) / (stop - start)
        values_for_cycle: dict[str, np.ndarray] = {}
        valid_cycle = True
        for name, values in series.items():
            selected = values[mask]
            finite = np.isfinite(phase) & np.isfinite(selected)
            if np.count_nonzero(finite) < 5:
                valid_cycle = False
                break
            values_for_cycle[name] = np.interp(grid, phase[finite], selected[finite])
        if not valid_cycle:
            continue
        for name, values in values_for_cycle.items():
            cycles[name].append(values)
        intervals.append([float(start), float(stop)])
    if not intervals:
        return {
            "available": False,
            "cycle_count": 0,
            "phase_percent": (100.0 * grid).tolist(),
            "intervals_s": [],
            "waveforms": {},
            "_cycles": {},
        }
    matrices = {name: np.vstack(values) for name, values in cycles.items()}
    return {
        "available": True,
        "cycle_count": len(intervals),
        "phase_percent": (100.0 * grid).tolist(),
        "intervals_s": intervals,
        "waveforms": {
            name: {
                "mean": np.mean(matrix, axis=0).tolist(),
                "std": np.std(matrix, axis=0).tolist(),
            }
            for name, matrix in matrices.items()
        },
        "_cycles": matrices,
    }


def _gait_profiles(
    table: dict[str, np.ndarray], start_s: float, end_s: float
) -> dict[str, Any]:
    time_s = _numeric(table, "time")
    activation = _activation_columns(table)
    result: dict[str, Any] = {}
    for side in ("left", "right"):
        suffix = "l" if side == "left" else "r"
        columns = [
            csv_column
            for label, (_, csv_column) in HUMAN_JOINT_COLUMNS.items()
            if label.endswith(f"_{suffix}")
        ]
        columns.extend(name for name in activation if name.endswith(f"_{suffix}"))
        for motor in MOTOR_CHANNELS:
            name = f"exo_actual_torque_{motor}_nm"
            if name in table and np.any(np.isfinite(_numeric(table, name))):
                columns.append(name)
        boundaries = _strike_times(table, side, start_s, end_s)
        result[side] = _normalize_cycles(
            time_s, boundaries, {name: _numeric(table, name) for name in columns}
        )
    return result


def _human_gait_rmse(
    profiles: dict[str, Any], reference_profiles: dict[str, Any]
) -> dict[str, Any]:
    per_joint: dict[str, float | None] = {}
    errors: list[np.ndarray] = []
    for label, (_, column) in HUMAN_JOINT_COLUMNS.items():
        side = "left" if label.endswith("_l") else "right"
        current = profiles[side]["waveforms"].get(column, {}).get("mean")
        reference = reference_profiles[side]["waveforms"].get(column, {}).get("mean")
        if current is None or reference is None:
            per_joint[label] = None
            continue
        difference = np.asarray(current, dtype=float) - np.asarray(reference, dtype=float)
        finite = difference[np.isfinite(difference)]
        value = float(np.sqrt(np.mean(np.square(finite)))) if finite.size else None
        per_joint[label] = value
        if finite.size:
            errors.append(finite)
    return {
        "overall_rad": (
            float(np.sqrt(np.mean(np.square(np.concatenate(errors)))))
            if errors else None
        ),
        "per_joint_rad": per_joint,
        "reference": str(HEALTHY_REFERENCE),
        "semantics": "RMSE of 101-point complete-stride Human joint mean profiles",
    }


def _stability_metrics(table: dict[str, np.ndarray]) -> dict[str, float]:
    velocity = _numeric(table, "forward_velocity")
    pitch = _numeric(table, "pelvis_pitch")
    height = _numeric(table, "pelvis_height")
    return {
        "average_speed_mps": float(np.nanmean(velocity)),
        # runner.ExperimentRecorder stores pelvis_pitch in degrees.
        "max_abs_pelvis_pitch_deg": float(np.nanmax(np.abs(pitch))),
        "min_pelvis_height_m": float(np.nanmin(height)),
    }


def _m5_metrics(
    result_directory: Path,
    timeseries: dict[str, np.ndarray],
    start_s: float,
    end_s: float,
) -> dict[str, Any]:
    physics_path = result_directory / "physics_timeseries_1khz.csv"
    physics = metrics.load_timeseries(physics_path)
    time_s = _numeric(physics, "time")
    columns = {label: f"coupling_qfrc_{label}_nm" for label in M5_LABELS}
    matrix = np.column_stack([_numeric(physics, name) for name in columns.values()])
    mask = (
        np.isfinite(time_s)
        & (time_s >= start_s)
        & (time_s <= end_s)
        & np.all(np.isfinite(matrix), axis=1)
    )
    if np.count_nonzero(mask) < 2:
        raise ValueError("M5 HRI window has fewer than two finite samples")
    selected_time = time_s[mask]
    selected = matrix[mask]
    power = np.sum(np.square(selected), axis=1)
    cumulative = np.zeros_like(power)
    cumulative[1:] = np.cumsum(
        0.5 * (power[1:] + power[:-1]) * np.diff(selected_time)
    )
    boundaries = _strike_times(timeseries, "left", start_s, end_s)
    cycle_series = {
        label: matrix[:, index] for index, label in enumerate(M5_LABELS)
    }
    cycle = _normalize_cycles(time_s, boundaries, cycle_series)
    if cycle["available"]:
        stacked = np.stack([cycle["_cycles"][label] for label in M5_LABELS], axis=2)
        norm_cycles = np.linalg.norm(stacked, axis=2)
        cycle["total_norm_mean_Nm"] = np.mean(norm_cycles, axis=0).tolist()
        cycle["total_norm_std_Nm"] = np.std(norm_cycles, axis=0).tolist()
    return {
        "HRI_RMS_total_Nm": float(np.sqrt(np.mean(power))),
        "HRI_RMS_per_joint_Nm": {
            label: float(np.sqrt(np.mean(np.square(selected[:, index]))))
            for index, label in enumerate(M5_LABELS)
        },
        "E_HRI_M5_N2m2s": float(cumulative[-1]),
        "evaluation_window_s": [float(selected_time[0]), float(selected_time[-1])],
        "gait_cycle": cycle,
        "_time_s": selected_time,
        "_cumulative": cumulative,
    }


def _motor_metrics(
    table: dict[str, np.ndarray], start_s: float, end_s: float
) -> dict[str, Any]:
    time_s = _numeric(table, "time")
    mask = np.isfinite(time_s) & (time_s >= start_s) & (time_s <= end_s)
    torque_arrays: list[np.ndarray] = []
    per_motor: dict[str, Any] = {}
    positive_work = 0.0
    negative_work = 0.0
    net_work = 0.0
    absolute_work = 0.0
    for motor in MOTOR_CHANNELS:
        torque = _numeric(table, f"exo_actual_torque_{motor}_nm")
        velocity = _numeric(table, f"exo_joint_velocity_{motor}_rad_s")
        valid = mask & np.isfinite(torque) & np.isfinite(velocity)
        selected_torque = torque[valid]
        selected_time = time_s[valid]
        power = selected_torque * velocity[valid]
        positive = float(np.trapezoid(np.maximum(power, 0.0), selected_time))
        negative = float(np.trapezoid(np.minimum(power, 0.0), selected_time))
        net = float(np.trapezoid(power, selected_time))
        absolute = float(np.trapezoid(np.abs(power), selected_time))
        positive_work += positive
        negative_work += negative
        net_work += net
        absolute_work += absolute
        torque_arrays.append(selected_torque)
        per_motor[motor] = {
            "torque_RMS_Nm": float(np.sqrt(np.mean(np.square(selected_torque)))),
            "torque_peak_abs_Nm": float(np.max(np.abs(selected_torque))),
            "positive_work_J": positive,
            "negative_work_J": negative,
            "net_work_J": net,
            "absolute_work_J": absolute,
        }
    rectangular = np.column_stack(torque_arrays)
    return {
        "per_motor": per_motor,
        "overall_torque_RMS_Nm": float(np.sqrt(np.mean(np.square(rectangular)))),
        "torque_peak_abs_Nm": float(np.max(np.abs(rectangular))),
        "positive_mechanical_work_J": positive_work,
        "negative_mechanical_work_J": negative_work,
        "net_mechanical_work_J": net_work,
        "absolute_mechanical_work_J": absolute_work,
    }


def _controller_metrics(result_directory: Path) -> dict[str, Any]:
    diagnostics = metrics.load_timeseries(result_directory / "controller_diagnostics.csv")
    bellman = _numeric(diagnostics, "bellman_global")
    bellman = bellman[np.isfinite(bellman)]
    intervals = _numeric(diagnostics, "inter_event_global_s")
    intervals = intervals[np.isfinite(intervals)]
    left = _numeric(diagnostics, "critic_weight_norm_l")
    right = _numeric(diagnostics, "critic_weight_norm_r")
    event_count = _numeric(diagnostics, "event_count_global")
    event_fraction = _numeric(diagnostics, "event_fraction_global")
    return {
        "Global_Bellman_RMS": float(np.sqrt(np.mean(np.square(bellman)))),
        "final_critic_norm_L": float(left[np.isfinite(left)][-1]),
        "final_critic_norm_R": float(right[np.isfinite(right)][-1]),
        "Event_count": int(event_count[np.isfinite(event_count)][-1]),
        "Event_fraction": float(event_fraction[np.isfinite(event_fraction)][-1]),
        "Policy_refresh_reduction_percent": float(
            100.0 * (1.0 - event_fraction[np.isfinite(event_fraction)][-1])
        ),
        "mean_inter_event_s": float(np.mean(intervals)),
        "median_inter_event_s": float(np.median(intervals)),
    }


def _initial_human_state(table: dict[str, np.ndarray]) -> dict[str, float]:
    columns = (
        "forward_position", "forward_velocity", "pelvis_height", "pelvis_pitch",
        "left_hip_angle_rad", "left_hip_velocity_rad_s",
        "left_knee_angle_rad", "left_knee_velocity_rad_s",
        "right_hip_angle_rad", "right_hip_velocity_rad_s",
        "right_knee_angle_rad", "right_knee_velocity_rad_s",
    )
    return {name: float(_numeric(table, name)[0]) for name in columns}


def _fairness_audit(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    manifests = {name: run["manifest"] for name, run in runs.items()}
    shared_fields = (
        "human_xml_sha256", "patient_config_sha256",
        "human_controller_parameter_sha256", "control_timestep_s",
        "mujoco_timestep_s", "seed",
    )
    shared_checks = {
        field: len({manifest.get(field) for manifest in manifests.values()}) == 1
        for field in shared_fields
    }
    pathology_checks = {
        name: all(
            math.isclose(value, 0.6, abs_tol=1.0e-12)
            for value in run["pathology_ratios"].values()
        )
        for name, run in runs.items()
    }
    exo_pair_fields = ("coupled_xml_sha256", "exo_model_version", "exo_mass_kg", "coupling_version")
    exo_pair_checks = {
        field: manifests["EXO_ZERO"].get(field) == manifests["FULL_YAC"].get(field)
        for field in exo_pair_fields
    }
    states = {name: _initial_human_state(run["table"]) for name, run in runs.items()}
    reference = states["NO_EXO"]
    initial_differences = {
        name: max(abs(values[key] - reference[key]) for key in reference)
        for name, values in states.items()
    }
    initial_state_check = all(value <= 1.0e-12 for value in initial_differences.values())
    checks = {
        **{f"shared_{key}": value for key, value in shared_checks.items()},
        **{f"pathology_{key}": value for key, value in pathology_checks.items()},
        **{f"exo_pair_{key}": value for key, value in exo_pair_checks.items()},
        "initial_shared_human_state": initial_state_check,
        "exo_zero_actual_commanded_torque_exactly_zero": all(
            runs["EXO_ZERO"]["validation"]["exoskeleton_channels_all_zero"][name]
            is True
            for name in MOTOR_CHANNELS
        ),
        "full_yac_uses_controller_source_of_truth": (
            standard_workflow._make_flexible_config(
                standard_workflow._condition("moderate60"),
                assistance="yac", duration_s=DEFAULT_DURATION_S,
                video=False, run_name="BENCHMARK_SOURCE_AUDIT",
                healthy_reference_result=HEALTHY_REFERENCE,
                controller_mode="full_yac", active_scale=1.0, adp_scale=1.0,
                event_trigger=True, event_parameters=dict(DEFAULT_EVENT_PARAMETERS),
                intention_mode="live_activation",
                output_root=PROJECT_ROOT / "results/runs/YAC_benchmark/DRY_RUN_NOT_CREATED",
                viewer=False, viewer_keep_open=False, record_git_metadata=False,
            ).ASSIST_CONTROLLER
            is YACController
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "shared_fields": {field: manifests["NO_EXO"].get(field) for field in shared_fields},
        "human_controller_file": manifests["NO_EXO"].get("human_controller_parameter_path"),
        "human_controller_sha256": manifests["NO_EXO"].get("human_controller_parameter_sha256"),
        "initial_human_state": states,
        "initial_state_max_abs_difference": initial_differences,
        "human_joint_state_only": True,
        "no_full_qpos_hash_requirement": True,
        "human_models": {name: manifest.get("human_model") for name, manifest in manifests.items()},
        "exo_models": {name: manifest.get("exo_model") for name, manifest in manifests.items()},
        "M5_version": manifests["EXO_ZERO"].get("coupling_version"),
        "exo_mass_kg": manifests["EXO_ZERO"].get("exo_mass_kg"),
        "reference_source": str(HEALTHY_REFERENCE),
    }


def _video_metadata(path: Path, survival_s: float) -> dict[str, Any]:
    count, duration = imageio_ffmpeg.count_frames_and_secs(str(path))
    reader = imageio.get_reader(path)
    try:
        metadata = reader.get_meta_data()
        first = reader.get_data(0)
    finally:
        reader.close()
    fps = float(metadata.get("fps", count / duration if duration > 0.0 else math.nan))
    result = {
        "path": str(path),
        "decode_succeeds": bool(first.size > 0),
        "frame_count": int(count),
        "duration_s": float(duration),
        "fps": fps,
        "size": [int(first.shape[1]), int(first.shape[0])],
        "duration_matches_survival": bool(abs(duration - survival_s) <= 0.25),
    }
    result["passed"] = bool(
        result["decode_succeeds"]
        and result["frame_count"] > 0
        and math.isfinite(fps)
        and fps > 0.0
        and result["duration_matches_survival"]
    )
    return result


def _comparison_video(
    output: Path, videos: list[tuple[str, Path, float]]
) -> dict[str, Any]:
    readers = [imageio.get_reader(path) for _, path, _ in videos]
    try:
        metadata = [reader.get_meta_data() for reader in readers]
        fps = min(float(item.get("fps", 25.0)) for item in metadata)
        counts = [int(imageio_ffmpeg.count_frames_and_secs(str(path))[0]) for _, path, _ in videos]
        frame_count = max(counts)
        panel_size = (480, 272)
        font = ImageFont.load_default()
        writer = imageio.get_writer(
            output, fps=fps, codec="libx264", quality=7, macro_block_size=16
        )
        last_frames: list[Image.Image | None] = [None, None, None]
        try:
            for index in range(frame_count):
                canvas = Image.new("RGB", (panel_size[0] * 3, panel_size[1] + 64), "white")
                draw = ImageDraw.Draw(canvas)
                for panel, ((label, _, survival), reader, count) in enumerate(
                    zip(videos, readers, counts)
                ):
                    terminated = index >= count
                    if not terminated:
                        frame = Image.fromarray(reader.get_data(index)).convert("RGB")
                        last_frames[panel] = frame
                    elif last_frames[panel] is not None:
                        frame = ImageEnhance.Brightness(last_frames[panel]).enhance(0.45)
                    else:
                        frame = Image.new("RGB", panel_size, "black")
                    frame.thumbnail(panel_size, Image.Resampling.LANCZOS)
                    x0 = panel * panel_size[0]
                    y0 = 64 + (panel_size[1] - frame.height) // 2
                    canvas.paste(frame, (x0 + (panel_size[0] - frame.width) // 2, y0))
                    sim_time = min(index / fps, survival)
                    status = " | TERMINATED" if terminated else ""
                    draw.text(
                        (x0 + 8, 8),
                        f"{label} | survival {survival:.2f}s{status}\nsim time {sim_time:.2f}s",
                        fill="black", font=font,
                    )
                writer.append_data(np.asarray(canvas))
        finally:
            writer.close()
    finally:
        for reader in readers:
            reader.close()
    return _video_metadata(output, max(item[2] for item in videos))


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Figure was not created: {path}")


def _generate_figures(
    figures_dir: Path,
    conditions: dict[str, dict[str, Any]],
    profiles: dict[str, Any],
    reference_profiles: dict[str, Any],
    family_effort: dict[str, dict[str, float]],
    m5: dict[str, Any],
    motor: dict[str, Any],
    controller: dict[str, Any],
    effort_changes: dict[str, float | None],
) -> dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    order = ("NO_EXO", "EXO_ZERO", "FULL_YAC")
    labels = ("No Exo", "Exo Zero", "Full YAC")
    paths: dict[str, str] = {}

    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    survival = [conditions[name]["survival_time_s"] for name in order]
    axes[0, 0].bar(labels, survival)
    axes[0, 0].axhline(DEFAULT_DURATION_S, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_title("A. Survival")
    axes[0, 0].set_ylabel("s")
    effort = [conditions[name]["Ja_common"] for name in order]
    axes[0, 1].bar(labels, effort)
    axes[0, 1].set_title("B. Human Effort Proxy")
    axes[0, 1].set_ylabel("Ja common")
    axes[0, 1].text(
        0.02, 0.96,
        f"YAC vs NoExo: {effort_changes['vs_noexo']:+.2f}%\nYAC vs ExoZero: {effort_changes['vs_exozero']:+.2f}%",
        transform=axes[0, 1].transAxes, va="top", fontsize=8,
    )
    gait = [conditions[name]["human_gait_RMSE"]["overall_rad"] for name in order]
    axes[0, 2].bar(labels, [np.nan if value is None else value for value in gait])
    axes[0, 2].set_title("C. Human gait vs Healthy")
    axes[0, 2].set_ylabel("RMSE (rad)")
    axes[1, 0].bar(
        ("Exo Zero", "Full YAC"),
        (m5["EXO_ZERO"]["E_HRI_M5_N2m2s"], m5["FULL_YAC"]["E_HRI_M5_N2m2s"]),
    )
    axes[1, 0].set_title("D. Physical M5 HRI")
    axes[1, 0].set_ylabel(r"$E_{HRI,M5}$ ($N^2m^2s$)")
    motor_names = [name.replace("Exo_", "") for name in MOTOR_CHANNELS]
    rms = [motor["per_motor"][name]["torque_RMS_Nm"] for name in MOTOR_CHANNELS]
    peak = [motor["per_motor"][name]["torque_peak_abs_Nm"] for name in MOTOR_CHANNELS]
    x = np.arange(4)
    axes[1, 1].bar(x - 0.18, rms, 0.36, label="RMS")
    axes[1, 1].bar(x + 0.18, peak, 0.36, label="Peak")
    axes[1, 1].set_xticks(x, motor_names, rotation=25)
    axes[1, 1].set_title("E. Full YAC motor torque")
    axes[1, 1].set_ylabel("Nm")
    axes[1, 1].legend()
    axes[1, 2].bar(
        ("Event fraction", "Refresh reduction"),
        (100.0 * controller["Event_fraction"], controller["Policy_refresh_reduction_percent"]),
    )
    axes[1, 2].set_title("F. Event policy")
    axes[1, 2].set_ylabel("%")
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
    main = figures_dir / "yac_benchmark_main.png"
    _save_figure(figure, main)
    paths["main"] = str(main)

    top_families = sorted(
        family_effort["FULL_YAC"],
        key=lambda name: max(family_effort[item][name] for item in order),
        reverse=True,
    )[:8]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(top_families))
    width = 0.25
    for index, name in enumerate(order):
        axes[0].bar(
            x + (index - 1) * width,
            [family_effort[name][family] for family in top_families],
            width, label=labels[index],
        )
    axes[0].set_xticks(x, top_families, rotation=40, ha="right")
    axes[0].set_title("A. Bilateral family effort")
    axes[0].set_ylabel(r"mean($a_i^2$), summed L/R")
    axes[0].legend(fontsize=8)
    for axis, baseline, title in (
        (axes[1], "NO_EXO", "B. Full YAC vs No Exo"),
        (axes[2], "EXO_ZERO", "C. Full YAC vs Exo Zero"),
    ):
        changes = [
            _percent_change(family_effort[baseline][family], family_effort["FULL_YAC"][family])
            for family in top_families
        ]
        axis.bar(top_families, changes)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.tick_params(axis="x", rotation=40)
        axis.set_title(title)
        axis.set_ylabel("Change (%)")
    muscle = figures_dir / "yac_benchmark_muscle_effort.png"
    _save_figure(figure, muscle)
    paths["muscle"] = str(muscle)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    phase = np.linspace(0.0, 100.0, 101)
    for axis, family in zip(axes.flat, ("vasti", "iliopsoas", "bifemsh", "hamstrings")):
        column = f"muscle_activation_{family}_l"
        for name, label in zip(order, labels):
            waveform = profiles[name]["left"]["waveforms"].get(column)
            if waveform is None:
                continue
            mean = np.asarray(waveform["mean"])
            std = np.asarray(waveform["std"])
            line = axis.plot(phase, mean, label=label)[0]
            axis.fill_between(phase, mean - std, mean + std, color=line.get_color(), alpha=0.10)
        axis.set_title(family)
        axis.set_xlabel("Gait cycle (%)")
        axis.set_ylabel("Simulation activation")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    activation = figures_dir / "yac_benchmark_gait_activation.png"
    _save_figure(figure, activation)
    paths["activation"] = str(activation)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, (joint, (_, column)) in zip(axes.flat, HUMAN_JOINT_COLUMNS.items()):
        side = "left" if joint.endswith("_l") else "right"
        reference = reference_profiles[side]["waveforms"].get(column)
        if reference is not None:
            axis.plot(phase, reference["mean"], color="black", linestyle="--", label="Healthy")
        for name, label in zip(order, labels):
            waveform = profiles[name][side]["waveforms"].get(column)
            if waveform is not None:
                axis.plot(phase, waveform["mean"], label=label)
        axis.set_title(joint.replace("_", " ").title())
        axis.set_xlabel("Gait cycle (%)")
        axis.set_ylabel("Human joint angle (rad)")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    kinematics = figures_dir / "yac_benchmark_gait_kinematics.png"
    _save_figure(figure, kinematics)
    paths["kinematics"] = str(kinematics)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for name, label in (("EXO_ZERO", "Exo Zero"), ("FULL_YAC", "Full YAC")):
        gait_cycle = m5[name]["gait_cycle"]
        if gait_cycle["available"]:
            axes[0, 0].plot(phase, gait_cycle["total_norm_mean_Nm"], label=label)
        axes[0, 1].plot(
            m5[name]["_time_s"] - DEFAULT_STARTUP_CUTOFF_S,
            m5[name]["_cumulative"], label=label,
        )
    axes[0, 0].set_title("A. M5 generalized torque norm")
    axes[0, 0].set_xlabel("Gait cycle (%)")
    axes[0, 0].set_ylabel("Nm")
    axes[0, 0].legend()
    axes[0, 1].set_title("B. Cumulative physical HRI cost")
    axes[0, 1].set_xlabel("Time after cutoff (s)")
    axes[0, 1].set_ylabel(r"$E_{HRI,M5}$ ($N^2m^2s$)")
    axes[0, 1].legend()
    x = np.arange(4)
    for offset, (name, label) in zip((-0.18, 0.18), (("EXO_ZERO", "Exo Zero"), ("FULL_YAC", "Full YAC"))):
        axes[1, 0].bar(
            x + offset,
            [m5[name]["HRI_RMS_per_joint_Nm"][item] for item in M5_LABELS],
            0.36, label=label,
        )
    axes[1, 0].set_xticks(x, M5_LABELS, rotation=25)
    axes[1, 0].set_title("C. Per-joint physical HRI RMS")
    axes[1, 0].set_ylabel("Nm")
    axes[1, 0].legend()
    full_left = profiles["FULL_YAC"]["left"]["waveforms"]
    for motor in MOTOR_CHANNELS:
        waveform = full_left.get(f"exo_actual_torque_{motor}_nm")
        if waveform is not None:
            axes[1, 1].plot(phase, waveform["mean"], label=motor)
    axes[1, 1].set_title("D. Full YAC motor torque")
    axes[1, 1].set_xlabel("Left gait cycle (%)")
    axes[1, 1].set_ylabel("Nm")
    axes[1, 1].legend(fontsize=7, ncol=2)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    hri = figures_dir / "yac_benchmark_physical_hri.png"
    _save_figure(figure, hri)
    paths["HRI"] = str(hri)
    return paths


def _strip_private(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_private(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, list):
        return [_strip_private(item) for item in value]
    return value


def _existing_ablation() -> dict[str, Any] | None:
    if not EXISTING_INTENTION_ABLATION.is_file():
        return None
    value = _read_json(EXISTING_INTENTION_ABLATION)
    return {
        "source": str(EXISTING_INTENTION_ABLATION),
        "tracking_improvement_percent": value.get("q_rms", {}).get("tracking_change_percent"),
        "Paper_HRI_reduction_percent": value.get("Paper_HRI_reduction_percent"),
        "constant_survival_s": value.get("survival", {}).get("baseline_s"),
        "live_survival_s": value.get("survival", {}).get("yac_s"),
        "rerun": False,
    }


def _write_summary_csv(path: Path, conditions: dict[str, dict[str, Any]]) -> None:
    columns = (
        "condition", "survival", "termination_reason", "speed", "pitch", "height",
        "Ja_common", "Ja_full", "human_q_rmse", "HRI_RMS", "E_HRI_M5",
        "torque_rms", "torque_peak", "event_fraction",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for name in ("NO_EXO", "EXO_ZERO", "FULL_YAC"):
            item = conditions[name]
            writer.writerow(
                {
                    "condition": name,
                    "survival": item["survival_time_s"],
                    "termination_reason": item["termination_reason"],
                    "speed": item["average_speed_mps"],
                    "pitch": item["max_abs_pelvis_pitch_deg"],
                    "height": item["min_pelvis_height_m"],
                    "Ja_common": item["Ja_common"],
                    "Ja_full": "" if item["Ja_full"] is None else item["Ja_full"],
                    "human_q_rmse": item["human_gait_RMSE"]["overall_rad"],
                    "HRI_RMS": "" if item["M5_HRI"] is None else item["M5_HRI"]["HRI_RMS_total_Nm"],
                    "E_HRI_M5": "" if item["M5_HRI"] is None else item["M5_HRI"]["E_HRI_M5_N2m2s"],
                    "torque_rms": "" if item["motor_assistance"] is None else item["motor_assistance"]["overall_torque_RMS_Nm"],
                    "torque_peak": "" if item["motor_assistance"] is None else item["motor_assistance"]["torque_peak_abs_Nm"],
                    "event_fraction": "" if item["ADP_event"] is None else item["ADP_event"]["Event_fraction"],
                }
            )


def _format(value: Any, digits: int = 4) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def _print_final(summary: dict[str, Any]) -> None:
    conditions = summary["conditions"]
    flags = summary["flags"]
    print("\n==========================================")
    print("YAC FINAL ASSISTANCE BENCHMARK")
    print("==========================================")
    print("\nPathology:\nbilateral vasti = 60%")
    print(
        f"\nCommon window:\n{summary['common_window_s'][0]:.2f} s -> "
        f"{summary['common_window_s'][1]:.2f} s"
    )
    print("\n[STABILITY]")
    for name in ("NO_EXO", "EXO_ZERO", "FULL_YAC"):
        item = conditions[name]
        print(f"\n{name}\nSurvival: {item['survival_time_s']:.2f} s\nReason: {item['termination_reason']}")
    print("\n[HUMAN EFFORT PROXY]")
    for name in ("NO_EXO", "EXO_ZERO", "FULL_YAC"):
        print(f"{name} Ja: {conditions[name]['Ja_common']:.6f}")
    print(f"YAC vs NO EXO: {summary['effort_change_percent']['vs_noexo']:+.2f}%")
    print(f"YAC vs EXO ZERO: {summary['effort_change_percent']['vs_exozero']:+.2f}%")
    print("\n[GAIT]")
    for name in ("NO_EXO", "EXO_ZERO", "FULL_YAC"):
        print(f"{name} Human q RMSE: {_format(conditions[name]['human_gait_RMSE']['overall_rad'], 6)} rad")
    print("\n[M5 PHYSICAL HRI]")
    print(f"EXO ZERO: {conditions['EXO_ZERO']['M5_HRI']['E_HRI_M5_N2m2s']:.3f}")
    print(f"FULL YAC: {conditions['FULL_YAC']['M5_HRI']['E_HRI_M5_N2m2s']:.3f}")
    print(f"Physical HRI change: {summary['physical_HRI_change_percent']:+.3f}%")
    motor = conditions["FULL_YAC"]["motor_assistance"]
    print("\n[MOTOR ASSISTANCE]")
    print(f"Torque RMS: {motor['overall_torque_RMS_Nm']:.4f} Nm")
    print(f"Peak: {motor['torque_peak_abs_Nm']:.4f} Nm")
    print(f"Absolute mechanical work: {motor['absolute_mechanical_work_J']:.4f} J")
    adp = conditions["FULL_YAC"]["ADP_event"]
    print("\n[ADP / EVENT]")
    print(f"Bellman RMS: {adp['Global_Bellman_RMS']:.6f}")
    print(f"Event fraction: {adp['Event_fraction']:.6f}")
    print(f"Policy refresh reduction: {adp['Policy_refresh_reduction_percent']:.3f}%")
    existing = summary["existing_intention_ablation"]
    print("\n[EXISTING INTENTION ABLATION]")
    if existing is None:
        print("Not available")
    else:
        print(f"Paper HRI reduction: {_format(existing['Paper_HRI_reduction_percent'], 3)}%")
        print(f"Tracking improvement: {_format(existing['tracking_improvement_percent'], 3)}%")
    print("\n[FINAL FLAGS]")
    for name, value in flags.items():
        print(f"{name}: {value}")
    print("\n[ARTIFACTS]")
    print(f"Benchmark root: {summary['paths']['benchmark_root']}")
    print(f"Summary JSON: {summary['paths']['summary_json']}")
    print(f"Summary CSV: {summary['paths']['summary_csv']}")
    for name, path in summary["figures"].items():
        print(f"{name.title()} figure: {path}")
    for name, item in summary["videos"].items():
        print(f"{name}: {item['path'] if item else 'N/A'}")
    print("==========================================")


def main() -> int:
    if DEFAULT_DURATION_S <= DEFAULT_STARTUP_CUTOFF_S:
        raise ValueError("Benchmark duration must exceed startup cutoff")
    if not HEALTHY_REFERENCE.is_dir():
        raise FileNotFoundError(f"Healthy reference is missing: {HEALTHY_REFERENCE}")
    controller_file = standard_workflow.FLEXIBLE_CONTROLLER.resolve()
    if not controller_file.is_file():
        raise FileNotFoundError(f"Frozen Human controller is missing: {controller_file}")
    human_joint_mapping = _validate_human_joint_mapping()
    headless = os.environ.get("YAC_BENCHMARK_HEADLESS") == "1"
    benchmark_root = _new_benchmark_root()
    print(f"Benchmark root: {benchmark_root}")
    print(f"Human controller: {controller_file}")
    print(f"Human controller SHA256: {_sha256(controller_file)}")

    runs: dict[str, dict[str, Any]] = {}
    for index, condition in enumerate(CONDITIONS, start=1):
        label, folder, exoskeleton, assistance = condition
        print("\n--------------------------------")
        print(f"RUN {index} / 3")
        print(label.replace("_", " "))
        print("--------------------------------", flush=True)
        runs[label] = _run_condition(
            label, folder, exoskeleton, assistance, benchmark_root, headless
        )
        validation = runs[label]["validation"]
        print(f"vasti_l ratio: {runs[label]['pathology_ratios']['vasti_l']:.6f}")
        print(f"vasti_r ratio: {runs[label]['pathology_ratios']['vasti_r']:.6f}")
        print(f"Survival: {float(validation['actual_duration_s']):.3f} s")
        print(f"Termination: {validation['termination_reason']}", flush=True)

    durations = [float(run["validation"]["actual_duration_s"]) for run in runs.values()]
    common_end = min(*durations, DEFAULT_DURATION_S)
    if common_end <= DEFAULT_STARTUP_CUTOFF_S:
        raise RuntimeError("Benchmark invalid: common window ends before startup cutoff")

    healthy_table = metrics.load_timeseries(HEALTHY_REFERENCE / "timeseries.csv")
    healthy_end = min(
        float(_read_json(HEALTHY_REFERENCE / "validation.json")["actual_duration_s"]),
        DEFAULT_DURATION_S,
    )
    reference_profiles = _gait_profiles(
        healthy_table, DEFAULT_STARTUP_CUTOFF_S, healthy_end
    )
    conditions: dict[str, dict[str, Any]] = {}
    profiles: dict[str, Any] = {}
    per_muscle: dict[str, dict[str, float]] = {}
    family_effort: dict[str, dict[str, float]] = {}
    for name, run in runs.items():
        result_directory = Path(run["result_directory"])
        table = metrics.load_timeseries(result_directory / "timeseries.csv")
        run["table"] = table
        validation = run["validation"]
        survival = float(validation["actual_duration_s"])
        stable_full = bool(
            survival >= DEFAULT_DURATION_S - 1.0e-6
            and validation["termination_reason"] == "time_limit"
            and validation["finite_state"] is True
        )
        ja_common, muscle_common, family_common = _window_effort(
            table, DEFAULT_STARTUP_CUTOFF_S, common_end
        )
        ja_full = None
        if stable_full:
            ja_full, _, _ = _window_effort(table, 0.0, DEFAULT_DURATION_S)
        profiles[name] = _gait_profiles(table, DEFAULT_STARTUP_CUTOFF_S, common_end)
        gait_rmse = _human_gait_rmse(profiles[name], reference_profiles)
        stability = _stability_metrics(table)
        per_muscle[name] = muscle_common
        family_effort[name] = family_common
        conditions[name] = {
            "result_directory": str(result_directory),
            "survival_time_s": survival,
            "termination_reason": validation["termination_reason"],
            "finite_state": validation["finite_state"],
            **stability,
            "Ja_full": ja_full,
            "Ja_common": ja_common,
            "per_muscle_Ja_common": muscle_common,
            "muscle_family_bilateral_Ja_common": family_common,
            "gait_cycle": _strip_private(profiles[name]),
            "human_gait_RMSE": gait_rmse,
            "M5_HRI": None,
            "motor_assistance": None,
            "ADP_event": None,
        }

    if not all(set(per_muscle[name]) == set(per_muscle["NO_EXO"]) for name in per_muscle):
        raise RuntimeError("Human muscle actuator sets differ across benchmark conditions")

    m5_end = min(
        conditions["EXO_ZERO"]["survival_time_s"],
        conditions["FULL_YAC"]["survival_time_s"],
        DEFAULT_DURATION_S,
    )
    m5_values = {
        name: _m5_metrics(
            Path(runs[name]["result_directory"]), runs[name]["table"],
            DEFAULT_STARTUP_CUTOFF_S, m5_end,
        )
        for name in ("EXO_ZERO", "FULL_YAC")
    }
    conditions["EXO_ZERO"]["M5_HRI"] = _strip_private(m5_values["EXO_ZERO"])
    conditions["FULL_YAC"]["M5_HRI"] = _strip_private(m5_values["FULL_YAC"])
    motor = _motor_metrics(runs["FULL_YAC"]["table"], 0.0, conditions["FULL_YAC"]["survival_time_s"])
    controller = _controller_metrics(Path(runs["FULL_YAC"]["result_directory"]))
    conditions["FULL_YAC"]["motor_assistance"] = motor
    conditions["FULL_YAC"]["ADP_event"] = controller

    effort_changes = {
        "vs_noexo": _percent_change(conditions["NO_EXO"]["Ja_common"], conditions["FULL_YAC"]["Ja_common"]),
        "vs_exozero": _percent_change(conditions["EXO_ZERO"]["Ja_common"], conditions["FULL_YAC"]["Ja_common"]),
    }
    physical_hri_change = _percent_change(
        m5_values["EXO_ZERO"]["E_HRI_M5_N2m2s"],
        m5_values["FULL_YAC"]["E_HRI_M5_N2m2s"],
    )
    fairness = _fairness_audit(runs)
    if not fairness["passed"]:
        raise RuntimeError(f"Benchmark fairness audit failed: {fairness['checks']}")

    videos: dict[str, Any] = {}
    for name in ("NO_EXO", "EXO_ZERO", "FULL_YAC"):
        path = Path(runs[name]["condition_root"]) / "video.mp4"
        videos[name] = _video_metadata(path, conditions[name]["survival_time_s"])
        if not videos[name]["passed"]:
            raise RuntimeError(f"Video validation failed for {name}: {videos[name]}")
    comparison_path = benchmark_root / "YAC_benchmark_comparison.mp4"
    if DEFAULT_GENERATE_COMPARISON_VIDEO:
        videos["comparison"] = _comparison_video(
            comparison_path,
            [
                (name.replace("_", " "), Path(videos[name]["path"]), conditions[name]["survival_time_s"])
                for name in ("NO_EXO", "EXO_ZERO", "FULL_YAC")
            ],
        )
        if not videos["comparison"]["passed"]:
            raise RuntimeError(f"Comparison-video validation failed: {videos['comparison']}")
    else:
        videos["comparison"] = None

    figures = {}
    if DEFAULT_GENERATE_FIGURES:
        figures = _generate_figures(
            benchmark_root / "figures", conditions, profiles, reference_profiles,
            family_effort, m5_values, motor, controller, effort_changes,
        )
    flags = {
        "YAC_STABLE_20S": conditions["FULL_YAC"]["survival_time_s"] >= DEFAULT_DURATION_S - 1.0e-6,
        "YAC_RESCUES_EXO_LOADED_GAIT": (
            conditions["EXO_ZERO"]["survival_time_s"] < DEFAULT_DURATION_S - 1.0e-6
            and conditions["FULL_YAC"]["survival_time_s"] >= DEFAULT_DURATION_S - 1.0e-6
        ),
        "YAC_REDUCES_EFFORT_VS_NOEXO": effort_changes["vs_noexo"] is not None and effort_changes["vs_noexo"] < 0.0,
        "YAC_REDUCES_EFFORT_VS_EXOZERO": effort_changes["vs_exozero"] is not None and effort_changes["vs_exozero"] < 0.0,
        "YAC_REDUCES_PHYSICAL_HRI_VS_EXOZERO": physical_hri_change is not None and physical_hri_change < 0.0,
        "YAC_IMPROVES_HUMAN_GAIT_VS_EXOZERO": (
            conditions["FULL_YAC"]["human_gait_RMSE"]["overall_rad"] is not None
            and conditions["EXO_ZERO"]["human_gait_RMSE"]["overall_rad"] is not None
            and conditions["FULL_YAC"]["human_gait_RMSE"]["overall_rad"]
            < conditions["EXO_ZERO"]["human_gait_RMSE"]["overall_rad"]
        ),
    }

    summary_dir = benchmark_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_json = summary_dir / "yac_benchmark_summary.json"
    summary_csv = summary_dir / "yac_benchmark_summary.csv"
    summary = {
        "schema_version": "YAC_FINAL_ASSISTANCE_BENCHMARK_V1",
        "benchmark_status": "PASS",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "paths": {
            "benchmark_root": str(benchmark_root),
            "summary_json": str(summary_json),
            "summary_csv": str(summary_csv),
            "condition_roots": {name: run["condition_root"] for name, run in runs.items()},
            "result_directories": {name: run["result_directory"] for name, run in runs.items()},
        },
        "common_window_s": [DEFAULT_STARTUP_CUTOFF_S, common_end],
        "M5_common_window_s": [DEFAULT_STARTUP_CUTOFF_S, m5_end],
        "conditions": conditions,
        "effort_change_percent": effort_changes,
        "effort_reduction_percent": {
            "vs_noexo": None if effort_changes["vs_noexo"] is None else -effort_changes["vs_noexo"],
            "vs_exozero": None if effort_changes["vs_exozero"] is None else -effort_changes["vs_exozero"],
        },
        "per_muscle_change_percent": {
            baseline: {
                muscle: _percent_change(per_muscle[baseline][muscle], per_muscle["FULL_YAC"][muscle])
                for muscle in per_muscle["FULL_YAC"]
            }
            for baseline in ("NO_EXO", "EXO_ZERO")
        },
        "muscle_family_change_percent": {
            baseline: {
                family: _percent_change(family_effort[baseline][family], family_effort["FULL_YAC"][family])
                for family in family_effort["FULL_YAC"]
            }
            for baseline in ("NO_EXO", "EXO_ZERO")
        },
        "physical_HRI_change_percent": physical_hri_change,
        "human_joint_mapping": human_joint_mapping,
        "fairness": fairness,
        "existing_intention_ablation": _existing_ablation(),
        "flags": flags,
        "videos": videos,
        "figures": figures,
        "scientific_labels": {
            "activation": "MyoAssist data.act simulation muscle activation; not measured EMG",
            "Ja": "Human Effort Proxy; not metabolic cost",
            "E_HRI_M5": "cumulative squared physical interaction torque cost; not mechanical energy",
            "paper_HRI": "evaluation-only YAC surrogate; not physical M5 HRI",
            "motor_work": "mechanical work from integral of actual motor torque times joint velocity",
        },
        "controller_source_of_truth": str(PROJECT_ROOT / "controllers/YAC.py"),
        "controllers_YAC_core_mathematics_modified": False,
        "calibration_or_optimization_performed": False,
    }
    _write_json(summary_json, summary)
    _write_summary_csv(summary_csv, conditions)
    # Final round-trip validation rejects NaN/Infinity and missing N/A semantics.
    reloaded = _read_json(summary_json)
    if reloaded["conditions"]["NO_EXO"]["M5_HRI"] is not None:
        raise RuntimeError("NoExo M5 HRI must be N/A/null, not zero")
    if not all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in figures.values()):
        raise RuntimeError("One or more benchmark figures are missing")
    _print_final(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
