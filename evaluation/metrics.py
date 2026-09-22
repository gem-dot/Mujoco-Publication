"""Unified, dependency-light metrics for saved MuJoCo experiments.

The evaluator intentionally reads CSV/JSON artifacts instead of importing the
MyoAssist training stack.  Missing historical signals remain unavailable; they
are never estimated from unrelated columns.
"""

from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = "2.0"
GRAVITY_MPS2 = 9.81
FALL_PELVIS_HEIGHT_M = 0.65
FALL_PELVIS_PITCH_DEG = 60.0
DEFAULT_TARGET_VELOCITY_MPS = 1.25
DEFAULT_HUMAN_MASS_KG = 83.42146649200212
CONTACT_THRESHOLD_BW = 0.10
GAIT_PHASE = np.linspace(0.0, 100.0, 101)
MIN_GAIT_CYCLE_S = 0.6
MAX_GAIT_CYCLE_S = 2.0

SIDES = ("right", "left")
JOINTS = ("hip", "knee", "ankle")
MUSCLES = (
    "hamstrings_r", "bifemsh_r", "edl_r", "fdl_r", "glutmax_r",
    "iliopsoas_r", "rectfem_r", "vasti_r", "gastroc_r", "soleus_r",
    "tibant_r", "hamstrings_l", "bifemsh_l", "edl_l", "fdl_l",
    "glutmax_l", "iliopsoas_l", "rectfem_l", "vasti_l", "gastroc_l",
    "soleus_l", "tibant_l",
)
KEY_MUSCLE_GROUPS = ("vasti", "hamstrings", "glutmax", "iliopsoas", "soleus")
KEY_MUSCLES = tuple(
    f"{muscle}_{side}" for muscle in KEY_MUSCLE_GROUPS for side in ("r", "l")
)
EXO_CHANNELS = (
    "HipExo_R", "HipExo_L", "KneeExo_R",
    "KneeExo_L", "Exo_R", "Exo_L",
)
EXO_JOINT = {
    "HipExo_R": ("right", "hip"),
    "HipExo_L": ("left", "hip"),
    "KneeExo_R": ("right", "knee"),
    "KneeExo_L": ("left", "knee"),
    "Exo_R": ("right", "ankle"),
    "Exo_L": ("left", "ankle"),
}
INTERFACE_NAMES = ("waist", "thigh_l", "thigh_r", "shank_l", "shank_r")
INTERFACE_VECTORS = {
    "translation": ("m", "Relative displacement"),
    "rotation": ("rad", "Relative rotation"),
    "force": ("N", "Human-exoskeleton interface force"),
    "moment": ("N*m", "Human-exoskeleton interface moment"),
}


def load_timeseries(path: Path) -> dict[str, np.ndarray]:
    """Load a CSV without requiring pandas and preserve text columns."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")
    return {
        name: np.asarray([row.get(name, "") for row in rows], dtype=object)
        for name in reader.fieldnames
    }


def load_json_if_present(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def find_column(table: dict[str, np.ndarray], aliases: Iterable[str]) -> str | None:
    for name in aliases:
        if name in table:
            return name
    return None


def numeric_column(
    table: dict[str, np.ndarray], aliases: Iterable[str]
) -> tuple[np.ndarray | None, str | None]:
    name = find_column(table, aliases)
    if name is None:
        return None, None
    values = np.full(len(table[name]), np.nan, dtype=float)
    for index, raw in enumerate(table[name]):
        try:
            if raw not in ("", None):
                values[index] = float(raw)
        except (TypeError, ValueError):
            pass
    return values, name


def text_column(
    table: dict[str, np.ndarray], aliases: Iterable[str]
) -> tuple[np.ndarray | None, str | None]:
    name = find_column(table, aliases)
    if name is None:
        return None, None
    return np.asarray([str(value or "") for value in table[name]], dtype=object), name


def bool_column(
    table: dict[str, np.ndarray], aliases: Iterable[str]
) -> tuple[np.ndarray | None, str | None]:
    raw, name = text_column(table, aliases)
    if raw is None:
        return None, None
    truthy = {"1", "true", "yes", "y", "on"}
    return np.asarray([value.strip().lower() in truthy for value in raw]), name


def _nested_find(mapping: dict[str, Any], key: str) -> Any:
    if key in mapping:
        return mapping[key]
    for value in mapping.values():
        if isinstance(value, dict):
            found = _nested_find(value, key)
            if found is not None:
                return found
    return None


def _finite(values: np.ndarray | None) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=float)
    return values[np.isfinite(values)]


def _new_category() -> dict[str, Any]:
    return {"available": [], "unavailable": {}, "metrics": {}}


def _put_metric(
    category: dict[str, Any],
    name: str,
    value: Any,
    unit: str = "",
    definition: str = "",
) -> None:
    if value is None or (
        isinstance(value, (float, np.floating)) and not math.isfinite(float(value))
    ):
        _put_unavailable(category, name, "calculation did not produce a finite value")
        return
    if isinstance(value, np.generic):
        value = value.item()
    category["metrics"][name] = {
        "available": True,
        "value": value,
        "unit": unit,
        "definition": definition,
    }
    category["available"].append(name)


def _put_unavailable(category: dict[str, Any], name: str, reason: str) -> None:
    category["metrics"][name] = {
        "available": False,
        "value": None,
        "unit": "",
        "reason": reason,
    }
    category["unavailable"][name] = reason


def metric_value(category: dict[str, Any], name: str) -> Any:
    item = category.get("metrics", {}).get(name, {})
    return item.get("value") if item.get("available") else None


def _angle_aliases(side: str, joint: str) -> list[str]:
    suffix = "r" if side == "right" else "l"
    legacy_joint = {"hip": "hip_flexion", "knee": "knee_angle", "ankle": "ankle_angle"}[joint]
    return [
        f"joint_{legacy_joint}_{suffix}_rad",
        f"joint_{legacy_joint}_{suffix}",
        f"{side}_{joint}_angle_rad",
        f"{side}_{joint}_angle",
        f"joint_{joint}_{suffix}_rad",
        f"joint_{joint}_{suffix}",
        f"{side}_{joint}_angle_deg",
    ]


def joint_angle_deg(
    table: dict[str, np.ndarray], side: str, joint: str
) -> tuple[np.ndarray | None, str | None]:
    values, name = numeric_column(table, _angle_aliases(side, joint))
    if values is None or name is None:
        return None, None
    if name.endswith("_deg"):
        return values, name
    return np.rad2deg(values), name


def _velocity_aliases(side: str, joint: str) -> list[str]:
    suffix = "r" if side == "right" else "l"
    legacy_joint = {"hip": "hip_flexion", "knee": "knee_angle", "ankle": "ankle_angle"}[joint]
    return [
        f"joint_{legacy_joint}_{suffix}_velocity_rad_s",
        f"joint_{legacy_joint}_{suffix}_qvel_rad_s",
        f"{side}_{joint}_velocity_rad_s",
        f"joint_velocity_{joint}_{suffix}_rad_s",
        f"exo_joint_velocity_{joint}_{suffix}_rad_s",
    ]


def joint_velocity(
    table: dict[str, np.ndarray], side: str, joint: str
) -> tuple[np.ndarray | None, str | None]:
    return numeric_column(table, _velocity_aliases(side, joint))


def _strike_times(
    table: dict[str, np.ndarray], time: np.ndarray, side: str
) -> tuple[np.ndarray, str]:
    labels, _ = text_column(table, ["heel_strike_side"])
    if labels is not None:
        selected = [
            float(t)
            for t, label in zip(time, labels)
            if np.isfinite(t) and label.strip().lower() in (side, "both")
        ]
        return _deduplicate_events(selected), "heel_strike_side"

    contact, name = bool_column(table, [f"{side}_heel_contact"])
    if contact is None:
        return np.asarray([], dtype=float), "unavailable"
    rising = np.flatnonzero(contact & np.concatenate(([True], ~contact[:-1])))
    return _deduplicate_events(time[rising]), name or "heel contact rising edge"


def _deduplicate_events(times: Iterable[float], refractory_s: float = 0.3) -> np.ndarray:
    result: list[float] = []
    for raw in times:
        value = float(raw)
        if not math.isfinite(value):
            continue
        if not result or value - result[-1] >= refractory_s:
            result.append(value)
    return np.asarray(result, dtype=float)


def _cycle_profile_stats(
    time: np.ndarray, values: np.ndarray | None, strikes: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    if values is None or len(strikes) < 2:
        return None, None, 0
    profiles: list[np.ndarray] = []
    for start, end in zip(strikes[:-1], strikes[1:]):
        duration = end - start
        if duration < MIN_GAIT_CYCLE_S or duration > MAX_GAIT_CYCLE_S:
            continue
        mask = (time >= start) & (time <= end) & np.isfinite(values)
        if int(np.count_nonzero(mask)) < 3:
            continue
        normalized = 100.0 * (time[mask] - start) / duration
        profiles.append(np.interp(GAIT_PHASE, normalized, values[mask]))
    if not profiles:
        return None, None, 0
    stacked = np.vstack(profiles)
    return np.mean(stacked, axis=0), np.std(stacked, axis=0), len(profiles)


def _cycle_profile(
    time: np.ndarray, values: np.ndarray | None, strikes: np.ndarray
) -> tuple[np.ndarray | None, int]:
    mean, _, count = _cycle_profile_stats(time, values, strikes)
    return mean, count


def _waveform_correlation(
    current: np.ndarray, reference: np.ndarray
) -> float | None:
    valid = np.isfinite(current) & np.isfinite(reference)
    if int(np.count_nonzero(valid)) < 3:
        return None
    current_valid = current[valid]
    reference_valid = reference[valid]
    if np.std(current_valid) <= 1e-12 or np.std(reference_valid) <= 1e-12:
        return None
    return float(np.corrcoef(current_valid, reference_valid)[0, 1])


def _symmetry_percent(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    denominator = abs(left) + abs(right)
    if denominator <= 1e-12:
        return None
    return 200.0 * abs(left - right) / denominator


def _mean_step_length(table: dict[str, np.ndarray], side: str) -> tuple[float | None, int]:
    values, _ = numeric_column(table, [f"{side}_step_length_m"])
    finite = _finite(values)
    if len(finite) == 0:
        return None, 0
    return float(np.mean(finite)), int(len(finite))


def _muscle_column(
    table: dict[str, np.ndarray], muscle: str, signal: str
) -> tuple[np.ndarray | None, str | None]:
    if signal == "ctrl":
        exact = [f"muscle_ctrl_{muscle}"]
        patterns = [rf"^ctrl_\d+_{re.escape(muscle)}$"]
    elif signal == "activation":
        exact = [f"muscle_activation_{muscle}", f"activation_{muscle}"]
        patterns = [rf"^activation_\d+_{re.escape(muscle)}$"]
    else:
        exact = [f"muscle_force_{muscle}_N", f"muscle_force_{muscle}"]
        patterns = [
            rf"^muscle_force_\d+_{re.escape(muscle)}_N$",
            rf"^actual_force_\d+_{re.escape(muscle)}_N$",
            rf"^force_\d+_{re.escape(muscle)}$",
        ]
    name = find_column(table, exact)
    if name is None:
        for candidate in table:
            if any(re.match(pattern, candidate) for pattern in patterns):
                name = candidate
                break
    if name is None:
        return None, None
    values, _ = numeric_column(table, [name])
    if values is not None and signal == "force":
        values = np.abs(values)
    return values, name


def _exo_column(
    table: dict[str, np.ndarray], channel: str, signal: str
) -> tuple[np.ndarray | None, str | None]:
    if signal == "command":
        exact = [
            f"exo_command_torque_{channel}_nm",
            f"command_torque_{channel}_Nm",
        ]
        patterns = [rf"^ctrl_\d+_{re.escape(channel)}$"]
    else:
        exact = [
            f"exo_actual_torque_{channel}_nm",
            f"actual_torque_{channel}_Nm",
        ]
        patterns = [
            rf"^actual_force_\d+_{re.escape(channel)}_N$",
            rf"^force_\d+_{re.escape(channel)}$",
        ]
    name = find_column(table, exact)
    if name is None:
        for candidate in table:
            if any(re.match(pattern, candidate) for pattern in patterns):
                name = candidate
                break
    if name is None:
        return None, None
    return numeric_column(table, [name])


def _infer_masses(validation: dict[str, Any]) -> dict[str, Any]:
    human = _nested_find(validation, "human_model_kg")
    if human is None:
        human = _nested_find(validation, "human_mass_kg")
    model = _nested_find(validation, "total_model_mass_kg")
    if model is None:
        model = _nested_find(validation, "coupled_model_kg")
    exo = _nested_find(validation, "exoskeleton_mass_kg")
    if exo is None:
        exo = _nested_find(validation, "exo_mass_kg")
    try:
        human = float(human) if human is not None else None
    except (TypeError, ValueError):
        human = None
    try:
        model = float(model) if model is not None else None
    except (TypeError, ValueError):
        model = None
    try:
        exo = float(exo) if exo is not None else 0.0
    except (TypeError, ValueError):
        exo = 0.0

    if human is None and model is not None and exo > 0:
        human = model - exo
        human_source = "validation total model mass - exoskeleton mass"
    elif human is not None:
        human_source = "validation human mass"
    else:
        human = DEFAULT_HUMAN_MASS_KG
        human_source = "project validated fixed human mass fallback"

    if model is None:
        model = human + exo
        model_source = "human mass + validation exoskeleton mass"
    else:
        model_source = "validation model mass"
    return {
        "human_mass_kg": float(human),
        "human_mass_source": human_source,
        "model_mass_kg": float(model),
        "model_mass_source": model_source,
        "exoskeleton_mass_kg": float(exo),
    }


def _target_velocity(validation: dict[str, Any]) -> tuple[float, str]:
    value = _nested_find(validation, "target_speed_m_per_s")
    if value is None:
        value = _nested_find(validation, "target_velocity_mps")
    try:
        return float(value), "validation metadata"
    except (TypeError, ValueError):
        return DEFAULT_TARGET_VELOCITY_MPS, "project default"


def _requested_duration(validation: dict[str, Any], observed: float) -> tuple[float, str]:
    value = _nested_find(validation, "requested_duration_seconds")
    if value is None:
        value = _nested_find(validation, "requested_duration_s")
    try:
        return float(value), "validation metadata"
    except (TypeError, ValueError):
        return observed, "not recorded; observed duration used"


def _select_reference(
    candidates: Iterable[Path],
) -> tuple[dict[str, np.ndarray] | None, Path | None, str]:
    skipped: list[str] = []
    for path in candidates:
        if not path.exists():
            skipped.append(f"{path}: missing")
            continue
        try:
            table = load_timeseries(path)
        except ValueError as exc:
            skipped.append(f"{path}: {exc}")
            continue
        if all(joint_angle_deg(table, side, joint)[0] is not None for side in SIDES for joint in JOINTS):
            reason = "first candidate containing all six hip/knee/ankle angle signals"
            if skipped:
                reason += "; skipped " + "; ".join(skipped)
            return table, path.resolve(), reason
        skipped.append(f"{path}: six joint-angle signals unavailable")
    return None, None, "; ".join(skipped) or "no healthy reference candidates supplied"


def _contact_segments(time: np.ndarray, contact: np.ndarray) -> tuple[list[float], list[float]]:
    if len(time) < 2:
        return [], []
    changes = np.diff(contact.astype(int))
    starts = list(np.flatnonzero(changes == 1) + 1)
    ends = list(np.flatnonzero(changes == -1) + 1)
    if contact[0]:
        starts.insert(0, 0)
    if contact[-1]:
        ends.append(len(contact) - 1)
    stance = [float(time[end] - time[start]) for start, end in zip(starts, ends) if end > start]
    toe_off_times = time[np.flatnonzero(changes == -1) + 1]
    heel_strike_times = time[np.flatnonzero(changes == 1) + 1]
    swing: list[float] = []
    for toe_off in toe_off_times:
        future = heel_strike_times[heel_strike_times > toe_off]
        if len(future):
            swing.append(float(future[0] - toe_off))
    return stance, swing


def _grf_data(
    table: dict[str, np.ndarray], time: np.ndarray, model_mass_kg: float
) -> tuple[dict[str, Any] | None, str]:
    aliases = {
        "right_heel": ["right_heel_force_n", "right_heel_force", "right_heel_force_N", "r_foot_force_N", "sensor_r_foot_N"],
        "right_toe": ["right_toe_force_n", "right_toe_force", "right_toe_force_N", "r_toes_force_N", "sensor_r_toes_N"],
        "left_heel": ["left_heel_force_n", "left_heel_force", "left_heel_force_N", "l_foot_force_N", "sensor_l_foot_N"],
        "left_toe": ["left_toe_force_n", "left_toe_force", "left_toe_force_N", "l_toes_force_N", "sensor_l_toes_N"],
    }
    forces: dict[str, np.ndarray] = {}
    missing: list[str] = []
    sources: dict[str, str] = {}
    for key, names in aliases.items():
        values, source = numeric_column(table, names)
        if values is None or source is None:
            missing.append(key)
        else:
            forces[key] = values
            sources[key] = source
    if missing:
        return None, "missing raw touch-sensor force fields: " + ", ".join(missing)
    if model_mass_kg <= 0:
        return None, "model mass is unavailable or non-positive"
    body_weight = model_mass_kg * GRAVITY_MPS2
    result: dict[str, Any] = {"source_columns": sources, "body_weight_N": body_weight}
    for side in SIDES:
        heel = forces[f"{side}_heel"]
        toe = forces[f"{side}_toe"]
        grf_n = heel + toe
        grf_bw = grf_n / body_weight
        total_contact = np.isfinite(grf_bw) & (grf_bw > CONTACT_THRESHOLD_BW)
        heel_contact = np.isfinite(heel) & ((heel / body_weight) > CONTACT_THRESHOLD_BW)
        heel_rising = np.flatnonzero(heel_contact & np.concatenate(([True], ~heel_contact[:-1])))
        total_falling = np.flatnonzero((~total_contact) & np.concatenate(([False], total_contact[:-1])))
        stance, swing = _contact_segments(time, total_contact)
        result[side] = {
            "grf_n": grf_n,
            "grf_bw": grf_bw,
            "heel_strike_times_s": time[heel_rising],
            "toe_off_times_s": time[total_falling],
            "contact": total_contact,
            "stance_durations_s": stance,
            "swing_durations_s": swing,
        }
    return result, ""


def _stability_metrics(
    table: dict[str, np.ndarray], validation: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    category = _new_category()
    time, time_name = numeric_column(table, ["time", "simulation_time"])
    if time is None or len(_finite(time)) == 0:
        raise ValueError("timeseries CSV has no usable time/simulation_time column")
    finite_time = _finite(time)
    duration = float(np.max(finite_time) - np.min(finite_time))
    requested, requested_source = _requested_duration(validation, duration)
    target, target_source = _target_velocity(validation)

    velocity, _ = numeric_column(table, ["forward_velocity", "forward_velocity_mps"])
    height, _ = numeric_column(table, ["pelvis_height", "pelvis_height_m"])
    pitch, _ = numeric_column(table, ["pelvis_pitch_deg", "pelvis_pitch"])
    done, _ = bool_column(table, ["is_done", "terminated_early"])
    reasons, _ = text_column(table, ["termination_reason"])

    min_height = float(np.nanmin(height)) if height is not None and len(_finite(height)) else None
    max_pitch = float(np.nanmax(np.abs(pitch))) if pitch is not None and len(_finite(pitch)) else None
    height_fall = min_height is not None and min_height < FALL_PELVIS_HEIGHT_M
    pitch_fall = max_pitch is not None and max_pitch > FALL_PELVIS_PITCH_DEG
    metadata_terminated = bool(_nested_find(validation, "terminated_early") or False)
    terminated = bool(height_fall or pitch_fall or metadata_terminated or (done is not None and np.any(done)))
    reason = ""
    if reasons is not None:
        nonempty = [str(value) for value in reasons if str(value).strip()]
        if nonempty:
            reason = nonempty[-1]
    if not reason:
        reason = str(_nested_find(validation, "termination_reason") or "")
    if not reason and height_fall:
        reason = "pelvis_height_below_0.65_m"
    if not reason and pitch_fall:
        reason = "pelvis_pitch_outside_plus_minus_60_deg"
    if not reason:
        reason = "none"
    fall_detected = bool(
        height_fall or pitch_fall or "fall" in reason.strip().lower()
    )
    success = bool(not terminated and duration >= requested - 0.011)

    _put_metric(category, "walking_duration_s", duration, "s", "last time minus first time")
    _put_metric(category, "success", success, "bool")
    _put_metric(
        category,
        "fall_detected",
        fall_detected,
        "bool",
        "pelvis threshold violation or explicit fall termination reason",
    )
    _put_metric(category, "terminated_early", terminated, "bool")
    _put_metric(category, "termination_reason", reason, "text")

    finite_velocity = _finite(velocity)
    if len(finite_velocity):
        samples = finite_velocity[1:] if len(finite_velocity) > 1 else finite_velocity
        mean_velocity = float(np.mean(samples))
        _put_metric(category, "mean_forward_velocity_mps", mean_velocity, "m/s")
        _put_metric(category, "velocity_std_mps", float(np.std(samples)), "m/s")
        _put_metric(category, "target_velocity_error_mps", abs(mean_velocity - target), "m/s")
    else:
        for name in ("mean_forward_velocity_mps", "velocity_std_mps", "target_velocity_error_mps"):
            _put_unavailable(category, name, "forward_velocity field is missing")
    if min_height is not None:
        _put_metric(category, "min_pelvis_height_m", min_height, "m")
    else:
        _put_unavailable(category, "min_pelvis_height_m", "pelvis_height field is missing")
    if pitch is not None and len(_finite(pitch)):
        finite_mask = np.isfinite(time) & np.isfinite(pitch)
        t = time[finite_mask]
        p = pitch[finite_mask]
        _put_metric(category, "max_abs_pelvis_pitch_deg", float(np.max(np.abs(p))), "deg")
        _put_metric(category, "rms_pelvis_pitch_deg", float(np.sqrt(np.mean(np.square(p)))), "deg")
        half = len(t) // 2
        trend = float(np.polyfit(t[half:], p[half:], 1)[0]) if len(t[half:]) >= 2 else None
        _put_metric(category, "pelvis_pitch_trend_deg_s", trend, "deg/s", "linear slope over second half")
    else:
        for name in ("max_abs_pelvis_pitch_deg", "rms_pelvis_pitch_deg", "pelvis_pitch_trend_deg_s"):
            _put_unavailable(category, name, "pelvis_pitch field is missing")
    context = {
        "time": time,
        "time_column": time_name,
        "target_velocity_mps": target,
        "target_velocity_source": target_source,
        "requested_duration_s": requested,
        "requested_duration_source": requested_source,
    }
    return category, context


def _gait_metrics(
    table: dict[str, np.ndarray],
    reference: dict[str, np.ndarray] | None,
    time: np.ndarray,
    grf: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    category = _new_category()
    profiles: dict[str, Any] = {"phase_percent": GAIT_PHASE.tolist(), "joints": {}}
    strike_times: dict[str, np.ndarray] = {}
    cycle_means: dict[str, float | None] = {}
    step_means: dict[str, float | None] = {}

    for side in SIDES:
        strikes, source = _strike_times(table, time, side)
        strike_times[side] = strikes
        cycles = np.diff(strikes)
        valid_cycles = cycles[(cycles >= MIN_GAIT_CYCLE_S) & (cycles <= MAX_GAIT_CYCLE_S)]
        cycle_mean = float(np.mean(valid_cycles)) if len(valid_cycles) else None
        cycle_means[side] = cycle_mean
        _put_metric(category, f"{side}_heel_strike_count", int(len(strikes)), "count", source)
        if cycle_mean is None:
            _put_unavailable(category, f"{side}_mean_cycle_time_s", "fewer than two valid same-side heel strikes")
        else:
            _put_metric(category, f"{side}_mean_cycle_time_s", cycle_mean, "s")
        step_mean, count = _mean_step_length(table, side)
        step_means[side] = step_mean
        if step_mean is None:
            _put_unavailable(category, f"{side}_mean_step_length_m", "step-length field is missing")
        else:
            _put_metric(category, f"{side}_mean_step_length_m", step_mean, "m", f"mean of {count} recorded steps")

        if grf is None:
            _put_unavailable(category, f"{side}_mean_stance_time_s", "raw heel+toe force fields are missing")
            _put_unavailable(category, f"{side}_mean_swing_time_s", "raw heel+toe force fields are missing")
        else:
            stance = grf[side]["stance_durations_s"]
            swing = grf[side]["swing_durations_s"]
            if stance:
                _put_metric(category, f"{side}_mean_stance_time_s", float(np.mean(stance)), "s")
            else:
                _put_unavailable(category, f"{side}_mean_stance_time_s", "no complete GRF stance interval")
            if swing:
                _put_metric(category, f"{side}_mean_swing_time_s", float(np.mean(swing)), "s")
            else:
                _put_unavailable(category, f"{side}_mean_swing_time_s", "no complete GRF swing interval")

        profiles["joints"][side] = {}
        ref_strikes = np.asarray([], dtype=float)
        ref_time = None
        if reference is not None:
            ref_time, _ = numeric_column(reference, ["time", "simulation_time"])
            if ref_time is not None:
                ref_strikes, _ = _strike_times(reference, ref_time, side)
        for joint in JOINTS:
            values, source_col = joint_angle_deg(table, side, joint)
            current_profile, current_std, current_cycles = _cycle_profile_stats(
                time, values, strikes
            )
            ref_profile = None
            ref_std = None
            ref_cycles = 0
            if reference is not None and ref_time is not None:
                ref_values, _ = joint_angle_deg(reference, side, joint)
                ref_profile, ref_std, ref_cycles = _cycle_profile_stats(
                    ref_time, ref_values, ref_strikes
                )
            profile_entry = {
                "source_column": source_col,
                "current_cycles": current_cycles,
                "reference_cycles": ref_cycles,
                "current_deg": current_profile.tolist() if current_profile is not None else None,
                "current_std_deg": current_std.tolist() if current_std is not None else None,
                "healthy_reference_deg": ref_profile.tolist() if ref_profile is not None else None,
                "healthy_reference_std_deg": ref_std.tolist() if ref_std is not None else None,
            }
            profiles["joints"][side][joint] = profile_entry
            current_prefix = f"current_{side}_{joint}"
            comparison_prefix = f"healthy_reference_{side}_{joint}"
            if current_profile is None:
                _put_unavailable(
                    category,
                    f"{current_prefix}_rom_deg",
                    "current joint signal or valid gait cycles unavailable",
                )
                for suffix in ("mae_deg", "rmse_deg", "max_abs_error_deg"):
                    _put_unavailable(
                        category,
                        f"{comparison_prefix}_{suffix}",
                        "current joint signal or valid gait cycles unavailable",
                    )
                if joint in ("hip", "knee"):
                    _put_unavailable(
                        category,
                        f"{comparison_prefix}_waveform_correlation",
                        "current joint signal or valid gait cycles unavailable",
                    )
                continue
            _put_metric(
                category,
                f"{current_prefix}_rom_deg",
                float(np.ptp(current_profile)),
                "deg",
            )
            if ref_profile is None:
                for suffix in ("mae_deg", "rmse_deg", "max_abs_error_deg"):
                    _put_unavailable(
                        category,
                        f"{comparison_prefix}_{suffix}",
                        "healthy reference profile unavailable",
                    )
                if joint in ("hip", "knee"):
                    _put_unavailable(
                        category,
                        f"{comparison_prefix}_waveform_correlation",
                        "healthy reference profile unavailable",
                    )
                continue
            error = current_profile - ref_profile
            _put_metric(category, f"{comparison_prefix}_mae_deg", float(np.mean(np.abs(error))), "deg")
            _put_metric(category, f"{comparison_prefix}_rmse_deg", float(np.sqrt(np.mean(np.square(error)))), "deg")
            _put_metric(category, f"{comparison_prefix}_max_abs_error_deg", float(np.max(np.abs(error))), "deg")
            if joint in ("hip", "knee"):
                correlation = _waveform_correlation(current_profile, ref_profile)
                if correlation is None:
                    _put_unavailable(
                        category,
                        f"{comparison_prefix}_waveform_correlation",
                        "Pearson correlation is undefined for a constant or insufficient waveform",
                    )
                else:
                    _put_metric(
                        category,
                        f"{comparison_prefix}_waveform_correlation",
                        correlation,
                        "1",
                        "Pearson correlation over the 101-point gait-cycle mean waveform",
                    )

    step_sym = _symmetry_percent(step_means["left"], step_means["right"])
    cycle_sym = _symmetry_percent(cycle_means["left"], cycle_means["right"])
    available_sym = [value for value in (step_sym, cycle_sym) if value is not None]
    if step_sym is None:
        _put_unavailable(category, "step_length_symmetry_index_percent", "bilateral step-length means unavailable")
    else:
        _put_metric(category, "step_length_symmetry_index_percent", step_sym, "%")
    if cycle_sym is None:
        _put_unavailable(category, "cycle_time_symmetry_index_percent", "bilateral cycle-time means unavailable")
    else:
        _put_metric(category, "cycle_time_symmetry_index_percent", cycle_sym, "%")
    if available_sym:
        _put_metric(
            category,
            "spatiotemporal_symmetry_index_percent",
            float(np.mean(available_sym)),
            "%",
            "mean of available 200*|L-R|/(|L|+|R|) step-length and cycle-time indices; lower is better",
        )
    else:
        _put_unavailable(category, "spatiotemporal_symmetry_index_percent", "step-length and cycle-time symmetry are both unavailable")
    profiles["heel_strike_times_s"] = {side: strike_times[side].tolist() for side in SIDES}
    return category, profiles


def _muscle_metrics(
    table: dict[str, np.ndarray], time: np.ndarray, gait_profiles: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    category = _new_category()
    profiles: dict[str, Any] = {
        "phase_percent": GAIT_PHASE.tolist(),
        "activation": {},
        "force": {},
    }
    activation_arrays: dict[str, np.ndarray] = {}
    strike_map = {
        side: np.asarray(gait_profiles.get("heel_strike_times_s", {}).get(side, []), dtype=float)
        for side in SIDES
    }
    for muscle in MUSCLES:
        activation, activation_source = _muscle_column(table, muscle, "activation")
        force, force_source = _muscle_column(table, muscle, "force")
        ctrl, ctrl_source = _muscle_column(table, muscle, "ctrl")
        if ctrl is None:
            _put_unavailable(category, f"{muscle}.mean_excitation", "muscle ctrl/excitation field is missing")
        else:
            _put_metric(category, f"{muscle}.mean_excitation", float(np.nanmean(ctrl)), "0-1", f"excitation/neural command from {ctrl_source}")
        if activation is None or len(_finite(activation)) == 0:
            for name in ("mean_activation", "rms_activation", "peak_activation"):
                _put_unavailable(category, f"{muscle}.{name}", "muscle activation/data.act field is missing")
            profile, cycles = None, 0
        else:
            activation_arrays[muscle] = activation
            _put_metric(category, f"{muscle}.mean_activation", float(np.nanmean(activation)), "0-1", "MuJoCo data.act; not measured EMG")
            _put_metric(category, f"{muscle}.rms_activation", float(np.sqrt(np.nanmean(np.square(activation)))), "0-1")
            _put_metric(category, f"{muscle}.peak_activation", float(np.nanmax(activation)), "0-1")
            side = "left" if muscle.endswith("_l") else "right"
            profile, cycles = _cycle_profile(time, activation, strike_map[side])
        profiles["activation"][muscle] = {
            "source_column": activation_source,
            "cycles": cycles,
            "profile": profile.tolist() if profile is not None else None,
        }
        if force is None or len(_finite(force)) == 0:
            for name in ("mean_force_N", "peak_force_N"):
                _put_unavailable(category, f"{muscle}.{name}", "muscle actuator-force field is missing")
            force_profile, force_cycles = None, 0
        else:
            _put_metric(category, f"{muscle}.mean_force_N", float(np.nanmean(force)), "N", f"absolute actuator force from {force_source}")
            _put_metric(category, f"{muscle}.peak_force_N", float(np.nanmax(force)), "N")
            side = "left" if muscle.endswith("_l") else "right"
            force_profile, force_cycles = _cycle_profile(
                time, force, strike_map[side]
            )
        profiles["force"][muscle] = {
            "source_column": force_source,
            "cycles": force_cycles,
            "profile_N": (
                force_profile.tolist() if force_profile is not None else None
            ),
        }

    key_summary: dict[str, Any] = {}
    for muscle in KEY_MUSCLES:
        key_summary[muscle] = {
            metric.split(".", 1)[1]: item
            for metric, item in category["metrics"].items()
            if metric.startswith(muscle + ".")
        }
    profiles["key_muscle_summary"] = key_summary
    if all(profiles["activation"].get(name, {}).get("profile") is not None for name in MUSCLES):
        _put_metric(category, "all_22_activation_gait_profiles", True, "bool")
    else:
        missing = [name for name in MUSCLES if profiles["activation"].get(name, {}).get("profile") is None]
        _put_unavailable(category, "all_22_activation_gait_profiles", "missing signal or valid gait cycle for: " + ", ".join(missing))
    return category, profiles, activation_arrays


def _grf_metrics(
    grf: dict[str, Any] | None,
    grf_reason: str,
    time: np.ndarray,
    gait_profiles: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    category = _new_category()
    profiles: dict[str, Any] = {"phase_percent": GAIT_PHASE.tolist(), "available": grf is not None}
    names = (
        "left_grf_bw", "right_grf_bw",
        "peak_left_grf_bw", "peak_right_grf_bw",
        "left_vertical_impulse_Ns", "right_vertical_impulse_Ns",
        "left_vertical_impulse_bw_s", "right_vertical_impulse_bw_s",
        "left_grf_gait_profile", "right_grf_gait_profile",
        "left_heel_strikes", "right_heel_strikes", "left_toe_offs", "right_toe_offs",
    )
    if grf is None:
        for name in names:
            _put_unavailable(category, name, grf_reason)
        profiles["reason"] = grf_reason
        return category, profiles

    profiles["normalization"] = {
        "body_weight_N": grf["body_weight_N"],
        "contact_threshold_bw": CONTACT_THRESHOLD_BW,
        "signal_interpretation": "aggregated foot/toe contact normal force; not 3D GRF or pressure distribution",
    }
    for side in SIDES:
        side_data = grf[side]
        grf_n = side_data["grf_n"]
        grf_bw = side_data["grf_bw"]
        _put_metric(category, f"{side}_grf_bw", True, "timeseries", "values stored in profiles.grf")
        _put_metric(category, f"peak_{side}_grf_bw", float(np.nanmax(grf_bw)), "BW")
        _put_metric(category, f"{side}_vertical_impulse_Ns", float(np.trapezoid(np.nan_to_num(grf_n), time)), "N*s", "whole recorded interval")
        _put_metric(category, f"{side}_vertical_impulse_bw_s", float(np.trapezoid(np.nan_to_num(grf_bw), time)), "BW*s", "whole recorded interval")
        strikes = np.asarray(gait_profiles.get("heel_strike_times_s", {}).get(side, []), dtype=float)
        profile, cycles = _cycle_profile(time, grf_bw, strikes)
        if profile is None:
            _put_unavailable(category, f"{side}_grf_gait_profile", "no valid same-side gait cycle")
        else:
            _put_metric(category, f"{side}_grf_gait_profile", True, "bool", f"mean of {cycles} cycles")
        _put_metric(category, f"{side}_heel_strikes", int(len(side_data["heel_strike_times_s"])), "count")
        _put_metric(category, f"{side}_toe_offs", int(len(side_data["toe_off_times_s"])), "count")
        profiles[side] = {
            "time_s": time.tolist(),
            "grf_bw": grf_bw.tolist(),
            "gait_profile_bw": profile.tolist() if profile is not None else None,
            "profile_cycles": cycles,
            "heel_strike_times_s": side_data["heel_strike_times_s"].tolist(),
            "toe_off_times_s": side_data["toe_off_times_s"].tolist(),
            "stance_intervals": _boolean_intervals(time, side_data["contact"]),
        }
    return category, profiles


def _boolean_intervals(time: np.ndarray, values: np.ndarray) -> list[list[float]]:
    changes = np.diff(values.astype(int))
    starts = list(np.flatnonzero(changes == 1) + 1)
    ends = list(np.flatnonzero(changes == -1) + 1)
    if values[0]:
        starts.insert(0, 0)
    if values[-1]:
        ends.append(len(values) - 1)
    return [[float(time[a]), float(time[b])] for a, b in zip(starts, ends) if b > a]


def _effort_metrics(
    activation_arrays: dict[str, np.ndarray],
    time: np.ndarray,
    forward_position: np.ndarray | None,
    masses: dict[str, Any],
) -> dict[str, Any]:
    category = _new_category()
    effort_names = (
        "activation_squared_mean",
        "activation_squared_integral_s",
        "activation_squared_per_distance_s_per_m",
        "human_effort_proxy",
        "myoassist_effort_proxy",
    )
    if len(activation_arrays) != len(MUSCLES):
        missing = [name for name in MUSCLES if name not in activation_arrays]
        reason = "all 22 activation/data.act signals are required; missing: " + ", ".join(missing)
        for name in effort_names:
            _put_unavailable(category, name, reason)
        return category
    if forward_position is None or len(_finite(forward_position)) < 2:
        reason = "forward_position field is missing"
        for name in effort_names:
            _put_unavailable(category, name, reason)
        return category
    distance = abs(float(forward_position[np.flatnonzero(np.isfinite(forward_position))[-1]] - forward_position[np.flatnonzero(np.isfinite(forward_position))[0]]))
    if distance <= 1e-9:
        reason = "forward distance is zero or too small"
        for name in effort_names:
            _put_unavailable(category, name, reason)
        return category
    matrix = np.column_stack([activation_arrays[name] for name in MUSCLES])
    valid = np.isfinite(time) & np.all(np.isfinite(matrix), axis=1)
    if int(np.count_nonzero(valid)) < 2:
        reason = "activation/time samples are not finite"
        for name in effort_names:
            _put_unavailable(category, name, reason)
        return category
    sum_squared = np.sum(np.square(matrix[valid]), axis=1)
    integral = float(np.trapezoid(sum_squared, time[valid]))
    duration = float(time[valid][-1] - time[valid][0])
    if duration <= 0.0:
        reason = "recorded activation duration is zero or negative"
        for name in effort_names:
            _put_unavailable(category, name, reason)
        return category
    mean_squared = integral / duration
    distance_normalized = integral / distance
    human_proxy = distance_normalized / masses["human_mass_kg"]
    official_proxy = float(np.sum(sum_squared) / (masses["model_mass_kg"] * distance))
    _put_metric(
        category,
        "activation_squared_mean",
        mean_squared,
        "1",
        "(1/T) * integral(sum_i activation_i^2, dt); 22 MuJoCo activation states",
    )
    _put_metric(
        category,
        "activation_squared_integral_s",
        integral,
        "s",
        "integral(sum_i activation_i^2, dt) over the recorded interval",
    )
    _put_metric(
        category,
        "activation_squared_per_distance_s_per_m",
        distance_normalized,
        "s/m",
        "integral(sum_i activation_i^2, dt) / forward distance",
    )
    _put_metric(
        category, "human_effort_proxy", human_proxy, "s/(kg*m)",
        "integral(sum(activation^2),dt)/(fixed human mass*forward distance); effort proxy, not metabolic energy",
    )
    _put_metric(
        category, "myoassist_effort_proxy", official_proxy, "samples/(kg*m)",
        "MyoAssist-compatible sum(activation^2)/(total model mass*distance), without dt; full recorded interval",
    )
    _put_metric(category, "effort_forward_distance_m", distance, "m")
    _put_metric(category, "effort_human_mass_kg", masses["human_mass_kg"], "kg", masses["human_mass_source"])
    _put_metric(category, "effort_model_mass_kg", masses["model_mass_kg"], "kg", masses["model_mass_source"])
    return category


def _exo_metrics(
    table: dict[str, np.ndarray], time: np.ndarray
) -> tuple[dict[str, Any], dict[str, Any]]:
    category = _new_category()
    profiles: dict[str, Any] = {"time_s": time.tolist(), "channels": {}}
    for channel in EXO_CHANNELS:
        command, command_source = _exo_column(table, channel, "command")
        actual, actual_source = _exo_column(table, channel, "actual")
        side, joint = EXO_JOINT[channel]
        velocity, velocity_source = numeric_column(
            table,
            [f"exo_joint_velocity_{channel}_rad_s", *_velocity_aliases(side, joint)],
        )
        finite_command = _finite(command) if command is not None else np.array([])
        finite_actual = _finite(actual) if actual is not None else np.array([])
        actuator_is_present = bool(
            len(finite_actual)
            or (len(finite_command) and np.any(np.abs(finite_command) > 1e-12))
        )
        if not actuator_is_present:
            # Schema 3.1 reserves ankle-channel columns even for four-actuator
            # hip/knee models.  Those placeholders can contain zero commands
            # and inherit Human ankle velocity aliases despite having no
            # actual actuator signal.  They are not unavailable measurements.
            continue
        entry: dict[str, Any] = {
            "command_source": command_source,
            "actual_torque_source": actual_source,
            "joint_velocity_source": velocity_source,
            "command_torque_Nm": command.tolist() if command is not None else None,
            "actual_torque_Nm": actual.tolist() if actual is not None else None,
            "joint_velocity_rad_s": velocity.tolist() if velocity is not None else None,
            "power_W": None,
        }
        profiles["channels"][channel] = entry
        if command is None or len(_finite(command)) == 0:
            _put_unavailable(
                category,
                f"{channel}.peak_abs_command_torque_Nm",
                "commanded actuator torque field is missing",
            )
            _put_unavailable(
                category,
                f"{channel}.rms_command_torque_Nm",
                "commanded actuator torque field is missing",
            )
        else:
            _put_metric(
                category,
                f"{channel}.peak_abs_command_torque_Nm",
                float(np.nanmax(np.abs(command))),
                "N*m",
            )
            _put_metric(
                category,
                f"{channel}.rms_command_torque_Nm",
                float(np.sqrt(np.nanmean(np.square(command)))),
                "N*m",
            )
        if actual is None or len(_finite(actual)) == 0:
            _put_unavailable(category, f"{channel}.peak_abs_torque_Nm", "actual actuator torque field is missing")
            _put_unavailable(category, f"{channel}.rms_torque_Nm", "actual actuator torque field is missing")
            for metric in ("peak_positive_power_W", "peak_negative_power_W", "rms_power_W", "positive_work_J", "negative_work_J", "absorbed_work_J", "absolute_work_J", "net_work_J"):
                _put_unavailable(category, f"{channel}.{metric}", "actual actuator torque field is missing")
            continue
        _put_metric(category, f"{channel}.peak_abs_torque_Nm", float(np.nanmax(np.abs(actual))), "N*m")
        _put_metric(category, f"{channel}.rms_torque_Nm", float(np.sqrt(np.nanmean(np.square(actual)))), "N*m")
        if velocity is None or len(_finite(velocity)) == 0:
            reason = f"corresponding {side} {joint} angular velocity field is missing"
            for metric in ("peak_positive_power_W", "peak_negative_power_W", "rms_power_W", "positive_work_J", "negative_work_J", "absorbed_work_J", "absolute_work_J", "net_work_J"):
                _put_unavailable(category, f"{channel}.{metric}", reason)
            continue
        valid = np.isfinite(time) & np.isfinite(actual) & np.isfinite(velocity)
        if int(np.count_nonzero(valid)) < 2:
            reason = "fewer than two finite torque/velocity/time samples"
            for metric in ("peak_positive_power_W", "peak_negative_power_W", "rms_power_W", "positive_work_J", "negative_work_J", "absorbed_work_J", "absolute_work_J", "net_work_J"):
                _put_unavailable(category, f"{channel}.{metric}", reason)
            continue
        power = actual * velocity
        entry["power_W"] = power.tolist()
        p = power[valid]
        t = time[valid]
        positive = np.maximum(p, 0.0)
        negative = np.minimum(p, 0.0)
        _put_metric(category, f"{channel}.peak_positive_power_W", float(np.max(positive)), "W")
        _put_metric(category, f"{channel}.peak_negative_power_W", float(np.min(negative)), "W")
        _put_metric(category, f"{channel}.rms_power_W", float(np.sqrt(np.mean(np.square(p)))), "W")
        _put_metric(category, f"{channel}.positive_work_J", float(np.trapezoid(positive, t)), "J")
        _put_metric(category, f"{channel}.negative_work_J", float(np.trapezoid(negative, t)), "J", "signed negative work")
        _put_metric(category, f"{channel}.absorbed_work_J", float(np.trapezoid(np.maximum(-p, 0.0), t)), "J")
        _put_metric(category, f"{channel}.absolute_work_J", float(np.trapezoid(np.abs(p), t)), "J")
        _put_metric(category, f"{channel}.net_work_J", float(np.trapezoid(p, t)), "J")
    active_channels = tuple(profiles["channels"])
    available_abs_work = [
        metric_value(category, f"{channel}.absolute_work_J")
        for channel in active_channels
    ]
    available_abs_work = [float(value) for value in available_abs_work if value is not None]
    if available_abs_work:
        _put_metric(category, "available_channels_total_absolute_work_J", float(sum(available_abs_work)), "J", "sum over channels with recorded joint velocity; may be incomplete")
    else:
        _put_unavailable(category, "available_channels_total_absolute_work_J", "no exoskeleton channel has both actual torque and joint velocity")
    return category, profiles


def _interaction_metrics(
    experiment_dir: Path,
    table: dict[str, np.ndarray],
    time: np.ndarray,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Evaluate M5 interface signals without degrading 1 kHz peak statistics."""
    physics_path = experiment_dir / "physics_timeseries_1khz.csv"
    physics_table: dict[str, np.ndarray] | None = None
    if physics_path.exists():
        try:
            physics_table = load_timeseries(physics_path)
        except ValueError:
            physics_table = None

    active_interfaces: list[str] = []
    profiles: dict[str, Any] = {
        "time_s": time.tolist(),
        "interfaces": {},
        "profile_source": "100 Hz control-endpoint timeseries",
    }
    for interface in INTERFACE_NAMES:
        interface_profiles: dict[str, Any] = {}
        for vector in INTERFACE_VECTORS:
            values, source = numeric_column(
                table, [f"coupling_{interface}_{vector}_norm"]
            )
            if values is not None and len(_finite(values)):
                interface_profiles[vector] = {
                    "source_column": source,
                    "norm": values.tolist(),
                }
        if interface_profiles:
            active_interfaces.append(interface)
            profiles["interfaces"][interface] = interface_profiles

    if not active_interfaces and physics_table is not None:
        for interface in INTERFACE_NAMES:
            if any(
                len(_finite(numeric_column(physics_table, [f"{interface}_{vector}_norm"])[0]))
                for vector in INTERFACE_VECTORS
            ):
                active_interfaces.append(interface)
                profiles["interfaces"][interface] = {}
    if not active_interfaces:
        return None, None

    category = _new_category()
    source_description: str
    if physics_table is not None:
        source_description = "physics_timeseries_1khz.csv pre-step samples"
        physics_time, _ = numeric_column(physics_table, ["time"])
        finite_time = _finite(physics_time)
        _put_metric(
            category,
            "physics_sample_count",
            int(len(finite_time)),
            "count",
            source_description,
        )
        if len(finite_time) >= 2:
            median_dt = float(np.median(np.diff(finite_time)))
            if median_dt > 0.0:
                _put_metric(
                    category,
                    "physics_sample_rate_hz",
                    1.0 / median_dt,
                    "Hz",
                    "inferred from median physics-sample time step",
                )
    else:
        source_description = "control-rate CSV substep aggregates"
        recorded_rate = manifest.get("physics_coupling_rate_hz")
        try:
            _put_metric(
                category,
                "physics_sample_rate_hz",
                float(recorded_rate),
                "Hz",
                "manifest value; full physics-rate CSV unavailable",
            )
        except (TypeError, ValueError):
            pass

    substeps, _ = numeric_column(table, ["coupling_physics_substeps"])
    for interface in active_interfaces:
        for vector, (unit, label) in INTERFACE_VECTORS.items():
            metric_prefix = f"{interface}.{vector}"
            values = None
            if physics_table is not None:
                values, _ = numeric_column(
                    physics_table, [f"{interface}_{vector}_norm"]
                )
            finite = _finite(values)
            if len(finite):
                _put_metric(
                    category,
                    f"{metric_prefix}.peak",
                    float(np.max(finite)),
                    unit,
                    f"{label} norm peak from {source_description}",
                )
                _put_metric(
                    category,
                    f"{metric_prefix}.rms",
                    float(np.sqrt(np.mean(np.square(finite)))),
                    unit,
                    f"{label} norm RMS from {source_description}",
                )
                _put_metric(
                    category,
                    f"{metric_prefix}.p95",
                    float(np.percentile(finite, 95.0)),
                    unit,
                    f"{label} norm p95 from {source_description}",
                )
                continue

            peak_values, _ = numeric_column(
                table, [f"coupling_{interface}_{vector}_norm_peak"]
            )
            rms_values, _ = numeric_column(
                table, [f"coupling_{interface}_{vector}_norm_rms"]
            )
            peak_finite = _finite(peak_values)
            if len(peak_finite):
                _put_metric(
                    category,
                    f"{metric_prefix}.peak",
                    float(np.max(peak_finite)),
                    unit,
                    f"{label} exact peak from per-control-interval substep peaks",
                )
            else:
                _put_unavailable(category, f"{metric_prefix}.peak", f"{label} is missing")
            valid_rms = (
                rms_values is not None
                and substeps is not None
                and np.any(np.isfinite(rms_values) & np.isfinite(substeps) & (substeps > 0))
            )
            if valid_rms:
                mask = np.isfinite(rms_values) & np.isfinite(substeps) & (substeps > 0)
                weighted_rms = float(
                    np.sqrt(
                        np.sum(substeps[mask] * np.square(rms_values[mask]))
                        / np.sum(substeps[mask])
                    )
                )
                _put_metric(
                    category,
                    f"{metric_prefix}.rms",
                    weighted_rms,
                    unit,
                    f"{label} exact RMS combined from substep count and interval RMS",
                )
            else:
                _put_unavailable(category, f"{metric_prefix}.rms", f"{label} substep RMS is missing")
            _put_unavailable(
                category,
                f"{metric_prefix}.p95",
                "global physics-rate p95 requires physics_timeseries_1khz.csv; interval p95 values cannot be combined exactly",
            )

    profiles["active_interfaces"] = active_interfaces
    profiles["statistics_source"] = source_description
    profiles["physics_timeseries_csv"] = (
        str(physics_path.resolve()) if physics_table is not None else None
    )
    return category, profiles


def evaluate_directory(
    experiment_dir: Path,
    project_root: Path,
    healthy_reference_candidates: Iterable[Path],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate one experiment directory and return JSON-ready metrics plus raw data."""
    experiment_dir = experiment_dir.resolve()
    csv_path = experiment_dir / "timeseries.csv"
    validation_path = experiment_dir / "validation.json"
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected timeseries.csv in experiment directory: {experiment_dir}")
    table = load_timeseries(csv_path)
    validation = load_json_if_present(validation_path)
    manifest_path = experiment_dir / "manifest.json"
    manifest = load_json_if_present(manifest_path)
    masses = _infer_masses(validation)

    stability, context = _stability_metrics(table, validation)
    time = context["time"]
    reference, reference_path, reference_reason = _select_reference(
        healthy_reference_candidates
    )
    grf_data, grf_reason = _grf_data(table, time, masses["model_mass_kg"])
    gait, gait_profiles = _gait_metrics(table, reference, time, grf_data)
    muscle, muscle_profiles, activation_arrays = _muscle_metrics(table, time, gait_profiles)
    grf, grf_profiles = _grf_metrics(grf_data, grf_reason, time, gait_profiles)
    forward_position, _ = numeric_column(table, ["forward_position", "forward_position_m"])
    effort = _effort_metrics(activation_arrays, time, forward_position, masses)
    exoskeleton, exo_profiles = _exo_metrics(table, time)
    interaction, interaction_profiles = _interaction_metrics(
        experiment_dir, table, time, manifest
    )

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": {
            "project_root": str(project_root.resolve()),
            "directory": str(experiment_dir),
            "timeseries_csv": str(csv_path),
            "validation_json": str(validation_path) if validation_path.exists() else None,
            "manifest_json": str(manifest_path) if manifest_path.exists() else None,
            "experiment_type": manifest.get("experiment_type"),
            "row_count": len(next(iter(table.values()))),
        },
        "healthy_reference": {
            "timeseries_csv": str(reference_path) if reference_path else None,
            "selection_reason": reference_reason,
            "official_ref_kinematics_used": False,
        },
        "normalization": {
            **masses,
            "gravity_mps2": GRAVITY_MPS2,
            "grf_body_weight_definition": "total_model_mass_kg * gravity_mps2",
            "human_effort_mass_definition": "fixed human mass excluding exoskeleton mass",
        },
        "definitions": {
            "fall": f"pelvis height < {FALL_PELVIS_HEIGHT_M} m or |pelvis pitch| > {FALL_PELVIS_PITCH_DEG} deg",
            "gait_cycle": "same-side heel strike to next same-side heel strike; valid duration 0.6-2.0 s; 101 points from 0-100%",
            "muscle_ctrl": "excitation/neural command",
            "muscle_activation": "MuJoCo data.act; not measured EMG",
            "muscle_force": "absolute MuJoCo actuator force",
            "grf": "aggregated foot/toe contact normal force; not pressure map, COP, or 3D GRF",
            "human_effort_proxy": "integral(sum(activation_i^2),dt)/(human_mass*forward_distance)",
            "myoassist_effort_proxy": "sum(activation_i^2)/(model_mass*forward_distance), without dt; compatible with ctrl_optim/optim/cost_functions/effort_costs.py",
            "exo_power": "actual actuator torque * corresponding joint angular velocity",
            "interaction": "M5 relative motion and human-exoskeleton interface-load norms; not comfort, pain, or a validated human-safety threshold",
        },
        "evaluation_context": {
            "target_velocity_mps": context["target_velocity_mps"],
            "target_velocity_source": context["target_velocity_source"],
            "requested_duration_s": context["requested_duration_s"],
            "requested_duration_source": context["requested_duration_source"],
        },
        "metrics": {
            "stability": stability,
            "gait": gait,
            "muscle": muscle,
            "grf": grf,
            "human_effort": effort,
            "exoskeleton": exoskeleton,
        },
        "profiles": {
            "gait": gait_profiles,
            "muscle": muscle_profiles,
            "grf": grf_profiles,
            "exoskeleton": exo_profiles,
        },
        "unsupported_by_current_model": {
            "plantar_pressure_map": "only aggregated foot/toe touch normal-force sensors exist",
            "center_of_pressure": "not supported by current saved/model signals",
            "three_dimensional_grf": "touch sensors provide scalar contact normal force, not a 3D force vector",
            "strap_or_interface_force": "rigidly attached exoskeleton has no strap/contact/force-torque sensor",
            "human_exoskeleton_true_interface_force": "motor torque at a shared joint is not an interface force",
            "metabolic_energy_calories_vo2": "activation effort proxies are not physiological metabolic measurements",
        },
        "figures": {},
        "source_code_policy": {
            "official_myoassist_source_modified": False,
            "xml_modified": False,
        },
    }
    if interaction is not None and interaction_profiles is not None:
        summary["metrics"]["interaction"] = interaction
        summary["profiles"]["interaction"] = interaction_profiles
        summary["unsupported_by_current_model"].pop("strap_or_interface_force", None)
        summary["unsupported_by_current_model"].pop(
            "human_exoskeleton_true_interface_force", None
        )
    return summary, table


def evaluation_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category_name, category in summary["metrics"].items():
        for metric_name, item in category["metrics"].items():
            rows.append({
                "category": category_name,
                "metric": metric_name,
                "available": bool(item.get("available")),
                "value": item.get("value"),
                "unit": item.get("unit", ""),
                "definition_or_reason": item.get("definition") or item.get("reason", ""),
            })
    return rows


def write_evaluation_csv(path: Path, summary: dict[str, Any]) -> None:
    rows = evaluation_rows(summary)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("category", "metric", "available", "value", "unit", "definition_or_reason"),
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    """Convert NumPy values and non-finite floats into strict JSON values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
