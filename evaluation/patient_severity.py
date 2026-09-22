"""Cross-condition analysis for the Flexible Patient Baseline V1 study."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation import metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONDITION_ORDER = ("healthy100", "mild80", "moderate60", "severe40")
CONDITION_LABELS = {
    "healthy100": "Healthy 100%",
    "mild80": "Mild 80%",
    "moderate60": "Moderate 60%",
    "severe40": "Severe 40%",
}
RETAINED_FORCE = {"healthy100": 1.0, "mild80": 0.8, "moderate60": 0.6, "severe40": 0.4}
COLORS = {
    "healthy100": "#1f77b4",
    "mild80": "#2ca02c",
    "moderate60": "#ff7f0e",
    "severe40": "#d62728",
}
INTERFACES = ("waist", "thigh_l", "thigh_r", "shank_l", "shank_r")
JOINTS = (("right", "hip"), ("left", "hip"), ("right", "knee"), ("left", "knee"))
MUSCLES = (
    "vasti_l",
    "vasti_r",
    "hamstrings_l",
    "hamstrings_r",
    "glutmax_l",
    "glutmax_r",
    "iliopsoas_l",
    "iliopsoas_r",
    "soleus_l",
    "soleus_r",
)
MIN_CYCLE_S = 0.6
MAX_CYCLE_S = 2.0


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _metric(summary: dict[str, Any], category: str, name: str) -> Any:
    entry = summary.get("metrics", {}).get(category, {}).get("metrics", {}).get(name, {})
    return entry.get("value") if entry.get("available") else None


def _symmetry(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    denominator = abs(left) + abs(right)
    if denominator <= 1e-12:
        return None
    return 200.0 * abs(left - right) / denominator


def _complete_cycle_step_metrics(
    table: dict[str, np.ndarray], time: np.ndarray
) -> dict[str, Any]:
    """Exclude steps without valid preceding and following same-side cycles."""
    labels, _ = metrics.text_column(table, ["heel_strike_side"])
    result: dict[str, Any] = {"definition": (
        "A retained step event must have both a preceding and following same-side "
        "heel-strike interval within 0.6-2.0 s; this deterministically removes the "
        "initial and terminal incomplete cycles."
    )}
    means: dict[str, float | None] = {}
    for side in ("left", "right"):
        values, _ = metrics.numeric_column(table, [f"{side}_step_length_m"])
        events: list[tuple[float, float]] = []
        if labels is not None and values is not None:
            for event_time, label, value in zip(time, labels, values):
                if (
                    np.isfinite(event_time)
                    and np.isfinite(value)
                    and str(label).strip().lower() in (side, "both")
                    and (not events or float(event_time) - events[-1][0] >= 0.3)
                ):
                    events.append((float(event_time), float(value)))
        retained = [
            value
            for index, (_, value) in enumerate(events[1:-1], start=1)
            if MIN_CYCLE_S <= events[index][0] - events[index - 1][0] <= MAX_CYCLE_S
            and MIN_CYCLE_S <= events[index + 1][0] - events[index][0] <= MAX_CYCLE_S
        ]
        means[side] = float(np.mean(retained)) if retained else None
        result[side] = {
            "all_recorded_step_count": len(events),
            "complete_cycle_step_count": len(retained),
            "mean_step_length_m": means[side],
        }
    result["step_length_symmetry_index_percent"] = _symmetry(
        means["left"], means["right"]
    )
    return result


def _support_metrics(
    summary: dict[str, Any], table: dict[str, np.ndarray]
) -> dict[str, Any]:
    profile = summary["profiles"]["grf"]
    result: dict[str, Any] = {}
    impulses: dict[str, float | None] = {}
    for side in ("left", "right"):
        side_profile = profile.get(side, {})
        time = np.asarray(side_profile.get("time_s", []), dtype=float)
        grf = np.asarray(side_profile.get("grf_bw", []), dtype=float)
        mask = np.isfinite(time) & np.isfinite(grf)
        if np.count_nonzero(mask) < 2:
            result[side] = {
                "mean_touch_force_bw": None,
                "peak_touch_force_bw": None,
                "touch_force_impulse_bw_s": None,
                "stance_fraction": None,
            }
            impulses[side] = None
            continue
        t = time[mask]
        force = np.maximum(grf[mask], 0.0)
        duration = float(t[-1] - t[0])
        impulse = float(np.trapezoid(force, t))
        stance_fraction = (
            float(np.trapezoid((force > 0.1).astype(float), t) / duration)
            if duration > 0.0
            else None
        )
        impulses[side] = impulse
        result[side] = {
            "mean_touch_force_bw": float(np.mean(force)),
            "peak_touch_force_bw": float(np.max(force)),
            "touch_force_impulse_bw_s": impulse,
            "stance_fraction": stance_fraction,
            "gait_profile_cycles": int(side_profile.get("profile_cycles", 0)),
        }
    result["impulse_symmetry_index_percent"] = _symmetry(
        impulses["left"], impulses["right"]
    )
    if impulses["left"] is None or impulses["right"] is None:
        result["signed_left_minus_right_impulse_percent"] = None
    else:
        denominator = abs(impulses["left"]) + abs(impulses["right"])
        result["signed_left_minus_right_impulse_percent"] = (
            200.0 * (impulses["left"] - impulses["right"]) / denominator
            if denominator > 1e-12
            else None
        )
    return result


def _interaction_metrics(
    summary: dict[str, Any], validation: dict[str, Any]
) -> dict[str, Any]:
    interaction: dict[str, Any] = {}
    global_values: dict[str, dict[str, Any]] = {}
    all_finite = True
    caps = {"translation": 0.25, "rotation": math.pi, "force": 5000.0, "moment": 1000.0}
    within_numerical_guards = True
    for vector in ("translation", "rotation", "force", "moment"):
        candidates: list[tuple[float, str]] = []
        for interface in INTERFACES:
            peak = _metric(summary, "interaction", f"{interface}.{vector}.peak")
            rms = _metric(summary, "interaction", f"{interface}.{vector}.rms")
            p95 = _metric(summary, "interaction", f"{interface}.{vector}.p95")
            interaction.setdefault(interface, {})[vector] = {
                "peak": peak,
                "rms": rms,
                "p95": p95,
            }
            for value in (peak, rms, p95):
                all_finite &= value is not None and math.isfinite(float(value))
            if peak is not None:
                candidates.append((float(peak), interface))
                within_numerical_guards &= float(peak) < caps[vector]
        if candidates:
            value, source = max(candidates)
            global_values[vector] = {"peak": value, "source_interface": source}
        else:
            global_values[vector] = {"peak": None, "source_interface": None}

    expected_substeps = int(round(float(validation["actual_duration_s"]) / 0.001))
    actual_substeps = validation.get("coupling_substep_count")
    interfaces_complete = set(
        validation.get("coupling_summary", {}).get("interfaces", {})
    ) == set(INTERFACES)
    bounded = bool(
        validation.get("finite_state")
        and interfaces_complete
        and actual_substeps == expected_substeps
        and all_finite
        and within_numerical_guards
    )
    return {
        "interfaces": interaction,
        "global": global_values,
        "waist_rotation_peak_deg": (
            math.degrees(float(interaction["waist"]["rotation"]["peak"]))
            if interaction.get("waist", {}).get("rotation", {}).get("peak") is not None
            else None
        ),
        "global_rotation_peak_deg": (
            math.degrees(float(global_values["rotation"]["peak"]))
            if global_values["rotation"]["peak"] is not None
            else None
        ),
        "coupling_substeps": actual_substeps,
        "expected_coupling_substeps": expected_substeps,
        "all_statistics_finite": all_finite,
        "numerical_guard_caps": caps,
        "within_numerical_guards": within_numerical_guards,
        "bounded": bounded,
        "interpretation": (
            "Numerical boundedness guard, not a validated comfort or safety threshold."
        ),
    }


def _condition_summary(condition: str, run_dir: Path) -> dict[str, Any]:
    summary = _read_json(run_dir / "evaluation/summary.json")
    validation = _read_json(run_dir / "validation.json")
    manifest = _read_json(run_dir / "manifest.json")
    table = metrics.load_timeseries(run_dir / "timeseries.csv")
    time, _ = metrics.numeric_column(table, ["time", "simulation_time"])
    if time is None:
        raise ValueError(f"Missing time signal: {run_dir}")

    complete_cycle = _complete_cycle_step_metrics(table, time)
    joint_rmse: dict[str, Any] = {}
    for side, joint in JOINTS:
        key = f"{side}_{joint}"
        joint_rmse[key] = (
            0.0
            if condition == "healthy100"
            else _metric(summary, "gait", f"healthy_reference_{side}_{joint}_rmse_deg")
        )

    muscle_values: dict[str, Any] = {}
    for muscle in MUSCLES:
        muscle_values[muscle] = {
            "mean_activation": _metric(summary, "muscle", f"{muscle}.mean_activation"),
            "rms_activation": _metric(summary, "muscle", f"{muscle}.rms_activation"),
            "mean_force_N": _metric(summary, "muscle", f"{muscle}.mean_force_N"),
            "peak_force_N": _metric(summary, "muscle", f"{muscle}.peak_force_N"),
        }

    exo_zero = validation.get("exoskeleton_channels_all_zero", {})
    real_channels = ("HipExo_R", "HipExo_L", "KneeExo_R", "KneeExo_L")
    return {
        "label": CONDITION_LABELS[condition],
        "retained_vasti_l_force_fraction": RETAINED_FORCE[condition],
        "run_directory": str(run_dir.resolve()),
        "damage_verification": validation.get("damage_verification"),
        "frozen_inputs": {
            "coupled_xml_sha256": manifest.get("coupled_xml_sha256"),
            "human_controller_sha256": manifest.get("human_controller_parameter_sha256"),
            "control_timestep_s": manifest.get("control_timestep_s"),
            "physics_timestep_s": manifest.get("mujoco_timestep_s"),
        },
        "stability": {
            "duration_s": _metric(summary, "stability", "walking_duration_s"),
            "termination_reason": _metric(summary, "stability", "termination_reason"),
            "fall_detected": _metric(summary, "stability", "fall_detected"),
            "mean_speed_mps": _metric(summary, "stability", "mean_forward_velocity_mps"),
            "min_pelvis_height_m": _metric(summary, "stability", "min_pelvis_height_m"),
            "max_abs_pelvis_pitch_deg": _metric(summary, "stability", "max_abs_pelvis_pitch_deg"),
            "late_window_pelvis_pitch_trend_deg_s": _metric(summary, "stability", "pelvis_pitch_trend_deg_s"),
        },
        "gait": {
            "joint_rmse_vs_healthy_deg": joint_rmse,
            "full_run_step_symmetry_percent": _metric(summary, "gait", "step_length_symmetry_index_percent"),
            "cycle_time_symmetry_percent": _metric(summary, "gait", "cycle_time_symmetry_index_percent"),
            "left_mean_step_length_m": _metric(summary, "gait", "left_mean_step_length_m"),
            "right_mean_step_length_m": _metric(summary, "gait", "right_mean_step_length_m"),
            "left_stride_period_s": _metric(summary, "gait", "left_mean_cycle_time_s"),
            "right_stride_period_s": _metric(summary, "gait", "right_mean_cycle_time_s"),
            "complete_cycle_symmetry": complete_cycle,
            "profiles": summary["profiles"]["gait"],
        },
        "support": _support_metrics(summary, table),
        "muscle": muscle_values,
        "effort": {
            "activation_squared_mean": _metric(summary, "human_effort", "activation_squared_mean"),
            "activation_squared_integral_s": _metric(summary, "human_effort", "activation_squared_integral_s"),
            "activation_squared_per_distance_s_per_m": _metric(summary, "human_effort", "activation_squared_per_distance_s_per_m"),
            "human_effort_proxy_s_per_kg_m": _metric(summary, "human_effort", "human_effort_proxy"),
        },
        "interaction": _interaction_metrics(summary, validation),
        "exo_zero_torque_verified": all(exo_zero.get(name) is True for name in real_channels),
        "artifacts": validation.get("artifacts"),
    }


def _setup_axes(axes: Any) -> None:
    for axis in np.asarray(axes).ravel():
        axis.grid(True, alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)


def _line(axis: Any, x: np.ndarray, values: list[Any], ylabel: str) -> None:
    array = np.asarray([np.nan if value is None else value for value in values], dtype=float)
    axis.plot(x, array, marker="o", linewidth=2.2, color="#334e68")
    axis.set_ylabel(ylabel)
    axis.set_xticks(x, ["100", "80", "60", "40"])
    axis.set_xlabel("Retained configured vasti force capability (%)")


def _plot_stability(conditions: dict[str, Any], path: Path) -> None:
    x = np.arange(4)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    _line(axes[0, 0], x, [conditions[c]["stability"]["duration_s"] for c in CONDITION_ORDER], "Walking duration (s)")
    _line(axes[0, 1], x, [conditions[c]["stability"]["mean_speed_mps"] for c in CONDITION_ORDER], "Mean speed (m/s)")
    _line(axes[1, 0], x, [conditions[c]["stability"]["min_pelvis_height_m"] for c in CONDITION_ORDER], "Minimum pelvis height (m)")
    _line(axes[1, 1], x, [conditions[c]["stability"]["max_abs_pelvis_pitch_deg"] for c in CONDITION_ORDER], "Peak |pelvis pitch| (deg)")
    _setup_axes(axes)
    fig.suptitle("Flexible Patient Baseline: stability across weakness severity", fontsize=15)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_gait(conditions: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    phase = np.arange(101)
    for axis, (side, joint) in zip(axes.ravel(), JOINTS):
        for condition in CONDITION_ORDER:
            profile = conditions[condition]["gait"]["profiles"]["joints"][side][joint]["current_deg"]
            if profile is not None:
                axis.plot(phase, profile, color=COLORS[condition], linewidth=2, label=CONDITION_LABELS[condition])
        axis.set_title(f"{side.title()} {joint.title()}")
        axis.set_xlabel("Gait cycle (%)")
        axis.set_ylabel("Angle (deg)")
    _setup_axes(axes)
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=9)
    fig.suptitle("Hip and knee gait-cycle response to vasti weakness", fontsize=15)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_support(conditions: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    phase = np.arange(101)
    for condition in CONDITION_ORDER:
        summary = _read_json(Path(conditions[condition]["run_directory"]) / "evaluation/summary.json")
        for side, linestyle in (("left", "-"), ("right", "--")):
            profile = summary["profiles"]["grf"][side].get("gait_profile_bw")
            if profile is not None:
                axes[0].plot(phase, profile, color=COLORS[condition], linestyle=linestyle, linewidth=1.8, label=f"{CONDITION_LABELS[condition]} {side[0].upper()}")
    axes[0].set_xlabel("Gait cycle (%)")
    axes[0].set_ylabel("Foot touch normal force / BW")
    axes[0].set_title("Solid: left; dashed: right")
    x = np.arange(4)
    impulse_sym = [conditions[c]["support"]["impulse_symmetry_index_percent"] for c in CONDITION_ORDER]
    left_stance = [conditions[c]["support"]["left"]["stance_fraction"] for c in CONDITION_ORDER]
    right_stance = [conditions[c]["support"]["right"]["stance_fraction"] for c in CONDITION_ORDER]
    axes[1].plot(x, impulse_sym, marker="o", linewidth=2.2, label="Impulse symmetry index (%)")
    axes[1].plot(x, np.asarray(left_stance) * 100.0, marker="s", label="Left stance fraction (%)")
    axes[1].plot(x, np.asarray(right_stance) * 100.0, marker="^", label="Right stance fraction (%)")
    axes[1].set_xticks(x, ["100", "80", "60", "40"])
    axes[1].set_xlabel("Retained configured vasti force capability (%)")
    axes[1].set_ylabel("Percent")
    axes[1].set_title("Support redistribution")
    axes[1].legend(frameon=False, fontsize=9)
    _setup_axes(axes)
    axes[0].legend(frameon=False, ncol=2, fontsize=7)
    fig.suptitle("Foot touch-force support response", fontsize=15)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_muscle(conditions: dict[str, Any], path: Path) -> None:
    healthy = conditions["healthy100"]["muscle"]
    activation = np.zeros((len(MUSCLES), 4))
    force = np.zeros((len(MUSCLES), 4))
    for row, muscle in enumerate(MUSCLES):
        base_activation = float(healthy[muscle]["mean_activation"])
        base_force = float(healthy[muscle]["mean_force_N"])
        for column, condition in enumerate(CONDITION_ORDER):
            activation[row, column] = 100.0 * (float(conditions[condition]["muscle"][muscle]["mean_activation"]) - base_activation)
            force[row, column] = 100.0 * (float(conditions[condition]["muscle"][muscle]["mean_force_N"]) - base_force) / max(abs(base_force), 1e-9)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), constrained_layout=True)
    for axis, matrix, title, label in (
        (axes[0], activation, "Mean activation change", "percentage points vs Healthy"),
        (axes[1], force, "Mean actuator-force change", "% vs Healthy"),
    ):
        limit = max(float(np.nanmax(np.abs(matrix))), 1e-6)
        image = axis.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.set_xticks(np.arange(4), ["100", "80", "60", "40"])
        axis.set_yticks(np.arange(len(MUSCLES)), MUSCLES)
        axis.set_xlabel("Retained configured vasti force capability (%)")
        axis.set_title(title)
        fig.colorbar(image, ax=axis, shrink=0.78, label=label)
    fig.suptitle("Muscle compensation and force redistribution", fontsize=15)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_effort(conditions: dict[str, Any], path: Path) -> None:
    x = np.arange(4)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    _line(axes[0], x, [conditions[c]["effort"]["activation_squared_mean"] for c in CONDITION_ORDER], r"$J_a=(1/T)\int \Sigma a_i^2 dt$")
    _line(axes[1], x, [conditions[c]["effort"]["activation_squared_per_distance_s_per_m"] for c in CONDITION_ORDER], r"$\int \Sigma a_i^2dt$ / distance (s/m)")
    _setup_axes(axes)
    fig.suptitle("Human Effort Proxy (not metabolic cost)", fontsize=15)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_interaction(conditions: dict[str, Any], path: Path) -> None:
    x = np.arange(4)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    _line(axes[0, 0], x, [conditions[c]["interaction"]["global"]["force"]["peak"] for c in CONDITION_ORDER], "Global interface force peak (N)")
    _line(axes[0, 1], x, [1000.0 * conditions[c]["interaction"]["global"]["translation"]["peak"] for c in CONDITION_ORDER], "Global relative displacement peak (mm)")
    _line(axes[1, 0], x, [conditions[c]["interaction"]["global_rotation_peak_deg"] for c in CONDITION_ORDER], "Global relative rotation peak (deg)")
    _line(axes[1, 1], x, [conditions[c]["interaction"]["waist_rotation_peak_deg"] for c in CONDITION_ORDER], "Waist relative rotation peak (deg)")
    _setup_axes(axes)
    fig.suptitle("M5 numerical boundedness across patient severity", fontsize=15)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def generate_comparison(
    run_directories: dict[str, Path], output_root: Path
) -> tuple[Path, dict[str, Any]]:
    if tuple(run_directories) != CONDITION_ORDER:
        raise ValueError(f"Condition order must be {CONDITION_ORDER}")
    timestamp = datetime.now()
    output_dir = output_root / f"{timestamp:%Y%m%d_%H%M%S}__PATIENT_SEVERITY_COMPARISON"
    output_dir.mkdir(parents=True, exist_ok=False)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir()
    conditions = {
        condition: _condition_summary(condition, Path(run_directories[condition]).resolve())
        for condition in CONDITION_ORDER
    }
    figures = {
        "01_stability_severity.png": _plot_stability,
        "02_gait_severity.png": _plot_gait,
        "03_support_severity.png": _plot_support,
        "04_muscle_compensation.png": _plot_muscle,
        "05_effort_severity.png": _plot_effort,
        "06_interaction_severity.png": _plot_interaction,
    }
    figure_paths: dict[str, str] = {}
    for name, function in figures.items():
        path = figures_dir / name
        function(conditions, path)
        figure_paths[name] = str(path.resolve())

    healthy_full = conditions["healthy100"]["gait"]["full_run_step_symmetry_percent"]
    healthy_complete = conditions["healthy100"]["gait"]["complete_cycle_symmetry"]["step_length_symmetry_index_percent"]
    comparison = {
        "schema_version": "1.0",
        "generated_at": timestamp.isoformat(timespec="seconds"),
        "study": "M1 Flexible Patient Baseline V1",
        "condition_order": list(CONDITION_ORDER),
        "healthy_reference": str(Path(run_directories["healthy100"]).resolve()),
        "definitions": {
            "pathology": "runtime scaling of configured vasti actuator gainprm[actuator_id,2] and biasprm[actuator_id,2]; Moderate60 is bilateral",
            "grf": "aggregated foot/toe touch normal force divided by total model body weight; not 3D GRF",
            "human_effort": "activation-squared proxy; not metabolic cost",
            "full_run_step_symmetry": "200*|mean left step length - mean right step length|/(|left|+|right|), using every recorded step",
            "complete_cycle_step_symmetry": conditions["healthy100"]["gait"]["complete_cycle_symmetry"]["definition"],
            "m5_bounded": conditions["healthy100"]["interaction"]["interpretation"],
        },
        "symmetry_metric_audit": {
            "healthy_full_run_percent": healthy_full,
            "healthy_complete_cycle_percent": healthy_complete,
            "absolute_difference_percentage_points": (
                abs(float(healthy_full) - float(healthy_complete))
                if healthy_full is not None and healthy_complete is not None
                else None
            ),
            "initial_or_terminal_event_bias_detected": (
                abs(float(healthy_full) - float(healthy_complete)) > 1.0
                if healthy_full is not None and healthy_complete is not None
                else None
            ),
        },
        "conditions": conditions,
        "all_m5_bounded": all(conditions[c]["interaction"]["bounded"] for c in CONDITION_ORDER),
        "all_real_exo_torques_zero": all(conditions[c]["exo_zero_torque_verified"] for c in CONDITION_ORDER),
        "figures": figure_paths,
    }
    summary_path = output_dir / "severity_summary.json"
    _write_json(summary_path, comparison)
    return output_dir, comparison
