#!/usr/bin/env python3
"""Run healthy and isolated left-vasti weakness conditions without an exoskeleton."""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path


EARLY_PROJECT_ROOT = Path(__file__).resolve().parents[1]
EARLY_PROJECT_PYTHON = EARLY_PROJECT_ROOT / ".venv" / "bin" / "python"
if (
    os.environ.get("PATIENT_SWEEP_VENV_REEXEC") != "1"
    and EARLY_PROJECT_PYTHON.is_file()
    and Path(sys.executable).resolve() != EARLY_PROJECT_PYTHON.resolve()
):
    environment = os.environ.copy()
    environment["PATIENT_SWEEP_VENV_REEXEC"] = "1"
    os.execve(
        str(EARLY_PROJECT_PYTHON),
        [str(EARLY_PROJECT_PYTHON), str(Path(__file__).resolve())],
        environment,
    )

import imageio.v2 as imageio
import mujoco
import numpy as np
import yaml
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_healthy_exo as healthy_utils


MYOASSIST_SOURCE = PROJECT_ROOT / "models" / "humans" / "myoassist" / "source"
BASELINE_XML = (
    MYOASSIST_SOURCE / "models" / "22muscle_2D" / "myoLeg22_2D_BASELINE.xml"
)
HEALTHY_CONTROLLER = (
    PROJECT_ROOT
    / "results"
    / "healthy_baseline"
    / "healthyhuman"
    / "healthy_controller.txt"
)
PATIENT_CONFIG_DIR = (
    PROJECT_ROOT / "models" / "humans" / "patients" / "left_knee_weakness"
)
OUTPUT_ROOT = PROJECT_ROOT / "results" / "patients" / "left_knee_weakness"

CONDITION_ORDER = ("healthy", "mild", "moderate", "severe")
CONFIG_FILES = {
    "mild": PATIENT_CONFIG_DIR / "mild.yaml",
    "moderate": PATIENT_CONFIG_DIR / "moderate.yaml",
    "severe": PATIENT_CONFIG_DIR / "severe.yaml",
}
RETENTION = {"healthy": 1.0, "mild": 0.8, "moderate": 0.6, "severe": 0.4}

DURATION_SECONDS = 20.0
FRAME_RATE = 25
VIDEO_WIDTH = 960
VIDEO_HEIGHT = 544
TARGET_SPEED_M_PER_S = 1.25
AFFECTED_MUSCLE = "vasti_l"
CONTRALATERAL_MUSCLE = "vasti_r"
ALLOWED_AFFECTED_MUSCLES = (AFFECTED_MUSCLE, CONTRALATERAL_MUSCLE)
PELVIS_HEIGHT_THRESHOLD_M = 0.65
PELVIS_PITCH_LIMIT_DEG = 60.0
HEEL_STRIKE_REFRACTORY_SECONDS = 0.6

JOINT_NAMES = (
    "hip_flexion_r",
    "knee_angle_r",
    "ankle_angle_r",
    "hip_flexion_l",
    "knee_angle_l",
    "ankle_angle_l",
)
HEEL_SITES = {"right": "r_heel_btm", "left": "l_heel_btm"}


def load_condition_configs() -> dict[str, dict[str, object]]:
    configs: dict[str, dict[str, object]] = {
        "healthy": {
            "patient_type": "healthy_control",
            "severity": "healthy",
            "affected_side": "none",
            "affected_muscle": AFFECTED_MUSCLE,
            "retained_force_fraction": 1.0,
            "description": "Unmodified healthy control condition.",
        }
    }
    for condition, path in CONFIG_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(f"Patient configuration not found: {path}")
        with path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        affected_muscles = tuple(
            config.get("affected_muscles", (config["affected_muscle"],))
        )
        expected_muscles = (
            ALLOWED_AFFECTED_MUSCLES
            if condition == "moderate"
            else (AFFECTED_MUSCLE,)
        )
        if affected_muscles != expected_muscles:
            raise ValueError(
                f"{path} modifies {affected_muscles}, expected {expected_muscles}"
            )
        if not np.isclose(
            float(config["retained_force_fraction"]), RETENTION[condition]
        ):
            raise ValueError(f"Unexpected retained force fraction in {path}")
        config["config_file"] = str(path.resolve())
        configs[condition] = config
    return configs


def make_environment(
    parameters: np.ndarray,
    config: dict[str, object],
    model_xml: Path = BASELINE_XML,
) -> tuple[object, dict[str, object]]:
    healthy_utils.DURATION_SECONDS = DURATION_SECONDS
    environment = healthy_utils.load_reflex_environment(model_xml, parameters)
    model = environment.env.sim.model
    raw_model = model.ptr
    retention = float(config["retained_force_fraction"])
    affected_muscles = tuple(
        str(name)
        for name in config.get(
            "affected_muscles", (config.get("affected_muscle", AFFECTED_MUSCLE),)
        )
    )
    if not affected_muscles or len(set(affected_muscles)) != len(affected_muscles):
        raise ValueError("affected_muscles must be a non-empty unique sequence")
    if any(name not in ALLOWED_AFFECTED_MUSCLES for name in affected_muscles):
        raise ValueError(
            f"Only {ALLOWED_AFFECTED_MUSCLES} may be weakened, got {affected_muscles}"
        )

    def configured_force(key: str, muscle: str) -> float | None:
        value = config.get(key)
        if value is None:
            return None
        if isinstance(value, dict):
            if muscle not in value:
                raise ValueError(f"{key} is missing {muscle}")
            return float(value[muscle])
        return float(value)

    gain_before = np.asarray(raw_model.actuator_gainprm).copy()
    bias_before = np.asarray(raw_model.actuator_biasprm).copy()
    muscle_verification: dict[str, dict[str, object]] = {}
    for muscle in affected_muscles:
        actuator_id = int(model.actuator(muscle).id)
        healthy_force = float(raw_model.actuator_gainprm[actuator_id, 2])
        healthy_bias_force = float(raw_model.actuator_biasprm[actuator_id, 2])
        damaged_force = healthy_force * retention
        expected_healthy = configured_force("healthy_max_isometric_force_N", muscle)
        expected_damaged = configured_force("damaged_max_isometric_force_N", muscle)
        if expected_healthy is not None and not np.isclose(
            expected_healthy, healthy_force
        ):
            raise ValueError(
                f"Configured healthy {muscle} force does not match model: "
                f"{expected_healthy} != {healthy_force}"
            )
        if expected_damaged is not None and not np.isclose(
            expected_damaged, damaged_force
        ):
            raise ValueError(
                f"Configured damaged {muscle} force does not match model: "
                f"{expected_damaged} != {damaged_force}"
            )
        raw_model.actuator_gainprm[actuator_id, 2] = damaged_force
        raw_model.actuator_biasprm[actuator_id, 2] = healthy_bias_force * retention
        muscle_verification[muscle] = {
            "actuator_index": actuator_id,
            "healthy_gain_force_N": healthy_force,
            "healthy_bias_force_N": healthy_bias_force,
            "applied_gain_force_N": damaged_force,
            "applied_bias_force_N": healthy_bias_force * retention,
            "gain_ratio": damaged_force / healthy_force,
            "bias_ratio": retention,
        }
    gain_after = np.asarray(raw_model.actuator_gainprm).copy()
    bias_after = np.asarray(raw_model.actuator_biasprm).copy()

    changed_gain = np.argwhere(gain_before != gain_after).tolist()
    changed_bias = np.argwhere(bias_before != bias_after).tolist()
    expected_changes = (
        []
        if retention == 1.0
        else sorted(
            [[int(model.actuator(name).id), 2] for name in affected_muscles]
        )
    )
    if changed_gain != expected_changes or changed_bias != expected_changes:
        raise ValueError(
            f"Unexpected muscle parameter changes: gain={changed_gain}, bias={changed_bias}"
        )

    environment.reset()
    primary = muscle_verification[affected_muscles[0]]
    verification = {
        "affected_muscles": list(affected_muscles),
        "muscles": muscle_verification,
        "affected_actuator_index": primary["actuator_index"],
        "healthy_gain_force_N": primary["healthy_gain_force_N"],
        "healthy_bias_force_N": primary["healthy_bias_force_N"],
        "retained_force_fraction": retention,
        "applied_gain_force_N": primary["applied_gain_force_N"],
        "applied_bias_force_N": primary["applied_bias_force_N"],
        "changed_gain_entries": changed_gain,
        "changed_bias_entries": changed_bias,
        "only_configured_muscle_force_scales_changed": (
            changed_gain == expected_changes and changed_bias == expected_changes
        ),
        "only_vasti_l_force_scale_changed": (
            affected_muscles == (AFFECTED_MUSCLE,)
            and changed_gain == expected_changes
            and changed_bias == expected_changes
        ),
    }
    return environment, verification


def overlay_frame(
    frame: np.ndarray,
    condition: str,
    retention: float,
    simulation_time: float,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, VIDEO_WIDTH, 52), fill=(12, 18, 24))
    draw.text(
        (14, 8),
        f"Vasti weakness: {condition} ({retention * 100:.0f}% Fmax)",
        fill=(255, 255, 255),
    )
    draw.text(
        (14, 29),
        f"healthy controller, no exoskeleton, t={simulation_time:5.2f}s",
        fill=(110, 220, 255),
    )
    return np.asarray(image)


def muscle_activation(model: object, data: object, actuator_id: int) -> float:
    activation_address = int(model.ptr.actuator_actadr[actuator_id])
    return float(data.act[activation_address]) if activation_address >= 0 else 0.0


def joint_angles(environment: object) -> dict[str, float]:
    data = environment.env.sim.data
    return {
        f"joint_{name}_rad": float(data.joint(name).qpos[0]) for name in JOINT_NAMES
    }


def state_row(
    environment: object,
    actuators: list[dict[str, object]],
    step: int,
    heel_strike_side: str,
    right_step_length: float | None,
    left_step_length: float | None,
    is_done: bool,
) -> dict[str, object]:
    model = environment.env.sim.model
    data = environment.env.sim.data
    pelvis = np.asarray(data.body("pelvis").xpos).copy()
    velocity = np.asarray(
        data.object_velocity("pelvis", "body", local_frame=False)[0]
    ).copy()
    pitch = healthy_utils.pelvis_pitch_deg(environment)
    reasons = healthy_utils.termination_reasons(environment)
    row: dict[str, object] = {
        "step": step,
        "simulation_time": float(data.time),
        "pelvis_height": float(pelvis[2]),
        "pelvis_pitch_deg": pitch,
        "forward_position": float(pelvis[0]),
        "forward_velocity": float(velocity[0]),
        "is_done": bool(is_done),
        "termination_reason": ";".join(reasons),
        "right_heel_contact": bool(environment.footstep["r_contact"]),
        "left_heel_contact": bool(environment.footstep["l_contact"]),
        "heel_strike_side": heel_strike_side,
        "right_step_length_m": "" if right_step_length is None else right_step_length,
        "left_step_length_m": "" if left_step_length is None else left_step_length,
        "right_heel_x_m": float(data.site(HEEL_SITES["right"]).xpos[0]),
        "left_heel_x_m": float(data.site(HEEL_SITES["left"]).xpos[0]),
    }
    row.update(joint_angles(environment))
    controls = np.asarray(data.ctrl)
    forces = np.asarray(data.actuator_force)
    for item in actuators:
        actuator_id = int(item["index"])
        name = str(item["name"])
        row[f"ctrl_{actuator_id:02d}_{name}"] = float(controls[actuator_id])
        row[f"actual_force_{actuator_id:02d}_{name}_N"] = float(forces[actuator_id])
        if item["is_muscle"]:
            row[f"activation_{actuator_id:02d}_{name}"] = muscle_activation(
                model, data, actuator_id
            )
            row[f"muscle_force_{actuator_id:02d}_{name}_N"] = abs(
                float(forces[actuator_id])
            )
    return row


def run_condition(
    condition: str,
    config: dict[str, object],
    parameters: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    environment, damage_verification = make_environment(parameters, config)
    environment.reset()
    model = environment.env.sim.model
    data = environment.env.sim.data
    actuators = healthy_utils.actuator_metadata(model)
    muscle_actuators = [item for item in actuators if item["is_muscle"]]
    non_muscle_actuators = [item for item in actuators if not item["is_muscle"]]
    if len(muscle_actuators) != 22 or non_muscle_actuators:
        raise ValueError(
            f"Expected the human-only 22-muscle model with no exoskeleton channels, found "
            f"{len(muscle_actuators)} and {len(non_muscle_actuators)}"
        )
    non_muscle_indices = [int(item["index"]) for item in non_muscle_actuators]

    condition_dir = OUTPUT_ROOT / condition
    condition_dir.mkdir(parents=True, exist_ok=True)
    video_path = condition_dir / "walk.mp4"
    csv_path = condition_dir / "timeseries.csv"
    json_path = condition_dir / "validation.json"

    rows: list[dict[str, object]] = []
    last_strike_time = {"right": -np.inf, "left": -np.inf}
    finite_state = True
    terminated_early = False
    capture_interval = max(1, round(1.0 / (FRAME_RATE * environment.dt)))
    total_steps = int(round(DURATION_SECONDS / environment.dt))

    with imageio.get_writer(
        video_path, fps=FRAME_RATE, codec="libx264", quality=8
    ) as video_writer:
        initial = environment.env.sim.renderer.render_offscreen(
            camera_id=4, width=VIDEO_WIDTH, height=VIDEO_HEIGHT
        )
        video_writer.append_data(
            overlay_frame(
                initial,
                condition,
                float(config["retained_force_fraction"]),
                float(data.time),
            )
        )
        rows.append(
            state_row(environment, actuators, 0, "", None, None, False)
        )

        for step in range(1, total_steps + 1):
            previous_contact = {
                "right": bool(environment.footstep["r_contact"]),
                "left": bool(environment.footstep["l_contact"]),
            }
            environment.get_sensor_data()
            action = environment.reflex2mujoco(
                environment.update(environment.SENSOR_DATA)
            )
            action[np.asarray(non_muscle_indices, dtype=int)] = 0.0
            environment.env.step(action)
            environment.update_footstep()

            state = np.concatenate(
                (
                    np.asarray(data.qpos),
                    np.asarray(data.qvel),
                    np.asarray(data.qacc),
                    np.asarray(data.ctrl),
                    np.asarray(data.actuator_force),
                )
            )
            if not np.isfinite(state).all():
                finite_state = False
                raise FloatingPointError(
                    f"{condition}: NaN or Inf detected at step {step}"
                )

            strikes: list[str] = []
            step_lengths: dict[str, float | None] = {"right": None, "left": None}
            for side, contact_key in (("right", "r_contact"), ("left", "l_contact")):
                current_contact = bool(environment.footstep[contact_key])
                if (
                    not previous_contact[side]
                    and current_contact
                    and float(data.time) - last_strike_time[side]
                    > HEEL_STRIKE_REFRACTORY_SECONDS
                ):
                    strikes.append(side)
                    last_strike_time[side] = float(data.time)
                    other_side = "left" if side == "right" else "right"
                    step_lengths[side] = float(
                        data.site(HEEL_SITES[side]).xpos[0]
                        - data.site(HEEL_SITES[other_side]).xpos[0]
                    )
            heel_strike_side = (
                "both" if len(strikes) == 2 else "".join(strikes)
            )
            is_done = bool(healthy_utils.termination_reasons(environment))
            rows.append(
                state_row(
                    environment,
                    actuators,
                    step,
                    heel_strike_side,
                    step_lengths["right"],
                    step_lengths["left"],
                    is_done,
                )
            )
            if step % capture_interval == 0:
                frame = environment.env.sim.renderer.render_offscreen(
                    camera_id=4, width=VIDEO_WIDTH, height=VIDEO_HEIGHT
                )
                video_writer.append_data(
                    overlay_frame(
                        frame,
                        condition,
                        float(config["retained_force_fraction"]),
                        float(data.time),
                    )
                )
            if is_done:
                terminated_early = True
                break

    write_csv(csv_path, rows)
    metrics = condition_metrics(rows, actuators)
    non_muscle_zero = {}
    for item in non_muscle_actuators:
        actuator_id = int(item["index"])
        name = str(item["name"])
        controls = np.asarray(
            [float(row[f"ctrl_{actuator_id:02d}_{name}"]) for row in rows]
        )
        forces = np.asarray(
            [float(row[f"actual_force_{actuator_id:02d}_{name}_N"]) for row in rows]
        )
        non_muscle_zero[name] = {
            "control_max_abs": float(np.max(np.abs(controls))),
            "force_max_abs_N": float(np.max(np.abs(forces))),
            "all_zero": bool(
                np.allclose(controls, 0.0, atol=0.0, rtol=0.0)
                and np.allclose(forces, 0.0, atol=0.0, rtol=0.0)
            ),
        }

    final = rows[-1]
    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "condition": condition,
        "patient_config": config,
        "model_xml": str(BASELINE_XML.resolve()),
        "controller": str(HEALTHY_CONTROLLER.resolve()),
        "parameter_count": int(parameters.size),
        "no_exoskeleton": True,
        "model_dimensions": {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
        },
        "damage_verification": damage_verification,
        "walking_duration_seconds": float(final["simulation_time"]),
        "terminated_early": terminated_early,
        "termination_reason": (
            str(final["termination_reason"]) if terminated_early else "duration_limit"
        ),
        "finite_state": finite_state,
        "walking_metrics": metrics,
        "dormant_ankle_exoskeleton_channels": non_muscle_zero,
        "exoskeleton_actuator_count": len(non_muscle_actuators),
        "all_exoskeleton_channels_zero": all(
            bool(item["all_zero"]) for item in non_muscle_zero.values()
        ),
        "official_reflex_controller_modified": False,
        "official_fall_conditions_modified": False,
        "artifacts": {
            "video": str(video_path.resolve()),
            "csv": str(csv_path.resolve()),
        },
    }
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return rows, result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def side_gait_metrics(rows: list[dict[str, object]], side: str) -> dict[str, object]:
    strike_times = [
        float(row["simulation_time"])
        for row in rows
        if row["heel_strike_side"] in (side, "both")
    ]
    cycle_durations = np.diff(strike_times) if len(strike_times) > 1 else np.asarray([])
    length_key = f"{side}_step_length_m"
    step_lengths = [
        float(row[length_key])
        for row in rows
        if row[length_key] not in ("", None)
    ]
    return {
        "heel_strike_count": len(strike_times),
        "heel_strike_times_seconds": strike_times,
        "cycle_count": int(len(cycle_durations)),
        "mean_cycle_seconds": (
            float(np.mean(cycle_durations)) if len(cycle_durations) else None
        ),
        "cycle_std_seconds": (
            float(np.std(cycle_durations)) if len(cycle_durations) else None
        ),
        "step_count": len(step_lengths),
        "mean_step_length_m": (
            float(np.mean(step_lengths)) if step_lengths else None
        ),
        "step_length_std_m": (
            float(np.std(step_lengths)) if step_lengths else None
        ),
        "step_lengths_m": step_lengths,
    }


def symmetry_percent(left_value: float | None, right_value: float | None) -> float | None:
    if left_value is None or right_value is None:
        return None
    denominator = abs(left_value) + abs(right_value)
    if denominator <= 1e-12:
        return None
    return float(200.0 * abs(left_value - right_value) / denominator)


def condition_metrics(
    rows: list[dict[str, object]], actuators: list[dict[str, object]]
) -> dict[str, object]:
    samples = rows[1:] if len(rows) > 1 else rows
    times = np.asarray([float(row["simulation_time"]) for row in samples])
    velocity = np.asarray([float(row["forward_velocity"]) for row in samples])
    pelvis_height = np.asarray([float(row["pelvis_height"]) for row in samples])
    pitch = np.asarray([float(row["pelvis_pitch_deg"]) for row in samples])
    half = len(times) // 2
    pitch_trend = (
        float(np.polyfit(times[half:], pitch[half:], 1)[0])
        if len(times[half:]) >= 2
        else 0.0
    )
    left_gait = side_gait_metrics(rows, "left")
    right_gait = side_gait_metrics(rows, "right")
    step_symmetry = (
        symmetry_percent(
            left_gait["mean_step_length_m"], right_gait["mean_step_length_m"]
        )
        if int(left_gait["step_count"]) >= 2 and int(right_gait["step_count"]) >= 2
        else None
    )
    cycle_symmetry = (
        symmetry_percent(
            left_gait["mean_cycle_seconds"], right_gait["mean_cycle_seconds"]
        )
        if int(left_gait["cycle_count"]) >= 2 and int(right_gait["cycle_count"]) >= 2
        else None
    )
    available_symmetry = [
        value for value in (step_symmetry, cycle_symmetry) if value is not None
    ]

    joint_statistics = {}
    for joint_name in JOINT_NAMES:
        values = np.asarray(
            [float(row[f"joint_{joint_name}_rad"]) for row in samples]
        )
        values_deg = np.rad2deg(values)
        joint_statistics[joint_name] = {
            "mean_deg": float(np.mean(values_deg)),
            "min_deg": float(np.min(values_deg)),
            "max_deg": float(np.max(values_deg)),
            "range_of_motion_deg": float(np.ptp(values_deg)),
        }

    muscle_statistics = {}
    for item in actuators:
        if not item["is_muscle"]:
            continue
        actuator_id = int(item["index"])
        name = str(item["name"])
        controls = np.asarray(
            [float(row[f"ctrl_{actuator_id:02d}_{name}"]) for row in samples]
        )
        activations = np.asarray(
            [float(row[f"activation_{actuator_id:02d}_{name}"]) for row in samples]
        )
        forces = np.asarray(
            [float(row[f"muscle_force_{actuator_id:02d}_{name}_N"]) for row in samples]
        )
        muscle_statistics[name] = {
            "control_mean": float(np.mean(controls)),
            "control_max": float(np.max(controls)),
            "activation_mean": float(np.mean(activations)),
            "activation_max": float(np.max(activations)),
            "force_mean_N": float(np.mean(forces)),
            "force_max_N": float(np.max(forces)),
        }

    return {
        "mean_forward_velocity_m_per_s": float(np.mean(velocity)),
        "forward_velocity_std_m_per_s": float(np.std(velocity)),
        "min_pelvis_height_m": float(np.min(pelvis_height)),
        "max_abs_pelvis_pitch_deg": float(np.max(np.abs(pitch))),
        "second_half_pelvis_pitch_trend_deg_per_s": pitch_trend,
        "left_gait": left_gait,
        "right_gait": right_gait,
        "gait_symmetry": {
            "step_length_symmetry_index_percent": step_symmetry,
            "cycle_time_symmetry_index_percent": cycle_symmetry,
            "combined_mean_symmetry_index_percent": (
                float(np.mean(available_symmetry)) if available_symmetry else None
            ),
            "heel_strike_count_difference": abs(
                int(left_gait["heel_strike_count"])
                - int(right_gait["heel_strike_count"])
            ),
            "sufficient_step_length_data": bool(
                int(left_gait["step_count"]) >= 2
                and int(right_gait["step_count"]) >= 2
            ),
            "sufficient_cycle_time_data": bool(
                int(left_gait["cycle_count"]) >= 2
                and int(right_gait["cycle_count"]) >= 2
            ),
            "definition": "0 percent is symmetric; larger values indicate greater left-right difference.",
        },
        "joint_statistics": joint_statistics,
        "muscle_statistics": muscle_statistics,
        "vasti_l": muscle_statistics[AFFECTED_MUSCLE],
        "vasti_r": muscle_statistics[CONTRALATERAL_MUSCLE],
    }


def aligned_joint_comparison(
    baseline_rows: list[dict[str, object]],
    condition_rows: list[dict[str, object]],
    joint_name: str,
) -> dict[str, float]:
    baseline_time = np.asarray(
        [float(row["simulation_time"]) for row in baseline_rows]
    )
    condition_time = np.asarray(
        [float(row["simulation_time"]) for row in condition_rows]
    )
    baseline_value = np.asarray(
        [float(row[f"joint_{joint_name}_rad"]) for row in baseline_rows]
    )
    condition_value = np.asarray(
        [float(row[f"joint_{joint_name}_rad"]) for row in condition_rows]
    )
    reference = np.interp(condition_time, baseline_time, baseline_value)
    difference_deg = np.rad2deg(condition_value - reference)
    condition_deg = np.rad2deg(condition_value)
    baseline_overlap_deg = np.rad2deg(
        np.interp(condition_time, baseline_time, baseline_value)
    )
    return {
        "mean_difference_deg": float(np.mean(difference_deg)),
        "mean_absolute_difference_deg": float(np.mean(np.abs(difference_deg))),
        "rmse_deg": float(np.sqrt(np.mean(np.square(difference_deg)))),
        "max_absolute_difference_deg": float(np.max(np.abs(difference_deg))),
        "condition_range_of_motion_deg": float(np.ptp(condition_deg)),
        "healthy_overlap_range_of_motion_deg": float(np.ptp(baseline_overlap_deg)),
        "range_of_motion_change_deg": float(
            np.ptp(condition_deg) - np.ptp(baseline_overlap_deg)
        ),
    }


def phase_normalized_joint_profile(
    rows: list[dict[str, object]], joint_name: str, side: str
) -> tuple[np.ndarray | None, int]:
    times = np.asarray([float(row["simulation_time"]) for row in rows])
    angles_deg = np.rad2deg(
        np.asarray([float(row[f"joint_{joint_name}_rad"]) for row in rows])
    )
    strike_times = [
        float(row["simulation_time"])
        for row in rows
        if row["heel_strike_side"] in (side, "both")
    ]
    phase = np.linspace(0.0, 1.0, 101)
    profiles = []
    for start, end in zip(strike_times[:-1], strike_times[1:]):
        duration = end - start
        if duration < 0.6 or duration > 2.0:
            continue
        mask = (times >= start) & (times <= end)
        if int(np.count_nonzero(mask)) < 3:
            continue
        normalized_time = (times[mask] - start) / duration
        profiles.append(np.interp(phase, normalized_time, angles_deg[mask]))
    if not profiles:
        return None, 0
    return np.mean(np.vstack(profiles), axis=0), len(profiles)


def phase_normalized_joint_comparison(
    healthy_rows: list[dict[str, object]],
    condition_rows: list[dict[str, object]],
    joint_name: str,
    side: str,
) -> dict[str, object]:
    healthy_profile, healthy_cycles = phase_normalized_joint_profile(
        healthy_rows, joint_name, side
    )
    condition_profile, condition_cycles = phase_normalized_joint_profile(
        condition_rows, joint_name, side
    )
    if healthy_profile is None or condition_profile is None:
        return {
            "available": False,
            "healthy_cycle_count": healthy_cycles,
            "condition_cycle_count": condition_cycles,
            "reason": "At least one condition has no complete gait cycle.",
        }
    difference = condition_profile - healthy_profile
    max_positive_index = int(np.argmax(difference))
    max_negative_index = int(np.argmin(difference))
    return {
        "available": True,
        "healthy_cycle_count": healthy_cycles,
        "condition_cycle_count": condition_cycles,
        "mean_difference_deg": float(np.mean(difference)),
        "mean_absolute_difference_deg": float(np.mean(np.abs(difference))),
        "rmse_deg": float(np.sqrt(np.mean(np.square(difference)))),
        "max_absolute_difference_deg": float(np.max(np.abs(difference))),
        "difference_at_10_percent_gait_cycle_deg": float(difference[10]),
        "difference_at_50_percent_gait_cycle_deg": float(difference[50]),
        "max_positive_difference_deg": float(difference[max_positive_index]),
        "max_positive_difference_phase_percent": max_positive_index,
        "max_negative_difference_deg": float(difference[max_negative_index]),
        "max_negative_difference_phase_percent": max_negative_index,
        "healthy_mean_cycle_range_of_motion_deg": float(np.ptp(healthy_profile)),
        "condition_mean_cycle_range_of_motion_deg": float(np.ptp(condition_profile)),
        "range_of_motion_change_deg": float(
            np.ptp(condition_profile) - np.ptp(healthy_profile)
        ),
    }


def percent_change(value: float, reference: float) -> float | None:
    if abs(reference) <= 1e-12:
        return None
    return float(100.0 * (value - reference) / abs(reference))


def build_comparison(
    rows_by_condition: dict[str, list[dict[str, object]]],
    results_by_condition: dict[str, dict[str, object]],
) -> dict[str, object]:
    healthy_result = results_by_condition["healthy"]
    healthy_metrics = healthy_result["walking_metrics"]
    healthy_vasti_l = healthy_metrics["vasti_l"]
    healthy_vasti_r = healthy_metrics["vasti_r"]
    healthy_symmetry = healthy_metrics["gait_symmetry"][
        "combined_mean_symmetry_index_percent"
    ]
    comparisons: dict[str, object] = {}
    for condition in CONDITION_ORDER[1:]:
        result = results_by_condition[condition]
        metrics = result["walking_metrics"]
        joint_changes = {
            joint_name: aligned_joint_comparison(
                rows_by_condition["healthy"],
                rows_by_condition[condition],
                joint_name,
            )
            for joint_name in JOINT_NAMES
        }
        phase_left_knee_change = phase_normalized_joint_comparison(
            rows_by_condition["healthy"],
            rows_by_condition[condition],
            "knee_angle_l",
            "left",
        )
        vasti_l = metrics["vasti_l"]
        vasti_r = metrics["vasti_r"]
        healthy_muscles = healthy_metrics["muscle_statistics"]
        condition_muscles = metrics["muscle_statistics"]
        right_side_compensation = []
        for muscle_name, muscle_values in condition_muscles.items():
            if not muscle_name.endswith("_r"):
                continue
            reference = healthy_muscles[muscle_name]
            right_side_compensation.append(
                {
                    "muscle": muscle_name,
                    "control_mean_percent": percent_change(
                        muscle_values["control_mean"], reference["control_mean"]
                    ),
                    "activation_mean_percent": percent_change(
                        muscle_values["activation_mean"],
                        reference["activation_mean"],
                    ),
                    "force_mean_percent": percent_change(
                        muscle_values["force_mean_N"], reference["force_mean_N"]
                    ),
                    "force_max_percent": percent_change(
                        muscle_values["force_max_N"], reference["force_max_N"]
                    ),
                }
            )
        right_side_compensation.sort(
            key=lambda item: float(item["force_mean_percent"] or -np.inf),
            reverse=True,
        )
        symmetry = metrics["gait_symmetry"]["combined_mean_symmetry_index_percent"]
        symmetry_change = (
            float(symmetry) - float(healthy_symmetry)
            if symmetry is not None and healthy_symmetry is not None
            else None
        )
        duration = float(result["walking_duration_seconds"])
        comparisons[condition] = {
            "retained_vasti_l_max_force_fraction": float(
                result["damage_verification"]["retained_force_fraction"]
            ),
            "walking_duration_seconds": duration,
            "terminated_early": bool(result["terminated_early"]),
            "termination_reason": result["termination_reason"],
            "mean_speed_delta_m_per_s": float(
                metrics["mean_forward_velocity_m_per_s"]
                - healthy_metrics["mean_forward_velocity_m_per_s"]
            ),
            "left_knee_trajectory_change": joint_changes["knee_angle_l"],
            "phase_normalized_left_knee_trajectory_change": phase_left_knee_change,
            "all_joint_trajectory_changes": joint_changes,
            "gait_symmetry_index_percent": symmetry,
            "gait_symmetry_change_from_healthy_percentage_points": symmetry_change,
            "vasti_l_output_change": {
                "control_mean_percent": percent_change(
                    vasti_l["control_mean"], healthy_vasti_l["control_mean"]
                ),
                "activation_mean_percent": percent_change(
                    vasti_l["activation_mean"],
                    healthy_vasti_l["activation_mean"],
                ),
                "force_mean_percent": percent_change(
                    vasti_l["force_mean_N"], healthy_vasti_l["force_mean_N"]
                ),
                "force_max_percent": percent_change(
                    vasti_l["force_max_N"], healthy_vasti_l["force_max_N"]
                ),
            },
            "vasti_r_compensation": {
                "control_mean_percent": percent_change(
                    vasti_r["control_mean"], healthy_vasti_r["control_mean"]
                ),
                "activation_mean_percent": percent_change(
                    vasti_r["activation_mean"],
                    healthy_vasti_r["activation_mean"],
                ),
                "force_mean_percent": percent_change(
                    vasti_r["force_mean_N"], healthy_vasti_r["force_mean_N"]
                ),
                "force_max_percent": percent_change(
                    vasti_r["force_max_N"], healthy_vasti_r["force_max_N"]
                ),
            },
            "contralateral_right_side_compensation": {
                "all_right_muscles": right_side_compensation,
                "top_five_mean_force_increases": right_side_compensation[:5],
                "any_right_muscle_mean_force_increase_above_5_percent": any(
                    float(item["force_mean_percent"] or 0.0) > 5.0
                    for item in right_side_compensation
                ),
                "interpretation_note": "Compensation can occur through muscles other than vasti_r because the unchanged reflex controller redistributes activity across the full right leg.",
            },
            "candidate_evidence": {
                "walks_at_least_10_seconds": duration >= 10.0,
                "completes_20_seconds": duration >= DURATION_SECONDS - 0.01,
                "left_knee_rmse_at_least_5_deg": (
                    bool(phase_left_knee_change.get("available"))
                    and float(phase_left_knee_change["rmse_deg"]) >= 5.0
                ),
                "gait_symmetry_worse_than_healthy": (
                    symmetry_change is not None and symmetry_change > 0.0
                ),
                "finite_state": bool(result["finite_state"]),
            },
        }

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "patient_family": "isolated_left_vasti_maximum_force_reduction",
        "controller": str(HEALTHY_CONTROLLER.resolve()),
        "model_xml": str(BASELINE_XML.resolve()),
        "automatic_final_patient_selection": False,
        "healthy_reference": {
            "walking_duration_seconds": healthy_result["walking_duration_seconds"],
            "mean_forward_velocity_m_per_s": healthy_metrics[
                "mean_forward_velocity_m_per_s"
            ],
            "gait_symmetry": healthy_metrics["gait_symmetry"],
            "vasti_l": healthy_vasti_l,
            "vasti_r": healthy_vasti_r,
        },
        "conditions": comparisons,
        "review_guidance": {
            "minimum_desired_duration_seconds": 10.0,
            "preferred_duration_seconds": 20.0,
            "desired_observations": [
                "visible left-knee trajectory deviation",
                "increased left-right gait asymmetry",
                "reduced speed or stability",
                "contralateral compensation",
                "finite numerical state",
            ],
            "note": "The report exposes candidate evidence but does not declare a final patient automatically.",
        },
    }


def main() -> int:
    try:
        if not BASELINE_XML.is_file():
            raise FileNotFoundError(f"Baseline model not found: {BASELINE_XML}")
        if not HEALTHY_CONTROLLER.is_file():
            raise FileNotFoundError(
                f"Healthy controller not found: {HEALTHY_CONTROLLER}"
            )
        parameters = np.loadtxt(HEALTHY_CONTROLLER).reshape(-1)
        if parameters.size != 77:
            raise ValueError(
                f"Expected 77 healthy controller parameters, found {parameters.size}"
            )
        configs = load_condition_configs()
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

        rows_by_condition: dict[str, list[dict[str, object]]] = {}
        results_by_condition: dict[str, dict[str, object]] = {}
        for condition in CONDITION_ORDER:
            print(
                f"Running {condition}: "
                f"vasti_l Fmax={RETENTION[condition] * 100:.0f}%"
            )
            rows, result = run_condition(condition, configs[condition], parameters)
            rows_by_condition[condition] = rows
            results_by_condition[condition] = result
            print(
                f"  duration={result['walking_duration_seconds']:.2f}s, "
                f"termination={result['termination_reason']}, "
                f"speed={result['walking_metrics']['mean_forward_velocity_m_per_s']:.3f}m/s"
            )

        comparison = build_comparison(rows_by_condition, results_by_condition)
        comparison_path = OUTPUT_ROOT / "comparison.json"
        comparison_path.write_text(
            json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
        )

        print("")
        print("Patient sweep complete.")
        for condition in CONDITION_ORDER:
            result = results_by_condition[condition]
            metrics = result["walking_metrics"]
            print(
                f"{condition:8s} | {result['walking_duration_seconds']:5.2f}s | "
                f"{result['termination_reason']:<32s} | "
                f"speed={metrics['mean_forward_velocity_m_per_s']:.3f} m/s | "
                f"symmetry={metrics['gait_symmetry']['combined_mean_symmetry_index_percent']}"
            )
        print(f"Comparison: {comparison_path.resolve()}")
        return 0
    except Exception as exc:
        print(f"ERROR: Patient sweep failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
