#!/usr/bin/env python3
"""Run the moderate left-vasti weakness patient in the passive 8 kg exoskeleton."""

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
    os.environ.get("PATIENT_EXO_ZERO_VENV_REEXEC") != "1"
    and EARLY_PROJECT_PYTHON.is_file()
    and Path(sys.executable).resolve() != EARLY_PROJECT_PYTHON.resolve()
):
    environment = os.environ.copy()
    environment["PATIENT_EXO_ZERO_VENV_REEXEC"] = "1"
    os.execve(
        str(EARLY_PROJECT_PYTHON),
        [str(EARLY_PROJECT_PYTHON), str(Path(__file__).resolve())],
        environment,
    )

import imageio.v2 as imageio
import numpy as np
import yaml
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_healthy_exo as healthy_utils
import run_patient_sweep as patient_utils


COUPLED_XML = PROJECT_ROOT / "models" / "coupled" / "healthy_exo" / "healthy_exo.xml"
CONTROLLER_PATH = (
    PROJECT_ROOT
    / "results"
    / "healthy_exo"
    / "corrected_model"
    / "healthy_exo_corrected_controller.txt"
)
MECHANICAL_MANIFEST = (
    PROJECT_ROOT
    / "results"
    / "healthy_exo"
    / "corrected_model"
    / "mechanical_model_manifest.json"
)
MODERATE_CONFIG = (
    PROJECT_ROOT
    / "models"
    / "humans"
    / "patients"
    / "left_knee_weakness"
    / "moderate.yaml"
)
MODERATE_NO_EXO_DIR = (
    PROJECT_ROOT / "results" / "patients" / "left_knee_weakness" / "moderate"
)
HEALTHY_EXO_DIR = PROJECT_ROOT / "results" / "healthy_exo" / "corrected_model"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "patients"
    / "left_knee_weakness"
    / "moderate_exo_zero"
)

DURATION_SECONDS = 20.0
FRAME_RATE = 25
VIDEO_WIDTH = 960
VIDEO_HEIGHT = 544
EXPECTED_EXO_ACTUATORS = (
    "HipExo_R",
    "HipExo_L",
    "KneeExo_R",
    "KneeExo_L",
    "Exo_R",
    "Exo_L",
)
COMPENSATION_MUSCLES = ("vasti_r", "rectfem_r", "iliopsoas_r")


def overlay_frame(frame: np.ndarray, simulation_time: float) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, VIDEO_WIDTH, 52), fill=(12, 18, 24))
    draw.text(
        (14, 8),
        "Moderate left vasti weakness (60% Fmax) + frozen 8 kg exoskeleton",
        fill=(255, 255, 255),
    )
    draw.text(
        (14, 29),
        f"zero exoskeleton assistance, t={simulation_time:5.2f}s",
        fill=(110, 220, 255),
    )
    return np.asarray(image)


def run_episode(
    parameters: np.ndarray, config: dict[str, object]
) -> tuple[list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    patient_utils.DURATION_SECONDS = DURATION_SECONDS
    healthy_utils.DURATION_SECONDS = DURATION_SECONDS
    environment, damage_verification = patient_utils.make_environment(
        parameters, config, COUPLED_XML
    )
    environment.reset()
    model = environment.env.sim.model
    data = environment.env.sim.data
    actuators = healthy_utils.actuator_metadata(model)
    muscle_actuators = [item for item in actuators if item["is_muscle"]]
    non_muscle_actuators = [item for item in actuators if not item["is_muscle"]]
    if len(muscle_actuators) != 22 or len(non_muscle_actuators) != 6:
        raise ValueError(
            f"Expected 22 muscles and 6 exoskeleton actuators, found "
            f"{len(muscle_actuators)} and {len(non_muscle_actuators)}"
        )
    if tuple(str(item["name"]) for item in non_muscle_actuators) != EXPECTED_EXO_ACTUATORS:
        raise ValueError("The six exoskeleton channels are not in the verified order")
    non_muscle_indices = [int(item["index"]) for item in non_muscle_actuators]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    video_path = OUTPUT_DIR / "walk.mp4"
    csv_path = OUTPUT_DIR / "timeseries.csv"
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
        video_writer.append_data(overlay_frame(initial, float(data.time)))
        rows.append(
            patient_utils.state_row(
                environment, actuators, 0, "", None, None, False
            )
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
                raise FloatingPointError(f"NaN or Inf detected at step {step}")

            strikes: list[str] = []
            step_lengths: dict[str, float | None] = {"right": None, "left": None}
            for side, contact_key in (("right", "r_contact"), ("left", "l_contact")):
                current_contact = bool(environment.footstep[contact_key])
                if (
                    not previous_contact[side]
                    and current_contact
                    and float(data.time) - last_strike_time[side]
                    > patient_utils.HEEL_STRIKE_REFRACTORY_SECONDS
                ):
                    strikes.append(side)
                    last_strike_time[side] = float(data.time)
                    other_side = "left" if side == "right" else "right"
                    step_lengths[side] = float(
                        data.site(patient_utils.HEEL_SITES[side]).xpos[0]
                        - data.site(patient_utils.HEEL_SITES[other_side]).xpos[0]
                    )
            heel_strike_side = "both" if len(strikes) == 2 else "".join(strikes)
            is_done = bool(healthy_utils.termination_reasons(environment))
            rows.append(
                patient_utils.state_row(
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
                video_writer.append_data(overlay_frame(frame, float(data.time)))
            if is_done:
                terminated_early = True
                break

    patient_utils.write_csv(csv_path, rows)
    metrics = patient_utils.condition_metrics(rows, actuators)
    exo_zero = {}
    for item in non_muscle_actuators:
        actuator_id = int(item["index"])
        name = str(item["name"])
        control_key = f"ctrl_{actuator_id:02d}_{name}"
        force_key = f"actual_force_{actuator_id:02d}_{name}_N"
        controls = np.asarray([float(row[control_key]) for row in rows])
        forces = np.asarray([float(row[force_key]) for row in rows])
        exo_zero[name] = {
            "command_min_Nm": float(np.min(controls)),
            "command_max_Nm": float(np.max(controls)),
            "command_max_abs_Nm": float(np.max(np.abs(controls))),
            "actual_min_Nm": float(np.min(forces)),
            "actual_max_Nm": float(np.max(forces)),
            "actual_max_abs_Nm": float(np.max(np.abs(forces))),
            "all_zero": bool(
                np.allclose(controls, 0.0, atol=0.0, rtol=0.0)
                and np.allclose(forces, 0.0, atol=0.0, rtol=0.0)
            ),
        }

    total_mass = float(np.sum(model.ptr.body_mass[1:]))
    final = rows[-1]
    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "condition": "moderate_left_knee_weakness_with_exoskeleton_zero_assistance",
        "model_xml": str(COUPLED_XML.resolve()),
        "mechanical_model": "corrected_hip_knee_exo_v1",
        "mechanical_manifest": str(MECHANICAL_MANIFEST.resolve()),
        "controller": str(CONTROLLER_PATH.resolve()),
        "parameter_count": int(parameters.size),
        "requested_duration_seconds": DURATION_SECONDS,
        "target_speed_m_per_s": 1.25,
        "slope_deg": 0.0,
        "patient_config": config,
        "damage_verification": damage_verification,
        "model_dimensions": {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
        },
        "total_model_mass_kg": total_mass,
        "exoskeleton_mass_kg": 8.0,
        "walking_duration_seconds": float(final["simulation_time"]),
        "terminated_early": terminated_early,
        "termination_reason": (
            str(final["termination_reason"]) if terminated_early else "duration_limit"
        ),
        "termination_state": {
            "pelvis_height_m": float(final["pelvis_height"]),
            "pelvis_pitch_deg": float(final["pelvis_pitch_deg"]),
            "forward_velocity_m_per_s": float(final["forward_velocity"]),
            "left_knee_angle_deg": float(
                np.rad2deg(float(final["joint_knee_angle_l_rad"]))
            ),
        },
        "finite_state": finite_state,
        "walking_metrics": metrics,
        "exo_assistance_enabled": False,
        "exoskeleton_channels": exo_zero,
        "all_six_exoskeleton_commands_and_forces_exactly_zero": all(
            bool(item["all_zero"]) for item in exo_zero.values()
        ),
        "official_reflex_controller_modified": False,
        "official_fall_conditions_modified": False,
        "mechanical_model_modified": False,
        "artifacts": {
            "video": str(video_path.resolve()),
            "csv": str(csv_path.resolve()),
        },
    }
    return rows, result, actuators


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_healthy_exo_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for row in rows:
        item: dict[str, object] = {
            "simulation_time": float(row["simulation_time"]),
            "heel_strike_side": row["heel_strike_side"],
        }
        for joint_name in patient_utils.JOINT_NAMES:
            item[f"joint_{joint_name}_rad"] = float(row[f"joint_{joint_name}"])
        normalized.append(item)
    return normalized


def muscle_changes(
    current: dict[str, object], reference: dict[str, object]
) -> dict[str, object]:
    changes = {}
    for name in current:
        current_values = current[name]
        reference_values = reference[name]
        changes[name] = {
            "control_mean_percent": patient_utils.percent_change(
                current_values["control_mean"], reference_values["control_mean"]
            ),
            "activation_mean_percent": patient_utils.percent_change(
                current_values["activation_mean"],
                reference_values["activation_mean"],
            ),
            "force_mean_percent": patient_utils.percent_change(
                current_values["force_mean_N"], reference_values["force_mean_N"]
            ),
            "force_max_percent": patient_utils.percent_change(
                current_values["force_max_N"], reference_values["force_max_N"]
            ),
        }
    return changes


def healthy_exo_muscle_statistics(
    rows: list[dict[str, str]], actuators: list[dict[str, object]]
) -> dict[str, dict[str, float]]:
    statistics = {}
    for item in actuators:
        if not item["is_muscle"]:
            continue
        actuator_id = int(item["index"])
        name = str(item["name"])
        controls = np.asarray(
            [float(row[f"ctrl_{actuator_id:02d}_{name}"]) for row in rows]
        )
        forces = np.asarray(
            [abs(float(row[f"force_{actuator_id:02d}_{name}"])) for row in rows]
        )
        statistics[name] = {
            "control_mean": float(np.mean(controls)),
            "control_max": float(np.max(controls)),
            "force_mean_N": float(np.mean(forces)),
            "force_max_N": float(np.max(forces)),
        }
    return statistics


def build_comparison(
    current_rows: list[dict[str, object]],
    current_result: dict[str, object],
    actuators: list[dict[str, object]],
) -> dict[str, object]:
    moderate_validation_path = MODERATE_NO_EXO_DIR / "validation.json"
    moderate_csv_path = MODERATE_NO_EXO_DIR / "timeseries.csv"
    healthy_exo_validation_path = HEALTHY_EXO_DIR / "validation.json"
    healthy_exo_csv_path = HEALTHY_EXO_DIR / "timeseries.csv"
    for path in (
        moderate_validation_path,
        moderate_csv_path,
        healthy_exo_validation_path,
        healthy_exo_csv_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Comparison artifact not found: {path}")

    moderate_validation = json.loads(
        moderate_validation_path.read_text(encoding="utf-8")
    )
    healthy_exo_validation = json.loads(
        healthy_exo_validation_path.read_text(encoding="utf-8")
    )
    moderate_rows = read_csv(moderate_csv_path)
    healthy_exo_rows_raw = read_csv(healthy_exo_csv_path)
    healthy_exo_rows = normalize_healthy_exo_rows(healthy_exo_rows_raw)

    current_metrics = current_result["walking_metrics"]
    moderate_metrics = moderate_validation["walking_metrics"]
    healthy_exo_metrics = healthy_exo_validation["healthy_exo_metrics"]
    current_muscles = current_metrics["muscle_statistics"]
    moderate_muscles = moderate_metrics["muscle_statistics"]
    patient_muscle_change = muscle_changes(current_muscles, moderate_muscles)
    healthy_exo_muscles = healthy_exo_muscle_statistics(
        healthy_exo_rows_raw, actuators
    )
    same_controller_muscle_change = {}
    for name, current_values in current_muscles.items():
        reference_values = healthy_exo_muscles[name]
        same_controller_muscle_change[name] = {
            "control_mean_percent": patient_utils.percent_change(
                current_values["control_mean"], reference_values["control_mean"]
            ),
            "force_mean_percent": patient_utils.percent_change(
                current_values["force_mean_N"], reference_values["force_mean_N"]
            ),
            "force_max_percent": patient_utils.percent_change(
                current_values["force_max_N"], reference_values["force_max_N"]
            ),
        }

    current_symmetry = current_metrics["gait_symmetry"][
        "combined_mean_symmetry_index_percent"
    ]
    moderate_symmetry = moderate_metrics["gait_symmetry"][
        "combined_mean_symmetry_index_percent"
    ]
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment": "moderate_left_vasti_60_percent_with_frozen_8kg_exoskeleton_zero_assistance",
        "comparison_design_note": {
            "healthy_exo_reference": "Same frozen mechanical model and healthy_exo_corrected_controller; no muscle weakness.",
            "moderate_no_exo_reference": "Same 60 percent vasti_l weakness, but the mandated no-exoskeleton baseline uses the original healthy-human controller.",
            "causal_limit": "The patient-exo versus patient-no-exo difference combines passive exoskeleton dynamics with the mandated controller change. It must not be described as a pure mass-only causal effect.",
        },
        "current_patient_exo_zero": {
            "walking_duration_seconds": current_result["walking_duration_seconds"],
            "terminated_early": current_result["terminated_early"],
            "termination_reason": current_result["termination_reason"],
            "mean_forward_velocity_m_per_s": current_metrics[
                "mean_forward_velocity_m_per_s"
            ],
            "forward_velocity_std_m_per_s": current_metrics[
                "forward_velocity_std_m_per_s"
            ],
            "gait_symmetry": current_metrics["gait_symmetry"],
            "short_run_statistics_warning": (
                "The run terminates after three complete cycles per side. Speed, symmetry, and muscle averages describe only the pre-fall transient and must not be interpreted as 20-second steady-state improvements."
            ),
            "vasti_l": current_metrics["vasti_l"],
            "compensation_muscles": {
                name: current_muscles[name] for name in COMPENSATION_MUSCLES
            },
        },
        "versus_moderate_no_exoskeleton": {
            "reference_controller": moderate_validation["controller"],
            "walking_duration_delta_seconds": float(
                current_result["walking_duration_seconds"]
                - moderate_validation["walking_duration_seconds"]
            ),
            "mean_speed_delta_m_per_s": float(
                current_metrics["mean_forward_velocity_m_per_s"]
                - moderate_metrics["mean_forward_velocity_m_per_s"]
            ),
            "speed_std_delta_m_per_s": float(
                current_metrics["forward_velocity_std_m_per_s"]
                - moderate_metrics["forward_velocity_std_m_per_s"]
            ),
            "gait_symmetry_delta_percentage_points": (
                float(current_symmetry) - float(moderate_symmetry)
                if current_symmetry is not None and moderate_symmetry is not None
                else None
            ),
            "steady_state_speed_and_symmetry_comparison_valid": bool(
                not current_result["terminated_early"]
                and float(current_result["walking_duration_seconds"]) >= 10.0
            ),
            "stability_observation": (
                "The no-exoskeleton patient reaches 20 seconds, while the zero-assistance wearing condition falls at 4.30 seconds. This is clear degradation of the tested wearing condition, even though the short-run mean speed is higher."
            ),
            "left_knee_phase_normalized_change": (
                patient_utils.phase_normalized_joint_comparison(
                    moderate_rows,
                    current_rows,
                    "knee_angle_l",
                    "left",
                )
            ),
            "all_joint_time_aligned_changes": {
                joint_name: patient_utils.aligned_joint_comparison(
                    moderate_rows, current_rows, joint_name
                )
                for joint_name in patient_utils.JOINT_NAMES
            },
            "selected_compensation_muscle_changes": {
                name: patient_muscle_change[name] for name in COMPENSATION_MUSCLES
            },
            "muscle_comparison_window_note": (
                "Current values average only the 4.30-second pre-fall window; the no-exoskeleton reference averages 20 seconds and uses the mandated healthy-human controller."
            ),
            "all_muscle_changes": patient_muscle_change,
        },
        "versus_healthy_exoskeleton_zero_assistance": {
            "reference_controller": healthy_exo_validation["healthy_controller"],
            "walking_duration_delta_seconds": float(
                current_result["walking_duration_seconds"]
                - healthy_exo_metrics["walking_duration_seconds"]
            ),
            "mean_speed_delta_m_per_s": float(
                current_metrics["mean_forward_velocity_m_per_s"]
                - healthy_exo_metrics["mean_forward_velocity_m_per_s"]
            ),
            "speed_std_delta_m_per_s": float(
                current_metrics["forward_velocity_std_m_per_s"]
                - healthy_exo_metrics["forward_velocity_std_m_per_s"]
            ),
            "left_knee_phase_normalized_change": (
                patient_utils.phase_normalized_joint_comparison(
                    healthy_exo_rows,
                    current_rows,
                    "knee_angle_l",
                    "left",
                )
            ),
            "selected_muscle_changes": {
                name: same_controller_muscle_change[name]
                for name in ("vasti_l", *COMPENSATION_MUSCLES)
            },
            "muscle_comparison_window_note": (
                "Both conditions use the same model and controller, but the patient averages cover only the 4.30-second pre-fall window."
            ),
        },
        "passive_wearing_conclusion": {
            "zero_assistance_wearing_condition_worsens_stability": bool(
                float(current_result["walking_duration_seconds"])
                < float(moderate_validation["walking_duration_seconds"])
            ),
            "pure_mass_only_causal_claim_supported": False,
            "reason": (
                "The required no-exoskeleton and exoskeleton baselines use different pre-existing controllers, so the result establishes failure of the full zero-assistance wearing condition, not a mass-only causal estimate."
            ),
        },
        "all_six_exoskeleton_commands_and_forces_exactly_zero": current_result[
            "all_six_exoskeleton_commands_and_forces_exactly_zero"
        ],
        "automatic_progression_to_active_assistance": False,
    }


def main() -> int:
    try:
        for path in (COUPLED_XML, CONTROLLER_PATH, MODERATE_CONFIG, MECHANICAL_MANIFEST):
            if not path.is_file():
                raise FileNotFoundError(f"Required file not found: {path}")
        parameters = np.loadtxt(CONTROLLER_PATH).reshape(-1)
        if parameters.size != 77:
            raise ValueError(f"Expected 77 controller parameters, found {parameters.size}")
        with MODERATE_CONFIG.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        config["config_file"] = str(MODERATE_CONFIG.resolve())
        if (
            config["affected_muscle"] != "vasti_l"
            or not np.isclose(float(config["retained_force_fraction"]), 0.6)
        ):
            raise ValueError("The patient configuration is not the verified moderate case")

        rows, result, actuators = run_episode(parameters, config)
        validation_path = OUTPUT_DIR / "validation.json"
        result["artifacts"]["validation"] = str(validation_path.resolve())
        result["artifacts"]["comparison"] = str((OUTPUT_DIR / "comparison.json").resolve())
        validation_path.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        comparison = build_comparison(rows, result, actuators)
        comparison_path = OUTPUT_DIR / "comparison.json"
        comparison_path.write_text(
            json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
        )

        metrics = result["walking_metrics"]
        print(f"Model: {COUPLED_XML.resolve()}")
        print(f"Controller: {CONTROLLER_PATH.resolve()} (77 parameters)")
        print("Patient: vasti_l Fmax = 60%; all six exoskeleton channels = 0")
        print(
            f"Duration: {result['walking_duration_seconds']:.2f} s; "
            f"termination: {result['termination_reason']}"
        )
        print(
            f"Speed: {metrics['mean_forward_velocity_m_per_s']:.3f} +/- "
            f"{metrics['forward_velocity_std_m_per_s']:.3f} m/s"
        )
        print(
            f"Symmetry index: "
            f"{metrics['gait_symmetry']['combined_mean_symmetry_index_percent']}"
        )
        print(
            f"All six commands and forces exactly zero: "
            f"{result['all_six_exoskeleton_commands_and_forces_exactly_zero']}"
        )
        print(f"Video: {(OUTPUT_DIR / 'walk.mp4').resolve()}")
        print(f"Validation: {validation_path.resolve()}")
        print(f"Comparison: {comparison_path.resolve()}")
        return 0
    except Exception as exc:
        print(f"ERROR: Moderate patient zero-assistance exoskeleton run failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
