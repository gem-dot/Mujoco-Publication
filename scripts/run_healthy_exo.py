#!/usr/bin/env python3
"""Run the healthy MyoAssist controller with a zero-assistance hip-knee exoskeleton."""

from __future__ import annotations

import csv
import argparse
import importlib.metadata
import json
import os
import sys
from datetime import datetime
from pathlib import Path


EARLY_PROJECT_ROOT = Path(__file__).resolve().parents[1]
EARLY_PROJECT_PYTHON = EARLY_PROJECT_ROOT / ".venv" / "bin" / "python"
if (
    os.environ.get("HEALTHY_EXO_VENV_REEXEC") != "1"
    and EARLY_PROJECT_PYTHON.is_file()
    and Path(sys.executable).resolve() != EARLY_PROJECT_PYTHON.resolve()
):
    environment = os.environ.copy()
    environment["HEALTHY_EXO_VENV_REEXEC"] = "1"
    os.execve(
        str(EARLY_PROJECT_PYTHON),
        [str(EARLY_PROJECT_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )

import imageio.v2 as imageio
import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MYOASSIST_SOURCE = PROJECT_ROOT / "models" / "humans" / "myoassist" / "source"
HEALTHY_PARAMS = PROJECT_ROOT / "results" / "healthy_baseline" / "healthyhuman" / "healthy_controller.txt"
OLD_HEALTHY_EXO_PARAMS = PROJECT_ROOT / "results" / "healthy_exo" / "healthyexo" / "healthy_exo_controller.txt"
CORRECTED_HEALTHY_EXO_PARAMS = PROJECT_ROOT / "results" / "healthy_exo" / "corrected_model" / "healthy_exo_corrected_controller.txt"
BASELINE_XML = MYOASSIST_SOURCE / "models" / "22muscle_2D" / "myoLeg22_2D_BASELINE.xml"
COUPLED_XML = PROJECT_ROOT / "models" / "coupled" / "healthy_exo" / "healthy_exo.xml"
HEALTHY_CONFIG = MYOASSIST_SOURCE / "ctrl_optim" / "optim" / "training_configs" / "healthy_no_exo.sh"
BASELINE_RESULT = PROJECT_ROOT / "results" / "healthy_baseline" / "healthyhuman" / "validation.json"
BASELINE_TIMESERIES = PROJECT_ROOT / "results" / "healthy_baseline" / "healthyhuman" / "timeseries.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "healthy_exo" / "corrected_model"
DURATION_SECONDS = 20.0
TARGET_SPEED = 1.25
FRAME_RATE = 50
PELVIS_HEIGHT_THRESHOLD = 0.65
PELVIS_PITCH_LIMIT_DEG = 60.0


def load_reflex_environment(model_xml: Path, parameters: np.ndarray) -> object:
    sys.path.insert(0, str(MYOASSIST_SOURCE))
    from ctrl_optim.ctrl.reflex.reflex_interface import myoLeg_reflex

    model_xml = Path(model_xml).expanduser().resolve()
    is_custom_model = model_xml != BASELINE_XML.resolve()
    environment = myoLeg_reflex(
        seed=0,
        dt=0.01,
        mode="2D",
        sim_time=DURATION_SECONDS,
        init_pose="walk_left",
        control_params=parameters,
        slope_deg=0,
        delayed=False,
        exo_bool=False,
        fixed_exo=False,
        n_points=0,
        use_4param_spline=False,
        max_torque=0.0,
        model="custom" if is_custom_model else "baseline",
        model_path=str(model_xml) if is_custom_model else None,
    )
    environment.reset()
    return environment


def enum_name(enum_type: object, value: int) -> str:
    return enum_type(int(value)).name


def actuator_metadata(model: object) -> list[dict[str, object]]:
    raw_model = model.ptr
    metadata: list[dict[str, object]] = []
    for actuator_id in range(model.nu):
        transmission = enum_name(mujoco.mjtTrn, raw_model.actuator_trntype[actuator_id])
        dynamics = enum_name(mujoco.mjtDyn, raw_model.actuator_dyntype[actuator_id])
        gain = enum_name(mujoco.mjtGain, raw_model.actuator_gaintype[actuator_id])
        bias = enum_name(mujoco.mjtBias, raw_model.actuator_biastype[actuator_id])
        target_id = int(raw_model.actuator_trnid[actuator_id, 0])
        if transmission == "mjTRN_TENDON":
            target = model.tendon(target_id).name
        elif transmission in ("mjTRN_JOINT", "mjTRN_JOINTINPARENT"):
            target = model.joint(target_id).name
        else:
            target = str(target_id)
        is_muscle = (
            transmission == "mjTRN_TENDON"
            and dynamics == "mjDYN_MUSCLE"
            and gain == "mjGAIN_MUSCLE"
            and bias == "mjBIAS_MUSCLE"
        )
        metadata.append({
            "index": actuator_id,
            "name": model.actuator(actuator_id).name,
            "transmission": transmission,
            "dynamics": dynamics,
            "gain": gain,
            "bias": bias,
            "target": target,
            "is_muscle": is_muscle,
        })
    return metadata


def pelvis_pitch_deg(environment: object) -> float:
    quat = np.asarray(environment.env.sim.data.body("pelvis").xquat).copy()
    euler = environment.get_intrinsic_EulerXYZ(quat)
    return float(np.rad2deg(euler[1]))


def termination_reasons(environment: object) -> list[str]:
    data = environment.env.sim.data
    reasons: list[str] = []
    if float(data.body("pelvis").xpos[2]) < PELVIS_HEIGHT_THRESHOLD:
        reasons.append("pelvis_height_below_0.65_m")
    pitch = pelvis_pitch_deg(environment)
    if pitch < -PELVIS_PITCH_LIMIT_DEG:
        reasons.append("pelvis_pitch_below_minus_60_deg")
    if pitch > PELVIS_PITCH_LIMIT_DEG:
        reasons.append("pelvis_pitch_above_plus_60_deg")
    return reasons


def joint_values(environment: object) -> dict[str, float]:
    data = environment.env.sim.data
    names = (
        "hip_flexion_r", "knee_angle_r", "ankle_angle_r",
        "hip_flexion_l", "knee_angle_l", "ankle_angle_l",
    )
    return {f"joint_{name}": float(data.joint(name).qpos[0]) for name in names}


def step_with_all_exos_zero(environment: object, non_muscle_indices: list[int]) -> bool:
    """Use the official reflex output, then force every exo channel to exactly zero."""
    environment.get_sensor_data()
    action = environment.reflex2mujoco(environment.update(environment.SENSOR_DATA))
    action[np.asarray(non_muscle_indices, dtype=int)] = 0.0
    environment.env.step(action)
    environment.update_footstep()
    return bool(termination_reasons(environment))


def run_episode(environment: object, actuators: list[dict[str, object]], capture_video: bool) -> tuple[list[dict[str, object]], list[np.ndarray]]:
    environment.reset()
    model = environment.env.sim.model
    data = environment.env.sim.data
    muscle_indices = [int(item["index"]) for item in actuators if item["is_muscle"]]
    exo_indices = [int(item["index"]) for item in actuators if not item["is_muscle"]]
    frames: list[np.ndarray] = []
    if capture_video:
        frames.append(environment.env.sim.renderer.render_offscreen(camera_id=4, width=960, height=544))

    def row(step: int, is_done: bool, strike: str = "") -> dict[str, object]:
        pelvis = np.asarray(data.body("pelvis").xpos).copy()
        velocity = np.asarray(data.object_velocity("pelvis", "body", local_frame=False)[0]).copy()
        pitch = pelvis_pitch_deg(environment)
        reasons = termination_reasons(environment)
        values: dict[str, object] = {
            "step": step,
            "simulation_time": float(data.time),
            "pelvis_height": float(pelvis[2]),
            "pelvis_pitch_deg": pitch,
            "torso_pitch_deg": float(np.rad2deg(environment.get_intrinsic_EulerXYZ(np.asarray(data.body("torso").xquat).copy())[1])),
            "forward_position": float(pelvis[0]),
            "forward_velocity": float(velocity[0]),
            "is_done": bool(is_done),
            "termination_reason": ";".join(reasons),
            "right_heel_contact": bool(environment.footstep["r_contact"]),
            "left_heel_contact": bool(environment.footstep["l_contact"]),
            "heel_strike_side": strike,
        }
        values.update(joint_values(environment))
        controls = np.asarray(data.ctrl).copy()
        forces = np.asarray(data.actuator_force).copy()
        for item in actuators:
            index = int(item["index"])
            values[f"ctrl_{index:02d}_{item['name']}"] = float(controls[index])
            values[f"force_{index:02d}_{item['name']}"] = float(forces[index])
        for index in exo_indices:
            values[f"exo_zero_control_{index:02d}"] = float(controls[index])
        return values

    rows = [row(0, False)]
    capture_interval = max(1, round(1.0 / (FRAME_RATE * environment.dt)))
    total_steps = int(DURATION_SECONDS / environment.dt)
    for step in range(1, total_steps + 1):
        previous_r = bool(environment.footstep["r_contact"])
        previous_l = bool(environment.footstep["l_contact"])
        is_done = step_with_all_exos_zero(environment, exo_indices)
        state = np.concatenate((np.asarray(data.qpos), np.asarray(data.qvel), np.asarray(data.qacc), np.asarray(data.ctrl)))
        if not np.isfinite(state).all():
            raise FloatingPointError(f"NaN or Inf detected at step {step}")
        strikes = []
        if not previous_r and environment.footstep["r_contact"]:
            strikes.append("right")
        if not previous_l and environment.footstep["l_contact"]:
            strikes.append("left")
        strike = "both" if len(strikes) == 2 else "".join(strikes)
        rows.append(row(step, bool(is_done), strike))
        if capture_video and step % capture_interval == 0:
            frames.append(environment.env.sim.renderer.render_offscreen(camera_id=4, width=960, height=544))
        if is_done:
            break
    return rows, frames


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def episode_metrics(rows: list[dict[str, object]]) -> dict[str, float | bool | str]:
    values = rows[1:] if len(rows) > 1 else rows
    velocity = np.asarray([float(row["forward_velocity"]) for row in values])
    pitch = np.asarray([float(row["pelvis_pitch_deg"]) for row in values])
    pelvis_height = np.asarray([float(row["pelvis_height"]) for row in values])
    time = np.asarray([float(row["simulation_time"]) for row in values])
    trend = float(np.polyfit(time[len(time)//2:], pitch[len(time)//2:], 1)[0]) if len(time) > 2 else 0.0
    final = rows[-1]
    terminated = bool(final["is_done"])
    return {
        "walking_duration_seconds": float(final["simulation_time"]),
        "terminated_early": terminated,
        "termination_reason": ";".join(termination_reasons_from_row(final)) if terminated else "duration_limit",
        "finite_state": True,
        "mean_forward_velocity_m_per_s": float(np.mean(velocity)),
        "forward_velocity_std_m_per_s": float(np.std(velocity)),
        "max_abs_pelvis_pitch_deg": float(np.max(np.abs(pitch))),
        "min_pelvis_height_m": float(np.min(pelvis_height)),
        "second_half_pelvis_pitch_trend_deg_per_s": trend,
        "final_forward_position_m": float(final["forward_position"]),
    }


def termination_reasons_from_row(row: dict[str, object]) -> list[str]:
    text = str(row.get("termination_reason", ""))
    return [item for item in text.split(";") if item]


def aligned_difference(
    baseline_rows: list[dict[str, object]],
    coupled_rows: list[dict[str, object]],
    baseline_key: str,
    coupled_key: str,
) -> np.ndarray:
    baseline_time = np.asarray([float(row["simulation_time"]) for row in baseline_rows])
    coupled_time = np.asarray([float(row["simulation_time"]) for row in coupled_rows])
    baseline_value = np.asarray([float(row[baseline_key]) for row in baseline_rows])
    coupled_value = np.asarray([float(row[coupled_key]) for row in coupled_rows])
    valid = coupled_time <= baseline_time[-1]
    reference = np.interp(coupled_time[valid], baseline_time, baseline_value)
    return coupled_value[valid] - reference


def joint_trajectory_comparison(
    baseline_rows: list[dict[str, object]], coupled_rows: list[dict[str, object]]
) -> dict[str, dict[str, float]]:
    comparison: dict[str, dict[str, float]] = {}
    for joint_name in (
        "hip_flexion_r", "knee_angle_r", "ankle_angle_r",
        "hip_flexion_l", "knee_angle_l", "ankle_angle_l",
    ):
        key = f"joint_{joint_name}"
        difference_deg = np.rad2deg(aligned_difference(baseline_rows, coupled_rows, key, key))
        comparison[joint_name] = {
            "mean_difference_deg": float(np.mean(difference_deg)),
            "mean_absolute_difference_deg": float(np.mean(np.abs(difference_deg))),
            "rmse_deg": float(np.sqrt(np.mean(np.square(difference_deg)))),
            "max_absolute_difference_deg": float(np.max(np.abs(difference_deg))),
        }
    return comparison


def muscle_control_comparison(
    baseline_rows: list[dict[str, object]],
    coupled_rows: list[dict[str, object]],
    actuators: list[dict[str, object]],
) -> dict[str, dict[str, float]]:
    comparison: dict[str, dict[str, float]] = {}
    for item in actuators:
        if not item["is_muscle"]:
            continue
        key = f"ctrl_{int(item['index']):02d}_{item['name']}"
        baseline_values = np.asarray([float(row[key]) for row in baseline_rows])
        coupled_values = np.asarray([float(row[key]) for row in coupled_rows])
        difference = aligned_difference(baseline_rows, coupled_rows, key, key)
        comparison[str(item["name"])] = {
            "baseline_mean": float(np.mean(baseline_values)),
            "coupled_mean": float(np.mean(coupled_values)),
            "mean_change": float(np.mean(coupled_values) - np.mean(baseline_values)),
            "rmse_change": float(np.sqrt(np.mean(np.square(difference)))),
            "baseline_max": float(np.max(baseline_values)),
            "coupled_max": float(np.max(coupled_values)),
        }
    return comparison


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen corrected 8 kg hip-knee exoskeleton with zero assistance."
    )
    parser.add_argument(
        "--controller",
        choices=("corrected", "old"),
        default="corrected",
        help="Use the corrected-model controller (default) or the preserved pre-correction controller.",
    )
    parser.add_argument("--params", type=Path, help="Explicit 77-parameter controller file.")
    parser.add_argument(
        "--model-xml",
        type=Path,
        help="Explicit coupled XML for a historical comparison; default is the frozen corrected model.",
    )
    parser.add_argument("--duration", type=float, default=20.0, help="Simulation duration in seconds.")
    parser.add_argument("--output-dir", type=Path, help="Output directory; defaults to corrected_model.")
    parser.add_argument("--no-video", action="store_true", help="Skip MP4 rendering for quick validation.")
    return parser.parse_args()


def main() -> int:
    global COUPLED_XML, DURATION_SECONDS
    try:
        args = parse_arguments()
        if args.duration <= 0:
            raise ValueError("--duration must be positive")
        DURATION_SECONDS = float(args.duration)
        if args.model_xml is not None:
            COUPLED_XML = args.model_xml.expanduser().resolve()
        if args.params is not None:
            controller_path = args.params.expanduser().resolve()
        elif args.controller == "old":
            controller_path = OLD_HEALTHY_EXO_PARAMS
        else:
            controller_path = CORRECTED_HEALTHY_EXO_PARAMS
        if not controller_path.is_file():
            raise FileNotFoundError(f"Healthy controller not found: {controller_path}")
        if not COUPLED_XML.is_file():
            raise FileNotFoundError(f"Coupled XML not found: {COUPLED_XML}")
        if not BASELINE_RESULT.is_file() or not BASELINE_TIMESERIES.is_file():
            raise FileNotFoundError("Stored healthy baseline validation artifacts are incomplete")
        parameters = np.loadtxt(controller_path).reshape(-1)
        if parameters.size != 77:
            raise ValueError(f"Expected 77 healthy parameters, found {parameters.size}")
        baseline_parameters = np.loadtxt(HEALTHY_PARAMS).reshape(-1)
        if baseline_parameters.size != 77:
            raise ValueError(f"Expected 77 baseline parameters, found {baseline_parameters.size}")

        baseline_env = load_reflex_environment(BASELINE_XML, baseline_parameters)
        coupled_env = load_reflex_environment(COUPLED_XML, parameters)
        baseline_model = baseline_env.env.sim.model
        coupled_model = coupled_env.env.sim.model
        if coupled_model.nu != 28:
            raise ValueError(f"Expected 28 coupled actuators, found {coupled_model.nu}")
        actuators = actuator_metadata(coupled_model)
        expected_exo_names = ["HipExo_R", "HipExo_L", "KneeExo_R", "KneeExo_L", "Exo_R", "Exo_L"]
        if [item["name"] for item in actuators[-6:]] != expected_exo_names:
            raise ValueError("The final six actuators do not match the hip/knee plus dormant ankle channels")

        print(f"Human XML: {BASELINE_XML.resolve()}")
        print(f"Coupled XML: {COUPLED_XML.resolve()}")
        print(f"Healthy controller: {controller_path.resolve()} (77 parameters)")
        print(f"Coupled dimensions: nq={coupled_model.nq}, nv={coupled_model.nv}, nu={coupled_model.nu}")
        print("HipExo_R=0, HipExo_L=0, KneeExo_R=0, KneeExo_L=0, Exo_R=0, Exo_L=0")
        print("Coupled actuator list:")
        for item in actuators:
            print(f"  {item['index']:02d} {item['name']} {'muscle' if item['is_muscle'] else 'non_muscle_control'}")

        baseline_rows, _ = run_episode(baseline_env, actuator_metadata(baseline_model), capture_video=False)
        coupled_rows, frames = run_episode(coupled_env, actuators, capture_video=not args.no_video)
        if args.output_dir is not None:
            output_dir = args.output_dir.expanduser().resolve()
        elif args.controller == "old" and args.params is None:
            output_dir = (OUTPUT_DIR / "historical_old_controller").resolve()
        else:
            output_dir = OUTPUT_DIR.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = output_dir / "healthy_exo_corrected_walk.mp4"
        if not args.no_video:
            with imageio.get_writer(video_path, fps=FRAME_RATE, codec="libx264", quality=8) as writer:
                for frame in frames:
                    writer.append_data(frame)
        csv_path = output_dir / "timeseries.csv"
        json_path = output_dir / "validation.json"
        write_csv(csv_path, coupled_rows)

        coupled_metrics = episode_metrics(coupled_rows)
        baseline_metrics = episode_metrics(baseline_rows)
        stored_baseline = json.loads(BASELINE_RESULT.read_text(encoding="utf-8"))
        with BASELINE_TIMESERIES.open(encoding="utf-8", newline="") as handle:
            stored_baseline_rows = list(csv.DictReader(handle))
        stored_baseline_metrics = stored_baseline["walking_metrics"]
        body_mass = float(np.sum(coupled_model.body_mass[1:]))
        baseline_mass = float(np.sum(baseline_model.body_mass[1:]))
        exo_mass = body_mass - baseline_mass
        non_muscle_actuators = [item for item in actuators if not item["is_muscle"]]
        exo_controls = {
            item["name"]: [float(row[f"ctrl_{item['index']:02d}_{item['name']}"]) for row in coupled_rows]
            for item in non_muscle_actuators
        }
        exo_forces = {
            item["name"]: [float(row[f"force_{item['index']:02d}_{item['name']}"]) for row in coupled_rows]
            for item in non_muscle_actuators
        }
        joint_comparison = joint_trajectory_comparison(baseline_rows, coupled_rows)
        muscle_comparison = muscle_control_comparison(
            baseline_rows, coupled_rows, actuators
        )
        muscle_control_statistics = {}
        for item in actuators:
            if not item["is_muscle"]:
                continue
            key = f"ctrl_{int(item['index']):02d}_{item['name']}"
            values = np.asarray([float(row[key]) for row in coupled_rows])
            muscle_control_statistics[str(item["name"])] = {
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "has_nonzero_control": bool(np.any(values != 0.0)),
            }
        all_exo_controls_zero = all(np.allclose(values, 0.0, atol=0.0, rtol=0.0) for values in exo_controls.values())
        all_exo_forces_zero = all(np.allclose(values, 0.0, atol=0.0, rtol=0.0) for values in exo_forces.values())
        all_22_muscles_active = len(muscle_control_statistics) == 22 and all(
            item["has_nonzero_control"] for item in muscle_control_statistics.values()
        )
        speed_in_target_range = 1.0 <= float(coupled_metrics["mean_forward_velocity_m_per_s"]) <= 1.5
        pelvis_height_above_threshold = float(coupled_metrics["min_pelvis_height_m"]) >= PELVIS_HEIGHT_THRESHOLD
        pelvis_pitch_not_sustained_increase = float(coupled_metrics["second_half_pelvis_pitch_trend_deg_per_s"]) <= 0.5
        passed = bool(
            not coupled_metrics["terminated_early"]
            and coupled_metrics["finite_state"]
            and float(coupled_metrics["walking_duration_seconds"]) >= DURATION_SECONDS - coupled_env.dt
            and speed_in_target_range
            and pelvis_height_above_threshold
            and pelvis_pitch_not_sustained_increase
            and all_22_muscles_active
            and all_exo_controls_zero
            and all_exo_forces_zero
        )
        result = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model_xml": str(COUPLED_XML.resolve()),
            "human_baseline_xml": str(BASELINE_XML.resolve()),
            "healthy_controller": str(controller_path.resolve()),
            "healthy_baseline_controller": str(HEALTHY_PARAMS.resolve()),
            "parameter_count": int(parameters.size),
            "requested_duration_seconds": DURATION_SECONDS,
            "validation_status": "PASSED" if passed else "FAILED",
            "validation_checks": {
                "duration_reached": bool(float(coupled_metrics["walking_duration_seconds"]) >= DURATION_SECONDS - coupled_env.dt),
                "not_terminated_early": bool(not coupled_metrics["terminated_early"]),
                "finite_state": bool(coupled_metrics["finite_state"]),
                "mean_speed_in_1.0_to_1.5_m_per_s": speed_in_target_range,
                "pelvis_height_never_below_0.65_m": pelvis_height_above_threshold,
                "second_half_pelvis_pitch_not_sustained_increase": pelvis_pitch_not_sustained_increase,
                "all_22_muscles_have_nonzero_control": all_22_muscles_active,
                "all_six_exo_controls_exactly_zero": all_exo_controls_zero,
                "all_six_exo_forces_exactly_zero": all_exo_forces_zero,
            },
            "source_commit": "c10883728bbe20894d1ab59b848d3f4ba57c2a29",
            "dependencies": {
                "python": sys.version.split()[0],
                "mujoco": mujoco.__version__,
                "numpy": np.__version__,
                "dm_control": importlib.metadata.version("dm-control"),
            },
            "coupled_dimensions": {"nq": int(coupled_model.nq), "nv": int(coupled_model.nv), "nu": int(coupled_model.nu)},
            "actuators": actuators,
            "exo_assistance_enabled": False,
            "exo_controls": {name: {"min": float(np.min(values)), "max": float(np.max(values)), "all_zero": bool(np.allclose(values, 0.0))} for name, values in exo_controls.items()},
            "exo_actuator_forces_Nm": {name: {"min": float(np.min(values)), "max": float(np.max(values)), "all_zero": bool(np.allclose(values, 0.0))} for name, values in exo_forces.items()},
            "muscle_control_statistics": muscle_control_statistics,
            "hip_knee_exo_controls": {name: {"min": float(np.min(exo_controls[name])), "max": float(np.max(exo_controls[name])), "all_zero": bool(np.allclose(exo_controls[name], 0.0))} for name in expected_exo_names[:4]},
            "ankle_exo_controls": {name: {"min": float(np.min(exo_controls[name])), "max": float(np.max(exo_controls[name])), "all_zero": bool(np.allclose(exo_controls[name], 0.0))} for name in expected_exo_names[4:]},
            "mass": {"human_model_kg": baseline_mass, "coupled_model_kg": body_mass, "added_exoskeleton_kg": exo_mass, "configured_exoskeleton_kg": 8.0},
            "coupling": {"method": "rigid_body_attachment", "independent_exoskeleton_dofs": 0, "collision_with_human": "disabled_for_ideal_alignment"},
            "healthy_exo_metrics": coupled_metrics,
            "healthy_baseline_metrics_replay": baseline_metrics,
            "stored_healthy_baseline": {
                "validation_json": str(BASELINE_RESULT.resolve()),
                "timeseries_csv": str(BASELINE_TIMESERIES.resolve()),
                "stored_row_count": len(stored_baseline_rows),
                "walking_duration_seconds": stored_baseline["walking_duration_seconds"],
                "mean_forward_velocity_m_per_s": stored_baseline_metrics["mean_forward_velocity_m_per_s"],
                "max_abs_pelvis_pitch_deg": stored_baseline_metrics["max_abs_pelvis_pitch_deg"],
            },
            "baseline_comparison": {
                "duration_delta_seconds": float(coupled_metrics["walking_duration_seconds"]) - float(stored_baseline["walking_duration_seconds"]),
                "mean_speed_delta_m_per_s": float(coupled_metrics["mean_forward_velocity_m_per_s"]) - float(stored_baseline_metrics["mean_forward_velocity_m_per_s"]),
                "max_pitch_delta_deg": float(coupled_metrics["max_abs_pelvis_pitch_deg"]) - float(stored_baseline_metrics["max_abs_pelvis_pitch_deg"]),
                "joint_trajectory_source": "baseline replay with the unchanged healthy controller because the stored baseline CSV does not contain joint angles",
                "joint_trajectory_changes": joint_comparison,
                "muscle_control_changes": muscle_comparison,
            },
            "video": str(video_path) if not args.no_video else None,
            "csv": str(csv_path),
        }
        json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        environment_versions = {
            "mechanical_model": "corrected_hip_knee_exo_v1",
            "myoassist_commit": "c10883728bbe20894d1ab59b848d3f4ba57c2a29",
            "python": sys.version.split()[0],
            "mujoco": mujoco.__version__,
            "dm_control": importlib.metadata.version("dm-control"),
            "numpy": np.__version__,
            "parameter_count": int(parameters.size),
            "model_xml": str(COUPLED_XML.resolve()),
            "controller": str(controller_path.resolve()),
            "timestep_seconds": float(coupled_model.opt.timestep),
        }
        (output_dir / "environment_versions.json").write_text(
            json.dumps(environment_versions, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({k: result[k] for k in ("mass", "healthy_exo_metrics", "baseline_comparison", "video", "csv")}, indent=2))
        return 0
    except Exception as exc:
        print(f"ERROR: Healthy zero-assistance exoskeleton run failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
