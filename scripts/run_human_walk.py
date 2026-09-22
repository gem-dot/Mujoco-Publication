#!/usr/bin/env python3
"""Run and diagnose the official MyoAssist 2D reflex controller."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MYOASSIST_SOURCE = PROJECT_ROOT / "models" / "humans" / "myoassist" / "source"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "myoassist_walk"
HEALTHY_OUTPUT_DIR = PROJECT_ROOT / "results" / "healthy_baseline"
TUTORIAL_DIR = (
    MYOASSIST_SOURCE
    / "ctrl_optim"
    / "results"
    / "optim_results"
    / "tutorial_prep"
)
TUTORIAL_CONFIG = TUTORIAL_DIR / "tutorial_prep.bat"
TUTORIAL_BESTLAST = (
    TUTORIAL_DIR
    / "myorfl_Kine_2D_1_25_2025Jul25_1827_None_BestLast.txt"
)
TUTORIAL_XML = (
    MYOASSIST_SOURCE
    / "models"
    / "22muscle_2D"
    / "myoLeg22_2D_TUTORIAL.xml"
)
BASELINE_CONFIG = (
    MYOASSIST_SOURCE
    / "ctrl_optim"
    / "optim"
    / "training_configs"
    / "baseline.bat"
)
HEALTHY_CONFIG = (
    MYOASSIST_SOURCE
    / "ctrl_optim"
    / "optim"
    / "training_configs"
    / "healthy_no_exo.sh"
)
BASELINE_XML = (
    MYOASSIST_SOURCE
    / "models"
    / "22muscle_2D"
    / "myoLeg22_2D_BASELINE.xml"
)
PARAMETER_COUNT = 77
PELVIS_HEIGHT_THRESHOLD = 0.65
PELVIS_PITCH_LIMIT_DEG = 60.0
HEALTHY_DURATION_SECONDS = 20.0
HEALTHY_SPEED_TOLERANCE = 0.25
HEALTHY_MAX_PITCH_TREND_DEG_PER_SECOND = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and diagnose MyoAssist's official 2D reflex controller."
    )
    parser.add_argument(
        "--mode",
        choices=("minimal", "optimized"),
        default="optimized",
        help="Controller mode (default: optimized).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Maximum simulation seconds.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for minimal mode.")
    parser.add_argument(
        "--params",
        type=Path,
        help="Optional 77-value human reflex parameter file.",
    )
    parser.add_argument(
        "--entry",
        choices=("wrapper", "official"),
        default="wrapper",
        help="Use the direct wrapper or MyoAssist's official config loader.",
    )
    parser.add_argument(
        "--healthy-validation",
        action="store_true",
        help="Run the 20-second baseline no-exoskeleton acceptance test.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip video generation while retaining JSON and CSV diagnostics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for videos and diagnostic logs.",
    )
    return parser.parse_args()


def require_source(config_path: Path, model_xml: Path) -> None:
    required = [
        MYOASSIST_SOURCE / "ctrl_optim" / "ctrl" / "reflex" / "reflex_interface.py",
        MYOASSIST_SOURCE / "ctrl_optim" / "optim" / "optim_utils" / "config_parser.py",
        MYOASSIST_SOURCE / "myosuite",
        config_path,
        model_xml,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        paths = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"MyoAssist sparse checkout is incomplete:\n{paths}")


def load_training_config(config_path: Path) -> dict[str, object]:
    sys.path.insert(0, str(MYOASSIST_SOURCE))
    from ctrl_optim.optim.optim_utils.config_parser import parse_bat_config

    config = dict(parse_bat_config(str(config_path)))
    content = config_path.read_text(encoding="utf-8")
    target_speed = re.search(r"--tgt_vel\s+([\d.-]+)", content)
    muscle_model = re.search(r"--musc_model\s+(\w+)", content)
    config["target_speed"] = float(target_speed.group(1)) if target_speed else None
    config["muscle_model"] = muscle_model.group(1) if muscle_model else None
    config["config_file"] = str(config_path)
    return config


def tutorial_inventory() -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    for path in sorted(TUTORIAL_DIR.iterdir()):
        if not path.is_file():
            continue
        lower_name = path.name.lower()
        if lower_name.endswith("_bestlast.txt"):
            kind = "BestLast"
        elif lower_name.endswith("_best.txt"):
            kind = "Best"
        elif path.suffix.lower() in (".pkl", ".pickle"):
            kind = "pickle"
        elif path.suffix.lower() in (".bat", ".sh", ".yaml", ".yml", ".json"):
            kind = "config"
        else:
            kind = "other"
        item: dict[str, object] = {"path": str(path), "kind": kind}
        if path.suffix.lower() == ".txt":
            item["parameter_count"] = int(np.loadtxt(path).reshape(-1).size)
        inventory.append(item)
    return inventory


def expected_parameter_count(config: dict[str, object]) -> int:
    base_count = PARAMETER_COUNT if config["mode"] == "2D" else 97
    if not config["exo_bool"]:
        return base_count
    if config["use_4param_spline"]:
        return base_count + 4
    return base_count + 2 * int(config["n_points"])


def load_parameters(
    mode: str, seed: int, parameter_file: Path | None
) -> tuple[np.ndarray, str, Path | None]:
    if mode == "minimal":
        if parameter_file is not None:
            raise ValueError("--params can only be used with --mode optimized")
        random_state = np.random.RandomState(seed)
        return random_state.randn(PARAMETER_COUNT), f"random seed {seed}", None

    path = (parameter_file or TUTORIAL_BESTLAST).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Parameter file not found: {path}")
    parameters = np.loadtxt(path).reshape(-1)
    return parameters, str(path), path


def create_environment(
    entry: str,
    parameters: np.ndarray,
    duration: float,
    training_config: dict[str, object],
    config_path: Path,
) -> tuple[object, dict[str, object]]:
    sys.path.insert(0, str(MYOASSIST_SOURCE))

    effective_config = {
        "sim_time": duration,
        "mode": training_config["mode"],
        "init_pose": training_config["init_pose"],
        "slope_deg": training_config["slope_deg"],
        "delayed": training_config["delayed"],
        "model": training_config["model"],
        "exo_bool": False,
        "fixed_exo": False,
        "n_points": 0,
        "use_4param_spline": False,
        "max_torque": 0.0,
    }

    if entry == "official":
        from ctrl_optim.optim.optim_utils.config_parser import create_testenv_from_bat

        environment, loaded_config = create_testenv_from_bat(
            str(config_path),
            parameters,
            **effective_config,
        )
        return environment, dict(loaded_config)

    from ctrl_optim.ctrl.reflex.reflex_interface import myoLeg_reflex

    environment = myoLeg_reflex(control_params=parameters, **effective_config)
    return environment, effective_config


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
            dynamics == "mjDYN_MUSCLE"
            and gain == "mjGAIN_MUSCLE"
            and bias == "mjBIAS_MUSCLE"
            and transmission == "mjTRN_TENDON"
        )
        metadata.append(
            {
                "index": actuator_id,
                "name": model.actuator(actuator_id).name,
                "transmission": transmission,
                "dynamics": dynamics,
                "gain": gain,
                "bias": bias,
                "target": target,
                "is_muscle": is_muscle,
            }
        )
    return metadata


def print_actuators(actuators: list[dict[str, object]]) -> None:
    print("Actuators:")
    for item in actuators:
        kind = "muscle" if item["is_muscle"] else "non-muscle"
        print(
            f"  {item['index']:02d} {item['name']:<16} {kind:<10} "
            f"{item['transmission']} {item['dynamics']} "
            f"{item['gain']} -> {item['target']}"
        )


def termination_reasons(pelvis_height: float, pelvis_pitch_deg: float) -> list[str]:
    reasons: list[str] = []
    if pelvis_height < PELVIS_HEIGHT_THRESHOLD:
        reasons.append("pelvis_height_below_0.65_m")
    if pelvis_pitch_deg < -PELVIS_PITCH_LIMIT_DEG:
        reasons.append("pelvis_pitch_below_minus_60_deg")
    if pelvis_pitch_deg > PELVIS_PITCH_LIMIT_DEG:
        reasons.append("pelvis_pitch_above_plus_60_deg")
    return reasons


def state_row(
    environment: object,
    step_index: int,
    is_done: bool,
    actuators: list[dict[str, object]],
    heel_strike_side: str = "",
) -> dict[str, object]:
    data = environment.env.sim.data
    pelvis_position = np.asarray(data.body("pelvis").xpos).copy()
    pelvis_velocity = np.asarray(
        data.object_velocity("pelvis", "body", local_frame=False)[0]
    ).copy()
    pelvis_euler = environment.get_intrinsic_EulerXYZ(
        np.asarray(data.body("pelvis").xquat).copy()
    )
    torso_euler = environment.get_intrinsic_EulerXYZ(
        np.asarray(data.body("torso").xquat).copy()
    )
    pelvis_pitch_deg = float(np.rad2deg(pelvis_euler[1]))
    reasons = termination_reasons(float(pelvis_position[2]), pelvis_pitch_deg)
    if is_done and not reasons:
        reasons = ["unclassified_official_termination"]

    row: dict[str, object] = {
        "step": step_index,
        "simulation_time": float(data.time),
        "pelvis_height": float(pelvis_position[2]),
        "pelvis_pitch_deg": pelvis_pitch_deg,
        "torso_pitch_deg": float(np.rad2deg(torso_euler[1])),
        "forward_position": float(pelvis_position[0]),
        "forward_velocity": float(pelvis_velocity[0]),
        "is_done": bool(is_done),
        "termination_reason": ";".join(reasons),
        "right_heel_contact": bool(environment.footstep["r_contact"]),
        "left_heel_contact": bool(environment.footstep["l_contact"]),
        "heel_strike_side": heel_strike_side,
    }
    controls = np.asarray(data.ctrl).copy()
    for item in actuators:
        row[f"ctrl_{item['index']:02d}_{item['name']}"] = float(
            controls[int(item["index"])]
        )
    return row


def render_frame(environment: object, width: int = 960, height: int = 544) -> np.ndarray:
    return environment.env.sim.renderer.render_offscreen(
        camera_id=4,
        width=width,
        height=height,
    )


def safe_label(parameter_path: Path | None, mode: str, entry: str) -> str:
    if mode == "minimal":
        return f"minimal_{entry}"
    if parameter_path is None:
        variant = "bestlast"
    elif re.search(r"_BestLast$", parameter_path.stem, re.IGNORECASE):
        variant = "bestlast"
    elif re.search(r"_Best$", parameter_path.stem, re.IGNORECASE):
        variant = "best"
    else:
        variant = re.sub(r"[^A-Za-z0-9_-]+", "_", parameter_path.stem).lower()
    return f"{variant}_{entry}"


def source_commit() -> str:
    completed = subprocess.run(
        ["git", "-C", str(MYOASSIST_SOURCE), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def dependency_versions() -> dict[str, str]:
    packages = ["mujoco", "numpy", "scipy", "gymnasium", "dm-control", "cma"]
    versions = {"python": sys.version.split()[0]}
    for package in packages:
        versions[package] = importlib.metadata.version(package)
    return versions


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def gait_cycle_metrics(rows: list[dict[str, object]], side: str) -> dict[str, object]:
    raw_strike_times = [
        float(row["simulation_time"])
        for row in rows
        if row["heel_strike_side"] in (side, "both")
    ]
    strike_times: list[float] = []
    for strike_time in raw_strike_times:
        if not strike_times or strike_time - strike_times[-1] > 0.3:
            strike_times.append(strike_time)
    cycle_durations = np.diff(strike_times).tolist() if len(strike_times) > 1 else []
    return {
        "raw_heel_strike_count": len(raw_strike_times),
        "heel_strike_count": len(strike_times),
        "heel_strike_times_seconds": strike_times,
        "cycle_count": len(cycle_durations),
        "cycle_durations_seconds": cycle_durations,
        "mean_cycle_seconds": (
            float(np.mean(cycle_durations)) if cycle_durations else None
        ),
        "cycle_std_seconds": (
            float(np.std(cycle_durations)) if cycle_durations else None
        ),
    }


def walking_metrics(
    rows: list[dict[str, object]],
    actuators: list[dict[str, object]],
    target_speed: float,
) -> dict[str, object]:
    sampled_rows = rows[1:] if len(rows) > 1 else rows
    times = np.asarray([row["simulation_time"] for row in sampled_rows], dtype=float)
    velocities = np.asarray([row["forward_velocity"] for row in sampled_rows], dtype=float)
    pitches = np.asarray([row["pelvis_pitch_deg"] for row in sampled_rows], dtype=float)
    half_index = len(times) // 2
    if len(times[half_index:]) >= 2:
        pitch_trend = float(
            np.polyfit(times[half_index:], pitches[half_index:], deg=1)[0]
        )
    else:
        pitch_trend = 0.0

    muscle_stats: dict[str, dict[str, float]] = {}
    for item in actuators:
        if not item["is_muscle"]:
            continue
        key = f"ctrl_{item['index']:02d}_{item['name']}"
        values = np.asarray([row[key] for row in sampled_rows], dtype=float)
        muscle_stats[str(item["name"])] = {
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
        }

    return {
        "target_speed_m_per_s": target_speed,
        "acceptable_speed_range_m_per_s": [
            target_speed - HEALTHY_SPEED_TOLERANCE,
            target_speed + HEALTHY_SPEED_TOLERANCE,
        ],
        "mean_forward_velocity_m_per_s": float(np.mean(velocities)),
        "forward_velocity_std_m_per_s": float(np.std(velocities)),
        "max_abs_pelvis_pitch_deg": float(np.max(np.abs(pitches))),
        "second_half_pelvis_pitch_trend_deg_per_s": pitch_trend,
        "left_gait_cycles": gait_cycle_metrics(rows, "left"),
        "right_gait_cycles": gait_cycle_metrics(rows, "right"),
        "muscle_control_stats": muscle_stats,
    }


def run(
    mode: str,
    duration: float,
    seed: int,
    output_dir: Path,
    parameter_file: Path | None = None,
    entry: str = "wrapper",
    make_video: bool = True,
    healthy_validation: bool = False,
) -> dict[str, object]:
    if healthy_validation:
        if mode != "optimized" or parameter_file is None:
            raise ValueError(
                "--healthy-validation requires --mode optimized and --params"
            )
        duration = HEALTHY_DURATION_SECONDS
        config_path = HEALTHY_CONFIG
        model_xml = BASELINE_XML
        if output_dir == DEFAULT_OUTPUT_DIR:
            output_dir = HEALTHY_OUTPUT_DIR
    else:
        config_path = TUTORIAL_CONFIG
        model_xml = TUTORIAL_XML

    require_source(config_path, model_xml)
    if duration <= 0:
        raise ValueError("Duration must be greater than zero")
    if mode == "minimal" and entry == "official":
        raise ValueError("The official config-loader comparison requires optimized mode")

    training_config = load_training_config(config_path)
    inventory = tutorial_inventory() if config_path == TUTORIAL_CONFIG else []
    parameters, parameter_source, resolved_parameter_path = load_parameters(
        mode, seed, parameter_file
    )
    if len(parameters) != PARAMETER_COUNT:
        raise ValueError(
            f"Expected {PARAMETER_COUNT} human-only parameters, found {len(parameters)}. "
            "Exoskeleton parameter vectors are intentionally not accepted."
        )

    environment, effective_config = create_environment(
        entry, parameters, duration, training_config, config_path
    )
    environment.reset()
    model = environment.env.sim.model
    actuators = actuator_metadata(model)
    print_actuators(actuators)

    muscle_indices = [int(item["index"]) for item in actuators if item["is_muscle"]]
    non_muscle_indices = [
        int(item["index"]) for item in actuators if not item["is_muscle"]
    ]
    if len(muscle_indices) != 22:
        raise ValueError(f"Expected 22 verified muscle actuators, found {len(muscle_indices)}")

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if healthy_validation:
        json_path = output_dir / "validation.json"
        csv_path = output_dir / "timeseries.csv"
        video_path = output_dir / "healthy_walk.mp4"
    else:
        label = safe_label(resolved_parameter_path, mode, entry)
        json_path = output_dir / f"{label}.json"
        csv_path = output_dir / f"{label}.csv"
        video_path = output_dir / f"{label}.mp4"

    frames: list[np.ndarray] = []
    capture_interval = max(1, round(1.0 / (50.0 * environment.dt)))
    if mode == "optimized" and make_video:
        frames.append(render_frame(environment))

    time_series = [state_row(environment, 0, False, actuators)]
    active_muscles: set[int] = set()
    max_muscle_activation = 0.0
    max_non_muscle_control = 0.0
    terminated_early = False

    total_steps = int(duration / environment.dt)
    for step_index in range(1, total_steps + 1):
        previous_right_contact = bool(environment.footstep["r_contact"])
        previous_left_contact = bool(environment.footstep["l_contact"])
        _, _, is_done = environment.run_reflex_step_Cost()
        data = environment.env.sim.data
        controls = np.asarray(data.ctrl)
        state = np.concatenate(
            (
                np.asarray(data.qpos),
                np.asarray(data.qvel),
                np.asarray(data.qacc),
                controls,
            )
        )
        if not np.isfinite(state).all():
            raise FloatingPointError(f"NaN or Inf detected at step {step_index}")

        muscle_controls = controls[muscle_indices]
        active_local = np.flatnonzero(np.abs(muscle_controls) > 1e-12)
        active_muscles.update(muscle_indices[index] for index in active_local)
        max_muscle_activation = max(
            max_muscle_activation,
            float(np.max(np.abs(muscle_controls))),
        )
        if non_muscle_indices:
            max_non_muscle_control = max(
                max_non_muscle_control,
                float(np.max(np.abs(controls[non_muscle_indices]))),
            )

        strike_sides = []
        if not previous_right_contact and environment.footstep["r_contact"]:
            strike_sides.append("right")
        if not previous_left_contact and environment.footstep["l_contact"]:
            strike_sides.append("left")
        heel_strike_side = "both" if len(strike_sides) == 2 else "".join(strike_sides)
        row = state_row(
            environment,
            step_index,
            bool(is_done),
            actuators,
            heel_strike_side=heel_strike_side,
        )
        time_series.append(row)
        if mode == "optimized" and make_video and step_index % capture_interval == 0:
            frames.append(render_frame(environment))
        if is_done:
            terminated_early = True
            break

    if mode == "optimized" and make_video and frames:
        import imageio.v2 as imageio

        with imageio.get_writer(
            video_path,
            fps=50,
            codec="libx264",
            quality=8,
        ) as writer:
            for frame in frames:
                writer.append_data(frame)

    final_state = time_series[-1]
    termination_reason = (
        final_state["termination_reason"] if terminated_early else "duration_limit"
    )
    raw_model = model.ptr
    target_speed = float(training_config.get("target_speed") or 1.25)
    metrics = walking_metrics(time_series, actuators, target_speed)
    duration_reached = (
        float(final_state["simulation_time"]) >= duration - float(environment.dt) / 2
    )
    speed_range = metrics["acceptable_speed_range_m_per_s"]
    speed_passed = (
        float(speed_range[0])
        <= float(metrics["mean_forward_velocity_m_per_s"])
        <= float(speed_range[1])
    )
    pitch_trend_passed = (
        float(metrics["second_half_pelvis_pitch_trend_deg_per_s"])
        <= HEALTHY_MAX_PITCH_TREND_DEG_PER_SECOND
    )
    acceptance = {
        "passed": bool(
            healthy_validation
            and duration_reached
            and not terminated_early
            and speed_passed
            and pitch_trend_passed
        ),
        "duration_reached": duration_reached,
        "finite_state": True,
        "no_official_fall": not terminated_early,
        "mean_speed_in_target_range": speed_passed,
        "pelvis_pitch_not_sustained_increasing": pitch_trend_passed,
        "max_allowed_pitch_trend_deg_per_s": (
            HEALTHY_MAX_PITCH_TREND_DEG_PER_SECOND
        ),
    }
    result: dict[str, object] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "entry": entry,
        "mode": mode,
        "model": effective_config["model"],
        "model_xml": str(model_xml),
        "parameter_source": parameter_source,
        "parameter_count": len(parameters),
        "source_commit": source_commit(),
        "dependencies": dependency_versions(),
        "training_config": training_config,
        "tutorial_prep_inventory": inventory,
        "paired_config_expected_parameter_count": expected_parameter_count(
            training_config
        ),
        "paired_config_matches_parameter_count": (
            expected_parameter_count(training_config) == len(parameters)
        ),
        "effective_runtime_config": effective_config,
        "requested_duration_seconds": duration,
        "walking_duration_seconds": float(final_state["simulation_time"]),
        "terminated_early": terminated_early,
        "termination_reason": termination_reason,
        "termination_conditions": {
            "pelvis_height": f"< {PELVIS_HEIGHT_THRESHOLD} m",
            "pelvis_pitch": (
                f"< -{PELVIS_PITCH_LIMIT_DEG} deg or > +{PELVIS_PITCH_LIMIT_DEG} deg"
            ),
        },
        "termination_state": {
            "pelvis_height": final_state["pelvis_height"],
            "pelvis_pitch_deg": final_state["pelvis_pitch_deg"],
            "torso_pitch_deg": final_state["torso_pitch_deg"],
            "forward_position": final_state["forward_position"],
            "forward_velocity": final_state["forward_velocity"],
            "simulation_time": final_state["simulation_time"],
        },
        "finite_state": True,
        "controller_timestep_seconds": float(environment.dt),
        "mujoco_timestep_seconds": float(raw_model.opt.timestep),
        "frame_skip": int(environment.frame_skip),
        "model_nu": int(model.nu),
        "actuators": actuators,
        "verified_muscle_actuator_count": len(muscle_indices),
        "non_muscle_actuator_count": len(non_muscle_indices),
        "active_muscle_channels_observed": len(active_muscles),
        "max_muscle_activation": max_muscle_activation,
        "max_non_muscle_control": max_non_muscle_control,
        "exo_assist_enabled": False,
        "walking_metrics": metrics,
        "healthy_acceptance": acceptance if healthy_validation else None,
        "video": str(video_path) if mode == "optimized" and make_video else None,
        "csv": str(csv_path),
        "time_series": time_series,
    }

    write_csv(csv_path, time_series)
    if healthy_validation and acceptance["passed"]:
        controller_path = output_dir / "healthy_controller.txt"
        config_copy_path = output_dir / "optimization_config.sh"
        versions_path = output_dir / "environment_versions.json"
        shutil.copy2(resolved_parameter_path, controller_path)
        shutil.copy2(config_path, config_copy_path)
        versions = {
            "myoassist_commit": result["source_commit"],
            "dependencies": result["dependencies"],
            "parameter_dimension": len(parameters),
            "model_xml": str(model_xml),
            "optimization_config": str(config_copy_path),
        }
        versions_path.write_text(
            json.dumps(versions, indent=2) + "\n", encoding="utf-8"
        )
        result["healthy_artifacts"] = {
            "controller": str(controller_path),
            "optimization_config": str(config_copy_path),
            "environment_versions": str(versions_path),
            "validation": str(json_path),
            "timeseries": str(csv_path),
            "video": str(video_path) if make_video else None,
        }
    else:
        result["healthy_artifacts"] = None
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["log"] = str(json_path)
    return result


def printable_summary(result: dict[str, object]) -> dict[str, object]:
    summary = dict(result)
    summary.pop("time_series", None)
    return summary


def main() -> int:
    args = parse_args()
    try:
        result = run(
            args.mode,
            args.duration,
            args.seed,
            args.output_dir,
            parameter_file=args.params,
            entry=args.entry,
            make_video=not args.no_video,
            healthy_validation=args.healthy_validation,
        )
        print(json.dumps(printable_summary(result), indent=2))
        if result["terminated_early"]:
            print(
                "The official fall condition ended the simulation: "
                f"{result['termination_reason']}"
            )
        if args.healthy_validation:
            print(
                "Healthy acceptance: "
                f"{'PASSED' if result['healthy_acceptance']['passed'] else 'FAILED'}"
            )
        return 0
    except Exception as exc:
        print(f"ERROR: MyoAssist walk failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
