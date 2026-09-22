"""M3 adapter for the standalone fixed-root exoskeleton test bench."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JOINTS = ("HipExo_R", "HipExo_L", "KneeExo_R", "KneeExo_L")
INITIAL_POSITION_RAD = {
    "HipExo_R": 0.30,
    "HipExo_L": -0.25,
    "KneeExo_R": -0.45,
    "KneeExo_L": -0.60,
}
VIDEO_WIDTH = 960
VIDEO_HEIGHT = 544


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


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
        json.dumps(
            _json_safe(value), indent=2, ensure_ascii=False, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )


def _git_value(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip() or "unavailable"


def _joint_metadata(model: mujoco.MjModel) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in JOINTS:
        joint_id = int(model.joint(name).id)
        actuator_id = int(model.actuator(name).id)
        target_joint_id = int(model.actuator_trnid[actuator_id, 0])
        result[name] = {
            "joint_id": joint_id,
            "qpos_address": int(model.jnt_qposadr[joint_id]),
            "dof_address": int(model.jnt_dofadr[joint_id]),
            "range_rad": np.asarray(model.jnt_range[joint_id], dtype=float),
            "actuator_id": actuator_id,
            "target_joint_id": target_joint_id,
            "target_joint_name": model.joint(target_joint_id).name,
            "gear_sign": float(model.actuator_gear[actuator_id, 0]),
        }
    return result


def _record_row(
    step: int,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    row: dict[str, Any] = {"step": step, "time_s": float(data.time)}
    for name, item in metadata.items():
        qpos = float(data.qpos[item["qpos_address"]])
        qvel = float(data.qvel[item["dof_address"]])
        command = float(data.ctrl[item["actuator_id"]])
        actual = float(
            data.actuator_force[item["actuator_id"]] * item["gear_sign"]
        )
        lower, upper = item["range_rad"]
        margin = min(qpos - float(lower), float(upper) - qpos)
        row[f"joint_position_{name}_rad"] = qpos
        row[f"joint_velocity_{name}_rad_s"] = qvel
        row[f"command_torque_{name}_Nm"] = command
        row[f"actual_torque_{name}_Nm"] = actual
        row[f"power_{name}_W"] = actual * qvel
        row[f"joint_limit_margin_{name}_rad"] = margin
        row[f"joint_limit_ok_{name}"] = margin >= -1e-8
    state = np.concatenate(
        (
            np.asarray(data.qpos),
            np.asarray(data.qvel),
            np.asarray(data.qacc),
            np.asarray(data.ctrl),
            np.asarray(data.actuator_force),
        )
    )
    row["finite_state"] = bool(np.isfinite(state).all())
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(config: Any, plan: dict[str, Any]) -> dict[str, Any]:
    """Run one fixed-root, zero-torque standalone Exo experiment."""
    model_xml = Path(config.MODEL_XML).expanduser().resolve()
    if not model_xml.is_file():
        raise FileNotFoundError(f"Standalone Exo XML not found: {model_xml}")
    if str(config.TEST_MODE) != "passive_zero_torque":
        raise ValueError(
            "Phase 1 L2 supports only TEST_MODE='passive_zero_torque'"
        )
    duration_s = float(config.DURATION_SECONDS)
    if duration_s <= 0.0:
        raise ValueError("DURATION_SECONDS must be positive")

    model = mujoco.MjModel.from_xml_path(str(model_xml))
    data = mujoco.MjData(model)
    metadata = _joint_metadata(model)
    pelvis_body_id = int(model.body("exo_pelvis_frame").id)
    root_fixed = int(model.body_jntnum[pelvis_body_id]) == 0
    mappings_valid = all(
        item["joint_id"] == item["target_joint_id"]
        and item["target_joint_name"] == name
        for name, item in metadata.items()
    )
    positive_torque_sign = all(
        item["gear_sign"] == 1.0 for item in metadata.values()
    )
    dimensions_valid = bool(model.nq == model.nv == model.nu == 4)
    body_mass = np.asarray(model.body_mass[1:], dtype=float)
    body_inertia = np.asarray(model.body_inertia[1:], dtype=float)
    mass_inertia_finite = bool(
        np.isfinite(body_mass).all()
        and np.isfinite(body_inertia).all()
        and np.all(body_mass > 0.0)
        and np.all(body_inertia > 0.0)
    )
    total_mass_kg = float(np.sum(body_mass))

    for name, position in INITIAL_POSITION_RAD.items():
        data.qpos[metadata[name]["qpos_address"]] = position
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    output_directory = Path(plan["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=False)
    manifest_path = output_directory / "manifest.json"
    timeseries_path = output_directory / "timeseries.csv"
    validation_path = output_directory / "validation.json"
    video_path = output_directory / "video.mp4"

    mapping_manifest = {
        name: {
            "joint_id": item["joint_id"],
            "actuator_id": item["actuator_id"],
            "target_joint": item["target_joint_name"],
            "gear_sign": item["gear_sign"],
            "joint_range_rad": item["range_rad"],
        }
        for name, item in metadata.items()
    }
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": plan["run_id"],
        "timestamp": plan["timestamp"].isoformat(timespec="seconds"),
        "experiment_name": str(config.EXPERIMENT_NAME),
        "experiment_type": "exo_only",
        "test_mode": str(config.TEST_MODE),
        "control_mode": "passive_zero_torque",
        "standalone_model": str(config.EXO_MODEL),
        "standalone_model_version": str(config.EXO_MODEL_VERSION),
        "model_xml_path": _relative(model_xml),
        "model_xml_sha256": _sha256(model_xml),
        "fixture_mode": "fixed_pelvis_root",
        "fixed_root_body": "exo_pelvis_frame",
        "test_bench_scope": {
            "type": "standalone_exoskeleton_mechanical_test_bench",
            "purpose": "isolate joint, actuator, mass/inertia, state, torque and power validation",
            "human_worn_exoskeleton": False,
            "hardware_test": False,
            "walking_experiment": False,
        },
        "free_dofs": list(JOINTS),
        "actuators": list(JOINTS),
        "actuator_mapping": mapping_manifest,
        "initial_joint_position_rad": dict(INITIAL_POSITION_RAD),
        "command_policy": "zero_torque",
        "requested_duration_s": duration_s,
        "actual_duration_s": 0.0,
        "mujoco_timestep_s": float(model.opt.timestep),
        "total_exoskeleton_mass_kg": total_mass_kg,
        "model_dimensions": {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "nbody": int(model.nbody),
        },
        "save_video": bool(config.SAVE_VIDEO),
        "python_version": sys.version.split()[0],
        "mujoco_version": mujoco.__version__,
        "git_branch": _git_value("branch", "--show-current"),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "success": False,
    }
    _write_json(manifest_path, manifest)

    rows = [_record_row(0, model, data, metadata)]
    renderer = None
    video_writer = None
    frame_count = 0
    capture_interval = max(
        1, round(1.0 / (float(config.VIDEO_FPS) * float(model.opt.timestep)))
    )
    if bool(config.SAVE_VIDEO):
        renderer = mujoco.Renderer(
            model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH
        )
        camera = mujoco.MjvCamera()
        camera.lookat[:] = (0.0, 0.0, 0.42)
        camera.distance = 2.0
        camera.azimuth = 90.0
        camera.elevation = -12.0
        video_writer = imageio.get_writer(
            video_path,
            fps=int(config.VIDEO_FPS),
            codec="libx264",
            quality=8,
        )
        renderer.update_scene(data, camera=camera)
        video_writer.append_data(renderer.render())
        frame_count += 1

    total_steps = int(round(duration_s / float(model.opt.timestep)))
    termination_reason = "time_limit"
    try:
        for step in range(1, total_steps + 1):
            data.ctrl[:] = 0.0
            mujoco.mj_step(model, data)
            row = _record_row(step, model, data, metadata)
            rows.append(row)
            if not bool(row["finite_state"]):
                termination_reason = "non_finite_state"
                break
            if video_writer is not None and step % capture_interval == 0:
                renderer.update_scene(data, camera=camera)
                video_writer.append_data(renderer.render())
                frame_count += 1
    finally:
        if video_writer is not None:
            video_writer.close()
        if renderer is not None:
            renderer.close()

    _write_csv(timeseries_path, rows)
    actual_duration_s = float(data.time)
    finite_state = bool(all(bool(row["finite_state"]) for row in rows))
    all_joint_limits_respected = bool(
        all(
            bool(row[f"joint_limit_ok_{name}"])
            for row in rows
            for name in JOINTS
        )
    )
    command_peak_nm = {
        name: max(abs(float(row[f"command_torque_{name}_Nm"])) for row in rows)
        for name in JOINTS
    }
    actual_peak_nm = {
        name: max(abs(float(row[f"actual_torque_{name}_Nm"])) for row in rows)
        for name in JOINTS
    }
    power_peak_w = {
        name: max(abs(float(row[f"power_{name}_W"])) for row in rows)
        for name in JOINTS
    }
    zero_torque_verified = bool(
        max((*command_peak_nm.values(), *actual_peak_nm.values())) <= 1e-12
    )
    video_valid = bool(
        not config.SAVE_VIDEO
        or (video_path.is_file() and video_path.stat().st_size > 0 and frame_count > 1)
    )
    success = bool(
        termination_reason == "time_limit"
        and actual_duration_s >= duration_s - float(model.opt.timestep)
        and finite_state
        and root_fixed
        and dimensions_valid
        and mappings_valid
        and positive_torque_sign
        and mass_inertia_finite
        and math.isclose(total_mass_kg, 8.0, abs_tol=1e-9)
        and all_joint_limits_respected
        and zero_torque_verified
        and video_valid
    )
    manifest.update(
        {
            "actual_duration_s": actual_duration_s,
            "termination_reason": termination_reason,
            "success": success,
        }
    )
    _write_json(manifest_path, manifest)

    validation = {
        **manifest,
        "xml_compile": True,
        "root_fixed": root_fixed,
        "four_free_dofs": dimensions_valid,
        "mass_inertia_finite_positive": mass_inertia_finite,
        "mass_is_8kg": math.isclose(total_mass_kg, 8.0, abs_tol=1e-9),
        "joint_mapping_valid": mappings_valid,
        "actuator_mapping_valid": mappings_valid,
        "positive_torque_sign_convention": positive_torque_sign,
        "finite_state": finite_state,
        "joint_limits_respected": all_joint_limits_respected,
        "zero_torque_verified": zero_torque_verified,
        "peak_abs_command_torque_Nm": command_peak_nm,
        "peak_abs_actual_torque_Nm": actual_peak_nm,
        "peak_abs_power_W": power_peak_w,
        "video_frame_count": frame_count,
        "video_valid": video_valid,
        "artifacts": {
            "manifest": _relative(manifest_path),
            "timeseries": _relative(timeseries_path),
            "validation": _relative(validation_path),
            "video": _relative(video_path) if bool(config.SAVE_VIDEO) else None,
        },
    }
    _write_json(validation_path, validation)
    latest_path = Path(config.OUTPUT_ROOT).expanduser().resolve() / "latest_run.json"
    _write_json(
        latest_path,
        {
            "run_id": plan["run_id"],
            "run_directory": _relative(output_directory),
        },
    )
    if not success:
        raise RuntimeError(
            f"Standalone Exo validation failed; inspect {validation_path}"
        )
    return validation
