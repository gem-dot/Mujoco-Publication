"""Unified experiment loop for the project's verified MyoAssist experiments."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np
import yaml

from simulation import model_setup
from simulation.model_setup import EXO_CHANNELS, PreparedModel
from simulation.recorder import ExperimentRecorder, TIMESERIES_SCHEMA_VERSION
from visualization import CouplingVisualizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PELVIS_HEIGHT_THRESHOLD_M = 0.65
PELVIS_PITCH_LIMIT_DEG = 60.0
CONTACT_THRESHOLD_BW = 0.10
CONTACT_GRAVITY_MPS2 = 9.8  # Regression-compatible MyoAssist heel-contact threshold.
HEEL_STRIKE_REFRACTORY_SECONDS = 0.6
HEEL_SITES = {"right": "r_heel_btm", "left": "l_heel_btm"}
VIDEO_WIDTH = 960
VIDEO_HEIGHT = 544
CAMERA_ID = 4


def _safe_experiment_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
    if not result:
        raise ValueError("EXPERIMENT_NAME must contain at least one safe character")
    return result


def plan_experiment(config: ModuleType | Any) -> dict[str, Any]:
    """Choose a unique run directory and read display-only patient metadata."""
    patient_path = Path(config.PATIENT_CONFIG_PATH).expanduser().resolve()
    patient = yaml.safe_load(patient_path.read_text(encoding="utf-8"))
    if not isinstance(patient, dict):
        raise ValueError(f"Patient config is not a mapping: {patient_path}")
    timestamp = datetime.now()
    stem = f"{timestamp:%Y%m%d_%H%M%S}__{_safe_experiment_name(config.EXPERIMENT_NAME)}"
    output_root = Path(config.OUTPUT_ROOT).expanduser().resolve()
    output_directory = output_root / stem
    suffix = 1
    while output_directory.exists():
        output_directory = output_root / f"{stem}__{suffix:02d}"
        suffix += 1
    return {
        "timestamp": timestamp,
        "run_id": output_directory.name,
        "output_directory": output_directory,
        "patient": patient,
    }


def print_experiment_summary(config: ModuleType | Any, plan: dict[str, Any]) -> None:
    patient = plan["patient"]
    assist = config.ASSIST_CONTROLLER
    assist_name = "none (all exoskeleton torques zero)" if assist is None else assist.__name__
    affected = patient.get("affected_muscles", [patient.get("affected_muscle")])
    damage = (
        f"{','.join(str(name) for name in affected)} retained at "
        f"{100.0 * float(patient['retained_force_fraction']):.0f}%"
    )
    print("=" * 40)
    print("EXPERIMENT")
    print("=" * 40)
    print(f"Experiment:       {config.EXPERIMENT_NAME}")
    print(f"Patient:          {patient.get('severity', patient.get('patient_type'))}")
    print(f"Damage:           {damage}")
    print(f"XML:              {Path(config.MODEL_XML).resolve()}")
    print(f"Exoskeleton:      {config.EXO_MODEL if config.EXO_ENABLED else 'disabled'}")
    print(f"Exo mass:         {float(config.EXO_MASS_KG):g} kg")
    print(f"Human controller: {Path(config.HUMAN_CONTROLLER_PATH).resolve()}")
    print(f"Assist controller: {assist_name}")
    print(f"Assist parameters: {dict(config.ASSIST_PARAMS)}")
    print(f"Duration:         {float(config.DURATION_SECONDS):g} s")
    print(f"Target velocity:  {float(config.TARGET_VELOCITY_MPS):g} m/s")
    print(f"Save video:       {bool(config.SAVE_VIDEO)}")
    print(f"Output:           {plan['output_directory']}")
    print("=" * 40)


def _touch_forces(prepared: PreparedModel) -> dict[str, float]:
    result: dict[str, float] = {}
    for logical_name, sensor_id in prepared.sensor_ids.items():
        address = int(prepared.model.ptr.sensor_adr[sensor_id])
        result[logical_name] = float(prepared.data.sensordata[address])
    return result


def _muscle_activation(prepared: PreparedModel, actuator_id: int) -> float:
    address = int(prepared.model.ptr.actuator_actadr[actuator_id])
    return float(prepared.data.act[address]) if address >= 0 else math.nan


def build_observation(
    prepared: PreparedModel,
    muscle_ctrl_override: np.ndarray | None = None,
    *,
    include_full_state_snapshot: bool = False,
) -> dict[str, Any]:
    """Read the runtime observation without stepping or mutating the model."""
    environment = prepared.environment
    data = prepared.data
    pelvis = np.asarray(data.body("pelvis").xpos).copy()
    velocity = np.asarray(
        data.object_velocity("pelvis", "body", local_frame=False)[0]
    ).copy()
    touch = _touch_forces(prepared)
    threshold_n = (
        CONTACT_THRESHOLD_BW
        * float(prepared.metadata["total_model_mass_kg"])
        * CONTACT_GRAVITY_MPS2
    )
    controls = (
        np.asarray(muscle_ctrl_override)
        if muscle_ctrl_override is not None
        else np.asarray(data.ctrl)
    )
    joints = {
        "hip_r": {
            "angle": float(data.joint("hip_flexion_r").qpos[0]),
            "velocity": float(data.joint("hip_flexion_r").qvel[0]),
        },
        "hip_l": {
            "angle": float(data.joint("hip_flexion_l").qpos[0]),
            "velocity": float(data.joint("hip_flexion_l").qvel[0]),
        },
        "knee_r": {
            "angle": float(data.joint("knee_angle_r").qpos[0]),
            "velocity": float(data.joint("knee_angle_r").qvel[0]),
        },
        "knee_l": {
            "angle": float(data.joint("knee_angle_l").qpos[0]),
            "velocity": float(data.joint("knee_angle_l").qvel[0]),
        },
        "ankle_r": {
            "angle": float(data.joint("ankle_angle_r").qpos[0]),
            "velocity": float(data.joint("ankle_angle_r").qvel[0]),
        },
        "ankle_l": {
            "angle": float(data.joint("ankle_angle_l").qpos[0]),
            "velocity": float(data.joint("ankle_angle_l").qvel[0]),
        },
    }
    exoskeleton: dict[str, Any] | None = None
    required_exo_joints = (
        "exo_root_forward",
        "exo_root_vertical",
        "exo_root_pitch",
        "exo_hip_l",
        "exo_knee_l",
        "exo_hip_r",
        "exo_knee_r",
    )
    if all(
        mujoco.mj_name2id(
            prepared.model.ptr, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        >= 0
        for name in required_exo_joints
    ):
        exo_joint_ids = {
            name: int(prepared.model.joint(name).id) for name in required_exo_joints
        }
        exo_qpos_addresses = np.asarray(
            [prepared.model.ptr.jnt_qposadr[exo_joint_ids[name]] for name in required_exo_joints],
            dtype=int,
        )
        exo_dof_addresses = np.asarray(
            [prepared.model.ptr.jnt_dofadr[exo_joint_ids[name]] for name in required_exo_joints],
            dtype=int,
        )
        hip_dof = int(exo_dof_addresses[3])
        knee_dof = int(exo_dof_addresses[4])
        coupling_qfrc = (
            np.zeros(int(prepared.model.nv), dtype=float)
            if prepared.coupling_runtime is None
            else np.asarray(prepared.coupling_runtime.coupling_qfrc)
        )
        base_external_qfrc = (
            np.asarray(data.qfrc_applied, dtype=float)
            if prepared.coupling_runtime is None
            else np.asarray(prepared.coupling_runtime.base_external_qfrc, dtype=float)
        )
        passive_exo_qfrc = np.asarray(
            data.qfrc_passive[exo_dof_addresses], dtype=float
        ).copy()
        coupling_exo_qfrc = np.asarray(
            coupling_qfrc[exo_dof_addresses], dtype=float
        ).copy()
        constraint_jacobian = np.zeros((0, 7), dtype=float)
        constraint_regularization = np.zeros(0, dtype=float)
        constraint_reference_acceleration = np.zeros(0, dtype=float)
        if int(data.nefc) > 0:
            if mujoco.mj_isSparse(prepared.model.ptr):
                full_jacobian = np.zeros(
                    (int(data.nefc), int(prepared.model.nv)), dtype=float
                )
                values = np.asarray(data.efc_J, dtype=float)
                for row in range(int(data.nefc)):
                    start = int(data.efc_J_rowadr[row])
                    count = int(data.efc_J_rownnz[row])
                    columns = np.asarray(
                        data.efc_J_colind[start : start + count], dtype=int
                    )
                    full_jacobian[row, columns] = values[start : start + count]
            else:
                full_jacobian = np.asarray(data.efc_J, dtype=float).reshape(
                    int(data.nefc), int(prepared.model.nv)
                )
            mask = np.max(np.abs(full_jacobian[:, exo_dof_addresses]), axis=1) > 1.0e-12
            selected = np.flatnonzero(mask)
            if selected.size:
                outside = np.setdiff1d(
                    np.arange(int(prepared.model.nv)), exo_dof_addresses
                )
                if np.max(np.abs(full_jacobian[np.ix_(selected, outside)])) > 1.0e-12:
                    raise RuntimeError(
                        "Bilateral YAC encountered a constraint row coupled outside the Exo tree"
                    )
                constraint_jacobian = full_jacobian[np.ix_(selected, exo_dof_addresses)]
                constraint_regularization = np.asarray(data.efc_R[selected], dtype=float)
                constraint_reference_acceleration = np.asarray(
                    data.efc_aref[selected], dtype=float
                )
        exoskeleton = {
            "root_pitch_rad": float(data.joint("exo_root_pitch").qpos[0]),
            "hip_l": {
                "angle_rad": float(data.joint("exo_hip_l").qpos[0]),
                "velocity_rad_s": float(data.joint("exo_hip_l").qvel[0]),
                "acceleration_rad_s2": float(data.qacc[hip_dof]),
            },
            "knee_l": {
                "angle_rad": float(data.joint("exo_knee_l").qpos[0]),
                "velocity_rad_s": float(data.joint("exo_knee_l").qvel[0]),
                "acceleration_rad_s2": float(data.qacc[knee_dof]),
            },
            "hip_r": {
                "angle_rad": float(data.joint("exo_hip_r").qpos[0]),
                "velocity_rad_s": float(data.joint("exo_hip_r").qvel[0]),
                "acceleration_rad_s2": float(data.qacc[exo_dof_addresses[5]]),
            },
            "knee_r": {
                "angle_rad": float(data.joint("exo_knee_r").qpos[0]),
                "velocity_rad_s": float(data.joint("exo_knee_r").qvel[0]),
                "acceleration_rad_s2": float(data.qacc[exo_dof_addresses[6]]),
            },
            "coupling_qfrc_nm": {
                "hip_l": float(coupling_qfrc[hip_dof]),
                "knee_l": float(coupling_qfrc[knee_dof]),
                "hip_r": float(coupling_qfrc[exo_dof_addresses[5]]),
                "knee_r": float(coupling_qfrc[exo_dof_addresses[6]]),
            },
            "constraint_qfrc_nm": {
                "hip_l": float(data.qfrc_constraint[hip_dof]),
                "knee_l": float(data.qfrc_constraint[knee_dof]),
                "hip_r": float(data.qfrc_constraint[exo_dof_addresses[5]]),
                "knee_r": float(data.qfrc_constraint[exo_dof_addresses[6]]),
            },
            "bilateral_dynamics": {
                "q_exo": np.asarray(data.qpos[exo_qpos_addresses], dtype=float).copy(),
                "qvel_exo": np.asarray(data.qvel[exo_dof_addresses], dtype=float).copy(),
                "external_generalized_force": coupling_exo_qfrc + passive_exo_qfrc,
                "coupling_generalized_force": coupling_exo_qfrc,
                "passive_generalized_force": passive_exo_qfrc,
                "base_applied_generalized_force": np.asarray(
                    base_external_qfrc[exo_dof_addresses], dtype=float
                ).copy(),
                "constraint_jacobian": constraint_jacobian,
                "constraint_regularization": constraint_regularization,
                "constraint_reference_acceleration": constraint_reference_acceleration,
            },
        }
        if include_full_state_snapshot:
            exoskeleton["runtime_snapshot"] = {
                "time": float(data.time),
                "qpos": np.asarray(data.qpos, dtype=float).copy(),
                "qvel": np.asarray(data.qvel, dtype=float).copy(),
                "act": np.asarray(data.act, dtype=float).copy(),
                "ctrl": np.asarray(controls, dtype=float).copy(),
                "qacc_warmstart": np.asarray(
                    data.qacc_warmstart, dtype=float
                ).copy(),
                "qfrc_applied": (
                    np.asarray(base_external_qfrc, dtype=float)
                    + np.asarray(coupling_qfrc, dtype=float)
                ).copy(),
                "xfrc_applied": np.asarray(
                    data.xfrc_applied, dtype=float
                ).copy(),
                "mocap_pos": np.asarray(data.mocap_pos, dtype=float).copy(),
                "mocap_quat": np.asarray(data.mocap_quat, dtype=float).copy(),
                "userdata": np.asarray(data.userdata, dtype=float).copy(),
            }
    return {
        "time": float(data.time),
        "pelvis": {
            "height": float(pelvis[2]),
            "pitch": float(model_setup.healthy_utils.pelvis_pitch_deg(environment)),
            "forward_position": float(pelvis[0]),
            "forward_velocity": float(velocity[0]),
        },
        "joints": joints,
        "contact": {
            "left_heel": bool(touch["left_heel"] > threshold_n),
            "left_toe": bool(touch["left_toe"] > threshold_n),
            "right_heel": bool(touch["right_heel"] > threshold_n),
            "right_toe": bool(touch["right_toe"] > threshold_n),
            # Regression requirement: these are the legacy heel-only footstep flags.
            "left_stance": bool(environment.footstep["l_contact"]),
            "right_stance": bool(environment.footstep["r_contact"]),
        },
        "touch_force_n": touch,
        "muscles": {
            "ctrl": {
                name: float(controls[actuator_id])
                for name, actuator_id in prepared.muscle_ids.items()
            },
            "activation": {
                name: _muscle_activation(prepared, actuator_id)
                for name, actuator_id in prepared.muscle_ids.items()
            },
            "force": {
                name: abs(float(data.actuator_force[actuator_id]))
                for name, actuator_id in prepared.muscle_ids.items()
            },
        },
        # Controller-only runtime context. It is intentionally not copied into
        # the public timeseries schema; algorithm diagnostics use a separate CSV.
        "exoskeleton": exoskeleton,
    }


def _exo_measurements(
    prepared: PreparedModel, observation: dict[str, Any]
) -> tuple[dict[str, float], dict[str, float]]:
    del observation
    actual: dict[str, float] = {}
    velocity: dict[str, float] = {}
    for channel in EXO_CHANNELS:
        if channel not in prepared.exo_ids:
            actual[channel] = math.nan
            velocity[channel] = math.nan
            continue
        actual[channel] = float(prepared.data.actuator_force[prepared.exo_ids[channel]])
        velocity[channel] = float(
            prepared.data.joint(prepared.exo_joint_names[channel]).qvel[0]
        )
    return actual, velocity


def _termination_reason(observation: dict[str, Any], finite_state: bool) -> str:
    if not finite_state:
        return "non_finite_state"
    pelvis = observation["pelvis"]
    if float(pelvis["height"]) < PELVIS_HEIGHT_THRESHOLD_M:
        return "pelvis_height_below_0.65_m"
    pitch = float(pelvis["pitch"])
    if pitch < -PELVIS_PITCH_LIMIT_DEG:
        return "pelvis_pitch_below_minus_60_deg"
    if pitch > PELVIS_PITCH_LIMIT_DEG:
        return "pelvis_pitch_above_plus_60_deg"
    return ""


def _git_value(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.strip() or "unavailable"


def _relative(path: Path) -> str:
    return model_setup.project_relative(path)


def _json_write(path: Path, value: dict[str, Any]) -> None:
    def ready(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: ready(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [ready(child) for child in item]
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        return item

    path.write_text(
        json.dumps(ready(value), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _manifest(
    config: ModuleType | Any,
    plan: dict[str, Any],
    prepared: PreparedModel,
) -> dict[str, Any]:
    patient = prepared.patient_config
    metadata = prepared.metadata
    assist = config.ASSIST_CONTROLLER
    native_actuator_limits: dict[str, Any] = {}
    for channel, actuator_id in prepared.exo_ids.items():
        limited = bool(prepared.model.ptr.actuator_ctrllimited[actuator_id])
        control_range = np.asarray(
            prepared.model.ptr.actuator_ctrlrange[actuator_id], dtype=float
        )
        native_actuator_limits[channel] = {
            "control_limited": limited,
            "control_range_nm": control_range.tolist() if limited else None,
            "gear": np.asarray(
                prepared.model.ptr.actuator_gear[actuator_id], dtype=float
            ).tolist(),
        }
    affected_muscles = list(
        patient.get("affected_muscles", [patient.get("affected_muscle")])
    )
    muscle_damage = prepared.damage_verification["muscles"]
    return {
        "schema_version": "1.0",
        "run_id": plan["run_id"],
        "timestamp": plan["timestamp"].isoformat(timespec="seconds"),
        "experiment_name": str(config.EXPERIMENT_NAME),
        "experiment_type": str(
            getattr(config, "EXPERIMENT_TYPE", "human_exo_control")
        ),
        "human_model": str(config.HUMAN_MODEL),
        "human_condition": str(
            getattr(
                config,
                "HUMAN_CONDITION",
                patient.get("severity", patient.get("patient_type")),
            )
        ),
        "human_xml_path": _relative(metadata["human_xml"]),
        "human_xml_sha256": metadata["human_xml_sha256"],
        "patient_type": patient.get("patient_type"),
        "pathology_target": "+".join(str(name) for name in affected_muscles),
        "affected_muscles": affected_muscles,
        "strength_scale": float(patient["retained_force_fraction"]),
        "retained_fmax_percent": 100.0
        * float(patient["retained_force_fraction"]),
        "healthy_fmax_N": float(
            prepared.damage_verification["healthy_gain_force_N"]
        ),
        "actual_fmax_N": float(
            prepared.damage_verification["applied_gain_force_N"]
        ),
        "healthy_fmax_by_muscle_N": {
            name: float(muscle_damage[name]["healthy_gain_force_N"])
            for name in affected_muscles
        },
        "actual_fmax_by_muscle_N": {
            name: float(muscle_damage[name]["applied_gain_force_N"])
            for name in affected_muscles
        },
        "patient_config_path": _relative(metadata["patient_config_path"]),
        "patient_config_sha256": metadata["patient_config_sha256"],
        "exo_enabled": bool(config.EXO_ENABLED),
        "exo_model": str(config.EXO_MODEL),
        "exo_model_version": str(config.EXO_MODEL_VERSION),
        "exo_mass_kg": float(metadata["exo_mass_kg"]),
        "coupled_xml_path": _relative(metadata["model_xml"]),
        "coupled_xml_sha256": metadata["model_xml_sha256"],
        "human_controller": "official MyoAssist 2D reflex controller",
        "human_controller_type": getattr(
            config, "HUMAN_CONTROLLER_TYPE", "reflex"
        ),
        "human_controller_optimized_for": getattr(
            config, "HUMAN_CONTROLLER_OPTIMIZED_FOR", None
        ),
        "source_initialization": getattr(config, "SOURCE_INITIALIZATION", None),
        "human_controller_parameter_path": _relative(metadata["controller_path"]),
        "human_controller_parameter_count": metadata["parameter_count"],
        "human_controller_parameter_sha256": metadata["controller_sha256"],
        "assist_controller": str(
            getattr(
                config,
                "ASSISTANCE_NAME",
                "none" if assist is None else assist.__name__,
            )
        ),
        "assist_parameters": dict(config.ASSIST_PARAMS),
        "controller_mode": getattr(config, "CONTROLLER_MODE", None),
        "yac_mode": getattr(config, "YAC_MODE", None),
        "active_assistance": bool(
            getattr(config, "ACTIVE_ASSISTANCE", assist is not None)
        ),
        "yac_stage": getattr(config, "YAC_STAGE", None),
        "technical_migration_stage": getattr(
            config, "TECHNICAL_MIGRATION_STAGE", None
        ),
        "z_damping_rate_s_inv": getattr(
            config, "Z_DAMPING_RATE_S_INV", None
        ),
        "active_scale": getattr(config, "YAC_ACTIVE_SCALE", None),
        "adp_scale": getattr(config, "YAC_ADP_SCALE", None),
        "engagement_ramp_duration_s": getattr(
            config, "ENGAGEMENT_RAMP_DURATION_S", None
        ),
        "stage_d_torque_limit_nm": getattr(
            config, "STAGE_D_TORQUE_LIMIT_NM", None
        ),
        "total_torque_limit_nm": getattr(
            config, "TOTAL_TORQUE_LIMIT_NM", None
        ),
        "adp_residual_limit_nm": getattr(
            config, "ADP_RESIDUAL_LIMIT_NM", None
        ),
        "adp_residual_software_clipping": bool(
            getattr(config, "ADP_RESIDUAL_SOFTWARE_CLIPPING", False)
        ),
        "software_total_torque_clipping": bool(
            getattr(config, "SOFTWARE_TOTAL_TORQUE_CLIPPING", False)
        ),
        "native_actuator_limits": native_actuator_limits,
        "critic_basis": getattr(config, "CRITIC_BASIS", None),
        "critic_input": getattr(config, "CRITIC_INPUT", None),
        "critic_weight_dimension": getattr(
            config, "CRITIC_WEIGHT_DIMENSION", None
        ),
        "critic_z_scale_rad_s": getattr(
            config, "CRITIC_Z_SCALE_RAD_S", None
        ),
        "Q": getattr(config, "CRITIC_Q", None),
        "R": getattr(config, "CRITIC_R", None),
        "kappa_c": getattr(config, "CRITIC_KAPPA", None),
        "critic_eta_c": getattr(config, "CRITIC_ETA_C", None),
        "critic_weight_radius": getattr(
            config, "CRITIC_WEIGHT_RADIUS", None
        ),
        "critic_D_chi_diagonal": getattr(
            config, "CRITIC_D_CHI_DIAGONAL", None
        ),
        "critic_update_mode": getattr(config, "CRITIC_UPDATE_MODE", None),
        "critic_update_semantics": getattr(
            config, "CRITIC_UPDATE_SEMANTICS", None
        ),
        "experience_replay": getattr(config, "EXPERIENCE_REPLAY", None),
        "event_trigger": getattr(config, "EVENT_TRIGGER", None),
        "event_formula": getattr(config, "EVENT_FORMULA", None),
        "event_lambda_eta": getattr(config, "EVENT_LAMBDA_ETA", None),
        "event_sigma_eta": getattr(config, "EVENT_SIGMA_ETA", None),
        "event_xi": getattr(config, "EVENT_XI", None),
        "event_rho_delta": getattr(config, "EVENT_RHO_DELTA", None),
        "event_epsilon_delta": getattr(
            config, "EVENT_EPSILON_DELTA", None
        ),
        "event_eta_initial": getattr(config, "EVENT_ETA_INITIAL", None),
        "event_eta_discretization": getattr(
            config, "EVENT_ETA_DISCRETIZATION", None
        ),
        "event_parameter_source": getattr(
            config, "EVENT_PARAMETER_SOURCE", None
        ),
        "event_trigger_policy_semantics": getattr(
            config, "EVENT_TRIGGER_POLICY_SEMANTICS", None
        ),
        "event_policy_semantics": getattr(
            config, "EVENT_TRIGGER_POLICY_SEMANTICS", None
        ),
        "strict_full_torque_zoh": bool(
            getattr(config, "STRICT_FULL_TORQUE_ZOH", False)
        ),
        "event_trigger_parameters": getattr(
            config, "EVENT_TRIGGER_PARAMETERS", None
        ),
        "phase_beta": getattr(config, "PHASE_EWMA_BETA", None),
        "controlled_dofs": list(getattr(config, "CONTROLLED_DOFS", ())),
        "reference_source": getattr(config, "REFERENCE_SOURCE", None),
        "intention_source": getattr(config, "INTENTION_SOURCE", None),
        "interaction_semantics": getattr(
            config, "INTERACTION_SEMANTICS", None
        ),
        "paper_exact_tau_cancellation": getattr(
            config, "PAPER_EXACT_TAU_CANCELLATION", None
        ),
        "paper_candidate_recorded": getattr(
            config, "PAPER_CANDIDATE_RECORDED", None
        ),
        "requested_duration_s": float(config.DURATION_SECONDS),
        "actual_duration_s": 0.0,
        "control_timestep_s": float(metadata["control_timestep_s"]),
        "mujoco_timestep_s": float(metadata["mujoco_timestep_s"]),
        "target_velocity_mps": float(config.TARGET_VELOCITY_MPS),
        "seed": int(config.SEED),
        "human_mass_kg": float(metadata["human_mass_kg"]),
        "total_model_mass_kg": float(metadata["total_model_mass_kg"]),
        "save_video": bool(config.SAVE_VIDEO),
        "viewer": bool(getattr(config, "VIEWER", False)),
        "viewer_keep_open": bool(getattr(config, "VIEWER_KEEP_OPEN", False)),
        "strap_visualization": bool(
            getattr(config, "STRAP_VISUALIZATION", False)
        ),
        "strap_hud": bool(getattr(config, "STRAP_HUD", False)),
        "strap_visualization_physics_coupled": False,
        "strap_force_visual_reference_n": float(
            getattr(config, "STRAP_FORCE_VISUAL_REFERENCE_N", 200.0)
        ),
        "timeseries_schema_version": TIMESERIES_SCHEMA_VERSION,
        "coupling_mode": str(metadata["coupling_mode"]),
        "coupling_version": getattr(config, "COUPLING_VERSION", None),
        "parameter_set": getattr(config, "COUPLING_PARAMETER_SET", None),
        "active_interfaces": list(getattr(config, "ACTIVE_INTERFACES", ())),
        "disabled_interfaces": list(getattr(config, "DISABLED_INTERFACES", ())),
        "exo_dynamics": getattr(config, "EXO_DYNAMICS", "rigid_human_parented"),
        "physics_coupling_rate_hz": (
            1.0 / float(metadata["mujoco_timestep_s"])
            if metadata["coupling_mode"] == "nature_2024_linear_bushing"
            else None
        ),
        "rigid_baseline": getattr(config, "RIGID_BASELINE", None),
        "coupling_topology": getattr(config, "COUPLING_TOPOLOGY", None),
        "shank_parameter_source": getattr(config, "SHANK_PARAMETER_SOURCE", None),
        "nature_original_topology": getattr(
            config, "NATURE_ORIGINAL_TOPOLOGY", None
        ),
        "nature_exact_topology": getattr(config, "NATURE_EXACT_TOPOLOGY", None),
        "diagnostic_knee_lock": bool(
            getattr(config, "DIAGNOSTIC_KNEE_LOCK", False)
        ),
        "diagnostic_knee_lock_references_rad": metadata[
            "diagnostic_knee_lock_references"
        ],
        "diagnostic_relative_pitch_lock": bool(
            getattr(config, "DIAGNOSTIC_RELATIVE_PITCH_LOCK", False)
        ),
        "diagnostic_relative_pitch_reference_rad": metadata[
            "diagnostic_relative_pitch_reference"
        ],
        "diagnostic_waist_ap_lock": bool(
            getattr(config, "DIAGNOSTIC_WAIST_AP_LOCK", False)
        ),
        "diagnostic_waist_ap_reference_m": metadata[
            "diagnostic_waist_ap_reference"
        ],
        "physics_timeseries_schema_version": (
            "1.0"
            if metadata["coupling_mode"] == "nature_2024_linear_bushing"
            else None
        ),
        "waist_rotation_parameter_set": getattr(
            config, "WAIST_ROTATION_PARAMETER_SET", None
        ),
        "waist_rotation_stiffness_nm_per_rad": metadata[
            "waist_rotation_stiffness_nm_per_rad"
        ],
        "waist_rotation_damping_nms_per_rad": metadata[
            "waist_rotation_damping_nms_per_rad"
        ],
        "waist_rotation_parameter_source": getattr(
            config, "WAIST_ROTATION_PARAMETER_SOURCE", None
        ),
        "waist_translation_parameter_set": getattr(
            config, "WAIST_TRANSLATION_PARAMETER_SET", None
        ),
        "waist_ap_translation_stiffness_n_per_m": metadata[
            "waist_ap_translation_stiffness_n_per_m"
        ],
        "waist_ap_translation_damping_ns_per_m": metadata[
            "waist_ap_translation_damping_ns_per_m"
        ],
        "waist_translation_parameter_source": getattr(
            config, "WAIST_TRANSLATION_PARAMETER_SOURCE", None
        ),
        "git_branch": (
            _git_value("branch", "--show-current")
            if bool(getattr(config, "RECORD_GIT_METADATA", True))
            else "not_recorded_by_request"
        ),
        "git_commit": (
            _git_value("rev-parse", "HEAD")
            if bool(getattr(config, "RECORD_GIT_METADATA", True))
            else "not_recorded_by_request"
        ),
        "python_version": sys.version.split()[0],
        "success": False,
    }


def _validate_torque_output(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        raise TypeError("Assist controller compute() must return a dictionary")
    if set(value) != set(EXO_CHANNELS):
        missing = sorted(set(EXO_CHANNELS) - set(value))
        extra = sorted(set(value) - set(EXO_CHANNELS))
        raise ValueError(f"Assist torque keys mismatch; missing={missing}, extra={extra}")
    result = {name: float(value[name]) for name in EXO_CHANNELS}
    if not np.isfinite(np.asarray(list(result.values()))).all():
        raise FloatingPointError("Assist controller returned NaN or Inf")
    return result


def _validate_controller_diagnostics(value: Any) -> dict[str, Any]:
    """Validate one flat, deterministic controller diagnostics row."""
    if not isinstance(value, dict) or not value:
        raise TypeError("controller.last_diagnostics must be a non-empty dictionary")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise TypeError("Controller diagnostics keys must be non-empty strings")
        if isinstance(item, np.generic):
            item = item.item()
        if item is None:
            result[key] = ""
        elif isinstance(item, (str, bool, int)):
            result[key] = item
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise FloatingPointError(
                    f"Controller diagnostic {key} is NaN or Inf"
                )
            result[key] = item
        else:
            raise TypeError(
                f"Controller diagnostic {key} must be a scalar, got {type(item).__name__}"
            )
    return result


def _full_dynamics_audit_diagnostics(
    prepared: PreparedModel, bias_shadow_data: Any | None = None
) -> dict[str, float]:
    """Read-only full-model terms aligned with the controller observation."""
    model = prepared.model.ptr
    data = prepared.data.ptr
    joint_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in (
            "exo_root_forward",
            "exo_root_vertical",
            "exo_root_pitch",
            "exo_hip_r",
            "exo_knee_r",
            "exo_hip_l",
            "exo_knee_l",
        )
    }
    missing = sorted(name for name, joint_id in joint_ids.items() if joint_id < 0)
    if missing:
        raise ValueError(
            "Full dynamics audit requires Flexible Exo joints: " + ", ".join(missing)
        )
    dof = {
        name: int(model.jnt_dofadr[joint_id]) for name, joint_id in joint_ids.items()
    }
    local = np.asarray((dof["exo_hip_l"], dof["exo_knee_l"]), dtype=int)
    exo_root = np.asarray(
        (
            dof["exo_root_forward"],
            dof["exo_root_vertical"],
            dof["exo_root_pitch"],
        ),
        dtype=int,
    )
    right_exo = np.asarray((dof["exo_hip_r"], dof["exo_knee_r"]), dtype=int)
    first_exo_root = int(exo_root[0])
    human = np.arange(first_exo_root, dtype=int)

    mass = np.empty((int(model.nv), int(model.nv)), dtype=float)
    mujoco.mj_fullM(model, mass, data.qM)
    qacc = np.asarray(data.qacc, dtype=float)
    bias = np.asarray(data.qfrc_bias, dtype=float)
    if not np.isfinite(np.concatenate((mass.reshape(-1), qacc, bias))).all():
        raise FloatingPointError("Full dynamics audit encountered NaN or Inf")

    row_names = ("hip", "knee")
    result: dict[str, float] = {
        "full_M_hip_hip_kgm2": float(mass[local[0], local[0]]),
        "full_M_hip_knee_kgm2": float(mass[local[0], local[1]]),
        "full_M_knee_hip_kgm2": float(mass[local[1], local[0]]),
        "full_M_knee_knee_kgm2": float(mass[local[1], local[1]]),
    }
    root_labels = ("forward", "vertical", "pitch")
    for label, index in zip(root_labels, exo_root, strict=True):
        joint_name = f"exo_root_{label}"
        result[f"root_{label}_position"] = float(
            prepared.data.joint(joint_name).qpos[0]
        )
        result[f"root_{label}_velocity"] = float(
            prepared.data.joint(joint_name).qvel[0]
        )
        result[f"root_{label}_acceleration"] = float(qacc[index])
    groups = {
        "human": human,
        "exo_root": exo_root,
        "right_exo": right_exo,
        "left_local": local,
    }
    for row_name, row_index in zip(row_names, local, strict=True):
        full_inertial = float(mass[row_index] @ qacc)
        grouped = {
            name: float(mass[row_index, indices] @ qacc[indices])
            for name, indices in groups.items()
        }
        result[f"full_Mqacc_{row_name}_nm"] = full_inertial
        result[f"full_qfrc_bias_{row_name}_nm"] = float(bias[row_index])
        for root_label, root_index in zip(root_labels, exo_root, strict=True):
            coefficient = float(mass[row_index, root_index])
            result[f"full_M_{row_name}_root_{root_label}"] = coefficient
            result[f"full_Mqacc_root_{root_label}_{row_name}_nm"] = float(
                coefficient * qacc[root_index]
            )
        for group_name, contribution in grouped.items():
            result[f"full_Mqacc_{group_name}_{row_name}_nm"] = contribution
        result[f"full_Mqacc_group_sum_error_{row_name}_nm"] = float(
            full_inertial - sum(grouped.values())
        )
    if bias_shadow_data is not None:
        bias_shadow_data.qpos[:] = data.qpos
        bias_shadow_data.qvel[:] = data.qvel
        bias_shadow_data.qvel[exo_root] = 0.0
        bias_shadow_data.time = data.time
        mujoco.mj_forward(model, bias_shadow_data)
        zero_root_bias = np.asarray(bias_shadow_data.qfrc_bias, dtype=float)
        for row_name, row_index in zip(row_names, local, strict=True):
            shadow_value = float(zero_root_bias[row_index])
            result[f"full_qfrc_bias_zero_root_velocity_{row_name}_nm"] = shadow_value
            result[f"full_qfrc_bias_root_velocity_part_{row_name}_nm"] = float(
                bias[row_index] - shadow_value
            )
    return result


def _write_controller_diagnostics(
    path: Path, rows: list[dict[str, Any]]
) -> None:
    """Write optional controller-owned scalar diagnostics beside common artifacts."""
    if not rows:
        return
    fieldnames = list(rows[0])
    expected = set(fieldnames)
    for index, row in enumerate(rows[1:], start=1):
        if set(row) != expected:
            raise ValueError(
                f"Controller diagnostics keys changed at row {index}"
            )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(
    config: ModuleType | Any, plan: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Execute one experiment and return its final validation dictionary."""
    plan = plan or plan_experiment(config)
    prepared = model_setup.prepare_model(config)
    output_directory = Path(plan["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=False)

    manifest_path = output_directory / "manifest.json"
    csv_path = output_directory / "timeseries.csv"
    validation_path = output_directory / "validation.json"
    physics_csv_path = output_directory / "physics_timeseries_1khz.csv"
    controller_diagnostics_path = output_directory / "controller_diagnostics.csv"
    root_inertia_shadow_path = (
        output_directory / "yac_root_inertia_shadow_1khz.csv"
    )
    delayed_root_inertia_path = (
        output_directory / "yac_delayed_root_inertia_1khz.csv"
    )
    video_path = output_directory / str(
        getattr(config, "VIDEO_FILENAME", "walk.mp4")
    )
    manifest = _manifest(config, plan, prepared)
    _json_write(manifest_path, manifest)

    controller = None
    if config.ASSIST_CONTROLLER is not None:
        controller_parameters = dict(config.ASSIST_PARAMS)
        if bool(getattr(config, "ASSIST_REQUIRES_MUJOCO_MODEL", False)):
            controller_parameters["mujoco_model"] = prepared.model.ptr
        controller = config.ASSIST_CONTROLLER(**controller_parameters)
        controller.reset()

    environment = prepared.environment
    data = prepared.data
    recorder = ExperimentRecorder(prepared.muscle_names)
    controller_diagnostics_rows: list[dict[str, Any]] = []
    full_dynamics_bias_shadow = (
        mujoco.MjData(prepared.model.ptr)
        if bool(getattr(config, "YAC_FULL_DYNAMICS_AUDIT", False))
        else None
    )
    root_inertia_shadow_enabled = bool(
        getattr(config, "YAC_ROOT_INERTIA_SHADOW", False)
    )
    delayed_root_inertia_active = bool(
        getattr(config, "YAC_DELAYED_ROOT_INERTIA_ACTIVE", False)
    )
    if root_inertia_shadow_enabled and delayed_root_inertia_active:
        raise ValueError("Root-inertia shadow and active overlay are mutually exclusive")
    root_inertia_shadow = None
    if root_inertia_shadow_enabled or delayed_root_inertia_active:
        if prepared.coupling_runtime is None:
            raise ValueError("YAC root-inertia shadow requires flexible coupling")
        from simulation.yac_root_inertia_shadow import (
            YACDelayedRootInertiaOverlay,
            YACRootInertiaShadowRecorder,
        )

        root_inertia_shadow = (
            YACDelayedRootInertiaOverlay(prepared)
            if delayed_root_inertia_active
            else YACRootInertiaShadowRecorder(prepared)
        )
    zero_torques = {name: 0.0 for name in EXO_CHANNELS}
    cycle_index = {"right": 0, "left": 0}
    last_strike_time = {"right": -np.inf, "left": -np.inf}

    initial_observation = build_observation(prepared)
    initial_actual, initial_velocity = _exo_measurements(
        prepared, initial_observation
    )
    interaction_diagnostics = (
        None
        if prepared.coupling_runtime is None
        else prepared.coupling_runtime.snapshot()
    )
    recorder.record(
        step=0,
        observation=initial_observation,
        exo_commands=zero_torques,
        exo_actual=initial_actual,
        exo_joint_velocity=initial_velocity,
        heel_strike_side="",
        right_step_length_m=None,
        left_step_length_m=None,
        cycle_index=cycle_index,
        is_done=False,
        termination_reason="",
        interaction_diagnostics=interaction_diagnostics,
    )

    video_writer = None
    coupling_visualizer = None
    interactive_viewer = None
    capture_interval = max(
        1, round(1.0 / (float(config.VIDEO_FPS) * float(environment.dt)))
    )
    if bool(config.SAVE_VIDEO):
        if bool(getattr(config, "STRAP_VISUALIZATION", False)) or bool(
            getattr(config, "STRAP_HUD", False)
        ):
            if interaction_diagnostics is None:
                raise ValueError(
                    "Strap visualization/HUD requires flexible coupling diagnostics"
                )
            coupling_visualizer = CouplingVisualizer(
                prepared.model,
                prepared.data,
                environment.env.sim.renderer,
                width=VIDEO_WIDTH,
                height=VIDEO_HEIGHT,
                camera_id=CAMERA_ID,
                force_visual_reference_n=float(
                    getattr(config, "STRAP_FORCE_VISUAL_REFERENCE_N", 200.0)
                ),
                show_connectors=bool(
                    getattr(config, "STRAP_VISUALIZATION", False)
                ),
                show_hud=bool(getattr(config, "STRAP_HUD", False)),
            )
        video_writer = imageio.get_writer(
            video_path, fps=int(config.VIDEO_FPS), codec="libx264", quality=8
        )
        initial_frame = (
            coupling_visualizer.render(interaction_diagnostics, float(data.time))
            if coupling_visualizer is not None
            else environment.env.sim.renderer.render_offscreen(
                camera_id=CAMERA_ID, width=VIDEO_WIDTH, height=VIDEO_HEIGHT
            )
        )
        video_writer.append_data(initial_frame)

    viewer_enabled = bool(getattr(config, "VIEWER", False)) and os.environ.get(
        "MUJOCO_HEADLESS"
    ) != "1"
    if viewer_enabled:
        from mujoco import viewer as mujoco_viewer

        interactive_viewer = mujoco_viewer.launch_passive(
            prepared.model.ptr, prepared.data.ptr
        )
        interactive_viewer.sync()

    termination_reason = "time_limit"
    finite_state = True
    total_steps = int(round(float(config.DURATION_SECONDS) / float(environment.dt)))
    viewer_wall_start = time.perf_counter()
    viewer_sim_start = float(data.time)
    try:
        for step in range(1, total_steps + 1):
            if interactive_viewer is not None and not interactive_viewer.is_running():
                termination_reason = "viewer_closed"
                break
            previous_contact = {
                "right": bool(environment.footstep["r_contact"]),
                "left": bool(environment.footstep["l_contact"]),
            }

            environment.get_sensor_data()
            human_output = environment.update(environment.SENSOR_DATA)
            action = np.asarray(environment.reflex2mujoco(human_output), dtype=float)
            if action.shape != (int(prepared.model.nu),):
                raise ValueError("Human reflex action has the wrong shape")
            for actuator_id in prepared.exo_ids.values():
                action[actuator_id] = 0.0

            control_observation = build_observation(
                prepared,
                muscle_ctrl_override=action,
                include_full_state_snapshot=bool(
                    controller is not None
                    and getattr(controller, "requires_full_state_snapshot", False)
                ),
            )
            if controller is None:
                exo_commands = dict(zero_torques)
            else:
                exo_commands = _validate_torque_output(
                    controller.compute(
                        observation=control_observation, dt=float(environment.dt)
                    )
                )
                diagnostics = getattr(controller, "last_diagnostics", None)
                if diagnostics is not None:
                    if bool(getattr(config, "YAC_FULL_DYNAMICS_AUDIT", False)):
                        diagnostics.update(
                            _full_dynamics_audit_diagnostics(
                                prepared, full_dynamics_bias_shadow
                            )
                        )
                    controller_diagnostics_rows.append(
                        _validate_controller_diagnostics(diagnostics)
                    )
                controller_termination = str(
                    getattr(controller, "termination_request", "") or ""
                )
                if controller_termination:
                    termination_reason = controller_termination
                    break
            for channel, command in exo_commands.items():
                if channel in prepared.exo_ids:
                    action[prepared.exo_ids[channel]] = command
                elif command != 0.0:
                    raise ValueError(
                        f"Controller commanded absent actuator {channel}: {command}"
                    )

            # The runner owns the full action. Flexible mode replaces only the
            # scene's physics advance, preserving MyoSuite action processing,
            # reward/termination bookkeeping and one environment step.
            if prepared.coupling_runtime is None:
                environment.env.step(action)
                interaction_diagnostics = None
            else:
                if root_inertia_shadow is not None:
                    root_inertia_shadow.begin_control_step(step)
                interaction_diagnostics = prepared.coupling_runtime.step_environment(
                    environment, action, substep_observer=root_inertia_shadow
                )
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
            finite_state = bool(np.isfinite(state).all())

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
                    cycle_index[side] += 1
                    other = "left" if side == "right" else "right"
                    step_lengths[side] = float(
                        data.site(HEEL_SITES[side]).xpos[0]
                        - data.site(HEEL_SITES[other]).xpos[0]
                    )
            heel_strike_side = "both" if len(strikes) == 2 else "".join(strikes)

            observation = build_observation(prepared)
            reason = _termination_reason(observation, finite_state)
            is_done = bool(reason)
            actual, exo_velocity = _exo_measurements(prepared, observation)
            if controller_diagnostics_rows:
                current_diagnostics = controller_diagnostics_rows[-1]
                if "u_command_hip_nm" in current_diagnostics:
                    hip_actual = float(actual["HipExo_L"])
                    knee_actual = float(actual["KneeExo_L"])
                    current_diagnostics.update(
                        {
                            "actual_actuator_torque_hip_nm": hip_actual,
                            "actual_actuator_torque_knee_nm": knee_actual,
                            "u_command_minus_actual_hip_nm": float(
                                current_diagnostics["u_command_hip_nm"]
                            )
                            - hip_actual,
                            "u_command_minus_actual_knee_nm": float(
                                current_diagnostics["u_command_knee_nm"]
                            )
                            - knee_actual,
                        }
                    )
                bilateral_channels = {
                    "hip_l": "HipExo_L",
                    "knee_l": "KneeExo_L",
                    "hip_r": "HipExo_R",
                    "knee_r": "KneeExo_R",
                }
                for label, channel in bilateral_channels.items():
                    command_key = f"u_command_{label}_nm"
                    if command_key not in current_diagnostics:
                        continue
                    actual_value = float(actual[channel])
                    current_diagnostics[f"actual_actuator_torque_{label}_nm"] = actual_value
                    current_diagnostics[f"u_command_minus_actual_{label}_nm"] = (
                        float(current_diagnostics[command_key]) - actual_value
                    )
            recorder.record(
                step=step,
                observation=observation,
                exo_commands=exo_commands,
                exo_actual=actual,
                exo_joint_velocity=exo_velocity,
                heel_strike_side=heel_strike_side,
                right_step_length_m=step_lengths["right"],
                left_step_length_m=step_lengths["left"],
                cycle_index=cycle_index,
                is_done=is_done,
                termination_reason=reason,
                interaction_diagnostics=interaction_diagnostics,
            )
            if video_writer is not None and step % capture_interval == 0:
                frame = (
                    coupling_visualizer.render(
                        interaction_diagnostics, float(data.time)
                    )
                    if coupling_visualizer is not None
                    else environment.env.sim.renderer.render_offscreen(
                        camera_id=CAMERA_ID,
                        width=VIDEO_WIDTH,
                        height=VIDEO_HEIGHT,
                    )
                )
                video_writer.append_data(frame)
            if interactive_viewer is not None:
                interactive_viewer.sync()
                target_wall_time = viewer_wall_start + (
                    float(data.time) - viewer_sim_start
                )
                remaining = target_wall_time - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(remaining)
            if is_done:
                termination_reason = reason
                break
    finally:
        if video_writer is not None:
            video_writer.close()
        if interactive_viewer is not None:
            if bool(getattr(config, "VIEWER_KEEP_OPEN", False)):
                print("Simulation ended; close the MuJoCo viewer to finish.", flush=True)
                while interactive_viewer.is_running():
                    interactive_viewer.sync()
                    time.sleep(0.05)
            interactive_viewer.close()

    recorder.write_csv(csv_path)
    _write_controller_diagnostics(
        controller_diagnostics_path, controller_diagnostics_rows
    )
    if prepared.coupling_runtime is not None:
        prepared.coupling_runtime.write_physics_csv(physics_csv_path)
    if root_inertia_shadow is not None:
        root_inertia_shadow.write_csv(
            delayed_root_inertia_path
            if delayed_root_inertia_active
            else root_inertia_shadow_path
        )
    actual_duration = float(data.time)
    success = bool(
        termination_reason == "time_limit"
        and finite_state
        and actual_duration
        >= float(config.DURATION_SECONDS) - float(environment.dt)
    )
    manifest.update(
        {
            "actual_duration_s": actual_duration,
            "termination_reason": termination_reason,
            "success": success,
        }
    )
    _json_write(manifest_path, manifest)

    exo_all_zero = {}
    for channel in EXO_CHANNELS:
        if channel not in prepared.exo_ids:
            exo_all_zero[channel] = None
            continue
        exo_all_zero[channel] = bool(
            all(float(row[f"exo_command_torque_{channel}_nm"]) == 0.0 for row in recorder.rows)
            and all(float(row[f"exo_actual_torque_{channel}_nm"]) == 0.0 for row in recorder.rows)
        )
    validation = {
        **manifest,
        "requested_duration_seconds": float(config.DURATION_SECONDS),
        "walking_duration_seconds": actual_duration,
        "terminated_early": termination_reason != "time_limit",
        "finite_state": finite_state,
        "damage_verification": prepared.damage_verification,
        "model_dimensions": {
            "nq": int(prepared.model.nq),
            "nv": int(prepared.model.nv),
            "nu": int(prepared.model.nu),
        },
        "exoskeleton_channels_all_zero": exo_all_zero,
        "controller_diagnostics_row_count": len(controller_diagnostics_rows),
        "coupling_substep_count": (
            None
            if prepared.coupling_runtime is None
            else prepared.coupling_runtime.total_substep_count
        ),
        "coupling_summary": (
            None
            if prepared.coupling_runtime is None
            else prepared.coupling_runtime.overall_summary()
        ),
        "yac_root_inertia_shadow": (
            None if root_inertia_shadow is None else root_inertia_shadow.summary()
        ),
        "yac_delayed_root_inertia_active": delayed_root_inertia_active,
        "strap_visualization_summary": (
            None
            if coupling_visualizer is None
            else coupling_visualizer.summary()
        ),
        "artifacts": {
            "manifest": _relative(manifest_path),
            "timeseries": _relative(csv_path),
            "validation": _relative(validation_path),
            "video": _relative(video_path) if bool(config.SAVE_VIDEO) else None,
            "physics_timeseries_1khz": (
                _relative(physics_csv_path)
                if prepared.coupling_runtime is not None
                else None
            ),
            "controller_diagnostics": (
                _relative(controller_diagnostics_path)
                if controller_diagnostics_rows
                else None
            ),
            "yac_root_inertia_shadow_1khz": (
                _relative(root_inertia_shadow_path)
                if root_inertia_shadow is not None
                and not delayed_root_inertia_active
                else None
            ),
            "yac_delayed_root_inertia_1khz": (
                _relative(delayed_root_inertia_path)
                if delayed_root_inertia_active
                else None
            ),
        },
    }
    _json_write(validation_path, validation)

    latest_path = Path(config.OUTPUT_ROOT).expanduser().resolve() / "latest_run.json"
    _json_write(
        latest_path,
        {
            "run_id": plan["run_id"],
            "run_directory": _relative(output_directory),
        },
    )
    return validation
