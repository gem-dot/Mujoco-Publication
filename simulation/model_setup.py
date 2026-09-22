"""Create one fresh verified MyoAssist environment for an experiment."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import mujoco
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_healthy_exo as healthy_utils  # noqa: E402
import run_patient_sweep as patient_utils  # noqa: E402
from simulation.flexible_coupling import (  # noqa: E402
    FlexibleCouplingRuntime,
    initialize_exo_from_human_pose,
)


EXO_CHANNELS = (
    "HipExo_R",
    "HipExo_L",
    "KneeExo_R",
    "KneeExo_L",
    "Exo_R",
    "Exo_L",
)
TOUCH_SENSORS = {
    "right_heel": "r_foot",
    "right_toe": "r_toes",
    "left_heel": "l_foot",
    "left_toe": "l_toes",
}
EXO_JOINTS = {
    "HipExo_R": "hip_flexion_r",
    "HipExo_L": "hip_flexion_l",
    "KneeExo_R": "knee_angle_r",
    "KneeExo_L": "knee_angle_l",
    "Exo_R": "ankle_angle_r",
    "Exo_L": "ankle_angle_l",
}


@dataclass
class PreparedModel:
    environment: Any
    model: Any
    data: Any
    actuator_ids: dict[str, int]
    muscle_names: tuple[str, ...]
    muscle_ids: dict[str, int]
    exo_ids: dict[str, int]
    exo_joint_names: dict[str, str]
    sensor_ids: dict[str, int]
    patient_config: dict[str, Any]
    damage_verification: dict[str, Any]
    metadata: dict[str, Any]
    coupling_runtime: FlexibleCouplingRuntime | None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_relative(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _load_patient_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Patient config is not a mapping: {path}")
    required = {"patient_type", "retained_force_fraction"}
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"Patient config is missing: {', '.join(missing)}")
    if "affected_muscle" not in value and "affected_muscles" not in value:
        raise ValueError("Patient config requires affected_muscle(s)")
    value["config_file"] = str(path.resolve())
    return value


def _set_diagnostic_knee_lock_references(model: Any, data: Any) -> dict[str, float]:
    """Set temporary equality references to the initialized Exo knee angles."""
    references: dict[str, float] = {}
    for side in ("r", "l"):
        joint_name = f"exo_knee_{side}"
        equality_name = f"diagnostic_lock_exo_knee_{side}"
        equality_id = mujoco.mj_name2id(
            model.ptr, mujoco.mjtObj.mjOBJ_EQUALITY, equality_name
        )
        if equality_id < 0:
            raise ValueError(
                f"Diagnostic knee-lock equality is missing: {equality_name}"
            )
        reference = float(data.joint(joint_name).qpos[0])
        model.ptr.eq_data[equality_id, 0] = reference
        references[joint_name] = reference
    mujoco.mj_forward(model.ptr, data.ptr)
    return references


def _set_diagnostic_relative_pitch_reference(model: Any, data: Any) -> float:
    """Set the Exo-minus-Human pitch equality to its initialized difference."""
    equality_name = "diagnostic_lock_relative_root_pitch"
    equality_id = mujoco.mj_name2id(
        model.ptr, mujoco.mjtObj.mjOBJ_EQUALITY, equality_name
    )
    if equality_id < 0:
        raise ValueError(
            f"Diagnostic relative-pitch equality is missing: {equality_name}"
        )
    reference = float(
        data.joint("exo_root_pitch").qpos[0]
        - data.joint("pelvis_tilt").qpos[0]
    )
    model.ptr.eq_data[equality_id, 0] = reference
    mujoco.mj_forward(model.ptr, data.ptr)
    return reference


def _set_diagnostic_waist_ap_reference(model: Any, data: Any) -> float:
    """Set the Exo-minus-Human forward-coordinate equality at initialization."""
    equality_name = "diagnostic_lock_relative_waist_ap"
    equality_id = mujoco.mj_name2id(
        model.ptr, mujoco.mjtObj.mjOBJ_EQUALITY, equality_name
    )
    if equality_id < 0:
        raise ValueError(f"Diagnostic waist-AP equality is missing: {equality_name}")
    reference = float(
        data.joint("exo_root_forward").qpos[0]
        - data.joint("pelvis_tx").qpos[0]
    )
    model.ptr.eq_data[equality_id, 0] = reference
    mujoco.mj_forward(model.ptr, data.ptr)
    return reference


def prepare_model(config: ModuleType | Any) -> PreparedModel:
    """Load a new model, then apply the existing verified patient damage path."""
    model_xml = Path(config.MODEL_XML).expanduser().resolve()
    controller_path = Path(config.HUMAN_CONTROLLER_PATH).expanduser().resolve()
    patient_path = Path(config.PATIENT_CONFIG_PATH).expanduser().resolve()
    for path in (model_xml, controller_path, patient_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required experiment input not found: {path}")
    if int(config.SEED) != 0:
        raise ValueError("The verified MyoAssist loader currently requires SEED = 0")

    parameters = np.loadtxt(controller_path).reshape(-1)
    if parameters.size != 77 or not np.isfinite(parameters).all():
        raise ValueError("Human reflex controller must contain 77 finite parameters")
    patient_config = _load_patient_config(patient_path)

    duration = float(config.DURATION_SECONDS)
    if duration <= 0.0:
        raise ValueError("DURATION_SECONDS must be positive")
    healthy_utils.DURATION_SECONDS = duration
    patient_utils.DURATION_SECONDS = duration

    # This is the same verified path used by the legacy patient runners.  It
    # creates a fresh environment before scaling the configured muscle Fmax
    # entries in gainprm/biasprm.
    environment, damage_verification = patient_utils.make_environment(
        parameters, patient_config, model_xml
    )
    environment.reset()
    model = environment.env.sim.model
    data = environment.env.sim.data

    actuator_metadata = healthy_utils.actuator_metadata(model)
    names = [str(item["name"]) for item in actuator_metadata]
    if len(names) != len(set(names)):
        raise ValueError("Actuator names must be unique")
    actuator_ids = {str(item["name"]): int(item["index"]) for item in actuator_metadata}
    muscle_names = tuple(
        str(item["name"]) for item in actuator_metadata if bool(item["is_muscle"])
    )
    if len(muscle_names) != 22:
        raise ValueError(f"Expected 22 muscle actuators, found {len(muscle_names)}")
    muscle_ids = {name: actuator_ids[name] for name in muscle_names}

    present_exo = tuple(name for name in EXO_CHANNELS if name in actuator_ids)
    expected_exo = tuple(getattr(config, "EXPECTED_EXO_CHANNELS", EXO_CHANNELS))
    if bool(config.EXO_ENABLED) and present_exo != expected_exo:
        missing = [name for name in expected_exo if name not in actuator_ids]
        unexpected = [name for name in present_exo if name not in expected_exo]
        raise ValueError(
            "Coupled model exoskeleton actuators mismatch; "
            f"missing={missing}, unexpected={unexpected}"
        )
    exo_ids = {name: actuator_ids[name] for name in present_exo}
    exo_joint_names = {
        name: str(actuator_metadata[actuator_ids[name]]["target"])
        for name in present_exo
    }

    coupling_mode = str(getattr(config, "COUPLING_MODE", "rigid"))
    coupling_runtime: FlexibleCouplingRuntime | None = None
    diagnostic_knee_lock_references: dict[str, float] = {}
    diagnostic_relative_pitch_reference: float | None = None
    diagnostic_waist_ap_reference: float | None = None
    if coupling_mode == "nature_2024_linear_bushing":
        required_targets = {
            "HipExo_R": "exo_hip_r",
            "HipExo_L": "exo_hip_l",
            "KneeExo_R": "exo_knee_r",
            "KneeExo_L": "exo_knee_l",
        }
        if exo_joint_names != required_targets:
            raise ValueError(
                "Flexible exoskeleton actuator targets are invalid: "
                f"{exo_joint_names}"
            )
        initialize_exo_from_human_pose(model, data)
        if bool(getattr(config, "DIAGNOSTIC_KNEE_LOCK", False)):
            diagnostic_knee_lock_references = (
                _set_diagnostic_knee_lock_references(model, data)
            )
        if bool(getattr(config, "DIAGNOSTIC_RELATIVE_PITCH_LOCK", False)):
            diagnostic_relative_pitch_reference = (
                _set_diagnostic_relative_pitch_reference(model, data)
            )
        if bool(getattr(config, "DIAGNOSTIC_WAIST_AP_LOCK", False)):
            diagnostic_waist_ap_reference = _set_diagnostic_waist_ap_reference(
                model, data
            )
        coupling_runtime = FlexibleCouplingRuntime(
            model,
            data,
            active_interface_names=tuple(
                getattr(
                    config,
                    "ACTIVE_INTERFACES",
                    ("waist", "thigh_l", "thigh_r"),
                )
            ),
            waist_rotation_stiffness_nm_per_rad=float(
                getattr(config, "WAIST_ROTATION_STIFFNESS_NM_PER_RAD", 0.0)
            ),
            waist_rotation_damping_nms_per_rad=float(
                getattr(config, "WAIST_ROTATION_DAMPING_NMS_PER_RAD", 0.0)
            ),
            waist_ap_translation_stiffness_n_per_m=float(
                getattr(config, "WAIST_AP_TRANSLATION_STIFFNESS_N_PER_M", 0.0)
            ),
            waist_ap_translation_damping_ns_per_m=float(
                getattr(config, "WAIST_AP_TRANSLATION_DAMPING_NS_PER_M", 0.0)
            ),
        )
    elif coupling_mode != "rigid":
        raise ValueError(f"Unsupported COUPLING_MODE: {coupling_mode}")

    sensor_ids: dict[str, int] = {}
    for logical_name, sensor_name in TOUCH_SENSORS.items():
        try:
            sensor_ids[logical_name] = int(model.sensor(sensor_name).id)
        except KeyError as exc:
            raise ValueError(f"Required touch sensor is missing: {sensor_name}") from exc

    total_mass = float(np.sum(model.ptr.body_mass[1:]))
    exo_mass = float(config.EXO_MASS_KG) if bool(config.EXO_ENABLED) else 0.0
    human_mass = total_mass - exo_mass
    if human_mass <= 0.0:
        raise ValueError("Computed human mass must be positive")

    metadata = {
        "human_xml": patient_utils.BASELINE_XML.resolve(),
        "model_xml": model_xml,
        "controller_path": controller_path,
        "patient_config_path": patient_path,
        "parameter_count": int(parameters.size),
        "model_xml_sha256": file_sha256(model_xml),
        "human_xml_sha256": file_sha256(patient_utils.BASELINE_XML.resolve()),
        "controller_sha256": file_sha256(controller_path),
        "patient_config_sha256": file_sha256(patient_path),
        "human_mass_kg": human_mass,
        "total_model_mass_kg": total_mass,
        "exo_mass_kg": exo_mass,
        "mujoco_timestep_s": float(model.ptr.opt.timestep),
        "control_timestep_s": float(environment.dt),
        "coupling_mode": coupling_mode,
        "diagnostic_knee_lock_references": diagnostic_knee_lock_references,
        "diagnostic_relative_pitch_reference": (
            diagnostic_relative_pitch_reference
        ),
        "diagnostic_waist_ap_reference": diagnostic_waist_ap_reference,
        "waist_rotation_stiffness_nm_per_rad": float(
            getattr(config, "WAIST_ROTATION_STIFFNESS_NM_PER_RAD", 0.0)
        ),
        "waist_rotation_damping_nms_per_rad": float(
            getattr(config, "WAIST_ROTATION_DAMPING_NMS_PER_RAD", 0.0)
        ),
        "waist_ap_translation_stiffness_n_per_m": float(
            getattr(config, "WAIST_AP_TRANSLATION_STIFFNESS_N_PER_M", 0.0)
        ),
        "waist_ap_translation_damping_ns_per_m": float(
            getattr(config, "WAIST_AP_TRANSLATION_DAMPING_NS_PER_M", 0.0)
        ),
    }
    return PreparedModel(
        environment=environment,
        model=model,
        data=data,
        actuator_ids=actuator_ids,
        muscle_names=muscle_names,
        muscle_ids=muscle_ids,
        exo_ids=exo_ids,
        exo_joint_names=exo_joint_names,
        sensor_ids=sensor_ids,
        patient_config=patient_config,
        damage_verification=damage_verification,
        metadata=metadata,
        coupling_runtime=coupling_runtime,
    )
