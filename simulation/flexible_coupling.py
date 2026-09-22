"""Nature-bushing runtime for the independent M5.3 exoskeleton model."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from coupling import FrameState, LinearBushing
from coupling.nature_2024_parameters import NATURE_2024_LEG_STRAP, NATURE_2024_WAIST
from simulation.waist_rotation_parameters import extended_waist_parameters


NATURE_INTERFACE_NAMES = ("waist", "thigh_l", "thigh_r")
SHANK_INTERFACE_NAMES = ("shank_l", "shank_r")
ALL_INTERFACE_NAMES = NATURE_INTERFACE_NAMES + SHANK_INTERFACE_NAMES


@dataclass(frozen=True)
class InterfaceDefinition:
    name: str
    human_site: str
    exo_site: str
    bushing: LinearBushing


def initialize_exo_from_human_pose(model: Any, data: Any) -> None:
    """Copy the Human planar pose/twist into the independent Exo coordinates."""
    pairs = (
        ("exo_root_forward", "pelvis_tx"),
        ("exo_root_vertical", "pelvis_ty"),
        ("exo_root_pitch", "pelvis_tilt"),
        ("exo_hip_r", "hip_flexion_r"),
        ("exo_knee_r", "knee_angle_r"),
        ("exo_hip_l", "hip_flexion_l"),
        ("exo_knee_l", "knee_angle_l"),
    )
    for exo_name, human_name in pairs:
        data.joint(exo_name).qpos[0] = data.joint(human_name).qpos[0]
        data.joint(exo_name).qvel[0] = data.joint(human_name).qvel[0]
    mujoco.mj_forward(model.ptr, data.ptr)


class FlexibleCouplingRuntime:
    """Recompute the selected interface bushings before every physics substep."""

    def __init__(
        self,
        model: Any,
        data: Any,
        *,
        active_interface_names: tuple[str, ...] = NATURE_INTERFACE_NAMES,
        waist_rotation_stiffness_nm_per_rad: float = 0.0,
        waist_rotation_damping_nms_per_rad: float = 0.0,
        waist_ap_translation_stiffness_n_per_m: float = 0.0,
        waist_ap_translation_damping_ns_per_m: float = 0.0,
    ) -> None:
        self.model = model
        self.data = data
        self.raw_model = model.ptr
        self.raw_data = data.ptr
        self.dt = float(self.raw_model.opt.timestep)
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("MuJoCo physics timestep must be finite and positive")
        definitions = {
            "waist": InterfaceDefinition(
                "waist", "human_waist_interface", "exo_waist_interface",
                LinearBushing(
                    extended_waist_parameters(
                        waist_rotation_stiffness_nm_per_rad,
                        waist_rotation_damping_nms_per_rad,
                        waist_ap_translation_stiffness_n_per_m,
                        waist_ap_translation_damping_ns_per_m,
                    )
                ),
            ),
            "thigh_l": InterfaceDefinition(
                "thigh_l", "human_thigh_l_interface", "exo_thigh_l_interface",
                LinearBushing(NATURE_2024_LEG_STRAP.parameters),
            ),
            "thigh_r": InterfaceDefinition(
                "thigh_r", "human_thigh_r_interface", "exo_thigh_r_interface",
                LinearBushing(NATURE_2024_LEG_STRAP.parameters),
            ),
            "shank_l": InterfaceDefinition(
                "shank_l", "human_shank_l_interface", "exo_shank_l_interface",
                LinearBushing(NATURE_2024_LEG_STRAP.parameters),
            ),
            "shank_r": InterfaceDefinition(
                "shank_r", "human_shank_r_interface", "exo_shank_r_interface",
                LinearBushing(NATURE_2024_LEG_STRAP.parameters),
            ),
        }
        if len(active_interface_names) != len(set(active_interface_names)):
            raise ValueError("Active interface names must be unique")
        unknown = sorted(set(active_interface_names) - set(definitions))
        if unknown:
            raise ValueError(f"Unknown M5 interface names: {unknown}")
        self.active_interface_names = tuple(active_interface_names)
        self.interfaces = tuple(definitions[name] for name in self.active_interface_names)
        self._site_ids: dict[str, tuple[int, int]] = {}
        self._body_ids: dict[str, tuple[int, int]] = {}
        for item in self.interfaces:
            human_id = self._site_id(item.human_site)
            exo_id = self._site_id(item.exo_site)
            self._site_ids[item.name] = (human_id, exo_id)
            self._body_ids[item.name] = (
                int(self.raw_model.site_bodyid[human_id]),
                int(self.raw_model.site_bodyid[exo_id]),
            )
        self._relative_pitch_equality_id = int(
            mujoco.mj_name2id(
                self.raw_model,
                mujoco.mjtObj.mjOBJ_EQUALITY,
                "diagnostic_lock_relative_root_pitch",
            )
        )
        self._waist_ap_equality_id = int(
            mujoco.mj_name2id(
                self.raw_model,
                mujoco.mjtObj.mjOBJ_EQUALITY,
                "diagnostic_lock_relative_waist_ap",
            )
        )
        self.coupling_qfrc = np.zeros(self.raw_model.nv, dtype=np.float64)
        self.base_external_qfrc = np.zeros(self.raw_model.nv, dtype=np.float64)
        self.last_substep_count = 0
        self.total_substep_count = 0
        self._history: list[dict[str, dict[str, Any]]] = []
        self._physics_history: list[dict[str, float]] = []
        self.reset()

    def _site_id(self, name: str) -> int:
        value = mujoco.mj_name2id(self.raw_model, mujoco.mjtObj.mjOBJ_SITE, name)
        if value < 0:
            raise ValueError(f"Required M5 interface site is missing: {name}")
        return int(value)

    def _frame_state(self, site_id: int) -> FrameState:
        velocity = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.raw_model, self.raw_data, mujoco.mjtObj.mjOBJ_SITE,
            site_id, velocity, 0,
        )
        return FrameState(
            position=np.asarray(self.raw_data.site_xpos[site_id]),
            orientation=np.asarray(self.raw_data.site_xmat[site_id]).reshape(3, 3),
            linear_velocity=velocity[3:6],
            angular_velocity=velocity[0:3],
        )

    def reset(self) -> None:
        for item in self.interfaces:
            item.bushing.reset()
        self.coupling_qfrc.fill(0.0)
        self.base_external_qfrc.fill(0.0)
        self.last_substep_count = 0
        self.total_substep_count = 0
        self._history.clear()
        self._physics_history.clear()

    @staticmethod
    def _sample(interaction: Any, human_orientation: np.ndarray) -> dict[str, Any]:
        translation = np.asarray(interaction.relative_translation, dtype=float)
        translation_velocity = np.asarray(
            interaction.relative_linear_velocity, dtype=float
        )
        rotation = np.asarray(interaction.relative_rotation, dtype=float)
        rotation_velocity = np.asarray(
            interaction.relative_angular_velocity, dtype=float
        )
        force = np.asarray(interaction.human_force, dtype=float)
        moment = np.asarray(interaction.human_moment, dtype=float)
        force_local = np.asarray(human_orientation, dtype=float).T @ force
        moment_local = np.asarray(human_orientation, dtype=float).T @ moment
        return {
            "translation": translation.copy(),
            "translation_velocity": translation_velocity.copy(),
            "rotation": rotation.copy(),
            "rotation_velocity": rotation_velocity.copy(),
            "force": force.copy(),
            "moment": moment.copy(),
            "force_local": force_local.copy(),
            "moment_local": moment_local.copy(),
            "translation_norm": float(np.linalg.norm(translation)),
            "translation_velocity_norm": float(np.linalg.norm(translation_velocity)),
            "rotation_norm": float(np.linalg.norm(rotation)),
            "rotation_velocity_norm": float(np.linalg.norm(rotation_velocity)),
            "force_norm": float(np.linalg.norm(force)),
            "moment_norm": float(np.linalg.norm(moment)),
            "force_local_norm": float(np.linalg.norm(force_local)),
            "moment_local_norm": float(np.linalg.norm(moment_local)),
        }

    def _evaluate_and_apply(self) -> dict[str, dict[str, Any]]:
        mujoco.mj_forward(self.raw_model, self.raw_data)
        self.coupling_qfrc.fill(0.0)
        samples: dict[str, dict[str, Any]] = {}
        for item in self.interfaces:
            human_site, exo_site = self._site_ids[item.name]
            human = self._frame_state(human_site)
            exo = self._frame_state(exo_site)
            interaction = item.bushing.compute(human, exo, self.dt)
            human_body, exo_body = self._body_ids[item.name]
            mujoco.mj_applyFT(
                self.raw_model, self.raw_data,
                interaction.human_force, interaction.human_moment,
                human.position, human_body, self.coupling_qfrc,
            )
            mujoco.mj_applyFT(
                self.raw_model, self.raw_data,
                interaction.exo_force, interaction.exo_moment,
                exo.position, exo_body, self.coupling_qfrc,
            )
            samples[item.name] = self._sample(interaction, human.orientation)
        # M5 owns only coupling_qfrc. The caller-owned contribution is explicit.
        self.raw_data.qfrc_applied[:] = self.base_external_qfrc + self.coupling_qfrc
        return samples

    def _aggregate(self, samples: list[dict[str, dict[str, Any]]]) -> dict[str, Any]:
        if not samples:
            raise RuntimeError("No M5 physics-substep samples were collected")
        result: dict[str, Any] = {"physics_substeps": len(samples), "interfaces": {}}
        for name in self.active_interface_names:
            values = samples[-1][name]
            summary: dict[str, Any] = {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in values.items()
            }
            for metric in ("translation_norm", "rotation_norm", "force_norm", "moment_norm"):
                series = np.asarray([sample[name][metric] for sample in samples])
                summary[f"{metric}_peak"] = float(np.max(series))
                summary[f"{metric}_rms"] = float(np.sqrt(np.mean(series * series)))
                summary[f"{metric}_p95"] = float(np.percentile(series, 95.0))
            result["interfaces"][name] = summary
        return result

    def _joint_value(self, name: str, *, velocity: bool = False) -> float:
        joint_id = mujoco.mj_name2id(
            self.raw_model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if joint_id < 0:
            raise ValueError(f"Required diagnostic joint is missing: {name}")
        address = int(
            self.raw_model.jnt_dofadr[joint_id]
            if velocity
            else self.raw_model.jnt_qposadr[joint_id]
        )
        source = self.raw_data.qvel if velocity else self.raw_data.qpos
        return float(source[address])

    def _body_velocity(self, name: str) -> np.ndarray:
        body_id = mujoco.mj_name2id(
            self.raw_model, mujoco.mjtObj.mjOBJ_BODY, name
        )
        if body_id < 0:
            raise ValueError(f"Required diagnostic body is missing: {name}")
        velocity = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.raw_model, self.raw_data, mujoco.mjtObj.mjOBJ_BODY,
            body_id, velocity, 0,
        )
        return velocity

    def _joint_anchor(self, name: str) -> np.ndarray:
        joint_id = mujoco.mj_name2id(
            self.raw_model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        return np.asarray(self.raw_data.xanchor[joint_id], dtype=float)

    def _physics_sample(
        self, interfaces: dict[str, dict[str, Any]]
    ) -> dict[str, float]:
        exo_body = self.raw_data.xpos[
            mujoco.mj_name2id(
                self.raw_model, mujoco.mjtObj.mjOBJ_BODY, "exo_pelvis_frame"
            )
        ]
        human_body = self.raw_data.xpos[
            mujoco.mj_name2id(
                self.raw_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
            )
        ]
        exo_velocity = self._body_velocity("exo_pelvis_frame")
        human_velocity = self._body_velocity("pelvis")
        row: dict[str, float] = {
            "time": float(self.raw_data.time),
            "exo_root_x": float(exo_body[0]),
            "exo_root_z": float(exo_body[2]),
            "exo_root_pitch": self._joint_value("exo_root_pitch"),
            "exo_root_vx": float(exo_velocity[3]),
            "exo_root_vz": float(exo_velocity[5]),
            "exo_root_pitch_rate": self._joint_value(
                "exo_root_pitch", velocity=True
            ),
            "human_pelvis_x": float(human_body[0]),
            "human_pelvis_z": float(human_body[2]),
            "human_pelvis_pitch": self._joint_value("pelvis_tilt"),
            "human_pelvis_vx": float(human_velocity[3]),
            "human_pelvis_vz": float(human_velocity[5]),
            "human_pelvis_pitch_rate": self._joint_value(
                "pelvis_tilt", velocity=True
            ),
        }
        if self._relative_pitch_equality_id >= 0:
            reference = float(
                self.raw_model.eq_data[self._relative_pitch_equality_id, 0]
            )
            relative_pitch = row["exo_root_pitch"] - row["human_pelvis_pitch"]
            mask = (
                (np.asarray(self.raw_data.efc_type) == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
                & (np.asarray(self.raw_data.efc_id) == self._relative_pitch_equality_id)
            )
            row["diagnostic_relative_pitch_reference"] = reference
            row["diagnostic_relative_pitch_error"] = relative_pitch - reference
            row["diagnostic_constraint_residual"] = (
                float(np.max(np.abs(np.asarray(self.raw_data.efc_pos)[mask])))
                if np.any(mask)
                else float("nan")
            )
            row["diagnostic_constraint_force"] = (
                float(np.sum(np.abs(np.asarray(self.raw_data.efc_force)[mask])))
                if np.any(mask)
                else float("nan")
            )
        if self._waist_ap_equality_id >= 0:
            reference = float(self.raw_model.eq_data[self._waist_ap_equality_id, 0])
            relative_ap = self._joint_value("exo_root_forward") - self._joint_value(
                "pelvis_tx"
            )
            mask = (
                (np.asarray(self.raw_data.efc_type) == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
                & (np.asarray(self.raw_data.efc_id) == self._waist_ap_equality_id)
            )
            row["diagnostic_waist_ap_reference"] = reference
            row["diagnostic_waist_ap_error"] = relative_ap - reference
            row["diagnostic_waist_ap_constraint_residual"] = (
                float(np.max(np.abs(np.asarray(self.raw_data.efc_pos)[mask])))
                if np.any(mask)
                else float("nan")
            )
            row["diagnostic_waist_ap_constraint_force"] = (
                float(np.sum(np.abs(np.asarray(self.raw_data.efc_force)[mask])))
                if np.any(mask)
                else float("nan")
            )
        for prefix, name in (
            ("exo_hip_r", "exo_hip_r"),
            ("exo_hip_l", "exo_hip_l"),
            ("exo_knee_r", "exo_knee_r"),
            ("exo_knee_l", "exo_knee_l"),
            ("human_hip_r", "hip_flexion_r"),
            ("human_hip_l", "hip_flexion_l"),
            ("human_knee_r", "knee_angle_r"),
            ("human_knee_l", "knee_angle_l"),
        ):
            row[f"{prefix}_angle"] = self._joint_value(name)
            row[f"{prefix}_velocity"] = self._joint_value(name, velocity=True)
        # Evaluation-only M5 generalized torques.  These are the exact current
        # coupling_qfrc entries already applied by _evaluate_and_apply(); logging
        # them here does not alter the force calculation or integration order.
        for label, joint_name in (
            ("hip_l", "exo_hip_l"),
            ("knee_l", "exo_knee_l"),
            ("hip_r", "exo_hip_r"),
            ("knee_r", "exo_knee_r"),
        ):
            joint_id = mujoco.mj_name2id(
                self.raw_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id < 0:
                raise ValueError(f"Required M5 diagnostic joint is missing: {joint_name}")
            dof_address = int(self.raw_model.jnt_dofadr[joint_id])
            row[f"coupling_qfrc_{label}_nm"] = float(
                self.coupling_qfrc[dof_address]
            )
        for side in ("r", "l"):
            human_anchor = self._joint_anchor(f"knee_angle_{side}")
            exo_anchor = self._joint_anchor(f"exo_knee_{side}")
            row[f"knee_center_misalignment_{side}"] = float(
                np.linalg.norm(exo_anchor - human_anchor)
            )
        for name, values in interfaces.items():
            for vector in (
                "translation",
                "translation_velocity",
                "rotation",
                "rotation_velocity",
                "force",
                "moment",
                "force_local",
                "moment_local",
            ):
                for index, axis in enumerate(("x", "y", "z")):
                    row[f"{name}_{vector}_{axis}"] = float(values[vector][index])
                row[f"{name}_{vector}_norm"] = float(values[f"{vector}_norm"])
        return row

    def snapshot(self) -> dict[str, Any]:
        """Evaluate diagnostics without advancing physics or retaining a force."""
        samples = self._evaluate_and_apply()
        self.raw_data.qfrc_applied[:] = self.base_external_qfrc
        return self._aggregate([samples])

    def step_environment(
        self,
        environment: Any,
        action: np.ndarray,
        *,
        substep_observer: Any | None = None,
    ) -> dict[str, Any]:
        """Preserve the MyoSuite step path while replacing only scene.advance."""
        scene = environment.env.sim
        original_advance = scene.advance
        samples: list[dict[str, dict[str, Any]]] = []

        def coupled_advance(*, substeps: int = 1, render: bool = True) -> None:
            del render
            requested = int(substeps)
            if requested <= 0 or requested != substeps:
                raise ValueError(f"Invalid physics substep count: {substeps}")
            for substep_index in range(requested):
                if substep_observer is not None:
                    substep_observer.start_substep(substep_index)
                interface_sample = self._evaluate_and_apply()
                samples.append(interface_sample)
                self._physics_history.append(self._physics_sample(interface_sample))
                if substep_observer is not None:
                    substep_observer.before_substep(substep_index)
                mujoco.mj_step(self.raw_model, self.raw_data)
                if substep_observer is not None:
                    substep_observer.after_substep(substep_index)

        scene.advance = coupled_advance
        try:
            environment.env.step(action)
        finally:
            scene.advance = original_advance
            self.raw_data.qfrc_applied[:] = self.base_external_qfrc
        self.last_substep_count = len(samples)
        self.total_substep_count += len(samples)
        self._history.extend(samples)
        expected = int(round(float(environment.dt) / self.dt))
        if self.last_substep_count != expected:
            raise RuntimeError(
                f"M5 executed {self.last_substep_count} substeps; expected {expected}"
            )
        return self._aggregate(samples)

    def overall_summary(self) -> dict[str, Any]:
        """Return exact peak/RMS/p95 values over every executed 1 kHz substep."""
        return self._aggregate(self._history) if self._history else self.snapshot()

    def write_physics_csv(self, path: Path) -> None:
        """Persist complete pre-step diagnostics at the MuJoCo physics rate."""
        if not self._physics_history:
            raise RuntimeError("No 1 kHz diagnostic samples are available")
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(self._physics_history[0].keys())
            )
            writer.writeheader()
            writer.writerows(self._physics_history)
