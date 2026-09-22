"""Causal 1 kHz shadow diagnostics for YAC root-inertia compensation.

The recorder never writes to the live ``MjData``.  It reconstructs the current
continuous-time dynamics in a private ``MjData`` after M5 has written the
current substep's generalized interaction force and before the live state is
integrated.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from controllers.YAC import (
    COORDINATE_NAMES,
    PlanarFloatingBaseLeftExoModel,
)


ROOT_NAMES = ("forward", "vertical", "pitch")
JOINT_NAMES = ("hip", "knee")


def _joint_addresses(
    model: mujoco.MjModel, names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    joint_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in names
        ],
        dtype=int,
    )
    if np.any(joint_ids < 0):
        missing = [name for name, joint_id in zip(names, joint_ids) if joint_id < 0]
        raise ValueError("Root-inertia shadow joints are missing: " + ", ".join(missing))
    return (
        np.asarray(model.jnt_qposadr[joint_ids], dtype=int),
        np.asarray(model.jnt_dofadr[joint_ids], dtype=int),
    )


class YACRootInertiaShadowRecorder:
    """Evaluate and retain time-aligned root inertial torque without actuation."""

    def __init__(self, prepared: Any) -> None:
        self.model: mujoco.MjModel = prepared.model.ptr
        self.live: mujoco.MjData = prepared.data.ptr
        self.shadow = mujoco.MjData(self.model)
        self.analytic = PlanarFloatingBaseLeftExoModel.from_mujoco(self.model)
        self.qpos_addresses, self.dof_addresses = _joint_addresses(
            self.model, COORDINATE_NAMES
        )
        self.root_dofs = self.dof_addresses[:3]
        self.joint_dofs = self.dof_addresses[3:]
        self.left_touch_sensor_ids = (
            int(prepared.sensor_ids["left_heel"]),
            int(prepared.sensor_ids["left_toe"]),
        )
        self.contact_threshold_n = (
            0.10 * float(prepared.metadata["total_model_mass_kg"]) * 9.8
        )
        self.left_actuator_ids = (
            int(prepared.exo_ids["HipExo_L"]),
            int(prepared.exo_ids["KneeExo_L"]),
        )
        self.control_step = 0
        self.rows: list[dict[str, float | int | bool]] = []
        self._pending_row: dict[str, float | int | bool] | None = None
        self._expected_qacc: np.ndarray | None = None
        self._zoh_root_torque: np.ndarray | None = None

    def begin_control_step(self, step: int) -> None:
        self.control_step = int(step)
        self._zoh_root_torque = None

    def start_substep(self, substep_index: int) -> None:
        """Optional pre-M5 hook; the read-only shadow path intentionally does nothing."""
        del substep_index

    def _copy_current_inputs(self) -> None:
        """Copy the exact continuous state and force inputs, never derived outputs."""
        source = self.live
        target = self.shadow
        target.time = source.time
        target.qpos[:] = source.qpos
        target.qvel[:] = source.qvel
        if self.model.na:
            target.act[:] = source.act
        target.ctrl[:] = source.ctrl
        target.qfrc_applied[:] = source.qfrc_applied
        target.xfrc_applied[:] = source.xfrc_applied
        target.qacc_warmstart[:] = source.qacc_warmstart
        target.eq_active[:] = source.eq_active
        if self.model.nmocap:
            target.mocap_pos[:] = source.mocap_pos
            target.mocap_quat[:] = source.mocap_quat
        if self.model.nuserdata:
            target.userdata[:] = source.userdata
        if getattr(self.model, "npluginstate", 0):
            target.plugin_state[:] = source.plugin_state

    def before_substep(self, substep_index: int) -> None:
        """Evaluate f(t_k, x_k, u_k) after M5 force construction, before integration."""
        if self._pending_row is not None:
            raise RuntimeError("Previous root-inertia shadow sample was not finalized")
        self._copy_current_inputs()
        mujoco.mj_forward(self.model, self.shadow)

        q_fb = np.asarray(self.shadow.qpos[self.qpos_addresses], dtype=float)
        qacc = np.asarray(self.shadow.qacc, dtype=float)
        root_acceleration = qacc[self.root_dofs]
        joint_acceleration = qacc[self.joint_dofs]
        mjr, mjj = self.analytic.joint_partitions(q_fb)
        components = mjr * root_acceleration[None, :]
        root_torque = np.sum(components, axis=1)
        joint_local = mjj @ joint_acceleration

        full_mass = np.empty((int(self.model.nv), int(self.model.nv)), dtype=float)
        mujoco.mj_fullM(self.model, full_mass, self.shadow.qM)
        full_inertia = full_mass[self.joint_dofs] @ qacc
        ground_truth = full_inertia - joint_local
        alignment_error = root_torque - ground_truth

        if int(substep_index) == 0:
            self._zoh_root_torque = root_torque.copy()
        if self._zoh_root_torque is None:
            raise RuntimeError("100 Hz ZOH sample was not initialized")

        touch = []
        for sensor_id in self.left_touch_sensor_ids:
            address = int(self.model.sensor_adr[sensor_id])
            touch.append(float(self.shadow.sensordata[address]))

        row: dict[str, float | int | bool] = {
            "time_s": float(self.shadow.time),
            "control_step": self.control_step,
            "physics_substep": int(substep_index),
            "left_heel_touch_n": touch[0],
            "left_toe_touch_n": touch[1],
            "left_contact": bool(
                touch[0] > self.contact_threshold_n
                or touch[1] > self.contact_threshold_n
            ),
            "contact_count": int(self.shadow.ncon),
            "constraint_count": int(self.shadow.nefc),
            "qfrc_applied_norm": float(np.linalg.norm(self.shadow.qfrc_applied)),
        }
        for index, root_name in enumerate(ROOT_NAMES):
            row[f"qacc_root_{root_name}"] = float(root_acceleration[index])
        for index, joint_name in enumerate(JOINT_NAMES):
            row[f"qacc_{joint_name}"] = float(joint_acceleration[index])
            row[f"tau_inertia_full_{joint_name}_nm"] = float(full_inertia[index])
            row[f"tau_joint_local_{joint_name}_nm"] = float(joint_local[index])
            row[f"tau_root_ground_truth_{joint_name}_nm"] = float(
                ground_truth[index]
            )
            row[f"tau_root_shadow_{joint_name}_nm"] = float(root_torque[index])
            row[f"tau_root_alignment_error_{joint_name}_nm"] = float(
                alignment_error[index]
            )
            row[f"tau_root_100hz_zoh_{joint_name}_nm"] = float(
                self._zoh_root_torque[index]
            )
            row[f"tau_root_1khz_minus_100hz_{joint_name}_nm"] = float(
                root_torque[index] - self._zoh_root_torque[index]
            )
            row[f"ctrl_{joint_name}_nm"] = float(
                self.shadow.ctrl[self.left_actuator_ids[index]]
            )
            row[f"actuator_force_{joint_name}_nm"] = float(
                self.shadow.actuator_force[self.left_actuator_ids[index]]
            )
            for root_index, root_name in enumerate(ROOT_NAMES):
                row[f"tau_root_{root_name}_{joint_name}_nm"] = float(
                    components[index, root_index]
                )
            row[f"tau_root_component_sum_error_{joint_name}_nm"] = float(
                root_torque[index] - np.sum(components[index])
            )

        finite_values = np.asarray(
            [float(value) for value in row.values() if not isinstance(value, bool)],
            dtype=float,
        )
        if not np.isfinite(finite_values).all():
            raise FloatingPointError("Root-inertia shadow sample contains NaN or Inf")
        self._pending_row = row
        self._expected_qacc = qacc.copy()

    def after_substep(self, substep_index: int) -> None:
        """Confirm live mj_step used the same pre-integration acceleration."""
        if self._pending_row is None or self._expected_qacc is None:
            raise RuntimeError("Root-inertia shadow sample is missing")
        if int(self._pending_row["physics_substep"]) != int(substep_index):
            raise RuntimeError("Root-inertia shadow substep index changed")
        live_qacc = np.asarray(self.live.qacc, dtype=float)
        difference = live_qacc - self._expected_qacc
        self._pending_row["post_step_time_s"] = float(self.live.time)
        self._pending_row["post_step_full_qacc_match_max_abs"] = float(
            np.max(np.abs(difference))
        )
        for index, root_name in enumerate(ROOT_NAMES):
            self._pending_row[f"post_step_qacc_match_{root_name}"] = float(
                difference[self.root_dofs[index]]
            )
        self.rows.append(self._pending_row)
        self._pending_row = None
        self._expected_qacc = None

    def write_csv(self, path: Path) -> None:
        if self._pending_row is not None:
            raise RuntimeError("Unfinalized root-inertia shadow sample")
        if not self.rows:
            raise RuntimeError("No root-inertia shadow samples were collected")
        fieldnames = list(self.rows[0])
        expected = set(fieldnames)
        if any(set(row) != expected for row in self.rows):
            raise ValueError("Root-inertia shadow diagnostic schema changed")
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)

    def summary(self) -> dict[str, Any]:
        maximum_live_match = max(
            float(row["post_step_full_qacc_match_max_abs"]) for row in self.rows
        )
        maximum_alignment = max(
            abs(float(row[f"tau_root_alignment_error_{joint}_nm"]))
            for row in self.rows
            for joint in JOINT_NAMES
        )
        return {
            "sample_count": len(self.rows),
            "physics_rate_hz": float(1.0 / self.model.opt.timestep),
            "live_post_step_qacc_match_max_abs": maximum_live_match,
            "root_torque_alignment_max_abs_nm": maximum_alignment,
            "writes_live_mjdata": False,
            "applies_shadow_torque": False,
            "uses_future_state": False,
            "mujoco_control_callback_present": bool(mujoco.get_mjcb_control()),
            "warmstart_disabled": bool(
                int(self.model.opt.disableflags)
                & int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
            ),
            "finite": bool(math.isfinite(maximum_live_match + maximum_alignment)),
        }


class YACDelayedRootInertiaOverlay(YACRootInertiaShadowRecorder):
    """Apply only the previous 1 kHz root-inertia sample to left Hip/Knee.

    Ordering is intentionally strict:

    1. ``start_substep`` applies ``previous_tau_root``.
    2. M5 is evaluated and written by ``FlexibleCouplingRuntime``.
    3. ``before_substep`` computes ``tau_root_current`` without changing ctrl.
    4. Live ``mj_step`` evaluates and integrates the current dynamics.
    5. ``after_substep`` stores ``tau_root_current`` for the next 1 ms sample.

    Thus the current sample can never affect the torque used to compute itself.
    """

    def __init__(self, prepared: Any) -> None:
        super().__init__(prepared)
        self.previous_tau_root = np.zeros(2, dtype=float)
        self._base_command: np.ndarray | None = None
        self._applied_delay = np.zeros(2, dtype=float)
        self._final_command = np.zeros(2, dtype=float)
        self._next_tau_root: np.ndarray | None = None
        self.right_actuator_ids = (
            int(prepared.exo_ids["HipExo_R"]),
            int(prepared.exo_ids["KneeExo_R"]),
        )
        self.ctrl_limits = np.asarray(
            [self.model.actuator_ctrlrange[index] for index in self.left_actuator_ids],
            dtype=float,
        )
        if not np.all(
            np.asarray(
                [self.model.actuator_ctrllimited[index] for index in self.left_actuator_ids]
            )
        ):
            raise ValueError("Delayed root-inertia overlay requires native ctrl limits")

    def begin_control_step(self, step: int) -> None:
        super().begin_control_step(step)
        self._base_command = None

    def start_substep(self, substep_index: int) -> None:
        """Apply k-1 correction before any dynamics evaluation of sample k."""
        if int(substep_index) == 0:
            # Robot.step has just written the new 100 Hz YAC command. Capture it
            # before this diagnostic actuator overlay changes the live ctrl.
            self._base_command = np.asarray(
                self.live.ctrl[list(self.left_actuator_ids)], dtype=float
            ).copy()
        if self._base_command is None:
            raise RuntimeError("YAC base command was not captured")
        self._applied_delay = self.previous_tau_root.copy()
        self._final_command = self._base_command + self._applied_delay
        self.live.ctrl[list(self.left_actuator_ids)] = self._final_command

    def before_substep(self, substep_index: int) -> None:
        """Compute current root inertia after M5; do not apply it in this sample."""
        super().before_substep(substep_index)
        if self._pending_row is None or self._base_command is None:
            raise RuntimeError("Delayed root-inertia sample was not constructed")
        row = self._pending_row
        self._next_tau_root = np.asarray(
            [float(row[f"tau_root_shadow_{joint}_nm"]) for joint in JOINT_NAMES],
            dtype=float,
        )
        for index, joint in enumerate(JOINT_NAMES):
            base = float(self._base_command[index])
            delayed = float(self._applied_delay[index])
            final = float(self._final_command[index])
            current = float(self._next_tau_root[index])
            row[f"yac_base_command_{joint}_nm"] = base
            row[f"tau_root_delay_applied_{joint}_nm"] = delayed
            row[f"final_command_before_native_saturation_{joint}_nm"] = final
            row[f"command_decomposition_error_{joint}_nm"] = final - base - delayed
            row[f"delay_error_vs_current_{joint}_nm"] = delayed - current
            row[f"native_saturation_active_{joint}"] = bool(
                final < self.ctrl_limits[index, 0]
                or final > self.ctrl_limits[index, 1]
            )
            row[f"right_ctrl_{joint}_nm"] = float(
                self.live.ctrl[self.right_actuator_ids[index]]
            )

    def after_substep(self, substep_index: int) -> None:
        """Commit current tau only after the same sample has been integrated."""
        if self._next_tau_root is None:
            raise RuntimeError("Current root-inertia correction is unavailable")
        super().after_substep(substep_index)
        completed = self.rows[-1]
        for index, joint in enumerate(JOINT_NAMES):
            completed[f"actual_actuator_force_{joint}_nm"] = float(
                self.live.actuator_force[self.left_actuator_ids[index]]
            )
        self.previous_tau_root = self._next_tau_root.copy()
        self._next_tau_root = None

    def summary(self) -> dict[str, Any]:
        result = super().summary()
        result.update(
            {
                "mode": "active_one_physics_step_delayed",
                "writes_live_mjdata": True,
                "writes_live_ctrl_only": True,
                "applies_shadow_torque": True,
                "delay_physics_steps": 1,
                "delay_seconds": float(self.model.opt.timestep),
                "controlled_channels": ["HipExo_L", "KneeExo_L"],
                "same_step_root_torque_applied": False,
            }
        )
        return result
