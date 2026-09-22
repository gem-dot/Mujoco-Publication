"""Minimal MuJoCo physics-substep adapter for the isolated M5 fixture.

This module is intentionally not imported by :mod:`coupling` so that the
standalone LinearBushing core keeps its NumPy-only runtime dependency.  The
adapter owns only ``coupling_qfrc``.  All non-coupling applied generalized
forces must be supplied explicitly by the caller as ``base_external_qfrc``.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .linear_bushing import FrameState, InteractionState, LinearBushing


Vector = NDArray[np.float64]


def _copy_array(value: ArrayLike) -> Vector:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CouplingStepRecord:
    """Diagnostics evaluated immediately before one MuJoCo physics step.

    ``qpos`` and ``qvel`` are copied after ``mj_step``.  The interface state,
    wrenches and generalized forces correspond to ``time`` immediately before
    that step.  Wrench fields are ``None`` when coupling is disabled.
    """

    time: float
    dt: float
    coupling_enabled: bool
    human_site_position: Vector
    exo_site_position: Vector
    relative_translation: Vector | None
    relative_rotation: Vector | None
    human_force: Vector | None
    exo_force: Vector | None
    human_moment: Vector | None
    exo_moment: Vector | None
    coupling_qfrc: Vector
    coupling_qfrc_norm: float
    applied_qfrc: Vector
    qpos: Vector
    qvel: Vector
    interaction: InteractionState | None


class CouplingStepper:
    """Read two sites, apply one M5 bushing wrench, and advance one step.

    This class exists only to validate the 1 kHz physics-substep lifecycle on
    the two-body M5 fixture.  It does not integrate with MyoSuite or the formal
    project runner.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        bushing: LinearBushing,
        *,
        human_site_name: str = "human_interface_frame",
        exo_site_name: str = "exo_interface_frame",
    ) -> None:
        self.model = model
        self.data = data
        self.bushing = bushing
        self.human_site_id = self._site_id(human_site_name)
        self.exo_site_id = self._site_id(exo_site_name)
        self.human_body_id = int(model.site_bodyid[self.human_site_id])
        self.exo_body_id = int(model.site_bodyid[self.exo_site_id])
        self.coupling_qfrc = np.zeros(model.nv, dtype=np.float64)

        timestep = float(model.opt.timestep)
        if not np.isfinite(timestep) or timestep <= 0.0:
            raise ValueError("model.opt.timestep must be finite and positive")

    @property
    def dt(self) -> float:
        """Return the MuJoCo physics timestep without hard-coding 1 ms."""

        return float(self.model.opt.timestep)

    def _site_id(self, name: str) -> int:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise ValueError(f"MuJoCo site not found: {name}")
        return int(site_id)

    def _frame_state(self, site_id: int) -> FrameState:
        velocity = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_SITE,
            site_id,
            velocity,
            0,
        )
        # MuJoCo uses spatial-vector order (angular, linear).
        return FrameState(
            position=np.asarray(self.data.site_xpos[site_id], dtype=np.float64),
            orientation=np.asarray(
                self.data.site_xmat[site_id], dtype=np.float64
            ).reshape(3, 3),
            linear_velocity=velocity[3:6],
            angular_velocity=velocity[0:3],
        )

    def reset(self) -> None:
        """Reset only state owned by this adapter."""

        self.bushing.reset()
        self.coupling_qfrc.fill(0.0)

    def step(
        self,
        *,
        coupling_enabled: bool = True,
        base_external_qfrc: ArrayLike | None = None,
    ) -> CouplingStepRecord:
        """Recompute coupling, compose applied forces, and call ``mj_step`` once.

        ``base_external_qfrc`` represents the complete contribution owned by
        callers other than M5.  Omitting it means that the isolated fixture has
        no other external generalized forces.
        """

        if base_external_qfrc is None:
            base = np.zeros(self.model.nv, dtype=np.float64)
        else:
            base = np.asarray(base_external_qfrc, dtype=np.float64)
            if base.shape != (self.model.nv,):
                raise ValueError(
                    "base_external_qfrc must have shape "
                    f"{(self.model.nv,)}, got {base.shape}"
                )
            if not np.all(np.isfinite(base)):
                raise ValueError("base_external_qfrc must contain only finite values")

        # Synchronize site pose/twist with the qpos/qvel integrated by the
        # preceding mj_step before evaluating the next substep's bushing.
        mujoco.mj_forward(self.model, self.data)
        human_state = self._frame_state(self.human_site_id)
        exo_state = self._frame_state(self.exo_site_id)
        human_position = np.array(human_state.position, copy=True)
        exo_position = np.array(exo_state.position, copy=True)

        self.coupling_qfrc.fill(0.0)
        interaction: InteractionState | None = None
        if coupling_enabled:
            interaction = self.bushing.compute(human_state, exo_state, self.dt)
            mujoco.mj_applyFT(
                self.model,
                self.data,
                interaction.human_force,
                interaction.human_moment,
                human_position,
                self.human_body_id,
                self.coupling_qfrc,
            )
            mujoco.mj_applyFT(
                self.model,
                self.data,
                interaction.exo_force,
                interaction.exo_moment,
                exo_position,
                self.exo_body_id,
                self.coupling_qfrc,
            )
        else:
            # A disabled interval breaks the Euler unwrapping trajectory.  A
            # later re-enable therefore starts with a fresh branch reference.
            self.bushing.reset()

        applied = np.asarray(base, dtype=np.float64) + self.coupling_qfrc
        self.data.qfrc_applied[:] = applied
        time = float(self.data.time)
        mujoco.mj_step(self.model, self.data)

        return CouplingStepRecord(
            time=time,
            dt=self.dt,
            coupling_enabled=bool(coupling_enabled),
            human_site_position=_copy_array(human_position),
            exo_site_position=_copy_array(exo_position),
            relative_translation=(
                None
                if interaction is None
                else _copy_array(interaction.relative_translation)
            ),
            relative_rotation=(
                None if interaction is None else _copy_array(interaction.relative_rotation)
            ),
            human_force=(
                None if interaction is None else _copy_array(interaction.human_force)
            ),
            exo_force=(
                None if interaction is None else _copy_array(interaction.exo_force)
            ),
            human_moment=(
                None if interaction is None else _copy_array(interaction.human_moment)
            ),
            exo_moment=(
                None if interaction is None else _copy_array(interaction.exo_moment)
            ),
            coupling_qfrc=_copy_array(self.coupling_qfrc),
            coupling_qfrc_norm=float(np.linalg.norm(self.coupling_qfrc)),
            applied_qfrc=_copy_array(applied),
            qpos=_copy_array(self.data.qpos),
            qvel=_copy_array(self.data.qvel),
            interaction=interaction,
        )
