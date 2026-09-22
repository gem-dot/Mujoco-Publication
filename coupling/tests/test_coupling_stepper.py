from __future__ import annotations

import math
from pathlib import Path
import unittest

import mujoco
import numpy as np

from coupling import BushingParameters, LinearBushing, NATURE_2024_LEG_STRAP
from coupling.coupling_stepper import CouplingStepRecord, CouplingStepper


FIXTURE = Path(__file__).parent / "fixtures" / "two_frame_bodies.xml"
ZERO3 = np.zeros(3)


def load_fixture(
    parameters: BushingParameters | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData, CouplingStepper]:
    model = mujoco.MjModel.from_xml_path(str(FIXTURE))
    data = mujoco.MjData(model)
    stepper = CouplingStepper(
        model,
        data,
        LinearBushing(parameters or NATURE_2024_LEG_STRAP.parameters),
    )
    return model, data, stepper


def set_exo_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation_x: float = 0.0,
    linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    mujoco.mj_resetData(model, data)
    joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "exo_fixture_freejoint"
    )
    qpos_adr = int(model.jnt_qposadr[joint_id])
    dof_adr = int(model.jnt_dofadr[joint_id])
    data.qpos[qpos_adr : qpos_adr + 3] = position
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = (
        math.cos(rotation_x / 2.0),
        math.sin(rotation_x / 2.0),
        0.0,
        0.0,
    )
    data.qvel[dof_adr : dof_adr + 3] = linear_velocity
    data.qvel[dof_adr + 3 : dof_adr + 6] = angular_velocity
    mujoco.mj_forward(model, data)


def body_translational_energy(
    model: mujoco.MjModel, data: mujoco.MjData, body_id: int
) -> float:
    dof_adr = int(model.body_dofadr[body_id])
    velocity = data.qvel[dof_adr : dof_adr + 3]
    return 0.5 * float(model.body_mass[body_id]) * float(np.dot(velocity, velocity))


def assert_record_finite(test: unittest.TestCase, record: CouplingStepRecord) -> None:
    arrays = (
        record.human_site_position,
        record.exo_site_position,
        record.coupling_qfrc,
        record.applied_qfrc,
        record.qpos,
        record.qvel,
    )
    for value in (
        record.relative_translation,
        record.relative_rotation,
        record.human_force,
        record.exo_force,
        record.human_moment,
        record.exo_moment,
    ):
        if value is not None:
            arrays += (value,)
    for array in arrays:
        test.assertTrue(np.all(np.isfinite(array)))
    test.assertTrue(math.isfinite(record.time))
    test.assertTrue(math.isfinite(record.dt))
    test.assertTrue(math.isfinite(record.coupling_qfrc_norm))


class CouplingStepperFixtureTests(unittest.TestCase):
    def test_g1_zero_state_produces_zero_force(self) -> None:
        model, _, stepper = load_fixture()
        record = stepper.step()
        self.assertEqual(model.opt.timestep, 0.001)
        np.testing.assert_allclose(record.relative_translation, ZERO3, atol=1e-14)
        np.testing.assert_allclose(record.relative_rotation, ZERO3, atol=1e-14)
        np.testing.assert_allclose(record.human_force, ZERO3, atol=1e-14)
        np.testing.assert_allclose(record.exo_force, ZERO3, atol=1e-14)
        np.testing.assert_allclose(record.human_moment, ZERO3, atol=1e-14)
        np.testing.assert_allclose(record.exo_moment, ZERO3, atol=1e-14)
        self.assertLessEqual(record.coupling_qfrc_norm, 1e-14)

    def test_g2_fixed_displacement_force_does_not_accumulate(self) -> None:
        model, data, stepper = load_fixture()
        set_exo_state(model, data, position=(0.005, 0.0, 0.0))
        fixed_qpos = np.array(data.qpos, copy=True)
        records: list[CouplingStepRecord] = []
        for _ in range(10):
            records.append(stepper.step())
            # Hold the fixture pose fixed without resetting simulation time or
            # qfrc_applied.  The next call must replace the previous M5 load.
            data.qpos[:] = fixed_qpos
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)

        forces = np.asarray([record.exo_force for record in records])
        qfrc = np.asarray([record.coupling_qfrc for record in records])
        expected_force = np.array((-7.5, 0.0, 0.0))
        np.testing.assert_allclose(
            forces, np.tile(expected_force, (10, 1)), atol=1e-12
        )
        np.testing.assert_allclose(qfrc, np.tile(qfrc[0], (10, 1)), atol=1e-12)
        np.testing.assert_allclose(
            [record.time for record in records],
            np.arange(10, dtype=float) * model.opt.timestep,
            atol=1e-15,
            rtol=0.0,
        )

    def test_g3_translation_perturbation_restores_toward_zero(self) -> None:
        model, data, stepper = load_fixture()
        set_exo_state(model, data, position=(0.005, 0.0, 0.0))
        records = [stepper.step() for _ in range(30)]
        self.assertLess(records[0].exo_force[0], 0.0)
        self.assertLess(abs(records[-1].relative_translation[0]), 0.005)
        self.assertLess(
            records[-1].relative_translation[0],
            records[0].relative_translation[0],
        )

    def test_g4_damping_opposes_velocity_and_removes_energy(self) -> None:
        damping_only = BushingParameters(
            translation_stiffness=(0.0, 0.0, 0.0),
            translation_damping=(10.0, 0.0, 0.0),
            rotation_stiffness=(0.0, 0.0, 0.0),
            rotation_damping=(0.0, 0.0, 0.0),
        )
        model, data, stepper = load_fixture(damping_only)
        set_exo_state(model, data, linear_velocity=(0.2, 0.0, 0.0))
        initial_energy = body_translational_energy(
            model, data, stepper.human_body_id
        ) + body_translational_energy(model, data, stepper.exo_body_id)
        records = [stepper.step() for _ in range(100)]
        initial_power = float(
            np.dot(records[0].exo_force, np.array((0.2, 0.0, 0.0)))
        )
        final_energy = body_translational_energy(
            model, data, stepper.human_body_id
        ) + body_translational_energy(model, data, stepper.exo_body_id)
        self.assertLess(initial_power, 0.0)
        self.assertLess(final_energy, initial_energy)

    def test_g5_rotation_perturbation_restores_toward_zero(self) -> None:
        angle = math.radians(2.0)
        model, data, stepper = load_fixture()
        set_exo_state(model, data, rotation_x=angle)
        records = [stepper.step() for _ in range(50)]
        self.assertLess(records[0].exo_moment[0], 0.0)
        self.assertLess(abs(records[-1].relative_rotation[0]), angle)

    def test_g6_newton_third_law_holds_each_substep(self) -> None:
        model, data, stepper = load_fixture()
        set_exo_state(
            model,
            data,
            position=(0.004, -0.003, 0.002),
            rotation_x=math.radians(1.0),
        )
        for _ in range(20):
            record = stepper.step()
            human_dof = int(model.body_dofadr[stepper.human_body_id])
            exo_dof = int(model.body_dofadr[stepper.exo_body_id])
            self.assertGreater(
                np.linalg.norm(record.coupling_qfrc[human_dof : human_dof + 6]),
                0.0,
            )
            self.assertGreater(
                np.linalg.norm(record.coupling_qfrc[exo_dof : exo_dof + 6]),
                0.0,
            )
            np.testing.assert_allclose(
                record.human_force + record.exo_force, ZERO3, atol=1e-12
            )
            exo_moment_at_human = record.interaction.exo_moment_about_reference(
                record.exo_site_position
            )
            np.testing.assert_allclose(
                record.human_moment + exo_moment_at_human, ZERO3, atol=1e-12
            )

    def test_g7_disabled_next_step_clears_only_coupling_contribution(self) -> None:
        model, data, stepper = load_fixture()
        set_exo_state(model, data, position=(0.005, 0.0, 0.0))
        enabled = stepper.step()
        self.assertGreater(enabled.coupling_qfrc_norm, 0.0)

        base = np.linspace(0.01, 0.06, model.nv)
        disabled = stepper.step(coupling_enabled=False, base_external_qfrc=base)
        self.assertEqual(disabled.coupling_qfrc_norm, 0.0)
        np.testing.assert_array_equal(disabled.coupling_qfrc, np.zeros(model.nv))
        np.testing.assert_allclose(disabled.applied_qfrc, base, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(data.qfrc_applied, base, atol=0.0, rtol=0.0)

    def test_g8_disabled_stepper_matches_plain_mj_step(self) -> None:
        model_a = mujoco.MjModel.from_xml_path(str(FIXTURE))
        data_a = mujoco.MjData(model_a)
        model_b, data_b, stepper = load_fixture()
        initial = dict(
            position=(0.01, -0.005, 0.003),
            rotation_x=math.radians(1.0),
            linear_velocity=(0.03, -0.02, 0.01),
            angular_velocity=(0.04, 0.02, -0.01),
        )
        set_exo_state(model_a, data_a, **initial)
        set_exo_state(model_b, data_b, **initial)

        for _ in range(50):
            mujoco.mj_step(model_a, data_a)
            stepper.step(coupling_enabled=False)

        np.testing.assert_allclose(data_b.qpos, data_a.qpos, atol=1e-14, rtol=0.0)
        np.testing.assert_allclose(data_b.qvel, data_a.qvel, atol=1e-14, rtol=0.0)

    def test_g9_dynamic_records_contain_no_nan_or_inf(self) -> None:
        model, data, stepper = load_fixture()
        set_exo_state(
            model,
            data,
            position=(0.001, 0.0, 0.0),
            rotation_x=math.radians(0.5),
            linear_velocity=(0.01, 0.0, 0.0),
            angular_velocity=(0.01, 0.0, 0.0),
        )
        for _ in range(200):
            assert_record_finite(self, stepper.step())


if __name__ == "__main__":
    unittest.main()
