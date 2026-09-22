from __future__ import annotations

import math
from pathlib import Path
import unittest

import numpy as np

from coupling import (
    FrameState,
    LinearBushing,
    NATURE_2024_LEG_STRAP,
    NATURE_2024_WAIST,
)


ZERO = np.zeros(3)
IDENTITY = np.eye(3)


def body_xyz_matrix(theta: np.ndarray) -> np.ndarray:
    x, y, z = theta
    sx, cx = math.sin(x), math.cos(x)
    sy, cy = math.sin(y), math.cos(y)
    sz, cz = math.sin(z), math.cos(z)
    rx = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)), dtype=float)
    ry = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)), dtype=float)
    rz = np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)), dtype=float)
    return rx @ ry @ rz


def space_omega_from_body_xyz_rates(theta: np.ndarray, rates: np.ndarray) -> np.ndarray:
    x, y, _ = theta
    x_rate, y_rate, z_rate = rates
    return np.array(
        (
            x_rate + math.sin(y) * z_rate,
            math.cos(x) * y_rate - math.sin(x) * math.cos(y) * z_rate,
            math.sin(x) * y_rate + math.cos(x) * math.cos(y) * z_rate,
        )
    )


def frame(
    *,
    position: np.ndarray = ZERO,
    orientation: np.ndarray = IDENTITY,
    linear_velocity: np.ndarray = ZERO,
    angular_velocity: np.ndarray = ZERO,
) -> FrameState:
    return FrameState(position, orientation, linear_velocity, angular_velocity)


class LinearBushingAnalyticalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = NATURE_2024_LEG_STRAP.parameters
        self.strap = LinearBushing(self.parameters)

    def assertVectorClose(self, actual: np.ndarray, expected: np.ndarray, atol: float = 1e-11) -> None:
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=1e-11)

    def test_zero_state(self) -> None:
        result = self.strap.compute(frame(), frame(), 0.001)
        for value in (
            result.relative_translation,
            result.relative_linear_velocity,
            result.relative_rotation,
            result.relative_angular_velocity,
            result.human_force,
            result.human_moment,
            result.exo_force,
            result.exo_moment,
        ):
            self.assertVectorClose(value, ZERO)
        self.assertEqual(result.force_norm, 0.0)
        self.assertEqual(result.moment_norm, 0.0)

    def test_translation_spring_xyz(self) -> None:
        displacement = 0.005
        for axis in range(3):
            with self.subTest(axis=axis):
                self.strap.reset()
                offset = np.zeros(3)
                offset[axis] = displacement
                result = self.strap.compute(frame(), frame(position=offset), 0.001)
                expected = np.zeros(3)
                expected[axis] = self.parameters.translation_stiffness[axis] * displacement
                self.assertVectorClose(result.human_force, expected)
                self.assertVectorClose(result.exo_force, -expected)
                self.assertLess(float(np.dot(result.exo_force, offset)), 0.0)

    def test_translation_damping_xyz(self) -> None:
        relative_velocity = 0.2
        for axis in range(3):
            with self.subTest(axis=axis):
                self.strap.reset()
                velocity = np.zeros(3)
                velocity[axis] = relative_velocity
                result = self.strap.compute(
                    frame(), frame(linear_velocity=velocity), 0.001
                )
                expected = np.zeros(3)
                expected[axis] = self.parameters.translation_damping[axis] * relative_velocity
                self.assertVectorClose(result.human_force, expected)
                self.assertLess(float(np.dot(result.exo_force, velocity)), 0.0)

    def test_rotation_spring_xyz(self) -> None:
        angle = math.radians(2.0)
        for axis in range(3):
            with self.subTest(axis=axis):
                self.strap.reset()
                theta = np.zeros(3)
                theta[axis] = angle
                result = self.strap.compute(
                    frame(), frame(orientation=body_xyz_matrix(theta)), 0.001
                )
                expected = np.zeros(3)
                expected[axis] = self.parameters.rotation_stiffness[axis] * angle
                self.assertVectorClose(result.relative_rotation, theta)
                self.assertVectorClose(result.human_moment, expected)
                self.assertLess(float(np.dot(result.exo_moment, theta)), 0.0)

    def test_rotation_damping_xyz(self) -> None:
        angular_rate = 0.3
        for axis in range(3):
            with self.subTest(axis=axis):
                self.strap.reset()
                omega = np.zeros(3)
                omega[axis] = angular_rate
                result = self.strap.compute(
                    frame(), frame(angular_velocity=omega), 0.001
                )
                expected = np.zeros(3)
                expected[axis] = self.parameters.rotation_damping[axis] * angular_rate
                self.assertVectorClose(result.relative_angular_velocity, omega)
                self.assertVectorClose(result.human_moment, expected)
                self.assertLess(float(np.dot(result.exo_moment, omega)), 0.0)

    def test_combined_six_dof_matches_analytical_result(self) -> None:
        displacement = np.array((0.004, -0.003, 0.002))
        translation_rate = np.array((0.12, -0.08, 0.03))
        theta = np.array((0.03, -0.02, 0.04))
        theta_rate = np.array((0.2, -0.1, 0.15))
        angular_velocity = space_omega_from_body_xyz_rates(theta, theta_rate)
        result = self.strap.compute(
            frame(),
            frame(
                position=displacement,
                orientation=body_xyz_matrix(theta),
                linear_velocity=translation_rate,
                angular_velocity=angular_velocity,
            ),
            0.001,
        )
        expected_force = (
            self.parameters.translation_stiffness * displacement
            + self.parameters.translation_damping * translation_rate
        )
        expected_moment = (
            self.parameters.rotation_stiffness * theta
            + self.parameters.rotation_damping * theta_rate
        )
        self.assertVectorClose(result.relative_translation, displacement)
        self.assertVectorClose(result.relative_linear_velocity, translation_rate)
        self.assertVectorClose(result.relative_rotation, theta)
        self.assertVectorClose(result.relative_angular_velocity, theta_rate)
        self.assertVectorClose(result.human_force, expected_force)
        self.assertVectorClose(result.human_moment, expected_moment)

    def test_newton_third_law_at_common_reference(self) -> None:
        exo_position = np.array((0.004, -0.003, 0.002))
        result = self.strap.compute(
            frame(),
            frame(
                position=exo_position,
                orientation=body_xyz_matrix(np.array((0.02, 0.01, -0.03))),
            ),
            0.001,
        )
        self.assertVectorClose(result.human_force + result.exo_force, ZERO)
        exo_moment_at_human = result.exo_moment_about_reference(exo_position)
        self.assertVectorClose(result.human_moment + exo_moment_at_human, ZERO)

    def test_transport_term_in_relative_translation_rate(self) -> None:
        position = np.array((0.2, 0.0, 0.0))
        human_omega = np.array((0.0, 0.0, 2.0))
        result = self.strap.compute(
            frame(angular_velocity=human_omega),
            frame(position=position, angular_velocity=human_omega),
            0.001,
        )
        self.assertVectorClose(
            result.relative_linear_velocity, -np.cross(human_omega, position)
        )

    def test_small_values_and_coincident_frames_remain_finite(self) -> None:
        tiny = np.array((1e-12, -1e-12, 1e-12))
        result = self.strap.compute(
            frame(),
            frame(
                position=tiny,
                orientation=body_xyz_matrix(tiny),
                linear_velocity=tiny,
                angular_velocity=tiny,
            ),
            1e-4,
        )
        for value in result.__dict__.values():
            if isinstance(value, np.ndarray):
                self.assertTrue(np.all(np.isfinite(value)))

    def test_euler_branch_is_continuous_across_pi(self) -> None:
        first = self.strap.compute(
            frame(),
            frame(orientation=body_xyz_matrix(np.array((0.0, 0.0, math.radians(179.0))))),
            0.001,
        )
        second = self.strap.compute(
            frame(),
            frame(orientation=body_xyz_matrix(np.array((0.0, 0.0, math.radians(-179.0))))),
            0.001,
        )
        self.assertAlmostEqual(
            second.relative_rotation[2] - first.relative_rotation[2],
            math.radians(2.0),
            places=12,
        )
        self.assertTrue(np.all(np.isfinite(second.human_moment)))

    def test_repeated_small_motion_is_finite(self) -> None:
        for index in range(200):
            phase = index * 0.01
            theta = np.array((1e-3 * math.sin(phase), 5e-4, 1e-3 * math.cos(phase)))
            result = self.strap.compute(
                frame(), frame(orientation=body_xyz_matrix(theta)), 0.001
            )
            self.assertTrue(np.all(np.isfinite(result.human_force)))
            self.assertTrue(np.all(np.isfinite(result.human_moment)))

    def test_nature_parameter_transcription_and_units_disclosure(self) -> None:
        self.assertEqual(NATURE_2024_WAIST.reported_components, (("k_y", 8000.0), ("c_y", 10.0)))
        self.assertVectorClose(
            NATURE_2024_WAIST.parameters.translation_stiffness,
            np.array((0.0, 8000.0, 0.0)),
        )
        self.assertVectorClose(
            NATURE_2024_WAIST.parameters.translation_damping,
            np.array((0.0, 10.0, 0.0)),
        )
        self.assertVectorClose(
            NATURE_2024_LEG_STRAP.parameters.translation_stiffness,
            np.array((1500.0, 500.0, 1500.0)),
        )
        self.assertVectorClose(
            NATURE_2024_LEG_STRAP.parameters.translation_damping,
            np.array((10.0, 1.0, 10.0)),
        )
        self.assertVectorClose(
            NATURE_2024_LEG_STRAP.parameters.rotation_stiffness,
            np.array((10.0, 3.0, 10.0)),
        )
        self.assertVectorClose(
            NATURE_2024_LEG_STRAP.parameters.rotation_damping,
            np.array((1.0, 0.1, 1.0)),
        )
        self.assertIn("not stated", NATURE_2024_LEG_STRAP.reported_units)

    def test_rejects_euler_rate_singularity_without_nan(self) -> None:
        singular = body_xyz_matrix(np.array((0.0, math.pi / 2.0, 0.0)))
        with self.assertRaisesRegex(ValueError, "singular"):
            self.strap.compute(frame(), frame(orientation=singular), 0.001)


class M5FixtureSmokeTest(unittest.TestCase):
    def test_two_body_mujoco_fixture_loads_when_mujoco_is_available(self) -> None:
        try:
            import mujoco
        except ImportError:
            self.skipTest("MuJoCo is optional for the standalone M5 analytical tests")
        fixture = Path(__file__).parent / "fixtures" / "two_frame_bodies.xml"
        model = mujoco.MjModel.from_xml_path(str(fixture))
        self.assertGreaterEqual(model.nbody, 3)
        self.assertGreaterEqual(model.nsite, 2)


if __name__ == "__main__":
    unittest.main()
