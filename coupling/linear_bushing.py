"""Nature-style linear bushing mechanics for two Cartesian frames.

FRAME_A is the human interface frame and FRAME_B is the exoskeleton
interface frame.  The relative transform is always B relative to A:

    d_A = R_A.T @ (p_B - p_A)
    R_AB = R_A.T @ R_B

where rotation matrices map local coordinates to world coordinates.  Eq. (5)
of Luo et al. (Nature, 2024) is evaluated component-wise in FRAME_A.  The
positive Eq. (5) vectors are assigned to FRAME_A and their restoring
opposites to FRAME_B, so positive relative displacement/velocity cannot
produce positive feedback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


Vector3: TypeAlias = NDArray[np.float64]
Matrix3: TypeAlias = NDArray[np.float64]


def _readonly_array(value: ArrayLike, shape: tuple[int, ...], name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    result = np.array(array, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _validate_rotation_matrix(rotation: Matrix3, name: str) -> None:
    identity_error = np.linalg.norm(rotation.T @ rotation - np.eye(3), ord=np.inf)
    determinant = float(np.linalg.det(rotation))
    if identity_error > 1e-9 or not math.isclose(determinant, 1.0, abs_tol=1e-9):
        raise ValueError(f"{name} must be a proper orthonormal rotation matrix")


@dataclass(frozen=True)
class FrameState:
    """World-frame state of one Cartesian interface frame.

    ``orientation`` maps vectors from the interface frame to the world frame.
    Linear and angular velocities are expressed in world coordinates and are
    evaluated at the interface-frame origin.
    """

    position: Vector3
    orientation: Matrix3
    linear_velocity: Vector3
    angular_velocity: Vector3

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _readonly_array(self.position, (3,), "position"))
        orientation = _readonly_array(self.orientation, (3, 3), "orientation")
        _validate_rotation_matrix(orientation, "orientation")
        object.__setattr__(self, "orientation", orientation)
        object.__setattr__(
            self,
            "linear_velocity",
            _readonly_array(self.linear_velocity, (3,), "linear_velocity"),
        )
        object.__setattr__(
            self,
            "angular_velocity",
            _readonly_array(self.angular_velocity, (3,), "angular_velocity"),
        )


@dataclass(frozen=True)
class BushingParameters:
    """Diagonal translational and torsional spring-damper coefficients."""

    translation_stiffness: Vector3
    translation_damping: Vector3
    rotation_stiffness: Vector3
    rotation_damping: Vector3

    def __post_init__(self) -> None:
        for name in (
            "translation_stiffness",
            "translation_damping",
            "rotation_stiffness",
            "rotation_damping",
        ):
            value = _readonly_array(getattr(self, name), (3,), name)
            if np.any(value < 0.0):
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class InteractionState:
    """One bushing evaluation, with forces/moments expressed in world axes.

    ``human_moment`` is about the human-frame origin. ``exo_moment`` is the
    dynamically equivalent moment about the exoskeleton-frame origin.  The
    latter includes the moment-arm correction needed so that the two complete
    wrenches are equal and opposite when shifted to a common reference point.
    ``torsional_moment`` is the Eq. (5) rotational spring-damper term on the
    human side, about the human-frame origin.
    """

    relative_translation: Vector3
    relative_linear_velocity: Vector3
    relative_rotation: Vector3
    relative_angular_velocity: Vector3
    human_force: Vector3
    human_moment: Vector3
    exo_force: Vector3
    exo_moment: Vector3
    torsional_moment: Vector3
    reference_position: Vector3
    force_norm: float
    moment_norm: float

    def exo_moment_about_reference(self, exo_position: ArrayLike) -> Vector3:
        """Shift the exoskeleton-side moment to ``reference_position``."""
        position = _readonly_array(exo_position, (3,), "exo_position")
        return self.exo_moment + np.cross(position - self.reference_position, self.exo_force)


def _body_fixed_xyz_from_matrix(rotation: Matrix3) -> Vector3:
    """Return intrinsic x-y-z (body-fixed) Euler angles from ``rotation``.

    For angles ``(x, y, z)``, the matrix convention is ``Rx(x) Ry(y) Rz(z)``.
    This is the body-fixed x-y-z convention named in Nature Eq. (5).
    """
    sin_y = float(np.clip(rotation[0, 2], -1.0, 1.0))
    y = math.asin(sin_y)
    cos_y = math.cos(y)
    if abs(cos_y) < 1e-8:
        raise ValueError(
            "body-fixed x-y-z Euler representation is singular near |theta_y| = pi/2"
        )
    x = math.atan2(-float(rotation[1, 2]), float(rotation[2, 2]))
    z = math.atan2(-float(rotation[0, 1]), float(rotation[0, 0]))
    return np.array((x, y, z), dtype=np.float64)


def _unwrap_near(current: Vector3, previous: Vector3 | None) -> Vector3:
    if previous is None:
        return current
    delta = (current - previous + math.pi) % (2.0 * math.pi) - math.pi
    return previous + delta


def _body_fixed_xyz_rates(theta: Vector3, relative_omega_a: Vector3) -> Vector3:
    """Map relative spatial angular velocity in FRAME_A to x-y-z Euler rates."""
    x, y, _ = theta
    sin_x, cos_x = math.sin(float(x)), math.cos(float(x))
    cos_y = math.cos(float(y))
    if abs(cos_y) < 1e-8:
        raise ValueError(
            "body-fixed x-y-z Euler rates are singular near |theta_y| = pi/2"
        )
    omega_x, omega_y, omega_z = relative_omega_a
    z_rate = (-sin_x * omega_y + cos_x * omega_z) / cos_y
    y_rate = cos_x * omega_y + sin_x * omega_z
    x_rate = omega_x - math.sin(float(y)) * z_rate
    return np.array((x_rate, y_rate, z_rate), dtype=np.float64)


@dataclass
class LinearBushing:
    """Compute the restoring interaction wrench between two frame states.

    The object stores only the previous Euler representative so consecutive
    calls can unwrap the +/-pi branch without a torque discontinuity.  Call
    :meth:`reset` between independent trajectories.
    """

    parameters: BushingParameters
    _previous_rotation: Vector3 | None = field(default=None, init=False, repr=False)

    def reset(self) -> None:
        self._previous_rotation = None

    def compute(
        self,
        human_frame_state: FrameState,
        exo_frame_state: FrameState,
        dt: float,
    ) -> InteractionState:
        """Evaluate Nature Eq. (5) and return equal-and-opposite wrenches.

        ``dt`` is part of the stable M5/M3 contract and is validated here.  The
        current implementation obtains Euler rates analytically from the input
        angular velocities rather than by finite differencing ``dt``.
        """
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")

        rotation_a = human_frame_state.orientation
        displacement_world = exo_frame_state.position - human_frame_state.position
        relative_translation = rotation_a.T @ displacement_world

        omega_a_in_a = rotation_a.T @ human_frame_state.angular_velocity
        relative_linear_velocity = (
            rotation_a.T
            @ (exo_frame_state.linear_velocity - human_frame_state.linear_velocity)
            - np.cross(omega_a_in_a, relative_translation)
        )

        relative_orientation = rotation_a.T @ exo_frame_state.orientation
        wrapped_rotation = _body_fixed_xyz_from_matrix(relative_orientation)
        relative_rotation = _unwrap_near(wrapped_rotation, self._previous_rotation)
        self._previous_rotation = np.array(relative_rotation, copy=True)

        relative_omega_a = rotation_a.T @ (
            exo_frame_state.angular_velocity - human_frame_state.angular_velocity
        )
        relative_angular_velocity = _body_fixed_xyz_rates(
            relative_rotation, relative_omega_a
        )

        # Eq. (5), component-wise in FRAME_A.  Assigning the positive vectors
        # to A and their opposites to B makes both terms restoring for d = B-A.
        force_a = (
            self.parameters.translation_stiffness * relative_translation
            + self.parameters.translation_damping * relative_linear_velocity
        )
        moment_a = (
            self.parameters.rotation_stiffness * relative_rotation
            + self.parameters.rotation_damping * relative_angular_velocity
        )

        human_force = rotation_a @ force_a
        human_moment = rotation_a @ moment_a
        exo_force = -human_force

        # The B-origin moment is chosen so the B wrench shifted to the A origin
        # is exactly the negative of the A wrench:
        # M_B@A = M_B@B + (p_B-p_A) x F_B = -M_A@A.
        exo_moment = -human_moment + np.cross(displacement_world, human_force)

        arrays = (
            relative_translation,
            relative_linear_velocity,
            relative_rotation,
            relative_angular_velocity,
            human_force,
            human_moment,
            exo_force,
            exo_moment,
            human_moment,
            human_frame_state.position,
        )
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise FloatingPointError("linear bushing produced NaN or Inf")

        return InteractionState(
            relative_translation=_readonly_array(relative_translation, (3,), "relative_translation"),
            relative_linear_velocity=_readonly_array(
                relative_linear_velocity, (3,), "relative_linear_velocity"
            ),
            relative_rotation=_readonly_array(relative_rotation, (3,), "relative_rotation"),
            relative_angular_velocity=_readonly_array(
                relative_angular_velocity, (3,), "relative_angular_velocity"
            ),
            human_force=_readonly_array(human_force, (3,), "human_force"),
            human_moment=_readonly_array(human_moment, (3,), "human_moment"),
            exo_force=_readonly_array(exo_force, (3,), "exo_force"),
            exo_moment=_readonly_array(exo_moment, (3,), "exo_moment"),
            torsional_moment=_readonly_array(human_moment, (3,), "torsional_moment"),
            reference_position=_readonly_array(
                human_frame_state.position, (3,), "reference_position"
            ),
            force_norm=float(np.linalg.norm(human_force)),
            moment_norm=float(np.linalg.norm(human_moment)),
        )
