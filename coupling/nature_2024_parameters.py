"""Parameters transcribed from Luo et al., Nature 2024, Supplementary Table 1.

The source table reports numerical values and paper x/y/z components, but it
does not print units.  That omission is preserved explicitly below.  The
``si_dimensional_interpretation`` metadata states the dimensional units that
result when M5 receives metres, seconds, radians, newtons and newton-metres;
it is an implementation convention, not a claim that the table printed units.
"""

from __future__ import annotations

from dataclasses import dataclass

from .linear_bushing import BushingParameters


@dataclass(frozen=True)
class Nature2024ParameterSet:
    name: str
    strap_type: str
    parameters: BushingParameters
    reported_components: tuple[tuple[str, float], ...]
    source: str = "Luo et al., Nature 630, 353-359 (2024)"
    doi: str = "10.1038/s41586-024-07382-4"
    table: str = "Supplementary Table 1"
    axis_convention: str = "paper x/y/z Cartesian-frame axes"
    reported_units: str = "not stated in Supplementary Table 1 or the associated Methods text"
    si_dimensional_interpretation: tuple[tuple[str, str], ...] = (
        ("translation_stiffness", "N/m"),
        ("translation_damping", "N*s/m"),
        ("rotation_stiffness", "N*m/rad"),
        ("rotation_damping", "N*m*s/rad"),
    )


NATURE_2024_WAIST = Nature2024ParameterSet(
    name="NATURE_2024_WAIST",
    strap_type="human pelvis and exoskeleton waist belt",
    parameters=BushingParameters(
        translation_stiffness=(0.0, 8000.0, 0.0),
        translation_damping=(0.0, 10.0, 0.0),
        rotation_stiffness=(0.0, 0.0, 0.0),
        rotation_damping=(0.0, 0.0, 0.0),
    ),
    reported_components=(("k_y", 8000.0), ("c_y", 10.0)),
)


NATURE_2024_LEG_STRAP = Nature2024ParameterSet(
    name="NATURE_2024_LEG_STRAP",
    strap_type="human legs and exoskeleton leg straps",
    parameters=BushingParameters(
        translation_stiffness=(1500.0, 500.0, 1500.0),
        translation_damping=(10.0, 1.0, 10.0),
        rotation_stiffness=(10.0, 3.0, 10.0),
        rotation_damping=(1.0, 0.1, 1.0),
    ),
    reported_components=(
        ("k_x", 1500.0),
        ("k_y", 500.0),
        ("k_z", 1500.0),
        ("c_x", 10.0),
        ("c_y", 1.0),
        ("c_z", 10.0),
        ("alpha_x", 10.0),
        ("alpha_y", 3.0),
        ("alpha_z", 10.0),
        ("beta_x", 1.0),
        ("beta_y", 0.1),
        ("beta_z", 1.0),
    ),
)
