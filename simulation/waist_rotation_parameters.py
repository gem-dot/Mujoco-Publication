"""M5 extension parameters for finite waist sagittal rotation/AP translation.

These parameters are deliberately separate from ``NATURE_2024_EXACT``.
Nature's reported waist translation coefficients are copied without mutation;
only the otherwise-unreported local-z (sagittal) rotation coefficients are
filled by the M5 hip-knee extension.
"""

from __future__ import annotations

from dataclasses import dataclass

from coupling.linear_bushing import BushingParameters
from coupling.nature_2024_parameters import NATURE_2024_WAIST


WAIST_AP_AXIS = 0
WAIST_SAGITTAL_AXIS = 2


@dataclass(frozen=True)
class WaistRotationParameterSet:
    name: str
    alpha_pitch_nm_per_rad: float
    beta_pitch_nms_per_rad: float
    estimated_damping_ratio: float
    parameter_source: str


@dataclass(frozen=True)
class WaistTranslationParameterSet:
    name: str
    k_ap_n_per_m: float
    c_ap_ns_per_m: float
    estimated_damping_ratio: float
    parameter_source: str


def extended_waist_parameters(
    alpha_pitch_nm_per_rad: float,
    beta_pitch_nms_per_rad: float,
    k_ap_n_per_m: float = 0.0,
    c_ap_ns_per_m: float = 0.0,
) -> BushingParameters:
    """Copy Nature coefficients, then add M5 pitch/AP-only extensions."""
    alpha = float(alpha_pitch_nm_per_rad)
    beta = float(beta_pitch_nms_per_rad)
    k_ap = float(k_ap_n_per_m)
    c_ap = float(c_ap_ns_per_m)
    if min(alpha, beta, k_ap, c_ap) < 0.0:
        raise ValueError("Waist extension coefficients must be non-negative")

    nature = NATURE_2024_WAIST.parameters
    translation_stiffness = list(nature.translation_stiffness)
    translation_damping = list(nature.translation_damping)
    rotation_stiffness = list(nature.rotation_stiffness)
    rotation_damping = list(nature.rotation_damping)
    translation_stiffness[WAIST_AP_AXIS] = k_ap
    translation_damping[WAIST_AP_AXIS] = c_ap
    rotation_stiffness[WAIST_SAGITTAL_AXIS] = alpha
    rotation_damping[WAIST_SAGITTAL_AXIS] = beta
    return BushingParameters(
        translation_stiffness=translation_stiffness,
        translation_damping=translation_damping,
        rotation_stiffness=rotation_stiffness,
        rotation_damping=rotation_damping,
    )


OUR_HIP_KNEE_WAIST_ROTATION_V1 = WaistRotationParameterSet(
    name="OUR_HIP_KNEE_WAIST_ROTATION_V1",
    alpha_pitch_nm_per_rad=150.0,
    beta_pitch_nms_per_rad=12.0,
    estimated_damping_ratio=2.245274557797055,
    parameter_source=(
        "engineering calibration based on Case C free-rotation instability + "
        "Case D diagnostic equality reaction + M5.6 finite-compliance "
        "sensitivity study"
    ),
)


OUR_HIP_KNEE_WAIST_ROTATION_V1_1 = WaistRotationParameterSet(
    name="OUR_HIP_KNEE_WAIST_ROTATION_V1_1",
    alpha_pitch_nm_per_rad=165.0,
    beta_pitch_nms_per_rad=13.5,
    estimated_damping_ratio=2.408347457258957,
    parameter_source=(
        "small local tightening around M5.6 V1; selected from V1, stiffer, "
        "more-damped, and stiffer-plus-more-damped 2 s comparisons"
    ),
)


OUR_HIP_KNEE_WAIST_TRANSLATION_V1 = WaistTranslationParameterSet(
    name="OUR_HIP_KNEE_WAIST_TRANSLATION_V1",
    k_ap_n_per_m=1500.0,
    c_ap_ns_per_m=60.0,
    estimated_damping_ratio=0.34924622713899245,
    parameter_source=(
        "M5 Case-E relative-AP-lock diagnosis plus three bounded 2 s finite "
        "candidates; LOW retained finite compliance and was promoted after the "
        "initial MEDIUM selection failed the 5 s pelvis-height gate"
    ),
)
