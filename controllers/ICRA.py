"""ICRA / IS-CGF staged MuJoCo controller.

Stage 0/1 shadow contracts remain available.  Stage 2R provides the verified
left-side Fixed-K ADP physical baseline.  Stage 3 adds an observational dual
endpoint and intention-scheduled game model.  Stage 4 adds normalized costs,
PPC/predefined-time signals, and future-Critic geometry.  Both shadow stages
leave physical motor commands exactly sourced from the Stage-2R baseline.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from controllers.YAC import HealthyGaitReference, PlanarFloatingBaseExoModel


SHADOW_CONTRACT_MODE = "shadow_contract"
SHADOW_PLANT_SCHEDULER_MODE = "shadow_plant_scheduler"
FIXED_K_SHADOW_MODE = "fixed_k_shadow"
FIXED_K_NOMINAL_MODE = "fixed_k_nominal"
FIXED_K_ADP_MODE = "fixed_k_adp"
GAME_SHADOW_MODE = "game_shadow"
PERFORMANCE_SHADOW_MODE = "performance_shadow"
ICRA_IMPEDANCE_BASELINE_MODE = "icra_impedance_baseline"
CRITIC_NASH_SHADOW_MODE = "critic_nash_shadow"
CONTINUOUS_ACTIVE_GAME_MODE = "continuous_active_game"
CALIBRATED_SCALAR_GAME_SHADOW_MODE = "calibrated_scalar_game_shadow"
PHYSICAL_ACTUATION_GAME_SHADOW_MODE = "physical_actuation_game_shadow"
PHYSICAL_ACTUATION_GAME_ACTIVE_MODE = "physical_actuation_game_active"
INTENTION_TARGET_GAME_SHADOW_MODE = "intention_target_game_shadow"
INTENTION_TARGET_GAME_ACTIVE_MODE = "intention_target_game_active"
EVENT_TRIGGERED_INTENTION_TARGET_GAME_ACTIVE_MODE = (
    "event_triggered_intention_target_game_active"
)
CONSERVATIVE_EVENT_TRIGGERED_INTENTION_TARGET_GAME_ACTIVE_MODE = (
    "conservative_event_triggered_intention_target_game_active"
)
DWELL_LIMITED_P25_EVENT_TRIGGERED_INTENTION_TARGET_GAME_ACTIVE_MODE = (
    "dwell_limited_p25_event_triggered_intention_target_game_active"
)
RESIDUAL_EVENT_TRIGGERED_INTENTION_TARGET_GAME_ACTIVE_MODE = (
    "residual_event_triggered_intention_target_game_active"
)
PPC_TIMING_CHARACTERIZATION_MODE = "ppc_timing_characterization"
PT_HORIZON_CHARACTERIZATION_MODE = "pt_horizon_characterization"
STAGE2_MODES = (FIXED_K_SHADOW_MODE, FIXED_K_NOMINAL_MODE, FIXED_K_ADP_MODE)
STAGE3_MODES = (GAME_SHADOW_MODE,)
STAGE4_MODES = (PERFORMANCE_SHADOW_MODE,)
STAGE5_MODES = (ICRA_IMPEDANCE_BASELINE_MODE,)
STAGE6A_MODES = (CRITIC_NASH_SHADOW_MODE,)
STAGE6B_MODES = (CONTINUOUS_ACTIVE_GAME_MODE,)
STAGE6E_MODES = (CALIBRATED_SCALAR_GAME_SHADOW_MODE,)
STAGE6F_MODES = (
    PHYSICAL_ACTUATION_GAME_SHADOW_MODE,
    PHYSICAL_ACTUATION_GAME_ACTIVE_MODE,
)
STAGE6G_MODES = (
    INTENTION_TARGET_GAME_SHADOW_MODE,
    INTENTION_TARGET_GAME_ACTIVE_MODE,
)
STAGE7_MODES = (EVENT_TRIGGERED_INTENTION_TARGET_GAME_ACTIVE_MODE,)
STAGE7R_MODES = (CONSERVATIVE_EVENT_TRIGGERED_INTENTION_TARGET_GAME_ACTIVE_MODE,)
STAGE7S_MODES = (DWELL_LIMITED_P25_EVENT_TRIGGERED_INTENTION_TARGET_GAME_ACTIVE_MODE,)
EVENT_EXECUTION_MODES = (*STAGE7_MODES, *STAGE7R_MODES, *STAGE7S_MODES)
STAGE7T_MODES = (RESIDUAL_EVENT_TRIGGERED_INTENTION_TARGET_GAME_ACTIVE_MODE,)
STAGE8A_MODES = (PPC_TIMING_CHARACTERIZATION_MODE, PT_HORIZON_CHARACTERIZATION_MODE)
FIXED_K_FAMILY_MODES = (*STAGE2_MODES, *STAGE3_MODES, *STAGE4_MODES)
FIXED_K_ADP_PHYSICAL_MODES = (
    FIXED_K_ADP_MODE,
    GAME_SHADOW_MODE,
    PERFORMANCE_SHADOW_MODE,
)
GAME_DIAGNOSTIC_MODES = (
    *STAGE3_MODES, *STAGE4_MODES, *STAGE5_MODES, *STAGE6A_MODES,
    *STAGE6B_MODES,
    *STAGE6E_MODES,
    *STAGE6F_MODES,
    *STAGE6G_MODES,
    *EVENT_EXECUTION_MODES,
    *STAGE7T_MODES,
    *STAGE8A_MODES,
)
CONTROL_DT_S = 0.01
HEEL_STRIKE_REFRACTORY_S = 0.6
MIN_STRIDE_PERIOD_S = 0.6
MAX_STRIDE_PERIOD_S = 2.0
PHASE_EWMA_BETA = 0.2
ORACLE_SAMPLE_STRIDE = 2
ORACLE_DELTAS_NM = (0.1, 0.5, 1.0)
ORACLE_TEST_TORQUES_NM = (
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (1.0, 1.0),
    (-1.0, 1.0),
)

EXO_TORQUE_CHANNELS = (
    "HipExo_R",
    "HipExo_L",
    "KneeExo_R",
    "KneeExo_L",
    "Exo_R",
    "Exo_L",
)
ZERO_EXO_TORQUES = {name: 0.0 for name in EXO_TORQUE_CHANNELS}

COORDINATE_ORDER = (
    "exo_root_forward",
    "exo_root_vertical",
    "exo_root_pitch",
    "exo_hip_l",
    "exo_knee_l",
    "exo_hip_r",
    "exo_knee_r",
)
CONTROLLED_INDICES = (3, 4)
UNACTUATED_INDICES = (0, 1, 2, 5, 6)
CONTROLLED_COORDINATES = tuple(COORDINATE_ORDER[index] for index in CONTROLLED_INDICES)
UNACTUATED_COORDINATES = tuple(
    COORDINATE_ORDER[index] for index in UNACTUATED_INDICES
)

LEFT_ACTUATORS = ("HipExo_L", "KneeExo_L")
RIGHT_ACTUATORS = ("HipExo_R", "KneeExo_R")
LEFT_INTENTION_MUSCLES = (
    "hamstrings_l",
    "bifemsh_l",
    "glutmax_l",
    "iliopsoas_l",
    "rectfem_l",
    "vasti_l",
)

# [PLATFORM ADAPTATION] data.act is already an activation state, so the
# derivative and integral terms in Paper Eq. 36--37 are retained structurally
# but have zero coefficient in the first MuJoCo contract.
BETA_1 = 1.0
BETA_2 = 0.0
BETA_3 = 0.0
ACTIVATION_EXPONENT_A = 1.0
ALPHA_LOWER = 0.05
ALPHA_UPPER = 0.95
ALPHA_TIME_CONSTANT_S = 0.05
ALPHA_ZOH_DECAY_100HZ = math.exp(-CONTROL_DT_S / ALPHA_TIME_CONSTANT_S)

DESIRED_INERTIA_KGM2 = np.asarray((0.60, 0.25), dtype=float)
STIFFNESS_MAX_NM_PER_RAD = np.asarray((55.0, 45.0), dtype=float)
FIXED_K_DAMPING_RATIO = 1.1
FIXED_K_DAMPING_NMS_PER_RAD = (
    2.0
    * FIXED_K_DAMPING_RATIO
    * np.sqrt(DESIRED_INERTIA_KGM2 * STIFFNESS_MAX_NM_PER_RAD)
)

# Phase-normalized bilateral means from the frozen Full-YAC trajectory.
# See results/audits/ICRA_stage0_contract/icra_stage0_contract.json.
FIXED_K_HIP_NM_PER_RAD = 23.179053896054274
FIXED_K_KNEE_NM_PER_RAD = 19.949467960723577
FIXED_K_NM_PER_RAD = np.asarray(
    (FIXED_K_HIP_NM_PER_RAD, FIXED_K_KNEE_NM_PER_RAD), dtype=float
)

# [PLATFORM BASELINE CONTINUITY] Fixed-K desired-impedance factorization.
# Values are calculated from the locked physical parameters, not rounded roots.
FIXED_D_S_INV = FIXED_K_DAMPING_NMS_PER_RAD / DESIRED_INERTIA_KGM2
FIXED_K_S_INV2 = FIXED_K_NM_PER_RAD / DESIRED_INERTIA_KGM2
FIXED_DISCRIMINANT_S_INV2 = FIXED_D_S_INV**2 - 4.0 * FIXED_K_S_INV2
if np.any(FIXED_DISCRIMINANT_S_INV2 <= 0.0):
    raise RuntimeError("Fixed-K impedance factorization requires positive roots")
FIXED_LAMBDA_S_INV = 0.5 * (
    FIXED_D_S_INV - np.sqrt(FIXED_DISCRIMINANT_S_INV2)
)
FIXED_GAMMA_S_INV = FIXED_D_S_INV - FIXED_LAMBDA_S_INV

# Stage-2 hard-KKT mean is retained only for recovery comparison.
FIXED_MBAR_KKT_KGM2 = np.asarray(
    (
        (0.06685506786602632, 0.006741692733158557),
        (0.006741692733158556, 0.02562358541615104),
    ),
    dtype=float,
)
# Arithmetic mean of all 17,301 states in the 15 retained Healthy reference
# cycles under the intrinsic smooth seven-coordinate Schur reduction.
FIXED_MBAR_SMOOTH_KGM2 = np.asarray(
    (
        (0.028299090288834657, 0.0064736128639314855),
        (0.0064736128639314855, 0.0057876651844901948),
    ),
    dtype=float,
)
FIXED_MBAR_EFF_KGM2 = FIXED_MBAR_SMOOTH_KGM2
FIXED_Z_DAMPING_RATE_S_INV = 5.0
FIXED_KEN_NMS_PER_RAD = FIXED_Z_DAMPING_RATE_S_INV * FIXED_MBAR_EFF_KGM2
FIXED_ENGAGEMENT_RAMP_S = 0.25

# Stage-2R smooth-HRI Healthy 20 s Fixed-K shadow, reference-valid samples
# after 2 s, pooled over 15 complete left strides: joint-wise p95(abs(z)).
# Source run: 20260904_160356__ICRA_STAGE2R_SMOOTH_SHADOW_HEALTHY_20S
FIXED_ADP_Z_SCALE_RAD_S = np.asarray(
    (2.3591666472203241, 3.7899765249265087), dtype=float
)
FIXED_ADP_Q = np.diag(1.0 / FIXED_ADP_Z_SCALE_RAD_S**2)
FIXED_ADP_R = np.diag((1.0 / 25.0, 1.0 / 4.0))
FIXED_ADP_R_INV = np.diag((25.0, 4.0))
FIXED_ADP_KAPPA = 1.0
FIXED_VALUE_EIGEN_TOLERANCE = 1.0e-10
FIXED_MAX_BACKTRACKS = 7
NATIVE_MOTOR_LIMIT_NM = 100.0

# Stage-3 dual endpoints.  The common endpoint separation is the geometric
# mean of the frozen-YAC-to-Fixed-K stiffness ratios; it is a derivation, not a
# tuned parameter.
GAME_GAIT_PERIOD_S = 1.15
GAME_LOADING_FRACTION = 0.20
GAME_LOADING_DURATION_S = GAME_LOADING_FRACTION * GAME_GAIT_PERIOD_S
GAME_CRITICAL_X90 = 3.88972016986743
GAME_OMEGA_A_S_INV = GAME_CRITICAL_X90 / GAME_LOADING_DURATION_S
GAME_OMEGA_G_S_INV = 2.0 * math.pi / GAME_GAIT_PERIOD_S
GAME_ENDPOINT_RATIO = math.sqrt(
    (STIFFNESS_MAX_NM_PER_RAD[0] / FIXED_K_NM_PER_RAD[0])
    * (STIFFNESS_MAX_NM_PER_RAD[1] / FIXED_K_NM_PER_RAD[1])
)
GAME_KA_NM_PER_RAD = GAME_ENDPOINT_RATIO * FIXED_K_NM_PER_RAD
GAME_KI_NM_PER_RAD = FIXED_K_NM_PER_RAD / GAME_ENDPOINT_RATIO
GAME_MA_KGM2 = GAME_KA_NM_PER_RAD / GAME_OMEGA_A_S_INV**2
GAME_DA_NMS_PER_RAD = 2.0 * GAME_OMEGA_A_S_INV * GAME_MA_KGM2
# Stage-5R sampled-data recovery.  The old diagonal damping is retained only
# as an audit reference; it is superseded for analytical and physical use.
GAME_DI_OLD_NMS_PER_RAD = np.diag(
    (
        1.9883637461605799,
        1.6098695414701274,
    )
)
GAME_I_MODAL_DAMPING_RATIO = 0.90
_game_mbar_eigenvalues, _game_mbar_eigenvectors = np.linalg.eigh(
    FIXED_MBAR_SMOOTH_KGM2
)
GAME_I_MHALF = (
    _game_mbar_eigenvectors
    * np.sqrt(_game_mbar_eigenvalues)
) @ _game_mbar_eigenvectors.T
GAME_I_MMINUS_HALF = (
    _game_mbar_eigenvectors
    / np.sqrt(_game_mbar_eigenvalues)
) @ _game_mbar_eigenvectors.T
GAME_I_OMEGA2 = (
    GAME_I_MMINUS_HALF
    @ np.diag(GAME_KI_NM_PER_RAD)
    @ GAME_I_MMINUS_HALF
)
_game_omega_eigenvalues, _game_omega_eigenvectors = np.linalg.eigh(
    GAME_I_OMEGA2
)
GAME_I_OMEGA = (
    _game_omega_eigenvectors
    * np.sqrt(_game_omega_eigenvalues)
) @ _game_omega_eigenvectors.T
GAME_DI_NMS_PER_RAD = (
    2.0
    * GAME_I_MODAL_DAMPING_RATIO
    * GAME_I_MHALF
    @ GAME_I_OMEGA
    @ GAME_I_MHALF
)
GAME_DI_NMS_PER_RAD = 0.5 * (
    GAME_DI_NMS_PER_RAD + GAME_DI_NMS_PER_RAD.T
)
GAME_K0_NM_PER_RAD = FIXED_K_NM_PER_RAD.copy()
GAME_RA = np.diag((0.004, 0.025))
GAME_RI = np.diag((0.08, 0.50))
GAME_HMU = np.diag(np.sqrt(np.diag(GAME_RA) * np.diag(GAME_RI)))
GAME_ETA0 = 0.5
GAME_ETA1 = 1.0

# Stage-4 robust block-isotropic normalizers.  These are pooled p95 scales from
# the locked Stage-3 Healthy/Moderate complete-cycle audit, not online gains.
PERFORMANCE_S_A_POS = 0.03858775153460374
PERFORMANCE_S_A_VEL = 0.4233737468613045
PERFORMANCE_S_I_POS = 0.2447821925117854
PERFORMANCE_S_I_VEL = 4.138674097370841
PERFORMANCE_S_EC_POS = 0.24125533731500468
PERFORMANCE_S_EC_VEL = 4.060547373463469
PERFORMANCE_DA = np.diag(
    (PERFORMANCE_S_A_POS, PERFORMANCE_S_A_POS,
     PERFORMANCE_S_A_VEL, PERFORMANCE_S_A_VEL)
)
PERFORMANCE_DI = np.diag(
    (PERFORMANCE_S_I_POS, PERFORMANCE_S_I_POS,
     PERFORMANCE_S_I_VEL, PERFORMANCE_S_I_VEL)
)
PERFORMANCE_DC = np.diag(
    (PERFORMANCE_S_EC_POS, PERFORMANCE_S_EC_POS,
     PERFORMANCE_S_EC_VEL, PERFORMANCE_S_EC_VEL)
)
PERFORMANCE_QBAR_A = np.diag((3.5, 3.5, 1.0, 1.0))
PERFORMANCE_QBAR_I = np.diag((2.0, 2.0, 1.0, 1.0))
PERFORMANCE_QA = np.diag(
    np.diag(PERFORMANCE_QBAR_A) / np.diag(PERFORMANCE_DA) ** 2
)
PERFORMANCE_QI = np.diag(
    np.diag(PERFORMANCE_QBAR_I) / np.diag(PERFORMANCE_DI) ** 2
)
PERFORMANCE_SA = np.diag(1.0 / np.diag(PERFORMANCE_DC) ** 2)
PERFORMANCE_SI = PERFORMANCE_SA.copy()
PERFORMANCE_LAMBDA_A = 0.25 * float(np.mean(np.diag(PERFORMANCE_QBAR_A)))
PERFORMANCE_LAMBDA_I = 0.25 * float(np.mean(np.diag(PERFORMANCE_QBAR_I)))
PERFORMANCE_C_P_RAD_PER_NM = 0.048

PPC_RHO0_DEV = 2.40
PPC_RHO_INF_DEV = 2.20
PPC_RHO0_TIGHT_CANDIDATE = 0.60
PPC_RHO_INF_TIGHT_CANDIDATE = 0.50
# Backward-compatible names denote the currently active development envelope.
PPC_RHO0 = PPC_RHO0_DEV
PPC_RHO_INF = PPC_RHO_INF_DEV
PPC_T_RHO_S = 2.30
PPC_NU = 2.0
PPC_DIRECT_INPUT_IN_FIRST_DERIVATIVE = False

PT_T_P_S = 3.45
PT_DELTA = 0.10
PT_P_S_INV = 1.0 / 1.15

if not np.allclose(
    np.sqrt(GAME_KA_NM_PER_RAD * GAME_KI_NM_PER_RAD),
    FIXED_K_NM_PER_RAD,
    rtol=1.0e-12,
    atol=0.0,
):
    raise RuntimeError("Stage-3 endpoint geometric-center derivation failed")
if not np.allclose(
    GAME_KA_NM_PER_RAD / GAME_MA_KGM2,
    GAME_OMEGA_A_S_INV**2,
    rtol=1.0e-14,
    atol=0.0,
) or not np.allclose(
    GAME_DA_NMS_PER_RAD / GAME_MA_KGM2,
    2.0 * GAME_OMEGA_A_S_INV,
    rtol=1.0e-14,
    atol=0.0,
):
    raise RuntimeError("Stage-3 critical-admittance derivation failed")
if not np.allclose(
    GAME_DI_NMS_PER_RAD,
    GAME_DI_NMS_PER_RAD.T,
    rtol=0.0,
    atol=1.0e-14,
) or np.min(np.linalg.eigvalsh(GAME_DI_NMS_PER_RAD)) <= 0.0:
    raise RuntimeError("Stage-5R impedance damping must be symmetric positive definite")

ACTION_SCALE_NM = np.asarray((5.0, 2.0), dtype=float)
R_A_NORMALIZED = 0.1 * np.eye(2, dtype=float)
R_I_NORMALIZED = 2.0 * np.eye(2, dtype=float)
ACTION_SCALE_INVERSE = np.diag(1.0 / ACTION_SCALE_NM)
R_A_PHYSICAL = ACTION_SCALE_INVERSE.T @ R_A_NORMALIZED @ ACTION_SCALE_INVERSE
R_I_PHYSICAL = ACTION_SCALE_INVERSE.T @ R_I_NORMALIZED @ ACTION_SCALE_INVERSE

CHI_DIMENSION = 12
CRITIC_FEATURE_COUNT = CHI_DIMENSION * (CHI_DIMENSION + 1) // 2
CRITIC_WEIGHT_SHAPE = (CRITIC_FEATURE_COUNT, 2)

# Stage-6A coordinates are locked from Stage-5R complete-cycle calibration.
STAGE6A_CHI_ORDER = (
    "ZA_hip_position", "ZA_knee_position",
    "ZA_hip_velocity", "ZA_knee_velocity",
    "ZI_hip_position", "ZI_knee_position",
    "ZI_hip_velocity", "ZI_knee_velocity",
    "epsilon_track_hip", "epsilon_track_knee",
    "epsilon_int_hip", "epsilon_int_knee",
)
STAGE6A_D_CHI_DIAGONAL = np.asarray(
    (
        0.0468633048914, 0.0468633048914,
        0.488526929055, 0.488526929055,
        0.199772901089, 0.199772901089,
        3.17281678936, 3.17281678936,
        0.143612797730, 0.143612797730,
        0.138624223405, 0.138624223405,
    ),
    dtype=float,
)
STAGE6A_D_CHI = np.diag(STAGE6A_D_CHI_DIAGONAL)
STAGE6A_D_CHI_INVERSE = np.diag(1.0 / STAGE6A_D_CHI_DIAGONAL)
STAGE6A_S_EPS_TRACK = 0.143612797730
STAGE6A_S_EPS_INT = 0.138624223405
STAGE6A_D_EPS = np.diag(
    (STAGE6A_S_EPS_TRACK, STAGE6A_S_EPS_TRACK,
     STAGE6A_S_EPS_INT, STAGE6A_S_EPS_INT)
)
STAGE6A_QBAR_EPS = 0.25 * np.eye(4, dtype=float)
STAGE6A_Q_EPS = (
    np.linalg.inv(STAGE6A_D_EPS).T
    @ STAGE6A_QBAR_EPS
    @ np.linalg.inv(STAGE6A_D_EPS)
)
STAGE6A_DISCOUNT_RATE_S_INV = 1.0 / PT_T_P_S
STAGE6A_GAMMA_C = np.eye(CRITIC_FEATURE_COUNT, dtype=float)
STAGE6A_WEIGHT_SAFETY_FACTOR = 0.50
STAGE6A_MUA_BUDGET_NM = 2.0 / np.sqrt(np.diag(GAME_RA))
STAGE6A_MUI_BUDGET_NM = 2.0 / np.sqrt(np.diag(GAME_RI))
STAGE6B_CRITIC_ETA = 0.05
STAGE6B_WEIGHT_RADIUS = 3.66457347596e-4
STAGE6B_MUI_ABORT_BUDGET_NM = np.asarray(
    (7.0710678, 2.8284271), dtype=float
)
STAGE6B_RAW_TOTAL_ABORT_NM = 20.0
STAGE7_STRATEGY_SCALE_NM = np.asarray(
    (1.2107171177701537, 0.4557594590456351,
     1.2570709247766185, 0.4852210652602563), dtype=float
)
STAGE7_THETA0 = 0.03785510066288972
STAGE7_DWELL_S = 0.02
STAGE7_LAMBDA_V_S_INV = 1.0 / 3.45
STAGE7_MEDIAN_XCHI_NORM_SQ = 1.041452234785662
STAGE7_KAPPA_E = 0.009087094779398164
STAGE7_C_CHI_EVT = 0.0026339405157675844
STAGE7R_THETA0 = 0.003704956407942091
STAGE7R_DWELL_S = 0.02
STAGE7R_LAMBDA_V_S_INV = 0.2898550724637681
STAGE7R_KAPPA_E = 0.000889372619355183
STAGE7R_C_CHI_EVT = 0.0002577891650304878
STAGE7S_THETA0 = 3.649642896138322e-06
STAGE7S_DWELL_S = 0.02
STAGE7S_LAMBDA_V_S_INV = 0.2898550724637681
STAGE7S_KAPPA_E = (
    0.25 * STAGE7S_THETA0 / (STAGE7_MEDIAN_XCHI_NORM_SQ + 1.0e-12)
)
STAGE7S_C_CHI_EVT = STAGE7S_LAMBDA_V_S_INV * STAGE7S_KAPPA_E
STAGE7T_RESIDUAL_SCALE_NM = np.asarray(
    (0.014950184223183957, 0.0024861161589295063,
     0.00364147970186807, 0.0027713121094356035), dtype=float
)
STAGE7T_THETA0 = 0.0003968219500891493
STAGE7T_DWELL_S = 0.02
STAGE7T_LAMBDA_V_S_INV = 1.0 / 3.45
STAGE7T_MEDIAN_XCHI_NORM_SQ = 1.041452234785662
STAGE7T_KAPPA_E = (
    0.25 * STAGE7T_THETA0 / (STAGE7T_MEDIAN_XCHI_NORM_SQ + 1.0e-12)
)
STAGE7T_C_CHI_EVT = STAGE7T_LAMBDA_V_S_INV * STAGE7T_KAPPA_E

# Stage-6E controller-independent Healthy ExoZero scalar fit.  These are the
# locked Stage-6D source-only feasibility result, not a pooled or online fit.
STAGE6E_THRESHOLD_LOW = 0.4000429561888558
STAGE6E_THRESHOLD_HIGH = 0.45966700879236977
STAGE6E_POOLED_SENSITIVITY_LOW = 0.3940429561888558
STAGE6E_POOLED_SENSITIVITY_HIGH = 0.46341700879236974

# [PLATFORM INTENTION-RANGE CALIBRATION] Stage-6C derives this single robust
# operating range from the retained complete left gait cycles in the frozen,
# controller-independent Healthy ExoZero result.  The percentile levels and
# resulting range are locked for every subsequent condition; the legacy
# scheduler above remains unchanged.
ACTIVATION_RANGE_CALIBRATED_SCHEDULER_MODE = "activation_range_calibrated"
STAGE6C_CALIBRATION_SOURCE = (
    "results/experiments/"
    "20260822_214246__HUMAN_EXO_HEALTHY100_ZERO"
)
STAGE6C_S_LOW = 0.08174746117299456
STAGE6C_S_HIGH = 0.4942108537428857


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"{name} must be finite")
    return result


def intention_scheduler_static(activation: Sequence[float]) -> dict[str, float]:
    """Evaluate locked Eq. 35--39 scheduler terms without filter state."""
    values = np.asarray(tuple(activation), dtype=float)
    if values.shape != (len(LEFT_INTENTION_MUSCLES),):
        raise ValueError(
            f"Expected {len(LEFT_INTENTION_MUSCLES)} intention activations"
        )
    if not np.isfinite(values).all():
        raise FloatingPointError("Intention activation contains NaN or Inf")
    normalized = np.clip(values, 0.0, 1.0) / math.sqrt(values.size)
    s_bar = float(np.sqrt(normalized @ normalized))
    s = float(np.clip(s_bar, 0.0, 1.0))
    mu_act = BETA_1 * s + BETA_2 * 0.0 + BETA_3 * 0.0
    a = float(
        np.expm1(ACTIVATION_EXPONENT_A * mu_act)
        / np.expm1(ACTIVATION_EXPONENT_A)
    )
    alpha_bar = float(np.clip(1.0 - a, ALPHA_LOWER, ALPHA_UPPER))
    return {
        "s": s,
        "mu_act": mu_act,
        "a": a,
        "alpha_bar": alpha_bar,
    }


def intention_scheduler_range_calibrated_static(
    activation: Sequence[float],
    *,
    s_low: float = STAGE6C_S_LOW,
    s_high: float = STAGE6C_S_HIGH,
) -> dict[str, float]:
    """Evaluate the Stage-6C activation-only calibrated scheduler.

    No gait phase, contact, kinematics, or controller signal enters this pure
    mapping.  ``s_cal`` is a normalized simulation-activation proxy, not EMG.
    """
    values = np.asarray(tuple(activation), dtype=float)
    if values.shape != (len(LEFT_INTENTION_MUSCLES),):
        raise ValueError(
            f"Expected {len(LEFT_INTENTION_MUSCLES)} intention activations"
        )
    if not np.isfinite(values).all():
        raise FloatingPointError("Intention activation contains NaN or Inf")
    low = _finite_float(s_low, "s_low")
    high = _finite_float(s_high, "s_high")
    if not 0.0 <= low < high <= 1.0:
        raise ValueError("Stage-6C calibration requires 0 <= s_low < s_high <= 1")
    normalized = np.clip(values, 0.0, 1.0) / math.sqrt(values.size)
    s_raw = float(np.sqrt(normalized @ normalized))
    s_cal = float(np.clip((s_raw - low) / (high - low), 0.0, 1.0))
    a_cal = float(
        np.expm1(ACTIVATION_EXPONENT_A * s_cal)
        / np.expm1(ACTIVATION_EXPONENT_A)
    )
    alpha_bar_cal = float(
        np.clip(1.0 - a_cal, ALPHA_LOWER, ALPHA_UPPER)
    )
    return {
        "s_raw": s_raw,
        "s_cal": s_cal,
        "mu_act": s_cal,
        "a": a_cal,
        "alpha_bar": alpha_bar_cal,
    }


def intention_scheduler_three_segment_static(
    activation: Sequence[float],
    *,
    threshold_low: float = STAGE6E_THRESHOLD_LOW,
    threshold_high: float = STAGE6E_THRESHOLD_HIGH,
) -> dict[str, float]:
    """Evaluate the locked Stage-6E source-only three-segment scheduler."""
    values = np.asarray(tuple(activation), dtype=float)
    if values.shape != (len(LEFT_INTENTION_MUSCLES),):
        raise ValueError(
            f"Expected {len(LEFT_INTENTION_MUSCLES)} intention activations"
        )
    if not np.isfinite(values).all():
        raise FloatingPointError("Intention activation contains NaN or Inf")
    low = _finite_float(threshold_low, "threshold_low")
    high = _finite_float(threshold_high, "threshold_high")
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(
            "Stage-6E calibration requires 0 <= threshold_low "
            "< threshold_high <= 1"
        )
    clipped = np.clip(values, 0.0, 1.0)
    s_raw = float(np.sqrt(np.mean(clipped**2)))
    if s_raw <= low:
        alpha_target = ALPHA_UPPER
    elif s_raw >= high:
        alpha_target = ALPHA_LOWER
    else:
        alpha_target = ALPHA_UPPER - (
            (ALPHA_UPPER - ALPHA_LOWER)
            * (s_raw - low)
            / (high - low)
        )
    return {
        "s": s_raw,
        "s_raw": s_raw,
        "mu_act": s_raw,
        "a": 1.0 - float(alpha_target),
        "alpha_bar": float(alpha_target),
    }


def alpha_exact_zoh_step(alpha: float, alpha_bar: float, dt: float) -> float:
    """Advance Paper Eq. 40 with exact zero-order hold discretization."""
    current = _finite_float(alpha, "alpha")
    target = _finite_float(alpha_bar, "alpha_bar")
    step = _finite_float(dt, "dt")
    if step <= 0.0:
        raise ValueError("dt must be positive")
    if not ALPHA_LOWER <= current <= ALPHA_UPPER:
        raise ValueError("alpha is outside the locked bounds")
    if not ALPHA_LOWER <= target <= ALPHA_UPPER:
        raise ValueError("alpha_bar is outside the locked bounds")
    decay = math.exp(-step / ALPHA_TIME_CONSTANT_S)
    result = decay * current + (1.0 - decay) * target
    if not ALPHA_LOWER <= result <= ALPHA_UPPER:
        raise RuntimeError("Exact-ZOH scheduler left the invariant alpha interval")
    return float(result)


def game_eta(alpha: float) -> float:
    """Return the locked intention-dependent strategy-coupling strength."""
    value = _finite_float(alpha, "game alpha")
    if not ALPHA_LOWER <= value <= ALPHA_UPPER:
        raise ValueError("game alpha is outside the locked scheduler bounds")
    return float(GAME_ETA0 + 4.0 * GAME_ETA1 * value * (1.0 - value))


def game_nash_matrix(alpha: float) -> np.ndarray:
    """Build the Stage-3 Nash matrix in [muA_H,muA_K,muI_H,muI_K] order."""
    value = _finite_float(alpha, "game alpha")
    eta = game_eta(value)
    matrix = np.block(
        [
            [value * GAME_RA + eta * GAME_HMU, -eta * GAME_HMU],
            [-eta * GAME_HMU, (1.0 - value) * GAME_RI + eta * GAME_HMU],
        ]
    )
    if not np.isfinite(matrix).all() or not np.allclose(
        matrix, matrix.T, rtol=0.0, atol=1.0e-15
    ):
        raise RuntimeError("Stage-3 Nash matrix is non-finite or asymmetric")
    return matrix


def admittance_exact_zoh_step(
    error: Sequence[float],
    error_velocity: Sequence[float],
    interaction_nm: Sequence[float],
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance the critically damped A endpoint under constant interaction."""
    e = np.asarray(tuple(error), dtype=float).reshape(2)
    velocity = np.asarray(tuple(error_velocity), dtype=float).reshape(2)
    interaction = np.asarray(tuple(interaction_nm), dtype=float).reshape(2)
    step = _finite_float(dt, "admittance dt")
    if step <= 0.0:
        raise ValueError("admittance dt must be positive")
    if not np.isfinite(np.concatenate((e, velocity, interaction))).all():
        raise FloatingPointError("Stage-3 admittance state is non-finite")
    steady = interaction / GAME_KA_NM_PER_RAD
    shifted = e - steady
    decay = math.exp(-GAME_OMEGA_A_S_INV * step)
    next_shifted = decay * (
        (1.0 + GAME_OMEGA_A_S_INV * step) * shifted
        + step * velocity
    )
    next_velocity = decay * (
        (1.0 - GAME_OMEGA_A_S_INV * step) * velocity
        - GAME_OMEGA_A_S_INV**2 * step * shifted
    )
    next_error = next_shifted + steady
    if not np.isfinite(np.concatenate((next_error, next_velocity))).all():
        raise FloatingPointError("Stage-3 admittance update is non-finite")
    return next_error, next_velocity


def symmetric_quadratic_feature_count(dimension: int) -> int:
    """Count unique ``chi_i * chi_j`` terms for ``i <= j``."""
    value = int(dimension)
    if value <= 0:
        raise ValueError("dimension must be positive")
    return value * (value + 1) // 2


def quadratic_basis_78(x12: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return the deterministic quadratic basis and its 12-by-78 Jacobian."""
    x = np.asarray(tuple(x12), dtype=float).reshape(CHI_DIMENSION)
    if not np.isfinite(x).all():
        raise FloatingPointError("Critic78 input contains NaN or Inf")
    features = np.empty(CRITIC_FEATURE_COUNT, dtype=float)
    jacobian = np.zeros((CHI_DIMENSION, CRITIC_FEATURE_COUNT), dtype=float)
    column = 0
    for i in range(CHI_DIMENSION):
        for j in range(i, CHI_DIMENSION):
            features[column] = x[i] * x[j]
            if i == j:
                jacobian[i, column] = 2.0 * x[i]
            else:
                jacobian[i, column] = x[j]
                jacobian[j, column] = x[i]
            column += 1
    if column != CRITIC_FEATURE_COUNT or not np.isfinite(
        np.concatenate((features, jacobian.reshape(-1)))
    ).all():
        raise RuntimeError("Critic78 basis construction failed")
    return features, jacobian


def quadratic_feature_names_78() -> tuple[str, ...]:
    """Return deterministic names matching :func:`quadratic_basis_78`."""
    return tuple(
        f"x{i}^2" if i == j else f"x{i}*x{j}"
        for i in range(CHI_DIMENSION)
        for j in range(i, CHI_DIMENSION)
    )


def stage6a_normalize_chi(chi12: Sequence[float]) -> np.ndarray:
    """Map physical ``chi`` to the locked Stage-6A Critic coordinate."""
    chi = np.asarray(tuple(chi12), dtype=float).reshape(CHI_DIMENSION)
    if not np.isfinite(chi).all():
        raise FloatingPointError("Stage-6A chi contains NaN or Inf")
    return STAGE6A_D_CHI_INVERSE @ chi


def stage6a_value_gradient_chi(
    chi12: Sequence[float], weights78: Sequence[float]
) -> np.ndarray:
    """Return ``grad_chi W^T sigma(D_chi^-1 chi)`` with chain rule."""
    x = stage6a_normalize_chi(chi12)
    weights = np.asarray(tuple(weights78), dtype=float).reshape(
        CRITIC_FEATURE_COUNT
    )
    _, jacobian_x_transpose = quadratic_basis_78(x)
    gradient_x = jacobian_x_transpose @ weights
    gradient_chi = STAGE6A_D_CHI_INVERSE.T @ gradient_x
    if not np.isfinite(gradient_chi).all():
        raise FloatingPointError("Stage-6A value gradient is non-finite")
    return gradient_chi


def stage6a_strategy_input_matrices(
    smooth_mass: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return physical-coordinate ``G_A_chi`` and ``G_I_chi`` matrices."""
    mass = np.asarray(smooth_mass, dtype=float).reshape(2, 2)
    if not np.isfinite(mass).all() or np.min(np.linalg.eigvalsh(mass)) <= 0.0:
        raise ValueError("Stage-6A smooth mass must be finite SPD")
    game_a = np.zeros((8, 2), dtype=float)
    game_i = np.zeros((8, 2), dtype=float)
    game_a[2:4] = np.linalg.solve(np.diag(GAME_MA_KGM2), np.eye(2))
    game_i[6:8] = np.linalg.solve(mass, np.eye(2))
    return (
        np.vstack((game_a, np.zeros((4, 2), dtype=float))),
        np.vstack((game_i, np.zeros((4, 2), dtype=float))),
    )


def stage6a_running_costs(
    chi12: Sequence[float],
    alpha: float,
    s_t: float,
    strategy4: Sequence[float],
) -> dict[str, float]:
    """Evaluate the Stage-6A PPC/PT-shaped player running costs."""
    chi = np.asarray(tuple(chi12), dtype=float).reshape(CHI_DIMENSION)
    strategy = np.asarray(tuple(strategy4), dtype=float).reshape(4)
    scheduler_alpha = _finite_float(alpha, "Stage-6A alpha")
    scaling = _finite_float(s_t, "Stage-6A sT")
    if not ALPHA_LOWER <= scheduler_alpha <= ALPHA_UPPER or scaling <= 0.0:
        raise ValueError("Stage-6A cost scheduling input is invalid")
    z_a, z_i, epsilon = chi[:4], chi[4:8], chi[8:12]
    disagreement = z_a - z_i
    mu_a, mu_i = strategy[:2], strategy[2:]
    eta = game_eta(scheduler_alpha)
    coupling = float((mu_a - mu_i) @ GAME_HMU @ (mu_a - mu_i))
    q_epsilon = float(epsilon @ STAGE6A_Q_EPS @ epsilon)
    q_a = float(
        z_a @ (scheduler_alpha * PERFORMANCE_QA) @ z_a
        + PERFORMANCE_LAMBDA_A * disagreement @ PERFORMANCE_SA @ disagreement
        + q_epsilon
    )
    q_i = float(
        z_i @ ((1.0 - scheduler_alpha) * PERFORMANCE_QI) @ z_i
        + PERFORMANCE_LAMBDA_I * disagreement @ PERFORMANCE_SI @ disagreement
        + q_epsilon
    )
    r_a = float(
        mu_a @ (scheduler_alpha * GAME_RA) @ mu_a + eta * coupling
    )
    r_i = float(
        mu_i @ ((1.0 - scheduler_alpha) * GAME_RI) @ mu_i + eta * coupling
    )
    result = {
        "q_A_chi": q_a,
        "q_I_chi": q_i,
        "r_A": r_a,
        "r_I": r_i,
        "ell_A_chi": scaling * q_a + r_a,
        "ell_I_chi": scaling * q_i + r_i,
    }
    if not np.isfinite(tuple(result.values())).all():
        raise FloatingPointError("Stage-6A running cost is non-finite")
    return result


def stage6a_actor_matrices(
    chi12: Sequence[float],
    smooth_mass: Sequence[Sequence[float]],
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``B_A``, ``B_I``, and the exact 4-by-156 Actor map."""
    x = stage6a_normalize_chi(chi12)
    _, jacobian_x_transpose = quadratic_basis_78(x)
    g_a, g_i = stage6a_strategy_input_matrices(smooth_mass)
    physical_feature_gradient = (
        jacobian_x_transpose / STAGE6A_D_CHI_DIAGONAL[:, None]
    )
    b_a = g_a.T @ physical_feature_gradient
    b_i = g_i.T @ physical_feature_gradient
    b_big = np.zeros((4, 2 * CRITIC_FEATURE_COUNT), dtype=float)
    b_big[:2, :CRITIC_FEATURE_COUNT] = b_a
    b_big[2:, CRITIC_FEATURE_COUNT:] = b_i
    linear_map = -0.5 * np.linalg.solve(game_nash_matrix(alpha), b_big)
    return b_a, b_i, linear_map


def stage6a_nash_actor(
    chi12: Sequence[float],
    smooth_mass: Sequence[Sequence[float]],
    alpha: float,
    weights: Sequence[Sequence[float]],
) -> np.ndarray:
    """Evaluate the unclipped analytical Nash strategy in fixed ordering."""
    critic = np.asarray(weights, dtype=float).reshape(CRITIC_WEIGHT_SHAPE)
    b_a, b_i, _ = stage6a_actor_matrices(chi12, smooth_mass, alpha)
    psi = np.concatenate((b_a @ critic[:, 0], b_i @ critic[:, 1]))
    strategy = -0.5 * np.linalg.solve(game_nash_matrix(alpha), psi)
    if not np.isfinite(strategy).all():
        raise FloatingPointError("Stage-6A Nash strategy is non-finite")
    return strategy


def stage6a_critic_update(
    chi12: Sequence[float],
    f_chi12: Sequence[float],
    smooth_mass: Sequence[Sequence[float]],
    alpha: float,
    s_t: float,
    weights: Sequence[Sequence[float]],
    eta_c: float,
) -> dict[str, Any]:
    """Return the raw Eq.-105-style shadow update at one 100-Hz sample."""
    chi = np.asarray(tuple(chi12), dtype=float).reshape(CHI_DIMENSION)
    drift = np.asarray(tuple(f_chi12), dtype=float).reshape(CHI_DIMENSION)
    critic = np.asarray(weights, dtype=float).reshape(CRITIC_WEIGHT_SHAPE)
    learning_rate = _finite_float(eta_c, "Stage-6A eta_c")
    if learning_rate <= 0.0:
        raise ValueError("Stage-6A eta_c must be positive")
    strategy = stage6a_nash_actor(chi, smooth_mass, alpha, critic)
    g_a, g_i = stage6a_strategy_input_matrices(smooth_mass)
    f_n = drift + g_a @ strategy[:2] + g_i @ strategy[2:]
    x = stage6a_normalize_chi(chi)
    sigma, jacobian_x_transpose = quadratic_basis_78(x)
    omega = (
        jacobian_x_transpose.T @ (STAGE6A_D_CHI_INVERSE @ f_n)
        - STAGE6A_DISCOUNT_RATE_S_INV * sigma
    )
    costs = stage6a_running_costs(chi, alpha, s_t, strategy)
    delta = np.asarray(
        (
            costs["ell_A_chi"] + critic[:, 0] @ omega,
            costs["ell_I_chi"] + critic[:, 1] @ omega,
        ),
        dtype=float,
    )
    denominator = (1.0 + float(omega @ omega)) ** 2
    # Gamma_c is the locked identity, so form the outer product directly.
    weight_dot = (
        -learning_rate * omega[:, None] * delta[None, :] / denominator
    )
    if weight_dot.shape != CRITIC_WEIGHT_SHAPE or not np.isfinite(
        np.concatenate((strategy, f_n, sigma, omega, delta,
                        weight_dot.reshape(-1)))
    ).all():
        raise FloatingPointError("Stage-6A Critic update is invalid")
    return {
        "strategy": strategy,
        "f_N_hat": f_n,
        "sigma": sigma,
        "omega": omega,
        "delta": delta,
        "weight_dot_raw": weight_dot,
        "sigma_e_t": np.zeros(CRITIC_FEATURE_COUNT, dtype=float),
        **costs,
    }


def stage6a_project_weight_columns(
    candidate: Sequence[Sequence[float]], radius: float
) -> tuple[np.ndarray, tuple[bool, bool]]:
    """Project each Critic output column onto the locked Euclidean ball."""
    weights = np.asarray(candidate, dtype=float).reshape(CRITIC_WEIGHT_SHAPE)
    bound = _finite_float(radius, "Stage-6A weight radius")
    if bound <= 0.0 or not np.isfinite(weights).all():
        raise ValueError("Stage-6A projection input is invalid")
    result = weights.copy()
    projected = []
    for column in range(2):
        norm = float(np.linalg.norm(result[:, column]))
        active = norm > bound
        if active:
            result[:, column] *= bound / norm
        projected.append(active)
    return result, (projected[0], projected[1])


def stage6a_value_matrix(weights78: Sequence[float]) -> np.ndarray:
    """Reconstruct symmetric ``P`` such that ``V(x)=x.T @ P @ x``."""
    weights = np.asarray(tuple(weights78), dtype=float).reshape(
        CRITIC_FEATURE_COUNT
    )
    matrix = np.zeros((CHI_DIMENSION, CHI_DIMENSION), dtype=float)
    column = 0
    for i in range(CHI_DIMENSION):
        for j in range(i, CHI_DIMENSION):
            if i == j:
                matrix[i, i] = weights[column]
            else:
                matrix[i, j] = matrix[j, i] = 0.5 * weights[column]
            column += 1
    return matrix


def stage6b_causal_control_step(
    chi12: Sequence[float],
    f_chi12: Sequence[float],
    smooth_mass: Sequence[Sequence[float]],
    alpha: float,
    s_t: float,
    weights: Sequence[Sequence[float]],
    dt: float = CONTROL_DT_S,
) -> dict[str, Any]:
    """Evaluate ``U_k`` from ``W_k`` and prepare projected ``W_{k+1}``.

    The returned strategy is computed before the projected Euler update.  A
    caller may therefore form the interval-k physical command from
    ``strategy_k`` and only then store ``weights_next`` for the next tick.
    """
    step = _finite_float(dt, "Stage-6B dt")
    if not math.isclose(step, CONTROL_DT_S, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("Stage-6B requires the locked 100 Hz update")
    current = np.asarray(weights, dtype=float).reshape(CRITIC_WEIGHT_SHAPE)
    update = stage6a_critic_update(
        chi12,
        f_chi12,
        smooth_mass,
        alpha,
        s_t,
        current,
        STAGE6B_CRITIC_ETA,
    )
    candidate = current + step * update["weight_dot_raw"]
    next_weights, projected = stage6a_project_weight_columns(
        candidate, STAGE6B_WEIGHT_RADIUS
    )
    strategy_from_current = stage6a_nash_actor(
        chi12, smooth_mass, alpha, current
    )
    if not np.array_equal(strategy_from_current, update["strategy"]):
        raise RuntimeError("Stage-6B Actor did not use the current Critic weight")
    gradients = np.column_stack(
        (
            stage6a_value_gradient_chi(chi12, current[:, 0]),
            stage6a_value_gradient_chi(chi12, current[:, 1]),
        )
    )
    p_a = stage6a_value_matrix(current[:, 0])
    p_i = stage6a_value_matrix(current[:, 1])
    values = np.concatenate(
        (
            current.reshape(-1),
            strategy_from_current,
            update["weight_dot_raw"].reshape(-1),
            next_weights.reshape(-1),
            gradients.reshape(-1),
            p_a.reshape(-1),
            p_i.reshape(-1),
        )
    )
    if not np.isfinite(values).all():
        raise FloatingPointError("Stage-6B causal control step is non-finite")
    return {
        **update,
        "weights_current": current.copy(),
        "weights_candidate": candidate,
        "weights_next": next_weights,
        "projected": projected,
        "value_gradients_chi": gradients,
        "P_A": p_a,
        "P_I": p_i,
        "causal_actor_uses_Wk": True,
        "next_weight_not_used_for_current_actor": True,
    }


def stage6f_total_actuation_beta(alpha: float) -> float:
    """Return the locked Stage-6F low-intention physical-actuation weight."""
    value = _finite_float(alpha, "Stage-6F alpha")
    if not ALPHA_LOWER <= value <= ALPHA_UPPER:
        raise ValueError("Stage-6F alpha is outside the locked bounds")
    return float((1.0 - value) ** 3)


def stage6f_nash_matrix(alpha: float) -> np.ndarray:
    """Build the physical-actuation-aware Stage-6F Nash FOC matrix."""
    value = _finite_float(alpha, "Stage-6F alpha")
    beta = stage6f_total_actuation_beta(value)
    matrix = game_nash_matrix(value).copy()
    matrix[2:, 2:] += beta * GAME_RI
    if not np.isfinite(matrix).all() or not np.allclose(
        matrix, matrix.T, rtol=0.0, atol=1.0e-15
    ):
        raise RuntimeError("Stage-6F Nash matrix is non-finite or asymmetric")
    return matrix


def stage6f_nash_actor(
    chi12: Sequence[float],
    smooth_mass: Sequence[Sequence[float]],
    alpha: float,
    weights: Sequence[Sequence[float]],
    mu0: Sequence[float],
) -> dict[str, Any]:
    """Solve the Stage-6F affine Nash FOC without an explicit inverse."""
    critic = np.asarray(weights, dtype=float).reshape(CRITIC_WEIGHT_SHAPE)
    baseline = np.asarray(tuple(mu0), dtype=float).reshape(2)
    if not np.isfinite(baseline).all():
        raise FloatingPointError("Stage-6F nominal torque is non-finite")
    b_a, b_i, _ = stage6a_actor_matrices(chi12, smooth_mass, alpha)
    psi = np.concatenate((b_a @ critic[:, 0], b_i @ critic[:, 1]))
    beta = stage6f_total_actuation_beta(alpha)
    b_total = np.concatenate((np.zeros(2), beta * GAME_RI @ baseline))
    matrix = stage6f_nash_matrix(alpha)
    strategy = np.linalg.solve(matrix, -0.5 * psi - b_total)
    residual = matrix @ strategy + 0.5 * psi + b_total
    if not np.isfinite(np.concatenate((strategy, residual))).all():
        raise FloatingPointError("Stage-6F Nash strategy is non-finite")
    return {
        "strategy": strategy,
        "psi_N": psi,
        "b_total": b_total,
        "RN_new": matrix,
        "FOC_residual": residual,
        "beta": beta,
    }


def stage6f_running_costs(
    chi12: Sequence[float],
    alpha: float,
    s_t: float,
    strategy4: Sequence[float],
    mu0: Sequence[float],
) -> dict[str, float]:
    """Add only the authorized total-physical-actuation I-player cost."""
    strategy = np.asarray(tuple(strategy4), dtype=float).reshape(4)
    baseline = np.asarray(tuple(mu0), dtype=float).reshape(2)
    original = stage6a_running_costs(chi12, alpha, s_t, strategy)
    mu_i = strategy[2:]
    total = baseline + mu_i
    beta = stage6f_total_actuation_beta(alpha)
    r_total = float(beta * total @ GAME_RI @ total)
    expanded = float(
        beta
        * (
            mu_i @ GAME_RI @ mu_i
            + 2.0 * baseline @ GAME_RI @ mu_i
            + baseline @ GAME_RI @ baseline
        )
    )
    result = dict(original)
    result.update(
        {
            "r_I_original": original["r_I"],
            "r_total": r_total,
            "r_total_expanded": expanded,
            "r_total_expansion_error": r_total - expanded,
            "r_I": original["r_I"] + r_total,
            "ell_I_chi": original["ell_I_chi"] + r_total,
            "beta": beta,
        }
    )
    if not np.isfinite(tuple(result.values())).all():
        raise FloatingPointError("Stage-6F running cost is non-finite")
    return result


def stage6f_critic_update(
    chi12: Sequence[float],
    f_chi12: Sequence[float],
    smooth_mass: Sequence[Sequence[float]],
    alpha: float,
    s_t: float,
    weights: Sequence[Sequence[float]],
    mu0: Sequence[float],
) -> dict[str, Any]:
    """Evaluate one locked-rate Stage-6F Critic update."""
    chi = np.asarray(tuple(chi12), dtype=float).reshape(CHI_DIMENSION)
    drift = np.asarray(tuple(f_chi12), dtype=float).reshape(CHI_DIMENSION)
    critic = np.asarray(weights, dtype=float).reshape(CRITIC_WEIGHT_SHAPE)
    actor = stage6f_nash_actor(chi, smooth_mass, alpha, critic, mu0)
    strategy = np.asarray(actor["strategy"], dtype=float)
    g_a, g_i = stage6a_strategy_input_matrices(smooth_mass)
    f_n = drift + g_a @ strategy[:2] + g_i @ strategy[2:]
    x = stage6a_normalize_chi(chi)
    sigma, jacobian_x_transpose = quadratic_basis_78(x)
    omega = (
        jacobian_x_transpose.T @ (STAGE6A_D_CHI_INVERSE @ f_n)
        - STAGE6A_DISCOUNT_RATE_S_INV * sigma
    )
    costs = stage6f_running_costs(chi, alpha, s_t, strategy, mu0)
    delta = np.asarray(
        (
            costs["ell_A_chi"] + critic[:, 0] @ omega,
            costs["ell_I_chi"] + critic[:, 1] @ omega,
        ),
        dtype=float,
    )
    denominator = (1.0 + float(omega @ omega)) ** 2
    weight_dot = -STAGE6B_CRITIC_ETA * omega[:, None] * delta[None, :] / denominator
    if not np.isfinite(
        np.concatenate((strategy, f_n, omega, delta, weight_dot.reshape(-1)))
    ).all():
        raise FloatingPointError("Stage-6F Critic update is invalid")
    return {
        **actor,
        **costs,
        "f_N_hat": f_n,
        "sigma": sigma,
        "omega": omega,
        "delta": delta,
        "weight_dot_raw": weight_dot,
    }


def stage6f_causal_control_step(
    chi12: Sequence[float],
    f_chi12: Sequence[float],
    smooth_mass: Sequence[Sequence[float]],
    alpha: float,
    s_t: float,
    weights: Sequence[Sequence[float]],
    mu0: Sequence[float],
    dt: float = CONTROL_DT_S,
) -> dict[str, Any]:
    """Form Stage-6F ``U_k`` from ``W_k`` before projecting ``W_{k+1}``."""
    step = _finite_float(dt, "Stage-6F dt")
    if not math.isclose(step, CONTROL_DT_S, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("Stage-6F requires the locked 100 Hz update")
    current = np.asarray(weights, dtype=float).reshape(CRITIC_WEIGHT_SHAPE)
    update = stage6f_critic_update(
        chi12, f_chi12, smooth_mass, alpha, s_t, current, mu0
    )
    candidate = current + step * update["weight_dot_raw"]
    next_weights, projected = stage6a_project_weight_columns(
        candidate, STAGE6B_WEIGHT_RADIUS
    )
    gradients = np.column_stack(
        (
            stage6a_value_gradient_chi(chi12, current[:, 0]),
            stage6a_value_gradient_chi(chi12, current[:, 1]),
        )
    )
    return {
        **update,
        "weights_current": current.copy(),
        "weights_candidate": candidate,
        "weights_next": next_weights,
        "projected": projected,
        "value_gradients_chi": gradients,
        "P_A": stage6a_value_matrix(current[:, 0]),
        "P_I": stage6a_value_matrix(current[:, 1]),
        "causal_actor_uses_Wk": True,
        "next_weight_not_used_for_current_actor": True,
    }


def stage6g_assistance_gamma(alpha: float) -> float:
    """Locked C1 intention-only assistance target gate."""
    value = _finite_float(alpha, "Stage-6G alpha")
    if not ALPHA_LOWER <= value <= ALPHA_UPPER:
        raise ValueError("Stage-6G alpha is outside the locked bounds")
    if value <= 0.45:
        return 0.0
    if value >= 0.75:
        return 1.0
    xi = (value - 0.45) / 0.30
    return float(3.0 * xi * xi - 2.0 * xi * xi * xi)


def stage6g_nash_matrix(alpha: float) -> np.ndarray:
    """Build the locked physical-target Nash FOC matrix."""
    matrix = game_nash_matrix(_finite_float(alpha, "Stage-6G alpha")).copy()
    matrix[2:, 2:] += GAME_RI
    if not np.isfinite(matrix).all() or not np.allclose(
        matrix, matrix.T, rtol=0.0, atol=1.0e-15
    ):
        raise RuntimeError("Stage-6G Nash matrix is non-finite or asymmetric")
    return matrix


def stage6g_nash_actor(
    chi12: Sequence[float], smooth_mass: Sequence[Sequence[float]],
    alpha: float, weights: Sequence[Sequence[float]], mu0: Sequence[float],
) -> dict[str, Any]:
    """Solve the affine intention-target Nash FOC without an inverse."""
    critic = np.asarray(weights, dtype=float).reshape(CRITIC_WEIGHT_SHAPE)
    baseline = np.asarray(tuple(mu0), dtype=float).reshape(2)
    b_a, b_i, _ = stage6a_actor_matrices(chi12, smooth_mass, alpha)
    psi = np.concatenate((b_a @ critic[:, 0], b_i @ critic[:, 1]))
    gamma = stage6g_assistance_gamma(alpha)
    b_target = np.concatenate((np.zeros(2), (1.0-gamma)*GAME_RI@baseline))
    matrix = stage6g_nash_matrix(alpha)
    strategy_affine = np.linalg.solve(matrix, -b_target)
    strategy_critic = np.linalg.solve(matrix, -0.5*psi)
    strategy = np.linalg.solve(matrix, -0.5*psi-b_target)
    residual = matrix@strategy + 0.5*psi + b_target
    decomposition_residual = strategy - (strategy_affine + strategy_critic)
    if not np.isfinite(np.concatenate((strategy, strategy_affine,
                                       strategy_critic, residual))).all():
        raise FloatingPointError("Stage-6G Nash strategy is non-finite")
    return {"strategy":strategy,"psi_N":psi,"b_target":b_target,
            "strategy_affine":strategy_affine,
            "strategy_critic":strategy_critic,
            "strategy_decomposition_residual":decomposition_residual,
            "RN_target":matrix,"FOC_residual":residual,"gamma":gamma}


def stage6g_running_costs(
    chi12: Sequence[float], alpha: float, s_t: float,
    strategy4: Sequence[float], mu0: Sequence[float],
) -> dict[str, float]:
    """Add only the coefficient-one intention-target I-player cost."""
    strategy=np.asarray(tuple(strategy4),dtype=float).reshape(4)
    baseline=np.asarray(tuple(mu0),dtype=float).reshape(2)
    original=stage6a_running_costs(chi12,alpha,s_t,strategy)
    mu_i=strategy[2:]; gamma=stage6g_assistance_gamma(alpha)
    error=mu_i+(1.0-gamma)*baseline
    r_target=float(error@GAME_RI@error)
    expanded=float(mu_i@GAME_RI@mu_i + 2.0*(1.0-gamma)*baseline@GAME_RI@mu_i + (1.0-gamma)**2*baseline@GAME_RI@baseline)
    result=dict(original); result.update({"r_I_original":original["r_I"],
        "r_target":r_target,"r_target_expanded":expanded,
        "r_target_expansion_error":r_target-expanded,
        "r_I":original["r_I"]+r_target,
        "ell_I_chi":original["ell_I_chi"]+r_target,"gamma":gamma})
    if not np.isfinite(tuple(result.values())).all():
        raise FloatingPointError("Stage-6G running cost is non-finite")
    return result


def stage6g_critic_update(
    chi12: Sequence[float], f_chi12: Sequence[float],
    smooth_mass: Sequence[Sequence[float]], alpha: float, s_t: float,
    weights: Sequence[Sequence[float]], mu0: Sequence[float],
) -> dict[str, Any]:
    chi=np.asarray(tuple(chi12),dtype=float).reshape(CHI_DIMENSION)
    drift=np.asarray(tuple(f_chi12),dtype=float).reshape(CHI_DIMENSION)
    critic=np.asarray(weights,dtype=float).reshape(CRITIC_WEIGHT_SHAPE)
    actor=stage6g_nash_actor(chi,smooth_mass,alpha,critic,mu0)
    strategy=np.asarray(actor["strategy"]); g_a,g_i=stage6a_strategy_input_matrices(smooth_mass)
    f_n=drift+g_a@strategy[:2]+g_i@strategy[2:]
    x=stage6a_normalize_chi(chi); sigma,jacobian=quadratic_basis_78(x)
    omega=jacobian.T@(STAGE6A_D_CHI_INVERSE@f_n)-STAGE6A_DISCOUNT_RATE_S_INV*sigma
    costs=stage6g_running_costs(chi,alpha,s_t,strategy,mu0)
    delta=np.asarray((costs["ell_A_chi"]+critic[:,0]@omega,costs["ell_I_chi"]+critic[:,1]@omega))
    weight_dot=-STAGE6B_CRITIC_ETA*omega[:,None]*delta[None,:]/(1.0+float(omega@omega))**2
    if not np.isfinite(np.concatenate((strategy,f_n,omega,delta,weight_dot.ravel()))).all():
        raise FloatingPointError("Stage-6G Critic update is invalid")
    return {**actor,**costs,"f_N_hat":f_n,"sigma":sigma,"omega":omega,
            "delta":delta,"weight_dot_raw":weight_dot}


def stage6g_causal_control_step(
    chi12: Sequence[float], f_chi12: Sequence[float],
    smooth_mass: Sequence[Sequence[float]], alpha: float, s_t: float,
    weights: Sequence[Sequence[float]], mu0: Sequence[float],
    dt: float=CONTROL_DT_S,
) -> dict[str, Any]:
    step=_finite_float(dt,"Stage-6G dt")
    if not math.isclose(step,CONTROL_DT_S,rel_tol=0.0,abs_tol=1e-12):
        raise ValueError("Stage-6G requires the locked 100 Hz update")
    current=np.asarray(weights,dtype=float).reshape(CRITIC_WEIGHT_SHAPE)
    update=stage6g_critic_update(chi12,f_chi12,smooth_mass,alpha,s_t,current,mu0)
    candidate=current+step*update["weight_dot_raw"]
    next_weights,projected=stage6a_project_weight_columns(candidate,STAGE6B_WEIGHT_RADIUS)
    gradients=np.column_stack((stage6a_value_gradient_chi(chi12,current[:,0]),stage6a_value_gradient_chi(chi12,current[:,1])))
    return {**update,"weights_current":current.copy(),"weights_candidate":candidate,
            "weights_next":next_weights,"projected":projected,
            "value_gradients_chi":gradients,"P_A":stage6a_value_matrix(current[:,0]),
            "P_I":stage6a_value_matrix(current[:,1]),"causal_actor_uses_Wk":True,
            "next_weight_not_used_for_current_actor":True}


def ppc_envelope_with_limits(
    local_time_s: float, rho0: float, rho_inf: float,
    transition_horizon_s: float = PPC_T_RHO_S,
    nu: float = PPC_NU,
) -> tuple[float, float]:
    """Return the PPC envelope and analytic derivative for fixed limits."""
    time_s = _finite_float(local_time_s, "PPC local time")
    initial = _finite_float(rho0, "PPC rho0")
    terminal = _finite_float(rho_inf, "PPC rho_inf")
    horizon = _finite_float(transition_horizon_s, "PPC transition horizon")
    exponent = _finite_float(nu, "PPC exponent")
    if initial <= 0.0 or terminal <= 0.0 or initial < terminal:
        raise ValueError("PPC envelope limits must satisfy rho0 >= rho_inf > 0")
    if horizon <= 0.0 or exponent <= 0.0:
        raise ValueError("PPC horizon and exponent must be positive")
    if time_s <= 0.0:
        return initial, 0.0
    if time_s >= horizon:
        return terminal, 0.0
    theta = math.pi * time_s / (2.0 * horizon)
    cosine = math.cos(theta)
    sine = math.sin(theta)
    rho = (initial - terminal) * cosine**exponent + terminal
    rho_dot = (
        -(initial - terminal)
        * exponent
        * math.pi
        / (2.0 * horizon)
        * cosine ** (exponent - 1.0)
        * sine
    )
    return float(rho), float(rho_dot)


def ppc_envelope(local_time_s: float) -> tuple[float, float]:
    """Return the Stage-4R development envelope and analytic derivative."""
    return ppc_envelope_with_limits(
        local_time_s, PPC_RHO0_DEV, PPC_RHO_INF_DEV
    )


def ppc_transform_given_envelope(
    error_performance: Sequence[float],
    error_performance_dot: Sequence[float],
    rho: float,
    rho_dot: float,
) -> dict[str, np.ndarray | float | bool]:
    """Apply PPC when valid, or return explicit unavailable diagnostics."""
    error = np.asarray(tuple(error_performance), dtype=float).reshape(4)
    error_dot = np.asarray(tuple(error_performance_dot), dtype=float).reshape(4)
    envelope = _finite_float(rho, "PPC rho")
    envelope_dot = _finite_float(rho_dot, "PPC rho_dot")
    if envelope <= 0.0 or not np.isfinite(np.concatenate((error, error_dot))).all():
        raise ValueError("PPC transform inputs must be finite with rho > 0")
    zeta = error / envelope
    valid = bool(np.all(np.abs(zeta) < 1.0))
    if not valid:
        unavailable = np.full(4, np.nan, dtype=float)
        return {
            "valid": False,
            "rho": envelope,
            "rho_dot": envelope_dot,
            "zeta": zeta,
            "epsilon": unavailable.copy(),
            "r_diagonal": unavailable.copy(),
            "epsilon_dot": unavailable.copy(),
        }
    angle = 0.5 * math.pi * zeta
    epsilon = np.tan(angle)
    r_diagonal = math.pi / (2.0 * envelope) / np.cos(angle) ** 2
    epsilon_dot = r_diagonal * (
        error_dot - (envelope_dot / envelope) * error
    )
    values = np.concatenate((zeta, epsilon, r_diagonal, epsilon_dot))
    if not np.isfinite(values).all():
        raise FloatingPointError("Stage-4R PPC transform contains NaN or Inf")
    return {
        "valid": True,
        "rho": envelope,
        "rho_dot": envelope_dot,
        "zeta": zeta,
        "epsilon": epsilon,
        "r_diagonal": r_diagonal,
        "epsilon_dot": epsilon_dot,
    }


def ppc_transform(
    error_performance: Sequence[float],
    error_performance_dot: Sequence[float],
    local_time_s: float,
) -> dict[str, np.ndarray | float | bool]:
    """Apply the diagnostic-only, non-terminating Stage-4R PPC transform."""
    rho, rho_dot = ppc_envelope(local_time_s)
    return ppc_transform_given_envelope(
        error_performance, error_performance_dot, rho, rho_dot
    )


def predefined_time_scaling_with_parameters(
    local_time_s: float, horizon_s: float, delta: float,
    p_s_inv: float,
) -> dict[str, float]:
    """Return endpoint-safe analytic phi1, phi1_dot, phiT, and sT."""
    time_s = _finite_float(local_time_s, "predefined-time local time")
    horizon = _finite_float(horizon_s, "predefined-time horizon")
    regularizer = _finite_float(delta, "predefined-time delta")
    p_value = _finite_float(p_s_inv, "predefined-time p")
    if horizon <= 0.0 or regularizer <= 0.0 or p_value <= 0.0:
        raise ValueError("Predefined-time parameters must be positive")
    if time_s <= 0.0:
        phi1 = 1.0
        phi1_dot = 0.0
    elif time_s >= horizon:
        phi1 = 0.0
        phi1_dot = 0.0
    else:
        log_a = -horizon / (horizon - time_s)
        log_b = -horizon / time_s
        log_ratio = log_b - log_a
        if log_ratio >= 0.0:
            exp_negative = math.exp(-log_ratio)
            phi1 = exp_negative / (1.0 + exp_negative)
        else:
            exp_positive = math.exp(log_ratio)
            phi1 = 1.0 / (1.0 + exp_positive)
        ratio_derivative = (
            horizon / time_s**2
            + horizon / (horizon - time_s) ** 2
        )
        phi1_dot = -phi1 * (1.0 - phi1) * ratio_derivative
    phi_t = 1.0 / (phi1 + regularizer)
    s_t = p_value - phi1_dot / (phi1 + regularizer)
    values = (phi1, phi1_dot, phi_t, s_t)
    if not all(math.isfinite(value) for value in values) or s_t < p_value:
        raise RuntimeError("Predefined-time scaling is invalid")
    return {
        "phi1": float(phi1),
        "phi1_dot": float(phi1_dot),
        "phiT": float(phi_t),
        "sT": float(s_t),
    }


def predefined_time_scaling(local_time_s: float) -> dict[str, float]:
    """Return the frozen final Stage-6H predefined-time scaling."""
    return predefined_time_scaling_with_parameters(
        local_time_s, PT_T_P_S, PT_DELTA, PT_P_S_INV
    )


def fixed_b3_basis(z_rad_s: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return the Fixed-K B3 features and Jacobian with respect to physical z."""
    z = np.asarray(tuple(z_rad_s), dtype=float).reshape(2)
    if not np.isfinite(z).all():
        raise FloatingPointError("Fixed-K Critic z contains NaN or Inf")
    x1, x2 = z / FIXED_ADP_Z_SCALE_RAD_S
    features = np.asarray((x1 * x1, x1 * x2, x2 * x2), dtype=float)
    s_hip, s_knee = FIXED_ADP_Z_SCALE_RAD_S
    jacobian = np.asarray(
        (
            (2.0 * x1 / s_hip, x2 / s_hip, 0.0),
            (0.0, x1 / s_knee, 2.0 * x2 / s_knee),
        ),
        dtype=float,
    )
    if not np.isfinite(np.concatenate((features, jacobian.reshape(-1)))).all():
        raise FloatingPointError("Fixed-K B3 basis contains NaN or Inf")
    return features, jacobian


def fixed_value_minimum_eigenvalue(weight: Sequence[float]) -> float:
    """Return lambda_min of the B3 quadratic value matrix."""
    values = np.asarray(tuple(weight), dtype=float).reshape(3)
    matrix = np.asarray(
        ((values[0], 0.5 * values[1]), (0.5 * values[1], values[2])),
        dtype=float,
    )
    return float(np.linalg.eigvalsh(matrix)[0])


class ICRAReducedPlant:
    """Left-2DOF adapter around the frozen seven-coordinate YAC model."""

    def __init__(self, mujoco_model: Any) -> None:
        self.model = PlanarFloatingBaseExoModel.from_mujoco(mujoco_model)

    @staticmethod
    def _row_labels(jacobian: np.ndarray) -> tuple[str, ...]:
        labels: list[str] = []
        for row in jacobian:
            if not np.any(np.abs(row) > 1.0e-12):
                labels.append("unknown")
            else:
                labels.append(COORDINATE_ORDER[int(np.argmax(np.abs(row)))])
        return tuple(labels)

    def evaluate(
        self,
        dynamics: Mapping[str, Any],
        target_left_qacc: Sequence[float],
    ) -> dict[str, Any]:
        """Return one fixed-active-set plant and an exactly-once HRI split."""
        q = np.asarray(dynamics["q_exo"], dtype=float).reshape(7)
        qvel = np.asarray(dynamics["qvel_exo"], dtype=float).reshape(7)
        external = np.asarray(
            dynamics["external_generalized_force"], dtype=float
        ).reshape(7)
        coupling = np.asarray(
            dynamics["coupling_generalized_force"], dtype=float
        ).reshape(7)
        passive = np.asarray(
            dynamics["passive_generalized_force"], dtype=float
        ).reshape(7)
        base_applied = np.asarray(
            dynamics["base_applied_generalized_force"], dtype=float
        ).reshape(7)
        jacobian = np.asarray(
            dynamics["constraint_jacobian"], dtype=float
        ).reshape(-1, 7)
        regularization = np.asarray(
            dynamics["constraint_regularization"], dtype=float
        ).reshape(-1)
        reference = np.asarray(
            dynamics["constraint_reference_acceleration"], dtype=float
        ).reshape(-1)
        target = np.asarray(target_left_qacc, dtype=float).reshape(2)
        all_values = np.concatenate(
            (q, qvel, external, coupling, passive, base_applied, target)
        )
        if not np.isfinite(all_values).all():
            raise FloatingPointError("ICRA plant input contains NaN or Inf")
        accounting_error = external - (coupling + passive)
        if np.max(np.abs(accounting_error)) > 1.0e-12:
            raise RuntimeError("External generalized force is not M5 plus passive")
        if np.max(np.abs(base_applied)) > 1.0e-12:
            raise RuntimeError("ICRA Stage 1 does not permit an unmodelled base force")

        # Stage 2R analytical control model: intrinsic smooth Schur dynamics.
        # No M5, passive force, or joint-limit KKT reaction is feedforward-
        # cancelled.  Those effects remain physical disturbances.
        zero_external = np.zeros(7, dtype=float)
        smooth_mass, smooth_bias = self.model.reduced_dynamics(
            q, qvel, zero_external
        )
        full_mass = self.model.mass_matrix(q)
        controlled = np.asarray(CONTROLLED_INDICES, dtype=int)
        unactuated = np.asarray(UNACTUATED_INDICES, dtype=int)
        muu = full_mass[np.ix_(unactuated, unactuated)]
        mcu = full_mass[np.ix_(controlled, unactuated)]

        def smooth_force_map(force: np.ndarray) -> np.ndarray:
            return force[controlled] - mcu @ np.linalg.solve(
                muu, force[unactuated]
            )

        tau_h_smooth = smooth_force_map(coupling)
        passive_smooth = smooth_force_map(passive)
        smooth_eigenvalues = np.linalg.eigvalsh(smooth_mass)
        if (
            not np.isfinite(
                np.concatenate(
                    (
                        smooth_mass.reshape(-1),
                        smooth_bias,
                        tau_h_smooth,
                        passive_smooth,
                        smooth_eigenvalues,
                    )
                )
            ).all()
            or np.any(smooth_eigenvalues <= 0.0)
        ):
            raise RuntimeError("ICRA smooth analytical model is non-finite or not SPD")

        selected = self.model.unilateral_inverse_dynamics(
            q,
            qvel,
            external,
            target,
            jacobian,
            regularization,
            reference,
        )
        active = np.asarray(selected["active_indices"], dtype=int)
        active_jacobian = jacobian[active]
        active_regularization = regularization[active]
        active_reference = reference[active]
        f_other = external - coupling
        if active.size:
            meff_other, b_no_hri = self.model.constrained_reduced_dynamics(
                q,
                qvel,
                f_other,
                active_jacobian,
                active_regularization,
                active_reference,
            )
            meff_total, b_with_hri = self.model.constrained_reduced_dynamics(
                q,
                qvel,
                external,
                active_jacobian,
                active_regularization,
                active_reference,
            )
        else:
            meff_other, b_no_hri = self.model.reduced_dynamics(q, qvel, f_other)
            meff_total, b_with_hri = self.model.reduced_dynamics(q, qvel, external)
        if not np.allclose(meff_other, meff_total, rtol=0.0, atol=1.0e-12):
            raise RuntimeError("M_eff changed during fixed-active-set HRI split")
        tau_h_eff = b_no_hri - b_with_hri
        identity_error = b_with_hri - (b_no_hri - tau_h_eff)
        eigenvalues = np.linalg.eigvalsh(meff_other)
        if (
            not np.isfinite(
                np.concatenate(
                    (meff_other.reshape(-1), b_no_hri, tau_h_eff, eigenvalues)
                )
            ).all()
            or np.any(eigenvalues <= 0.0)
        ):
            raise RuntimeError("ICRA reduced plant is non-finite or not SPD")
        labels = self._row_labels(jacobian)
        active_labels = tuple(labels[index] for index in active)
        return {
            "M_eff": meff_other,
            "b_noHRI": b_no_hri,
            "b_withHRI": b_with_hri,
            "tau_h_eff": tau_h_eff,
            "tau_h_direct": coupling[np.asarray(CONTROLLED_INDICES)],
            "f_M5": coupling,
            "f_other": f_other,
            "external": external,
            "external_accounting_error": accounting_error,
            "base_applied": base_applied,
            "identity_error": identity_error,
            "eigenvalues": eigenvalues,
            "condition": float(np.linalg.cond(meff_other)),
            "M_smooth": smooth_mass,
            "b_smooth": smooth_bias,
            "tau_h_smooth": tau_h_smooth,
            "passive_smooth": passive_smooth,
            "smooth_eigenvalues": smooth_eigenvalues,
            "smooth_condition": float(np.linalg.cond(smooth_mass)),
            "active_indices": active,
            "active_labels": active_labels,
            "candidate_labels": labels,
        }


class ICRAController:
    """Staged ICRA controller with a left-side Fixed-K ADP baseline."""

    requires_full_state_snapshot = True

    def __init__(
        self,
        *,
        mode: str = FIXED_K_ADP_MODE,
        reference_result: str | Path | None = None,
        mujoco_model: Any | None = None,
        ppc_rho0: float | None = None,
        ppc_rho_inf: float | None = None,
        ppc_t_rho_s: float | None = None,
        pt_tp_s: float | None = None,
    ) -> None:
        normalized = str(mode).strip().lower()
        allowed_modes = (
            SHADOW_CONTRACT_MODE,
            SHADOW_PLANT_SCHEDULER_MODE,
            *FIXED_K_FAMILY_MODES,
            *STAGE5_MODES,
            *STAGE6A_MODES,
            *STAGE6B_MODES,
            *STAGE6E_MODES,
            *STAGE6F_MODES, *STAGE6G_MODES, *EVENT_EXECUTION_MODES,
            *STAGE7T_MODES,
            *STAGE8A_MODES,
        )
        if normalized not in allowed_modes:
            raise ValueError(
                "Unsupported ICRA mode; expected one of " + ", ".join(allowed_modes)
            )
        if normalized != SHADOW_CONTRACT_MODE and (
            reference_result is None or mujoco_model is None
        ):
            raise ValueError(
                f"{normalized} requires reference_result and mujoco_model"
            )
        self.mode = normalized
        self._ppc_rho0 = PPC_RHO0_DEV
        self._ppc_rho_inf = PPC_RHO_INF_DEV
        self._ppc_t_rho_s = PPC_T_RHO_S
        self._pt_tp_s = PT_T_P_S
        if normalized == PPC_TIMING_CHARACTERIZATION_MODE:
            values = (ppc_rho0, ppc_rho_inf, ppc_t_rho_s)
            if any(value is None for value in values) or pt_tp_s not in (None, PT_T_P_S):
                raise ValueError("Stage8A PPC mode requires only rho0/rho_inf/Trho")
            if (float(ppc_rho0), float(ppc_rho_inf)) != (1.0, 0.8) or float(ppc_t_rho_s) not in (1.0, 2.3, 5.0, 8.0):
                raise ValueError("Stage8A PPC parameters are outside the authorized grid")
            self._ppc_rho0, self._ppc_rho_inf, self._ppc_t_rho_s = map(float, values)
        elif normalized == PT_HORIZON_CHARACTERIZATION_MODE:
            if pt_tp_s is None or float(pt_tp_s) not in (1.0, 2.0, 3.45, 5.0, 8.0) or any(value is not None for value in (ppc_rho0, ppc_rho_inf, ppc_t_rho_s)):
                raise ValueError("Stage8A PT parameters are outside the authorized grid")
            self._pt_tp_s = float(pt_tp_s)
        elif any(value is not None for value in (ppc_rho0, ppc_rho_inf, ppc_t_rho_s, pt_tp_s)):
            raise ValueError("Experimental PPC/PT parameters require a Stage8A mode")
        self.reference = (
            None
            if reference_result is None
            else HealthyGaitReference(reference_result, "l")
        )
        self._mujoco_model = mujoco_model
        self.plant = (
            None if mujoco_model is None else ICRAReducedPlant(mujoco_model)
        )
        self._scratch = None if mujoco_model is None else mujoco.MjData(mujoco_model)
        self._exo_dof_addresses: np.ndarray | None = None
        self._left_dof_addresses: np.ndarray | None = None
        self._exo_actuator_ids: dict[str, int] = {}
        self._exo_joint_ids: dict[str, int] = {}
        if mujoco_model is not None:
            self._exo_joint_ids = {
                name: int(
                    mujoco.mj_name2id(
                        mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, name
                    )
                )
                for name in COORDINATE_ORDER
            }
            if any(value < 0 for value in self._exo_joint_ids.values()):
                raise ValueError("ICRA model is missing a locked Exo coordinate")
            self._exo_dof_addresses = np.asarray(
                [
                    int(mujoco_model.jnt_dofadr[self._exo_joint_ids[name]])
                    for name in COORDINATE_ORDER
                ],
                dtype=int,
            )
            self._left_dof_addresses = self._exo_dof_addresses[
                np.asarray(CONTROLLED_INDICES)
            ]
            for name in LEFT_ACTUATORS + RIGHT_ACTUATORS:
                actuator_id = int(
                    mujoco.mj_name2id(
                        mujoco_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
                    )
                )
                if actuator_id < 0:
                    raise ValueError(f"ICRA model is missing actuator {name}")
                self._exo_actuator_ids[name] = actuator_id
        self.reset()

    def reset(self) -> None:
        """Reset phase/scheduler state without touching the live MuJoCo data."""
        self._alpha: float | None = None
        self._previous_left_heel: bool | None = None
        self._last_left_strike_s: float | None = None
        self._period_s = (
            1.15 if self.reference is None else float(self.reference.median_period_s)
        )
        self._reference_valid = False
        self._step_index = 0
        self._t1_rad_s = np.zeros(2, dtype=float)
        self._engagement_start_s: float | None = None
        self._learning_active = False
        self._critic_weight = np.zeros(3, dtype=float)
        self._critic_pending: dict[str, Any] | None = None
        self._model_pending: dict[str, Any] | None = None
        self._impedance_pending: dict[str, Any] | None = None
        self._critic_valid_samples = 0
        self._critic_skipped_samples = 0
        self._game_ea_rad: np.ndarray | None = None
        self._game_eadot_rad_s: np.ndarray | None = None
        self._game_pending_interaction_nm: np.ndarray | None = None
        self._ppc_start_s: float | None = None
        self._first_ppc_violation: dict[str, Any] | None = None
        self._stage6b_weight = np.zeros(CRITIC_WEIGHT_SHAPE, dtype=float)
        self._stage6b_learning_samples = 0
        self._stage6b_projection_count = np.zeros(2, dtype=int)
        self._stage6f_game_ramp_start_s: float | None = None
        self._stage6f_game_engaged = False
        self._stage6g_game_ramp_start_s: float | None = None
        self._stage6g_game_engaged = False
        self._stage7_held_strategy: np.ndarray | None = None
        self._stage7_vartheta = 0.0
        self._stage7_event_age_s = 0.0
        self._stage7_event_count = 0
        self._stage7_eligible_ticks = 0
        self._stage7_critic_updates = 0
        self._stage7_last_event_time_s: float | None = None
        self._stage7t_held_critic: np.ndarray | None = None
        self._stage7t_vartheta = 0.0
        self._stage7t_event_age_s = 0.0
        self._stage7t_event_count = 0
        self._stage7t_eligible_ticks = 0
        self._stage7t_critic_updates = 0
        self._stage7t_last_event_time_s: float | None = None
        self.termination_request = ""
        self.last_diagnostics: dict[str, Any] = {
            "icra_stage": 8 if self.mode in STAGE8A_MODES else 7 if self.mode in (*EVENT_EXECUTION_MODES, *STAGE7T_MODES) else 6 if self.mode in (
                *STAGE6A_MODES, *STAGE6B_MODES, *STAGE6E_MODES,
                *STAGE6F_MODES, *STAGE6G_MODES, *EVENT_EXECUTION_MODES,
                *STAGE7T_MODES,
                *STAGE8A_MODES,
            ) else (
                5 if self.mode in STAGE5_MODES else (
                4 if self.mode in STAGE4_MODES else (
                    3 if self.mode in STAGE3_MODES else (
                        2 if self.mode in STAGE2_MODES else (
                            1 if self.mode == SHADOW_PLANT_SCHEDULER_MODE else 0
                        )
                    )
                )
                )
            ),
            "mode": self.mode,
            "controlled_dof_count": 2,
            "reference_valid": False,
            "phase": "",
            "activation_rms": "",
            "scheduler_alpha_shadow": "",
            "K_H0": FIXED_K_HIP_NM_PER_RAD,
            "K_K0": FIXED_K_KNEE_NM_PER_RAD,
        }

    def _phase(self, time_s: float, contact: Mapping[str, Any]) -> tuple[Any, bool]:
        heel = bool(contact["left_heel"])
        strike = False
        if (
            self._previous_left_heel is not None
            and not self._previous_left_heel
            and heel
        ):
            elapsed = (
                math.inf
                if self._last_left_strike_s is None
                else time_s - self._last_left_strike_s
            )
            if self._last_left_strike_s is None:
                strike = True
                self._last_left_strike_s = time_s
                self._reference_valid = True
            elif MIN_STRIDE_PERIOD_S <= elapsed <= MAX_STRIDE_PERIOD_S:
                strike = True
                self._period_s = (
                    (1.0 - PHASE_EWMA_BETA) * self._period_s
                    + PHASE_EWMA_BETA * elapsed
                )
                self._last_left_strike_s = time_s
        self._previous_left_heel = heel
        if not self._reference_valid or self._last_left_strike_s is None:
            return "", strike
        return (
            float(
                np.clip(
                    (time_s - self._last_left_strike_s) / self._period_s,
                    0.0,
                    1.0,
                )
            ),
            strike,
        )

    def _scratch_constraint_signature(self, data: Any) -> tuple[str, ...]:
        if int(data.nefc) == 0:
            return ()
        limit_type = int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)
        labels: list[str] = []
        for row in range(int(data.nefc)):
            if int(data.efc_type[row]) != limit_type:
                continue
            joint_id = int(data.efc_id[row])
            for name, expected_id in self._exo_joint_ids.items():
                if joint_id == expected_id and float(data.efc_force[row]) > 1.0e-8:
                    labels.append(name)
                    break
        return tuple(sorted(labels))

    def _scratch_forward(
        self, snapshot: Mapping[str, Any], left_torque_nm: Sequence[float]
    ) -> tuple[np.ndarray, tuple[str, ...], Any]:
        if self._scratch is None or self._mujoco_model is None:
            raise RuntimeError("Scratch MuJoCo state is unavailable")
        data = self._scratch
        mujoco.mj_resetData(self._mujoco_model, data)
        data.time = _finite_float(snapshot["time"], "snapshot time")
        for name in (
            "qpos",
            "qvel",
            "act",
            "ctrl",
            "qacc_warmstart",
            "qfrc_applied",
            "xfrc_applied",
            "mocap_pos",
            "mocap_quat",
            "userdata",
        ):
            target = getattr(data, name)
            source = np.asarray(snapshot[name], dtype=float)
            if target.shape != source.shape:
                raise ValueError(f"Scratch snapshot shape mismatch for {name}")
            target[...] = source
        for actuator_id in self._exo_actuator_ids.values():
            data.ctrl[actuator_id] = 0.0
        torque = np.asarray(left_torque_nm, dtype=float).reshape(2)
        data.ctrl[self._exo_actuator_ids["HipExo_L"]] = torque[0]
        data.ctrl[self._exo_actuator_ids["KneeExo_L"]] = torque[1]
        mujoco.mj_forward(self._mujoco_model, data)
        if self._left_dof_addresses is None:
            raise RuntimeError("ICRA left DOF addresses are unavailable")
        qacc = np.asarray(data.qacc[self._left_dof_addresses], dtype=float).copy()
        return qacc, self._scratch_constraint_signature(data), data

    def _oracle_diagnostics(
        self, snapshot: Mapping[str, Any], plant_result: Mapping[str, Any]
    ) -> dict[str, Any]:
        baseline_qacc, baseline_mode, baseline_data = self._scratch_forward(
            snapshot, (0.0, 0.0)
        )
        applied = (
            np.asarray(baseline_data.qfrc_actuator, dtype=float)
            + np.asarray(baseline_data.qfrc_applied, dtype=float)
        )
        if np.max(np.abs(baseline_data.xfrc_applied)) > 1.0e-12:
            raise RuntimeError("Stage 1 inverse check requires zero xfrc_applied")
        forward_qacc = np.asarray(baseline_data.qacc, dtype=float).copy()
        baseline_data.qacc[:] = forward_qacc
        mujoco.mj_inverse(self._mujoco_model, baseline_data)
        inverse_residual = np.asarray(
            baseline_data.qfrc_inverse, dtype=float
        ) - applied
        inverse_finite = bool(
            np.isfinite(
                np.concatenate((forward_qacc, baseline_data.qfrc_inverse))
            ).all()
        )

        model_map = np.linalg.inv(np.asarray(plant_result["M_eff"], dtype=float))
        result: dict[str, Any] = {
            "oracle_sampled": True,
            "oracle_baseline_mode": "+".join(baseline_mode) or "none",
            "inverse_finite": inverse_finite,
            "inverse_residual_rms": float(
                np.sqrt(np.mean(inverse_residual * inverse_residual))
            ),
            "inverse_residual_max_abs": float(np.max(np.abs(inverse_residual))),
        }
        for delta in ORACLE_DELTAS_NM:
            columns = []
            modes = []
            for joint in range(2):
                unit = np.zeros(2, dtype=float)
                unit[joint] = delta
                plus, plus_mode, _ = self._scratch_forward(snapshot, unit)
                minus, minus_mode, _ = self._scratch_forward(snapshot, -unit)
                columns.append((plus - minus) / (2.0 * delta))
                modes.extend((plus_mode, minus_mode))
            finite_difference = np.column_stack(columns)
            switched = any(mode != baseline_mode for mode in modes)
            token = str(delta).replace(".", "p")
            result[f"oracle_mode_switch_{token}"] = switched
            result[f"oracle_relative_error_{token}"] = (
                ""
                if switched
                else float(
                    np.linalg.norm(finite_difference - model_map)
                    / (np.linalg.norm(finite_difference) + 1.0e-12)
                )
            )

        absolute_errors: list[float] = []
        relative_errors: list[float] = []
        test_switches = 0
        b_no_hri = np.asarray(plant_result["b_noHRI"], dtype=float)
        tau_h_eff = np.asarray(plant_result["tau_h_eff"], dtype=float)
        for torque in ORACLE_TEST_TORQUES_NM:
            command = np.asarray(torque, dtype=float)
            actual, mode, _ = self._scratch_forward(snapshot, command)
            if mode != baseline_mode:
                test_switches += 1
                continue
            predicted = model_map @ (command + tau_h_eff - b_no_hri)
            error = float(np.linalg.norm(actual - predicted))
            absolute_errors.append(error)
            relative_errors.append(error / (float(np.linalg.norm(actual)) + 1.0e-12))
        result.update(
            {
                "oracle_test_smooth_count": len(absolute_errors),
                "oracle_test_mode_switch_fraction": test_switches
                / len(ORACLE_TEST_TORQUES_NM),
                "oracle_test_abs_error_rms": (
                    ""
                    if not absolute_errors
                    else float(np.sqrt(np.mean(np.square(absolute_errors))))
                ),
                "oracle_test_relative_error_rms": (
                    ""
                    if not relative_errors
                    else float(np.sqrt(np.mean(np.square(relative_errors))))
                ),
                "oracle_test_relative_error_max": (
                    "" if not relative_errors else float(max(relative_errors))
                ),
            }
        )
        return result

    def _fixed_transform(
        self,
        q: np.ndarray,
        qdot: np.ndarray,
        qacc: np.ndarray,
        tau_h_eff: np.ndarray,
        phase: float,
        dt: float,
    ) -> dict[str, np.ndarray | float]:
        """Evaluate the constant-root impedance transform and exact-ZOH t1 step."""
        if self.reference is None:
            raise RuntimeError("Fixed-K transform requires the Healthy reference")
        qd, qdotd, qddotd = self.reference.evaluate(phase, self._period_s)
        error = q - qd
        error_velocity = qdot - qdotd
        t1 = self._t1_rad_s.copy()
        t1_dot = tau_h_eff / DESIRED_INERTIA_KGM2 - FIXED_GAMMA_S_INV * t1
        qdot_r = qdotd - FIXED_LAMBDA_S_INV * error + t1
        z = qdot - qdot_r
        qddot_r = (
            qddotd - FIXED_LAMBDA_S_INV * error_velocity + t1_dot
        )
        impedance_residual = (
            qacc
            - qddotd
            + FIXED_D_S_INV * error_velocity
            + FIXED_K_S_INV2 * error
            - tau_h_eff / DESIRED_INERTIA_KGM2
        )
        transformed_residual = qacc - qddot_r + FIXED_GAMMA_S_INV * z
        mismatch = impedance_residual - transformed_residual
        decay = np.exp(-FIXED_GAMMA_S_INV * dt)
        next_t1 = decay * t1 + (1.0 - decay) * (
            tau_h_eff / (DESIRED_INERTIA_KGM2 * FIXED_GAMMA_S_INV)
        )
        values = np.concatenate(
            (
                qd,
                qdotd,
                qddotd,
                error,
                error_velocity,
                t1,
                t1_dot,
                qdot_r,
                qddot_r,
                z,
                mismatch,
                next_t1,
            )
        )
        if not np.isfinite(values).all():
            raise FloatingPointError("Fixed-K transform contains NaN or Inf")
        self._t1_rad_s = next_t1
        return {
            "qd": qd,
            "qdotd": qdotd,
            "qddotd": qddotd,
            "error": error,
            "error_velocity": error_velocity,
            "t1": t1,
            "t1_dot": t1_dot,
            "qdot_r": qdot_r,
            "qddot_r": qddot_r,
            "z": z,
            "mismatch": mismatch,
            "mismatch_norm": float(np.linalg.norm(mismatch)),
        }

    def _engagement_ramp(self, time_s: float) -> float:
        if self._engagement_start_s is None:
            self._engagement_start_s = time_s
        elapsed = max(0.0, time_s - self._engagement_start_s)
        if elapsed >= FIXED_ENGAGEMENT_RAMP_S:
            return 1.0
        return float(
            0.5
            - 0.5 * math.cos(math.pi * elapsed / FIXED_ENGAGEMENT_RAMP_S)
        )

    def _complete_pending_critic(
        self, qacc_realized: np.ndarray, dt: float
    ) -> dict[str, Any]:
        """Complete interval k-1 using qacc first observed at control call k."""
        result: dict[str, Any] = {
            "critic_sample_completed": False,
            "bellman": "",
            "m_norm": "",
            "critic_update_norm": 0.0,
            "critic_backtracking_count": 0,
            "critic_update_skipped": False,
        }
        pending = self._critic_pending
        if pending is None:
            return result
        qddot_r = np.asarray(pending["qddot_r"], dtype=float)
        z = np.asarray(pending["z"], dtype=float)
        g = np.asarray(pending["g"], dtype=float)
        mu_applied = np.asarray(pending["mu_applied"], dtype=float)
        mu_policy = np.asarray(pending["mu_policy"], dtype=float)
        jacobian = np.asarray(pending["jacobian"], dtype=float)
        gradient = np.asarray(pending["gradient"], dtype=float)
        weight = np.asarray(pending["weight"], dtype=float)
        zdot = np.asarray(qacc_realized, dtype=float) - qddot_r
        f_hat = zdot - g @ mu_applied
        closed_loop_direction = f_hat + g @ mu_policy
        bellman = float(
            gradient @ closed_loop_direction
            + z @ FIXED_ADP_Q @ z
            + mu_policy @ FIXED_ADP_R @ mu_policy
        )
        regressor = jacobian.T @ closed_loop_direction
        denominator = (1.0 + float(regressor @ regressor)) ** 2
        proposed = -dt * FIXED_ADP_KAPPA * regressor * bellman / denominator
        if not np.isfinite(
            np.concatenate((zdot, f_hat, closed_loop_direction, proposed, [bellman]))
        ).all():
            raise RuntimeError("Fixed-K aligned Critic sample is non-finite")
        accepted: np.ndarray | None = None
        accepted_count = 0
        for count in range(FIXED_MAX_BACKTRACKS + 1):
            candidate = weight + (0.5**count) * proposed
            if (
                np.isfinite(candidate).all()
                and fixed_value_minimum_eigenvalue(candidate)
                >= -FIXED_VALUE_EIGEN_TOLERANCE
            ):
                accepted = candidate
                accepted_count = count
                break
        skipped = accepted is None
        if skipped:
            accepted = weight
            accepted_count = FIXED_MAX_BACKTRACKS
            self._critic_skipped_samples += 1
        self._critic_valid_samples += 1
        self._critic_weight = accepted.copy()
        applied_update = accepted - weight
        result.update(
            {
                "critic_sample_completed": True,
                "bellman": bellman,
                "m_norm": float(np.linalg.norm(regressor)),
                "critic_update_norm": float(np.linalg.norm(applied_update)),
                "critic_backtracking_count": accepted_count,
                "critic_update_skipped": skipped,
            }
        )
        return result

    def _complete_model_pending(self, qacc_realized: np.ndarray) -> dict[str, Any]:
        """Evaluate the previous interval's uncancelled physical disturbance."""
        result: dict[str, Any] = {
            "disturbance_sample_completed": False,
            "d_residual_hip": "",
            "d_residual_knee": "",
            "d_residual_norm": "",
            "d_residual_phase": "",
            "d_residual_constraint_active": False,
            "d_z_hip": "",
            "d_z_knee": "",
            "d_z_minus_residual_norm": "",
            "d_z_accounting_expected": False,
        }
        pending = self._model_pending
        if pending is None:
            return result
        mass = np.asarray(pending["M_smooth"], dtype=float)
        bias = np.asarray(pending["b_smooth"], dtype=float)
        torque = np.asarray(pending["u_actual"], dtype=float)
        interaction = np.asarray(pending["tau_h_smooth"], dtype=float)
        qddot_r = np.asarray(pending["qddot_r"], dtype=float)
        z = np.asarray(pending["z"], dtype=float)
        qacc = np.asarray(qacc_realized, dtype=float)
        residual = mass @ qacc + bias - torque - interaction
        d_z = (
            mass @ (qacc - qddot_r)
            + FIXED_KEN_NMS_PER_RAD @ z
            - interaction
        )
        mismatch = d_z - residual
        if not np.isfinite(np.concatenate((residual, d_z, mismatch))).all():
            raise RuntimeError("ICRA smooth-model disturbance audit is non-finite")
        result.update(
            {
                "disturbance_sample_completed": True,
                "d_residual_hip": float(residual[0]),
                "d_residual_knee": float(residual[1]),
                "d_residual_norm": float(np.linalg.norm(residual)),
                "d_residual_phase": float(pending["phase"]),
                "d_residual_constraint_active": bool(
                    pending["constraint_active"]
                ),
                "d_z_hip": float(d_z[0]),
                "d_z_knee": float(d_z[1]),
                "d_z_minus_residual_norm": float(np.linalg.norm(mismatch)),
                "d_z_accounting_expected": bool(pending["accounting_expected"]),
            }
        )
        return result

    def _complete_impedance_pending(
        self, qacc_realized: np.ndarray
    ) -> dict[str, Any]:
        """Complete the previous Stage-5 interval's impedance identity."""
        result: dict[str, Any] = {
            "impedance_sample_completed": False,
            "impedance_accounting_expected": False,
            "impedance_residual_hip": "",
            "impedance_residual_knee": "",
            "impedance_residual_norm": "",
            "impedance_phase": "",
            "impedance_constraint_active": False,
            "stage5_d_residual_hip": "",
            "stage5_d_residual_knee": "",
            "stage5_d_residual_norm": "",
        }
        pending = self._impedance_pending
        if pending is None:
            return result
        mass = np.asarray(pending["M_smooth"], dtype=float).reshape(2, 2)
        bias = np.asarray(pending["b_smooth"], dtype=float).reshape(2)
        torque = np.asarray(pending["u_actual"], dtype=float).reshape(2)
        interaction = np.asarray(
            pending["tau_h_smooth"], dtype=float
        ).reshape(2)
        qddot_d = np.asarray(pending["qddot_d"], dtype=float).reshape(2)
        error = np.asarray(pending["error"], dtype=float).reshape(2)
        error_velocity = np.asarray(
            pending["error_velocity"], dtype=float
        ).reshape(2)
        qacc = np.asarray(qacc_realized, dtype=float).reshape(2)
        strategic_mu_i = np.asarray(
            pending.get("strategic_muI", np.zeros(2, dtype=float)), dtype=float
        ).reshape(2)
        disturbance = mass @ qacc + bias - torque - interaction
        impedance_residual = (
            mass @ (qacc - qddot_d)
            + GAME_DI_NMS_PER_RAD @ error_velocity
            + GAME_KI_NM_PER_RAD * error
            - interaction
            - disturbance
            - strategic_mu_i
        )
        values = np.concatenate((disturbance, impedance_residual))
        if not np.isfinite(values).all():
            raise FloatingPointError("Stage-5 impedance identity is non-finite")
        result.update(
            {
                "impedance_sample_completed": True,
                "impedance_accounting_expected": bool(
                    pending["accounting_expected"]
                ),
                "impedance_residual_hip": float(impedance_residual[0]),
                "impedance_residual_knee": float(impedance_residual[1]),
                "impedance_residual_norm": float(
                    np.linalg.norm(impedance_residual)
                ),
                "impedance_phase": float(pending["phase"]),
                "impedance_constraint_active": bool(
                    pending["constraint_active"]
                ),
                "stage5_d_residual_hip": float(disturbance[0]),
                "stage5_d_residual_knee": float(disturbance[1]),
                "stage5_d_residual_norm": float(np.linalg.norm(disturbance)),
            }
        )
        return result

    def _stage5_defaults(self) -> dict[str, Any]:
        """Return a flat schema for the active ICRA impedance baseline."""
        active = self.mode in (
            *STAGE5_MODES, *STAGE6A_MODES, *STAGE6B_MODES, *STAGE6E_MODES,
            *STAGE6F_MODES, *STAGE6G_MODES, *EVENT_EXECUTION_MODES,
            *STAGE7T_MODES,
            *STAGE8A_MODES,
        )
        result: dict[str, Any] = {
            "stage5_impedance_active": active,
            "stage5_physical_source": (
                "shadow_only_zero_physical_torque"
                if self.mode in (
                    *STAGE6E_MODES, PHYSICAL_ACTUATION_GAME_SHADOW_MODE,
                    INTENTION_TARGET_GAME_SHADOW_MODE,
                )
                else "mu0_plus_stage6f_muI"
                if self.mode == PHYSICAL_ACTUATION_GAME_ACTIVE_MODE
                else "mu0_plus_stage6g_muI"
                if self.mode == INTENTION_TARGET_GAME_ACTIVE_MODE
                else "mu0_current_plus_event_held_stage6g_muI"
                if self.mode in EVENT_EXECUTION_MODES
                else "mu0_current_plus_current_affine_and_event_held_critic_muI"
                if self.mode in STAGE7T_MODES
                else "mu0_plus_stage6g_muI_with_experimental_ppc_pt"
                if self.mode in STAGE8A_MODES
                else "mu0_component" if self.mode in STAGE6B_MODES
                else "mu0" if active else ""
            ),
            "stage5_muA_hip": 0.0,
            "stage5_muA_knee": 0.0,
            "stage5_muI_hip": 0.0,
            "stage5_muI_knee": 0.0,
            "stage5_game_physical_contribution_nm": 0.0,
            "stage5_critic78_physical_contribution_nm": 0.0,
            "stage5_fixed_k_adp_residual_contribution_nm": 0.0,
            "stage5_nash_active": False,
            "stage5_critic78_learning_active": False,
            "stage5_event_active": False,
            "stage5_output_blending_active": False,
            "stage5_hard_kkt_feedforward_used": False,
            "stage5_m5_cancelled": False,
            "stage5_joint_limit_cancelled": False,
            "stage5_qfrc_passive_cancelled": False,
            "stage5_right_motor_zero": True,
            "stage6a_shadow_active": self.mode in STAGE6A_MODES,
            "stage6a_shadow_physical_contribution_nm": 0.0,
            "stage6a_critic_physical_applied": False,
            "stage6a_nash_physical_applied": False,
            "stage6a_event_physical_applied": False,
            "impedance_sample_completed": False,
            "impedance_accounting_expected": False,
            "impedance_constraint_active": False,
        }
        blank = (
            "stage5_q_ref_hip", "stage5_q_ref_knee",
            "stage5_qdot_ref_hip", "stage5_qdot_ref_knee",
            "stage5_qddot_ref_hip", "stage5_qddot_ref_knee",
            "stage5_error_hip", "stage5_error_knee",
            "stage5_error_velocity_hip", "stage5_error_velocity_knee",
            "stage5_u_inertia_hip", "stage5_u_inertia_knee",
            "stage5_u_bias_hip", "stage5_u_bias_knee",
            "stage5_u_damping_hip", "stage5_u_damping_knee",
            "stage5_u_stiffness_hip", "stage5_u_stiffness_knee",
            "stage5_mu0_hip", "stage5_mu0_knee",
            "stage5_command_hip", "stage5_command_knee",
            "stage5_engagement_ramp",
            "impedance_residual_hip", "impedance_residual_knee",
            "impedance_residual_norm", "impedance_phase",
            "stage5_d_residual_hip", "stage5_d_residual_knee",
            "stage5_d_residual_norm",
        )
        result.update({name: "" for name in blank})
        return result

    def _stage6b_defaults(self) -> dict[str, Any]:
        """Return a stable flat schema for continuous active IS-CGF."""
        active = self.mode in (
            *STAGE6B_MODES, *STAGE6E_MODES, *STAGE6F_MODES,
            *STAGE6G_MODES, *EVENT_EXECUTION_MODES, *STAGE7T_MODES,
            *STAGE8A_MODES,
        )
        result: dict[str, Any] = {
            "stage6b_active": active,
            "stage6b_control_law": "mu0_plus_muI" if active else "",
            "stage6b_continuous_strategy": active,
            "stage6b_event_active": False,
            "stage6b_muA_physically_applied": False,
            "stage6b_muI_physically_applied": self.mode in STAGE6B_MODES,
            "stage6b_learning_active": False,
            "stage6b_causal_actor_uses_Wk": False,
            "stage6b_next_weight_not_used_for_current_actor": False,
            "stage6b_weight_radius": STAGE6B_WEIGHT_RADIUS,
            "stage6b_eta_c": STAGE6B_CRITIC_ETA,
            "stage6b_projection_A": False,
            "stage6b_projection_I": False,
            "stage6b_projection_count_A": self._stage6b_projection_count[0],
            "stage6b_projection_count_I": self._stage6b_projection_count[1],
            "stage6b_projection_fraction_A": 0.0,
            "stage6b_projection_fraction_I": 0.0,
            "stage6b_learning_sample_count": self._stage6b_learning_samples,
            "stage6b_W_A_current_norm": float(
                np.linalg.norm(self._stage6b_weight[:, 0])
            ),
            "stage6b_W_I_current_norm": float(
                np.linalg.norm(self._stage6b_weight[:, 1])
            ),
            "stage6b_W_A_next_norm": float(
                np.linalg.norm(self._stage6b_weight[:, 0])
            ),
            "stage6b_W_I_next_norm": float(
                np.linalg.norm(self._stage6b_weight[:, 1])
            ),
            "stage6b_muA_hip": 0.0,
            "stage6b_muA_knee": 0.0,
            "stage6b_muI_hip": 0.0,
            "stage6b_muI_knee": 0.0,
            "stage6b_mu0_hip": 0.0,
            "stage6b_mu0_knee": 0.0,
            "stage6b_raw_total_hip": 0.0,
            "stage6b_raw_total_knee": 0.0,
            "stage6b_command_hip": 0.0,
            "stage6b_command_knee": 0.0,
            "stage6b_delta_A": "",
            "stage6b_delta_I": "",
            "stage6b_bellman_vector_norm": "",
            "stage6b_weight_update_A_norm": 0.0,
            "stage6b_weight_update_I_norm": 0.0,
            "stage6b_q_A_chi": "",
            "stage6b_q_I_chi": "",
            "stage6b_r_A": "",
            "stage6b_r_I": "",
            "stage6b_ell_A_chi": "",
            "stage6b_ell_I_chi": "",
            "stage6b_omega_norm": "",
            "stage6b_P_A_lambda_min": "",
            "stage6b_P_A_lambda_max": "",
            "stage6b_P_A_trace": "",
            "stage6b_P_I_lambda_min": "",
            "stage6b_P_I_lambda_max": "",
            "stage6b_P_I_trace": "",
            "stage6b_muI_budget_exceeded": False,
            "stage6b_raw_20Nm_guard_exceeded": False,
            "stage6b_ppc_invalid": False,
            "stage6b_RN_SPD_failure": False,
            "stage6b_native_saturation": False,
            "stage6b_right_motor_nonzero": False,
            "stage6b_scientific_abort": False,
            "stage6b_scientific_abort_reason": "",
        }
        for player in ("A", "I"):
            for index in range(CHI_DIMENSION):
                result[f"stage6b_gradV{player}_chi_{index:02d}"] = ""
        return result

    def _stage6f_defaults(self) -> dict[str, Any]:
        """Return the isolated Stage-6F physical-actuation-aware schema."""
        active = self.mode in STAGE6F_MODES
        result: dict[str, Any] = {
            "stage6f_active": active,
            "stage6f_shadow_only": self.mode == PHYSICAL_ACTUATION_GAME_SHADOW_MODE,
            "stage6f_physical_active": self.mode == PHYSICAL_ACTUATION_GAME_ACTIVE_MODE,
            "stage6f_objective": (
                "physical_actuation_aware_I_player_regularization" if active else ""
            ),
            "stage6f_beta_exponent": 3 if active else "",
            "stage6f_beta": "",
            "stage6f_game_ramp": 0.0,
            "stage6f_game_ramp_active": False,
            "stage6f_weights_frozen_for_game_ramp": False,
            "stage6f_weights_reset_at_full_engagement": False,
            "stage6f_learning_active": False,
            "stage6f_mu0_hip": 0.0,
            "stage6f_mu0_knee": 0.0,
            "stage6f_muA_hip": 0.0,
            "stage6f_muA_knee": 0.0,
            "stage6f_muI_hip": 0.0,
            "stage6f_muI_knee": 0.0,
            "stage6f_proposed_hip": 0.0,
            "stage6f_proposed_knee": 0.0,
            "stage6f_physical_hip": 0.0,
            "stage6f_physical_knee": 0.0,
            "stage6f_r_I_original": "",
            "stage6f_r_total": "",
            "stage6f_r_total_expanded": "",
            "stage6f_r_total_expansion_error": "",
            "stage6f_ell_A": "",
            "stage6f_ell_I": "",
            "stage6f_FOC_residual_max_abs": "",
            "stage6f_RN_min_eigenvalue": "",
            "stage6f_RN_max_eigenvalue": "",
            "stage6f_RN_condition": "",
            "stage6f_projection_A": False,
            "stage6f_projection_I": False,
            "stage6f_projection_fraction_A": 0.0,
            "stage6f_projection_fraction_I": 0.0,
            "stage6f_W_A_norm": float(np.linalg.norm(self._stage6b_weight[:, 0])),
            "stage6f_W_I_norm": float(np.linalg.norm(self._stage6b_weight[:, 1])),
            "stage6f_delta_A": "",
            "stage6f_delta_I": "",
            "stage6f_abort": False,
            "stage6f_abort_reason": "",
            "stage6f_event_active": False,
        }
        for player in ("A", "I"):
            for index in range(CHI_DIMENSION):
                result[f"stage6f_gradV{player}_chi_{index:02d}"] = ""
        return result

    def _stage6g_defaults(self) -> dict[str, Any]:
        """Return the isolated Stage-6G diagnostic schema."""
        active=self.mode in (*STAGE6G_MODES, *EVENT_EXECUTION_MODES, *STAGE7T_MODES, *STAGE8A_MODES)
        result: dict[str,Any]={
            "stage6g_active":active,"stage6g_shadow_only":self.mode==INTENTION_TARGET_GAME_SHADOW_MODE,
            "stage6g_physical_active":self.mode in (INTENTION_TARGET_GAME_ACTIVE_MODE, *EVENT_EXECUTION_MODES, *STAGE7T_MODES, *STAGE8A_MODES),
            "stage6g_objective":"INTENTION-SCHEDULED PHYSICAL ASSISTANCE TARGET" if active else "",
            "stage6g_gamma":"","stage6g_game_ramp":0.0,"stage6g_game_ramp_active":False,
            "stage6g_weights_frozen_for_game_ramp":False,"stage6g_weights_reset_at_full_engagement":False,
            "stage6g_learning_active":False,"stage6g_mu0_hip":0.0,"stage6g_mu0_knee":0.0,
            "stage6g_mu_target_hip":0.0,"stage6g_mu_target_knee":0.0,
            "stage6g_muA_hip":0.0,"stage6g_muA_knee":0.0,"stage6g_muI_hip":0.0,"stage6g_muI_knee":0.0,
            "stage6g_muA_affine_hip":0.0,"stage6g_muA_affine_knee":0.0,"stage6g_muI_affine_hip":0.0,"stage6g_muI_affine_knee":0.0,
            "stage6g_muA_critic_hip":0.0,"stage6g_muA_critic_knee":0.0,"stage6g_muI_critic_hip":0.0,"stage6g_muI_critic_knee":0.0,
            "stage6g_strategy_decomposition_error_max_abs":"",
            "stage6g_affine_physical_hip":0.0,"stage6g_affine_physical_knee":0.0,
            "stage6g_affine_sign_pass":False,"stage6g_affine_magnitude_pass":False,"stage6g_affine_near_zero_pass":False,"stage6g_affine_safety_passed":False,
            "stage6g_legacy_total_budget_exceeded":False,"stage6g_corrected_critic_budget_exceeded":False,
            "stage6g_legacy_budget_applies_to":"CRITIC_COMPONENT_ONLY" if active else "",
            "stage6g_total_muI_old_budget":"DIAGNOSTIC_ONLY" if active else "",
            "stage6g_proposed_hip":0.0,"stage6g_proposed_knee":0.0,"stage6g_physical_hip":0.0,"stage6g_physical_knee":0.0,
            "stage6g_r_I_original":"","stage6g_r_target":"","stage6g_r_target_expanded":"","stage6g_r_target_expansion_error":"",
            "stage6g_ell_A":"","stage6g_ell_I":"","stage6g_FOC_residual_max_abs":"",
            "stage6g_RN_min_eigenvalue":"","stage6g_RN_max_eigenvalue":"","stage6g_RN_condition":"",
            "stage6g_projection_A":False,"stage6g_projection_I":False,"stage6g_projection_fraction_A":0.0,"stage6g_projection_fraction_I":0.0,
            "stage6g_W_A_norm":float(np.linalg.norm(self._stage6b_weight[:,0])),"stage6g_W_I_norm":float(np.linalg.norm(self._stage6b_weight[:,1])),
            "stage6g_delta_A":"","stage6g_delta_I":"","stage6g_abort":False,"stage6g_abort_reason":"","stage6g_event_active":False,
        }
        for player in ("A","I"):
            for index in range(CHI_DIMENSION): result[f"stage6g_gradV{player}_chi_{index:02d}"]=""
        return result

    def _stage7_defaults(self) -> dict[str, Any]:
        """Return the event-execution schema without changing Stage-6G science."""
        active = self.mode in EVENT_EXECUTION_MODES
        parameters = self._event_execution_parameters()
        result: dict[str, Any] = {
            "stage7_active": active,
            "stage7_event_logic_active": False,
            "stage7_event_accepted": False,
            "stage7_event_index": -1,
            "stage7_event_count": self._stage7_event_count,
            "stage7_eligible_tick_count": self._stage7_eligible_ticks,
            "stage7_critic_update_count": self._stage7_critic_updates,
            "stage7_critic_continuous_100hz": False,
            "stage7_current_strategy_reconstructed_100hz": False,
            "stage7_mu0_continuous_100hz": active,
            "stage7_held_strategy_exact_between_events": True,
            "stage7_trigger_uses_phase": False,
            "stage7_candidate": parameters["candidate"] if active else "",
            "stage7_theta0": parameters["theta0"] if active else "",
            "stage7_dwell_s": parameters["dwell_s"] if active else "",
            "stage7_lambda_v_s_inv": parameters["lambda_v"] if active else "",
            "stage7_kappa_e": parameters["kappa_e"] if active else "",
            "stage7_c_chi_evt": parameters["c_chi_evt"] if active else "",
            "stage7_cv": 1.0 if active else "",
            "stage7_vartheta": 0.0,
            "stage7_threshold": "",
            "stage7_xchi_norm_sq": "",
            "stage7_event_age_s": 0.0,
            "stage7_inter_event_interval_s": "",
            "stage7_dwell_violation": False,
            "stage7_update_reduction": "",
            "stage7_physical_source": (
                "mu0_current_plus_projection_I_event_held_Uh" if active else ""
            ),
            "stage7_physical_contract_error_max_abs": "",
            "stage7_raw_physical_hip": 0.0,
            "stage7_raw_physical_knee": 0.0,
            "stage7_abort": False,
            "stage7_abort_reason": "",
        }
        labels = ("muA_hip", "muA_knee", "muI_hip", "muI_knee")
        for index, label in enumerate(labels):
            result[f"stage7_DU_{label}"] = (
                float(STAGE7_STRATEGY_SCALE_NM[index]) if active else ""
            )
            result[f"stage7_Uhat_{label}"] = 0.0
            result[f"stage7_Uheld_{label}"] = 0.0
            result[f"stage7_e_evt_{label}"] = 0.0
            result[f"stage7_e_bar_{label}"] = 0.0
        result.update(
            {
                "stage7_e_bar_norm_sq": 0.0,
                "stage7_e_bar_norm": 0.0,
                "stage7_muI_hold_error_hip": 0.0,
                "stage7_muI_hold_error_knee": 0.0,
            }
        )
        return result

    def _stage7t_defaults(self) -> dict[str, Any]:
        """Return diagnostics for continuous-affine/residual-only execution."""
        active = self.mode in STAGE7T_MODES
        result: dict[str, Any] = {
            "stage7t_active": active,
            "stage7t_event_logic_active": False,
            "stage7t_event_accepted": False,
            "stage7t_event_index": -1,
            "stage7t_event_count": self._stage7t_event_count,
            "stage7t_eligible_tick_count": self._stage7t_eligible_ticks,
            "stage7t_critic_update_count": self._stage7t_critic_updates,
            "stage7t_critic_continuous_100hz": False,
            "stage7t_affine_continuous_100hz": False,
            "stage7t_current_critic_reconstructed_100hz": False,
            "stage7t_mu0_continuous_100hz": active,
            "stage7t_only_critic_residual_held": active,
            "stage7t_held_critic_exact_between_events": True,
            "stage7t_trigger_uses_phase": False,
            "stage7t_trigger_uses_affine_error": False,
            "stage7t_trigger_uses_mu0_error": False,
            "stage7t_candidate": "p25_Td20ms_residual" if active else "",
            "stage7t_theta0": STAGE7T_THETA0 if active else "",
            "stage7t_dwell_s": STAGE7T_DWELL_S if active else "",
            "stage7t_lambda_v_s_inv": STAGE7T_LAMBDA_V_S_INV if active else "",
            "stage7t_kappa_e": STAGE7T_KAPPA_E if active else "",
            "stage7t_c_chi_evt": STAGE7T_C_CHI_EVT if active else "",
            "stage7t_cv": 1.0 if active else "",
            "stage7t_vartheta": 0.0,
            "stage7t_threshold": "",
            "stage7t_xchi_norm_sq": "",
            "stage7t_event_age_s": 0.0,
            "stage7t_inter_event_interval_s": "",
            "stage7t_dwell_violation": False,
            "stage7t_residual_refresh_reduction": "",
            "stage7t_physical_source": (
                "mu0_current_plus_projection_I_current_affine_plus_event_held_critic"
                if active else ""
            ),
            "stage7t_decomposition_contract_error_max_abs": "",
            "stage7t_affine_execution_contract_error_max_abs": "",
            "stage7t_physical_contract_error_max_abs": "",
            "stage7t_raw_physical_hip": 0.0,
            "stage7t_raw_physical_knee": 0.0,
            "stage7t_abort": False,
            "stage7t_abort_reason": "",
            "stage7t_e_bar_res_norm_sq": 0.0,
            "stage7t_e_bar_res_norm": 0.0,
        }
        for index, label in enumerate(
            ("muA_hip", "muA_knee", "muI_hip", "muI_knee")
        ):
            result[f"stage7t_Dres_{label}"] = (
                float(STAGE7T_RESIDUAL_SCALE_NM[index]) if active else ""
            )
            result[f"stage7t_Uaffine_current_{label}"] = 0.0
            result[f"stage7t_Ucritic_current_{label}"] = 0.0
            result[f"stage7t_Ucritic_held_{label}"] = 0.0
            result[f"stage7t_Uexec_{label}"] = 0.0
            result[f"stage7t_e_res_{label}"] = 0.0
            result[f"stage7t_e_bar_res_{label}"] = 0.0
        return result

    def _event_execution_parameters(self) -> dict[str, float | str]:
        if self.mode in STAGE7S_MODES:
            return {
                "candidate": "p25_Td20ms",
                "theta0": STAGE7S_THETA0,
                "dwell_s": STAGE7S_DWELL_S,
                "lambda_v": STAGE7S_LAMBDA_V_S_INV,
                "kappa_e": STAGE7S_KAPPA_E,
                "c_chi_evt": STAGE7S_C_CHI_EVT,
            }
        if self.mode in STAGE7R_MODES:
            return {
                "candidate": "p50_Td20ms",
                "theta0": STAGE7R_THETA0,
                "dwell_s": STAGE7R_DWELL_S,
                "lambda_v": STAGE7R_LAMBDA_V_S_INV,
                "kappa_e": STAGE7R_KAPPA_E,
                "c_chi_evt": STAGE7R_C_CHI_EVT,
            }
        return {
            "candidate": "p75_Td20ms",
            "theta0": STAGE7_THETA0,
            "dwell_s": STAGE7_DWELL_S,
            "lambda_v": STAGE7_LAMBDA_V_S_INV,
            "kappa_e": STAGE7_KAPPA_E,
            "c_chi_evt": STAGE7_C_CHI_EVT,
        }

    def _game_shadow_diagnostics(
        self,
        *,
        error_i: np.ndarray,
        error_velocity_i: np.ndarray,
        smooth_mass: np.ndarray,
        smooth_interaction: np.ndarray,
        alpha: float,
        fixed_k_command: np.ndarray,
        dt: float,
    ) -> dict[str, Any]:
        """Advance the observational dual endpoint without changing torque."""
        e_i = np.asarray(error_i, dtype=float).reshape(2)
        edot_i = np.asarray(error_velocity_i, dtype=float).reshape(2)
        mass = np.asarray(smooth_mass, dtype=float).reshape(2, 2)
        interaction = np.asarray(smooth_interaction, dtype=float).reshape(2)
        command = np.asarray(fixed_k_command, dtype=float).reshape(2)
        scheduler_alpha = _finite_float(alpha, "game scheduler alpha")
        initialized = self._game_ea_rad is None
        advanced = False
        if initialized:
            # [PLATFORM INITIALIZATION] Match the measured endpoint at the
            # first reference-valid sample to avoid an invented disagreement.
            self._game_ea_rad = e_i.copy()
            self._game_eadot_rad_s = edot_i.copy()
        else:
            if (
                self._game_eadot_rad_s is None
                or self._game_pending_interaction_nm is None
            ):
                raise RuntimeError("Stage-3 admittance pending state is incomplete")
            self._game_ea_rad, self._game_eadot_rad_s = (
                admittance_exact_zoh_step(
                    self._game_ea_rad,
                    self._game_eadot_rad_s,
                    self._game_pending_interaction_nm,
                    dt,
                )
            )
            advanced = True
        self._game_pending_interaction_nm = interaction.copy()
        e_a = np.asarray(self._game_ea_rad, dtype=float)
        edot_a = np.asarray(self._game_eadot_rad_s, dtype=float)
        z_a = np.concatenate((e_a, edot_a))
        z_i = np.concatenate((e_i, edot_i))
        disagreement = z_a - z_i
        tau_int = GAME_K0_NM_PER_RAD * (e_a - e_i)
        tau_int_dot = GAME_K0_NM_PER_RAD * (edot_a - edot_i)

        identity = np.eye(2, dtype=float)
        mass_a = np.diag(GAME_MA_KGM2)
        f_a_acceleration = -np.linalg.solve(
            mass_a,
            GAME_KA_NM_PER_RAD * e_a + GAME_DA_NMS_PER_RAD * edot_a,
        )
        g_a_acceleration = np.linalg.solve(mass_a, identity)
        f_i_acceleration = -np.linalg.solve(
            mass,
            GAME_KI_NM_PER_RAD * e_i + GAME_DI_NMS_PER_RAD @ edot_i,
        )
        g_i_acceleration = np.linalg.solve(mass, identity)
        f_x = np.concatenate(
            (edot_a, f_a_acceleration, edot_i, f_i_acceleration)
        )
        game_g_a = np.zeros((8, 2), dtype=float)
        game_g_i = np.zeros((8, 2), dtype=float)
        game_g_a[2:4] = g_a_acceleration
        game_g_i[6:8] = g_i_acceleration

        eta = game_eta(scheduler_alpha)
        nash = game_nash_matrix(scheduler_alpha)
        nash_eigenvalues = np.linalg.eigvalsh(nash)
        nash_condition = float(np.linalg.cond(nash))
        test_a = np.asarray((0.3, -0.2), dtype=float)
        test_i = np.asarray((-0.1, 0.4), dtype=float)
        test_strategy = np.concatenate((test_a, test_i))
        quadratic_lhs = float(test_strategy @ nash @ test_strategy)
        quadratic_rhs = float(
            scheduler_alpha * test_a @ GAME_RA @ test_a
            + (1.0 - scheduler_alpha) * test_i @ GAME_RI @ test_i
            + eta * (test_a - test_i) @ GAME_HMU @ (test_a - test_i)
        )
        values = np.concatenate(
            (
                z_a,
                z_i,
                disagreement,
                tau_int,
                tau_int_dot,
                f_x,
                game_g_a.reshape(-1),
                game_g_i.reshape(-1),
                nash.reshape(-1),
                nash_eigenvalues,
            )
        )
        if not np.isfinite(values).all() or np.any(nash_eigenvalues <= 0.0):
            raise RuntimeError("Stage-3 game shadow state or Nash matrix is invalid")

        return {
            "game_shadow_active": True,
            "game_physical_contribution_nm": 0.0,
            "fixed_k_adp_physical_source": self.mode in (*STAGE3_MODES, *STAGE4_MODES),
            "game_state_order": "eA_H,eA_K,eAdotA_H,eAdotA_K,eI_H,eI_K,edotI_H,edotI_K",
            "game_strategy_order": "muA_H,muA_K,muI_H,muI_K",
            "game_endpoint_initialized": initialized,
            "game_endpoint_advanced": advanced,
            "alpha": scheduler_alpha,
            "eta": eta,
            "KA_hip": float(GAME_KA_NM_PER_RAD[0]),
            "KA_knee": float(GAME_KA_NM_PER_RAD[1]),
            "KI_hip": float(GAME_KI_NM_PER_RAD[0]),
            "KI_knee": float(GAME_KI_NM_PER_RAD[1]),
            "MA_hip": float(GAME_MA_KGM2[0]),
            "MA_knee": float(GAME_MA_KGM2[1]),
            "DA_hip": float(GAME_DA_NMS_PER_RAD[0]),
            "DA_knee": float(GAME_DA_NMS_PER_RAD[1]),
            "DI_hip": float(GAME_DI_NMS_PER_RAD[0, 0]),
            "DI_knee": float(GAME_DI_NMS_PER_RAD[1, 1]),
            "DI_00": float(GAME_DI_NMS_PER_RAD[0, 0]),
            "DI_01": float(GAME_DI_NMS_PER_RAD[0, 1]),
            "DI_10": float(GAME_DI_NMS_PER_RAD[1, 0]),
            "DI_11": float(GAME_DI_NMS_PER_RAD[1, 1]),
            "DI_full_spd": True,
            "DI_design": "sampled_data_modal_zeta_0p90",
            "eA_hip": float(e_a[0]),
            "eA_knee": float(e_a[1]),
            "eAdotA_hip": float(edot_a[0]),
            "eAdotA_knee": float(edot_a[1]),
            "eI_hip": float(e_i[0]),
            "eI_knee": float(e_i[1]),
            "edotI_hip": float(edot_i[0]),
            "edotI_knee": float(edot_i[1]),
            "ec_e_hip": float(disagreement[0]),
            "ec_e_knee": float(disagreement[1]),
            "ec_edot_hip": float(disagreement[2]),
            "ec_edot_knee": float(disagreement[3]),
            "tau_int_hip": float(tau_int[0]),
            "tau_int_knee": float(tau_int[1]),
            "tau_int_dot_hip": float(tau_int_dot[0]),
            "tau_int_dot_knee": float(tau_int_dot[1]),
            "RN_min_eigenvalue": float(nash_eigenvalues[0]),
            "RN_max_eigenvalue": float(nash_eigenvalues[-1]),
            "RN_condition": nash_condition,
            "RN_quadratic_identity_error": abs(quadratic_lhs - quadratic_rhs),
            "fA_accel_hip": float(f_a_acceleration[0]),
            "fA_accel_knee": float(f_a_acceleration[1]),
            "fI_accel_hip": float(f_i_acceleration[0]),
            "fI_accel_knee": float(f_i_acceleration[1]),
            "gA_00": float(g_a_acceleration[0, 0]),
            "gA_01": float(g_a_acceleration[0, 1]),
            "gA_10": float(g_a_acceleration[1, 0]),
            "gA_11": float(g_a_acceleration[1, 1]),
            "gI_00": float(g_i_acceleration[0, 0]),
            "gI_01": float(g_i_acceleration[0, 1]),
            "gI_10": float(g_i_acceleration[1, 0]),
            "gI_11": float(g_i_acceleration[1, 1]),
            "hI_equals_gI": True,
            "game_GA_fro": float(np.linalg.norm(game_g_a)),
            "game_GI_fro": float(np.linalg.norm(game_g_i)),
            "fixed_k_u_total_hip": float(command[0]),
            "fixed_k_u_total_knee": float(command[1]),
            "game_u_total_hip": 0.0,
            "game_u_total_knee": 0.0,
            "game_eA_position_bounded": bool(np.max(np.abs(e_a)) < 1.0),
            "game_eA_velocity_bounded": bool(np.max(np.abs(edot_a)) < 20.0),
        }

    def _game_shadow_defaults(self) -> dict[str, Any]:
        """Return a stable flat schema before the first valid gait reference."""
        active = self.mode in GAME_DIAGNOSTIC_MODES
        result: dict[str, Any] = {
            "game_shadow_active": active,
            "game_physical_contribution_nm": 0.0,
            "fixed_k_adp_physical_source": self.mode in (
                *STAGE3_MODES, *STAGE4_MODES
            ),
            "game_state_order": "eA_H,eA_K,eAdotA_H,eAdotA_K,eI_H,eI_K,edotI_H,edotI_K",
            "game_strategy_order": "muA_H,muA_K,muI_H,muI_K",
            "game_endpoint_initialized": False,
            "game_endpoint_advanced": False,
            "alpha": "" if self._alpha is None else float(self._alpha),
            "eta": "" if self._alpha is None else game_eta(self._alpha),
            "KA_hip": float(GAME_KA_NM_PER_RAD[0]),
            "KA_knee": float(GAME_KA_NM_PER_RAD[1]),
            "KI_hip": float(GAME_KI_NM_PER_RAD[0]),
            "KI_knee": float(GAME_KI_NM_PER_RAD[1]),
            "MA_hip": float(GAME_MA_KGM2[0]),
            "MA_knee": float(GAME_MA_KGM2[1]),
            "DA_hip": float(GAME_DA_NMS_PER_RAD[0]),
            "DA_knee": float(GAME_DA_NMS_PER_RAD[1]),
            "DI_hip": float(GAME_DI_NMS_PER_RAD[0, 0]),
            "DI_knee": float(GAME_DI_NMS_PER_RAD[1, 1]),
            "DI_00": float(GAME_DI_NMS_PER_RAD[0, 0]),
            "DI_01": float(GAME_DI_NMS_PER_RAD[0, 1]),
            "DI_10": float(GAME_DI_NMS_PER_RAD[1, 0]),
            "DI_11": float(GAME_DI_NMS_PER_RAD[1, 1]),
            "DI_full_spd": True,
            "DI_design": "sampled_data_modal_zeta_0p90",
            "hI_equals_gI": active,
            "fixed_k_u_total_hip": 0.0,
            "fixed_k_u_total_knee": 0.0,
            "game_u_total_hip": 0.0,
            "game_u_total_knee": 0.0,
            "game_eA_position_bounded": True,
            "game_eA_velocity_bounded": True,
        }
        blank = (
            "eA_hip",
            "eA_knee",
            "eAdotA_hip",
            "eAdotA_knee",
            "eI_hip",
            "eI_knee",
            "edotI_hip",
            "edotI_knee",
            "ec_e_hip",
            "ec_e_knee",
            "ec_edot_hip",
            "ec_edot_knee",
            "tau_int_hip",
            "tau_int_knee",
            "tau_int_dot_hip",
            "tau_int_dot_knee",
            "RN_min_eigenvalue",
            "RN_max_eigenvalue",
            "RN_condition",
            "RN_quadratic_identity_error",
            "fA_accel_hip",
            "fA_accel_knee",
            "fI_accel_hip",
            "fI_accel_knee",
            "gA_00",
            "gA_01",
            "gA_10",
            "gA_11",
            "gI_00",
            "gI_01",
            "gI_10",
            "gI_11",
            "game_GA_fro",
            "game_GI_fro",
        )
        result.update({name: "" for name in blank})
        return result

    def _performance_shadow_defaults(self) -> dict[str, Any]:
        """Return the flat Stage-4 schema before PPC becomes active."""
        active = self.mode in (
            *STAGE4_MODES, *STAGE5_MODES, *STAGE6A_MODES, *STAGE6B_MODES,
            *STAGE6E_MODES,
            *STAGE6F_MODES, *STAGE6G_MODES, *EVENT_EXECUTION_MODES,
            *STAGE7T_MODES,
            *STAGE8A_MODES,
        )
        result: dict[str, Any] = {
            "performance_shadow_active": active,
            "ppc_physical_contribution_nm": 0.0,
            "mu0_shadow_physical_contribution_nm": 0.0,
            "critic78_physical_contribution_nm": 0.0,
            "ppc_direct_input_in_first_derivative": False,
            "critic78_learning_active": False,
            "critic78_weight_norm_A": 0.0,
            "critic78_weight_norm_I": 0.0,
            "performance_state_order": (
                "eA_H,eA_K,eAdotA_H,eAdotA_K,"
                "eI_H,eI_K,edotI_H,edotI_K"
            ),
            "performance_error_order": (
                "eI_H,eI_K,cP_tau_int_H,cP_tau_int_K"
            ),
            "chi_order": (
                "eA_H,eA_K,eAdotA_H,eAdotA_K,"
                "eI_H,eI_K,edotI_H,edotI_K,"
                "epsilon_H_track,epsilon_K_track,"
                "epsilon_H_int,epsilon_K_int"
            ),
            "chi_dimension": CHI_DIMENSION,
            "critic78_feature_count": CRITIC_FEATURE_COUNT,
            "critic78_jacobian_rows": CHI_DIMENSION,
            "critic78_jacobian_columns": CRITIC_FEATURE_COUNT,
            "Fchi_dimension": CHI_DIMENSION,
            "GAchi_rows": CHI_DIMENSION,
            "GAchi_columns": 2,
            "GIch_rows": CHI_DIMENSION,
            "GIch_columns": 2,
            "performance_cP": PERFORMANCE_C_P_RAD_PER_NM,
            "performance_lambda_A": PERFORMANCE_LAMBDA_A,
            "performance_lambda_I": PERFORMANCE_LAMBDA_I,
            "ppc_rho0": self._ppc_rho0,
            "ppc_rho_inf": self._ppc_rho_inf,
            "ppc_tight_candidate_rho0": PPC_RHO0_TIGHT_CANDIDATE,
            "ppc_tight_candidate_rho_inf": PPC_RHO_INF_TIGHT_CANDIDATE,
            "ppc_T_rho_s": self._ppc_t_rho_s,
            "ppc_nu": PPC_NU,
            "ppc_envelope_role": "development_only_not_publication_claim",
            "pt_Tp_s": self._pt_tp_s,
            "pt_delta": PT_DELTA,
            "pt_p_s_inv": PT_P_S_INV,
            "ppc_active": False,
            "ppc_valid": "",
            "ppc_activation_sample": False,
            "ppc_initial_condition_pass": "",
            "ppc_boundary_violation": False,
            "first_ppc_violation_time": "",
            "first_ppc_violation_channel": "",
            "first_ppc_violation_eP": "",
            "first_ppc_violation_rho": "",
            "first_ppc_violation_zeta": "",
            "chi_all_finite": "",
            "Fchi_all_finite": "",
        }
        blank = (
            "qA_state", "qI_state", "epsilon_norm_sq",
            "eP_track_hip", "eP_track_knee", "eP_int_hip", "eP_int_knee",
            "ePdot_track_hip", "ePdot_track_knee",
            "ePdot_int_hip", "ePdot_int_knee",
            "ppc_time_s", "rho", "rho_dot",
            "zeta_track_hip", "zeta_track_knee",
            "zeta_int_hip", "zeta_int_knee",
            "epsilon_track_hip", "epsilon_track_knee",
            "epsilon_int_hip", "epsilon_int_knee",
            "epsilon_dot_track_hip", "epsilon_dot_track_knee",
            "epsilon_dot_int_hip", "epsilon_dot_int_knee",
            "phi1", "phi1_dot", "phiT", "sT",
            "mu0_shadow_hip", "mu0_shadow_knee",
            "mu0_shadow_native_limit_counterfactual",
            "critic78_feature_norm", "critic78_jacobian_fro",
            "Fchi_norm", "GAchi_fro", "GIch_fro",
        )
        blank += tuple(f"chi_{index:02d}" for index in range(CHI_DIMENSION))
        blank += tuple(f"Fchi_{index:02d}" for index in range(CHI_DIMENSION))
        result.update({name: "" for name in blank})
        return result

    def _performance_shadow_diagnostics(
        self,
        *,
        game: Mapping[str, Any],
        qddot_reference: np.ndarray,
        smooth_mass: np.ndarray,
        smooth_bias: np.ndarray,
        ramp: float,
        time_s: float,
    ) -> dict[str, Any]:
        """Evaluate Stage-4 costs and geometry without changing motor torque."""
        z_a = np.asarray(
            (
                game["eA_hip"], game["eA_knee"],
                game["eAdotA_hip"], game["eAdotA_knee"],
            ),
            dtype=float,
        )
        z_i = np.asarray(
            (
                game["eI_hip"], game["eI_knee"],
                game["edotI_hip"], game["edotI_knee"],
            ),
            dtype=float,
        )
        disagreement = z_a - z_i
        alpha = _finite_float(game["alpha"], "performance alpha")
        q_a_state = float(
            alpha * z_a @ PERFORMANCE_QA @ z_a
            + PERFORMANCE_LAMBDA_A
            * disagreement @ PERFORMANCE_SA @ disagreement
        )
        q_i_state = float(
            (1.0 - alpha) * z_i @ PERFORMANCE_QI @ z_i
            + PERFORMANCE_LAMBDA_I
            * disagreement @ PERFORMANCE_SI @ disagreement
        )
        e_p = np.asarray(
            (
                z_i[0], z_i[1],
                PERFORMANCE_C_P_RAD_PER_NM * game["tau_int_hip"],
                PERFORMANCE_C_P_RAD_PER_NM * game["tau_int_knee"],
            ),
            dtype=float,
        )
        e_p_dot = np.asarray(
            (
                z_i[2], z_i[3],
                PERFORMANCE_C_P_RAD_PER_NM
                * GAME_K0_NM_PER_RAD[0] * (z_a[2] - z_i[2]),
                PERFORMANCE_C_P_RAD_PER_NM
                * GAME_K0_NM_PER_RAD[1] * (z_a[3] - z_i[3]),
            ),
            dtype=float,
        )
        qddot_d = np.asarray(qddot_reference, dtype=float).reshape(2)
        mass = np.asarray(smooth_mass, dtype=float).reshape(2, 2)
        bias = np.asarray(smooth_bias, dtype=float).reshape(2)
        mu0_shadow = (
            mass @ qddot_d
            + bias
            - GAME_DI_NMS_PER_RAD @ z_i[2:4]
            - GAME_KI_NM_PER_RAD * z_i[0:2]
        )
        f_x = np.asarray(
            (
                z_a[2], z_a[3],
                game["fA_accel_hip"], game["fA_accel_knee"],
                z_i[2], z_i[3],
                game["fI_accel_hip"], game["fI_accel_knee"],
            ),
            dtype=float,
        )
        g_a = np.zeros((8, 2), dtype=float)
        g_i = np.zeros((8, 2), dtype=float)
        g_a[2:4] = np.asarray(
            ((game["gA_00"], game["gA_01"]),
             (game["gA_10"], game["gA_11"])),
            dtype=float,
        )
        g_i[6:8] = np.asarray(
            ((game["gI_00"], game["gI_01"]),
             (game["gI_10"], game["gI_11"])),
            dtype=float,
        )
        values = np.concatenate((z_a, z_i, disagreement, e_p, e_p_dot,
                                 mu0_shadow, f_x, g_a.reshape(-1),
                                 g_i.reshape(-1)))
        if not np.isfinite(values).all():
            raise FloatingPointError("Stage-4 performance shadow is non-finite")
        result = self._performance_shadow_defaults()
        result.update(
            {
                "qA_state": q_a_state,
                "qI_state": q_i_state,
                "eP_track_hip": float(e_p[0]),
                "eP_track_knee": float(e_p[1]),
                "eP_int_hip": float(e_p[2]),
                "eP_int_knee": float(e_p[3]),
                "ePdot_track_hip": float(e_p_dot[0]),
                "ePdot_track_knee": float(e_p_dot[1]),
                "ePdot_int_hip": float(e_p_dot[2]),
                "ePdot_int_knee": float(e_p_dot[3]),
                "mu0_shadow_hip": float(mu0_shadow[0]),
                "mu0_shadow_knee": float(mu0_shadow[1]),
                "mu0_shadow_native_limit_counterfactual": bool(
                    np.any(np.abs(mu0_shadow) >= NATIVE_MOTOR_LIMIT_NM)
                ),
            }
        )
        if ramp < 1.0:
            return result
        activation_sample = self._ppc_start_s is None
        if activation_sample:
            self._ppc_start_s = time_s
        if self._ppc_start_s is None:
            raise RuntimeError("PPC local clock failed to initialize")
        ppc_time = max(0.0, time_s - self._ppc_start_s)
        rho_value, rho_dot_value = ppc_envelope_with_limits(
            ppc_time, self._ppc_rho0, self._ppc_rho_inf,
            self._ppc_t_rho_s, PPC_NU,
        )
        ppc = ppc_transform_given_envelope(
            e_p, e_p_dot, rho_value, rho_dot_value
        )
        ppc_valid = bool(ppc["valid"])
        zeta = np.asarray(ppc["zeta"], dtype=float)
        epsilon = np.asarray(ppc["epsilon"], dtype=float)
        epsilon_dot = np.asarray(ppc["epsilon_dot"], dtype=float)
        r_diagonal = np.asarray(ppc["r_diagonal"], dtype=float)
        rho = float(ppc["rho"])
        rho_dot = float(ppc["rho_dot"])
        pt = predefined_time_scaling_with_parameters(
            ppc_time, self._pt_tp_s, PT_DELTA, PT_P_S_INV
        )
        if not ppc_valid and self._first_ppc_violation is None:
            channel_index = int(np.argmax(np.abs(zeta)))
            channel_names = (
                "Hip track",
                "Knee track",
                "Hip interaction",
                "Knee interaction",
            )
            self._first_ppc_violation = {
                "time": float(time_s),
                "channel": channel_names[channel_index],
                "eP": float(e_p[channel_index]),
                "rho": rho,
                "zeta": float(zeta[channel_index]),
            }
        first = self._first_ppc_violation
        result.update(
            {
                "ppc_active": True,
                "ppc_valid": ppc_valid,
                "ppc_activation_sample": activation_sample,
                "ppc_initial_condition_pass": bool(
                    not activation_sample or np.all(np.abs(e_p) < self._ppc_rho0)
                ),
                "ppc_boundary_violation": not ppc_valid,
                "ppc_time_s": ppc_time,
                "rho": rho,
                "rho_dot": rho_dot,
                "zeta_track_hip": float(zeta[0]),
                "zeta_track_knee": float(zeta[1]),
                "zeta_int_hip": float(zeta[2]),
                "zeta_int_knee": float(zeta[3]),
                "phi1": pt["phi1"],
                "phi1_dot": pt["phi1_dot"],
                "phiT": pt["phiT"],
                "sT": pt["sT"],
                "first_ppc_violation_time": "" if first is None else first["time"],
                "first_ppc_violation_channel": "" if first is None else first["channel"],
                "first_ppc_violation_eP": "" if first is None else first["eP"],
                "first_ppc_violation_rho": "" if first is None else first["rho"],
                "first_ppc_violation_zeta": "" if first is None else first["zeta"],
            }
        )
        if not ppc_valid:
            result.update(
                {
                    "epsilon_track_hip": math.nan,
                    "epsilon_track_knee": math.nan,
                    "epsilon_int_hip": math.nan,
                    "epsilon_int_knee": math.nan,
                    "epsilon_dot_track_hip": math.nan,
                    "epsilon_dot_track_knee": math.nan,
                    "epsilon_dot_int_hip": math.nan,
                    "epsilon_dot_int_knee": math.nan,
                    "epsilon_norm_sq": math.nan,
                    "chi_all_finite": False,
                    "Fchi_all_finite": False,
                }
            )
            return result
        chi = np.concatenate((z_a, z_i, epsilon))
        features, jacobian = quadratic_basis_78(chi)
        f_chi = np.concatenate((f_x, epsilon_dot))
        ga_chi = np.vstack((g_a, np.zeros((4, 2), dtype=float)))
        gi_chi = np.vstack((g_i, np.zeros((4, 2), dtype=float)))
        geometry = np.concatenate(
            (chi, features, jacobian.reshape(-1), f_chi,
             ga_chi.reshape(-1), gi_chi.reshape(-1), r_diagonal)
        )
        if not np.isfinite(geometry).all():
            raise FloatingPointError("Stage-4 chi/Critic geometry is non-finite")
        result.update(
            {
                "epsilon_track_hip": float(epsilon[0]),
                "epsilon_track_knee": float(epsilon[1]),
                "epsilon_int_hip": float(epsilon[2]),
                "epsilon_int_knee": float(epsilon[3]),
                "epsilon_dot_track_hip": float(epsilon_dot[0]),
                "epsilon_dot_track_knee": float(epsilon_dot[1]),
                "epsilon_dot_int_hip": float(epsilon_dot[2]),
                "epsilon_dot_int_knee": float(epsilon_dot[3]),
                "epsilon_norm_sq": float(epsilon @ epsilon),
                "phi1": pt["phi1"],
                "phi1_dot": pt["phi1_dot"],
                "phiT": pt["phiT"],
                "sT": pt["sT"],
                "critic78_feature_norm": float(np.linalg.norm(features)),
                "critic78_jacobian_fro": float(np.linalg.norm(jacobian)),
                "Fchi_norm": float(np.linalg.norm(f_chi)),
                "GAchi_fro": float(np.linalg.norm(ga_chi)),
                "GIch_fro": float(np.linalg.norm(gi_chi)),
                "chi_all_finite": True,
                "Fchi_all_finite": True,
            }
        )
        result.update(
            {f"chi_{index:02d}": float(value)
             for index, value in enumerate(chi)}
        )
        result.update(
            {f"Fchi_{index:02d}": float(value)
             for index, value in enumerate(f_chi)}
        )
        return result

    def _stage6b_active_step(
        self,
        *,
        performance: Mapping[str, Any],
        smooth_mass: np.ndarray,
        alpha: float,
        ramp: float,
        mu0: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Form the active Stage-6B command and then store the next Critic."""
        result = self._stage6b_defaults()
        baseline = np.asarray(mu0, dtype=float).reshape(2)
        command = ramp * baseline
        result.update(
            {
                "stage6b_mu0_hip": float(baseline[0]),
                "stage6b_mu0_knee": float(baseline[1]),
                "stage6b_raw_total_hip": float(baseline[0]),
                "stage6b_raw_total_knee": float(baseline[1]),
                "stage6b_command_hip": float(command[0]),
                "stage6b_command_knee": float(command[1]),
            }
        )
        if ramp < 1.0:
            if np.any(self._stage6b_weight != 0.0):
                raise RuntimeError("Stage-6B Critic changed during engagement ramp")
            return command, result
        if not bool(performance.get("ppc_active")) or not bool(
            performance.get("ppc_valid")
        ):
            reason = "ppc_development_envelope_invalid"
            self.termination_request = f"stage6b_scientific_abort_{reason}"
            result.update(
                {
                    "stage6b_ppc_invalid": True,
                    "stage6b_scientific_abort": True,
                    "stage6b_scientific_abort_reason": reason,
                    "stage6b_command_hip": 0.0,
                    "stage6b_command_knee": 0.0,
                }
            )
            return np.zeros(2, dtype=float), result

        chi = np.asarray(
            [performance[f"chi_{index:02d}"] for index in range(CHI_DIMENSION)],
            dtype=float,
        )
        f_chi = np.asarray(
            [performance[f"Fchi_{index:02d}"] for index in range(CHI_DIMENSION)],
            dtype=float,
        )
        try:
            critic = stage6b_causal_control_step(
                chi,
                f_chi,
                smooth_mass,
                alpha,
                float(performance["sT"]),
                self._stage6b_weight,
                dt,
            )
        except (FloatingPointError, np.linalg.LinAlgError) as error:
            reason = "nonfinite_or_singular_controller_quantity"
            self.termination_request = f"stage6b_scientific_abort_{reason}"
            result.update(
                {
                    "stage6b_scientific_abort": True,
                    "stage6b_scientific_abort_reason": f"{reason}: {error}",
                    "stage6b_command_hip": 0.0,
                    "stage6b_command_knee": 0.0,
                }
            )
            return np.zeros(2, dtype=float), result

        strategy = np.asarray(critic["strategy"], dtype=float).reshape(4)
        mu_a = strategy[:2]
        mu_i = strategy[2:]
        raw_total = baseline + mu_i
        next_weights = np.asarray(critic["weights_next"], dtype=float)
        current_weights = np.asarray(critic["weights_current"], dtype=float)
        current_norm = np.linalg.norm(current_weights, axis=0)
        next_norm = np.linalg.norm(next_weights, axis=0)
        update_norm = np.linalg.norm(next_weights - current_weights, axis=0)
        projected = tuple(bool(value) for value in critic["projected"])
        p_a_eigen = np.linalg.eigvalsh(critic["P_A"])
        p_i_eigen = np.linalg.eigvalsh(critic["P_I"])
        gradients = np.asarray(critic["value_gradients_chi"], dtype=float)

        mu_i_exceeded = bool(
            np.any(np.abs(mu_i) > STAGE6B_MUI_ABORT_BUDGET_NM)
        )
        raw_exceeded = bool(
            np.any(np.abs(raw_total) >= STAGE6B_RAW_TOTAL_ABORT_NM)
        )
        weight_invalid = bool(
            np.any(next_norm > STAGE6B_WEIGHT_RADIUS + 1.0e-12)
        )
        abort_reasons = []
        if mu_i_exceeded:
            abort_reasons.append("muI_scientific_budget_exceeded")
        if raw_exceeded:
            abort_reasons.append("raw_total_20Nm_guard_exceeded")
        if weight_invalid:
            abort_reasons.append("critic_compact_domain_violation")
        scientific_abort = bool(abort_reasons)
        if scientific_abort:
            self.termination_request = (
                "stage6b_scientific_abort_" + "+".join(abort_reasons)
            )
            command = np.zeros(2, dtype=float)
        else:
            # The interval-k command above uses only W_k.  Persist W_{k+1}
            # after it has been fully formed for the next 100 Hz tick.
            command = ramp * raw_total
            self._stage6b_weight = next_weights.copy()
            self._stage6b_learning_samples += 1
            self._stage6b_projection_count += np.asarray(projected, dtype=int)

        denominator = max(self._stage6b_learning_samples, 1)
        projection_fraction = self._stage6b_projection_count / denominator
        result.update(
            {
                "stage6b_learning_active": True,
                "stage6b_causal_actor_uses_Wk": bool(
                    critic["causal_actor_uses_Wk"]
                ),
                "stage6b_next_weight_not_used_for_current_actor": bool(
                    critic["next_weight_not_used_for_current_actor"]
                ),
                "stage6b_projection_A": projected[0],
                "stage6b_projection_I": projected[1],
                "stage6b_projection_count_A": int(
                    self._stage6b_projection_count[0]
                ),
                "stage6b_projection_count_I": int(
                    self._stage6b_projection_count[1]
                ),
                "stage6b_projection_fraction_A": float(
                    projection_fraction[0]
                ),
                "stage6b_projection_fraction_I": float(
                    projection_fraction[1]
                ),
                "stage6b_learning_sample_count": self._stage6b_learning_samples,
                "stage6b_W_A_current_norm": float(current_norm[0]),
                "stage6b_W_I_current_norm": float(current_norm[1]),
                "stage6b_W_A_next_norm": float(next_norm[0]),
                "stage6b_W_I_next_norm": float(next_norm[1]),
                "stage6b_muA_hip": float(mu_a[0]),
                "stage6b_muA_knee": float(mu_a[1]),
                "stage6b_muI_hip": float(mu_i[0]),
                "stage6b_muI_knee": float(mu_i[1]),
                "stage6b_raw_total_hip": float(raw_total[0]),
                "stage6b_raw_total_knee": float(raw_total[1]),
                "stage6b_command_hip": float(command[0]),
                "stage6b_command_knee": float(command[1]),
                "stage6b_delta_A": float(critic["delta"][0]),
                "stage6b_delta_I": float(critic["delta"][1]),
                "stage6b_bellman_vector_norm": float(
                    np.linalg.norm(critic["delta"])
                ),
                "stage6b_weight_update_A_norm": float(update_norm[0]),
                "stage6b_weight_update_I_norm": float(update_norm[1]),
                "stage6b_q_A_chi": float(critic["q_A_chi"]),
                "stage6b_q_I_chi": float(critic["q_I_chi"]),
                "stage6b_r_A": float(critic["r_A"]),
                "stage6b_r_I": float(critic["r_I"]),
                "stage6b_ell_A_chi": float(critic["ell_A_chi"]),
                "stage6b_ell_I_chi": float(critic["ell_I_chi"]),
                "stage6b_omega_norm": float(np.linalg.norm(critic["omega"])),
                "stage6b_P_A_lambda_min": float(p_a_eigen[0]),
                "stage6b_P_A_lambda_max": float(p_a_eigen[-1]),
                "stage6b_P_A_trace": float(np.trace(critic["P_A"])),
                "stage6b_P_I_lambda_min": float(p_i_eigen[0]),
                "stage6b_P_I_lambda_max": float(p_i_eigen[-1]),
                "stage6b_P_I_trace": float(np.trace(critic["P_I"])),
                "stage6b_muI_budget_exceeded": mu_i_exceeded,
                "stage6b_raw_20Nm_guard_exceeded": raw_exceeded,
                "stage6b_native_saturation": bool(
                    np.any(np.abs(raw_total) >= NATIVE_MOTOR_LIMIT_NM)
                ),
                "stage6b_scientific_abort": scientific_abort,
                "stage6b_scientific_abort_reason": "+".join(abort_reasons),
            }
        )
        for player_index, player in enumerate(("A", "I")):
            result.update(
                {
                    f"stage6b_gradV{player}_chi_{index:02d}": float(
                        gradients[index, player_index]
                    )
                    for index in range(CHI_DIMENSION)
                }
            )
        return command, result

    def _stage6f_step(
        self,
        *,
        performance: Mapping[str, Any],
        smooth_mass: np.ndarray,
        alpha: float,
        nominal_ramp: float,
        mu0: np.ndarray,
        time_s: float,
        dt: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Advance isolated Stage-6F game state and return physical command."""
        result = self._stage6f_defaults()
        baseline = np.asarray(mu0, dtype=float).reshape(2)
        proposed = nominal_ramp * baseline
        result.update(
            {
                "stage6f_mu0_hip": float(baseline[0]),
                "stage6f_mu0_knee": float(baseline[1]),
                "stage6f_proposed_hip": float(proposed[0]),
                "stage6f_proposed_knee": float(proposed[1]),
            }
        )
        if nominal_ramp < 1.0:
            if np.any(self._stage6b_weight != 0.0):
                raise RuntimeError("Stage-6F Critic changed during nominal ramp")
            physical = (
                proposed
                if self.mode == PHYSICAL_ACTUATION_GAME_ACTIVE_MODE
                else np.zeros(2, dtype=float)
            )
            result["stage6f_physical_hip"] = float(physical[0])
            result["stage6f_physical_knee"] = float(physical[1])
            return physical, result
        if not bool(performance.get("ppc_active")) or not bool(
            performance.get("ppc_valid")
        ):
            reason = "ppc_development_envelope_invalid"
            result.update({"stage6f_abort": True, "stage6f_abort_reason": reason})
            if self.mode == PHYSICAL_ACTUATION_GAME_ACTIVE_MODE:
                self.termination_request = f"stage6f_scientific_abort_{reason}"
            return np.zeros(2, dtype=float), result

        if self._stage6f_game_ramp_start_s is None:
            self._stage6f_game_ramp_start_s = time_s
        elapsed = max(0.0, time_s - self._stage6f_game_ramp_start_s)
        game_ramp = (
            1.0
            if elapsed >= FIXED_ENGAGEMENT_RAMP_S
            else 0.5
            - 0.5 * math.cos(math.pi * elapsed / FIXED_ENGAGEMENT_RAMP_S)
        )
        chi = np.asarray(
            [performance[f"chi_{index:02d}"] for index in range(CHI_DIMENSION)],
            dtype=float,
        )
        f_chi = np.asarray(
            [performance[f"Fchi_{index:02d}"] for index in range(CHI_DIMENSION)],
            dtype=float,
        )
        reset_now = False
        if game_ramp < 1.0:
            if np.any(self._stage6b_weight != 0.0):
                raise RuntimeError("Stage-6F Critic changed during game ramp")
            critic = stage6f_causal_control_step(
                chi, f_chi, smooth_mass, alpha, float(performance["sT"]),
                self._stage6b_weight, baseline, dt,
            )
            # The actor is evaluated for the ramped proposal, but W stays
            # exactly frozen until full game engagement.
            next_weights = self._stage6b_weight.copy()
            projected = (False, False)
        else:
            if not self._stage6f_game_engaged:
                self._stage6b_weight = np.zeros(CRITIC_WEIGHT_SHAPE, dtype=float)
                self._stage6b_learning_samples = 0
                self._stage6b_projection_count = np.zeros(2, dtype=int)
                self._stage6f_game_engaged = True
                reset_now = True
            critic = stage6f_causal_control_step(
                chi, f_chi, smooth_mass, alpha, float(performance["sT"]),
                self._stage6b_weight, baseline, dt,
            )
            next_weights = np.asarray(critic["weights_next"], dtype=float)
            projected = tuple(bool(value) for value in critic["projected"])

        strategy = np.asarray(critic["strategy"], dtype=float).reshape(4)
        mu_a, mu_i = strategy[:2], strategy[2:]
        proposed = baseline + game_ramp * mu_i
        next_norm = np.linalg.norm(next_weights, axis=0)
        mu_i_exceeded = bool(
            np.any(np.abs(mu_i) >= STAGE6B_MUI_ABORT_BUDGET_NM)
        )
        raw_exceeded = bool(np.any(np.abs(proposed) >= STAGE6B_RAW_TOTAL_ABORT_NM))
        weight_invalid = bool(
            np.any(next_norm > STAGE6B_WEIGHT_RADIUS + 1.0e-12)
        )
        eigenvalues = np.linalg.eigvalsh(critic["RN_new"])
        rn_invalid = bool(eigenvalues[0] <= 1.0e-4)
        abort_reasons = []
        if mu_i_exceeded:
            abort_reasons.append("muI_scientific_budget_exceeded")
        if raw_exceeded:
            abort_reasons.append("raw_total_20Nm_guard_exceeded")
        if weight_invalid:
            abort_reasons.append("critic_compact_domain_violation")
        if rn_invalid:
            abort_reasons.append("RN_new_safety_violation")
        scientific_abort = bool(abort_reasons)
        if not scientific_abort and game_ramp >= 1.0:
            self._stage6b_weight = next_weights.copy()
            self._stage6b_learning_samples += 1
            self._stage6b_projection_count += np.asarray(projected, dtype=int)
        elif scientific_abort and self.mode == PHYSICAL_ACTUATION_GAME_ACTIVE_MODE:
            self.termination_request = (
                "stage6f_scientific_abort_" + "+".join(abort_reasons)
            )
        denominator = max(self._stage6b_learning_samples, 1)
        projection_fraction = self._stage6b_projection_count / denominator
        physical = (
            proposed
            if self.mode == PHYSICAL_ACTUATION_GAME_ACTIVE_MODE
            and not scientific_abort
            else np.zeros(2, dtype=float)
        )
        gradients = np.asarray(critic["value_gradients_chi"], dtype=float)
        result.update(
            {
                "stage6f_beta": float(critic["beta"]),
                "stage6f_game_ramp": float(game_ramp),
                "stage6f_game_ramp_active": game_ramp < 1.0,
                "stage6f_weights_frozen_for_game_ramp": game_ramp < 1.0,
                "stage6f_weights_reset_at_full_engagement": reset_now,
                "stage6f_learning_active": game_ramp >= 1.0 and not scientific_abort,
                "stage6f_muA_hip": float(mu_a[0]),
                "stage6f_muA_knee": float(mu_a[1]),
                "stage6f_muI_hip": float(mu_i[0]),
                "stage6f_muI_knee": float(mu_i[1]),
                "stage6f_proposed_hip": float(proposed[0]),
                "stage6f_proposed_knee": float(proposed[1]),
                "stage6f_physical_hip": float(physical[0]),
                "stage6f_physical_knee": float(physical[1]),
                "stage6f_r_I_original": float(critic["r_I_original"]),
                "stage6f_r_total": float(critic["r_total"]),
                "stage6f_r_total_expanded": float(critic["r_total_expanded"]),
                "stage6f_r_total_expansion_error": float(
                    critic["r_total_expansion_error"]
                ),
                "stage6f_ell_A": float(critic["ell_A_chi"]),
                "stage6f_ell_I": float(critic["ell_I_chi"]),
                "stage6f_FOC_residual_max_abs": float(
                    np.max(np.abs(critic["FOC_residual"]))
                ),
                "stage6f_RN_min_eigenvalue": float(eigenvalues[0]),
                "stage6f_RN_max_eigenvalue": float(eigenvalues[-1]),
                "stage6f_RN_condition": float(np.linalg.cond(critic["RN_new"])),
                "stage6f_projection_A": projected[0],
                "stage6f_projection_I": projected[1],
                "stage6f_projection_fraction_A": float(projection_fraction[0]),
                "stage6f_projection_fraction_I": float(projection_fraction[1]),
                "stage6f_W_A_norm": float(next_norm[0]),
                "stage6f_W_I_norm": float(next_norm[1]),
                "stage6f_delta_A": float(critic["delta"][0]),
                "stage6f_delta_I": float(critic["delta"][1]),
                "stage6f_abort": scientific_abort,
                "stage6f_abort_reason": "+".join(abort_reasons),
            }
        )
        for player_index, player in enumerate(("A", "I")):
            for index in range(CHI_DIMENSION):
                result[f"stage6f_gradV{player}_chi_{index:02d}"] = float(
                    gradients[index, player_index]
                )
        return physical, result

    def _stage6g_step(
        self, *, performance: Mapping[str, Any], smooth_mass: np.ndarray,
        alpha: float, nominal_ramp: float, mu0: np.ndarray,
        time_s: float, dt: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Advance the isolated intention-target game state."""
        result=self._stage6g_defaults(); baseline=np.asarray(mu0,dtype=float).reshape(2)
        gamma=stage6g_assistance_gamma(alpha); target=gamma*baseline
        proposed=nominal_ramp*baseline
        result.update({"stage6g_gamma":gamma,"stage6g_mu0_hip":float(baseline[0]),"stage6g_mu0_knee":float(baseline[1]),
                       "stage6g_mu_target_hip":float(target[0]),"stage6g_mu_target_knee":float(target[1]),
                       "stage6g_proposed_hip":float(proposed[0]),"stage6g_proposed_knee":float(proposed[1])})
        if nominal_ramp<1.0:
            if np.any(self._stage6b_weight!=0.0): raise RuntimeError("Stage-6G Critic changed during nominal ramp")
            physical=proposed if self.mode in (INTENTION_TARGET_GAME_ACTIVE_MODE, *EVENT_EXECUTION_MODES, *STAGE7T_MODES, *STAGE8A_MODES) else np.zeros(2)
            result.update({"stage6g_physical_hip":float(physical[0]),"stage6g_physical_knee":float(physical[1])})
            return physical,result
        if not bool(performance.get("ppc_active")) or not bool(performance.get("ppc_valid")):
            reason="ppc_development_envelope_invalid"; result.update({"stage6g_abort":True,"stage6g_abort_reason":reason})
            if self.mode in (INTENTION_TARGET_GAME_ACTIVE_MODE, *EVENT_EXECUTION_MODES, *STAGE7T_MODES, *STAGE8A_MODES): self.termination_request=f"stage6g_scientific_abort_{reason}"
            return np.zeros(2),result
        if self._stage6g_game_ramp_start_s is None: self._stage6g_game_ramp_start_s=time_s
        elapsed=max(0.0,time_s-self._stage6g_game_ramp_start_s)
        game_ramp=1.0 if elapsed>=FIXED_ENGAGEMENT_RAMP_S else .5-.5*math.cos(math.pi*elapsed/FIXED_ENGAGEMENT_RAMP_S)
        chi=np.asarray([performance[f"chi_{i:02d}"] for i in range(CHI_DIMENSION)])
        f_chi=np.asarray([performance[f"Fchi_{i:02d}"] for i in range(CHI_DIMENSION)])
        reset_now=False
        if game_ramp<1.0:
            if np.any(self._stage6b_weight!=0.0): raise RuntimeError("Stage-6G Critic changed during game ramp")
            critic=stage6g_causal_control_step(chi,f_chi,smooth_mass,alpha,float(performance["sT"]),self._stage6b_weight,baseline,dt)
            next_weights=self._stage6b_weight.copy(); projected=(False,False)
        else:
            if not self._stage6g_game_engaged:
                self._stage6b_weight=np.zeros(CRITIC_WEIGHT_SHAPE); self._stage6b_learning_samples=0; self._stage6b_projection_count=np.zeros(2,dtype=int)
                self._stage6g_game_engaged=True; reset_now=True
            critic=stage6g_causal_control_step(chi,f_chi,smooth_mass,alpha,float(performance["sT"]),self._stage6b_weight,baseline,dt)
            next_weights=np.asarray(critic["weights_next"]); projected=tuple(bool(v) for v in critic["projected"])
        strategy=np.asarray(critic["strategy"]).reshape(4); mu_a,mu_i=strategy[:2],strategy[2:]
        strategy_affine=np.asarray(critic["strategy_affine"]).reshape(4)
        strategy_critic=np.asarray(critic["strategy_critic"]).reshape(4)
        decomposition_error=float(np.max(np.abs(
            strategy-(strategy_affine+strategy_critic)
        )))
        affine_physical=baseline+strategy_affine[2:]
        sign_mask=np.abs(baseline)>0.05
        affine_sign_pass=bool(np.all(
            baseline[sign_mask]*affine_physical[sign_mask]>=-1.0e-9
        ))
        affine_magnitude_pass=bool(np.all(
            np.abs(affine_physical[sign_mask])
            <=np.abs(baseline[sign_mask])+1.0e-9
        ))
        affine_near_zero_pass=bool(np.all(
            np.abs(affine_physical[~sign_mask])<5.0
        ))
        affine_safety_pass=bool(
            affine_sign_pass and affine_magnitude_pass
            and affine_near_zero_pass and decomposition_error<1.0e-12
        )
        critic_budget=np.concatenate((STAGE6A_MUA_BUDGET_NM,
                                      STAGE6A_MUI_BUDGET_NM))
        critic_budget_exceeded=bool(np.any(
            np.abs(strategy_critic)>critic_budget
        ))
        legacy_total_budget_exceeded=bool(np.any(
            np.abs(mu_i)>STAGE6A_MUI_BUDGET_NM
        ))
        proposed=baseline+game_ramp*mu_i; next_norm=np.linalg.norm(next_weights,axis=0)
        eigenvalues=np.linalg.eigvalsh(critic["RN_target"]); abort_reasons=[]
        if critic_budget_exceeded: abort_reasons.append("critic_component_strategy_budget_exceeded")
        if not affine_safety_pass: abort_reasons.append("affine_target_safety_contract_failure")
        if np.any(np.abs(proposed)>=STAGE6B_RAW_TOTAL_ABORT_NM): abort_reasons.append("raw_total_20Nm_guard_exceeded")
        if np.any(next_norm>STAGE6B_WEIGHT_RADIUS+1e-12): abort_reasons.append("critic_compact_domain_violation")
        if eigenvalues[0]<=1e-4: abort_reasons.append("RN_target_safety_violation")
        scientific_abort=bool(abort_reasons)
        if not scientific_abort and game_ramp>=1.0:
            self._stage6b_weight=next_weights.copy(); self._stage6b_learning_samples+=1; self._stage6b_projection_count+=np.asarray(projected,dtype=int)
        elif scientific_abort and self.mode in (INTENTION_TARGET_GAME_ACTIVE_MODE, *EVENT_EXECUTION_MODES, *STAGE7T_MODES, *STAGE8A_MODES):
            self.termination_request="stage6g_scientific_abort_"+"+".join(abort_reasons)
        projection_fraction=self._stage6b_projection_count/max(self._stage6b_learning_samples,1)
        physical=proposed if self.mode in (INTENTION_TARGET_GAME_ACTIVE_MODE, *EVENT_EXECUTION_MODES, *STAGE7T_MODES, *STAGE8A_MODES) and not scientific_abort else np.zeros(2)
        result.update({"stage6g_gamma":float(critic["gamma"]),"stage6g_game_ramp":game_ramp,"stage6g_game_ramp_active":game_ramp<1.0,
            "stage6g_weights_frozen_for_game_ramp":game_ramp<1.0,"stage6g_weights_reset_at_full_engagement":reset_now,
            "stage6g_learning_active":game_ramp>=1.0 and not scientific_abort,"stage6g_muA_hip":float(mu_a[0]),"stage6g_muA_knee":float(mu_a[1]),
            "stage6g_muI_hip":float(mu_i[0]),"stage6g_muI_knee":float(mu_i[1]),"stage6g_proposed_hip":float(proposed[0]),"stage6g_proposed_knee":float(proposed[1]),
            "stage6g_muA_affine_hip":float(strategy_affine[0]),"stage6g_muA_affine_knee":float(strategy_affine[1]),
            "stage6g_muI_affine_hip":float(strategy_affine[2]),"stage6g_muI_affine_knee":float(strategy_affine[3]),
            "stage6g_muA_critic_hip":float(strategy_critic[0]),"stage6g_muA_critic_knee":float(strategy_critic[1]),
            "stage6g_muI_critic_hip":float(strategy_critic[2]),"stage6g_muI_critic_knee":float(strategy_critic[3]),
            "stage6g_strategy_decomposition_error_max_abs":decomposition_error,
            "stage6g_affine_physical_hip":float(affine_physical[0]),"stage6g_affine_physical_knee":float(affine_physical[1]),
            "stage6g_affine_sign_pass":affine_sign_pass,"stage6g_affine_magnitude_pass":affine_magnitude_pass,
            "stage6g_affine_near_zero_pass":affine_near_zero_pass,"stage6g_affine_safety_passed":affine_safety_pass,
            "stage6g_legacy_total_budget_exceeded":legacy_total_budget_exceeded,
            "stage6g_corrected_critic_budget_exceeded":critic_budget_exceeded,
            "stage6g_physical_hip":float(physical[0]),"stage6g_physical_knee":float(physical[1]),"stage6g_r_I_original":float(critic["r_I_original"]),
            "stage6g_r_target":float(critic["r_target"]),"stage6g_r_target_expanded":float(critic["r_target_expanded"]),
            "stage6g_r_target_expansion_error":float(critic["r_target_expansion_error"]),"stage6g_ell_A":float(critic["ell_A_chi"]),"stage6g_ell_I":float(critic["ell_I_chi"]),
            "stage6g_FOC_residual_max_abs":float(np.max(np.abs(critic["FOC_residual"]))),"stage6g_RN_min_eigenvalue":float(eigenvalues[0]),
            "stage6g_RN_max_eigenvalue":float(eigenvalues[-1]),"stage6g_RN_condition":float(np.linalg.cond(critic["RN_target"])),
            "stage6g_projection_A":projected[0],"stage6g_projection_I":projected[1],"stage6g_projection_fraction_A":float(projection_fraction[0]),
            "stage6g_projection_fraction_I":float(projection_fraction[1]),"stage6g_W_A_norm":float(next_norm[0]),"stage6g_W_I_norm":float(next_norm[1]),
            "stage6g_delta_A":float(critic["delta"][0]),"stage6g_delta_I":float(critic["delta"][1]),"stage6g_abort":scientific_abort,
            "stage6g_abort_reason":"+".join(abort_reasons)})
        gradients=np.asarray(critic["value_gradients_chi"])
        for player_index,player in enumerate(("A","I")):
            for index in range(CHI_DIMENSION): result[f"stage6g_gradV{player}_chi_{index:02d}"]=float(gradients[index,player_index])
        return physical,result

    def _stage7_event_step(
        self, *, performance: Mapping[str, Any],
        stage6g: Mapping[str, Any], mu0: np.ndarray,
        continuous_command: np.ndarray, time_s: float, dt: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Hold only the accepted Nash strategy while Stage-6G keeps updating."""
        result = self._stage7_defaults()
        parameters = self._event_execution_parameters()
        theta0 = float(parameters["theta0"])
        dwell_s = float(parameters["dwell_s"])
        lambda_v = float(parameters["lambda_v"])
        kappa_e = float(parameters["kappa_e"])
        c_chi_evt = float(parameters["c_chi_evt"])
        if not bool(stage6g.get("stage6g_learning_active")):
            return np.asarray(continuous_command, dtype=float).reshape(2), result

        current = np.asarray(
            (
                stage6g["stage6g_muA_hip"], stage6g["stage6g_muA_knee"],
                stage6g["stage6g_muI_hip"], stage6g["stage6g_muI_knee"],
            ),
            dtype=float,
        )
        chi = np.asarray(
            [performance[f"chi_{index:02d}"] for index in range(CHI_DIMENSION)],
            dtype=float,
        )
        xchi = chi / STAGE6A_D_CHI_DIAGONAL
        xchi_norm_sq = float(xchi @ xchi)
        if not np.isfinite(np.concatenate((current, xchi))).all():
            raise FloatingPointError("Stage-7 event inputs are non-finite")

        previous_held = (
            None if self._stage7_held_strategy is None
            else self._stage7_held_strategy.copy()
        )
        accepted = False
        interval: float | str = ""
        self._stage7_eligible_ticks += 1
        self._stage7_critic_updates += 1
        if self._stage7_held_strategy is None:
            # Event zero occurs exactly when the frozen continuous game first
            # reaches full engagement.  Startup mu0/game ramps remain unheld.
            self._stage7_held_strategy = current.copy()
            self._stage7_vartheta = 0.0
            self._stage7_event_age_s = 0.0
            self._stage7_event_count = 1
            self._stage7_last_event_time_s = time_s
            accepted = True
        else:
            decay = math.exp(-lambda_v * dt)
            self._stage7_vartheta = max(
                0.0,
                decay * self._stage7_vartheta
                + c_chi_evt * xchi_norm_sq
                * (1.0 - decay) / lambda_v,
            )
            self._stage7_event_age_s += dt
            pre_error = (self._stage7_held_strategy - current) / STAGE7_STRATEGY_SCALE_NM
            pre_error_norm_sq = float(pre_error @ pre_error)
            threshold = (
                theta0 + kappa_e * xchi_norm_sq
                + self._stage7_vartheta
            )
            if (
                self._stage7_event_age_s + 1.0e-12 >= dwell_s
                and pre_error_norm_sq >= threshold
            ):
                if self._stage7_last_event_time_s is not None:
                    interval = time_s - self._stage7_last_event_time_s
                self._stage7_held_strategy = current.copy()
                self._stage7_event_age_s = 0.0
                self._stage7_event_count += 1
                self._stage7_last_event_time_s = time_s
                accepted = True

        held = np.asarray(self._stage7_held_strategy, dtype=float).reshape(4)
        error = held - current
        normalized_error = error / STAGE7_STRATEGY_SCALE_NM
        error_norm_sq = float(normalized_error @ normalized_error)
        threshold = (
            theta0 + kappa_e * xchi_norm_sq
            + self._stage7_vartheta
        )
        baseline = np.asarray(mu0, dtype=float).reshape(2)
        physical = baseline + held[2:]
        contract_error = physical - (baseline + held[2:])
        dwell_violation = bool(
            accepted and interval != ""
            and float(interval) < dwell_s - 1.0e-12
        )
        abort_reasons: list[str] = []
        if np.any(np.abs(physical) >= STAGE6B_RAW_TOTAL_ABORT_NM):
            abort_reasons.append("raw_total_20Nm_guard_exceeded")
        if dwell_violation:
            abort_reasons.append("dwell_violation")
        if self._stage7_critic_updates != self._stage7_eligible_ticks:
            abort_reasons.append("critic_event_tick_count_mismatch")
        scientific_abort = bool(abort_reasons)
        if scientific_abort:
            self.termination_request = "stage7_scientific_abort_" + "+".join(abort_reasons)
            physical = np.zeros(2, dtype=float)

        labels = ("muA_hip", "muA_knee", "muI_hip", "muI_knee")
        result.update(
            {
                "stage7_event_logic_active": True,
                "stage7_event_accepted": accepted,
                "stage7_event_index": self._stage7_event_count - 1 if accepted else -1,
                "stage7_event_count": self._stage7_event_count,
                "stage7_eligible_tick_count": self._stage7_eligible_ticks,
                "stage7_critic_update_count": self._stage7_critic_updates,
                "stage7_critic_continuous_100hz": True,
                "stage7_current_strategy_reconstructed_100hz": True,
                "stage7_held_strategy_exact_between_events": bool(
                    accepted or previous_held is None
                    or np.array_equal(previous_held, held)
                ),
                "stage7_vartheta": self._stage7_vartheta,
                "stage7_threshold": threshold,
                "stage7_xchi_norm_sq": xchi_norm_sq,
                "stage7_event_age_s": self._stage7_event_age_s,
                "stage7_inter_event_interval_s": interval,
                "stage7_dwell_violation": dwell_violation,
                "stage7_update_reduction": 1.0 - (
                    self._stage7_event_count / self._stage7_eligible_ticks
                ),
                "stage7_physical_contract_error_max_abs": float(
                    np.max(np.abs(contract_error))
                ),
                "stage7_raw_physical_hip": float(baseline[0] + held[2]),
                "stage7_raw_physical_knee": float(baseline[1] + held[3]),
                "stage7_abort": scientific_abort,
                "stage7_abort_reason": "+".join(abort_reasons),
                "stage7_e_bar_norm_sq": error_norm_sq,
                "stage7_e_bar_norm": math.sqrt(error_norm_sq),
                "stage7_muI_hold_error_hip": float(error[2]),
                "stage7_muI_hold_error_knee": float(error[3]),
            }
        )
        for index, label in enumerate(labels):
            result[f"stage7_Uhat_{label}"] = float(current[index])
            result[f"stage7_Uheld_{label}"] = float(held[index])
            result[f"stage7_e_evt_{label}"] = float(error[index])
            result[f"stage7_e_bar_{label}"] = float(normalized_error[index])
        return physical, result

    def _stage7t_residual_event_step(
        self, *, performance: Mapping[str, Any],
        stage6g: Mapping[str, Any], mu0: np.ndarray,
        continuous_command: np.ndarray, time_s: float, dt: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Continuously execute the affine target and hold only Critic residual."""
        result = self._stage7t_defaults()
        if not bool(stage6g.get("stage6g_learning_active")):
            return np.asarray(continuous_command, dtype=float).reshape(2), result

        labels = ("muA_hip", "muA_knee", "muI_hip", "muI_knee")
        affine = np.asarray(
            [stage6g[f"stage6g_{label.split('_')[0]}_affine_{label.split('_')[1]}"]
             for label in labels], dtype=float,
        )
        critic = np.asarray(
            [stage6g[f"stage6g_{label.split('_')[0]}_critic_{label.split('_')[1]}"]
             for label in labels], dtype=float,
        )
        total = np.asarray(
            [stage6g[f"stage6g_{label}"] for label in labels], dtype=float,
        )
        chi = np.asarray(
            [performance[f"chi_{index:02d}"] for index in range(CHI_DIMENSION)],
            dtype=float,
        )
        xchi = chi / STAGE6A_D_CHI_DIAGONAL
        xchi_norm_sq = float(xchi @ xchi)
        if not np.isfinite(np.concatenate((affine, critic, total, xchi))).all():
            raise FloatingPointError("Stage-7T residual event inputs are non-finite")

        decomposition_error = float(np.max(np.abs(total - affine - critic)))
        previous_held = (
            None if self._stage7t_held_critic is None
            else self._stage7t_held_critic.copy()
        )
        accepted = False
        interval: float | str = ""
        self._stage7t_eligible_ticks += 1
        self._stage7t_critic_updates += 1
        if self._stage7t_held_critic is None:
            self._stage7t_held_critic = critic.copy()
            self._stage7t_vartheta = 0.0
            self._stage7t_event_age_s = 0.0
            self._stage7t_event_count = 1
            self._stage7t_last_event_time_s = time_s
            accepted = True
        else:
            decay = math.exp(-STAGE7T_LAMBDA_V_S_INV * dt)
            self._stage7t_vartheta = max(
                0.0,
                decay * self._stage7t_vartheta
                + STAGE7T_C_CHI_EVT * xchi_norm_sq
                * (1.0 - decay) / STAGE7T_LAMBDA_V_S_INV,
            )
            self._stage7t_event_age_s += dt
            pre_error = (
                self._stage7t_held_critic - critic
            ) / STAGE7T_RESIDUAL_SCALE_NM
            pre_error_norm_sq = float(pre_error @ pre_error)
            threshold = (
                STAGE7T_THETA0
                + STAGE7T_KAPPA_E * xchi_norm_sq
                + self._stage7t_vartheta
            )
            if (
                self._stage7t_event_age_s + 1.0e-12 >= STAGE7T_DWELL_S
                and pre_error_norm_sq >= threshold
            ):
                if self._stage7t_last_event_time_s is not None:
                    interval = time_s - self._stage7t_last_event_time_s
                self._stage7t_held_critic = critic.copy()
                self._stage7t_event_age_s = 0.0
                self._stage7t_event_count += 1
                self._stage7t_last_event_time_s = time_s
                accepted = True

        held = np.asarray(self._stage7t_held_critic, dtype=float).reshape(4)
        executed = affine + held
        error = held - critic
        normalized_error = error / STAGE7T_RESIDUAL_SCALE_NM
        error_norm_sq = float(normalized_error @ normalized_error)
        threshold = (
            STAGE7T_THETA0
            + STAGE7T_KAPPA_E * xchi_norm_sq
            + self._stage7t_vartheta
        )
        baseline = np.asarray(mu0, dtype=float).reshape(2)
        physical = baseline + executed[2:]
        affine_execution_error = executed - held - affine
        physical_error = physical - (baseline + affine[2:] + held[2:])
        dwell_violation = bool(
            accepted and interval != ""
            and float(interval) < STAGE7T_DWELL_S - 1.0e-12
        )
        abort_reasons: list[str] = []
        if decomposition_error >= 1.0e-12:
            abort_reasons.append("nash_decomposition_contract_failure")
        if np.any(np.abs(physical) >= STAGE6B_RAW_TOTAL_ABORT_NM):
            abort_reasons.append("raw_total_20Nm_guard_exceeded")
        if dwell_violation:
            abort_reasons.append("dwell_violation")
        if self._stage7t_critic_updates != self._stage7t_eligible_ticks:
            abort_reasons.append("critic_residual_tick_count_mismatch")
        scientific_abort = bool(abort_reasons)
        if scientific_abort:
            self.termination_request = (
                "stage7t_scientific_abort_" + "+".join(abort_reasons)
            )
            physical = np.zeros(2, dtype=float)

        result.update({
            "stage7t_event_logic_active": True,
            "stage7t_event_accepted": accepted,
            "stage7t_event_index": self._stage7t_event_count - 1 if accepted else -1,
            "stage7t_event_count": self._stage7t_event_count,
            "stage7t_eligible_tick_count": self._stage7t_eligible_ticks,
            "stage7t_critic_update_count": self._stage7t_critic_updates,
            "stage7t_critic_continuous_100hz": True,
            "stage7t_affine_continuous_100hz": True,
            "stage7t_current_critic_reconstructed_100hz": True,
            "stage7t_held_critic_exact_between_events": bool(
                accepted or previous_held is None
                or np.array_equal(previous_held, held)
            ),
            "stage7t_vartheta": self._stage7t_vartheta,
            "stage7t_threshold": threshold,
            "stage7t_xchi_norm_sq": xchi_norm_sq,
            "stage7t_event_age_s": self._stage7t_event_age_s,
            "stage7t_inter_event_interval_s": interval,
            "stage7t_dwell_violation": dwell_violation,
            "stage7t_residual_refresh_reduction": 1.0 - (
                self._stage7t_event_count / self._stage7t_eligible_ticks
            ),
            "stage7t_decomposition_contract_error_max_abs": decomposition_error,
            "stage7t_affine_execution_contract_error_max_abs": float(
                np.max(np.abs(affine_execution_error))
            ),
            "stage7t_physical_contract_error_max_abs": float(
                np.max(np.abs(physical_error))
            ),
            "stage7t_raw_physical_hip": float(baseline[0] + executed[2]),
            "stage7t_raw_physical_knee": float(baseline[1] + executed[3]),
            "stage7t_abort": scientific_abort,
            "stage7t_abort_reason": "+".join(abort_reasons),
            "stage7t_e_bar_res_norm_sq": error_norm_sq,
            "stage7t_e_bar_res_norm": math.sqrt(error_norm_sq),
        })
        for index, label in enumerate(labels):
            result[f"stage7t_Uaffine_current_{label}"] = float(affine[index])
            result[f"stage7t_Ucritic_current_{label}"] = float(critic[index])
            result[f"stage7t_Ucritic_held_{label}"] = float(held[index])
            result[f"stage7t_Uexec_{label}"] = float(executed[index])
            result[f"stage7t_e_res_{label}"] = float(error[index])
            result[f"stage7t_e_bar_res_{label}"] = float(normalized_error[index])
        return physical, result

    def compute(
        self, observation: Mapping[str, Any], dt: float
    ) -> dict[str, float]:
        """Run the selected staged controller and return six torque channels."""
        step = _finite_float(dt, "dt")
        if not math.isclose(step, CONTROL_DT_S, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("ICRA requires the locked 100 Hz control step")
        if not isinstance(observation, Mapping):
            raise TypeError("observation must be a mapping")
        time_s = _finite_float(observation.get("time", 0.0), "time")
        contact = observation.get("contact")
        if not isinstance(contact, Mapping):
            contact = {"left_heel": False}
        phase, heel_strike = self._phase(time_s, contact)
        muscles = observation.get("muscles")
        activation_map = muscles.get("activation") if isinstance(muscles, Mapping) else None
        scheduler: dict[str, float] | None = None
        scheduler_legacy: dict[str, float] | None = None
        scheduler_stage6c: dict[str, float] | None = None
        if isinstance(activation_map, Mapping) and all(
            name in activation_map for name in LEFT_INTENTION_MUSCLES
        ):
            activation_values = [
                activation_map[name] for name in LEFT_INTENTION_MUSCLES
            ]
            scheduler_legacy = intention_scheduler_static(activation_values)
            scheduler_stage6c = intention_scheduler_range_calibrated_static(
                activation_values
            )
            scheduler = (
                intention_scheduler_three_segment_static(activation_values)
                if self.mode in (*STAGE6E_MODES, *STAGE6F_MODES, *STAGE6G_MODES, *EVENT_EXECUTION_MODES, *STAGE7T_MODES, *STAGE8A_MODES)
                else scheduler_legacy
            )
            if self._alpha is None:
                # [NUMERICAL INITIALIZATION] Avoid an invented filter transient.
                self._alpha = scheduler["alpha_bar"]
            else:
                self._alpha = alpha_exact_zoh_step(
                    self._alpha, scheduler["alpha_bar"], step
                )

        stage = 8 if self.mode in STAGE8A_MODES else 7 if self.mode in (*EVENT_EXECUTION_MODES, *STAGE7T_MODES) else 6 if self.mode in (
            *STAGE6A_MODES, *STAGE6B_MODES, *STAGE6E_MODES,
            *STAGE6F_MODES, *STAGE6G_MODES, *EVENT_EXECUTION_MODES,
            *STAGE7T_MODES,
            *STAGE8A_MODES,
        ) else (
            5 if self.mode in STAGE5_MODES else (
            4 if self.mode in STAGE4_MODES else (
                3 if self.mode in STAGE3_MODES else (
                    2 if self.mode in STAGE2_MODES else (
                        1 if self.mode == SHADOW_PLANT_SCHEDULER_MODE else 0
                    )
                )
            )
            )
        )
        diagnostics: dict[str, Any] = {
            "icra_stage": stage,
            "mode": self.mode,
            "controlled_dof_count": 2,
            "reference_valid": self._reference_valid,
            "phase": phase,
            "heel_strike": heel_strike,
            "activation_rms": "" if scheduler is None else scheduler["s"],
            "scheduler_s": "" if scheduler is None else scheduler["s"],
            "scheduler_a": "" if scheduler is None else scheduler["a"],
            "scheduler_alpha_bar": "" if scheduler is None else scheduler["alpha_bar"],
            "scheduler_alpha": "" if self._alpha is None else self._alpha,
            "scheduler_alpha_shadow": "" if self._alpha is None else self._alpha,
            "scheduler_mapping": (
                "stage6d_primary_three_segment_source_only"
                if self.mode in (*STAGE6E_MODES, *STAGE6F_MODES, *STAGE6G_MODES, *EVENT_EXECUTION_MODES, *STAGE7T_MODES, *STAGE8A_MODES)
                else "legacy_exponential"
            ),
            "scheduler_s_raw": "" if scheduler is None else scheduler["s"],
            "scheduler_legacy_alpha_target": (
                "" if scheduler_legacy is None
                else scheduler_legacy["alpha_bar"]
            ),
            "scheduler_stage6c_alpha_target": (
                "" if scheduler_stage6c is None
                else scheduler_stage6c["alpha_bar"]
            ),
            "stage6e_shadow_only": self.mode in STAGE6E_MODES,
            "stage6e_physical_torque_asserted_zero": False,
            "stage6e_proposed_hip_l_nm": 0.0,
            "stage6e_proposed_knee_l_nm": 0.0,
            "stage6e_physical_hip_l_nm": 0.0,
            "stage6e_physical_knee_l_nm": 0.0,
            "K_H0": FIXED_K_HIP_NM_PER_RAD,
            "K_K0": FIXED_K_KNEE_NM_PER_RAD,
            "K_fixed_hip": FIXED_K_HIP_NM_PER_RAD,
            "K_fixed_knee": FIXED_K_KNEE_NM_PER_RAD,
            "analytical_model": "smooth_schur_2d",
            "physical_oracle": "constrained_kkt_2d",
            "baseline_continuity_confirmed": True,
            "hard_kkt_feedforward_used": False,
            "joint_limit_cancelled": False,
            "m5_cancelled": False,
            "lambda_fixed_hip": float(FIXED_LAMBDA_S_INV[0]),
            "lambda_fixed_knee": float(FIXED_LAMBDA_S_INV[1]),
            "gamma_fixed_hip": float(FIXED_GAMMA_S_INV[0]),
            "gamma_fixed_knee": float(FIXED_GAMMA_S_INV[1]),
            "fixed_z_scale_hip": float(FIXED_ADP_Z_SCALE_RAD_S[0]),
            "fixed_z_scale_knee": float(FIXED_ADP_Z_SCALE_RAD_S[1]),
            "plant_valid": False,
            "plant_constraint_count": 0,
            "M_eff_00": "",
            "M_eff_01": "",
            "M_eff_10": "",
            "M_eff_11": "",
            "M_eff_lambda_min": "",
            "M_eff_lambda_max": "",
            "M_eff_condition": "",
            "M_smooth_00": "",
            "M_smooth_01": "",
            "M_smooth_10": "",
            "M_smooth_11": "",
            "M_smooth_lambda_min": "",
            "M_smooth_lambda_max": "",
            "M_smooth_condition": "",
            "b_noHRI_hip": "",
            "b_noHRI_knee": "",
            "b_smooth_hip": "",
            "b_smooth_knee": "",
            "tau_h_direct_hip": "",
            "tau_h_direct_knee": "",
            "tau_h_eff_hip": "",
            "tau_h_eff_knee": "",
            "tau_h_smooth_hip": "",
            "tau_h_smooth_knee": "",
            "passive_smooth_hip": "",
            "passive_smooth_knee": "",
            "hri_identity_error_norm": "",
            "external_accounting_error_max_abs": "",
            "external_contains_m5": True,
            "external_other_terms": "qfrc_passive",
            "active_exo_hip_l": False,
            "active_exo_knee_l": False,
            "active_exo_hip_r": False,
            "active_exo_knee_r": False,
            "active_row_identities": "none",
            "active_set_selection": "state_rows_plus_current_left_qacc_no_measured_unactuated_qacc_or_force",
            "q_ref_hip": "",
            "q_ref_knee": "",
            "qdot_ref_hip": "",
            "qdot_ref_knee": "",
            "qddot_ref_hip": "",
            "qddot_ref_knee": "",
            "error_hip": "",
            "error_knee": "",
            "t1_hip": "",
            "t1_knee": "",
            "t1_dot_hip": "",
            "t1_dot_knee": "",
            "qdot_r_hip": "",
            "qdot_r_knee": "",
            "qddot_r_hip": "",
            "qddot_r_knee": "",
            "z_hip": "",
            "z_knee": "",
            "transform_mismatch_norm": "",
            "Ken_00": float(FIXED_KEN_NMS_PER_RAD[0, 0]),
            "Ken_01": float(FIXED_KEN_NMS_PER_RAD[0, 1]),
            "Ken_10": float(FIXED_KEN_NMS_PER_RAD[1, 0]),
            "Ken_11": float(FIXED_KEN_NMS_PER_RAD[1, 1]),
            "engagement_ramp": 0.0,
            "u0_hip": 0.0,
            "u0_knee": 0.0,
            "nominal_z_equation_smooth": False,
            "nominal_z_equation_error_norm": "",
            "disturbance_sample_completed": False,
            "d_residual_hip": "",
            "d_residual_knee": "",
            "d_residual_norm": "",
            "d_residual_phase": "",
            "d_residual_constraint_active": False,
            "d_z_hip": "",
            "d_z_knee": "",
            "d_z_minus_residual_norm": "",
            "d_z_accounting_expected": False,
            "critic_enabled": self.mode in FIXED_K_ADP_PHYSICAL_MODES,
            "critic_learning_active": False,
            "critic_sample_completed": False,
            "W0": float(self._critic_weight[0]),
            "W1": float(self._critic_weight[1]),
            "W2": float(self._critic_weight[2]),
            "critic_norm": float(np.linalg.norm(self._critic_weight)),
            "value_hat": 0.0,
            "value_min_eigenvalue": fixed_value_minimum_eigenvalue(
                self._critic_weight
            ),
            "bellman": "",
            "m_norm": "",
            "critic_update_norm": 0.0,
            "critic_backtracking_count": 0,
            "critic_update_skipped": False,
            "critic_valid_sample_count": self._critic_valid_samples,
            "critic_skip_fraction": 0.0,
            "mu_hip": 0.0,
            "mu_knee": 0.0,
            "u_total_hip": 0.0,
            "u_total_knee": 0.0,
            "native_saturation_command": False,
            "event_enabled": False,
            "right_motor_zero": True,
            "motor_hip_l": 0.0,
            "motor_knee_l": 0.0,
            "motor_hip_r": 0.0,
            "motor_knee_r": 0.0,
            "oracle_sampled": False,
            "oracle_baseline_mode": "",
            "oracle_mode_switch_0p1": "",
            "oracle_mode_switch_0p5": "",
            "oracle_mode_switch_1p0": "",
            "oracle_relative_error_0p1": "",
            "oracle_relative_error_0p5": "",
            "oracle_relative_error_1p0": "",
            "oracle_test_smooth_count": "",
            "oracle_test_mode_switch_fraction": "",
            "oracle_test_abs_error_rms": "",
            "oracle_test_relative_error_rms": "",
            "oracle_test_relative_error_max": "",
            "inverse_finite": "",
            "inverse_residual_rms": "",
            "inverse_residual_max_abs": "",
        }
        diagnostics.update(self._game_shadow_defaults())
        diagnostics.update(self._performance_shadow_defaults())
        diagnostics.update(self._stage5_defaults())
        diagnostics.update(self._stage6b_defaults())
        diagnostics.update(self._stage6f_defaults())
        diagnostics.update(self._stage6g_defaults())
        diagnostics.update(self._stage7_defaults())
        diagnostics.update(self._stage7t_defaults())
        exoskeleton = observation.get("exoskeleton")
        plant_result: dict[str, Any] | None = None
        if self.mode != SHADOW_CONTRACT_MODE:
            if not isinstance(exoskeleton, Mapping) or self.plant is None:
                raise KeyError("ICRA plant modes require exoskeleton dynamics")
            dynamics = exoskeleton.get("bilateral_dynamics")
            if not isinstance(dynamics, Mapping):
                raise KeyError("ICRA bilateral dynamics are missing")
            qacc = np.asarray(
                (
                    exoskeleton["hip_l"]["acceleration_rad_s2"],
                    exoskeleton["knee_l"]["acceleration_rad_s2"],
                ),
                dtype=float,
            )
            plant_result = self.plant.evaluate(dynamics, qacc)
            meff = np.asarray(plant_result["M_eff"], dtype=float)
            b_no_hri = np.asarray(plant_result["b_noHRI"], dtype=float)
            direct = np.asarray(plant_result["tau_h_direct"], dtype=float)
            effective = np.asarray(plant_result["tau_h_eff"], dtype=float)
            smooth_meff = np.asarray(plant_result["M_smooth"], dtype=float)
            smooth_bias = np.asarray(plant_result["b_smooth"], dtype=float)
            smooth_interaction = np.asarray(
                plant_result["tau_h_smooth"], dtype=float
            )
            active_labels = set(plant_result["active_labels"])
            diagnostics.update(
                {
                    "plant_valid": True,
                    "plant_constraint_count": len(active_labels),
                    "M_eff_00": float(meff[0, 0]),
                    "M_eff_01": float(meff[0, 1]),
                    "M_eff_10": float(meff[1, 0]),
                    "M_eff_11": float(meff[1, 1]),
                    "M_eff_lambda_min": float(plant_result["eigenvalues"][0]),
                    "M_eff_lambda_max": float(plant_result["eigenvalues"][1]),
                    "M_eff_condition": float(plant_result["condition"]),
                    "M_smooth_00": float(smooth_meff[0, 0]),
                    "M_smooth_01": float(smooth_meff[0, 1]),
                    "M_smooth_10": float(smooth_meff[1, 0]),
                    "M_smooth_11": float(smooth_meff[1, 1]),
                    "M_smooth_lambda_min": float(
                        plant_result["smooth_eigenvalues"][0]
                    ),
                    "M_smooth_lambda_max": float(
                        plant_result["smooth_eigenvalues"][1]
                    ),
                    "M_smooth_condition": float(
                        plant_result["smooth_condition"]
                    ),
                    "b_noHRI_hip": float(b_no_hri[0]),
                    "b_noHRI_knee": float(b_no_hri[1]),
                    "b_smooth_hip": float(smooth_bias[0]),
                    "b_smooth_knee": float(smooth_bias[1]),
                    "tau_h_direct_hip": float(direct[0]),
                    "tau_h_direct_knee": float(direct[1]),
                    "tau_h_eff_hip": float(effective[0]),
                    "tau_h_eff_knee": float(effective[1]),
                    "tau_h_smooth_hip": float(smooth_interaction[0]),
                    "tau_h_smooth_knee": float(smooth_interaction[1]),
                    "passive_smooth_hip": float(
                        plant_result["passive_smooth"][0]
                    ),
                    "passive_smooth_knee": float(
                        plant_result["passive_smooth"][1]
                    ),
                    "hri_identity_error_norm": float(
                        np.linalg.norm(plant_result["identity_error"])
                    ),
                    "external_accounting_error_max_abs": float(
                        np.max(np.abs(plant_result["external_accounting_error"]))
                    ),
                    "active_exo_hip_l": "exo_hip_l" in active_labels,
                    "active_exo_knee_l": "exo_knee_l" in active_labels,
                    "active_exo_hip_r": "exo_hip_r" in active_labels,
                    "active_exo_knee_r": "exo_knee_r" in active_labels,
                    "active_row_identities": "+".join(sorted(active_labels))
                    or "none",
                }
            )
            snapshot = exoskeleton.get("runtime_snapshot")
            if (
                self.mode == SHADOW_PLANT_SCHEDULER_MODE
                and self._step_index % ORACLE_SAMPLE_STRIDE == 0
            ):
                if not isinstance(snapshot, Mapping):
                    raise KeyError("ICRA scratch runtime snapshot is missing")
                diagnostics.update(self._oracle_diagnostics(snapshot, plant_result))

            if (
                self.mode in (
                    *STAGE5_MODES, *STAGE6A_MODES, *STAGE6B_MODES,
                    *STAGE6E_MODES,
                    *STAGE6F_MODES, *STAGE6G_MODES, *EVENT_EXECUTION_MODES,
                    *STAGE7T_MODES,
                    *STAGE8A_MODES,
                )
                and self._reference_valid
            ):
                diagnostics.update(self._complete_impedance_pending(qacc))
                if self.reference is None or self._alpha is None:
                    raise RuntimeError(
                        "Stage-5 impedance requires reference and scheduler"
                    )
                q_all = np.asarray(dynamics["q_exo"], dtype=float).reshape(7)
                qvel_all = np.asarray(
                    dynamics["qvel_exo"], dtype=float
                ).reshape(7)
                q = q_all[np.asarray(CONTROLLED_INDICES)]
                qdot = qvel_all[np.asarray(CONTROLLED_INDICES)]
                qd, qdotd, qddotd = self.reference.evaluate(
                    float(phase), self._period_s
                )
                error = q - qd
                error_velocity = qdot - qdotd
                u_inertia = smooth_meff @ qddotd
                u_bias = smooth_bias.copy()
                u_damping = -GAME_DI_NMS_PER_RAD @ error_velocity
                u_stiffness = -GAME_KI_NM_PER_RAD * error
                mu0 = u_inertia + u_bias + u_damping + u_stiffness
                ramp = self._engagement_ramp(time_s)
                command = ramp * mu0
                values = np.concatenate(
                    (
                        qd, qdotd, qddotd, error, error_velocity,
                        u_inertia, u_bias, u_damping, u_stiffness,
                        mu0, command,
                    )
                )
                if not np.isfinite(values).all():
                    raise FloatingPointError(
                        "Stage-5 impedance command is non-finite"
                    )

                self._impedance_pending = {
                    "M_smooth": smooth_meff.copy(),
                    "b_smooth": smooth_bias.copy(),
                    # MuJoCo applies the native actuator ctrlrange after the
                    # controller returns.  Retain the raw command above, but
                    # use the physically realized native-limited torque for
                    # the next-interval accounting residual.
                    "u_actual": np.clip(
                        command, -NATIVE_MOTOR_LIMIT_NM,
                        NATIVE_MOTOR_LIMIT_NM,
                    ),
                    "tau_h_smooth": smooth_interaction.copy(),
                    "qddot_d": qddotd.copy(),
                    "error": error.copy(),
                    "error_velocity": error_velocity.copy(),
                    "phase": float(phase),
                    "constraint_active": bool(active_labels),
                    "accounting_expected": bool(
                        ramp == 1.0
                        and np.all(np.abs(command) < NATIVE_MOTOR_LIMIT_NM)
                    ),
                }

                game_diagnostics = self._game_shadow_diagnostics(
                    error_i=error,
                    error_velocity_i=error_velocity,
                    smooth_mass=smooth_meff,
                    smooth_interaction=smooth_interaction,
                    alpha=self._alpha,
                    fixed_k_command=np.zeros(2, dtype=float),
                    dt=step,
                )
                diagnostics.update(game_diagnostics)
                performance_diagnostics = self._performance_shadow_diagnostics(
                    game=game_diagnostics,
                    qddot_reference=qddotd,
                    smooth_mass=smooth_meff,
                    smooth_bias=smooth_bias,
                    ramp=ramp,
                    time_s=time_s,
                )
                if self.mode in STAGE6B_MODES:
                    command, stage6b_diagnostics = self._stage6b_active_step(
                        performance=performance_diagnostics,
                        smooth_mass=smooth_meff,
                        alpha=self._alpha,
                        ramp=ramp,
                        mu0=mu0,
                        dt=step,
                    )
                    if bool(stage6b_diagnostics["stage6b_ppc_invalid"]):
                        # Preserve the finite pre-abort diagnostic row.  The
                        # PPC transform itself is not clipped or repaired.
                        performance_diagnostics = dict(performance_diagnostics)
                        for key, value in performance_diagnostics.items():
                            if isinstance(value, float) and not math.isfinite(value):
                                performance_diagnostics[key] = ""
                    diagnostics.update(stage6b_diagnostics)
                elif self.mode in STAGE6E_MODES:
                    proposed, stage6e_diagnostics = self._stage6b_active_step(
                        performance=performance_diagnostics,
                        smooth_mass=smooth_meff,
                        alpha=self._alpha,
                        ramp=ramp,
                        mu0=mu0,
                        dt=step,
                    )
                    # Stage-6E audits the complete frozen game/Critic proposal
                    # but never permits a controller-side scientific guard to
                    # terminate or alter the zero-torque physical trajectory.
                    self.termination_request = ""
                    diagnostics.update(stage6e_diagnostics)
                    diagnostics.update(
                        {
                            "stage6e_proposed_hip_l_nm": float(proposed[0]),
                            "stage6e_proposed_knee_l_nm": float(proposed[1]),
                        }
                    )
                    command = np.zeros(2, dtype=float)
                elif self.mode in STAGE6F_MODES:
                    command, stage6f_diagnostics = self._stage6f_step(
                        performance=performance_diagnostics,
                        smooth_mass=smooth_meff,
                        alpha=self._alpha,
                        nominal_ramp=ramp,
                        mu0=mu0,
                        time_s=time_s,
                        dt=step,
                    )
                    if bool(stage6f_diagnostics["stage6f_abort"]):
                        performance_diagnostics = dict(performance_diagnostics)
                        for key, value in performance_diagnostics.items():
                            if isinstance(value, float) and not math.isfinite(value):
                                performance_diagnostics[key] = ""
                    diagnostics.update(stage6f_diagnostics)
                elif self.mode in (*STAGE6G_MODES, *EVENT_EXECUTION_MODES, *STAGE7T_MODES, *STAGE8A_MODES):
                    command, stage6g_diagnostics = self._stage6g_step(
                        performance=performance_diagnostics,
                        smooth_mass=smooth_meff,
                        alpha=self._alpha,
                        nominal_ramp=ramp,
                        mu0=mu0,
                        time_s=time_s,
                        dt=step,
                    )
                    if bool(stage6g_diagnostics["stage6g_abort"]):
                        performance_diagnostics = dict(performance_diagnostics)
                        for key, value in performance_diagnostics.items():
                            if isinstance(value, float) and not math.isfinite(value):
                                performance_diagnostics[key] = ""
                    diagnostics.update(stage6g_diagnostics)
                    if self.mode in EVENT_EXECUTION_MODES:
                        command, stage7_diagnostics = self._stage7_event_step(
                            performance=performance_diagnostics,
                            stage6g=stage6g_diagnostics,
                            mu0=mu0,
                            continuous_command=command,
                            time_s=time_s,
                            dt=step,
                        )
                        diagnostics.update(stage7_diagnostics)
                    elif self.mode in STAGE7T_MODES:
                        command, stage7t_diagnostics = self._stage7t_residual_event_step(
                            performance=performance_diagnostics,
                            stage6g=stage6g_diagnostics,
                            mu0=mu0,
                            continuous_command=command,
                            time_s=time_s,
                            dt=step,
                        )
                        diagnostics.update(stage7t_diagnostics)
                diagnostics.update(performance_diagnostics)

                strategic_mu_i = (
                    np.asarray(
                        (
                            (
                                diagnostics["stage7_Uheld_muI_hip"]
                                if diagnostics["stage7_event_logic_active"]
                                else command[0] - mu0[0]
                            )
                            if self.mode in EVENT_EXECUTION_MODES
                            else command[0] - mu0[0]
                            if self.mode in STAGE7T_MODES
                            else diagnostics["stage6g_muI_hip"]
                            if self.mode in STAGE8A_MODES
                            else diagnostics["stage6g_muI_hip"]
                            if self.mode == INTENTION_TARGET_GAME_ACTIVE_MODE
                            else diagnostics["stage6f_muI_hip"]
                            if self.mode == PHYSICAL_ACTUATION_GAME_ACTIVE_MODE
                            else diagnostics["stage6b_muI_hip"],
                            (
                                diagnostics["stage7_Uheld_muI_knee"]
                                if diagnostics["stage7_event_logic_active"]
                                else command[1] - mu0[1]
                            )
                            if self.mode in EVENT_EXECUTION_MODES
                            else command[1] - mu0[1]
                            if self.mode in STAGE7T_MODES
                            else diagnostics["stage6g_muI_knee"]
                            if self.mode in STAGE8A_MODES
                            else diagnostics["stage6g_muI_knee"]
                            if self.mode == INTENTION_TARGET_GAME_ACTIVE_MODE
                            else diagnostics["stage6f_muI_knee"]
                            if self.mode == PHYSICAL_ACTUATION_GAME_ACTIVE_MODE
                            else diagnostics["stage6b_muI_knee"],
                        ),
                        dtype=float,
                    )
                    if self.mode in (
                        *STAGE6B_MODES, PHYSICAL_ACTUATION_GAME_ACTIVE_MODE,
                        INTENTION_TARGET_GAME_ACTIVE_MODE, *EVENT_EXECUTION_MODES,
                        *STAGE7T_MODES,
                        *STAGE8A_MODES,
                    )
                    else np.zeros(2, dtype=float)
                )
                self._impedance_pending["u_actual"] = np.clip(
                    command, -NATIVE_MOTOR_LIMIT_NM, NATIVE_MOTOR_LIMIT_NM
                )
                self._impedance_pending["strategic_muI"] = strategic_mu_i
                diagnostics.update(
                    {
                        "q_ref_hip": float(qd[0]),
                        "q_ref_knee": float(qd[1]),
                        "qdot_ref_hip": float(qdotd[0]),
                        "qdot_ref_knee": float(qdotd[1]),
                        "qddot_ref_hip": float(qddotd[0]),
                        "qddot_ref_knee": float(qddotd[1]),
                        "error_hip": float(error[0]),
                        "error_knee": float(error[1]),
                        "engagement_ramp": ramp,
                        "u0_hip": float(mu0[0]),
                        "u0_knee": float(mu0[1]),
                        "u_total_hip": float(command[0]),
                        "u_total_knee": float(command[1]),
                        "native_saturation_command": bool(
                            np.any(np.abs(command) >= NATIVE_MOTOR_LIMIT_NM)
                        ),
                        "motor_hip_l": float(command[0]),
                        "motor_knee_l": float(command[1]),
                        "stage5_q_ref_hip": float(qd[0]),
                        "stage5_q_ref_knee": float(qd[1]),
                        "stage5_qdot_ref_hip": float(qdotd[0]),
                        "stage5_qdot_ref_knee": float(qdotd[1]),
                        "stage5_qddot_ref_hip": float(qddotd[0]),
                        "stage5_qddot_ref_knee": float(qddotd[1]),
                        "stage5_error_hip": float(error[0]),
                        "stage5_error_knee": float(error[1]),
                        "stage5_error_velocity_hip": float(
                            error_velocity[0]
                        ),
                        "stage5_error_velocity_knee": float(
                            error_velocity[1]
                        ),
                        "stage5_u_inertia_hip": float(u_inertia[0]),
                        "stage5_u_inertia_knee": float(u_inertia[1]),
                        "stage5_u_bias_hip": float(u_bias[0]),
                        "stage5_u_bias_knee": float(u_bias[1]),
                        "stage5_u_damping_hip": float(u_damping[0]),
                        "stage5_u_damping_knee": float(u_damping[1]),
                        "stage5_u_stiffness_hip": float(u_stiffness[0]),
                        "stage5_u_stiffness_knee": float(u_stiffness[1]),
                        "stage5_mu0_hip": float(mu0[0]),
                        "stage5_mu0_knee": float(mu0[1]),
                        "stage5_command_hip": float(ramp * mu0[0]),
                        "stage5_command_knee": float(ramp * mu0[1]),
                        "stage5_engagement_ramp": ramp,
                    }
                )
                torques = dict(ZERO_EXO_TORQUES)
                torques["HipExo_L"] = float(command[0])
                torques["KneeExo_L"] = float(command[1])
                if self.mode in (
                    *STAGE6E_MODES, PHYSICAL_ACTUATION_GAME_SHADOW_MODE,
                    INTENTION_TARGET_GAME_SHADOW_MODE,
                ):
                    if any(value != 0.0 for value in torques.values()):
                        raise RuntimeError(
                            "Stage-6E shadow attempted nonzero physical torque"
                        )
                    diagnostics["stage6e_physical_torque_asserted_zero"] = True
                self.last_diagnostics = diagnostics
                self._step_index += 1
                return torques

            if self.mode in FIXED_K_FAMILY_MODES and self._reference_valid:
                model_step = self._complete_model_pending(qacc)
                diagnostics.update(model_step)
                q_all = np.asarray(dynamics["q_exo"], dtype=float).reshape(7)
                qvel_all = np.asarray(dynamics["qvel_exo"], dtype=float).reshape(7)
                q = q_all[np.asarray(CONTROLLED_INDICES)]
                qdot = qvel_all[np.asarray(CONTROLLED_INDICES)]
                transform = self._fixed_transform(
                    q,
                    qdot,
                    qacc,
                    smooth_interaction,
                    float(phase),
                    step,
                )
                qddot_r = np.asarray(transform["qddot_r"], dtype=float)
                z = np.asarray(transform["z"], dtype=float)
                g = np.linalg.solve(smooth_meff, np.eye(2, dtype=float))
                u0 = (
                    smooth_meff @ qddot_r
                    + smooth_bias
                    - FIXED_KEN_NMS_PER_RAD @ z
                )
                ramp = 0.0
                if self.mode in (
                    FIXED_K_NOMINAL_MODE,
                    *FIXED_K_ADP_PHYSICAL_MODES,
                ):
                    ramp = self._engagement_ramp(time_s)

                critic_step = {
                    "critic_sample_completed": False,
                    "bellman": "",
                    "m_norm": "",
                    "critic_update_norm": 0.0,
                    "critic_backtracking_count": 0,
                    "critic_update_skipped": False,
                }
                mu = np.zeros(2, dtype=float)
                features, jacobian = fixed_b3_basis(z)
                if self.mode in FIXED_K_ADP_PHYSICAL_MODES:
                    if ramp < 1.0:
                        self._critic_pending = None
                        self._learning_active = False
                    else:
                        if not self._learning_active:
                            self._critic_pending = None
                            self._learning_active = True
                        else:
                            critic_step = self._complete_pending_critic(qacc, step)
                        gradient = jacobian @ self._critic_weight
                        mu = -0.5 * FIXED_ADP_R_INV @ g.T @ gradient
                else:
                    gradient = jacobian @ self._critic_weight
                    self._critic_pending = None
                    self._learning_active = False

                total_unramped = u0 + mu
                command = (
                    np.zeros(2, dtype=float)
                    if self.mode == FIXED_K_SHADOW_MODE
                    else ramp * total_unramped
                )
                if not np.isfinite(
                    np.concatenate((u0, mu, total_unramped, command))
                ).all():
                    raise FloatingPointError("Fixed-K controller torque is non-finite")

                nominal_smooth = bool(model_step["d_z_accounting_expected"])
                nominal_error: float | str = (
                    model_step["d_z_minus_residual_norm"]
                    if nominal_smooth
                    else ""
                )

                value_hat = float(self._critic_weight @ features)
                value_min = fixed_value_minimum_eigenvalue(self._critic_weight)
                skip_fraction = (
                    0.0
                    if self._critic_valid_samples == 0
                    else self._critic_skipped_samples / self._critic_valid_samples
                )
                if (
                    self.mode in FIXED_K_ADP_PHYSICAL_MODES
                    and self._learning_active
                ):
                    gradient = jacobian @ self._critic_weight
                    mu = -0.5 * FIXED_ADP_R_INV @ g.T @ gradient
                    total_unramped = u0 + mu
                    command = ramp * total_unramped
                    features, jacobian = fixed_b3_basis(z)
                    value_hat = float(self._critic_weight @ features)
                    value_min = fixed_value_minimum_eigenvalue(self._critic_weight)
                    self._critic_pending = {
                        "time_s": time_s,
                        "z": z.copy(),
                        "qddot_r": qddot_r.copy(),
                        "g": g.copy(),
                        "mu_applied": mu.copy(),
                        "mu_policy": mu.copy(),
                        "features": features.copy(),
                        "jacobian": jacobian.copy(),
                        "gradient": gradient.copy(),
                        "weight": self._critic_weight.copy(),
                        "u0": u0.copy(),
                    }

                self._model_pending = {
                    "M_smooth": smooth_meff.copy(),
                    "b_smooth": smooth_bias.copy(),
                    # Stage 2R has no software torque clipping.  Under its hard
                    # no-saturation gate the commanded and native applied
                    # torques are identical, so the aligned residual uses the
                    # unmodified command from the completed interval.
                    "u_actual": command.copy(),
                    "tau_h_smooth": smooth_interaction.copy(),
                    "qddot_r": qddot_r.copy(),
                    "z": z.copy(),
                    "phase": float(phase),
                    "constraint_active": bool(active_labels),
                    "accounting_expected": bool(
                        self.mode == FIXED_K_NOMINAL_MODE
                        and ramp == 1.0
                        and np.all(np.abs(command) < NATIVE_MOTOR_LIMIT_NM)
                    ),
                }

                diagnostics.update(critic_step)
                diagnostics.update(
                    {
                        "q_ref_hip": float(transform["qd"][0]),
                        "q_ref_knee": float(transform["qd"][1]),
                        "qdot_ref_hip": float(transform["qdotd"][0]),
                        "qdot_ref_knee": float(transform["qdotd"][1]),
                        "qddot_ref_hip": float(transform["qddotd"][0]),
                        "qddot_ref_knee": float(transform["qddotd"][1]),
                        "error_hip": float(transform["error"][0]),
                        "error_knee": float(transform["error"][1]),
                        "t1_hip": float(transform["t1"][0]),
                        "t1_knee": float(transform["t1"][1]),
                        "t1_dot_hip": float(transform["t1_dot"][0]),
                        "t1_dot_knee": float(transform["t1_dot"][1]),
                        "qdot_r_hip": float(transform["qdot_r"][0]),
                        "qdot_r_knee": float(transform["qdot_r"][1]),
                        "qddot_r_hip": float(qddot_r[0]),
                        "qddot_r_knee": float(qddot_r[1]),
                        "z_hip": float(z[0]),
                        "z_knee": float(z[1]),
                        "transform_mismatch_norm": float(
                            transform["mismatch_norm"]
                        ),
                        "engagement_ramp": ramp,
                        "u0_hip": float(u0[0]),
                        "u0_knee": float(u0[1]),
                        "nominal_z_equation_smooth": nominal_smooth,
                        "nominal_z_equation_error_norm": nominal_error,
                        "critic_learning_active": self._learning_active,
                        "W0": float(self._critic_weight[0]),
                        "W1": float(self._critic_weight[1]),
                        "W2": float(self._critic_weight[2]),
                        "critic_norm": float(np.linalg.norm(self._critic_weight)),
                        "value_hat": value_hat,
                        "value_min_eigenvalue": value_min,
                        "critic_valid_sample_count": self._critic_valid_samples,
                        "critic_skip_fraction": skip_fraction,
                        "mu_hip": float(mu[0]),
                        "mu_knee": float(mu[1]),
                        "u_total_hip": float(command[0]),
                        "u_total_knee": float(command[1]),
                        "native_saturation_command": bool(
                            np.any(np.abs(command) >= NATIVE_MOTOR_LIMIT_NM)
                        ),
                        "motor_hip_l": float(command[0]),
                        "motor_knee_l": float(command[1]),
                    }
                )
                if self.mode in (*STAGE3_MODES, *STAGE4_MODES):
                    if self._alpha is None:
                        raise RuntimeError(
                            "Game/performance shadow requires the locked scheduler"
                        )
                    game_diagnostics = self._game_shadow_diagnostics(
                        error_i=np.asarray(transform["error"], dtype=float),
                        error_velocity_i=np.asarray(
                            transform["error_velocity"], dtype=float
                        ),
                        smooth_mass=smooth_meff,
                        smooth_interaction=smooth_interaction,
                        alpha=self._alpha,
                        fixed_k_command=command,
                        dt=step,
                    )
                    diagnostics.update(game_diagnostics)
                    if self.mode == PERFORMANCE_SHADOW_MODE:
                        diagnostics.update(
                            self._performance_shadow_diagnostics(
                                game=game_diagnostics,
                                qddot_reference=np.asarray(
                                    transform["qddotd"], dtype=float
                                ),
                                smooth_mass=smooth_meff,
                                smooth_bias=smooth_bias,
                                ramp=ramp,
                                time_s=time_s,
                            )
                        )

                torques = dict(ZERO_EXO_TORQUES)
                torques["HipExo_L"] = float(command[0])
                torques["KneeExo_L"] = float(command[1])
                self.last_diagnostics = diagnostics
                self._step_index += 1
                return torques
        torques = dict(ZERO_EXO_TORQUES)
        if self.mode in (
            *STAGE6E_MODES, PHYSICAL_ACTUATION_GAME_SHADOW_MODE,
            INTENTION_TARGET_GAME_SHADOW_MODE,
        ):
            if any(value != 0.0 for value in torques.values()):
                raise RuntimeError(
                    "ICRA full-shadow mode attempted nonzero physical torque"
                )
            diagnostics["stage6e_physical_torque_asserted_zero"] = True
        self.last_diagnostics = diagnostics
        self._step_index += 1
        return torques


__all__ = [
    "ICRAController",
    "ICRAReducedPlant",
    "ICRA_IMPEDANCE_BASELINE_MODE",
    "CRITIC_NASH_SHADOW_MODE",
    "CONTINUOUS_ACTIVE_GAME_MODE",
    "CALIBRATED_SCALAR_GAME_SHADOW_MODE",
    "PHYSICAL_ACTUATION_GAME_SHADOW_MODE",
    "PHYSICAL_ACTUATION_GAME_ACTIVE_MODE",
    "INTENTION_TARGET_GAME_SHADOW_MODE",
    "INTENTION_TARGET_GAME_ACTIVE_MODE",
    "EVENT_TRIGGERED_INTENTION_TARGET_GAME_ACTIVE_MODE",
    "CONSERVATIVE_EVENT_TRIGGERED_INTENTION_TARGET_GAME_ACTIVE_MODE",
    "DWELL_LIMITED_P25_EVENT_TRIGGERED_INTENTION_TARGET_GAME_ACTIVE_MODE",
    "RESIDUAL_EVENT_TRIGGERED_INTENTION_TARGET_GAME_ACTIVE_MODE",
    "PPC_TIMING_CHARACTERIZATION_MODE",
    "PT_HORIZON_CHARACTERIZATION_MODE",
    "STAGE7_STRATEGY_SCALE_NM",
    "STAGE7_THETA0",
    "STAGE7_DWELL_S",
    "STAGE7_LAMBDA_V_S_INV",
    "STAGE7_KAPPA_E",
    "STAGE7_C_CHI_EVT",
    "STAGE7R_THETA0",
    "STAGE7R_DWELL_S",
    "STAGE7R_LAMBDA_V_S_INV",
    "STAGE7R_KAPPA_E",
    "STAGE7R_C_CHI_EVT",
    "STAGE7S_THETA0",
    "STAGE7S_DWELL_S",
    "STAGE7S_LAMBDA_V_S_INV",
    "STAGE7S_KAPPA_E",
    "STAGE7S_C_CHI_EVT",
    "STAGE7T_RESIDUAL_SCALE_NM",
    "STAGE7T_THETA0",
    "STAGE7T_DWELL_S",
    "STAGE7T_LAMBDA_V_S_INV",
    "STAGE7T_KAPPA_E",
    "STAGE7T_C_CHI_EVT",
    "STAGE6A_CHI_ORDER",
    "STAGE6A_D_CHI_DIAGONAL",
    "STAGE6A_Q_EPS",
    "STAGE6A_DISCOUNT_RATE_S_INV",
    "intention_scheduler_static",
    "intention_scheduler_range_calibrated_static",
    "intention_scheduler_three_segment_static",
    "alpha_exact_zoh_step",
    "admittance_exact_zoh_step",
    "fixed_b3_basis",
    "fixed_value_minimum_eigenvalue",
    "game_eta",
    "game_nash_matrix",
    "ppc_envelope",
    "ppc_envelope_with_limits",
    "ppc_transform",
    "ppc_transform_given_envelope",
    "predefined_time_scaling",
    "predefined_time_scaling_with_parameters",
    "quadratic_basis_78",
    "quadratic_feature_names_78",
    "stage6a_normalize_chi",
    "stage6a_value_gradient_chi",
    "stage6a_strategy_input_matrices",
    "stage6a_running_costs",
    "stage6a_actor_matrices",
    "stage6a_nash_actor",
    "stage6a_critic_update",
    "stage6a_project_weight_columns",
    "stage6a_value_matrix",
    "stage6b_causal_control_step",
    "stage6f_total_actuation_beta",
    "stage6f_nash_matrix",
    "stage6f_nash_actor",
    "stage6f_running_costs",
    "stage6f_critic_update",
    "stage6f_causal_control_step",
    "stage6g_assistance_gamma",
    "stage6g_nash_matrix",
    "stage6g_nash_actor",
    "stage6g_running_costs",
    "stage6g_critic_update",
    "stage6g_causal_control_step",
    "symmetric_quadratic_feature_count",
]
