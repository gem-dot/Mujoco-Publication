"""Flexible-HRI adapted YAC：Nominal、Continuous ADP 与 Dynamic Event。

本文件是 YAC 算法的唯一源码文件。Stage D-R 冻结 Stage A-D 的参考、
阻抗变换、nominal dynamics、Ken 与 Flexible-HRI interaction 语义。
Stage E1 在该冻结 nominal baseline 上增加 B6 single-Critic、Paper exact
normalized update 与 continuous ADP residual。Full YAC 进一步按 Paper 显式
Eq. (28) 对 policy 的 z 做 dynamic-event sampling；Critic 仍以 100 Hz 连续更新。
Paper Eq. (18) 候选仍只作诊断；不使用 Experience Replay 或 Actor network。
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

# VS Code "Run Python File" may start outside the project runtime.  Re-enter
# through mjpython on macOS before importing MuJoCo when this file is executed.
if __name__ == "__main__":
    _BOOT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
    _BOOT_PROJECT_PYTHON = _BOOT_PROJECT_ROOT / ".venv/bin/python"
    _BOOT_MJ_PYTHON = _BOOT_PROJECT_ROOT / ".venv/bin/mjpython"
    _BOOT_HEADLESS = os.environ.get("MUJOCO_HEADLESS") == "1"
    _BOOT_VIEWER_READY = os.environ.get("YAC_DIRECT_MJPYTHON") == "1"
    if platform.system() == "Darwin" and not _BOOT_HEADLESS and not _BOOT_VIEWER_READY:
        if not _BOOT_PROJECT_PYTHON.is_file() or not _BOOT_MJ_PYTHON.is_file():
            raise FileNotFoundError("Project .venv Python/mjpython is missing")
        _boot_environment = os.environ.copy()
        _boot_environment["YAC_DIRECT_MJPYTHON"] = "1"
        os.execve(
            str(_BOOT_PROJECT_PYTHON),
            [
                str(_BOOT_PROJECT_PYTHON),
                str(_BOOT_MJ_PYTHON),
                str(Path(__file__).resolve()),
            ],
            _boot_environment,
        )
    if not _BOOT_VIEWER_READY and Path(sys.executable).resolve() != _BOOT_PROJECT_PYTHON.resolve():
        if not _BOOT_PROJECT_PYTHON.is_file():
            raise FileNotFoundError(f"Project Python is missing: {_BOOT_PROJECT_PYTHON}")
        os.execve(
            str(_BOOT_PROJECT_PYTHON),
            [str(_BOOT_PROJECT_PYTHON), str(Path(__file__).resolve())],
            os.environ.copy(),
        )

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline


# ============================================================
# [PLATFORM] Final bilateral Flexible Exo control-channel contract
#
# HipExo_L/KneeExo_L/HipExo_R/KneeExo_R are active.  The two ankle channels
# remain exactly zero because YAC controls only bilateral Hip/Knee errors.
# ============================================================
EXO_CHANNELS = (
    "HipExo_R",
    "HipExo_L",
    "KneeExo_R",
    "KneeExo_L",
    "Exo_R",
    "Exo_L",
)
ZERO_TORQUES = {name: 0.0 for name in EXO_CHANNELS}


# ============================================================
# [PLATFORM] MuJoCo simulation activation 意图代理
#
# 这六个量来自 data.act，是仿真肌肉 activation，不是实测肌电信号。
# ============================================================
INTENTION_MUSCLES = (
    "hamstrings_l",
    "bifemsh_l",
    "glutmax_l",
    "iliopsoas_l",
    "rectfem_l",
    "vasti_l",
)
INTENTION_MUSCLE_FAMILIES = (
    "hamstrings",
    "bifemsh",
    "glutmax",
    "iliopsoas",
    "rectfem",
    "vasti",
)


# ============================================================
# [NUMERICAL] Stage A/Stage B 集中数值参数
#
# EWMA_BETA 只是 Shadow 周期诊断的初始值，不是论文定义。
# CONSTRAINT_ACTIVE_THRESHOLD_NM 用于区分求解器数值零与真实关节限位力。
# ============================================================
EWMA_BETA = 0.2
HEEL_STRIKE_REFRACTORY_S = 0.6
MIN_STRIDE_PERIOD_S = 0.6
MAX_STRIDE_PERIOD_S = 2.0
CONSTRAINT_ACTIVE_THRESHOLD_NM = 1.0e-8
REFERENCE_PHASE_POINTS = 101
EXPECTED_REFERENCE_CYCLES = 15
EXPECTED_REFERENCE_MEDIAN_PERIOD_S = 1.150


# ============================================================
# [CROSS-PAPER CONTINUITY] YAC-MuJoCo V1 desired impedance 参数
#
# Md 与 Kmin/Kmax 不是最终 YAC 论文明确给出的参数；它们来自后续
# ROBIO 算法的明确设置。本轮沿用这一参数基础，是为了让后续算法比较
# 使用一致的 desired-inertia 与 stiffness envelope，不能称为
# “YAC paper parameter”。
# ============================================================
DESIRED_INERTIA_KGM2 = np.asarray((0.60, 0.25), dtype=float)
STIFFNESS_MIN_NM_PER_RAD = np.asarray((8.0, 8.0), dtype=float)
STIFFNESS_MAX_NM_PER_RAD = np.asarray((55.0, 45.0), dtype=float)


# ============================================================
# [NUMERICAL] MuJoCo V1 desired impedance damping 设计
#
# 最终 YAC 论文只要求 Dd 正定，并未给出这组数值。这里固定最大刚度处
# 的阻尼比 zeta=1.1，并由 Dd_i=2*zeta*sqrt(Md_i*Kmax_i) 现场计算，
# 以确保整个 Kr 范围具有正实阻尼分解；不硬编码四舍五入结果。
# ============================================================
DESIRED_DAMPING_RATIO_AT_KMAX = 1.1
DESIRED_DAMPING_NMS_PER_RAD = (
    2.0
    * DESIRED_DAMPING_RATIO_AT_KMAX
    * np.sqrt(DESIRED_INERTIA_KGM2 * STIFFNESS_MAX_NM_PER_RAD)
)
IMPEDANCE_UPDATE_DT_S = 0.01


# ============================================================
# [PAPER] transformed z subsystem 的 nominal damping family
#
# YAC 论文只要求 Ken 正定，没有给出当前 MuJoCo 对应的数值。Stage C
# 仅审计 lambda_z={0,2,5,10} s^-1，其中 0 只是无 z-feedback 的
# feedforward 诊断；2/5/10 才是正定候选，本轮不选择 FINAL_KEN。
# ============================================================
KEN_CANDIDATE_RATES_S_INV = (0.0, 2.0, 5.0, 10.0)


# ============================================================
# [NUMERICAL] Stage D active nominal baseline 与首次部署保护
#
# lambda_z=5 s^-1 是 Stage D MuJoCo nominal baseline 选择，不是 YAC
# 论文指定参数。Ken 仍由 lambda_z*M_bar 计算，不能硬编码 rounded matrix。
# engagement ramp 是 Stage D-R nominal-only 历史部署保护；Continuous/Full
# YAC 不使用 ramp 或软件 torque clipping。
# ============================================================
Z_DAMPING_RATE_S_INV = 5.0
ENGAGEMENT_RAMP_DURATION_S = 0.25
STAGE_D_TORQUE_LIMIT_NM = 20.0
STAGE_D_ACTIVE_SCALES = (0.25, 0.50, 1.00)
INTERACTION_SEMANTICS_MODE = "internal_m5_sensed_not_cancelled"
PAPER_EXACT_TAU_CANCELLATION = False
PAPER_CANDIDATE_RECORDED = True
FLEXIBLE_NOMINAL_SEMANTICS = "flexible_hri"
PAPER_EXACT_NOMINAL_SEMANTICS = "paper_exact"


# ============================================================
# [NUMERICAL] Stage E0 在冻结 D-R 5 s trajectory 上审计后固定
#
# z scale 是 D-R reference-valid samples 的 joint-wise p95 |z|，只用于
# Critic B6 basis。物理 z 仍是 rad/s。Q 使单关节 z 达到该 p95
# 时的 state cost 约为 1。E0 按 Eq. (42) 的单步增量 gate
# 得到 kappa_c=1.0。这些都不是 Paper 给定的 MuJoCo 数值。
# ============================================================
# Recalibrated from the stable Moderate60 bilateral-nominal 20 s run.  Samples
# are reference-valid, after the startup ramp, and inside complete strides.
# Per-side p95 |z| was Hip L/R=3.052541/3.041337 and
# Knee L/R=4.651972/5.189957 rad/s.  The larger bilateral value is used for
# each joint family so the mirrored B6 basis keeps one symmetric side scale.
CRITIC_Z_SCALE_RAD_S = np.asarray(
    (3.0525411155496047, 5.189957457811682), dtype=float
)
CRITIC_Q = np.diag(1.0 / CRITIC_Z_SCALE_RAD_S**2)

# ============================================================
# [NUMERICAL CALIBRATION] ADP action-cost scaling
#
# R 是当前 platform implementation 的 control-cost scaling，而不是
# YAC Paper 指定的 MuJoCo 常数。按 R_i ≈ 1/mu_scale_i^2：
# Hip 保持 5 N·m 尺度（R=0.04），Knee 采用 2 N·m 尺度
#（R=0.25）。这不是 torque clipping，也不是额外 controller gain；
# 它只在 J=∫(z^T Q z + mu^T R mu)dt 中提高 Knee residual torque
# 的 action cost，使 ADP 仍可做 Knee residual compensation，但不因
# Knee nominal dynamics/input-map 尺度较小而长期压过 nominal control。
# ============================================================
ADP_ACTION_SCALE_NM = np.asarray((5.0, 2.0), dtype=float)
ADP_R_SOURCE = "platform_numerical_calibration"
CRITIC_R = np.diag(1.0 / ADP_ACTION_SCALE_NM**2)
CRITIC_R_INV = np.diag(ADP_ACTION_SCALE_NM**2)
CRITIC_KAPPA = 1.0
CRITIC_WEIGHT_DIMENSION = 6
CRITIC_WEIGHT_NORM_LIMIT = 100.0
CONTINUOUS_ADP_MODE = "continuous_adp"
NOMINAL_MODE = "active_nominal_flexible_hri"
FULL_YAC_MODE = "full_yac"
EVENT_FORMULA = "YAC Eq.43-Eq.44"
EVENT_POLICY_SEMANTICS = (
    "eq28_sampled_z_current_a_current_critic_current_g"
)
EVENT_ETA_DISCRETIZATION = "exact_zoh_per_control_step"
EVENT_PARAMETER_SOURCE = "platform_numerical_calibration"
CRITIC_UPDATE_SEMANTICS = "continuous_100hz"
STRICT_FULL_TORQUE_ZOH = False
EVENT_ETA_INITIAL = 0.0
EVENT_REASON_NONE = 0
EVENT_REASON_INITIALIZATION = 1
EVENT_REASON_THRESHOLD = 2


# ============================================================
# [PAPER AMBIGUITY] Dynamic Event policy 语义
#
# 论文 Eq. (28) 显式使用 sampled z 与 current activation，而正文也出现
# 整个输入 ZOH 的描述。Flexible-HRI adapted YAC V1 选择前者：event 之间
# 只保持 z_sampled；a_current、W_current、g_current 仍逐控制周期更新。
# ============================================================


def _require_finite(value: Any, name: str) -> float:
    """将平台标量转为有限 float，并在输入异常时立即停止。"""
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"{name} is NaN or Inf")
    return result


def _rotation_2d(angle_rad: float) -> np.ndarray:
    """[NUMERICAL] 返回平面二维旋转矩阵。"""
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return np.asarray(((cosine, -sine), (sine, cosine)), dtype=float)


@dataclass(frozen=True)
class EventTriggerDecision:
    """Dynamic Event 当前步的纯数学判定结果。"""

    sampled_z_rad_s: np.ndarray
    delta_pretrigger_rad_s: np.ndarray
    delta_norm_sq_pretrigger: float
    z_norm_sq: float
    threshold_sq: float
    triggered: bool
    reason_code: int


def _event_trigger_decision(
    sampled_z_rad_s: np.ndarray | None,
    current_z_rad_s: np.ndarray,
    eta: float,
    *,
    xi: float,
    rho_delta: float,
    epsilon_delta: float,
) -> EventTriggerDecision:
    """[PAPER] 按 delta=z_sampled-z_current 评估 Eq.43--44。"""
    current_z = np.asarray(current_z_rad_s, dtype=float)
    eta_value = _require_finite(eta, "event eta")
    xi_value = _require_finite(xi, "event xi")
    rho_value = _require_finite(rho_delta, "event rho_delta")
    epsilon_value = _require_finite(epsilon_delta, "event epsilon_delta")
    if current_z.shape != (2,) or not np.isfinite(current_z).all():
        raise FloatingPointError("Dynamic Event current z must be finite shape (2,)")
    if (
        eta_value < 0.0
        or not 0.0 < xi_value < 1.0
        or rho_value <= 0.0
        or epsilon_value <= 0.0
    ):
        raise ValueError("Dynamic Event trigger parameters/state are invalid")

    z_norm_sq = float(current_z @ current_z)
    threshold_sq = float(
        (xi_value * z_norm_sq + eta_value + epsilon_value) / rho_value
    )
    if not math.isfinite(threshold_sq) or threshold_sq <= 0.0:
        raise FloatingPointError("Dynamic Event threshold must be finite and positive")

    if sampled_z_rad_s is None:
        delta_pretrigger = np.zeros(2, dtype=float)
        triggered = True
        reason_code = EVENT_REASON_INITIALIZATION
    else:
        sampled_z = np.asarray(sampled_z_rad_s, dtype=float)
        if sampled_z.shape != (2,) or not np.isfinite(sampled_z).all():
            raise FloatingPointError(
                "Dynamic Event sampled z must be finite shape (2,)"
            )
        # [PAPER] 必须保持 delta=z_sampled-z_current 的符号语义。
        delta_pretrigger = sampled_z - current_z
        triggered = bool(float(delta_pretrigger @ delta_pretrigger) >= threshold_sq)
        reason_code = EVENT_REASON_THRESHOLD if triggered else EVENT_REASON_NONE

    delta_norm_sq = float(delta_pretrigger @ delta_pretrigger)
    refreshed_sample = (
        current_z.copy()
        if triggered
        else np.asarray(sampled_z_rad_s, dtype=float).copy()
    )
    return EventTriggerDecision(
        sampled_z_rad_s=refreshed_sample,
        delta_pretrigger_rad_s=delta_pretrigger,
        delta_norm_sq_pretrigger=delta_norm_sq,
        z_norm_sq=z_norm_sq,
        threshold_sq=threshold_sq,
        triggered=triggered,
        reason_code=reason_code,
    )


def _event_eta_exact_zoh_step(
    eta: float,
    z_norm_sq: float,
    dt: float,
    *,
    lambda_eta: float,
    sigma_eta: float,
) -> tuple[float, float, float]:
    """[NUMERICAL] 对 eta_dot=-lambda*eta+sigma*||z||^2 精确 ZOH 离散。"""
    eta_value = _require_finite(eta, "event eta")
    z_norm_sq_value = _require_finite(z_norm_sq, "event z norm squared")
    dt_value = _require_finite(dt, "event control dt")
    lambda_value = _require_finite(lambda_eta, "event lambda_eta")
    sigma_value = _require_finite(sigma_eta, "event sigma_eta")
    if (
        eta_value < 0.0
        or z_norm_sq_value < 0.0
        or dt_value <= 0.0
        or lambda_value <= 0.0
        or sigma_value <= 0.0
    ):
        raise ValueError("Dynamic Event eta ZOH inputs must satisfy positivity")

    decay = math.exp(-lambda_value * dt_value)
    input_coefficient = (
        sigma_value / lambda_value
    ) * (-math.expm1(-lambda_value * dt_value))
    eta_next = decay * eta_value + input_coefficient * z_norm_sq_value
    if not all(math.isfinite(value) for value in (decay, input_coefficient, eta_next)):
        raise FloatingPointError("Dynamic Event eta ZOH update is non-finite")
    if eta_next < 0.0:
        raise FloatingPointError("Dynamic Event eta ZOH update became negative")
    return float(eta_next), float(decay), float(input_coefficient)


@dataclass(frozen=True)
class NominalDynamicsResult:
    """Stage A/C 单步 2DOF nominal dynamics 诊断量。"""

    mass: np.ndarray
    coriolis_matrix: np.ndarray
    coriolis_torque: np.ndarray
    gravity_torque: np.ndarray
    absolute_thigh_angle_rad: float


class CurrentExoNominalModel:
    """[PAPER] 2DOF 髋-膝 nominal model；[PLATFORM] 参数来自当前 XML。"""

    # ========================================================
    # [PLATFORM] 正式 Flexible XML 的精确质量与惯量数值
    #
    # 来源：
    # models/coupled/flexible_exo/healthy_exo_nature_flexible_v1.xml
    # 不使用旧 YAC.m 的 m1=2、m2=0.85 等失配参数。
    # 单位：mass kg，position m，inertia kg·m²。
    # ========================================================
    HIP_MODULE_MASS_KG = 0.7
    HIP_MODULE_I_ZZ = 0.00126
    THIGH_BAR_MASS_KG = 1.2
    THIGH_BAR_COM_FROM_HIP_M = np.asarray((0.0, -0.10), dtype=float)
    THIGH_BAR_I_ZZ = 0.00449

    KNEE_MODULE_MASS_KG = 0.15
    KNEE_MODULE_I_ZZ = 0.0001875
    SHANK_BAR_MASS_KG = 0.15
    SHANK_BAR_COM_FROM_KNEE_M = np.asarray((0.0, -0.20), dtype=float)
    SHANK_BAR_I_ZZ = 0.002045

    KNEE_FROM_HIP_M = np.asarray((0.00411, -0.395), dtype=float)
    JOINT_ARMATURE_KGM2 = 1.0e-5
    GRAVITY_MPS2 = 9.81

    def __init__(self) -> None:
        # [PLATFORM] 等效 link 1：髋模块 + 大腿杆。
        self.m1_kg = self.HIP_MODULE_MASS_KG + self.THIGH_BAR_MASS_KG
        self.link1_com_from_hip_m = (
            self.THIGH_BAR_MASS_KG * self.THIGH_BAR_COM_FROM_HIP_M
        ) / self.m1_kg
        self.lc1_m = float(np.linalg.norm(self.link1_com_from_hip_m))
        self.i1_com_kgm2 = float(
            self.HIP_MODULE_I_ZZ
            + self.HIP_MODULE_MASS_KG * self.lc1_m**2
            + self.THIGH_BAR_I_ZZ
            + self.THIGH_BAR_MASS_KG
            * float(
                np.dot(
                    self.THIGH_BAR_COM_FROM_HIP_M - self.link1_com_from_hip_m,
                    self.THIGH_BAR_COM_FROM_HIP_M - self.link1_com_from_hip_m,
                )
            )
        )

        # [PLATFORM] 等效 link 2：膝模块 + 小腿杆。
        self.m2_kg = self.KNEE_MODULE_MASS_KG + self.SHANK_BAR_MASS_KG
        self.link2_com_from_knee_m = (
            self.SHANK_BAR_MASS_KG * self.SHANK_BAR_COM_FROM_KNEE_M
        ) / self.m2_kg
        self.lc2_m = float(np.linalg.norm(self.link2_com_from_knee_m))
        self.i2_com_kgm2 = float(
            self.KNEE_MODULE_I_ZZ
            + self.KNEE_MODULE_MASS_KG * self.lc2_m**2
            + self.SHANK_BAR_I_ZZ
            + self.SHANK_BAR_MASS_KG
            * float(
                np.dot(
                    self.SHANK_BAR_COM_FROM_KNEE_M
                    - self.link2_com_from_knee_m,
                    self.SHANK_BAR_COM_FROM_KNEE_M
                    - self.link2_com_from_knee_m,
                )
            )
        )
        self.l1_m = float(np.linalg.norm(self.KNEE_FROM_HIP_M))

    def evaluate(
        self,
        q_rad: np.ndarray,
        qdot_rad_s: np.ndarray,
        root_pitch_rad: float,
    ) -> NominalDynamicsResult:
        """计算当前 XML 参数下的 fixed-base-style 2DOF M/C/G。"""
        q = np.asarray(q_rad, dtype=float).reshape(2)
        qdot = np.asarray(qdot_rad_s, dtype=float).reshape(2)
        if not np.isfinite(np.concatenate((q, qdot, [root_pitch_rad]))).all():
            raise FloatingPointError("Nominal dynamics input contains NaN or Inf")
        q1, q2 = float(q[0]), float(q[1])
        q1dot, q2dot = float(qdot[0]), float(qdot[1])

        # ====================================================
        # [PAPER] 2-link 质量矩阵 M(q)
        #
        # [PLATFORM] 保留 XML 中髋-膝中心 4.11 mm 的前后偏置，
        # 用几何向量而非 Prompt 四舍五入参数精确重建 Mcc。
        # 单位：kg·m²。joint armature 只加在对应对角元素。
        # ====================================================
        rotated_link2_com = _rotation_2d(q2) @ self.link2_com_from_knee_m
        cross_geometry = float(np.dot(self.KNEE_FROM_HIP_M, rotated_link2_com))
        link1_about_hip = self.i1_com_kgm2 + self.m1_kg * self.lc1_m**2
        link2_about_knee = self.i2_com_kgm2 + self.m2_kg * self.lc2_m**2
        m11 = (
            link1_about_hip
            + self.i2_com_kgm2
            + self.m2_kg
            * float(
                np.dot(
                    self.KNEE_FROM_HIP_M + rotated_link2_com,
                    self.KNEE_FROM_HIP_M + rotated_link2_com,
                )
            )
            + self.JOINT_ARMATURE_KGM2
        )
        m12 = link2_about_knee + self.m2_kg * cross_geometry
        m22 = link2_about_knee + self.JOINT_ARMATURE_KGM2
        mass = np.asarray(((m11, m12), (m12, m22)), dtype=float)

        # ====================================================
        # [PAPER] C(q,qdot)qdot
        #
        # b(q2)=m2*r_hip_knee^T*R(q2)*r_knee_com，b' 生成标准
        # 2-link Christoffel 项。单位：N·m。
        # ====================================================
        rotation_derivative = np.asarray(
            ((-math.sin(q2), -math.cos(q2)),
             (math.cos(q2), -math.sin(q2))),
            dtype=float,
        )
        b_prime = self.m2_kg * float(
            np.dot(
                self.KNEE_FROM_HIP_M,
                rotation_derivative @ self.link2_com_from_knee_m,
            )
        )
        coriolis = np.asarray(
            (
                (b_prime * q2dot, b_prime * (q1dot + q2dot)),
                (-b_prime * q1dot, 0.0),
            ),
            dtype=float,
        )
        coriolis_torque = coriolis @ qdot

        # ====================================================
        # [PAPER] 2-link 重力项 G
        #
        # [PLATFORM] exo_hip_l 是相对 pelvis 角，所以世界重力方向
        # 必须同时使用 exo_root_pitch。当前平面约定中，
        # absolute_thigh_angle = root_pitch + hip_angle，零角对应连杆向下。
        # Root translation/acceleration 不进入 Stage A nominal model。
        # 单位：N·m。
        # ====================================================
        absolute_thigh_angle = float(root_pitch_rad + q1)
        absolute_shank_angle = float(absolute_thigh_angle + q2)
        link1_com_world = (
            _rotation_2d(absolute_thigh_angle) @ self.link1_com_from_hip_m
        )
        knee_world = _rotation_2d(absolute_thigh_angle) @ self.KNEE_FROM_HIP_M
        link2_com_world = (
            _rotation_2d(absolute_shank_angle) @ self.link2_com_from_knee_m
        )
        g1 = self.GRAVITY_MPS2 * (
            self.m1_kg * float(link1_com_world[0])
            + self.m2_kg * float(knee_world[0] + link2_com_world[0])
        )
        g2 = self.GRAVITY_MPS2 * self.m2_kg * float(link2_com_world[0])
        gravity_torque = np.asarray((g1, g2), dtype=float)

        result_values = np.concatenate((mass.reshape(-1), coriolis_torque, gravity_torque))
        if not np.isfinite(result_values).all():
            raise FloatingPointError("Nominal dynamics output contains NaN or Inf")
        return NominalDynamicsResult(
            mass=mass,
            coriolis_matrix=coriolis,
            coriolis_torque=coriolis_torque,
            gravity_torque=gravity_torque,
            absolute_thigh_angle_rad=absolute_thigh_angle,
        )


class HealthyGaitReference:
    """[PLATFORM] 从唯一指定的 Healthy Flexible Zero Result 建立周期参考。"""

    def __init__(self, result_directory: str | Path, side: str = "l") -> None:
        normalized_side = str(side).strip().lower()
        if normalized_side not in ("l", "r"):
            raise ValueError("Healthy gait reference side must be 'l' or 'r'")
        self.side = normalized_side
        self.result_directory = Path(result_directory).expanduser().resolve()
        if not self.result_directory.is_dir():
            raise FileNotFoundError(
                f"Fixed Healthy reference result does not exist: {self.result_directory}"
            )
        control_path = self.result_directory / "timeseries.csv"
        physics_path = self.result_directory / "physics_timeseries_1khz.csv"
        for path in (control_path, physics_path):
            if not path.is_file():
                raise FileNotFoundError(f"Healthy reference artifact is missing: {path}")

        control_rows = self._read_rows(control_path)
        physics_rows = self._read_rows(physics_path)
        strike_name = "left" if self.side == "l" else "right"
        side_strikes = np.asarray(
            [
                _require_finite(row["time"], "Healthy heel-strike time")
                for row in control_rows
                if strike_name in row["heel_strike_side"].strip().lower()
            ],
            dtype=float,
        )
        complete_cycles = [
            (float(side_strikes[index]), float(side_strikes[index + 1]))
            for index in range(max(0, side_strikes.size - 1))
        ]
        period_valid = [
            cycle
            for cycle in complete_cycles
            if MIN_STRIDE_PERIOD_S <= cycle[1] - cycle[0] <= MAX_STRIDE_PERIOD_S
        ]
        if len(period_valid) <= 2:
            raise RuntimeError(
                f"Healthy result has too few complete {strike_name} gait cycles"
            )
        after_startup = period_valid[2:]
        self.median_period_s = float(
            np.median([end - start for start, end in after_startup])
        )
        retained = [
            cycle
            for cycle in after_startup
            if abs((cycle[1] - cycle[0]) - self.median_period_s)
            <= 0.20 * self.median_period_s
        ]
        self.retained_cycles = tuple(retained)
        self.cycle_count = len(retained)
        if self.cycle_count != EXPECTED_REFERENCE_CYCLES:
            raise RuntimeError(
                "Healthy reference cycle count differs from the audited value: "
                f"expected {EXPECTED_REFERENCE_CYCLES}, got {self.cycle_count}"
            )
        if not math.isclose(
            self.median_period_s,
            EXPECTED_REFERENCE_MEDIAN_PERIOD_S,
            rel_tol=0.0,
            abs_tol=0.02,
        ):
            raise RuntimeError(
                "Healthy reference median period differs materially from the audit: "
                f"{self.median_period_s:.6f} s"
            )

        physics_time = np.asarray(
            [_require_finite(row["time"], "Healthy physics time") for row in physics_rows]
        )
        source_q = np.asarray(
            [
                (
                    _require_finite(
                        row[f"exo_hip_{self.side}_angle"], "Healthy Hip q"
                    ),
                    _require_finite(
                        row[f"exo_knee_{self.side}_angle"], "Healthy Knee q"
                    ),
                )
                for row in physics_rows
            ],
            dtype=float,
        )
        source_qdot = np.asarray(
            [
                (
                    _require_finite(
                        row[f"exo_hip_{self.side}_velocity"], "Healthy Hip qdot"
                    ),
                    _require_finite(
                        row[f"exo_knee_{self.side}_velocity"], "Healthy Knee qdot"
                    ),
                )
                for row in physics_rows
            ],
            dtype=float,
        )
        phase = np.linspace(0.0, 1.0, REFERENCE_PHASE_POINTS)
        q_cycles: list[np.ndarray] = []
        qdot_cycles: list[np.ndarray] = []
        for start, end in retained:
            target_time = start + phase * (end - start)
            q_cycles.append(
                np.column_stack(
                    [
                        np.interp(target_time, physics_time, source_q[:, joint])
                        for joint in range(2)
                    ]
                )
            )
            qdot_cycles.append(
                np.column_stack(
                    [
                        np.interp(target_time, physics_time, source_qdot[:, joint])
                        for joint in range(2)
                    ]
                )
            )
        profile = np.mean(np.stack(q_cycles), axis=0)
        self.measured_qdot_profile_rad_s = np.mean(np.stack(qdot_cycles), axis=0)
        self.raw_endpoint_gap_rad = profile[-1] - profile[0]

        # ====================================================
        # [NUMERICAL] Periodic CubicSpline 周期边界处理
        #
        # 论文没有规定样条边界。为使 q_d、dq_d/dphi 和
        # d²q_d/dphi² 在 0/100% gait boundary 一致，先将 mean profile
        # 的首尾取平均，再建立 bc_type="periodic" 的样条。
        # ====================================================
        reconciled = profile.copy()
        endpoint_average = 0.5 * (reconciled[0] + reconciled[-1])
        reconciled[0] = endpoint_average
        reconciled[-1] = endpoint_average
        self.spline = CubicSpline(phase, reconciled, axis=0, bc_type="periodic")
        self.periodic_position_gap_rad = np.asarray(
            self.spline(1.0) - self.spline(0.0), dtype=float
        )
        self.periodic_first_derivative_gap = np.asarray(
            self.spline(1.0, 1) - self.spline(0.0, 1), dtype=float
        )
        self.periodic_second_derivative_gap = np.asarray(
            self.spline(1.0, 2) - self.spline(0.0, 2), dtype=float
        )
        closure = np.concatenate(
            (
                self.periodic_position_gap_rad,
                self.periodic_first_derivative_gap,
                self.periodic_second_derivative_gap,
            )
        )
        if not np.isfinite(closure).all() or np.max(np.abs(closure)) > 1.0e-10:
            raise RuntimeError("Periodic Healthy reference failed boundary continuity")

    @staticmethod
    def _read_rows(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def evaluate(self, phase: float, period_s: float) -> tuple[np.ndarray, ...]:
        """返回 q_d、qdot_d、qddot_d；phase 无量纲，周期单位为 s。"""
        phase_value = float(np.clip(_require_finite(phase, "gait phase"), 0.0, 1.0))
        period = _require_finite(period_s, "stride period")
        if period <= 0.0:
            raise ValueError("Stride period must be positive")
        q_d = np.asarray(self.spline(phase_value), dtype=float)
        qdot_d = np.asarray(self.spline(phase_value, 1), dtype=float) / period
        # [NUMERICAL] stride 内 phi_dot=1/T_hat，phi_ddot=0，不制造冲击项。
        qddot_d = np.asarray(self.spline(phase_value, 2), dtype=float) / period**2
        if not np.isfinite(np.concatenate((q_d, qdot_d, qddot_d))).all():
            raise FloatingPointError("Healthy reference evaluation contains NaN or Inf")
        return q_d, qdot_d, qddot_d


def _critic_basis_b6(
    z_rad_s: np.ndarray, activation: float
) -> tuple[np.ndarray, np.ndarray]:
    """[NUMERICAL] 返回 B6 features 与对物理 z 的 analytic Jacobian。"""
    z = np.asarray(z_rad_s, dtype=float).reshape(2)
    a_value = _require_finite(activation, "Critic activation")
    if not np.isfinite(z).all():
        raise FloatingPointError("Critic z contains NaN or Inf")
    x1, x2 = z / CRITIC_Z_SCALE_RAD_S
    features = np.asarray(
        (
            x1**2,
            x1 * x2,
            x2**2,
            a_value * x1**2,
            a_value * x1 * x2,
            a_value * x2**2,
        ),
        dtype=float,
    )
    # ========================================================
    # [PAPER AMBIGUITY] Paper 要求 phi_c(z,a)，但未给 basis。
    # [NUMERICAL] B6 使 V(0,a)=0 且 grad_z V(0,a)=0，避免
    # z=0 时凭空生成 residual torque。Jacobian 对物理 z
    # 求导，因此显式包含 1/s_hip 与 1/s_knee。
    # shape=(2,6)，与 Paper Eq. (32) 的 grad_z phi_c 一致。
    # ========================================================
    jacobian = np.asarray(
        (
            (
                2.0 * x1 / CRITIC_Z_SCALE_RAD_S[0],
                x2 / CRITIC_Z_SCALE_RAD_S[0],
                0.0,
                2.0 * a_value * x1 / CRITIC_Z_SCALE_RAD_S[0],
                a_value * x2 / CRITIC_Z_SCALE_RAD_S[0],
                0.0,
            ),
            (
                0.0,
                x1 / CRITIC_Z_SCALE_RAD_S[1],
                2.0 * x2 / CRITIC_Z_SCALE_RAD_S[1],
                0.0,
                a_value * x1 / CRITIC_Z_SCALE_RAD_S[1],
                2.0 * a_value * x2 / CRITIC_Z_SCALE_RAD_S[1],
            ),
        ),
        dtype=float,
    )
    if not np.isfinite(np.concatenate((features, jacobian.reshape(-1)))).all():
        raise FloatingPointError("Critic B6 basis/Jacobian contains NaN or Inf")
    return features, jacobian


class _LegacyUnilateralYACController:
    """Flexible-HRI adapted YAC nominal、continuous ADP 与 Full YAC。"""

    def __init__(
        self,
        *,
        reference_result: str | Path,
        mode: str = NOMINAL_MODE,
        active_scale: float = 1.0,
        adp_scale: float = 1.0,
        event_trigger: bool = False,
        event_lambda_eta: float | None = None,
        event_sigma_eta: float | None = None,
        event_xi: float | None = None,
        event_rho_delta: float | None = None,
        event_epsilon_delta: float | None = None,
        nominal_torque_semantics: str = FLEXIBLE_NOMINAL_SEMANTICS,
        ewma_beta: float = EWMA_BETA,
        constraint_active_threshold_nm: float = CONSTRAINT_ACTIVE_THRESHOLD_NM,
    ) -> None:
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in (
            NOMINAL_MODE,
            CONTINUOUS_ADP_MODE,
            FULL_YAC_MODE,
        ):
            raise ValueError(
                "YAC currently permits mode='active_nominal_flexible_hri' "
                "mode='continuous_adp' or mode='full_yac'"
            )
        self.mode = normalized_mode
        normalized_nominal_semantics = str(nominal_torque_semantics).strip().lower()
        if normalized_nominal_semantics not in (
            FLEXIBLE_NOMINAL_SEMANTICS,
            PAPER_EXACT_NOMINAL_SEMANTICS,
        ):
            raise ValueError(
                "nominal_torque_semantics must be 'flexible_hri' or 'paper_exact'"
            )
        self.nominal_torque_semantics = normalized_nominal_semantics
        self.active_scale = _require_finite(active_scale, "nominal active scale")
        if not any(
            math.isclose(self.active_scale, candidate, abs_tol=1.0e-12)
            for candidate in STAGE_D_ACTIVE_SCALES
        ):
            raise ValueError("active_scale must be 0.25, 0.50 or 1.00")
        if self.mode in (CONTINUOUS_ADP_MODE, FULL_YAC_MODE) and not math.isclose(
            self.active_scale, 1.0, abs_tol=1.0e-12
        ):
            raise ValueError("ADP modes freeze nominal active_scale at 1.0")
        self.adp_scale = _require_finite(adp_scale, "ADP scale")
        if not math.isclose(self.adp_scale, 1.0, abs_tol=1.0e-12):
            raise ValueError("Flexible-HRI adapted YAC freezes adp_scale at 1.0")
        self.event_trigger_enabled = bool(event_trigger)
        if self.mode == FULL_YAC_MODE and not self.event_trigger_enabled:
            raise ValueError("Full YAC requires event_trigger=True")
        if self.mode != FULL_YAC_MODE and self.event_trigger_enabled:
            raise ValueError("Dynamic Event Trigger is active only in full_yac mode")
        event_values = (
            event_lambda_eta,
            event_sigma_eta,
            event_xi,
            event_rho_delta,
            event_epsilon_delta,
        )
        if self.event_trigger_enabled:
            if any(value is None for value in event_values):
                raise ValueError("Full YAC requires all Dynamic Event parameters")
            self.event_lambda_eta = _require_finite(
                event_lambda_eta, "event lambda_eta"
            )
            self.event_sigma_eta = _require_finite(
                event_sigma_eta, "event sigma_eta"
            )
            self.event_xi = _require_finite(event_xi, "event xi")
            self.event_rho_delta = _require_finite(
                event_rho_delta, "event rho_delta"
            )
            self.event_epsilon_delta = _require_finite(
                event_epsilon_delta, "event epsilon_delta"
            )
            if (
                self.event_lambda_eta <= 0.0
                or self.event_sigma_eta <= 0.0
                or not 0.0 < self.event_xi < 1.0
                or self.event_rho_delta <= 0.0
                or self.event_epsilon_delta <= 0.0
            ):
                raise ValueError("Dynamic Event parameters must have valid signs")
        else:
            self.event_lambda_eta = 0.0
            self.event_sigma_eta = 0.0
            self.event_xi = 0.0
            self.event_rho_delta = 1.0
            self.event_epsilon_delta = 0.0
        self.reference = HealthyGaitReference(reference_result)
        self.nominal_model = CurrentExoNominalModel()
        self.m_bar_kgm2, self.ken_candidates_nms_per_rad = (
            self._build_ken_candidates()
        )
        self.ken_active_nms_per_rad = self.ken_candidates_nms_per_rad[
            Z_DAMPING_RATE_S_INV
        ]
        self.ewma_beta = _require_finite(ewma_beta, "EWMA beta")
        if not 0.0 < self.ewma_beta <= 1.0:
            raise ValueError("EWMA beta must be within (0, 1]")
        self.constraint_active_threshold_nm = _require_finite(
            constraint_active_threshold_nm, "constraint threshold"
        )
        if self.constraint_active_threshold_nm < 0.0:
            raise ValueError("Constraint threshold must be non-negative")
        self.last_diagnostics: dict[str, Any] = {}
        self.reset()

    def _build_ken_candidates(
        self,
    ) -> tuple[np.ndarray, dict[float, np.ndarray]]:
        """以 Healthy reference 的平均 nominal inertia 建立 Ken family。"""
        phase_grid = np.linspace(0.0, 1.0, REFERENCE_PHASE_POINTS)
        masses = []
        for phase in phase_grid:
            q_ref = np.asarray(self.reference.spline(float(phase)), dtype=float)
            masses.append(
                self.nominal_model.evaluate(
                    q_ref, np.zeros(2, dtype=float), 0.0
                ).mass
            )
        m_bar = np.mean(np.stack(masses), axis=0)
        m_bar_eigenvalues = np.linalg.eigvalsh(m_bar)
        if (
            not np.isfinite(
                np.concatenate((m_bar.reshape(-1), m_bar_eigenvalues))
            ).all()
            or not np.allclose(m_bar, m_bar.T, rtol=0.0, atol=1.0e-14)
            or np.any(m_bar_eigenvalues <= 0.0)
        ):
            raise ValueError("Healthy-reference M_bar must be symmetric positive definite")

        # ====================================================
        # [NUMERICAL] Ken=lambda_z*M_bar，而不是 alpha*Dd
        #
        # Dd 属于 desired-impedance Delta-q subsystem；Ken 属于 transformed
        # z subsystem 的 nominal damping。两者控制层次不同，不能为了方便
        # 令 Ken=alpha*Dd。M_bar 保留完整 SPD 非对角结构，不强制对角化。
        # ====================================================
        candidates = {
            rate: rate * m_bar for rate in KEN_CANDIDATE_RATES_S_INV
        }
        for rate, matrix in candidates.items():
            eigenvalues = np.linalg.eigvalsh(matrix)
            valid = bool(
                np.isfinite(
                    np.concatenate((matrix.reshape(-1), eigenvalues))
                ).all()
                and np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-14)
                and (
                    np.all(eigenvalues > 0.0)
                    if rate > 0.0
                    else np.all(eigenvalues == 0.0)
                )
            )
            if not valid:
                raise ValueError(f"Invalid Ken candidate for lambda_z={rate:g}")
        return m_bar, candidates

    def reset(self) -> None:
        """[NUMERICAL] 清空 phase/impedance/Critic 的全部在线状态。"""
        self._previous_left_heel: bool | None = None
        self._last_left_heel_strike_time_s: float | None = None
        self._stride_period_hat_s = float(self.reference.median_period_s)
        self._reference_valid = False
        self._accepted_left_heel_strikes = 0
        self._lambda_s_inv: np.ndarray | None = None
        self._t1_rad_s: np.ndarray | None = None
        self._engagement_start_time_s: float | None = None
        self._consecutive_numerical_failures = 0
        # [PAPER] Wc(0)=0 保证 grad V=0 且 mu=0。不使用随机初值、
        # exploration noise、projection 或 weight clipping。
        self._critic_weight = np.zeros(CRITIC_WEIGHT_DIMENSION, dtype=float)
        # [PLATFORM TIMING] 下一次 observation 的 qacc 对应本次真实施加
        # 的 residual。reset 后首次 compute 没有 previous action。
        self._previous_applied_mu_nm: np.ndarray | None = None
        self._consecutive_critic_failures = 0
        self._event_active = False
        self._event_sampled_z_rad_s: np.ndarray | None = None
        self._event_eta = EVENT_ETA_INITIAL
        self._event_count = 0
        self._event_initialization_count = 0
        self._event_threshold_count = 0
        self._event_policy_sample_count = 0
        self._event_age_steps = 0
        self._last_event_time_s: float | None = None
        self._last_event_valid_step: int | None = None
        self.last_diagnostics = {}

    def compute(self, observation: dict[str, object], dt: float) -> dict[str, float]:
        """运行 YAC core；任何非有限计算立即上抛并停止仿真。"""
        return self._compute_active_nominal(observation, dt)

    def _compute_active_nominal(
        self, observation: dict[str, object], dt: float
    ) -> dict[str, float]:
        """计算 Stage D-R nominal u0、一次性 ramp 与 component-wise limit。"""
        control_dt = _require_finite(dt, "control dt")
        if control_dt <= 0.0:
            raise ValueError("Control dt must be positive")
        if not math.isclose(
            control_dt, IMPEDANCE_UPDATE_DT_S, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(
                "YAC Stage C impedance transformation requires dt=0.01 s"
            )
        time_s = _require_finite(observation["time"], "observation time")
        contact = observation.get("contact")
        muscles = observation.get("muscles")
        exo = observation.get("exoskeleton")
        if not isinstance(contact, dict):
            raise KeyError("observation['contact'] is required")
        if not isinstance(muscles, dict) or not isinstance(muscles.get("activation"), dict):
            raise KeyError("observation['muscles']['activation'] is required")
        if not isinstance(exo, dict):
            raise KeyError("observation['exoskeleton'] is required")

        left_heel = bool(contact["left_heel"])
        left_heel_strike = False
        if (
            self._previous_left_heel is not None
            and not self._previous_left_heel
            and left_heel
        ):
            elapsed = (
                math.inf
                if self._last_left_heel_strike_time_s is None
                else time_s - self._last_left_heel_strike_time_s
            )
            if elapsed >= HEEL_STRIKE_REFRACTORY_S:
                if self._last_left_heel_strike_time_s is None:
                    left_heel_strike = True
                    self._last_left_heel_strike_time_s = time_s
                    self._reference_valid = True
                    self._accepted_left_heel_strikes += 1
                elif MIN_STRIDE_PERIOD_S <= elapsed <= MAX_STRIDE_PERIOD_S:
                    left_heel_strike = True
                    self._stride_period_hat_s = (
                        (1.0 - self.ewma_beta) * self._stride_period_hat_s
                        + self.ewma_beta * elapsed
                    )
                    self._last_left_heel_strike_time_s = time_s
                    self._accepted_left_heel_strikes += 1
        self._previous_left_heel = left_heel

        phase: float | str = ""
        q_d: np.ndarray | None = None
        qdot_d: np.ndarray | None = None
        qddot_d: np.ndarray | None = None
        if self._reference_valid and self._last_left_heel_strike_time_s is not None:
            phase = float(
                np.clip(
                    (time_s - self._last_left_heel_strike_time_s)
                    / self._stride_period_hat_s,
                    0.0,
                    1.0,
                )
            )
            q_d, qdot_d, qddot_d = self.reference.evaluate(
                phase, self._stride_period_hat_s
            )

        hip = exo.get("hip_l")
        knee = exo.get("knee_l")
        coupling = exo.get("coupling_qfrc_nm")
        constraint = exo.get("constraint_qfrc_nm")
        if not all(isinstance(item, dict) for item in (hip, knee, coupling, constraint)):
            raise KeyError("Exoskeleton Hip/Knee, coupling and constraint observations are required")
        q = np.asarray(
            (
                _require_finite(hip["angle_rad"], "Exo Hip q"),
                _require_finite(knee["angle_rad"], "Exo Knee q"),
            )
        )
        qdot = np.asarray(
            (
                _require_finite(hip["velocity_rad_s"], "Exo Hip qdot"),
                _require_finite(knee["velocity_rad_s"], "Exo Knee qdot"),
            )
        )
        qacc = np.asarray(
            (
                _require_finite(
                    hip["acceleration_rad_s2"], "Exo Hip qacc"
                ),
                _require_finite(
                    knee["acceleration_rad_s2"], "Exo Knee qacc"
                ),
            )
        )
        root_pitch = _require_finite(exo["root_pitch_rad"], "Exo root pitch")
        # ====================================================
        # [PLATFORM] Flexible-HRI interaction sensing
        #
        # 这是 M5 作用在 Exo 左髋/左膝广义坐标上的真实 interaction
        # torque，保持 Stage A 已审计的 direct sign，不取反。它继续进入
        # t1 desired-impedance transformation；不能因为取消 motor-side
        # cancellation 而删除 interaction sensing。
        # ====================================================
        tau_interaction_sense = np.asarray(
            (
                _require_finite(coupling["hip_l"], "Hip direct interaction torque"),
                _require_finite(coupling["knee_l"], "Knee direct interaction torque"),
            )
        )
        constraint_torque = np.asarray(
            (
                _require_finite(constraint["hip_l"], "Hip constraint torque"),
                _require_finite(constraint["knee_l"], "Knee constraint torque"),
            )
        )

        activation = muscles["activation"]
        single_activation = {
            name: _require_finite(activation[name], f"activation {name}")
            for name in INTENTION_MUSCLES
        }
        activation_proxy = float(
            math.sqrt(
                sum(value * value for value in single_activation.values())
                / len(INTENTION_MUSCLES)
            )
        )

        # ====================================================
        # [PAPER] activation-dependent stiffness scheduling
        #
        # a_sched 只对原始 activation RMS 做 [0,1] 截断；不做 p5/p95
        # normalization，不把 0.5 强行放大到 1，也不人为扩大患者与健康
        # 条件的差异。Kr 按论文线性 stiffness law 逐关节计算。
        # ====================================================
        intention_mode = getattr(self, "intention_mode", "live_activation")
        if intention_mode == "constant_one":
            # [EXPERIMENTAL ABLATION] Paper-equivalent constant activation.
            # The measured simulation activation proxy is still retained for
            # diagnostics; only the scheduling input to Kr and phi(z,a) is 1.
            a_sched = 1.0
        else:
            a_sched = float(np.clip(activation_proxy, 0.0, 1.0))
        stiffness = (
            STIFFNESS_MIN_NM_PER_RAD
            + a_sched
            * (STIFFNESS_MAX_NM_PER_RAD - STIFFNESS_MIN_NM_PER_RAD)
        )
        stiffness_valid = bool(
            np.isfinite(stiffness).all()
            and np.all(stiffness >= STIFFNESS_MIN_NM_PER_RAD)
            and np.all(stiffness <= STIFFNESS_MAX_NM_PER_RAD)
            and np.all(stiffness > 0.0)
        )

        # ====================================================
        # [PAPER] Stage B 使用跟踪偏差 Delta q / Delta qdot
        #
        # 这是 Patient-to-Healthy reference deviation，不是 YAC tracking
        # performance，因为 Stage B 仍没有施加任何 YAC torque。
        # ====================================================
        delta_q: np.ndarray | None = None
        delta_qdot: np.ndarray | None = None
        if q_d is not None and qdot_d is not None:
            delta_q = q - q_d
            delta_qdot = qdot - qdot_d

        # ====================================================
        # [DERIVATION] 量纲一致的 YAC 阻抗变换
        #
        # 最终论文 Eq. (13) 若按物理单位字面代入，会把 rad/s² 的代数项
        # 加到 qdot_r(rad/s) 上。作者推导稿将 t1 定义为一阶动态状态，
        # 并同时引入 Lambda/Gamma 动力学，从而保证 t1 单位为 rad/s，
        # 可以与 qdot_d 和 Lambda*Deltaq 相加。
        #
        # 这里不是新控制算法，而是论文 Eq. (13)-(15) 的量纲一致数值实现。
        # 本步先用当前状态解析计算全部量，再显式 Euler 积分到下一控制步，
        # 因而不使用未来 observation，也不对 qdot_r 做 finite difference。
        # ====================================================
        damping_over_inertia = (
            DESIRED_DAMPING_NMS_PER_RAD / DESIRED_INERTIA_KGM2
        )
        discriminant = (
            damping_over_inertia**2
            - 4.0 * stiffness / DESIRED_INERTIA_KGM2
        )
        impedance_transform_valid = False
        lambda_s_inv: np.ndarray | None = None
        gamma_s_inv: np.ndarray | None = None
        dot_lambda_s_inv2: np.ndarray | None = None
        t1_rad_s: np.ndarray | None = None
        dot_t1_rad_s2: np.ndarray | None = None
        qdot_r_rad_s: np.ndarray | None = None
        qddot_r_rad_s2: np.ndarray | None = None
        z_rad_s: np.ndarray | None = None
        z_identity_error = ""
        r_imp_rad_s2: np.ndarray | None = None
        r_transform_rad_s2: np.ndarray | None = None
        transform_mismatch: np.ndarray | None = None
        qddot_r_component_ref: np.ndarray | None = None
        qddot_r_component_dot_lambda: np.ndarray | None = None
        qddot_r_component_lambda_delta_qdot: np.ndarray | None = None
        qddot_r_component_dot_t1: np.ndarray | None = None

        reference_arrays_available = all(
            item is not None
            for item in (q_d, qdot_d, qddot_d, delta_q, delta_qdot)
        )
        if self._reference_valid and reference_arrays_available:
            # [DERIVATION] 第一次参考有效时，使用二次方程的较小正实根。
            if self._lambda_s_inv is None or self._t1_rad_s is None:
                initialization_valid = bool(
                    stiffness_valid
                    and np.isfinite(discriminant).all()
                    and np.all(discriminant > 0.0)
                )
                if initialization_valid:
                    initial_lambda = 0.5 * (
                        damping_over_inertia - np.sqrt(discriminant)
                    )
                    initial_gamma = damping_over_inertia - initial_lambda
                    initialization_valid = bool(
                        np.isfinite(
                            np.concatenate((initial_lambda, initial_gamma))
                        ).all()
                        and np.all(initial_lambda > 0.0)
                        and np.all(initial_gamma > 0.0)
                    )
                    if initialization_valid:
                        self._lambda_s_inv = initial_lambda
                        self._t1_rad_s = np.zeros(2, dtype=float)

            if self._lambda_s_inv is not None and self._t1_rad_s is not None:
                lambda_s_inv = self._lambda_s_inv.copy()
                t1_rad_s = self._t1_rad_s.copy()
                gamma_s_inv = damping_over_inertia - lambda_s_inv
                dot_lambda_s_inv2 = (
                    stiffness / DESIRED_INERTIA_KGM2
                    - gamma_s_inv * lambda_s_inv
                )
                dot_t1_rad_s2 = (
                    tau_interaction_sense / DESIRED_INERTIA_KGM2
                    - gamma_s_inv * t1_rad_s
                )

                # [NUMERICAL] 只做 finite/positive guard；绝不 clip Lambda。
                transform_state = np.concatenate(
                    (
                        discriminant,
                        lambda_s_inv,
                        gamma_s_inv,
                        dot_lambda_s_inv2,
                        t1_rad_s,
                        dot_t1_rad_s2,
                    )
                )
                state_valid = bool(
                    stiffness_valid
                    and np.isfinite(transform_state).all()
                    and np.all(discriminant > 0.0)
                    and np.all(lambda_s_inv > 0.0)
                    and np.all(gamma_s_inv > 0.0)
                )
                if state_valid:
                    assert qdot_d is not None
                    assert qddot_d is not None
                    assert delta_q is not None
                    assert delta_qdot is not None
                    qdot_r_rad_s = (
                        qdot_d - lambda_s_inv * delta_q + t1_rad_s
                    )
                    z_rad_s = qdot - qdot_r_rad_s
                    z_check = (
                        delta_qdot
                        + lambda_s_inv * delta_q
                        - t1_rad_s
                    )
                    z_identity_error = float(
                        np.max(np.abs(z_rad_s - z_check))
                    )

                    # [DERIVATION] qddot_r 直接使用解析导数，不做差分或滤波。
                    qddot_r_component_ref = qddot_d.copy()
                    qddot_r_component_dot_lambda = (
                        -dot_lambda_s_inv2 * delta_q
                    )
                    qddot_r_component_lambda_delta_qdot = (
                        -lambda_s_inv * delta_qdot
                    )
                    qddot_r_component_dot_t1 = dot_t1_rad_s2.copy()
                    qddot_r_rad_s2 = (
                        qddot_r_component_ref
                        + qddot_r_component_dot_lambda
                        + qddot_r_component_lambda_delta_qdot
                        + qddot_r_component_dot_t1
                    )

                    # [DERIVATION] 两条 residual 路径必须表示同一 desired impedance。
                    delta_qddot = qacc - qddot_d
                    r_imp_rad_s2 = (
                        delta_qddot
                        + damping_over_inertia * delta_qdot
                        + (stiffness / DESIRED_INERTIA_KGM2) * delta_q
                        - tau_interaction_sense / DESIRED_INERTIA_KGM2
                    )
                    z_dot_rad_s2 = qacc - qddot_r_rad_s2
                    r_transform_rad_s2 = (
                        z_dot_rad_s2 + gamma_s_inv * z_rad_s
                    )
                    transform_mismatch = r_imp_rad_s2 - r_transform_rad_s2

                    derived_values = np.concatenate(
                        (
                            qdot_r_rad_s,
                            qddot_r_rad_s2,
                            z_rad_s,
                            r_imp_rad_s2,
                            r_transform_rad_s2,
                            transform_mismatch,
                            [float(z_identity_error)],
                        )
                    )
                    next_lambda = (
                        lambda_s_inv + control_dt * dot_lambda_s_inv2
                    )
                    next_t1 = t1_rad_s + control_dt * dot_t1_rad_s2
                    next_gamma = damping_over_inertia - next_lambda
                    next_state_valid = bool(
                        np.isfinite(
                            np.concatenate(
                                (derived_values, next_lambda, next_t1, next_gamma)
                            )
                        ).all()
                        and np.all(next_lambda > 0.0)
                        and np.all(next_gamma > 0.0)
                    )
                    if next_state_valid:
                        self._lambda_s_inv = next_lambda
                        self._t1_rad_s = next_t1
                        impedance_transform_valid = True
                    else:
                        self._lambda_s_inv = None
                        self._t1_rad_s = None

        nominal = self.nominal_model.evaluate(q, qdot, root_pitch)
        eigenvalues = np.linalg.eigvalsh(nominal.mass)
        condition_number = float(np.linalg.cond(nominal.mass))
        if not np.isfinite(np.concatenate((eigenvalues, [condition_number]))).all():
            raise FloatingPointError("Nominal mass diagnostics contain NaN or Inf")

        # ====================================================
        # [PAPER] YAC Eq. (18) direct nominal torque candidate
        #
        # u0_paper = M*qddot_r + C*qdot_r + G - tau_sense - Ken*z。
        # 这是论文 Eq. (18) 的直接形式，继续完整计算并保留 Stage C
        # lambda_z candidates，但 Stage D-R 绝不把它施加到 actuator。
        #
        # [PLATFORM ADAPTATION] 当前 M5 interaction 是 Human-Exo 相对
        # 位移/速度产生的内部恢复性 coupling torque，不是独立 exogenous
        # disturbance。因此 interaction sensing 仍进入 t1，而实际 motor
        # command 使用：
        # u0_flexible = M*qddot_r + C*qdot_r + G - Ken*z，
        # 即 tau_interaction_cancel=[0,0]，不逐点取消内部 coupling。
        # ====================================================
        u0_inertia_nm: np.ndarray | None = None
        u0_coriolis_nm: np.ndarray | None = None
        u0_gravity_nm: np.ndarray | None = None
        u0_interaction_nm: np.ndarray | None = None
        tau_interaction_cancel_nm = np.zeros(2, dtype=float)
        ken_feedback_nm: dict[float, np.ndarray] = {}
        u0_candidates_nm: dict[float, np.ndarray] = {}
        u0_paper_raw_nm: np.ndarray | None = None
        u0_flexible_raw_nm: np.ndarray | None = None
        paper_minus_flexible_nm: np.ndarray | None = None
        paper_flexible_identity_error_nm: np.ndarray | None = None
        if (
            impedance_transform_valid
            and qddot_r_rad_s2 is not None
            and qdot_r_rad_s is not None
            and z_rad_s is not None
        ):
            u0_inertia_nm = nominal.mass @ qddot_r_rad_s2

            # [PAPER] Eq. (18) 需要 C(q,qdot_actual) 乘 qdot_r，不能复用
            # Stage A 中为了诊断而计算的 C(q,qdot_actual)*qdot_actual。
            u0_coriolis_nm = nominal.coriolis_matrix @ qdot_r_rad_s
            u0_gravity_nm = nominal.gravity_torque.copy()
            u0_interaction_nm = -tau_interaction_sense
            flexible_feedforward_nm = (
                u0_inertia_nm
                + u0_coriolis_nm
                + u0_gravity_nm
            )
            for rate, ken_matrix in self.ken_candidates_nms_per_rad.items():
                feedback_nm = -(ken_matrix @ z_rad_s)
                candidate_nm = (
                    flexible_feedforward_nm + u0_interaction_nm + feedback_nm
                )
                if not np.isfinite(
                    np.concatenate((feedback_nm, candidate_nm))
                ).all():
                    raise FloatingPointError(
                        f"Stage C u0 candidate is non-finite at lambda_z={rate:g}"
                    )
                ken_feedback_nm[rate] = feedback_nm
                u0_candidates_nm[rate] = candidate_nm
            active_feedback_nm = ken_feedback_nm[Z_DAMPING_RATE_S_INV]
            u0_paper_raw_nm = u0_candidates_nm[Z_DAMPING_RATE_S_INV]
            u0_flexible_raw_nm = flexible_feedforward_nm + active_feedback_nm
            paper_minus_flexible_nm = u0_paper_raw_nm - u0_flexible_raw_nm
            paper_flexible_identity_error_nm = (
                paper_minus_flexible_nm + tau_interaction_sense
            )
            if not np.isfinite(
                np.concatenate(
                    (
                        u0_paper_raw_nm,
                        u0_flexible_raw_nm,
                        paper_minus_flexible_nm,
                        paper_flexible_identity_error_nm,
                    )
                )
            ).all():
                raise FloatingPointError(
                    "Stage D-R Paper/Flexible candidates are non-finite"
                )

        # ====================================================
        # [PLATFORM] reduced nominal-model residual diagnostic
        #
        # 该字段沿用 Stage C 的 unactuated-balance 定义，不减 Stage D motor
        # torque；Active run 中它不再是完整 plant residual。该量还包含
        # floating root、right-leg coupling、fixed-base reduction error、
        # joint constraints、M5 timing mismatch 与其他未建模动力学，仍不能
        # 宣称为 true disturbance d(t)。
        # ====================================================
        nominal_residual_nm = (
            nominal.mass @ qacc
            + nominal.coriolis_torque
            + nominal.gravity_torque
            - tau_interaction_sense
        )
        if not np.isfinite(nominal_residual_nm).all():
            raise FloatingPointError("Nominal 2D model residual is non-finite")

        # ====================================================
        # [PAPER] Stage E1 B6 single-Critic 与 continuous ADP policy
        #
        # Vhat=Wc^T phi_c(z,a)，grad_z Vhat=J_phi Wc，
        # mu=-1/2 R^-1 g_nom^T grad_z Vhat。Paper 未给 basis/Q/R/kappa
        # 的 MuJoCo 数值，故使用 Stage E0 已冻结的 B6/Q/R/kappa。
        #
        # [PLATFORM TIMING] compute 发生在新 physics interval 之前。当前
        # qacc 由上一区间的 previous_applied_mu 生成，因此
        # f_hat=zdot_measured-g_nom*previous_applied_mu。本行 policy 使用
        # 更新前的 current W，delta_W 在下一行才生效，保证时序因果。
        # ====================================================
        u0_raw_nm = (
            u0_flexible_raw_nm
            if self.nominal_torque_semantics == FLEXIBLE_NOMINAL_SEMANTICS
            else u0_paper_raw_nm
        )
        active_window_ready = bool(
            self._reference_valid
            and impedance_transform_valid
            and u0_raw_nm is not None
        )
        critic_enabled = self.mode in (CONTINUOUS_ADP_MODE, FULL_YAC_MODE)
        critic_valid = bool(
            critic_enabled
            and np.isfinite(self._critic_weight).all()
            and np.linalg.norm(self._critic_weight) < CRITIC_WEIGHT_NORM_LIMIT
        )
        critic_update_valid = False
        critic_fallback_nominal = False
        critic_weight_logged = self._critic_weight.copy()
        critic_weight_next = self._critic_weight.copy()
        critic_weight_norm = float(np.linalg.norm(critic_weight_logged))
        critic_features: np.ndarray | None = None
        critic_jacobian: np.ndarray | None = None
        value_hat: float | str = ""
        grad_v_z: np.ndarray | None = None
        bellman_error: float | str = ""
        m_regressor: np.ndarray | None = None
        critic_update: np.ndarray | None = None
        f_hat_rad_s2: np.ndarray | None = None
        g_nom = np.linalg.inv(nominal.mass)
        mu_raw_nm: np.ndarray | None = None
        mu_continuous_nm: np.ndarray | None = None
        mu_event_nm: np.ndarray | None = None
        mu_event_error_nm: np.ndarray | None = None
        event_delta_pretrigger_rad_s: np.ndarray | None = None
        event_eta_logged: float | str = ""
        event_eta_next: float | str = ""
        event_eta_decay: float | str = ""
        event_eta_input_coefficient: float | str = ""
        event_z_norm_sq: float | str = ""
        event_threshold_sq: float | str = ""
        event_triggered = False
        event_reason_code = EVENT_REASON_NONE
        event_age_steps: int | str = ""
        event_age_s: float | str = ""
        inter_event_steps: int | str = ""
        inter_event_time_s: float | str = ""
        inter_event_valid = False
        event_sampled_z_logged: np.ndarray | None = None

        if critic_enabled and active_window_ready:
            assert z_rad_s is not None
            assert qddot_r_rad_s2 is not None
            try:
                if not critic_valid:
                    raise FloatingPointError("Critic weight watchdog failed")
                critic_features, critic_jacobian = _critic_basis_b6(
                    z_rad_s, a_sched
                )
                grad_v_z = critic_jacobian @ critic_weight_logged
                value_hat = float(critic_weight_logged @ critic_features)
                mu_continuous_nm = -0.5 * (
                    CRITIC_R_INV @ g_nom.T @ grad_v_z
                )

                if self.event_trigger_enabled:
                    # ================================================
                    # [PAPER] Dynamic Event Trigger Eq.43--44
                    # delta_j=z_sampled-z_current；trigger 使用刷新前 delta。
                    #
                    # [PAPER AMBIGUITY] Eq. (28) 的显式 policy 采用
                    # equation-priority interpretation：z 在 event 间保持，
                    # a/W/g 仍使用当前值。这不是 strict full-torque ZOH；
                    # Critic 继续用 current z/a 在每个有效 100 Hz step 更新。
                    #
                    # [NUMERICAL CALIBRATION] lambda/sigma/xi/rho/epsilon
                    # 是当前 MuJoCo platform 数值，Paper 只规定正性条件和
                    # trigger structure。eta 在当前 interval 将 z 视为常值，
                    # 使用 compute() 实际 dt 的 exact ZOH 更新，event 时不清零。
                    # ================================================
                    if not self._event_active:
                        self._event_sampled_z_rad_s = None
                        self._event_eta = EVENT_ETA_INITIAL
                        self._event_age_steps = 0
                        self._last_event_time_s = None
                        self._last_event_valid_step = None
                        self._event_active = True
                    self._event_policy_sample_count += 1
                    event_eta_logged = float(self._event_eta)
                    event_decision = _event_trigger_decision(
                        self._event_sampled_z_rad_s,
                        z_rad_s,
                        float(event_eta_logged),
                        xi=self.event_xi,
                        rho_delta=self.event_rho_delta,
                        epsilon_delta=self.event_epsilon_delta,
                    )
                    event_delta_pretrigger_rad_s = (
                        event_decision.delta_pretrigger_rad_s.copy()
                    )
                    event_z_norm_sq = event_decision.z_norm_sq
                    event_threshold_sq = event_decision.threshold_sq
                    event_triggered = event_decision.triggered
                    event_reason_code = event_decision.reason_code
                    if event_triggered:
                        if (
                            self._last_event_time_s is not None
                            and self._last_event_valid_step is not None
                        ):
                            inter_event_steps = int(
                                self._event_policy_sample_count
                                - self._last_event_valid_step
                            )
                            inter_event_time_s = float(
                                time_s - self._last_event_time_s
                            )
                            inter_event_valid = True
                        self._event_sampled_z_rad_s = (
                            event_decision.sampled_z_rad_s.copy()
                        )
                        self._last_event_time_s = time_s
                        self._last_event_valid_step = (
                            self._event_policy_sample_count
                        )
                        self._event_age_steps = 0
                        self._event_count += 1
                        if event_reason_code == EVENT_REASON_INITIALIZATION:
                            self._event_initialization_count += 1
                        elif event_reason_code == EVENT_REASON_THRESHOLD:
                            self._event_threshold_count += 1
                    else:
                        self._event_age_steps += 1
                    assert self._event_sampled_z_rad_s is not None
                    event_sampled_z_logged = self._event_sampled_z_rad_s.copy()
                    event_age_steps = self._event_age_steps
                    event_age_s = (
                        0.0
                        if self._last_event_time_s is None
                        else float(time_s - self._last_event_time_s)
                    )
                    _, event_policy_jacobian = _critic_basis_b6(
                        event_sampled_z_logged, a_sched
                    )
                    event_policy_grad = (
                        event_policy_jacobian @ critic_weight_logged
                    )
                    mu_event_nm = -0.5 * (
                        CRITIC_R_INV @ g_nom.T @ event_policy_grad
                    )
                    mu_event_error_nm = mu_event_nm - mu_continuous_nm
                    mu_raw_nm = mu_event_nm.copy()
                    (
                        event_eta_next,
                        event_eta_decay,
                        event_eta_input_coefficient,
                    ) = _event_eta_exact_zoh_step(
                        float(event_eta_logged),
                        float(event_z_norm_sq),
                        control_dt,
                        lambda_eta=self.event_lambda_eta,
                        sigma_eta=self.event_sigma_eta,
                    )
                    event_values = np.concatenate(
                        (
                            event_delta_pretrigger_rad_s,
                            event_sampled_z_logged,
                            event_policy_grad,
                            mu_raw_nm,
                            mu_event_error_nm,
                            [
                                float(event_eta_logged),
                                float(event_eta_next),
                                float(event_eta_decay),
                                float(event_eta_input_coefficient),
                                float(event_threshold_sq),
                                float(event_z_norm_sq),
                            ],
                        )
                    )
                    if not np.isfinite(event_values).all():
                        raise FloatingPointError(
                            "Dynamic Event state/policy is non-finite"
                        )
                    self._event_eta = float(event_eta_next)
                else:
                    mu_raw_nm = mu_continuous_nm.copy()
                if self._previous_applied_mu_nm is not None:
                    # [PAPER] e=Hhat；m=de/dW。因 mu 满足 dH/dmu=0，
                    # policy 对 W 的间接导数项由 envelope condition 相消，
                    # 故 m=J_phi^T(f_hat+g_nom*mu)。Eq. (42) 分母整体平方。
                    z_dot_measured = qacc - qddot_r_rad_s2
                    f_hat_rad_s2 = (
                        z_dot_measured
                        - g_nom @ self._previous_applied_mu_nm
                    )
                    closed_loop_direction = f_hat_rad_s2 + g_nom @ mu_raw_nm
                    bellman_error = float(
                        grad_v_z @ closed_loop_direction
                        + z_rad_s @ CRITIC_Q @ z_rad_s
                        + mu_raw_nm @ CRITIC_R @ mu_raw_nm
                    )
                    m_regressor = critic_jacobian.T @ closed_loop_direction
                    denominator = (1.0 + float(m_regressor @ m_regressor)) ** 2
                    critic_update = (
                        -control_dt
                        * CRITIC_KAPPA
                        * m_regressor
                        * bellman_error
                        / denominator
                    )
                    critic_weight_next = critic_weight_logged + critic_update
                    next_norm = float(np.linalg.norm(critic_weight_next))
                    if (
                        not np.isfinite(
                            np.concatenate(
                                (
                                    g_nom.reshape(-1),
                                    grad_v_z,
                                    mu_raw_nm,
                                    f_hat_rad_s2,
                                    m_regressor,
                                    critic_update,
                                    critic_weight_next,
                                    [float(value_hat), float(bellman_error), next_norm],
                                )
                            )
                        ).all()
                        or next_norm >= CRITIC_WEIGHT_NORM_LIMIT
                    ):
                        raise FloatingPointError("Critic update watchdog failed")
                    critic_update_valid = True
                elif not np.isfinite(
                    np.concatenate((g_nom.reshape(-1), grad_v_z, mu_raw_nm))
                ).all():
                    raise FloatingPointError("Initial Critic policy is non-finite")
            except (
                FloatingPointError,
                OverflowError,
                np.linalg.LinAlgError,
            ) as exc:
                raise RuntimeError("YAC Critic/Event numerical failure") from exc
            else:
                self._consecutive_critic_failures = 0
                self._critic_weight = critic_weight_next
        elif critic_enabled:
            # reference 尚未有效不是 Critic failure。
            self._consecutive_critic_failures = 0
            if self.event_trigger_enabled:
                # ADP invalid 期间不累计 eta/event。下次 inactive->active
                # 必须以 z_sampled=z_current 重新产生 initialization event。
                self._event_active = False
                self._event_sampled_z_rad_s = None
                self._event_eta = EVENT_ETA_INITIAL
                self._event_age_steps = 0
                self._last_event_time_s = None
                self._last_event_valid_step = None

        # ====================================================
        # [PLATFORM ADAPTATION] Unclipped nominal + residual deployment
        #
        # reference/transform 无效时严格输出零。Continuous/Full YAC 真实施加
        # u=u0_flexible+mu_raw：不做 residual clip、total clip、ramp、滤波、
        # slew limiter 或额外 scale。XML 中原生 ±100 N·m ctrlrange 保持不变，
        # command/actual 的饱和诊断由 runner 保存的 actuator_force 离线判定。
        # 旧 nominal-only 模式保留 Stage D-R engagement ramp 以维护历史兼容。
        # ====================================================
        controller_valid = bool(
            (not self._reference_valid or active_window_ready)
            and (not critic_enabled or critic_valid)
        )
        if active_window_ready and self._engagement_start_time_s is None:
            self._engagement_start_time_s = time_s
        engagement_active = self._engagement_start_time_s is not None
        ramp_factor = 0.0
        mu_scaled_nm = np.zeros(2, dtype=float)
        mu_limited_nm = np.zeros(2, dtype=float)
        mu_applied_nm = np.zeros(2, dtype=float)
        mu_limit_active = np.zeros(2, dtype=bool)
        mu_clip_amount_nm = np.zeros(2, dtype=float)
        u_total_raw_nm: np.ndarray | None = None
        u_scaled_nm = np.zeros(2, dtype=float)
        u_applied_nm = np.zeros(2, dtype=float)
        torque_limit_active = np.zeros(2, dtype=bool)
        torque_clip_amount_nm = np.zeros(2, dtype=float)
        if active_window_ready:
            assert u0_raw_nm is not None
            assert self._engagement_start_time_s is not None
            if critic_enabled:
                assert mu_raw_nm is not None
                ramp_factor = 1.0
                mu_scaled_nm = mu_raw_nm.copy()
                mu_limited_nm = mu_raw_nm.copy()
                mu_applied_nm = mu_raw_nm.copy()
                u_total_raw_nm = u0_raw_nm + mu_applied_nm
                u_scaled_nm = u_total_raw_nm.copy()
                u_applied_nm = u_total_raw_nm.copy()
            else:
                ramp_factor = float(
                    np.clip(
                        (time_s - self._engagement_start_time_s)
                        / ENGAGEMENT_RAMP_DURATION_S,
                        0.0,
                        1.0,
                    )
                )
                u_total_raw_nm = self.active_scale * u0_raw_nm
                u_scaled_nm = ramp_factor * u_total_raw_nm
                u_applied_nm = u_scaled_nm.copy()
            active_values = np.concatenate(
                (
                    u0_raw_nm,
                    mu_scaled_nm,
                    mu_limited_nm,
                    mu_applied_nm,
                    mu_clip_amount_nm,
                    u_total_raw_nm,
                    u_scaled_nm,
                    u_applied_nm,
                    torque_clip_amount_nm,
                    [ramp_factor],
                )
            )
            if not np.isfinite(active_values).all():
                raise FloatingPointError("YAC active torque path is non-finite")

        # 本次实际 residual 与下一次 observation 对齐。在 reference 无效、
        # nominal mode 或 Critic fallback 中均为真实的零 residual，不伪造数据。
        self._previous_applied_mu_nm = mu_applied_nm.copy()

        unavailable = ""

        def diagnostic_component(
            value: np.ndarray | None, joint_index: int
        ) -> float | str:
            """[NUMERICAL] 无效变换状态留空，绝不把 NaN/Inf 写入诊断。"""
            if value is None:
                return unavailable
            component = float(value[joint_index])
            return component if math.isfinite(component) else unavailable

        diagnostics: dict[str, Any] = {
            "time_s": time_s,
            "mode": self.mode,
            "reference_valid": self._reference_valid,
            "gait_phase": phase,
            "stride_period_hat_s": self._stride_period_hat_s,
            "left_heel_strike": left_heel_strike,
            "left_heel_strike_count": self._accepted_left_heel_strikes,
            "q_exo_hip_l_rad": q[0],
            "q_exo_knee_l_rad": q[1],
            "qdot_exo_hip_l_rad_s": qdot[0],
            "qdot_exo_knee_l_rad_s": qdot[1],
            "q_ref_hip_rad": unavailable if q_d is None else q_d[0],
            "q_ref_knee_rad": unavailable if q_d is None else q_d[1],
            "qdot_ref_hip_rad_s": unavailable if qdot_d is None else qdot_d[0],
            "qdot_ref_knee_rad_s": unavailable if qdot_d is None else qdot_d[1],
            "qddot_ref_hip_rad_s2": unavailable if qddot_d is None else qddot_d[0],
            "qddot_ref_knee_rad_s2": unavailable if qddot_d is None else qddot_d[1],
            "error_hip_rad": unavailable if delta_q is None else delta_q[0],
            "error_knee_rad": unavailable if delta_q is None else delta_q[1],
            "error_velocity_hip_rad_s": (
                unavailable if delta_qdot is None else delta_qdot[0]
            ),
            "error_velocity_knee_rad_s": (
                unavailable if delta_qdot is None else delta_qdot[1]
            ),
            "error_position_norm_rad": (
                unavailable if delta_q is None else float(np.linalg.norm(delta_q))
            ),
            "error_velocity_norm_rad_s": (
                unavailable
                if delta_qdot is None
                else float(np.linalg.norm(delta_qdot))
            ),
            "activation_proxy": activation_proxy,
            "a_sched": a_sched,
            **{
                f"activation_{name}": value
                for name, value in single_activation.items()
            },
            "Kr_hip_nm_per_rad": stiffness[0],
            "Kr_knee_nm_per_rad": stiffness[1],
            "Md_hip": DESIRED_INERTIA_KGM2[0],
            "Md_knee": DESIRED_INERTIA_KGM2[1],
            "Dd_hip_nms_per_rad": DESIRED_DAMPING_NMS_PER_RAD[0],
            "Dd_knee_nms_per_rad": DESIRED_DAMPING_NMS_PER_RAD[1],
            "Lambda_hip_s_inv": diagnostic_component(lambda_s_inv, 0),
            "Lambda_knee_s_inv": diagnostic_component(lambda_s_inv, 1),
            "Gamma_hip_s_inv": diagnostic_component(gamma_s_inv, 0),
            "Gamma_knee_s_inv": diagnostic_component(gamma_s_inv, 1),
            "dot_Lambda_hip_s_inv2": diagnostic_component(
                dot_lambda_s_inv2, 0
            ),
            "dot_Lambda_knee_s_inv2": diagnostic_component(
                dot_lambda_s_inv2, 1
            ),
            "t1_hip_rad_s": diagnostic_component(t1_rad_s, 0),
            "t1_knee_rad_s": diagnostic_component(t1_rad_s, 1),
            "dot_t1_hip_rad_s2": diagnostic_component(dot_t1_rad_s2, 0),
            "dot_t1_knee_rad_s2": diagnostic_component(dot_t1_rad_s2, 1),
            "qdot_r_hip_rad_s": diagnostic_component(qdot_r_rad_s, 0),
            "qdot_r_knee_rad_s": diagnostic_component(qdot_r_rad_s, 1),
            "qddot_r_hip_rad_s2": diagnostic_component(qddot_r_rad_s2, 0),
            "qddot_r_knee_rad_s2": diagnostic_component(qddot_r_rad_s2, 1),
            "z_hip_rad_s": diagnostic_component(z_rad_s, 0),
            "z_knee_rad_s": diagnostic_component(z_rad_s, 1),
            "z_norm_rad_s": (
                unavailable if z_rad_s is None else float(np.linalg.norm(z_rad_s))
            ),
            "z_identity_error": z_identity_error,
            "qacc_exo_hip_l_rad_s2": qacc[0],
            "qacc_exo_knee_l_rad_s2": qacc[1],
            "r_imp_hip_rad_s2": diagnostic_component(r_imp_rad_s2, 0),
            "r_imp_knee_rad_s2": diagnostic_component(r_imp_rad_s2, 1),
            "r_transform_hip_rad_s2": diagnostic_component(
                r_transform_rad_s2, 0
            ),
            "r_transform_knee_rad_s2": diagnostic_component(
                r_transform_rad_s2, 1
            ),
            "transform_mismatch_hip": diagnostic_component(
                transform_mismatch, 0
            ),
            "transform_mismatch_knee": diagnostic_component(
                transform_mismatch, 1
            ),
            "impedance_transform_valid": impedance_transform_valid,
            "qddot_r_component_ref_hip_rad_s2": diagnostic_component(
                qddot_r_component_ref, 0
            ),
            "qddot_r_component_ref_knee_rad_s2": diagnostic_component(
                qddot_r_component_ref, 1
            ),
            "qddot_r_component_dot_Lambda_hip_rad_s2": diagnostic_component(
                qddot_r_component_dot_lambda, 0
            ),
            "qddot_r_component_dot_Lambda_knee_rad_s2": diagnostic_component(
                qddot_r_component_dot_lambda, 1
            ),
            "qddot_r_component_Lambda_Deltaqdot_hip_rad_s2": diagnostic_component(
                qddot_r_component_lambda_delta_qdot, 0
            ),
            "qddot_r_component_Lambda_Deltaqdot_knee_rad_s2": diagnostic_component(
                qddot_r_component_lambda_delta_qdot, 1
            ),
            "qddot_r_component_dot_t1_hip_rad_s2": diagnostic_component(
                qddot_r_component_dot_t1, 0
            ),
            "qddot_r_component_dot_t1_knee_rad_s2": diagnostic_component(
                qddot_r_component_dot_t1, 1
            ),
            "u0_inertia_hip_nm": diagnostic_component(u0_inertia_nm, 0),
            "u0_inertia_knee_nm": diagnostic_component(u0_inertia_nm, 1),
            "u0_coriolis_hip_nm": diagnostic_component(u0_coriolis_nm, 0),
            "u0_coriolis_knee_nm": diagnostic_component(u0_coriolis_nm, 1),
            "u0_gravity_hip_nm": diagnostic_component(u0_gravity_nm, 0),
            "u0_gravity_knee_nm": diagnostic_component(u0_gravity_nm, 1),
            "u0_interaction_hip_nm": diagnostic_component(
                u0_interaction_nm, 0
            ),
            "u0_interaction_knee_nm": diagnostic_component(
                u0_interaction_nm, 1
            ),
            "interaction_semantics_mode": INTERACTION_SEMANTICS_MODE,
            "nominal_torque_semantics": self.nominal_torque_semantics,
            "tau_interaction_sense_hip_nm": tau_interaction_sense[0],
            "tau_interaction_sense_knee_nm": tau_interaction_sense[1],
            "tau_interaction_cancel_hip_nm": tau_interaction_cancel_nm[0],
            "tau_interaction_cancel_knee_nm": tau_interaction_cancel_nm[1],
            "u0_paper_raw_hip_nm": diagnostic_component(u0_paper_raw_nm, 0),
            "u0_paper_raw_knee_nm": diagnostic_component(u0_paper_raw_nm, 1),
            "u0_flexible_raw_hip_nm": diagnostic_component(
                u0_flexible_raw_nm, 0
            ),
            "u0_flexible_raw_knee_nm": diagnostic_component(
                u0_flexible_raw_nm, 1
            ),
            "paper_minus_flexible_hip_nm": diagnostic_component(
                paper_minus_flexible_nm, 0
            ),
            "paper_minus_flexible_knee_nm": diagnostic_component(
                paper_minus_flexible_nm, 1
            ),
            "paper_flexible_identity_error_hip_nm": diagnostic_component(
                paper_flexible_identity_error_nm, 0
            ),
            "paper_flexible_identity_error_knee_nm": diagnostic_component(
                paper_flexible_identity_error_nm, 1
            ),
            "nominal_residual_hip_nm": nominal_residual_nm[0],
            "nominal_residual_knee_nm": nominal_residual_nm[1],
            "u0_lambda0_hip_nm": diagnostic_component(
                u0_candidates_nm.get(0.0), 0
            ),
            "u0_lambda0_knee_nm": diagnostic_component(
                u0_candidates_nm.get(0.0), 1
            ),
            "ken2_feedback_hip_nm": diagnostic_component(
                ken_feedback_nm.get(2.0), 0
            ),
            "ken2_feedback_knee_nm": diagnostic_component(
                ken_feedback_nm.get(2.0), 1
            ),
            "u0_lambda2_hip_nm": diagnostic_component(
                u0_candidates_nm.get(2.0), 0
            ),
            "u0_lambda2_knee_nm": diagnostic_component(
                u0_candidates_nm.get(2.0), 1
            ),
            "ken5_feedback_hip_nm": diagnostic_component(
                ken_feedback_nm.get(5.0), 0
            ),
            "ken5_feedback_knee_nm": diagnostic_component(
                ken_feedback_nm.get(5.0), 1
            ),
            "u0_lambda5_hip_nm": diagnostic_component(
                u0_candidates_nm.get(5.0), 0
            ),
            "u0_lambda5_knee_nm": diagnostic_component(
                u0_candidates_nm.get(5.0), 1
            ),
            "ken10_feedback_hip_nm": diagnostic_component(
                ken_feedback_nm.get(10.0), 0
            ),
            "ken10_feedback_knee_nm": diagnostic_component(
                ken_feedback_nm.get(10.0), 1
            ),
            "u0_lambda10_hip_nm": diagnostic_component(
                u0_candidates_nm.get(10.0), 0
            ),
            "u0_lambda10_knee_nm": diagnostic_component(
                u0_candidates_nm.get(10.0), 1
            ),
            # =================================================
            # Stage E1 Critic/ADP 独立诊断。这些字段不进入
            # public timeseries，但能重建 B6 -> V -> Bellman -> mu
            # -> total torque 的整条算法链。
            # =================================================
            "critic_enabled": critic_enabled,
            "R_hip": float(CRITIC_R[0, 0]),
            "R_knee": float(CRITIC_R[1, 1]),
            "adp_action_scale_hip_nm": float(ADP_ACTION_SCALE_NM[0]),
            "adp_action_scale_knee_nm": float(ADP_ACTION_SCALE_NM[1]),
            "adp_R_source": ADP_R_SOURCE,
            "critic_valid": critic_valid,
            "critic_update_valid": critic_update_valid,
            "critic_fallback_nominal": critic_fallback_nominal,
            **{
                f"critic_weight_{index}": float(critic_weight_logged[index])
                for index in range(CRITIC_WEIGHT_DIMENSION)
            },
            "critic_weight_norm": critic_weight_norm,
            **{
                f"phi_{index}": (
                    unavailable
                    if critic_features is None
                    else float(critic_features[index])
                )
                for index in range(CRITIC_WEIGHT_DIMENSION)
            },
            "value_hat": value_hat,
            "gradV_z_hip": diagnostic_component(grad_v_z, 0),
            "gradV_z_knee": diagnostic_component(grad_v_z, 1),
            "bellman_error": bellman_error,
            "bellman_error_abs": (
                unavailable
                if bellman_error == unavailable
                else abs(float(bellman_error))
            ),
            "m_norm": (
                unavailable
                if m_regressor is None
                else float(np.linalg.norm(m_regressor))
            ),
            "critic_update_norm": (
                unavailable
                if critic_update is None
                else float(np.linalg.norm(critic_update))
            ),
            "f_hat_hip_rad_s2": diagnostic_component(f_hat_rad_s2, 0),
            "f_hat_knee_rad_s2": diagnostic_component(f_hat_rad_s2, 1),
            "g_nom_11": g_nom[0, 0],
            "g_nom_12": g_nom[0, 1],
            "g_nom_21": g_nom[1, 0],
            "g_nom_22": g_nom[1, 1],
            "mu_raw_hip_nm": diagnostic_component(mu_raw_nm, 0),
            "mu_raw_knee_nm": diagnostic_component(mu_raw_nm, 1),
            "mu_continuous_hip_nm": diagnostic_component(
                mu_continuous_nm, 0
            ),
            "mu_continuous_knee_nm": diagnostic_component(
                mu_continuous_nm, 1
            ),
            "adp_scale": self.adp_scale,
            "mu_scaled_hip_nm": mu_scaled_nm[0],
            "mu_scaled_knee_nm": mu_scaled_nm[1],
            "mu_applied_hip_nm": mu_applied_nm[0],
            "mu_applied_knee_nm": mu_applied_nm[1],
            "mu_raw_minus_applied_hip_nm": (
                unavailable
                if mu_raw_nm is None
                else float(mu_raw_nm[0] - mu_applied_nm[0])
            ),
            "mu_raw_minus_applied_knee_nm": (
                unavailable
                if mu_raw_nm is None
                else float(mu_raw_nm[1] - mu_applied_nm[1])
            ),
            "adp_residual_software_clipping": False,
            "mu_limit_active_hip": bool(mu_limit_active[0]),
            "mu_limit_active_knee": bool(mu_limit_active[1]),
            "mu_clip_amount_hip_nm": mu_clip_amount_nm[0],
            "mu_clip_amount_knee_nm": mu_clip_amount_nm[1],
            "event_enabled": self.event_trigger_enabled,
            "event_trigger_enabled": self.event_trigger_enabled,
            "event_adp_valid": bool(
                self.event_trigger_enabled and active_window_ready and critic_valid
            ),
            "event_formula": (
                EVENT_FORMULA if self.event_trigger_enabled else unavailable
            ),
            "event_policy_semantics": (
                EVENT_POLICY_SEMANTICS
                if self.event_trigger_enabled
                else unavailable
            ),
            "critic_update_semantics": CRITIC_UPDATE_SEMANTICS,
            "strict_full_torque_zoh": STRICT_FULL_TORQUE_ZOH,
            "event_eta_discretization": (
                EVENT_ETA_DISCRETIZATION
                if self.event_trigger_enabled
                else unavailable
            ),
            "event_parameter_source": (
                EVENT_PARAMETER_SOURCE
                if self.event_trigger_enabled
                else unavailable
            ),
            "event_lambda_eta": (
                self.event_lambda_eta
                if self.event_trigger_enabled
                else unavailable
            ),
            "event_sigma_eta": (
                self.event_sigma_eta
                if self.event_trigger_enabled
                else unavailable
            ),
            "event_xi": (
                self.event_xi if self.event_trigger_enabled else unavailable
            ),
            "event_rho_delta": (
                self.event_rho_delta
                if self.event_trigger_enabled
                else unavailable
            ),
            "event_epsilon_delta": (
                self.event_epsilon_delta
                if self.event_trigger_enabled
                else unavailable
            ),
            "event_eta_initial": (
                EVENT_ETA_INITIAL if self.event_trigger_enabled else unavailable
            ),
            "eta": event_eta_logged,
            "event_eta": event_eta_logged,
            "event_eta_next": event_eta_next,
            "event_eta_decay": event_eta_decay,
            "event_eta_input_coefficient": event_eta_input_coefficient,
            "event_reason_code": event_reason_code,
            "event_age_steps": event_age_steps,
            "event_age_s": event_age_s,
            "z_sampled_hip": diagnostic_component(
                event_sampled_z_logged, 0
            ),
            "z_sampled_knee": diagnostic_component(
                event_sampled_z_logged, 1
            ),
            "delta_hip_pretrigger": diagnostic_component(
                event_delta_pretrigger_rad_s, 0
            ),
            "delta_knee_pretrigger": diagnostic_component(
                event_delta_pretrigger_rad_s, 1
            ),
            "delta_norm_pretrigger": (
                unavailable
                if event_delta_pretrigger_rad_s is None
                else float(np.linalg.norm(event_delta_pretrigger_rad_s))
            ),
            "delta_norm_sq_pretrigger": (
                unavailable
                if event_delta_pretrigger_rad_s is None
                else float(
                    event_delta_pretrigger_rad_s
                    @ event_delta_pretrigger_rad_s
                )
            ),
            "z_norm": (
                unavailable
                if event_z_norm_sq == unavailable
                else math.sqrt(float(event_z_norm_sq))
            ),
            "z_norm_sq": event_z_norm_sq,
            "trigger_threshold_norm": (
                unavailable
                if event_threshold_sq == unavailable
                else math.sqrt(float(event_threshold_sq))
            ),
            "trigger_threshold_sq": event_threshold_sq,
            "event_delta_hip_rad_s": diagnostic_component(
                event_delta_pretrigger_rad_s, 0
            ),
            "event_delta_knee_rad_s": diagnostic_component(
                event_delta_pretrigger_rad_s, 1
            ),
            "event_delta_norm_sq": (
                unavailable
                if event_delta_pretrigger_rad_s is None
                else float(
                    event_delta_pretrigger_rad_s
                    @ event_delta_pretrigger_rad_s
                )
            ),
            "event_threshold": event_threshold_sq,
            "event_triggered": event_triggered,
            "event_count": self._event_count,
            "initialization_event_count": self._event_initialization_count,
            "threshold_event_count": self._event_threshold_count,
            "event_policy_sample_count": self._event_policy_sample_count,
            "event_fraction_running": (
                unavailable
                if self._event_policy_sample_count == 0
                else self._event_count / self._event_policy_sample_count
            ),
            "inter_event_valid": inter_event_valid,
            "inter_event_steps": inter_event_steps,
            "inter_event_time_s": inter_event_time_s,
            "event_interval_s": inter_event_time_s,
            "event_sampled_z_hip_rad_s": diagnostic_component(
                event_sampled_z_logged, 0
            ),
            "event_sampled_z_knee_rad_s": diagnostic_component(
                event_sampled_z_logged, 1
            ),
            "mu_continuous_candidate_hip": diagnostic_component(
                mu_continuous_nm, 0
            ),
            "mu_continuous_candidate_knee": diagnostic_component(
                mu_continuous_nm, 1
            ),
            "mu_event_hip": diagnostic_component(mu_event_nm, 0),
            "mu_event_knee": diagnostic_component(mu_event_nm, 1),
            "mu_event_error_hip": diagnostic_component(
                mu_event_error_nm, 0
            ),
            "mu_event_error_knee": diagnostic_component(
                mu_event_error_nm, 1
            ),
            "mu_event_error_norm": (
                unavailable
                if mu_event_error_nm is None
                else float(np.linalg.norm(mu_event_error_nm))
            ),
            "u_total_raw_hip_nm": diagnostic_component(u_total_raw_nm, 0),
            "u_total_raw_knee_nm": diagnostic_component(u_total_raw_nm, 1),
            "active_scale": self.active_scale,
            "engagement_ramp_factor": ramp_factor,
            "engagement_active": engagement_active,
            "u0_raw_hip_nm": diagnostic_component(u0_raw_nm, 0),
            "u0_raw_knee_nm": diagnostic_component(u0_raw_nm, 1),
            "u_scaled_hip_nm": u_scaled_nm[0],
            "u_scaled_knee_nm": u_scaled_nm[1],
            "u_applied_hip_nm": u_applied_nm[0],
            "u_applied_knee_nm": u_applied_nm[1],
            "u_command_hip_nm": u_applied_nm[0],
            "u_command_knee_nm": u_applied_nm[1],
            "software_total_torque_clipping": False,
            "torque_limit_active_hip": bool(torque_limit_active[0]),
            "torque_limit_active_knee": bool(torque_limit_active[1]),
            "torque_clip_amount_hip_nm": torque_clip_amount_nm[0],
            "torque_clip_amount_knee_nm": torque_clip_amount_nm[1],
            "controller_valid": controller_valid,
            "fallback_zero_torque": False,
            "active_assistance": True,
            # Stage A-D 兼容字段；数值与 tau_interaction_sense 完全相同。
            "tau_e_hip_direct_nm": tau_interaction_sense[0],
            "tau_e_knee_direct_nm": tau_interaction_sense[1],
            "exo_root_pitch_rad": root_pitch,
            "absolute_thigh_angle_rad": nominal.absolute_thigh_angle_rad,
            "constraint_hip_nm": constraint_torque[0],
            "constraint_knee_nm": constraint_torque[1],
            "constraint_hip_active": bool(
                abs(constraint_torque[0]) > self.constraint_active_threshold_nm
            ),
            "constraint_knee_active": bool(
                abs(constraint_torque[1]) > self.constraint_active_threshold_nm
            ),
            "nominal_M11": nominal.mass[0, 0],
            "nominal_M12": nominal.mass[0, 1],
            "nominal_M22": nominal.mass[1, 1],
            "nominal_Cqdot_hip_nm": nominal.coriolis_torque[0],
            "nominal_Cqdot_knee_nm": nominal.coriolis_torque[1],
            "nominal_G_hip_nm": nominal.gravity_torque[0],
            "nominal_G_knee_nm": nominal.gravity_torque[1],
            "nominal_eig_min": eigenvalues[0],
            "nominal_eig_max": eigenvalues[1],
            "nominal_condition_number": condition_number,
            "reference_cycle_count": self.reference.cycle_count,
            "reference_median_period_s": self.reference.median_period_s,
            "reference_raw_endpoint_gap_hip_rad": self.reference.raw_endpoint_gap_rad[0],
            "reference_raw_endpoint_gap_knee_rad": self.reference.raw_endpoint_gap_rad[1],
            "reference_periodic_position_gap_hip_rad": self.reference.periodic_position_gap_rad[0],
            "reference_periodic_position_gap_knee_rad": self.reference.periodic_position_gap_rad[1],
            "reference_periodic_qphi_gap_norm": float(
                np.linalg.norm(self.reference.periodic_first_derivative_gap)
            ),
            "reference_periodic_qphiphi_gap_norm": float(
                np.linalg.norm(self.reference.periodic_second_derivative_gap)
            ),
        }
        self.last_diagnostics = diagnostics

        # ====================================================
        # [PLATFORM] YAC Stage D-R 患侧主动通道映射
        #
        # reference/transform 未就绪或 numerical fallback 时患侧也是零；
        # 右侧和两个保留通道在所有情况下严格保持零。
        # ====================================================
        torques = dict(ZERO_TORQUES)
        torques["HipExo_L"] = float(u_applied_nm[0])
        torques["KneeExo_L"] = float(u_applied_nm[1])
        return torques


# ============================================================
# Bilateral floating-base deployment
# ============================================================
BILATERAL_NOMINAL_MODE = "bilateral_nominal"
BILATERAL_FULL_YAC_MODE = "bilateral_full_yac"


class _BilateralKKTController:
    """Four-motor YAC with exact bilateral nominal dynamics.

    The nominal term uses the validated seven-coordinate Schur/KKT model.
    The two ADP residuals intentionally remain mirrored 2DOF critics; their
    local input maps are the diagonal 2x2 blocks of the exact 4x4 ``g``.
    """

    def __init__(
        self,
        *,
        reference_result: str | Path,
        mujoco_model: Any,
        mode: str = BILATERAL_NOMINAL_MODE,
        active_scale: float = 1.0,
        adp_scale: float = 1.0,
        event_trigger: bool = False,
        event_lambda_eta: float | None = None,
        event_sigma_eta: float | None = None,
        event_xi: float | None = None,
        event_rho_delta: float | None = None,
        event_epsilon_delta: float | None = None,
        ewma_beta: float = EWMA_BETA,
        **_: Any,
    ) -> None:
        normalized = str(mode).strip().lower()
        if normalized not in (BILATERAL_NOMINAL_MODE, BILATERAL_FULL_YAC_MODE):
            raise ValueError("Bilateral YAC mode must be bilateral_nominal or bilateral_full_yac")
        if not math.isclose(float(active_scale), 1.0, abs_tol=1.0e-12):
            raise ValueError("Bilateral deployment freezes active_scale at 1.0")
        if not math.isclose(float(adp_scale), 1.0, abs_tol=1.0e-12):
            raise ValueError("Bilateral deployment freezes adp_scale at 1.0")
        if normalized == BILATERAL_FULL_YAC_MODE and not event_trigger:
            raise ValueError("Bilateral Full YAC requires Dynamic Event Trigger")
        if normalized == BILATERAL_NOMINAL_MODE and event_trigger:
            raise ValueError("Bilateral nominal-only mode cannot enable Dynamic Event")
        self.mode = normalized
        self.event_trigger_enabled = bool(event_trigger)
        self.event_lambda_eta = 0.0 if event_lambda_eta is None else float(event_lambda_eta)
        self.event_sigma_eta = 0.0 if event_sigma_eta is None else float(event_sigma_eta)
        self.event_xi = 0.0 if event_xi is None else float(event_xi)
        self.event_rho_delta = 1.0 if event_rho_delta is None else float(event_rho_delta)
        self.event_epsilon_delta = (
            0.0 if event_epsilon_delta is None else float(event_epsilon_delta)
        )
        if self.event_trigger_enabled and (
            self.event_lambda_eta <= 0.0
            or self.event_sigma_eta <= 0.0
            or not 0.0 < self.event_xi < 1.0
            or self.event_rho_delta <= 0.0
            or self.event_epsilon_delta <= 0.0
        ):
            raise ValueError("Bilateral Dynamic Event parameters have invalid signs")
        self.ewma_beta = float(ewma_beta)
        if not 0.0 < self.ewma_beta <= 1.0:
            raise ValueError("EWMA beta must be within (0, 1]")

        self.references = {
            "l": HealthyGaitReference(reference_result, "l"),
            "r": HealthyGaitReference(reference_result, "r"),
        }
        self.nominal_model = PlanarFloatingBaseExoModel.from_mujoco(mujoco_model)
        fixed_model = CurrentExoNominalModel()
        masses = [
            fixed_model.evaluate(
                np.asarray(self.references["l"].spline(float(phase))),
                np.zeros(2),
                0.0,
            ).mass
            for phase in np.linspace(0.0, 1.0, REFERENCE_PHASE_POINTS)
        ]
        self.ken_side = Z_DAMPING_RATE_S_INV * np.mean(np.stack(masses), axis=0)
        self.ken_4 = np.zeros((4, 4), dtype=float)
        self.ken_4[:2, :2] = self.ken_side
        self.ken_4[2:, 2:] = self.ken_side
        self.reset()

    @staticmethod
    def _fresh_side(reference: HealthyGaitReference) -> dict[str, Any]:
        return {
            "previous_heel": None,
            "last_strike_time": None,
            "period": float(reference.median_period_s),
            "reference_valid": False,
            "strike_count": 0,
            "lambda": None,
            "t1": None,
            "critic_weight": np.zeros(CRITIC_WEIGHT_DIMENSION, dtype=float),
            "previous_mu": None,
            "sampled_z": None,
            "eta": EVENT_ETA_INITIAL,
            "event_count": 0,
            "event_samples": 0,
            "event_age_steps": 0,
            "last_event_time": None,
            "last_event_sample": None,
        }

    def reset(self) -> None:
        self.side = {
            name: self._fresh_side(reference)
            for name, reference in self.references.items()
        }
        self._engagement_start_time_s: float | None = None
        self.last_diagnostics: dict[str, Any] = {}

    def _reference(
        self, side: str, time_s: float, contact: dict[str, Any]
    ) -> tuple[float | str, np.ndarray | None, np.ndarray | None, np.ndarray | None, bool]:
        state = self.side[side]
        heel = bool(contact["left_heel" if side == "l" else "right_heel"])
        strike = False
        if state["previous_heel"] is not None and not state["previous_heel"] and heel:
            elapsed = (
                math.inf
                if state["last_strike_time"] is None
                else time_s - float(state["last_strike_time"])
            )
            if elapsed >= HEEL_STRIKE_REFRACTORY_S:
                if state["last_strike_time"] is None:
                    strike = True
                    state["last_strike_time"] = time_s
                    state["reference_valid"] = True
                    state["strike_count"] += 1
                elif MIN_STRIDE_PERIOD_S <= elapsed <= MAX_STRIDE_PERIOD_S:
                    strike = True
                    state["period"] = (
                        (1.0 - self.ewma_beta) * float(state["period"])
                        + self.ewma_beta * elapsed
                    )
                    state["last_strike_time"] = time_s
                    state["strike_count"] += 1
        state["previous_heel"] = heel
        if not state["reference_valid"] or state["last_strike_time"] is None:
            return "", None, None, None, strike
        phase = float(
            np.clip(
                (time_s - float(state["last_strike_time"])) / float(state["period"]),
                0.0,
                1.0,
            )
        )
        qd, qdotd, qddotd = self.references[side].evaluate(
            phase, float(state["period"])
        )
        return phase, qd, qdotd, qddotd, strike

    def _side_transform(
        self,
        side: str,
        q: np.ndarray,
        qdot: np.ndarray,
        qacc: np.ndarray,
        tau_interaction: np.ndarray,
        activation: dict[str, Any],
        reference_values: tuple[
            float | str, np.ndarray | None, np.ndarray | None, np.ndarray | None, bool
        ],
        dt: float,
    ) -> dict[str, Any]:
        state = self.side[side]
        phase, qd, qdotd, qddotd, strike = reference_values
        names = [f"{family}_{side}" for family in INTENTION_MUSCLE_FAMILIES]
        muscle_values = {
            name: _require_finite(activation[name], f"activation {name}")
            for name in names
        }
        activation_proxy = float(
            np.sqrt(np.mean(np.square(np.asarray(list(muscle_values.values())))))
        )
        intention_mode = getattr(self, "intention_mode", "live_activation")
        if intention_mode == "constant_one":
            # [EXPERIMENTAL ABLATION] Paper-equivalent constant activation.
            # Keep activation_proxy as the real MyoAssist data.act RMS; only
            # the scheduling value entering Kr and phi(z,a) is forced to one.
            a_sched = 1.0
        else:
            a_sched = float(np.clip(activation_proxy, 0.0, 1.0))
        stiffness = STIFFNESS_MIN_NM_PER_RAD + a_sched * (
            STIFFNESS_MAX_NM_PER_RAD - STIFFNESS_MIN_NM_PER_RAD
        )
        result: dict[str, Any] = {
            "phase": phase,
            "qd": qd,
            "qdotd": qdotd,
            "qddotd": qddotd,
            "strike": strike,
            "activation_proxy": activation_proxy,
            "a_sched": a_sched,
            "muscles": muscle_values,
            "stiffness": stiffness,
            "valid": False,
            "qdotr": None,
            "qddotr": None,
            "z": None,
            "error": None,
            "error_velocity": None,
            "lambda": None,
            "gamma": None,
            "t1": None,
        }
        if qd is None or qdotd is None or qddotd is None:
            return result
        error = q - qd
        error_velocity = qdot - qdotd
        damping_over_inertia = DESIRED_DAMPING_NMS_PER_RAD / DESIRED_INERTIA_KGM2
        discriminant = damping_over_inertia**2 - 4.0 * stiffness / DESIRED_INERTIA_KGM2
        if state["lambda"] is None or state["t1"] is None:
            if np.all(discriminant > 0.0):
                state["lambda"] = 0.5 * (
                    damping_over_inertia - np.sqrt(discriminant)
                )
                state["t1"] = np.zeros(2, dtype=float)
        if state["lambda"] is None or state["t1"] is None:
            return result
        lambda_value = np.asarray(state["lambda"], dtype=float).copy()
        t1 = np.asarray(state["t1"], dtype=float).copy()
        gamma = damping_over_inertia - lambda_value
        dot_lambda = stiffness / DESIRED_INERTIA_KGM2 - gamma * lambda_value
        dot_t1 = tau_interaction / DESIRED_INERTIA_KGM2 - gamma * t1
        qdotr = qdotd - lambda_value * error + t1
        z = qdot - qdotr
        qddotr = qddotd - dot_lambda * error - lambda_value * error_velocity + dot_t1
        next_lambda = lambda_value + dt * dot_lambda
        next_t1 = t1 + dt * dot_t1
        finite = np.concatenate(
            (lambda_value, gamma, dot_lambda, t1, dot_t1, qdotr, z, qddotr, next_lambda, next_t1)
        )
        valid = bool(
            np.isfinite(finite).all()
            and np.all(lambda_value > 0.0)
            and np.all(gamma > 0.0)
            and np.all(next_lambda > 0.0)
            and np.all(damping_over_inertia - next_lambda > 0.0)
        )
        if not valid:
            state["lambda"] = None
            state["t1"] = None
            return result
        state["lambda"] = next_lambda
        state["t1"] = next_t1
        result.update(
            {
                "valid": True,
                "qdotr": qdotr,
                "qddotr": qddotr,
                "z": z,
                "error": error,
                "error_velocity": error_velocity,
                "lambda": lambda_value,
                "gamma": gamma,
                "t1": t1,
                "transform_mismatch": (
                    qacc
                    - qddotd
                    + damping_over_inertia * error_velocity
                    + stiffness / DESIRED_INERTIA_KGM2 * error
                    - tau_interaction / DESIRED_INERTIA_KGM2
                    - (qacc - qddotr + gamma * z)
                ),
            }
        )
        return result

    def _adp(
        self,
        side: str,
        transform: dict[str, Any],
        g_local: np.ndarray,
        qacc: np.ndarray,
        dt: float,
        time_s: float,
    ) -> dict[str, Any]:
        state = self.side[side]
        weight = np.asarray(state["critic_weight"], dtype=float).copy()
        z = np.asarray(transform["z"], dtype=float)
        qddotr = np.asarray(transform["qddotr"], dtype=float)
        features, jacobian = _critic_basis_b6(z, float(transform["a_sched"]))
        grad_v = jacobian @ weight
        mu_continuous = -0.5 * (CRITIC_R_INV @ g_local.T @ grad_v)
        sampled_z = z.copy()
        triggered = False
        reason = EVENT_REASON_NONE
        inter_event_s: float | str = ""
        event_age_s: float | str = ""
        eta_logged = float(state["eta"])
        threshold_sq: float | str = ""
        delta_norm_sq: float | str = ""
        if self.event_trigger_enabled:
            state["event_samples"] += 1
            decision = _event_trigger_decision(
                state["sampled_z"],
                z,
                eta_logged,
                xi=self.event_xi,
                rho_delta=self.event_rho_delta,
                epsilon_delta=self.event_epsilon_delta,
            )
            sampled_z = decision.sampled_z_rad_s
            state["sampled_z"] = sampled_z.copy()
            triggered = decision.triggered
            reason = decision.reason_code
            threshold_sq = decision.threshold_sq
            delta_norm_sq = decision.delta_norm_sq_pretrigger
            if triggered:
                if state["last_event_time"] is not None:
                    inter_event_s = time_s - float(state["last_event_time"])
                state["last_event_time"] = time_s
                state["event_age_steps"] = 0
                state["event_count"] += 1
            else:
                state["event_age_steps"] += 1
            event_age_s = state["event_age_steps"] * dt
            _, policy_jacobian = _critic_basis_b6(
                sampled_z, float(transform["a_sched"])
            )
            mu = -0.5 * (CRITIC_R_INV @ g_local.T @ (policy_jacobian @ weight))
            state["eta"], _, _ = _event_eta_exact_zoh_step(
                eta_logged,
                float(z @ z),
                dt,
                lambda_eta=self.event_lambda_eta,
                sigma_eta=self.event_sigma_eta,
            )
        else:
            mu = mu_continuous

        bellman: float | str = ""
        update_norm: float | str = ""
        if state["previous_mu"] is not None:
            previous_mu = np.asarray(state["previous_mu"], dtype=float)
            z_dot = qacc - qddotr
            f_hat = z_dot - g_local @ previous_mu
            direction = f_hat + g_local @ mu
            bellman = float(
                grad_v @ direction + z @ CRITIC_Q @ z + mu @ CRITIC_R @ mu
            )
            regressor = jacobian.T @ direction
            update = (
                -dt
                * CRITIC_KAPPA
                * regressor
                * bellman
                / (1.0 + float(regressor @ regressor)) ** 2
            )
            weight = weight + update
            update_norm = float(np.linalg.norm(update))
            if not np.isfinite(np.concatenate((mu, weight))).all() or np.linalg.norm(weight) >= CRITIC_WEIGHT_NORM_LIMIT:
                raise RuntimeError(f"Bilateral {side} Critic numerical watchdog failed")
            state["critic_weight"] = weight
        state["previous_mu"] = mu.copy()
        return {
            "mu": mu,
            "mu_continuous": mu_continuous,
            "weight": weight,
            "weight_norm": float(np.linalg.norm(weight)),
            "bellman": bellman,
            "update_norm": update_norm,
            "triggered": triggered,
            "event_reason": reason,
            "event_count": int(state["event_count"]),
            "event_samples": int(state["event_samples"]),
            "event_fraction": (
                float(state["event_count"] / state["event_samples"])
                if state["event_samples"]
                else 0.0
            ),
            "inter_event_s": inter_event_s,
            "event_age_s": event_age_s,
            "eta": eta_logged,
            "threshold_sq": threshold_sq,
            "delta_norm_sq": delta_norm_sq,
            "sampled_z": sampled_z,
        }

    @staticmethod
    def _component(value: Any, index: int) -> float | str:
        if value is None:
            return ""
        return float(np.asarray(value, dtype=float)[index])

    def compute(self, observation: dict[str, object], dt: float) -> dict[str, float]:
        control_dt = _require_finite(dt, "control dt")
        if not math.isclose(control_dt, IMPEDANCE_UPDATE_DT_S, abs_tol=1.0e-12):
            raise ValueError("Bilateral YAC requires the frozen 100 Hz control step")
        time_s = _require_finite(observation["time"], "observation time")
        contact = observation.get("contact")
        muscles = observation.get("muscles")
        exo = observation.get("exoskeleton")
        if not isinstance(contact, dict) or not isinstance(muscles, dict) or not isinstance(exo, dict):
            raise KeyError("Bilateral YAC observation is incomplete")
        activation = muscles.get("activation")
        dynamics = exo.get("bilateral_dynamics")
        coupling = exo.get("coupling_qfrc_nm")
        if not isinstance(activation, dict) or not isinstance(dynamics, dict) or not isinstance(coupling, dict):
            raise KeyError("Bilateral activation/dynamics/coupling observation is missing")

        q = np.asarray(dynamics["q_exo"], dtype=float).reshape(7)
        qvel = np.asarray(dynamics["qvel_exo"], dtype=float).reshape(7)
        external = np.asarray(dynamics["external_generalized_force"], dtype=float).reshape(7)
        jacobian = np.asarray(dynamics["constraint_jacobian"], dtype=float).reshape(-1, 7)
        regularization = np.asarray(dynamics["constraint_regularization"], dtype=float).reshape(-1)
        reference_acceleration = np.asarray(
            dynamics["constraint_reference_acceleration"], dtype=float
        ).reshape(-1)
        q4 = q[3:7]
        qdot4 = qvel[3:7]
        qacc4 = np.asarray(
            (
                exo["hip_l"]["acceleration_rad_s2"],
                exo["knee_l"]["acceleration_rad_s2"],
                exo["hip_r"]["acceleration_rad_s2"],
                exo["knee_r"]["acceleration_rad_s2"],
            ),
            dtype=float,
        )
        references = {
            side: self._reference(side, time_s, contact) for side in ("l", "r")
        }
        transforms: dict[str, dict[str, Any]] = {}
        for side, offset in (("l", 0), ("r", 2)):
            transforms[side] = self._side_transform(
                side,
                q4[offset : offset + 2],
                qdot4[offset : offset + 2],
                qacc4[offset : offset + 2],
                np.asarray(
                    (coupling[f"hip_{side}"], coupling[f"knee_{side}"]),
                    dtype=float,
                ),
                activation,
                references[side],
                control_dt,
            )

        ready = bool(all(transforms[side]["valid"] for side in ("l", "r")))
        u0 = np.zeros(4, dtype=float)
        mu = np.zeros(4, dtype=float)
        command = np.zeros(4, dtype=float)
        meff: np.ndarray | None = None
        beff: np.ndarray | None = None
        g4: np.ndarray | None = None
        active_count = 0
        adp = {"l": None, "r": None}
        ramp_factor = 0.0
        if ready:
            qddotr4 = np.concatenate(
                (transforms["l"]["qddotr"], transforms["r"]["qddotr"])
            )
            z4 = np.concatenate((transforms["l"]["z"], transforms["r"]["z"]))
            inverse = self.nominal_model.bilateral_unilateral_inverse_dynamics(
                q,
                qvel,
                external,
                qddotr4,
                jacobian,
                regularization,
                reference_acceleration,
            )
            meff = np.asarray(inverse["M_eff"], dtype=float)
            beff = np.asarray(inverse["b_eff"], dtype=float)
            active_count = len(inverse["active_indices"])
            u0 = meff @ qddotr4 + beff - self.ken_4 @ z4
            g4 = np.linalg.inv(meff)
            if self.mode == BILATERAL_FULL_YAC_MODE:
                adp["l"] = self._adp(
                    "l", transforms["l"], g4[:2, :2], qacc4[:2], control_dt, time_s
                )
                adp["r"] = self._adp(
                    "r", transforms["r"], g4[2:, 2:], qacc4[2:], control_dt, time_s
                )
                mu = np.concatenate((adp["l"]["mu"], adp["r"]["mu"]))
                ramp_factor = 1.0
                command = u0 + mu
            else:
                if self._engagement_start_time_s is None:
                    self._engagement_start_time_s = time_s
                ramp_factor = float(
                    np.clip(
                        (time_s - self._engagement_start_time_s)
                        / ENGAGEMENT_RAMP_DURATION_S,
                        0.0,
                        1.0,
                    )
                )
                command = ramp_factor * u0
                for side in ("l", "r"):
                    self.side[side]["previous_mu"] = np.zeros(2, dtype=float)
        else:
            for side in ("l", "r"):
                self.side[side]["previous_mu"] = None
                self.side[side]["sampled_z"] = None
                self.side[side]["eta"] = EVENT_ETA_INITIAL

        if not np.isfinite(np.concatenate((u0, mu, command))).all():
            raise FloatingPointError("Bilateral YAC torque path contains NaN or Inf")

        diagnostics: dict[str, Any] = {
            "time_s": time_s,
            "mode": self.mode,
            "controller_valid": ready,
            "reference_valid": ready,
            "ramp_factor": ramp_factor,
            "active_constraint_count": active_count,
            "measured_root_acceleration_used": False,
            "paper_tau_e_cancellation_used": False,
            "delayed_root_overlay_used": False,
            "u0_formula": "M_eff_4*qdd_r_4+b_eff_4-Ken_4*z_4",
            "adp_architecture": "two_mirrored_2d_critics",
        }
        labels = ("hip_l", "knee_l", "hip_r", "knee_r")
        for index, label in enumerate(labels):
            diagnostics[f"u0_{label}_nm"] = float(u0[index])
            diagnostics[f"mu_{label}_nm"] = float(mu[index])
            diagnostics[f"u_command_{label}_nm"] = float(command[index])
            diagnostics[f"q_exo_{label}_rad"] = float(q4[index])
            diagnostics[f"qdot_exo_{label}_rad_s"] = float(qdot4[index])
            diagnostics[f"qacc_exo_{label}_rad_s2"] = float(qacc4[index])
        if meff is None or g4 is None or beff is None:
            for row in range(4):
                diagnostics[f"b_eff_{row}"] = ""
                for column in range(4):
                    diagnostics[f"M_eff_{row}{column}"] = ""
                    diagnostics[f"g_4_{row}{column}"] = ""
            for name in ("g_ll_fro", "g_lr_fro", "g_rl_fro", "g_rr_fro"):
                diagnostics[name] = ""
        else:
            for row in range(4):
                diagnostics[f"b_eff_{row}"] = float(beff[row])
                for column in range(4):
                    diagnostics[f"M_eff_{row}{column}"] = float(meff[row, column])
                    diagnostics[f"g_4_{row}{column}"] = float(g4[row, column])
            diagnostics.update(
                {
                    "g_ll_fro": float(np.linalg.norm(g4[:2, :2])),
                    "g_lr_fro": float(np.linalg.norm(g4[:2, 2:])),
                    "g_rl_fro": float(np.linalg.norm(g4[2:, :2])),
                    "g_rr_fro": float(np.linalg.norm(g4[2:, 2:])),
                }
            )

        for side in ("l", "r"):
            item = transforms[side]
            diagnostics.update(
                {
                    f"phase_{side}": item["phase"],
                    f"stride_period_{side}_s": float(self.side[side]["period"]),
                    f"heel_strike_{side}": bool(item["strike"]),
                    f"heel_strike_count_{side}": int(self.side[side]["strike_count"]),
                    f"activation_proxy_{side}": float(item["activation_proxy"]),
                    f"a_sched_{side}": float(item["a_sched"]),
                    f"transform_valid_{side}": bool(item["valid"]),
                }
            )
            for muscle_name, value in item["muscles"].items():
                diagnostics[f"activation_{muscle_name}"] = float(value)
            offset = 0 if side == "l" else 2
            for joint, joint_index in (("hip", 0), ("knee", 1)):
                diagnostics[f"q_ref_{joint}_{side}_rad"] = self._component(item["qd"], joint_index)
                diagnostics[f"qdot_ref_{joint}_{side}_rad_s"] = self._component(item["qdotd"], joint_index)
                diagnostics[f"qddot_ref_{joint}_{side}_rad_s2"] = self._component(item["qddotd"], joint_index)
                diagnostics[f"error_{joint}_{side}_rad"] = self._component(item["error"], joint_index)
                diagnostics[f"error_velocity_{joint}_{side}_rad_s"] = self._component(item["error_velocity"], joint_index)
                diagnostics[f"qddot_r_{joint}_{side}_rad_s2"] = self._component(item["qddotr"], joint_index)
                diagnostics[f"z_{joint}_{side}_rad_s"] = self._component(item["z"], joint_index)
                diagnostics[f"Kr_{joint}_{side}_nm_per_rad"] = float(item["stiffness"][joint_index])
                diagnostics[f"Lambda_{joint}_{side}_s_inv"] = self._component(item["lambda"], joint_index)
                diagnostics[f"Gamma_{joint}_{side}_s_inv"] = self._component(item["gamma"], joint_index)
                diagnostics[f"t1_{joint}_{side}_rad_s"] = self._component(item["t1"], joint_index)
                diagnostics[f"transform_mismatch_{joint}_{side}"] = self._component(item.get("transform_mismatch"), joint_index)
            critic = adp[side]
            if critic is None:
                weight = np.asarray(self.side[side]["critic_weight"], dtype=float)
                diagnostics.update(
                    {
                        f"critic_weight_norm_{side}": float(np.linalg.norm(weight)),
                        f"bellman_error_{side}": "",
                        f"critic_update_norm_{side}": "",
                        f"event_triggered_{side}": False,
                        f"event_reason_{side}": EVENT_REASON_NONE,
                        f"event_count_{side}": int(self.side[side]["event_count"]),
                        f"event_fraction_{side}": 0.0,
                        f"inter_event_{side}_s": "",
                        f"event_age_{side}_s": "",
                        f"event_eta_{side}": float(self.side[side]["eta"]),
                        f"event_threshold_sq_{side}": "",
                        f"event_delta_norm_sq_{side}": "",
                    }
                )
            else:
                weight = critic["weight"]
                diagnostics.update(
                    {
                        f"critic_weight_norm_{side}": critic["weight_norm"],
                        f"bellman_error_{side}": critic["bellman"],
                        f"critic_update_norm_{side}": critic["update_norm"],
                        f"event_triggered_{side}": critic["triggered"],
                        f"event_reason_{side}": critic["event_reason"],
                        f"event_count_{side}": critic["event_count"],
                        f"event_fraction_{side}": critic["event_fraction"],
                        f"inter_event_{side}_s": critic["inter_event_s"],
                        f"event_age_{side}_s": critic["event_age_s"],
                        f"event_eta_{side}": critic["eta"],
                        f"event_threshold_sq_{side}": critic["threshold_sq"],
                        f"event_delta_norm_sq_{side}": critic["delta_norm_sq"],
                    }
                )
            for weight_index in range(CRITIC_WEIGHT_DIMENSION):
                diagnostics[f"critic_weight_{weight_index}_{side}"] = float(weight[weight_index])
        self.last_diagnostics = diagnostics
        torques = dict(ZERO_TORQUES)
        torques.update(
            {
                "HipExo_L": float(command[0]),
                "KneeExo_L": float(command[1]),
                "HipExo_R": float(command[2]),
                "KneeExo_R": float(command[3]),
            }
        )
        return torques

# ============================================================
# Validated analytic 7DOF floating-base Exoskeleton model
# (co-located here so controllers/YAC.py is the sole controller source)
# ============================================================
COORDINATE_NAMES = (
    "exo_root_forward",
    "exo_root_vertical",
    "exo_root_pitch",
    "exo_hip_l",
    "exo_knee_l",
)
FULL_COORDINATE_NAMES = (
    "exo_root_forward",
    "exo_root_vertical",
    "exo_root_pitch",
    "exo_hip_l",
    "exo_knee_l",
    "exo_hip_r",
    "exo_knee_r",
)
UNACTUATED_INDICES = np.asarray((0, 1, 2, 5, 6), dtype=int)
ACTUATED_INDICES = np.asarray((3, 4), dtype=int)
BILATERAL_UNACTUATED_INDICES = np.asarray((0, 1, 2), dtype=int)
BILATERAL_ACTUATED_INDICES = np.asarray((3, 4, 5, 6), dtype=int)
ROOT_DIMENSION = 3
JOINT_DIMENSION = 2


def _rotation(angle_rad: float) -> np.ndarray:
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    return np.asarray(((cosine, -sine), (sine, cosine)), dtype=float)


SKEW_2D = np.asarray(((0.0, -1.0), (1.0, 0.0)), dtype=float)


def _body_id(model: Any, name: str) -> int:
    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
    if body_id < 0:
        raise ValueError(f"Required exoskeleton body is missing: {name}")
    return body_id


def _joint_id(model: Any, name: str) -> int:
    joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
    if joint_id < 0:
        raise ValueError(f"Required exoskeleton joint is missing: {name}")
    return joint_id


def _planar_inertia_zz(model: Any, body_id: int) -> float:
    rotation = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(rotation, np.asarray(model.body_iquat[body_id]))
    rotation = rotation.reshape(3, 3)
    inertia_body = rotation @ np.diag(model.body_inertia[body_id]) @ rotation.T
    value = float(inertia_body[2, 2])
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("Planar body inertia must be finite and positive")
    return value


def _combine_planar_bodies(
    masses_kg: tuple[float, ...],
    com_positions_m: tuple[np.ndarray, ...],
    inertias_zz_kgm2: tuple[float, ...],
) -> tuple[float, np.ndarray, float]:
    masses = np.asarray(masses_kg, dtype=float)
    positions = np.asarray(com_positions_m, dtype=float)
    inertias = np.asarray(inertias_zz_kgm2, dtype=float)
    total_mass = float(np.sum(masses))
    combined_com = np.sum(masses[:, None] * positions, axis=0) / total_mass
    offsets = positions - combined_com
    combined_inertia = float(
        np.sum(inertias + masses * np.sum(offsets * offsets, axis=1))
    )
    return total_mass, combined_com, combined_inertia


@dataclass(frozen=True)
class FloatingBaseParameters:
    root_mass_kg: float
    root_com_m: np.ndarray
    root_inertia_zz_kgm2: float
    hip_origin_from_root_m: np.ndarray
    link1_mass_kg: float
    link1_com_from_hip_m: np.ndarray
    link1_inertia_zz_kgm2: float
    knee_origin_from_hip_m: np.ndarray
    link2_mass_kg: float
    link2_com_from_knee_m: np.ndarray
    link2_inertia_zz_kgm2: float
    root_translation_basis: np.ndarray
    rotation_signs: np.ndarray
    armature: np.ndarray


class PlanarFloatingBaseLeftExoModel:
    """Five-coordinate planar root + left hip/knee rigid-body mass model."""

    def __init__(self, parameters: FloatingBaseParameters) -> None:
        self.parameters = parameters

    @classmethod
    def from_mujoco(cls, model: Any) -> "PlanarFloatingBaseLeftExoModel":
        """Extract physical parameters from a compiled model, never its qM."""
        root_id = _body_id(model, "exo_pelvis_frame")
        hip_module_id = _body_id(model, "exo_l_hip_module")
        thigh_id = _body_id(model, "exo_l_thigh_bar")
        knee_module_id = _body_id(model, "exo_l_knee_module")
        shank_id = _body_id(model, "exo_l_shank_bar")

        link1 = _combine_planar_bodies(
            (
                float(model.body_mass[hip_module_id]),
                float(model.body_mass[thigh_id]),
            ),
            (
                np.asarray(model.body_ipos[hip_module_id, :2], dtype=float),
                np.asarray(
                    model.body_pos[thigh_id, :2] + model.body_ipos[thigh_id, :2],
                    dtype=float,
                ),
            ),
            (
                _planar_inertia_zz(model, hip_module_id),
                _planar_inertia_zz(model, thigh_id),
            ),
        )
        link2 = _combine_planar_bodies(
            (
                float(model.body_mass[knee_module_id]),
                float(model.body_mass[shank_id]),
            ),
            (
                np.asarray(model.body_ipos[knee_module_id, :2], dtype=float),
                np.asarray(
                    model.body_pos[shank_id, :2] + model.body_ipos[shank_id, :2],
                    dtype=float,
                ),
            ),
            (
                _planar_inertia_zz(model, knee_module_id),
                _planar_inertia_zz(model, shank_id),
            ),
        )

        joint_ids = {name: _joint_id(model, name) for name in COORDINATE_NAMES}
        forward_axis = np.asarray(
            model.jnt_axis[joint_ids["exo_root_forward"]], dtype=float
        )
        vertical_axis = np.asarray(
            model.jnt_axis[joint_ids["exo_root_vertical"]], dtype=float
        )
        if (
            abs(float(forward_axis[2])) > 1.0e-12
            or abs(float(vertical_axis[2])) > 1.0e-12
        ):
            raise ValueError("Root slide axes must lie in the exoskeleton plane")
        translation_basis = np.column_stack((forward_axis[:2], vertical_axis[:2]))
        if not np.allclose(
            translation_basis.T @ translation_basis,
            np.eye(2),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("Root slide axes must form an orthonormal planar basis")

        rotation_signs = []
        for name in ("exo_root_pitch", "exo_hip_l", "exo_knee_l"):
            axis = np.asarray(model.jnt_axis[joint_ids[name]], dtype=float)
            if not np.allclose(axis[:2], 0.0, rtol=0.0, atol=1.0e-12) or not np.isclose(
                abs(float(axis[2])), 1.0, rtol=0.0, atol=1.0e-12
            ):
                raise ValueError(f"{name} must rotate about the planar normal")
            rotation_signs.append(float(axis[2]))

        dof_addresses = np.asarray(
            [int(model.jnt_dofadr[joint_ids[name]]) for name in COORDINATE_NAMES],
            dtype=int,
        )
        parameters = FloatingBaseParameters(
            root_mass_kg=float(model.body_mass[root_id]),
            root_com_m=np.asarray(model.body_ipos[root_id, :2], dtype=float).copy(),
            root_inertia_zz_kgm2=_planar_inertia_zz(model, root_id),
            hip_origin_from_root_m=np.asarray(
                model.body_pos[hip_module_id, :2], dtype=float
            ).copy(),
            link1_mass_kg=link1[0],
            link1_com_from_hip_m=link1[1].copy(),
            link1_inertia_zz_kgm2=link1[2],
            knee_origin_from_hip_m=np.asarray(
                model.body_pos[knee_module_id, :2], dtype=float
            ).copy(),
            link2_mass_kg=link2[0],
            link2_com_from_knee_m=link2[1].copy(),
            link2_inertia_zz_kgm2=link2[2],
            root_translation_basis=translation_basis,
            rotation_signs=np.asarray(rotation_signs, dtype=float),
            armature=np.asarray(model.dof_armature[dof_addresses], dtype=float).copy(),
        )
        return cls(parameters)

    @staticmethod
    def _add_body(
        mass_matrix: np.ndarray,
        mass_kg: float,
        inertia_zz_kgm2: float,
        linear_jacobian: np.ndarray,
        angular_jacobian: np.ndarray,
    ) -> None:
        mass_matrix += mass_kg * linear_jacobian.T @ linear_jacobian
        mass_matrix += inertia_zz_kgm2 * np.outer(
            angular_jacobian, angular_jacobian
        )

    def mass_matrix(self, q_fb: np.ndarray) -> np.ndarray:
        """Return analytic M(q) in [x, y, theta, hip, knee] coordinates."""
        q = np.asarray(q_fb, dtype=float).reshape(5)
        if not np.isfinite(q).all():
            raise FloatingPointError("Floating-base configuration contains NaN or Inf")
        p = self.parameters
        theta_sign, hip_sign, knee_sign = p.rotation_signs
        theta = theta_sign * float(q[2])
        hip = hip_sign * float(q[3])
        knee = knee_sign * float(q[4])
        root_rotation = _rotation(theta)
        hip_rotation = _rotation(hip)
        knee_rotation = _rotation(knee)

        mass = np.zeros((5, 5), dtype=float)
        translation_columns = p.root_translation_basis

        root_jacobian = np.zeros((2, 5), dtype=float)
        root_jacobian[:, :2] = translation_columns
        root_jacobian[:, 2] = (
            theta_sign * SKEW_2D @ root_rotation @ p.root_com_m
        )
        root_angular = np.asarray((0.0, 0.0, theta_sign, 0.0, 0.0))
        self._add_body(
            mass,
            p.root_mass_kg,
            p.root_inertia_zz_kgm2,
            root_jacobian,
            root_angular,
        )

        link1_inside = p.hip_origin_from_root_m + hip_rotation @ p.link1_com_from_hip_m
        link1_jacobian = np.zeros((2, 5), dtype=float)
        link1_jacobian[:, :2] = translation_columns
        link1_jacobian[:, 2] = theta_sign * SKEW_2D @ root_rotation @ link1_inside
        link1_jacobian[:, 3] = (
            hip_sign
            * root_rotation
            @ SKEW_2D
            @ hip_rotation
            @ p.link1_com_from_hip_m
        )
        link1_angular = np.asarray(
            (0.0, 0.0, theta_sign, hip_sign, 0.0), dtype=float
        )
        self._add_body(
            mass,
            p.link1_mass_kg,
            p.link1_inertia_zz_kgm2,
            link1_jacobian,
            link1_angular,
        )

        knee_to_com = p.knee_origin_from_hip_m + knee_rotation @ p.link2_com_from_knee_m
        link2_inside = p.hip_origin_from_root_m + hip_rotation @ knee_to_com
        link2_jacobian = np.zeros((2, 5), dtype=float)
        link2_jacobian[:, :2] = translation_columns
        link2_jacobian[:, 2] = theta_sign * SKEW_2D @ root_rotation @ link2_inside
        link2_jacobian[:, 3] = (
            hip_sign * root_rotation @ SKEW_2D @ hip_rotation @ knee_to_com
        )
        link2_jacobian[:, 4] = (
            knee_sign
            * root_rotation
            @ hip_rotation
            @ SKEW_2D
            @ knee_rotation
            @ p.link2_com_from_knee_m
        )
        link2_angular = np.asarray(
            (0.0, 0.0, theta_sign, hip_sign, knee_sign), dtype=float
        )
        self._add_body(
            mass,
            p.link2_mass_kg,
            p.link2_inertia_zz_kgm2,
            link2_jacobian,
            link2_angular,
        )

        mass += np.diag(p.armature)
        if not np.isfinite(mass).all():
            raise FloatingPointError("Floating-base mass matrix contains NaN or Inf")
        return mass

    def joint_partitions(self, q_fb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return M_jr (2x3) and M_jj (2x2)."""
        mass = self.mass_matrix(q_fb)
        return mass[ROOT_DIMENSION:, :ROOT_DIMENSION], mass[ROOT_DIMENSION:, ROOT_DIMENSION:]


@dataclass(frozen=True)
class PlanarLegParameters:
    hip_origin_from_root_m: np.ndarray
    link1_mass_kg: float
    link1_com_from_hip_m: np.ndarray
    link1_inertia_zz_kgm2: float
    knee_origin_from_hip_m: np.ndarray
    link2_mass_kg: float
    link2_com_from_knee_m: np.ndarray
    link2_inertia_zz_kgm2: float


@dataclass(frozen=True)
class FullPlanarExoParameters:
    root_mass_kg: float
    root_com_m: np.ndarray
    root_inertia_zz_kgm2: float
    left: PlanarLegParameters
    right: PlanarLegParameters
    root_translation_basis: np.ndarray
    reference_rotation_world_from_planar: np.ndarray
    rotation_signs: np.ndarray
    armature: np.ndarray
    gravity_world_m_s2: np.ndarray


class PlanarFloatingBaseExoModel:
    """Independent seven-coordinate planar exoskeleton dynamics model.

    Coordinates are ``[root_x, root_y, root_pitch, hip_l, knee_l,
    hip_r, knee_r]``.  The human is deliberately absent: M5 enters only as
    a known generalized external force on this independent exoskeleton tree.
    """

    def __init__(self, parameters: FullPlanarExoParameters) -> None:
        self.parameters = parameters

    @staticmethod
    def _leg_parameters(model: Any, side: str) -> PlanarLegParameters:
        hip_module_id = _body_id(model, f"exo_{side}_hip_module")
        thigh_id = _body_id(model, f"exo_{side}_thigh_bar")
        knee_module_id = _body_id(model, f"exo_{side}_knee_module")
        shank_id = _body_id(model, f"exo_{side}_shank_bar")
        link1 = _combine_planar_bodies(
            (float(model.body_mass[hip_module_id]), float(model.body_mass[thigh_id])),
            (
                np.asarray(model.body_ipos[hip_module_id, :2], dtype=float),
                np.asarray(
                    model.body_pos[thigh_id, :2] + model.body_ipos[thigh_id, :2],
                    dtype=float,
                ),
            ),
            (
                _planar_inertia_zz(model, hip_module_id),
                _planar_inertia_zz(model, thigh_id),
            ),
        )
        link2 = _combine_planar_bodies(
            (float(model.body_mass[knee_module_id]), float(model.body_mass[shank_id])),
            (
                np.asarray(model.body_ipos[knee_module_id, :2], dtype=float),
                np.asarray(
                    model.body_pos[shank_id, :2] + model.body_ipos[shank_id, :2],
                    dtype=float,
                ),
            ),
            (
                _planar_inertia_zz(model, knee_module_id),
                _planar_inertia_zz(model, shank_id),
            ),
        )
        return PlanarLegParameters(
            hip_origin_from_root_m=np.asarray(
                model.body_pos[hip_module_id, :2], dtype=float
            ).copy(),
            link1_mass_kg=link1[0],
            link1_com_from_hip_m=link1[1].copy(),
            link1_inertia_zz_kgm2=link1[2],
            knee_origin_from_hip_m=np.asarray(
                model.body_pos[knee_module_id, :2], dtype=float
            ).copy(),
            link2_mass_kg=link2[0],
            link2_com_from_knee_m=link2[1].copy(),
            link2_inertia_zz_kgm2=link2[2],
        )

    @classmethod
    def from_mujoco(cls, model: Any) -> "PlanarFloatingBaseExoModel":
        """Extract compiled XML parameters without reading ``mjData.qM``."""
        root_id = _body_id(model, "exo_pelvis_frame")
        joint_ids = {name: _joint_id(model, name) for name in FULL_COORDINATE_NAMES}
        translation_basis = np.column_stack(
            (
                np.asarray(model.jnt_axis[joint_ids["exo_root_forward"], :2]),
                np.asarray(model.jnt_axis[joint_ids["exo_root_vertical"], :2]),
            )
        ).astype(float)
        if not np.allclose(
            translation_basis.T @ translation_basis,
            np.eye(2),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("Root slide axes must form an orthonormal planar basis")

        rotation_names = (
            "exo_root_pitch",
            "exo_hip_l",
            "exo_knee_l",
            "exo_hip_r",
            "exo_knee_r",
        )
        rotation_signs = []
        for name in rotation_names:
            axis = np.asarray(model.jnt_axis[joint_ids[name]], dtype=float)
            if not np.allclose(axis[:2], 0.0, rtol=0.0, atol=1.0e-12) or not np.isclose(
                abs(float(axis[2])), 1.0, rtol=0.0, atol=1.0e-12
            ):
                raise ValueError(f"{name} must rotate about the planar normal")
            rotation_signs.append(float(axis[2]))

        reference_rotation = np.empty(9, dtype=float)
        mujoco.mju_quat2Mat(reference_rotation, np.asarray(model.body_quat[root_id]))
        dof_addresses = np.asarray(
            [int(model.jnt_dofadr[joint_ids[name]]) for name in FULL_COORDINATE_NAMES],
            dtype=int,
        )
        return cls(
            FullPlanarExoParameters(
                root_mass_kg=float(model.body_mass[root_id]),
                root_com_m=np.asarray(model.body_ipos[root_id, :2], dtype=float).copy(),
                root_inertia_zz_kgm2=_planar_inertia_zz(model, root_id),
                left=cls._leg_parameters(model, "l"),
                right=cls._leg_parameters(model, "r"),
                root_translation_basis=translation_basis,
                reference_rotation_world_from_planar=reference_rotation.reshape(3, 3),
                rotation_signs=np.asarray(rotation_signs, dtype=float),
                armature=np.asarray(model.dof_armature[dof_addresses], dtype=float).copy(),
                gravity_world_m_s2=np.asarray(model.opt.gravity, dtype=float).copy(),
            )
        )

    @staticmethod
    def _add_body_terms(
        mass: np.ndarray,
        bodies: list[tuple[float, float, np.ndarray, np.ndarray]],
        mass_kg: float,
        inertia_zz_kgm2: float,
        linear_jacobian: np.ndarray,
        angular_jacobian: np.ndarray,
    ) -> None:
        PlanarFloatingBaseLeftExoModel._add_body(
            mass, mass_kg, inertia_zz_kgm2, linear_jacobian, angular_jacobian
        )
        bodies.append(
            (mass_kg, inertia_zz_kgm2, linear_jacobian, angular_jacobian)
        )

    def _body_terms(
        self, q_exo: np.ndarray
    ) -> tuple[np.ndarray, list[tuple[float, float, np.ndarray, np.ndarray]]]:
        q = np.asarray(q_exo, dtype=float).reshape(7)
        if not np.isfinite(q).all():
            raise FloatingPointError("Seven-DOF configuration contains NaN or Inf")
        p = self.parameters
        theta_sign, hip_l_sign, knee_l_sign, hip_r_sign, knee_r_sign = (
            p.rotation_signs
        )
        root_rotation = _rotation(theta_sign * float(q[2]))
        mass = np.zeros((7, 7), dtype=float)
        bodies: list[tuple[float, float, np.ndarray, np.ndarray]] = []

        root_jacobian = np.zeros((2, 7), dtype=float)
        root_jacobian[:, :2] = p.root_translation_basis
        root_jacobian[:, 2] = (
            theta_sign * SKEW_2D @ root_rotation @ p.root_com_m
        )
        root_angular = np.zeros(7, dtype=float)
        root_angular[2] = theta_sign
        self._add_body_terms(
            mass,
            bodies,
            p.root_mass_kg,
            p.root_inertia_zz_kgm2,
            root_jacobian,
            root_angular,
        )

        for leg, hip_index, knee_index, hip_sign, knee_sign in (
            (p.left, 3, 4, hip_l_sign, knee_l_sign),
            (p.right, 5, 6, hip_r_sign, knee_r_sign),
        ):
            hip_rotation = _rotation(hip_sign * float(q[hip_index]))
            knee_rotation = _rotation(knee_sign * float(q[knee_index]))
            link1_inside = (
                leg.hip_origin_from_root_m
                + hip_rotation @ leg.link1_com_from_hip_m
            )
            link1_jacobian = np.zeros((2, 7), dtype=float)
            link1_jacobian[:, :2] = p.root_translation_basis
            link1_jacobian[:, 2] = (
                theta_sign * SKEW_2D @ root_rotation @ link1_inside
            )
            link1_jacobian[:, hip_index] = (
                hip_sign
                * root_rotation
                @ SKEW_2D
                @ hip_rotation
                @ leg.link1_com_from_hip_m
            )
            link1_angular = np.zeros(7, dtype=float)
            link1_angular[2] = theta_sign
            link1_angular[hip_index] = hip_sign
            self._add_body_terms(
                mass,
                bodies,
                leg.link1_mass_kg,
                leg.link1_inertia_zz_kgm2,
                link1_jacobian,
                link1_angular,
            )

            knee_to_com = (
                leg.knee_origin_from_hip_m
                + knee_rotation @ leg.link2_com_from_knee_m
            )
            link2_inside = (
                leg.hip_origin_from_root_m + hip_rotation @ knee_to_com
            )
            link2_jacobian = np.zeros((2, 7), dtype=float)
            link2_jacobian[:, :2] = p.root_translation_basis
            link2_jacobian[:, 2] = (
                theta_sign * SKEW_2D @ root_rotation @ link2_inside
            )
            link2_jacobian[:, hip_index] = (
                hip_sign * root_rotation @ SKEW_2D @ hip_rotation @ knee_to_com
            )
            link2_jacobian[:, knee_index] = (
                knee_sign
                * root_rotation
                @ hip_rotation
                @ SKEW_2D
                @ knee_rotation
                @ leg.link2_com_from_knee_m
            )
            link2_angular = link1_angular.copy()
            link2_angular[knee_index] = knee_sign
            self._add_body_terms(
                mass,
                bodies,
                leg.link2_mass_kg,
                leg.link2_inertia_zz_kgm2,
                link2_jacobian,
                link2_angular,
            )
        mass += np.diag(p.armature)
        return mass, bodies

    def mass_matrix(self, q_exo: np.ndarray) -> np.ndarray:
        """Return analytic ``M(q)`` in the documented seven-coordinate order."""
        mass, _ = self._body_terms(q_exo)
        if not np.isfinite(mass).all():
            raise FloatingPointError("Seven-DOF mass matrix contains NaN or Inf")
        return mass

    def bias_forces(
        self, q_exo: np.ndarray, qvel_exo: np.ndarray, *, derivative_step: float = 1.0e-6
    ) -> np.ndarray:
        """Return analytic-Lagrangian ``C(q,v)v + G(q)`` in generalized units."""
        q = np.asarray(q_exo, dtype=float).reshape(7)
        velocity = np.asarray(qvel_exo, dtype=float).reshape(7)
        if not np.isfinite(np.concatenate((q, velocity))).all():
            raise FloatingPointError("Seven-DOF state contains NaN or Inf")
        derivative = np.zeros((7, 7, 7), dtype=float)
        for coordinate in range(2, 7):
            plus = q.copy()
            minus = q.copy()
            plus[coordinate] += derivative_step
            minus[coordinate] -= derivative_step
            derivative[coordinate] = (
                self.mass_matrix(plus) - self.mass_matrix(minus)
            ) / (2.0 * derivative_step)

        coriolis = 0.5 * (
            np.einsum("kij,j,k->i", derivative, velocity, velocity)
            + np.einsum("jik,j,k->i", derivative, velocity, velocity)
            - np.einsum("ijk,j,k->i", derivative, velocity, velocity)
        )

        _, bodies = self._body_terms(q)
        gravity_planar = (
            self.parameters.reference_rotation_world_from_planar.T
            @ self.parameters.gravity_world_m_s2
        )[:2]
        gravity_bias = np.zeros(7, dtype=float)
        for mass_kg, _, linear_jacobian, _ in bodies:
            gravity_bias -= mass_kg * linear_jacobian.T @ gravity_planar
        result = coriolis + gravity_bias
        if not np.isfinite(result).all():
            raise FloatingPointError("Seven-DOF bias vector contains NaN or Inf")
        return result

    def reduced_dynamics(
        self,
        q_exo: np.ndarray,
        qvel_exo: np.ndarray,
        external_generalized_force: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``M_eff, b_eff`` after eliminating root and right-leg qacc.

        The force convention is ``M qdd + h = S_a u_a + f_external``.
        Consequently ``u_a = M_eff qdd_a + b_eff`` with
        ``b_eff = h_a - f_a + M_au M_uu^-1 (f_u - h_u)``.
        """
        force = np.asarray(external_generalized_force, dtype=float).reshape(7)
        mass = self.mass_matrix(q_exo)
        bias = self.bias_forces(q_exo, qvel_exo)
        muu = mass[np.ix_(UNACTUATED_INDICES, UNACTUATED_INDICES)]
        mua = mass[np.ix_(UNACTUATED_INDICES, ACTUATED_INDICES)]
        mau = mass[np.ix_(ACTUATED_INDICES, UNACTUATED_INDICES)]
        maa = mass[np.ix_(ACTUATED_INDICES, ACTUATED_INDICES)]
        solve_mua = np.linalg.solve(muu, mua)
        meff = maa - mau @ solve_mua
        beff = (
            bias[ACTUATED_INDICES]
            - force[ACTUATED_INDICES]
            + mau
            @ np.linalg.solve(
                muu, force[UNACTUATED_INDICES] - bias[UNACTUATED_INDICES]
            )
        )
        return meff, beff

    def constrained_reduced_dynamics(
        self,
        q_exo: np.ndarray,
        qvel_exo: np.ndarray,
        external_generalized_force: np.ndarray,
        constraint_jacobian: np.ndarray,
        constraint_regularization: np.ndarray,
        constraint_reference_acceleration: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the affine inverse dynamics including active soft constraints.

        MuJoCo's active constraint rows satisfy
        ``J qdd + R lambda = a_ref`` while the rigid-body equation is
        ``M qdd + h = S_a u_a + f_external + J.T lambda``.  Root and
        right-leg accelerations, together with ``lambda``, are solved inside
        this KKT reduction; none is a measured input.
        """
        force = np.asarray(external_generalized_force, dtype=float).reshape(7)
        jacobian = np.asarray(constraint_jacobian, dtype=float).reshape(-1, 7)
        reference = np.asarray(
            constraint_reference_acceleration, dtype=float
        ).reshape(-1)
        regularization = np.asarray(constraint_regularization, dtype=float)
        if regularization.ndim == 1:
            regularization = np.diag(regularization)
        count = jacobian.shape[0]
        if reference.shape != (count,) or regularization.shape != (count, count):
            raise ValueError("Constraint J, R and a_ref dimensions are inconsistent")
        if count == 0:
            return self.reduced_dynamics(q_exo, qvel_exo, force)

        mass = self.mass_matrix(q_exo)
        bias = self.bias_forces(q_exo, qvel_exo)
        muu = mass[np.ix_(UNACTUATED_INDICES, UNACTUATED_INDICES)]
        mua = mass[np.ix_(UNACTUATED_INDICES, ACTUATED_INDICES)]
        mau = mass[np.ix_(ACTUATED_INDICES, UNACTUATED_INDICES)]
        maa = mass[np.ix_(ACTUATED_INDICES, ACTUATED_INDICES)]
        ju = jacobian[:, UNACTUATED_INDICES]
        ja = jacobian[:, ACTUATED_INDICES]
        kkt = np.block([[muu, -ju.T], [ju, regularization]])
        affine_rhs = np.concatenate(
            (
                force[UNACTUATED_INDICES] - bias[UNACTUATED_INDICES],
                reference,
            )
        )
        acceleration_rhs = np.vstack((-mua, -ja))
        affine_solution = np.linalg.solve(kkt, affine_rhs)
        acceleration_solution = np.linalg.solve(kkt, acceleration_rhs)
        output_map = np.hstack((mau, -ja.T))
        meff = maa + output_map @ acceleration_solution
        beff = (
            bias[ACTUATED_INDICES]
            - force[ACTUATED_INDICES]
            + output_map @ affine_solution
        )
        if not np.isfinite(np.concatenate((meff.reshape(-1), beff))).all():
            raise FloatingPointError("Constrained reduced dynamics contains NaN or Inf")
        return meff, beff

    def unilateral_inverse_dynamics(
        self,
        q_exo: np.ndarray,
        qvel_exo: np.ndarray,
        external_generalized_force: np.ndarray,
        target_actuated_acceleration: np.ndarray,
        candidate_constraint_jacobian: np.ndarray,
        candidate_constraint_regularization: np.ndarray,
        candidate_constraint_reference_acceleration: np.ndarray,
        *,
        complementarity_tolerance: float = 1.0e-8,
    ) -> dict[str, Any]:
        """Solve the soft unilateral joint-limit active set explicitly.

        Candidate rows are state-derived MuJoCo limit rows.  Every subset is
        checked because this model has at most four Exo joint limits.  Active
        rows require ``lambda >= 0``; inactive rows require
        ``a_ref - J qdd <= 0``.  This avoids using measured unactuated qacc or
        a measured constraint force to choose the inverse-dynamics solution.
        """
        q = np.asarray(q_exo, dtype=float).reshape(7)
        qvel = np.asarray(qvel_exo, dtype=float).reshape(7)
        force = np.asarray(external_generalized_force, dtype=float).reshape(7)
        target = np.asarray(target_actuated_acceleration, dtype=float).reshape(2)
        jacobian = np.asarray(candidate_constraint_jacobian, dtype=float).reshape(-1, 7)
        reference = np.asarray(
            candidate_constraint_reference_acceleration, dtype=float
        ).reshape(-1)
        regularization = np.asarray(candidate_constraint_regularization, dtype=float)
        if regularization.ndim == 2:
            if regularization.shape != (jacobian.shape[0], jacobian.shape[0]):
                raise ValueError("Constraint regularization matrix has wrong shape")
            regularization = np.diag(regularization)
        regularization = regularization.reshape(-1)
        count = jacobian.shape[0]
        if reference.shape != (count,) or regularization.shape != (count,):
            raise ValueError("Candidate constraint dimensions are inconsistent")

        mass = self.mass_matrix(q)
        bias = self.bias_forces(q, qvel)
        muu = mass[np.ix_(UNACTUATED_INDICES, UNACTUATED_INDICES)]
        mua = mass[np.ix_(UNACTUATED_INDICES, ACTUATED_INDICES)]
        mau = mass[np.ix_(ACTUATED_INDICES, UNACTUATED_INDICES)]
        maa = mass[np.ix_(ACTUATED_INDICES, ACTUATED_INDICES)]
        valid_solutions: list[dict[str, Any]] = []
        best_invalid: dict[str, Any] | None = None

        for active_count in range(count + 1):
            for active_tuple in combinations(range(count), active_count):
                active = np.asarray(active_tuple, dtype=int)
                active_mask = np.zeros(count, dtype=bool)
                active_mask[active] = True
                selected_jacobian = jacobian[active]
                selected_regularization = regularization[active]
                selected_reference = reference[active]
                meff, beff = self.constrained_reduced_dynamics(
                    q,
                    qvel,
                    force,
                    selected_jacobian,
                    selected_regularization,
                    selected_reference,
                )
                torque = meff @ target + beff
                if active.size:
                    ju = selected_jacobian[:, UNACTUATED_INDICES]
                    ja = selected_jacobian[:, ACTUATED_INDICES]
                    kkt = np.block(
                        [[muu, -ju.T], [ju, np.diag(selected_regularization)]]
                    )
                    solution = np.linalg.solve(
                        kkt,
                        np.concatenate(
                            (
                                force[UNACTUATED_INDICES]
                                - bias[UNACTUATED_INDICES]
                                - mua @ target,
                                selected_reference - ja @ target,
                            )
                        ),
                    )
                    qdd_unactuated = solution[: UNACTUATED_INDICES.size]
                    active_force = solution[UNACTUATED_INDICES.size :]
                else:
                    qdd_unactuated = np.linalg.solve(
                        muu,
                        force[UNACTUATED_INDICES]
                        - bias[UNACTUATED_INDICES]
                        - mua @ target,
                    )
                    active_force = np.zeros(0)
                qdd = np.empty(7, dtype=float)
                qdd[UNACTUATED_INDICES] = qdd_unactuated
                qdd[ACTUATED_INDICES] = target
                residual = reference - jacobian @ qdd
                inactive_violation = (
                    float(np.max(residual[~active_mask]))
                    if np.any(~active_mask)
                    else -np.inf
                )
                active_violation = (
                    float(np.max(-active_force)) if active_force.size else -np.inf
                )
                violation = max(0.0, inactive_violation, active_violation)
                item = {
                    "torque": torque,
                    "M_eff": meff,
                    "b_eff": beff,
                    "qdd": qdd,
                    "constraint_force": np.zeros(count, dtype=float),
                    "active_indices": active,
                    "complementarity_violation": violation,
                }
                item["constraint_force"][active] = active_force
                if violation <= complementarity_tolerance:
                    valid_solutions.append(item)
                if best_invalid is None or violation < best_invalid["complementarity_violation"]:
                    best_invalid = item

        if not valid_solutions:
            raise RuntimeError(
                "No unilateral constraint active set satisfied complementarity; "
                f"minimum violation={best_invalid['complementarity_violation'] if best_invalid else np.nan}"
            )
        valid_solutions.sort(
            key=lambda item: (
                item["complementarity_violation"], len(item["active_indices"])
            )
        )
        return valid_solutions[0]


    def bilateral_reduced_dynamics(
        self,
        q_exo: np.ndarray,
        qvel_exo: np.ndarray,
        external_generalized_force: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the exact four-actuator Schur reduction.

        Only the three floating-root coordinates are eliminated.  Both legs
        remain in the actuated partition, so the returned 4x4 matrix retains
        the cross-leg coupling transmitted through the shared pelvis.
        """
        force = np.asarray(external_generalized_force, dtype=float).reshape(7)
        mass = self.mass_matrix(q_exo)
        bias = self.bias_forces(q_exo, qvel_exo)
        unactuated = BILATERAL_UNACTUATED_INDICES
        actuated = BILATERAL_ACTUATED_INDICES
        muu = mass[np.ix_(unactuated, unactuated)]
        mua = mass[np.ix_(unactuated, actuated)]
        mau = mass[np.ix_(actuated, unactuated)]
        maa = mass[np.ix_(actuated, actuated)]
        meff = maa - mau @ np.linalg.solve(muu, mua)
        beff = (
            bias[actuated]
            - force[actuated]
            + mau @ np.linalg.solve(muu, force[unactuated] - bias[unactuated])
        )
        if not np.isfinite(np.concatenate((meff.reshape(-1), beff))).all():
            raise FloatingPointError("Bilateral reduced dynamics contains NaN or Inf")
        return meff, beff

    def bilateral_constrained_reduced_dynamics(
        self,
        q_exo: np.ndarray,
        qvel_exo: np.ndarray,
        external_generalized_force: np.ndarray,
        constraint_jacobian: np.ndarray,
        constraint_regularization: np.ndarray,
        constraint_reference_acceleration: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the four-actuator affine inverse dynamics for fixed rows."""
        force = np.asarray(external_generalized_force, dtype=float).reshape(7)
        jacobian = np.asarray(constraint_jacobian, dtype=float).reshape(-1, 7)
        reference = np.asarray(
            constraint_reference_acceleration, dtype=float
        ).reshape(-1)
        regularization = np.asarray(constraint_regularization, dtype=float)
        if regularization.ndim == 1:
            regularization = np.diag(regularization)
        count = jacobian.shape[0]
        if reference.shape != (count,) or regularization.shape != (count, count):
            raise ValueError("Constraint J, R and a_ref dimensions are inconsistent")
        if count == 0:
            return self.bilateral_reduced_dynamics(q_exo, qvel_exo, force)

        mass = self.mass_matrix(q_exo)
        bias = self.bias_forces(q_exo, qvel_exo)
        unactuated = BILATERAL_UNACTUATED_INDICES
        actuated = BILATERAL_ACTUATED_INDICES
        muu = mass[np.ix_(unactuated, unactuated)]
        mua = mass[np.ix_(unactuated, actuated)]
        mau = mass[np.ix_(actuated, unactuated)]
        maa = mass[np.ix_(actuated, actuated)]
        ju = jacobian[:, unactuated]
        ja = jacobian[:, actuated]
        kkt = np.block([[muu, -ju.T], [ju, regularization]])
        affine_rhs = np.concatenate(
            (force[unactuated] - bias[unactuated], reference)
        )
        acceleration_rhs = np.vstack((-mua, -ja))
        affine_solution = np.linalg.solve(kkt, affine_rhs)
        acceleration_solution = np.linalg.solve(kkt, acceleration_rhs)
        output_map = np.hstack((mau, -ja.T))
        meff = maa + output_map @ acceleration_solution
        beff = bias[actuated] - force[actuated] + output_map @ affine_solution
        if not np.isfinite(np.concatenate((meff.reshape(-1), beff))).all():
            raise FloatingPointError(
                "Bilateral constrained reduced dynamics contains NaN or Inf"
            )
        return meff, beff

    def bilateral_unilateral_inverse_dynamics(
        self,
        q_exo: np.ndarray,
        qvel_exo: np.ndarray,
        external_generalized_force: np.ndarray,
        target_actuated_acceleration: np.ndarray,
        candidate_constraint_jacobian: np.ndarray,
        candidate_constraint_regularization: np.ndarray,
        candidate_constraint_reference_acceleration: np.ndarray,
        *,
        complementarity_tolerance: float = 1.0e-8,
    ) -> dict[str, Any]:
        """Select the four-joint soft-limit active set without measured qacc."""
        q = np.asarray(q_exo, dtype=float).reshape(7)
        qvel = np.asarray(qvel_exo, dtype=float).reshape(7)
        force = np.asarray(external_generalized_force, dtype=float).reshape(7)
        target = np.asarray(target_actuated_acceleration, dtype=float).reshape(4)
        jacobian = np.asarray(candidate_constraint_jacobian, dtype=float).reshape(-1, 7)
        reference = np.asarray(
            candidate_constraint_reference_acceleration, dtype=float
        ).reshape(-1)
        regularization = np.asarray(candidate_constraint_regularization, dtype=float)
        if regularization.ndim == 2:
            if regularization.shape != (jacobian.shape[0], jacobian.shape[0]):
                raise ValueError("Constraint regularization matrix has wrong shape")
            regularization = np.diag(regularization)
        regularization = regularization.reshape(-1)
        count = jacobian.shape[0]
        if reference.shape != (count,) or regularization.shape != (count,):
            raise ValueError("Candidate constraint dimensions are inconsistent")

        mass = self.mass_matrix(q)
        bias = self.bias_forces(q, qvel)
        unactuated = BILATERAL_UNACTUATED_INDICES
        actuated = BILATERAL_ACTUATED_INDICES
        muu = mass[np.ix_(unactuated, unactuated)]
        mua = mass[np.ix_(unactuated, actuated)]
        valid_solutions: list[dict[str, Any]] = []
        best_invalid: dict[str, Any] | None = None

        for active_count in range(count + 1):
            for active_tuple in combinations(range(count), active_count):
                active = np.asarray(active_tuple, dtype=int)
                active_mask = np.zeros(count, dtype=bool)
                active_mask[active] = True
                selected_jacobian = jacobian[active]
                selected_regularization = regularization[active]
                selected_reference = reference[active]
                meff, beff = self.bilateral_constrained_reduced_dynamics(
                    q,
                    qvel,
                    force,
                    selected_jacobian,
                    selected_regularization,
                    selected_reference,
                )
                torque = meff @ target + beff
                if active.size:
                    ju = selected_jacobian[:, unactuated]
                    ja = selected_jacobian[:, actuated]
                    kkt = np.block(
                        [[muu, -ju.T], [ju, np.diag(selected_regularization)]]
                    )
                    solution = np.linalg.solve(
                        kkt,
                        np.concatenate(
                            (
                                force[unactuated]
                                - bias[unactuated]
                                - mua @ target,
                                selected_reference - ja @ target,
                            )
                        ),
                    )
                    qdd_unactuated = solution[: unactuated.size]
                    active_force = solution[unactuated.size :]
                else:
                    qdd_unactuated = np.linalg.solve(
                        muu,
                        force[unactuated] - bias[unactuated] - mua @ target,
                    )
                    active_force = np.zeros(0)
                qdd = np.empty(7, dtype=float)
                qdd[unactuated] = qdd_unactuated
                qdd[actuated] = target
                residual = reference - jacobian @ qdd
                inactive_violation = (
                    float(np.max(residual[~active_mask]))
                    if np.any(~active_mask)
                    else -np.inf
                )
                active_violation = (
                    float(np.max(-active_force)) if active_force.size else -np.inf
                )
                violation = max(0.0, inactive_violation, active_violation)
                item = {
                    "torque": torque,
                    "M_eff": meff,
                    "b_eff": beff,
                    "qdd": qdd,
                    "constraint_force": np.zeros(count, dtype=float),
                    "active_indices": active,
                    "complementarity_violation": violation,
                }
                item["constraint_force"][active] = active_force
                if violation <= complementarity_tolerance:
                    valid_solutions.append(item)
                if best_invalid is None or violation < best_invalid[
                    "complementarity_violation"
                ]:
                    best_invalid = item

        if not valid_solutions:
            minimum = (
                best_invalid["complementarity_violation"]
                if best_invalid is not None
                else np.nan
            )
            raise RuntimeError(
                "No bilateral unilateral active set satisfied complementarity; "
                f"minimum violation={minimum}"
            )
        valid_solutions.sort(
            key=lambda item: (
                item["complementarity_violation"], len(item["active_indices"])
            )
        )
        return valid_solutions[0]


# ============================================================
# Final public bilateral YAC controller
# ============================================================
DEFAULT_MODE = "full_yac"
DEFAULT_DURATION_S = 20.0
DEFAULT_VIEWER = True
DEFAULT_VIDEO = True
DEFAULT_EVALUATION = True
DEFAULT_PAPER_COMPARISON = True
DEFAULT_PATHOLOGY = "bilateral_moderate60"
YAC_HRI_TRANSIENT_CUTOFF_S = 2.0
DEFAULT_HEALTHY_REFERENCE_RESULT = (
    Path(__file__).resolve().parents[1]
    / "results/experiments/20260822_214246__HUMAN_EXO_HEALTHY100_ZERO"
)
DEFAULT_EVENT_PARAMETERS = {
    "lambda_eta": 5.0,
    "sigma_eta": 0.025,
    "xi": 0.005,
    "rho_delta": 1.0,
    "epsilon_delta": 0.0025,
}


class YACController(_BilateralKKTController):
    """Final bilateral Flexible-HRI YAC using a smooth 4DOF Schur model.

    MuJoCo retains its soft joint limits in the physical plant, but their
    state-dependent KKT forces are never cancelled by active feedforward.
    M5 is sensed by the two impedance transformations and likewise is not
    hidden in the nominal effective bias.  Two side-specific B6 critics are
    retained while their physical residual policy uses the full coupled g4.
    """

    def __init__(
        self,
        *,
        reference_result: str | Path,
        mujoco_model: Any,
        mode: str = DEFAULT_MODE,
        active_scale: float = 1.0,
        adp_scale: float = 1.0,
        event_trigger: bool = True,
        event_lambda_eta: float | None = None,
        event_sigma_eta: float | None = None,
        event_xi: float | None = None,
        event_rho_delta: float | None = None,
        event_epsilon_delta: float | None = None,
        intention_mode: str = "live_activation",
        ewma_beta: float = EWMA_BETA,
        **kwargs: Any,
    ) -> None:
        requested = str(mode).strip().lower()
        aliases = {
            "nominal": "nominal",
            "bilateral_nominal": "nominal",
            "active_nominal_flexible_hri": "nominal",
            "continuous_adp": "continuous_adp",
            "full_yac": "full_yac",
            "bilateral_full_yac": "full_yac",
        }
        if requested not in aliases:
            raise ValueError("YAC mode must be nominal, continuous_adp, or full_yac")
        normalized_intention = str(intention_mode).strip().lower()
        if normalized_intention not in ("live_activation", "constant_one"):
            raise ValueError(
                "intention_mode must be live_activation or constant_one"
            )
        self.intention_mode = normalized_intention
        final_mode = aliases[requested]
        event_enabled = final_mode == "full_yac"
        if final_mode == "full_yac" and not event_trigger:
            raise ValueError("Full YAC requires the frozen Dynamic Event law")
        super().__init__(
            reference_result=reference_result,
            mujoco_model=mujoco_model,
            mode=(BILATERAL_FULL_YAC_MODE if event_enabled else BILATERAL_NOMINAL_MODE),
            active_scale=active_scale,
            adp_scale=adp_scale,
            event_trigger=event_enabled,
            event_lambda_eta=event_lambda_eta,
            event_sigma_eta=event_sigma_eta,
            event_xi=event_xi,
            event_rho_delta=event_rho_delta,
            event_epsilon_delta=event_epsilon_delta,
            ewma_beta=ewma_beta,
            **kwargs,
        )
        self.mode = final_mode
        self.event_trigger_enabled = event_enabled
        self.mbar_eff_4, self.ken_healthy_sample_count = self._healthy_effective_mass(
            Path(reference_result), mujoco_model
        )
        eigenvalues = np.linalg.eigvalsh(self.mbar_eff_4)
        if (
            not np.allclose(self.mbar_eff_4, self.mbar_eff_4.T, atol=1.0e-12)
            or not np.all(eigenvalues > 0.0)
        ):
            raise ValueError("Healthy bilateral Mbar_eff_4 must be SPD")
        self.ken_4 = Z_DAMPING_RATE_S_INV * self.mbar_eff_4
        self.ken_eigenvalues = np.linalg.eigvalsh(self.ken_4)
        self._critic_pending: dict[str, Any] | None = None
        self._event4: dict[str, Any] = self._fresh_event4()

    def reset(self) -> None:
        super().reset()
        self._critic_pending = None
        self._event4 = self._fresh_event4()

    @staticmethod
    def _fresh_event4() -> dict[str, Any]:
        return {
            "sampled_z4": None,
            "eta": EVENT_ETA_INITIAL,
            "count": 0,
            "samples": 0,
            "age_steps": 0,
            "last_time": None,
        }

    def _healthy_effective_mass(
        self, result_directory: Path, mujoco_model: Any
    ) -> tuple[np.ndarray, int]:
        physics_path = result_directory.expanduser().resolve() / "physics_timeseries_1khz.csv"
        rows = HealthyGaitReference._read_rows(physics_path)
        data = mujoco.MjData(mujoco_model)
        mujoco.mj_forward(mujoco_model, data)
        vertical_joint = mujoco.mj_name2id(
            mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, "exo_root_vertical"
        )
        vertical_world_scale = float(data.xaxis[vertical_joint, 2])
        left_cycles = self.references["l"].retained_cycles
        right_cycles = self.references["r"].retained_cycles

        def retained(time_s: float, cycles: tuple[tuple[float, float], ...]) -> bool:
            return any(start <= time_s <= end for start, end in cycles)

        matrices = []
        for index, row in enumerate(rows):
            if index % 10:
                continue
            time_s = float(row["time"])
            if not retained(time_s, left_cycles) or not retained(time_s, right_cycles):
                continue
            q = np.asarray(
                (
                    float(row["exo_root_x"]),
                    float(row["exo_root_z"]) / vertical_world_scale,
                    float(row["exo_root_pitch"]),
                    float(row["exo_hip_l_angle"]),
                    float(row["exo_knee_l_angle"]),
                    float(row["exo_hip_r_angle"]),
                    float(row["exo_knee_r_angle"]),
                )
            )
            mass = self.nominal_model.mass_matrix(q)
            u = BILATERAL_UNACTUATED_INDICES
            a = BILATERAL_ACTUATED_INDICES
            muu = mass[np.ix_(u, u)]
            matrices.append(
                mass[np.ix_(a, a)]
                - mass[np.ix_(a, u)] @ np.linalg.solve(muu, mass[np.ix_(u, a)])
            )
        if not matrices:
            raise RuntimeError("Healthy result has no retained bilateral dynamics samples")
        return np.mean(np.stack(matrices), axis=0), len(matrices)

    def _event_policy_gradient(
        self,
        side: str,
        transform: dict[str, Any],
        current_gradient: np.ndarray,
        time_s: float,
        dt: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        state = self.side[side]
        z = np.asarray(transform["z"], dtype=float)
        diagnostics = {
            "triggered": False,
            "reason": EVENT_REASON_NONE,
            "fraction": 0.0,
            "inter_event_s": "",
            "age_s": "",
            "eta": float(state["eta"]),
            "threshold_sq": "",
            "delta_norm_sq": "",
        }
        if not self.event_trigger_enabled:
            return current_gradient, diagnostics
        state["event_samples"] += 1
        decision = _event_trigger_decision(
            state["sampled_z"],
            z,
            float(state["eta"]),
            xi=self.event_xi,
            rho_delta=self.event_rho_delta,
            epsilon_delta=self.event_epsilon_delta,
        )
        state["sampled_z"] = decision.sampled_z_rad_s.copy()
        inter_event: float | str = ""
        if decision.triggered:
            if state["last_event_time"] is not None:
                inter_event = time_s - float(state["last_event_time"])
            state["last_event_time"] = time_s
            state["event_age_steps"] = 0
            state["event_count"] += 1
        else:
            state["event_age_steps"] += 1
        sampled_features, sampled_jacobian = _critic_basis_b6(
            state["sampled_z"], float(transform["a_sched"])
        )
        del sampled_features
        policy_gradient = sampled_jacobian @ np.asarray(
            state["critic_weight"], dtype=float
        )
        eta_logged = float(state["eta"])
        state["eta"], _, _ = _event_eta_exact_zoh_step(
            eta_logged,
            float(z @ z),
            dt,
            lambda_eta=self.event_lambda_eta,
            sigma_eta=self.event_sigma_eta,
        )
        diagnostics.update(
            {
                "triggered": decision.triggered,
                "reason": decision.reason_code,
                "fraction": float(state["event_count"] / state["event_samples"]),
                "inter_event_s": inter_event,
                "age_s": state["event_age_steps"] * dt,
                "eta": eta_logged,
                "threshold_sq": decision.threshold_sq,
                "delta_norm_sq": decision.delta_norm_sq_pretrigger,
            }
        )
        return policy_gradient, diagnostics

    @staticmethod
    def _diagnostic_component(value: Any, index: int) -> float | str:
        if value is None:
            return ""
        return float(np.asarray(value, dtype=float)[index])

    def _compute_legacy_misaligned(
        self, observation: dict[str, object], dt: float
    ) -> dict[str, float]:
        control_dt = _require_finite(dt, "control dt")
        if not math.isclose(control_dt, IMPEDANCE_UPDATE_DT_S, abs_tol=1.0e-12):
            raise ValueError("YAC requires the frozen 100 Hz control step")
        time_s = _require_finite(observation["time"], "observation time")
        contact = observation.get("contact")
        muscles = observation.get("muscles")
        exo = observation.get("exoskeleton")
        if not isinstance(contact, dict) or not isinstance(muscles, dict) or not isinstance(exo, dict):
            raise KeyError("Bilateral YAC observation is incomplete")
        activation = muscles.get("activation")
        dynamics = exo.get("bilateral_dynamics")
        coupling = exo.get("coupling_qfrc_nm")
        constraint = exo.get("constraint_qfrc_nm")
        if not all(isinstance(value, dict) for value in (activation, dynamics, coupling, constraint)):
            raise KeyError("Bilateral activation/dynamics/coupling observation is missing")

        q = np.asarray(dynamics["q_exo"], dtype=float).reshape(7)
        qvel = np.asarray(dynamics["qvel_exo"], dtype=float).reshape(7)
        q4 = q[BILATERAL_ACTUATED_INDICES]
        qdot4 = qvel[BILATERAL_ACTUATED_INDICES]
        qacc4 = np.asarray(
            (
                exo["hip_l"]["acceleration_rad_s2"],
                exo["knee_l"]["acceleration_rad_s2"],
                exo["hip_r"]["acceleration_rad_s2"],
                exo["knee_r"]["acceleration_rad_s2"],
            ),
            dtype=float,
        )
        references = {
            side: self._reference(side, time_s, contact) for side in ("l", "r")
        }
        transforms: dict[str, dict[str, Any]] = {}
        for side, offset in (("l", 0), ("r", 2)):
            transforms[side] = self._side_transform(
                side,
                q4[offset : offset + 2],
                qdot4[offset : offset + 2],
                qacc4[offset : offset + 2],
                np.asarray(
                    (coupling[f"hip_{side}"], coupling[f"knee_{side}"]),
                    dtype=float,
                ),
                activation,
                references[side],
                control_dt,
            )

        ready = bool(all(transforms[side]["valid"] for side in ("l", "r")))
        critic_enabled = self.mode in ("continuous_adp", "full_yac")
        u0 = np.zeros(4, dtype=float)
        mu_raw = np.zeros(4, dtype=float)
        mu_applied = np.zeros(4, dtype=float)
        command = np.zeros(4, dtype=float)
        meff: np.ndarray | None = None
        beff: np.ndarray | None = None
        g4: np.ndarray | None = None
        ramp_factor = 0.0
        critic_diagnostics: dict[str, dict[str, Any]] = {}

        if ready:
            qddotr4 = np.concatenate(
                (transforms["l"]["qddotr"], transforms["r"]["qddotr"])
            )
            z4 = np.concatenate((transforms["l"]["z"], transforms["r"]["z"]))
            # Smooth intrinsic Exoskeleton dynamics only.  M5 and MuJoCo joint
            # limit forces remain in the plant and are not feedforward-cancelled.
            meff, beff = self.nominal_model.bilateral_reduced_dynamics(
                q, qvel, np.zeros(7, dtype=float)
            )
            g4 = np.linalg.inv(meff)
            u0 = meff @ qddotr4 + beff - self.ken_4 @ z4

            features: dict[str, np.ndarray] = {}
            jacobians: dict[str, np.ndarray] = {}
            current_gradients: dict[str, np.ndarray] = {}
            policy_gradients: dict[str, np.ndarray] = {}
            event_diagnostics: dict[str, dict[str, Any]] = {}
            if critic_enabled:
                for side in ("l", "r"):
                    features[side], jacobians[side] = _critic_basis_b6(
                        transforms[side]["z"], transforms[side]["a_sched"]
                    )
                    current_gradients[side] = jacobians[side] @ np.asarray(
                        self.side[side]["critic_weight"], dtype=float
                    )
                    policy_gradients[side], event_diagnostics[side] = (
                        self._event_policy_gradient(
                            side,
                            transforms[side],
                            current_gradients[side],
                            time_s,
                            control_dt,
                        )
                    )
                grad4 = np.concatenate(
                    (policy_gradients["l"], policy_gradients["r"])
                )
                r4_inv = np.zeros((4, 4), dtype=float)
                r4_inv[:2, :2] = CRITIC_R_INV
                r4_inv[2:, 2:] = CRITIC_R_INV
                mu_raw = -0.5 * (r4_inv @ g4.T @ grad4)

            if self._engagement_start_time_s is None:
                self._engagement_start_time_s = time_s
            progress = float(
                np.clip(
                    (time_s - self._engagement_start_time_s)
                    / ENGAGEMENT_RAMP_DURATION_S,
                    0.0,
                    1.0,
                )
            )
            ramp_factor = 0.5 - 0.5 * math.cos(math.pi * progress)
            mu_applied = ramp_factor * mu_raw
            command = ramp_factor * (u0 + mu_raw)

            if critic_enabled:
                direction4: np.ndarray | None = None
                if self._previous_applied_mu_4 is not None:
                    z_dot4 = qacc4 - qddotr4
                    f_hat4 = z_dot4 - g4 @ self._previous_applied_mu_4
                    direction4 = f_hat4 + g4 @ mu_applied
                for side, offset in (("l", 0), ("r", 2)):
                    bellman: float | str = ""
                    update_norm: float | str = ""
                    if direction4 is not None:
                        direction = direction4[offset : offset + 2]
                        z_side = np.asarray(transforms[side]["z"], dtype=float)
                        mu_side = mu_applied[offset : offset + 2]
                        bellman = float(
                            current_gradients[side] @ direction
                            + z_side @ CRITIC_Q @ z_side
                            + mu_side @ CRITIC_R @ mu_side
                        )
                        regressor = jacobians[side].T @ direction
                        update = (
                            -control_dt
                            * CRITIC_KAPPA
                            * regressor
                            * bellman
                            / (1.0 + float(regressor @ regressor)) ** 2
                        )
                        next_weight = (
                            np.asarray(self.side[side]["critic_weight"], dtype=float)
                            + update
                        )
                        if (
                            not np.isfinite(next_weight).all()
                            or np.linalg.norm(next_weight) >= CRITIC_WEIGHT_NORM_LIMIT
                        ):
                            raise RuntimeError(
                                f"Bilateral {side} Critic numerical watchdog failed"
                            )
                        self.side[side]["critic_weight"] = next_weight
                        update_norm = float(np.linalg.norm(update))
                    critic_diagnostics[side] = {
                        "bellman": bellman,
                        "update_norm": update_norm,
                        "event": event_diagnostics[side],
                    }
            self._previous_applied_mu_4 = mu_applied.copy()
        else:
            self._previous_applied_mu_4 = None
            self._engagement_start_time_s = None
            for side in ("l", "r"):
                self.side[side]["sampled_z"] = None
                self.side[side]["eta"] = EVENT_ETA_INITIAL
                self.side[side]["event_age_steps"] = 0

        if not np.isfinite(np.concatenate((u0, mu_raw, command))).all():
            raise FloatingPointError("YAC active torque contains NaN or Inf")

        diagnostics: dict[str, Any] = {
            "time_s": time_s,
            "mode": self.mode,
            "controller_valid": ready,
            "reference_valid": ready,
            "startup_ramp_factor": ramp_factor,
            "hard_kkt_active_feedforward": False,
            "m5_cancelled_in_nominal_bias": False,
            "m5_sensed_in_impedance": True,
            "measured_root_acceleration_used": False,
            "delayed_root_overlay_used": False,
            "ken_source": "lambda_z_times_healthy_Mbar_eff_4",
            "ken_healthy_sample_count": self.ken_healthy_sample_count,
            "adp_policy_input_map": "full_g4_transpose",
            "critic_learning_direction": "local_components_of_full_zdot4_and_g4_mu4",
        }
        labels = ("hip_l", "knee_l", "hip_r", "knee_r")
        for index, label in enumerate(labels):
            diagnostics[f"u0_{label}_nm"] = float(u0[index])
            diagnostics[f"mu_{label}_nm"] = float(mu_raw[index])
            diagnostics[f"mu_applied_{label}_nm"] = float(mu_applied[index])
            diagnostics[f"u_command_{label}_nm"] = float(command[index])
            diagnostics[f"q_exo_{label}_rad"] = float(q4[index])
            diagnostics[f"qdot_exo_{label}_rad_s"] = float(qdot4[index])
            diagnostics[f"qacc_exo_{label}_rad_s2"] = float(qacc4[index])
            diagnostics[f"constraint_{label}_nm"] = float(constraint[label])
            diagnostics[f"constraint_{label}_active"] = bool(
                abs(float(constraint[label])) > CONSTRAINT_ACTIVE_THRESHOLD_NM
            )
        for row in range(4):
            diagnostics[f"b_eff_{row}"] = "" if beff is None else float(beff[row])
            for column in range(4):
                diagnostics[f"Ken_4_{row}{column}"] = float(self.ken_4[row, column])
                diagnostics[f"Mbar_eff_4_{row}{column}"] = float(
                    self.mbar_eff_4[row, column]
                )
                diagnostics[f"M_eff_{row}{column}"] = (
                    "" if meff is None else float(meff[row, column])
                )
                diagnostics[f"g_4_{row}{column}"] = (
                    "" if g4 is None else float(g4[row, column])
                )
        if g4 is None:
            diagnostics.update(
                {name: "" for name in ("g_ll_fro", "g_lr_fro", "g_rl_fro", "g_rr_fro")}
            )
        else:
            diagnostics.update(
                {
                    "g_ll_fro": float(np.linalg.norm(g4[:2, :2])),
                    "g_lr_fro": float(np.linalg.norm(g4[:2, 2:])),
                    "g_rl_fro": float(np.linalg.norm(g4[2:, :2])),
                    "g_rr_fro": float(np.linalg.norm(g4[2:, 2:])),
                }
            )
        for side in ("l", "r"):
            transform = transforms[side]
            state = self.side[side]
            diagnostics.update(
                {
                    f"phase_{side}": transform["phase"],
                    f"stride_period_{side}_s": float(state["period"]),
                    f"heel_strike_{side}": bool(transform["strike"]),
                    f"heel_strike_count_{side}": int(state["strike_count"]),
                    f"activation_proxy_{side}": float(transform["activation_proxy"]),
                    f"a_sched_{side}": float(transform["a_sched"]),
                    f"transform_valid_{side}": bool(transform["valid"]),
                }
            )
            for muscle_name, value in transform["muscles"].items():
                diagnostics[f"activation_{muscle_name}"] = float(value)
            for joint, index in (("hip", 0), ("knee", 1)):
                diagnostics[f"q_ref_{joint}_{side}_rad"] = self._diagnostic_component(transform["qd"], index)
                diagnostics[f"qdot_ref_{joint}_{side}_rad_s"] = self._diagnostic_component(transform["qdotd"], index)
                diagnostics[f"qddot_ref_{joint}_{side}_rad_s2"] = self._diagnostic_component(transform["qddotd"], index)
                diagnostics[f"error_{joint}_{side}_rad"] = self._diagnostic_component(transform["error"], index)
                diagnostics[f"error_velocity_{joint}_{side}_rad_s"] = self._diagnostic_component(transform["error_velocity"], index)
                diagnostics[f"qddot_r_{joint}_{side}_rad_s2"] = self._diagnostic_component(transform["qddotr"], index)
                diagnostics[f"z_{joint}_{side}_rad_s"] = self._diagnostic_component(transform["z"], index)
                diagnostics[f"Kr_{joint}_{side}_nm_per_rad"] = float(transform["stiffness"][index])
                diagnostics[f"Lambda_{joint}_{side}_s_inv"] = self._diagnostic_component(transform["lambda"], index)
                diagnostics[f"Gamma_{joint}_{side}_s_inv"] = self._diagnostic_component(transform["gamma"], index)
                diagnostics[f"t1_{joint}_{side}_rad_s"] = self._diagnostic_component(transform["t1"], index)
            weight = np.asarray(state["critic_weight"], dtype=float)
            critic = critic_diagnostics.get(side)
            event = None if critic is None else critic["event"]
            diagnostics[f"critic_weight_norm_{side}"] = float(np.linalg.norm(weight))
            diagnostics[f"bellman_error_{side}"] = "" if critic is None else critic["bellman"]
            diagnostics[f"critic_update_norm_{side}"] = "" if critic is None else critic["update_norm"]
            diagnostics[f"event_triggered_{side}"] = False if event is None else event["triggered"]
            diagnostics[f"event_reason_{side}"] = EVENT_REASON_NONE if event is None else event["reason"]
            diagnostics[f"event_count_{side}"] = int(state["event_count"])
            diagnostics[f"event_fraction_{side}"] = 0.0 if event is None else event["fraction"]
            diagnostics[f"inter_event_{side}_s"] = "" if event is None else event["inter_event_s"]
            diagnostics[f"event_age_{side}_s"] = "" if event is None else event["age_s"]
            diagnostics[f"event_eta_{side}"] = float(state["eta"]) if event is None else event["eta"]
            diagnostics[f"event_threshold_sq_{side}"] = "" if event is None else event["threshold_sq"]
            diagnostics[f"event_delta_norm_sq_{side}"] = "" if event is None else event["delta_norm_sq"]
            for index in range(CRITIC_WEIGHT_DIMENSION):
                diagnostics[f"critic_weight_{index}_{side}"] = float(weight[index])
        self.last_diagnostics = diagnostics
        torques = dict(ZERO_TORQUES)
        torques.update(
            {
                "HipExo_L": float(command[0]),
                "KneeExo_L": float(command[1]),
                "HipExo_R": float(command[2]),
                "KneeExo_R": float(command[3]),
            }
        )
        return torques

    @staticmethod
    def _global_critic_basis(
        z4: np.ndarray, activation_l: float, activation_r: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return phi12 and the 4x12 Jacobian of V_L + V_R."""
        phi_l, jacobian_l = _critic_basis_b6(z4[:2], activation_l)
        phi_r, jacobian_r = _critic_basis_b6(z4[2:], activation_r)
        jacobian4 = np.zeros((4, 2 * CRITIC_WEIGHT_DIMENSION), dtype=float)
        jacobian4[:2, :CRITIC_WEIGHT_DIMENSION] = jacobian_l
        jacobian4[2:, CRITIC_WEIGHT_DIMENSION:] = jacobian_r
        return np.concatenate((phi_l, phi_r)), jacobian4

    @staticmethod
    def _value_minimum_eigenvalue(weight: np.ndarray) -> float:
        """Minimum quadratic-form eigenvalue over the frozen activation grid."""
        values = []
        for activation in (0.0, 0.25, 0.5, 0.75, 1.0):
            diagonal_0 = weight[0] + activation * weight[3]
            cross = weight[1] + activation * weight[4]
            diagonal_1 = weight[2] + activation * weight[5]
            matrix = np.asarray(
                ((diagonal_0, 0.5 * cross), (0.5 * cross, diagonal_1)),
                dtype=float,
            )
            values.append(float(np.linalg.eigvalsh(matrix)[0]))
        return min(values)

    def _complete_pending_critic_sample(
        self, qacc_realized4: np.ndarray, dt: float
    ) -> dict[str, Any]:
        """Finish interval k-1 using acceleration first observed at sample k.

        MuJoCo's qacc at control sample k was realized under the command held
        over interval k-1.  Every other Hamiltonian quantity below is therefore
        read from the pending k-1 record; no current z, qddotr, g, basis, or
        value gradient is mixed into that physical sample.
        """
        result: dict[str, Any] = {
            "completed": False,
            "sample_time_s": "",
            "bellman_global": "",
            "m12_norm": "",
            "critic_update_12_norm": "",
            "backtracking_count": 0,
            "update_skipped": False,
        }
        pending = self._critic_pending
        if pending is None:
            return result
        zdot4 = np.asarray(qacc_realized4, dtype=float) - np.asarray(
            pending["qddotr4"], dtype=float
        )
        z4 = np.asarray(pending["z4"], dtype=float)
        mu4 = np.asarray(pending["mu_applied4"], dtype=float)
        jacobian4 = np.asarray(pending["Jphi4"], dtype=float)
        gradient4 = np.asarray(pending["gradV4"], dtype=float)
        weight12 = np.asarray(pending["weight12"], dtype=float)
        q4_matrix = np.kron(np.eye(2), CRITIC_Q)
        r4_matrix = np.kron(np.eye(2), CRITIC_R)
        bellman = float(
            gradient4 @ zdot4
            + z4 @ q4_matrix @ z4
            + mu4 @ r4_matrix @ mu4
        )
        regressor12 = jacobian4.T @ zdot4
        denominator = (1.0 + float(regressor12 @ regressor12)) ** 2
        proposed_update = (
            -dt * CRITIC_KAPPA * regressor12 * bellman / denominator
        )
        if not np.isfinite(
            np.concatenate((zdot4, regressor12, proposed_update, [bellman]))
        ).all():
            raise RuntimeError("Global aligned Critic sample is non-finite")

        admissibility_tolerance = 1.0e-9
        maximum_backtracks = 12
        accepted: np.ndarray | None = None
        accepted_count = 0
        for backtracking_count in range(maximum_backtracks + 1):
            candidate = weight12 + (0.5**backtracking_count) * proposed_update
            minimum_l = self._value_minimum_eigenvalue(
                candidate[:CRITIC_WEIGHT_DIMENSION]
            )
            minimum_r = self._value_minimum_eigenvalue(
                candidate[CRITIC_WEIGHT_DIMENSION:]
            )
            if (
                np.isfinite(candidate).all()
                and np.linalg.norm(candidate) < CRITIC_WEIGHT_NORM_LIMIT
                and minimum_l >= -admissibility_tolerance
                and minimum_r >= -admissibility_tolerance
            ):
                accepted = candidate
                accepted_count = backtracking_count
                break
        skipped = accepted is None
        if skipped:
            accepted = weight12
            accepted_count = maximum_backtracks
        self.side["l"]["critic_weight"] = accepted[
            :CRITIC_WEIGHT_DIMENSION
        ].copy()
        self.side["r"]["critic_weight"] = accepted[
            CRITIC_WEIGHT_DIMENSION:
        ].copy()
        applied_update = accepted - weight12
        result.update(
            {
                "completed": True,
                "sample_time_s": float(pending["time_s"]),
                "bellman_global": bellman,
                "m12_norm": float(np.linalg.norm(regressor12)),
                "critic_update_12_norm": float(np.linalg.norm(applied_update)),
                "backtracking_count": accepted_count,
                "update_skipped": skipped,
            }
        )
        return result

    def _synchronized_event_policy_gradient(
        self,
        z4: np.ndarray,
        activation_l: float,
        activation_r: float,
        weight12: np.ndarray,
        time_s: float,
        dt: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply one bilateral event decision to all four sampled z states."""
        state = self._event4
        state["samples"] += 1
        eta_logged = float(state["eta"])
        current_norm_sq = float(z4 @ z4)
        threshold_sq = float(
            (
                self.event_xi * current_norm_sq
                + eta_logged
                + self.event_epsilon_delta
            )
            / self.event_rho_delta
        )
        previous = state["sampled_z4"]
        triggered = previous is None
        reason = EVENT_REASON_INITIALIZATION if triggered else EVENT_REASON_NONE
        delta4 = np.zeros(4, dtype=float) if previous is None else np.asarray(previous) - z4
        delta_norm_sq = float(delta4 @ delta4)
        if previous is not None and delta_norm_sq >= threshold_sq:
            triggered = True
            reason = EVENT_REASON_THRESHOLD
        inter_event: float | str = ""
        if triggered:
            if state["last_time"] is not None:
                inter_event = time_s - float(state["last_time"])
            state["sampled_z4"] = z4.copy()
            state["last_time"] = time_s
            state["age_steps"] = 0
            state["count"] += 1
        else:
            state["age_steps"] += 1
        state["eta"], _, _ = _event_eta_exact_zoh_step(
            eta_logged,
            current_norm_sq,
            dt,
            lambda_eta=self.event_lambda_eta,
            sigma_eta=self.event_sigma_eta,
        )
        _, sampled_jacobian4 = self._global_critic_basis(
            np.asarray(state["sampled_z4"], dtype=float),
            activation_l,
            activation_r,
        )
        return sampled_jacobian4 @ weight12, {
            "triggered": triggered,
            "reason": reason,
            "count": int(state["count"]),
            "fraction": float(state["count"] / state["samples"]),
            "inter_event_s": inter_event,
            "age_s": float(state["age_steps"] * dt),
            "eta": eta_logged,
            "threshold_sq": threshold_sq,
            "delta_norm_sq": delta_norm_sq,
        }

    def compute(self, observation: dict[str, object], dt: float) -> dict[str, float]:
        """Run the causal, globally coupled bilateral ADP/Event controller."""
        control_dt = _require_finite(dt, "control dt")
        if not math.isclose(control_dt, IMPEDANCE_UPDATE_DT_S, abs_tol=1.0e-12):
            raise ValueError("YAC requires the frozen 100 Hz control step")
        time_s = _require_finite(observation["time"], "observation time")
        contact = observation.get("contact")
        muscles = observation.get("muscles")
        exo = observation.get("exoskeleton")
        if not isinstance(contact, dict) or not isinstance(muscles, dict) or not isinstance(exo, dict):
            raise KeyError("Bilateral YAC observation is incomplete")
        activation = muscles.get("activation")
        dynamics = exo.get("bilateral_dynamics")
        coupling = exo.get("coupling_qfrc_nm")
        constraint = exo.get("constraint_qfrc_nm")
        if not all(isinstance(value, dict) for value in (activation, dynamics, coupling, constraint)):
            raise KeyError("Bilateral activation/dynamics/coupling observation is missing")

        q = np.asarray(dynamics["q_exo"], dtype=float).reshape(7)
        qvel = np.asarray(dynamics["qvel_exo"], dtype=float).reshape(7)
        q4 = q[BILATERAL_ACTUATED_INDICES]
        qdot4 = qvel[BILATERAL_ACTUATED_INDICES]
        qacc4 = np.asarray(
            (
                exo["hip_l"]["acceleration_rad_s2"],
                exo["knee_l"]["acceleration_rad_s2"],
                exo["hip_r"]["acceleration_rad_s2"],
                exo["knee_r"]["acceleration_rad_s2"],
            ),
            dtype=float,
        )
        references = {
            side: self._reference(side, time_s, contact) for side in ("l", "r")
        }
        transforms: dict[str, dict[str, Any]] = {}
        for side, offset in (("l", 0), ("r", 2)):
            transforms[side] = self._side_transform(
                side,
                q4[offset : offset + 2],
                qdot4[offset : offset + 2],
                qacc4[offset : offset + 2],
                np.asarray(
                    (coupling[f"hip_{side}"], coupling[f"knee_{side}"]),
                    dtype=float,
                ),
                activation,
                references[side],
                control_dt,
            )

        ready = bool(all(transforms[side]["valid"] for side in ("l", "r")))
        critic_enabled = self.mode in ("continuous_adp", "full_yac")
        u0 = np.zeros(4, dtype=float)
        mu_raw = np.zeros(4, dtype=float)
        mu_applied = np.zeros(4, dtype=float)
        command = np.zeros(4, dtype=float)
        meff: np.ndarray | None = None
        beff: np.ndarray | None = None
        g4: np.ndarray | None = None
        ramp_factor = 0.0
        residual_state_power: float | str = ""
        update_diagnostics = {
            "completed": False,
            "sample_time_s": "",
            "bellman_global": "",
            "m12_norm": "",
            "critic_update_12_norm": "",
            "backtracking_count": 0,
            "update_skipped": False,
        }
        event_diagnostics = {
            "triggered": False,
            "reason": EVENT_REASON_NONE,
            "count": int(self._event4["count"]),
            "fraction": 0.0,
            "inter_event_s": "",
            "age_s": "",
            "eta": float(self._event4["eta"]),
            "threshold_sq": "",
            "delta_norm_sq": "",
        }

        if ready:
            qddotr4 = np.concatenate(
                (transforms["l"]["qddotr"], transforms["r"]["qddotr"])
            )
            z4 = np.concatenate((transforms["l"]["z"], transforms["r"]["z"]))
            meff, beff = self.nominal_model.bilateral_reduced_dynamics(
                q, qvel, np.zeros(7, dtype=float)
            )
            g4 = np.linalg.inv(meff)
            u0 = meff @ qddotr4 + beff - self.ken_4 @ z4
            if self._engagement_start_time_s is None:
                self._engagement_start_time_s = time_s
            progress = float(
                np.clip(
                    (time_s - self._engagement_start_time_s)
                    / ENGAGEMENT_RAMP_DURATION_S,
                    0.0,
                    1.0,
                )
            )
            ramp_factor = 0.5 - 0.5 * math.cos(math.pi * progress)
            learning_active = bool(critic_enabled and progress >= 1.0)

            if critic_enabled:
                if learning_active:
                    # Complete k-1 first; current policy k then uses the updated W12.
                    update_diagnostics = self._complete_pending_critic_sample(
                        qacc4, control_dt
                    )
                else:
                    # Startup dynamics do not satisfy the steady YAC law.
                    self._critic_pending = None
                    self._event4 = self._fresh_event4()

                weight12 = np.concatenate(
                    (
                        np.asarray(self.side["l"]["critic_weight"], dtype=float),
                        np.asarray(self.side["r"]["critic_weight"], dtype=float),
                    )
                )
                phi12, jacobian4 = self._global_critic_basis(
                    z4,
                    float(transforms["l"]["a_sched"]),
                    float(transforms["r"]["a_sched"]),
                )
                gradient4 = jacobian4 @ weight12
                policy_gradient4 = gradient4
                if self.mode == "full_yac" and learning_active:
                    policy_gradient4, event_diagnostics = (
                        self._synchronized_event_policy_gradient(
                            z4,
                            float(transforms["l"]["a_sched"]),
                            float(transforms["r"]["a_sched"]),
                            weight12,
                            time_s,
                            control_dt,
                        )
                    )
                r4_inverse = np.kron(np.eye(2), CRITIC_R_INV)
                mu_raw = -0.5 * (r4_inverse @ g4.T @ policy_gradient4)
                mu_applied = ramp_factor * mu_raw
                if learning_active:
                    residual_state_power = float(z4 @ g4 @ mu_applied)
                    self._critic_pending = {
                        "time_s": time_s,
                        "z4": z4.copy(),
                        "qddotr4": qddotr4.copy(),
                        "g4": g4.copy(),
                        "mu_applied4": mu_applied.copy(),
                        "phi12": phi12.copy(),
                        "Jphi4": jacobian4.copy(),
                        "gradV4": gradient4.copy(),
                        "weight12": weight12.copy(),
                        "residual_state_power": residual_state_power,
                    }
            command = ramp_factor * (u0 + mu_raw)
        else:
            self._critic_pending = None
            self._event4 = self._fresh_event4()
            self._engagement_start_time_s = None

        if not np.isfinite(np.concatenate((u0, mu_raw, command))).all():
            raise FloatingPointError("YAC active torque contains NaN or Inf")

        weight_l = np.asarray(self.side["l"]["critic_weight"], dtype=float)
        weight_r = np.asarray(self.side["r"]["critic_weight"], dtype=float)
        diagnostics: dict[str, Any] = {
            "time_s": time_s,
            "mode": self.mode,
            "controller_valid": ready,
            "reference_valid": ready,
            "startup_ramp_factor": ramp_factor,
            "critic_learning_active": bool(critic_enabled and ramp_factor >= 1.0),
            "critic_pending_valid": self._critic_pending is not None,
            "critic_sample_completed": update_diagnostics["completed"],
            "critic_sample_time_s": update_diagnostics["sample_time_s"],
            "bellman_global": update_diagnostics["bellman_global"],
            "m12_norm": update_diagnostics["m12_norm"],
            "critic_update_12_norm": update_diagnostics["critic_update_12_norm"],
            "critic_backtracking_count": update_diagnostics["backtracking_count"],
            "critic_update_skipped": update_diagnostics["update_skipped"],
            "value_min_eigenvalue_L": self._value_minimum_eigenvalue(weight_l),
            "value_min_eigenvalue_R": self._value_minimum_eigenvalue(weight_r),
            "residual_state_power": residual_state_power,
            "hard_kkt_active_feedforward": False,
            "m5_cancelled_in_nominal_bias": False,
            "m5_sensed_in_impedance": True,
            "measured_root_acceleration_used": False,
            "delayed_root_overlay_used": False,
            "ken_source": "lambda_z_times_healthy_Mbar_eff_4",
            "ken_healthy_sample_count": self.ken_healthy_sample_count,
            "adp_sample_timing": "qacc_k_completes_stored_interval_k_minus_1",
            "adp_value_structure": "V4_equals_VL_plus_VR_12_weights",
            "adp_policy_input_map": "full_g4_transpose",
            "critic_learning_direction": "global_aligned_zdot4",
            "event_scope": "synchronized_bilateral_z4",
            "critic_z_scale_hip_rad_s": float(CRITIC_Z_SCALE_RAD_S[0]),
            "critic_z_scale_knee_rad_s": float(CRITIC_Z_SCALE_RAD_S[1]),
            "intention_mode": self.intention_mode,
            "intention_ablation": (
                "paper_equivalent_constant_activation_a1"
                if self.intention_mode == "constant_one"
                else "live_simulation_muscle_activation"
            ),
        }
        labels = ("hip_l", "knee_l", "hip_r", "knee_r")
        for index, label in enumerate(labels):
            diagnostics[f"u0_{label}_nm"] = float(u0[index])
            diagnostics[f"mu_{label}_nm"] = float(mu_raw[index])
            diagnostics[f"mu_applied_{label}_nm"] = float(mu_applied[index])
            diagnostics[f"u_command_{label}_nm"] = float(command[index])
            diagnostics[f"q_exo_{label}_rad"] = float(q4[index])
            diagnostics[f"qdot_exo_{label}_rad_s"] = float(qdot4[index])
            diagnostics[f"qacc_exo_{label}_rad_s2"] = float(qacc4[index])
            diagnostics[f"constraint_{label}_nm"] = float(constraint[label])
            diagnostics[f"constraint_{label}_active"] = bool(
                abs(float(constraint[label])) > CONSTRAINT_ACTIVE_THRESHOLD_NM
            )
            diagnostics[f"tau_interaction_{label}_nm"] = float(
                coupling[label]
            )
        for row in range(4):
            diagnostics[f"b_eff_{row}"] = "" if beff is None else float(beff[row])
            for column in range(4):
                diagnostics[f"Ken_4_{row}{column}"] = float(self.ken_4[row, column])
                diagnostics[f"Mbar_eff_4_{row}{column}"] = float(self.mbar_eff_4[row, column])
                diagnostics[f"M_eff_{row}{column}"] = "" if meff is None else float(meff[row, column])
                diagnostics[f"g_4_{row}{column}"] = "" if g4 is None else float(g4[row, column])
        if g4 is None:
            diagnostics.update({name: "" for name in ("g_ll_fro", "g_lr_fro", "g_rl_fro", "g_rr_fro")})
        else:
            diagnostics.update(
                {
                    "g_ll_fro": float(np.linalg.norm(g4[:2, :2])),
                    "g_lr_fro": float(np.linalg.norm(g4[:2, 2:])),
                    "g_rl_fro": float(np.linalg.norm(g4[2:, :2])),
                    "g_rr_fro": float(np.linalg.norm(g4[2:, 2:])),
                }
            )
        for side, weight in (("l", weight_l), ("r", weight_r)):
            transform = transforms[side]
            state = self.side[side]
            diagnostics.update(
                {
                    f"phase_{side}": transform["phase"],
                    f"stride_period_{side}_s": float(state["period"]),
                    f"heel_strike_{side}": bool(transform["strike"]),
                    f"heel_strike_count_{side}": int(state["strike_count"]),
                    f"activation_proxy_{side}": float(transform["activation_proxy"]),
                    f"a_sched_{side}": float(transform["a_sched"]),
                    f"transform_valid_{side}": bool(transform["valid"]),
                    f"critic_weight_norm_{side}": float(np.linalg.norm(weight)),
                    f"bellman_error_{side}": "",
                    f"critic_update_norm_{side}": "",
                    f"event_triggered_{side}": event_diagnostics["triggered"],
                    f"event_reason_{side}": event_diagnostics["reason"],
                    f"event_count_{side}": event_diagnostics["count"],
                    f"event_fraction_{side}": event_diagnostics["fraction"],
                    f"inter_event_{side}_s": event_diagnostics["inter_event_s"],
                    f"event_age_{side}_s": event_diagnostics["age_s"],
                    f"event_eta_{side}": event_diagnostics["eta"],
                    f"event_threshold_sq_{side}": event_diagnostics["threshold_sq"],
                    f"event_delta_norm_sq_{side}": event_diagnostics["delta_norm_sq"],
                }
            )
            for muscle_name, value in transform["muscles"].items():
                diagnostics[f"activation_{muscle_name}"] = float(value)
            for joint, index in (("hip", 0), ("knee", 1)):
                diagnostics[f"q_ref_{joint}_{side}_rad"] = self._diagnostic_component(transform["qd"], index)
                diagnostics[f"qdot_ref_{joint}_{side}_rad_s"] = self._diagnostic_component(transform["qdotd"], index)
                diagnostics[f"qddot_ref_{joint}_{side}_rad_s2"] = self._diagnostic_component(transform["qddotd"], index)
                diagnostics[f"error_{joint}_{side}_rad"] = self._diagnostic_component(transform["error"], index)
                diagnostics[f"error_velocity_{joint}_{side}_rad_s"] = self._diagnostic_component(transform["error_velocity"], index)
                diagnostics[f"qddot_r_{joint}_{side}_rad_s2"] = self._diagnostic_component(transform["qddotr"], index)
                diagnostics[f"z_{joint}_{side}_rad_s"] = self._diagnostic_component(transform["z"], index)
                diagnostics[f"Kr_{joint}_{side}_nm_per_rad"] = float(transform["stiffness"][index])
                diagnostics[f"Lambda_{joint}_{side}_s_inv"] = self._diagnostic_component(transform["lambda"], index)
                diagnostics[f"Gamma_{joint}_{side}_s_inv"] = self._diagnostic_component(transform["gamma"], index)
                diagnostics[f"t1_{joint}_{side}_rad_s"] = self._diagnostic_component(transform["t1"], index)
            for index in range(CRITIC_WEIGHT_DIMENSION):
                diagnostics[f"critic_weight_{index}_{side}"] = float(weight[index])
        diagnostics.update(
            {
                "event_triggered_global": event_diagnostics["triggered"],
                "event_reason_global": event_diagnostics["reason"],
                "event_count_global": event_diagnostics["count"],
                "event_fraction_global": event_diagnostics["fraction"],
                "inter_event_global_s": event_diagnostics["inter_event_s"],
                "event_age_global_s": event_diagnostics["age_s"],
                "event_eta_global": event_diagnostics["eta"],
                "event_threshold_global_sq": event_diagnostics["threshold_sq"],
                "event_delta_global_norm_sq": event_diagnostics["delta_norm_sq"],
            }
        )
        self.last_diagnostics = diagnostics
        torques = dict(ZERO_TORQUES)
        torques.update(
            {
                "HipExo_L": float(command[0]),
                "KneeExo_L": float(command[1]),
                "HipExo_R": float(command[2]),
                "KneeExo_R": float(command[3]),
            }
        )
        return torques


def _direct_run_summary(
    project_root: Path, validation: dict[str, Any]
) -> dict[str, Any]:
    manifest_path = Path(str(validation["artifacts"]["manifest"]))
    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path
    result_directory = manifest_path.parent
    with (result_directory / "timeseries.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    time = np.asarray([float(row["time"]) for row in rows], dtype=float)
    speed = np.asarray([float(row["forward_velocity"]) for row in rows], dtype=float)
    pitch = np.asarray([float(row["pelvis_pitch"]) for row in rows], dtype=float)
    height = np.asarray([float(row["pelvis_height"]) for row in rows], dtype=float)
    activation_keys = [
        key for key in rows[0] if key.startswith("muscle_activation_")
    ]
    activations = np.column_stack(
        [np.asarray([float(row[key]) for row in rows]) for key in activation_keys]
    )
    duration = float(time[-1] - time[0])
    ja = (
        float(np.trapezoid(np.sum(activations * activations, axis=1), time) / duration)
        if duration > 0.0
        else 0.0
    )
    saturation = {}
    for motor in ("HipExo_L", "KneeExo_L", "HipExo_R", "KneeExo_R"):
        actual = np.asarray(
            [float(row[f"exo_actual_torque_{motor}_nm"]) for row in rows]
        )
        saturation[motor] = float(np.mean(np.abs(actual) >= 100.0 - 1.0e-9))
    return {
        "survival_time_s": float(validation["actual_duration_s"]),
        "termination_reason": str(validation["termination_reason"]),
        "average_speed_mps": float(np.mean(speed[1:] if speed.size > 1 else speed)),
        "max_abs_pelvis_pitch_deg": float(np.max(np.abs(pitch))),
        "min_pelvis_height_m": float(np.min(height)),
        "Ja": ja,
        "saturation": saturation,
    }


def _single_run_main() -> int:
    """Launch the final real bilateral YAC experiment without CLI arguments."""
    import contextlib
    import io
    import yaml

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from orchestration import run_human_exo

    smoke = os.environ.get("YAC_DIRECT_SMOKE") == "1"
    duration = min(DEFAULT_DURATION_S, 0.05) if smoke else DEFAULT_DURATION_S
    viewer = (
        DEFAULT_VIEWER
        and not smoke
        and os.environ.get("MUJOCO_HEADLESS") != "1"
    )
    if DEFAULT_PATHOLOGY != "bilateral_moderate60":
        raise ValueError("Final direct YAC experiment freezes bilateral_moderate60")
    if DEFAULT_MODE not in ("full_yac", "continuous_adp", "nominal"):
        raise ValueError("DEFAULT_MODE must be full_yac, continuous_adp, or nominal")
    pathology_path = (
        project_root
        / "models/humans/patients/left_knee_weakness/moderate.yaml"
    )
    pathology = yaml.safe_load(pathology_path.read_text(encoding="utf-8"))
    affected = set(pathology["affected_muscles"])
    if affected != {"vasti_l", "vasti_r"}:
        raise ValueError("Final pathology must affect exactly vasti_l and vasti_r")
    pathology_ratios = {
        muscle: float(pathology["damaged_max_isometric_force_N"][muscle])
        / float(pathology["healthy_max_isometric_force_N"][muscle])
        for muscle in ("vasti_l", "vasti_r")
    }
    if not all(
        math.isclose(ratio, 0.6, abs_tol=1.0e-12)
        for ratio in pathology_ratios.values()
    ):
        raise ValueError("Final bilateral vasti pathology is not exactly 60%")
    controller_mode = (
        "bilateral_nominal" if DEFAULT_MODE == "nominal" else DEFAULT_MODE
    )
    event_trigger = DEFAULT_MODE == "full_yac"

    print("==========================================")
    print("YAC Bilateral Experiment")
    print("Pathology: vasti L/R = 60%")
    print(f"vasti_l ratio: {pathology_ratios['vasti_l']:.6f}")
    print(f"vasti_r ratio: {pathology_ratios['vasti_r']:.6f}")
    print(
        "Controller: Full YAC"
        if DEFAULT_MODE == "full_yac"
        else f"Controller: {DEFAULT_MODE}"
    )
    print("Motors: HipL KneeL HipR KneeR")
    print(f"Duration: {duration:g} s")
    print(f"Viewer: {'ON' if viewer else 'OFF'}")
    print("==========================================", flush=True)

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        validation = run_human_exo(
            human_condition="moderate60",
            exoskeleton="flexible",
            assistance="yac",
            duration_s=duration,
            video=False,
            evaluation=False,
            run_name="YAC_FINAL_BILATERAL_MODERATE60_FULL",
            healthy_reference_result=DEFAULT_HEALTHY_REFERENCE_RESULT,
            controller_mode=controller_mode,
            active_scale=1.0,
            adp_scale=1.0,
            event_trigger=event_trigger,
            event_parameters=(
                dict(DEFAULT_EVENT_PARAMETERS) if event_trigger else None
            ),
            output_root=project_root / "results/runs/YAC_final",
            enforce_regression=False,
            viewer=viewer,
            viewer_keep_open=viewer,
            record_git_metadata=False,
        )
    damage = validation["damage_verification"]["muscles"]
    for muscle in ("vasti_l", "vasti_r"):
        if not math.isclose(float(damage[muscle]["gain_ratio"]), 0.6, abs_tol=1.0e-12):
            raise RuntimeError(f"{muscle} pathology ratio is not exactly 0.60")
    summary = _direct_run_summary(project_root, validation)
    print(f"Survival time: {summary['survival_time_s']:.3f} s")
    print(f"Termination reason: {summary['termination_reason']}")
    print(f"Average speed: {summary['average_speed_mps']:.4f} m/s")
    print(f"Max |pelvis pitch|: {summary['max_abs_pelvis_pitch_deg']:.3f} deg")
    print(f"Min pelvis height: {summary['min_pelvis_height_m']:.4f} m")
    print(f"Ja: {summary['Ja']:.6f}")
    for motor in ("HipExo_L", "KneeExo_L", "HipExo_R", "KneeExo_R"):
        print(f"{motor} saturation: {summary['saturation'][motor]:.3%}")
    return 0


def _artifact_path(
    project_root: Path, validation: dict[str, Any], artifact: str
) -> Path | None:
    raw = validation.get("artifacts", {}).get(artifact)
    if not raw:
        return None
    path = Path(str(raw))
    return path if path.is_absolute() else project_root / path


def _paper_run_metrics(
    project_root: Path,
    result_directory: Path,
    common_end_s: float,
) -> dict[str, Any]:
    """Compose the minimal paper metrics from M6 and YAC diagnostics."""
    from evaluation import metrics as m6_metrics

    m6_summary, table = m6_metrics.evaluate_directory(
        result_directory,
        project_root,
        (DEFAULT_HEALTHY_REFERENCE_RESULT / "timeseries.csv",),
    )
    diagnostics = m6_metrics.load_timeseries(
        result_directory / "controller_diagnostics.csv"
    )
    validation = m6_metrics.load_json_if_present(
        result_directory / "validation.json"
    )

    def metric(category: str, name: str) -> Any:
        return m6_metrics.metric_value(m6_summary["metrics"][category], name)

    time, _ = m6_metrics.numeric_column(table, ["time"])
    diagnostic_time, _ = m6_metrics.numeric_column(diagnostics, ["time_s"])
    if time is None or diagnostic_time is None:
        raise ValueError("Paper evaluation requires saved simulation time")
    reference_valid, _ = m6_metrics.bool_column(
        diagnostics, ["reference_valid"]
    )
    tracking_mask = (
        np.isfinite(diagnostic_time)
        & (diagnostic_time >= YAC_HRI_TRANSIENT_CUTOFF_S)
        & (diagnostic_time <= common_end_s)
    )
    if reference_valid is not None:
        tracking_mask &= reference_valid

    labels = ("hip_l", "knee_l", "hip_r", "knee_r")
    tracking: dict[str, float | None] = {}
    all_tracking_values = []
    plot_actual: dict[str, list[float]] = {}
    plot_reference: dict[str, list[float]] = {}
    for label in labels:
        joint, side = label.split("_")
        error, _ = m6_metrics.numeric_column(
            diagnostics, [f"error_{joint}_{side}_rad"]
        )
        actual, _ = m6_metrics.numeric_column(
            diagnostics, [f"q_exo_{joint}_{side}_rad"]
        )
        reference, _ = m6_metrics.numeric_column(
            diagnostics, [f"q_ref_{joint}_{side}_rad"]
        )
        if error is None:
            tracking[label] = None
            continue
        valid = tracking_mask & np.isfinite(error)
        tracking[label] = (
            float(np.sqrt(np.mean(np.square(error[valid]))))
            if np.any(valid)
            else None
        )
        if np.any(valid):
            all_tracking_values.append(error[valid])
        if actual is not None and reference is not None:
            plot_actual[label] = actual[tracking_mask].tolist()
            plot_reference[label] = reference[tracking_mask].tolist()
    tracking_all = (
        float(
            np.sqrt(
                np.mean(np.square(np.concatenate(all_tracking_values)))
            )
        )
        if all_tracking_values
        else None
    )

    activation_columns = sorted(
        name for name in table if name.startswith("muscle_activation_")
    )
    activation_arrays = []
    for name in activation_columns:
        values, _ = m6_metrics.numeric_column(table, [name])
        if values is not None:
            activation_arrays.append(values)
    common_ja: float | None = None
    common_activation_mask = (
        np.isfinite(time) & (time >= 0.0) & (time <= common_end_s)
    )
    if activation_arrays:
        activation_matrix = np.column_stack(activation_arrays)
        common_activation_mask &= np.all(np.isfinite(activation_matrix), axis=1)
        if np.count_nonzero(common_activation_mask) >= 2:
            selected_time = time[common_activation_mask]
            common_ja = float(
                np.trapezoid(
                    np.sum(
                        np.square(activation_matrix[common_activation_mask]), axis=1
                    ),
                    selected_time,
                )
                / (selected_time[-1] - selected_time[0])
            )

    hri_columns = {
        label: f"tau_interaction_{label}_nm" for label in labels
    }
    hri_arrays = []
    hri_per_joint: dict[str, float | None] = {}
    for label, name in hri_columns.items():
        values, _ = m6_metrics.numeric_column(diagnostics, [name])
        if values is None:
            hri_per_joint[label] = None
        else:
            hri_arrays.append(values)
    hri_cost: float | None = None
    hri_rms: float | None = None
    hri_plot_time: list[float] = []
    hri_cumulative: list[float] = []
    if len(hri_arrays) == 4 and common_end_s > YAC_HRI_TRANSIENT_CUTOFF_S:
        hri_matrix = np.column_stack(hri_arrays)
        hri_mask = (
            np.isfinite(diagnostic_time)
            & np.all(np.isfinite(hri_matrix), axis=1)
            & (diagnostic_time >= YAC_HRI_TRANSIENT_CUTOFF_S)
            & (diagnostic_time <= common_end_s)
        )
        if np.count_nonzero(hri_mask) >= 2:
            selected_time = diagnostic_time[hri_mask]
            selected_hri = hri_matrix[hri_mask]
            squared_norm = np.sum(np.square(selected_hri), axis=1)
            hri_cost = float(np.trapezoid(squared_norm, selected_time))
            hri_rms = float(np.sqrt(np.mean(squared_norm)))
            for index, label in enumerate(labels):
                hri_per_joint[label] = float(
                    np.sqrt(np.mean(np.square(selected_hri[:, index])))
                )
            cumulative = np.zeros_like(squared_norm)
            cumulative[1:] = np.cumsum(
                0.5
                * (squared_norm[1:] + squared_norm[:-1])
                * np.diff(selected_time)
            )
            hri_plot_time = (
                selected_time - YAC_HRI_TRANSIENT_CUTOFF_S
            ).tolist()
            hri_cumulative = cumulative.tolist()

    # Evaluation-only reconstruction of the original-paper interaction term:
    # tau_e = Kr(a) * e.  This channel is never injected into the MuJoCo plant.
    paper_hri_arrays = []
    for label in labels:
        joint, side = label.split("_")
        stiffness, _ = m6_metrics.numeric_column(
            diagnostics, [f"Kr_{joint}_{side}_nm_per_rad"]
        )
        error, _ = m6_metrics.numeric_column(
            diagnostics, [f"error_{joint}_{side}_rad"]
        )
        if stiffness is not None and error is not None:
            paper_hri_arrays.append(stiffness * error)
    paper_hri_cost: float | None = None
    paper_hri_rms: float | None = None
    paper_hri_plot_time: list[float] = []
    paper_hri_cumulative: list[float] = []
    if len(paper_hri_arrays) == 4 and common_end_s > YAC_HRI_TRANSIENT_CUTOFF_S:
        paper_hri_matrix = np.column_stack(paper_hri_arrays)
        paper_hri_mask = (
            np.isfinite(diagnostic_time)
            & np.all(np.isfinite(paper_hri_matrix), axis=1)
            & (diagnostic_time >= YAC_HRI_TRANSIENT_CUTOFF_S)
            & (diagnostic_time <= common_end_s)
        )
        if reference_valid is not None:
            paper_hri_mask &= reference_valid
        if np.count_nonzero(paper_hri_mask) >= 2:
            selected_time = diagnostic_time[paper_hri_mask]
            selected_paper_hri = paper_hri_matrix[paper_hri_mask]
            squared_norm = np.sum(np.square(selected_paper_hri), axis=1)
            paper_hri_cost = float(np.trapezoid(squared_norm, selected_time))
            paper_hri_rms = float(np.sqrt(np.mean(squared_norm)))
            cumulative = np.zeros_like(squared_norm)
            cumulative[1:] = np.cumsum(
                0.5
                * (squared_norm[1:] + squared_norm[:-1])
                * np.diff(selected_time)
            )
            paper_hri_plot_time = (
                selected_time - YAC_HRI_TRANSIENT_CUTOFF_S
            ).tolist()
            paper_hri_cumulative = cumulative.tolist()

    motor_channels = ("HipExo_L", "KneeExo_L", "HipExo_R", "KneeExo_R")
    motor_rms = []
    motor_peak = []
    saturation: dict[str, float | None] = {}
    for motor in motor_channels:
        rms = metric("exoskeleton", f"{motor}.rms_torque_Nm")
        peak = metric("exoskeleton", f"{motor}.peak_abs_torque_Nm")
        if rms is not None:
            motor_rms.append(float(rms))
        if peak is not None:
            motor_peak.append(float(peak))
        actual, _ = m6_metrics.numeric_column(
            table, [f"exo_actual_torque_{motor}_nm"]
        )
        finite = np.asarray([]) if actual is None else actual[np.isfinite(actual)]
        saturation[motor] = (
            float(np.mean(np.abs(finite) >= 100.0 - 1.0e-9))
            if finite.size
            else None
        )

    def finite_diagnostic(name: str) -> np.ndarray:
        values, _ = m6_metrics.numeric_column(diagnostics, [name])
        return (
            np.asarray([], dtype=float)
            if values is None
            else values[np.isfinite(values)]
        )

    bellman = finite_diagnostic("bellman_global")
    intervals = finite_diagnostic("inter_event_global_s")
    event_triggered, _ = m6_metrics.bool_column(
        diagnostics, ["event_triggered_global"]
    )
    event_times = (
        diagnostic_time[event_triggered & (diagnostic_time >= YAC_HRI_TRANSIENT_CUTOFF_S)]
        if event_triggered is not None
        else np.asarray([], dtype=float)
    )
    critic_l = finite_diagnostic("critic_weight_norm_l")
    critic_r = finite_diagnostic("critic_weight_norm_r")
    event_count = finite_diagnostic("event_count_global")
    event_fraction = finite_diagnostic("event_fraction_global")
    survival = float(validation.get("actual_duration_s", time[-1] - time[0]))
    result = {
        "survival_s": survival,
        "termination_reason": str(validation.get("termination_reason", "")),
        "average_speed_mps": metric("stability", "mean_forward_velocity_mps"),
        "max_abs_pelvis_pitch_deg": metric(
            "stability", "max_abs_pelvis_pitch_deg"
        ),
        "min_pelvis_height_m": metric("stability", "min_pelvis_height_m"),
        "q_rmse_rad": tracking,
        "q_rms_overall_rad": tracking_all,
        "Ja_full_run": metric("human_effort", "activation_squared_mean"),
        "Ja_common_window": common_ja,
        "activation_channel_count": len(activation_arrays),
        "E_HRI_M5_N2m2s": hri_cost,
        "HRI_RMS_total_Nm": hri_rms,
        "HRI_RMS_per_joint_Nm": hri_per_joint,
        "E_HRI_paper_N2m2s": paper_hri_cost,
        "HRI_paper_RMS_total_Nm": paper_hri_rms,
        "torque_rms_Nm": (
            float(np.sqrt(np.mean(np.square(motor_rms))))
            if len(motor_rms) == 4
            else None
        ),
        "torque_peak_Nm": max(motor_peak) if len(motor_peak) == 4 else None,
        "saturation_fraction": saturation,
        "Bellman_RMS": (
            float(np.sqrt(np.mean(np.square(bellman)))) if bellman.size else None
        ),
        "critic_norm_L": float(critic_l[-1]) if critic_l.size else None,
        "critic_norm_R": float(critic_r[-1]) if critic_r.size else None,
        "event_count": int(event_count[-1]) if event_count.size else 0,
        "event_fraction": float(event_fraction[-1]) if event_fraction.size else 0.0,
        "policy_refresh_reduction_percent": (
            100.0 * (1.0 - float(event_fraction[-1]))
            if event_fraction.size
            else 100.0
        ),
        "mean_inter_event_s": float(np.mean(intervals)) if intervals.size else None,
        "median_inter_event_s": float(np.median(intervals)) if intervals.size else None,
        "_plot": {
            "tracking_time_s": diagnostic_time[tracking_mask].tolist(),
            "actual": plot_actual,
            "reference": plot_reference,
            "hri_time_s": hri_plot_time,
            "hri_cumulative": hri_cumulative,
            "paper_hri_time_s": paper_hri_plot_time,
            "paper_hri_cumulative": paper_hri_cumulative,
            "event_times_s": event_times.tolist(),
        },
    }
    return result


def _intention_numeric_column(
    table: dict[str, Any], name: str
) -> np.ndarray:
    """Load one required finite/numeric audit column without inventing data."""
    from evaluation import metrics as m6_metrics

    values, reason = m6_metrics.numeric_column(table, [name])
    if values is None:
        raise ValueError(f"Intention audit requires {name}: {reason}")
    return np.asarray(values, dtype=float)


def _complete_gait_cycles(
    time_s: np.ndarray,
    phase: np.ndarray,
    heel_strike: np.ndarray,
    series: dict[str, np.ndarray],
    common_end_s: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[list[float]]]:
    """Interpolate complete post-startup heel-strike cycles to 101 points."""
    grid = np.linspace(0.0, 1.0, REFERENCE_PHASE_POINTS)
    starts = np.flatnonzero(heel_strike & np.isfinite(phase))
    interpolated: dict[str, list[np.ndarray]] = {name: [] for name in series}
    intervals: list[list[float]] = []
    for start, stop in zip(starts[:-1], starts[1:]):
        if (
            time_s[start] < YAC_HRI_TRANSIENT_CUTOFF_S
            or time_s[stop] > common_end_s + 1.0e-9
            or stop - start < 5
        ):
            continue
        cycle_phase = phase[start:stop]
        phase_finite = np.isfinite(cycle_phase)
        cycle_phase = cycle_phase[phase_finite]
        if (
            cycle_phase.size < 5
            or cycle_phase[0] > 0.05
            or cycle_phase[-1] < 0.90
        ):
            continue
        strictly_increasing = np.concatenate(
            (np.asarray([True]), np.diff(cycle_phase) > 1.0e-12)
        )
        cycle_phase = cycle_phase[strictly_increasing]
        cycle_values: dict[str, np.ndarray] = {}
        cycle_is_valid = True
        for name, values in series.items():
            selected = values[start:stop][phase_finite][strictly_increasing]
            finite = np.isfinite(selected) & np.isfinite(cycle_phase)
            if np.count_nonzero(finite) < 5:
                cycle_is_valid = False
                break
            cycle_values[name] = np.interp(
                grid, cycle_phase[finite], selected[finite]
            )
        if not cycle_is_valid:
            continue
        for name, values in cycle_values.items():
            interpolated[name].append(values)
        intervals.append([float(time_s[start]), float(time_s[stop])])
    if not intervals:
        raise ValueError("No complete post-startup gait cycles available for audit")
    return (
        grid,
        {name: np.vstack(values) for name, values in interpolated.items()},
        intervals,
    )


def _distribution_summary(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot summarize an empty intention channel")
    p5, p50, p95 = np.percentile(finite, (5.0, 50.0, 95.0))
    mean = float(np.mean(finite))
    std = float(np.std(finite))
    return {
        "min": float(np.min(finite)),
        "mean": mean,
        "std": std,
        "max": float(np.max(finite)),
        "p5": float(p5),
        "p50": float(p50),
        "p95": float(p95),
        "dynamic_range_90": float(p95 - p5),
        "coefficient_of_variation": float(std / mean) if mean > 0.0 else 0.0,
        "fraction_below_0_05": float(np.mean(finite < 0.05)),
        "fraction_above_0_95": float(np.mean(finite > 0.95)),
    }


def _safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    valid = np.isfinite(first) & np.isfinite(second)
    if np.count_nonzero(valid) < 3:
        return 0.0
    x = first[valid]
    y = second[valid]
    if np.std(x) <= 1.0e-15 or np.std(y) <= 1.0e-15:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _save_intention_audit_figure(
    path: Path,
    phase_percent: np.ndarray,
    left_reference_cycles: dict[str, np.ndarray],
) -> None:
    """Save the required four-panel trajectory-native intention audit."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    left_a = np.mean(left_reference_cycles["a_sched_l"], axis=0)
    right_a = np.mean(left_reference_cycles["a_sched_r"], axis=0)
    left_std = np.std(left_reference_cycles["a_sched_l"], axis=0)
    right_std = np.std(left_reference_cycles["a_sched_r"], axis=0)
    right_shifted = np.concatenate((np.roll(right_a[:-1], -50), [right_a[50]]))
    right_shifted_std = np.concatenate(
        (np.roll(right_std[:-1], -50), [right_std[50]])
    )
    axes[0, 0].plot(phase_percent, left_a, label="Left", color="tab:blue")
    axes[0, 0].fill_between(
        phase_percent, left_a - left_std, left_a + left_std,
        color="tab:blue", alpha=0.12,
    )
    axes[0, 0].plot(phase_percent, right_a, label="Right", color="tab:orange")
    axes[0, 0].plot(
        phase_percent, right_shifted, label="Right shifted 50%",
        color="tab:orange", linestyle="--",
    )
    axes[0, 0].fill_between(
        phase_percent, right_shifted - right_shifted_std,
        right_shifted + right_shifted_std, color="tab:orange", alpha=0.10,
    )
    axes[0, 0].set_title("A. Bilateral scalar intention")
    axes[0, 0].set_ylabel("a_sched")
    axes[0, 0].legend(fontsize=8)

    colors = {
        "hamstrings": "tab:brown", "bifemsh": "tab:purple",
        "glutmax": "tab:green", "iliopsoas": "tab:red",
        "rectfem": "tab:pink", "vasti": "tab:cyan",
    }
    for muscle, color in colors.items():
        left = np.mean(left_reference_cycles[f"activation_{muscle}_l"], axis=0)
        right = np.mean(left_reference_cycles[f"activation_{muscle}_r"], axis=0)
        right_shift = np.concatenate((np.roll(right[:-1], -50), [right[50]]))
        axes[0, 1].plot(phase_percent, left, label=muscle, color=color)
        axes[0, 1].plot(
            phase_percent, right_shift, color=color, linestyle="--", alpha=0.45
        )
    axes[0, 1].set_title("B. Muscle activation (L solid, R shift-50 dashed)")
    axes[0, 1].set_ylabel("MyoAssist data.act")
    axes[0, 1].legend(fontsize=7, ncol=2)

    for joint, side, color, style in (
        ("hip", "l", "tab:blue", "-"),
        ("knee", "l", "tab:green", "-"),
        ("hip", "r", "tab:orange", "--"),
        ("knee", "r", "tab:red", "--"),
    ):
        values = np.mean(
            left_reference_cycles[f"Kr_{joint}_{side}_nm_per_rad"], axis=0
        )
        axes[1, 0].plot(
            phase_percent, values, label=f"{joint.title()} {side.upper()}",
            color=color, linestyle=style,
        )
    axes[1, 0].set_title("C. Intention-dependent stiffness")
    axes[1, 0].set_ylabel("Kr (Nm/rad)")
    axes[1, 0].legend(fontsize=8, ncol=2)

    paper_power = np.mean(left_reference_cycles["paper_hri_power"], axis=0)
    m5_power = np.mean(left_reference_cycles["m5_hri_power"], axis=0)
    paper_axis = axes[1, 1]
    m5_axis = paper_axis.twinx()
    paper_line = paper_axis.plot(
        phase_percent, paper_power, color="tab:blue", label="Paper-equivalent"
    )[0]
    m5_line = m5_axis.plot(
        phase_percent, m5_power, color="tab:red", label="M5 physical"
    )[0]
    paper_axis.set_title("D. Interaction cost density")
    paper_axis.set_ylabel(r"$||K_r e||^2$ ($N^2m^2$)", color="tab:blue")
    m5_axis.set_ylabel(r"$||tau_{M5}||^2$ ($N^2m^2$)", color="tab:red")
    paper_axis.legend(handles=(paper_line, m5_line), fontsize=8)
    for axis in axes.flat:
        axis.set_xlabel("Gait cycle (%)")
        axis.grid(alpha=0.2)
    figure.suptitle("YAC Intention Mapping Audit (trajectory-native; not measured EMG)")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _run_intention_mapping_audit(
    baseline_directory: Path,
    yac_directory: Path,
    common_end_s: float,
    baseline: dict[str, Any],
    yac: dict[str, Any],
    ja_baseline: float | None,
    ja_yac: float | None,
) -> tuple[dict[str, Any], Path, Path]:
    """Audit the live per-leg scalar mapping and apply the fast decision gate."""
    from evaluation import metrics as m6_metrics

    diagnostics = m6_metrics.load_timeseries(
        yac_directory / "controller_diagnostics.csv"
    )
    time_s = _intention_numeric_column(diagnostics, "time_s")
    reference_valid, reason = m6_metrics.bool_column(diagnostics, ["reference_valid"])
    if reference_valid is None:
        raise ValueError(f"Intention audit requires reference_valid: {reason}")
    heel_strikes: dict[str, np.ndarray] = {}
    phases: dict[str, np.ndarray] = {}
    series: dict[str, np.ndarray] = {}
    for side in ("l", "r"):
        strike, reason = m6_metrics.bool_column(
            diagnostics, [f"heel_strike_{side}"]
        )
        if strike is None:
            raise ValueError(f"Intention audit requires heel_strike_{side}: {reason}")
        heel_strikes[side] = np.asarray(strike, dtype=bool)
        phases[side] = _intention_numeric_column(diagnostics, f"phase_{side}")
        for name in ("activation_proxy", "a_sched"):
            series[f"{name}_{side}"] = _intention_numeric_column(
                diagnostics, f"{name}_{side}"
            )
        for muscle in INTENTION_MUSCLE_FAMILIES:
            series[f"activation_{muscle}_{side}"] = _intention_numeric_column(
                diagnostics, f"activation_{muscle}_{side}"
            )
        for joint in ("hip", "knee"):
            for prefix, suffix in (
                ("Kr", "nm_per_rad"), ("error", "rad"),
                ("tau_interaction", "nm"),
            ):
                key = f"{prefix}_{joint}_{side}_{suffix}"
                series[key] = _intention_numeric_column(diagnostics, key)

    paper_power = np.zeros_like(time_s)
    m5_power = np.zeros_like(time_s)
    for side in ("l", "r"):
        for joint in ("hip", "knee"):
            paper_tau = (
                series[f"Kr_{joint}_{side}_nm_per_rad"]
                * series[f"error_{joint}_{side}_rad"]
            )
            paper_power += np.square(paper_tau)
            m5_power += np.square(series[f"tau_interaction_{joint}_{side}_nm"])
    series["paper_hri_power"] = paper_power
    series["m5_hri_power"] = m5_power

    grid, left_reference_cycles, left_intervals = _complete_gait_cycles(
        time_s, phases["l"], heel_strikes["l"], series, common_end_s
    )
    own_cycles: dict[str, dict[str, np.ndarray]] = {"l": left_reference_cycles}
    own_intervals: dict[str, list[list[float]]] = {"l": left_intervals}
    _, own_cycles["r"], own_intervals["r"] = _complete_gait_cycles(
        time_s, phases["r"], heel_strikes["r"], series, common_end_s
    )
    phase_percent = 100.0 * grid
    analysis_mask = (
        np.isfinite(time_s)
        & (time_s >= YAC_HRI_TRANSIENT_CUTOFF_S)
        & (time_s <= common_end_s)
        & reference_valid
    )

    activation_summary: dict[str, dict[str, float]] = {}
    stiffness_summary: dict[str, dict[str, float]] = {}
    muscle_summary: dict[str, dict[str, dict[str, Any]]] = {"left": {}, "right": {}}
    gait_waveforms: dict[str, Any] = {"phase_percent": phase_percent.tolist()}
    shadow_summary: dict[str, Any] = {}
    muscle_sanity = True
    for side, side_name in (("l", "left"), ("r", "right")):
        activation_summary[side_name] = _distribution_summary(
            series[f"a_sched_{side}"][analysis_mask]
        )
        side_waveforms: dict[str, Any] = {}
        for channel in (
            f"activation_proxy_{side}",
            f"a_sched_{side}",
            f"Kr_hip_{side}_nm_per_rad",
            f"Kr_knee_{side}_nm_per_rad",
        ):
            channel_cycles = own_cycles[side][channel]
            side_waveforms[channel] = {
                "mean": np.mean(channel_cycles, axis=0).tolist(),
                "std": np.std(channel_cycles, axis=0).tolist(),
            }
        for muscle in INTENTION_MUSCLE_FAMILIES:
            values = series[f"activation_{muscle}_{side}"]
            cycles = own_cycles[side][f"activation_{muscle}_{side}"]
            mean_wave = np.mean(cycles, axis=0)
            std_wave = np.std(cycles, axis=0)
            distribution = _distribution_summary(values[analysis_mask])
            peak_index = int(np.argmax(mean_wave))
            stance_mean = float(np.mean(mean_wave[phase_percent <= 60.0]))
            swing_mean = float(np.mean(mean_wave[phase_percent > 60.0]))
            muscle_summary[side_name][muscle] = {
                "peak_phase_percent": float(phase_percent[peak_index]),
                "peak_activation": float(mean_wave[peak_index]),
                "p5": distribution["p5"],
                "p50": distribution["p50"],
                "p95": distribution["p95"],
                "stance_mean": stance_mean,
                "swing_mean": swing_mean,
            }
            side_waveforms[muscle] = {
                "mean": mean_wave.tolist(), "std": std_wave.tolist()
            }
            peak = float(phase_percent[peak_index])
            if muscle == "vasti":
                muscle_sanity &= peak <= 60.0 and stance_mean > swing_mean
            elif muscle == "iliopsoas":
                muscle_sanity &= 50.0 <= peak <= 90.0 and swing_mean > stance_mean
            elif muscle == "bifemsh":
                muscle_sanity &= 60.0 <= peak <= 100.0 and swing_mean > stance_mean
            elif muscle in ("hamstrings", "glutmax"):
                muscle_sanity &= peak <= 40.0 or peak >= 80.0
        gait_waveforms[side_name] = side_waveforms

        for joint in ("hip", "knee"):
            key = f"Kr_{joint}_{side}_nm_per_rad"
            stiffness_summary[f"{joint}_{side_name}"] = _distribution_summary(
                series[key][analysis_mask]
            )

        current = series[f"a_sched_{side}"][analysis_mask]
        hip = np.sqrt(
            np.mean(
                np.square(
                    np.vstack(
                        [
                            series[f"activation_{muscle}_{side}"][analysis_mask]
                            for muscle in ("glutmax", "iliopsoas", "hamstrings", "rectfem")
                        ]
                    )
                ),
                axis=0,
            )
        )
        knee = np.sqrt(
            np.mean(
                np.square(
                    np.vstack(
                        [
                            series[f"activation_{muscle}_{side}"][analysis_mask]
                            for muscle in ("vasti", "rectfem", "hamstrings", "bifemsh")
                        ]
                    )
                ),
                axis=0,
            )
        )
        hip_stats = _distribution_summary(hip)
        knee_stats = _distribution_summary(knee)
        hip_wave = np.sqrt(
            np.mean(
                np.square(
                    np.vstack(
                        [
                            np.mean(own_cycles[side][f"activation_{muscle}_{side}"], axis=0)
                            for muscle in ("glutmax", "iliopsoas", "hamstrings", "rectfem")
                        ]
                    )
                ),
                axis=0,
            )
        )
        knee_wave = np.sqrt(
            np.mean(
                np.square(
                    np.vstack(
                        [
                            np.mean(own_cycles[side][f"activation_{muscle}_{side}"], axis=0)
                            for muscle in ("vasti", "rectfem", "hamstrings", "bifemsh")
                        ]
                    )
                ),
                axis=0,
            )
        )
        shadow_summary[side_name] = {
            "hip": hip_stats,
            "knee": knee_stats,
            "current_vs_hip_correlation": _safe_correlation(current, hip),
            "current_vs_knee_correlation": _safe_correlation(current, knee),
            "hip_vs_knee_correlation": _safe_correlation(hip, knee),
            "hip_vs_knee_rms_difference": float(
                np.sqrt(np.mean(np.square(hip - knee)))
            ),
            "hip_peak_phase_percent": float(phase_percent[np.argmax(hip_wave)]),
            "knee_peak_phase_percent": float(phase_percent[np.argmax(knee_wave)]),
            "phase_dependent_difference": (hip_wave - knee_wave).tolist(),
        }

    left_mean = np.mean(left_reference_cycles["a_sched_l"], axis=0)[:-1]
    right_mean = np.mean(left_reference_cycles["a_sched_r"], axis=0)[:-1]
    shift_correlations = np.asarray(
        [
            _safe_correlation(left_mean, np.roll(right_mean, -shift))
            for shift in range(100)
        ]
    )
    best_shift = int(np.argmax(shift_correlations))
    corr_raw = _safe_correlation(left_mean, right_mean)
    corr_shift50 = _safe_correlation(left_mean, np.roll(right_mean, -50))
    best_correlation = float(shift_correlations[best_shift])
    bilateral_alternation = bool(
        35 <= best_shift <= 65 and best_correlation >= 0.50 and corr_raw < best_correlation
    )

    left_flat = bool(
        activation_summary["left"]["dynamic_range_90"] < 0.12
        or activation_summary["left"]["coefficient_of_variation"] < 0.15
    )
    right_flat = bool(
        activation_summary["right"]["dynamic_range_90"] < 0.12
        or activation_summary["right"]["coefficient_of_variation"] < 0.15
    )
    shadow_distinct = bool(
        all(
            abs(
                shadow_summary[side]["hip_peak_phase_percent"]
                - shadow_summary[side]["knee_peak_phase_percent"]
            )
            >= 15.0
            and shadow_summary[side]["hip_vs_knee_correlation"] < 0.50
            for side in ("left", "right")
        )
    )
    scalar_information_loss = bool(shadow_distinct and (left_flat or right_flat))
    paper_reduction = _percent_reduction(
        baseline["E_HRI_paper_N2m2s"], yac["E_HRI_paper_N2m2s"]
    )
    m5_change = (
        None
        if baseline["E_HRI_M5_N2m2s"] in (None, 0.0)
        or yac["E_HRI_M5_N2m2s"] is None
        else 100.0
        * (yac["E_HRI_M5_N2m2s"] - baseline["E_HRI_M5_N2m2s"])
        / baseline["E_HRI_M5_N2m2s"]
    )
    ja_change = (
        None
        if ja_baseline in (None, 0.0) or ja_yac is None
        else 100.0 * (ja_yac - ja_baseline) / ja_baseline
    )
    stable_20s = bool(yac["survival_s"] >= DEFAULT_DURATION_S - 1.0e-6)
    tracking_not_worse = bool(
        baseline["q_rms_overall_rad"] is not None
        and yac["q_rms_overall_rad"] is not None
        and yac["q_rms_overall_rad"] <= baseline["q_rms_overall_rad"] * 1.05
    )
    paper_mechanism_reproduced = bool(
        paper_reduction is not None
        and paper_reduction >= 5.0
        and stable_20s
        and tracking_not_worse
    )
    if not bilateral_alternation or not muscle_sanity:
        decision = "HUMAN ACTIVATION REQUIRES REVIEW"
    elif paper_mechanism_reproduced:
        decision = "KEEP CURRENT MAPPING"
    elif left_flat or right_flat:
        decision = "TEST GAMMA 1.5/2.0"
    else:
        decision = "KEEP CURRENT MAPPING"

    summary: dict[str, Any] = {
        "schema_version": "YAC_INTENTION_MAPPING_AUDIT_V1",
        "source": {
            "baseline_result_dir": str(baseline_directory),
            "yac_result_dir": str(yac_directory),
            "evaluation_window_s": [YAC_HRI_TRANSIENT_CUTOFF_S, common_end_s],
            "activation_semantics": "MyoAssist data.act; not measured experimental EMG",
            "paper_hri_semantics": "evaluation-only tau_e=Kr(a)*e; never injected into plant",
        },
        "complete_stride_count": {
            "left": len(own_intervals["l"]), "right": len(own_intervals["r"])
        },
        "complete_stride_intervals_s": {
            "left": own_intervals["l"], "right": own_intervals["r"]
        },
        "left_a_mean": activation_summary["left"]["mean"],
        "left_a_p5": activation_summary["left"]["p5"],
        "left_a_p95": activation_summary["left"]["p95"],
        "left_a_range": activation_summary["left"]["dynamic_range_90"],
        "right_a_mean": activation_summary["right"]["mean"],
        "right_a_p5": activation_summary["right"]["p5"],
        "right_a_p95": activation_summary["right"]["p95"],
        "right_a_range": activation_summary["right"]["dynamic_range_90"],
        "activation_summary": activation_summary,
        "corr_raw": corr_raw,
        "corr_shift50": corr_shift50,
        "best_lr_phase_shift_percent": float(best_shift),
        "best_lr_correlation": best_correlation,
        "bilateral_alternation": "PASS" if bilateral_alternation else "SUSPICIOUS",
        "muscle_phase": muscle_summary,
        "muscle_phase_overall": "PASS" if muscle_sanity else "SUSPICIOUS",
        "stiffness_summary": stiffness_summary,
        "hip_shadow_range_L": shadow_summary["left"]["hip"]["dynamic_range_90"],
        "hip_shadow_range_R": shadow_summary["right"]["hip"]["dynamic_range_90"],
        "knee_shadow_range_L": shadow_summary["left"]["knee"]["dynamic_range_90"],
        "knee_shadow_range_R": shadow_summary["right"]["knee"]["dynamic_range_90"],
        "shadow_proxy_audit": shadow_summary,
        "hip_knee_shadow_structure_distinct": shadow_distinct,
        "scalar_information_loss": scalar_information_loss,
        "Paper_E_HRI_baseline": baseline["E_HRI_paper_N2m2s"],
        "Paper_E_HRI_live": yac["E_HRI_paper_N2m2s"],
        "Paper_HRI_reduction_percent": paper_reduction,
        "E_HRI_M5_baseline": baseline["E_HRI_M5_N2m2s"],
        "E_HRI_M5_live": yac["E_HRI_M5_N2m2s"],
        "M5_HRI_change_percent": m5_change,
        "Ja_baseline": ja_baseline,
        "Ja_live": ja_yac,
        "Ja_change_percent": ja_change,
        "tracking_change_percent": _percent_reduction(
            baseline["q_rms_overall_rad"], yac["q_rms_overall_rad"]
        ),
        "full_yac_survival_s": yac["survival_s"],
        "paper_mechanism_reproduced": paper_mechanism_reproduced,
        "physical_m5_benefit_reproduced": bool(m5_change is not None and m5_change < 0.0),
        "mapping_selected": "per_leg_six_muscle_rms_clip",
        "gamma_selected": 1.0,
        "calibration_executed": False,
        "decision": decision,
        "gait_cycle_waveforms": gait_waveforms,
        "left_phase_bilateral_waveforms": {
            name: {
                "mean": np.mean(left_reference_cycles[name], axis=0).tolist(),
                "std": np.std(left_reference_cycles[name], axis=0).tolist(),
            }
            for name in (
                "a_sched_l",
                "a_sched_r",
                "paper_hri_power",
                "m5_hri_power",
            )
        },
    }
    summary_path = yac_directory / "yac_intention_mapping_audit.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    figure_path = yac_directory / "yac_intention_mapping_audit.png"
    _save_intention_audit_figure(figure_path, phase_percent, left_reference_cycles)
    if not figure_path.is_file() or figure_path.stat().st_size == 0:
        raise RuntimeError("Intention audit figure was not created")
    return summary, summary_path, figure_path


def _print_intention_mapping_audit(summary: dict[str, Any]) -> None:
    print("\n==========================================")
    print("YAC INTENTION MAPPING AUDIT")
    print("==========================================")
    print("\n[LEFT / RIGHT]")
    left = summary["activation_summary"]["left"]
    right = summary["activation_summary"]["right"]
    print(f"Left a p5 / p50 / p95: {left['p5']:.4f} / {left['p50']:.4f} / {left['p95']:.4f}")
    print(f"Right a p5 / p50 / p95: {right['p5']:.4f} / {right['p50']:.4f} / {right['p95']:.4f}")
    print(f"Best bilateral phase shift: {summary['best_lr_phase_shift_percent']:.1f}%")
    print(f"Correlation: {summary['best_lr_correlation']:.4f}")
    print(f"Bilateral alternation: {summary['bilateral_alternation']}")
    print("\n[MUSCLE PHASE]")
    for muscle in INTENTION_MUSCLE_FAMILIES:
        left_peak = summary["muscle_phase"]["left"][muscle]["peak_phase_percent"]
        right_peak = summary["muscle_phase"]["right"][muscle]["peak_phase_percent"]
        print(f"{muscle}: L {left_peak:.1f}% / R {right_peak:.1f}%")
    print(f"Overall: {summary['muscle_phase_overall']}")
    print("\n[SCALAR MAPPING]")
    print(f"Dynamic range L: {summary['left_a_range']:.4f}")
    print(f"Dynamic range R: {summary['right_a_range']:.4f}")
    print(
        "Hip/Knee information loss: "
        + ("YES" if summary["scalar_information_loss"] else "NO")
    )
    print("\n[ORIGINAL PAPER METRIC]")
    print(f"Constant a=1 E_HRI_paper: {summary['Paper_E_HRI_baseline']:.3f}")
    print(f"Live YAC E_HRI_paper: {summary['Paper_E_HRI_live']:.3f}")
    print(f"Reduction: {summary['Paper_HRI_reduction_percent']:.3f}%")
    print(
        "Paper mechanism reproduced: "
        + ("YES" if summary["paper_mechanism_reproduced"] else "NO")
    )
    print("\n[PHYSICAL HRI]")
    print(f"M5 change: {summary['M5_HRI_change_percent']:+.3f}%")
    print("\n[HUMAN EFFORT]")
    print(f"Ja change: {summary['Ja_change_percent']:+.3f}%")
    print("\n[DECISION]")
    print(summary["decision"])
    print("\n[FINAL ACTIVE TEST if executed]")
    print("Not executed: CASE A passed with gamma=1.0.")
    print("==========================================")


def _verify_paper_fairness(
    baseline_directory: Path, yac_directory: Path
) -> dict[str, Any]:
    baseline = json.loads((baseline_directory / "manifest.json").read_text())
    yac = json.loads((yac_directory / "manifest.json").read_text())
    frozen_keys = (
        "human_xml_sha256",
        "patient_config_sha256",
        "coupled_xml_sha256",
        "human_controller_parameter_sha256",
        "exo_mass_kg",
        "control_timestep_s",
        "mujoco_timestep_s",
        "seed",
        "coupling_version",
        "reference_source",
    )
    checks = {key: baseline.get(key) == yac.get(key) for key in frozen_keys}
    baseline_params = dict(baseline.get("assist_parameters", {}))
    yac_params = dict(yac.get("assist_parameters", {}))
    baseline_intention = baseline_params.pop("intention_mode", None)
    yac_intention = yac_params.pop("intention_mode", None)
    checks["same_yac_parameters_except_intention"] = baseline_params == yac_params
    checks["constant_one_vs_live_activation"] = (
        baseline_intention == "constant_one"
        and yac_intention == "live_activation"
    )
    if not all(checks.values()):
        raise RuntimeError(f"Paper baseline fairness verification failed: {checks}")
    return {"passed": True, "checks": checks}


def _percent_reduction(baseline: float | None, current: float | None) -> float | None:
    if baseline is None or current is None or abs(baseline) <= 1.0e-15:
        return None
    return 100.0 * (baseline - current) / baseline


def _save_paper_figure(
    path: Path,
    baseline: dict[str, Any],
    yac: dict[str, Any],
    ja_baseline: float | None,
    ja_yac: float | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    tracking = yac["_plot"]
    t = np.asarray(tracking["tracking_time_s"], dtype=float)
    colors = {"hip_l": "tab:blue", "knee_l": "tab:orange", "hip_r": "tab:green", "knee_r": "tab:red"}
    for label, color in colors.items():
        actual = np.asarray(tracking["actual"].get(label, []), dtype=float)
        reference = np.asarray(tracking["reference"].get(label, []), dtype=float)
        if t.size and actual.size == t.size:
            axes[0, 0].plot(t, actual, color=color, linewidth=1.0, label=label)
        if t.size and reference.size == t.size:
            axes[0, 0].plot(t, reference, color=color, linestyle="--", linewidth=0.8)
    axes[0, 0].set_title("A. Healthy reference (dashed) vs Full YAC")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Joint angle (rad)")
    if axes[0, 0].lines:
        axes[0, 0].legend(ncol=2, fontsize=8)

    for data, label, color in (
        (baseline, "Constant a=1", "0.4"),
        (yac, "Live intention", "tab:blue"),
    ):
        axes[0, 1].plot(
            data["_plot"]["hri_time_s"],
            data["_plot"]["hri_cumulative"],
            label=label,
            color=color,
        )
    axes[0, 1].set_title("B. Cumulative squared M5 interaction cost")
    axes[0, 1].set_xlabel("Time after 2 s cutoff (s)")
    axes[0, 1].set_ylabel(r"$E_{HRI,M5}$ ($N^2m^2s$)")
    axes[0, 1].legend()

    axes[1, 0].bar(
        ("Constant a=1", "Live intention"),
        (np.nan if ja_baseline is None else ja_baseline, np.nan if ja_yac is None else ja_yac),
        color=("0.5", "tab:blue"),
    )
    axes[1, 0].set_title("C. Human Effort Proxy")
    axes[1, 0].set_ylabel(r"$J_a=(1/T)\int\sum a_i^2dt$")

    event_times = np.asarray(yac["_plot"]["event_times_s"], dtype=float)
    if event_times.size:
        axes[1, 1].eventplot(event_times, colors="tab:blue", lineoffsets=1.0)
    axes[1, 1].set_title(
        f"D. Event raster (fraction={yac['event_fraction']:.3f})"
    )
    axes[1, 1].set_xlabel("Simulation time (s)")
    axes[1, 1].set_yticks([])
    figure.suptitle("YAC One-Click Paper Experiment V1")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _format_metric(value: Any, digits: int = 4, suffix: str = "") -> str:
    if value is None:
        return "INVALID"
    return f"{float(value):.{digits}f}{suffix}"


def main() -> int:
    """Run the constant-intention baseline and proposed Full YAC once."""
    import contextlib
    import io
    import yaml

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from orchestration import run_human_exo

    smoke = os.environ.get("YAC_DIRECT_SMOKE") == "1"
    duration = min(DEFAULT_DURATION_S, 0.05) if smoke else DEFAULT_DURATION_S
    viewer = DEFAULT_VIEWER and not smoke and os.environ.get("MUJOCO_HEADLESS") != "1"
    if DEFAULT_PATHOLOGY != "bilateral_moderate60":
        raise ValueError("Final direct YAC experiment freezes bilateral_moderate60")
    if DEFAULT_MODE != "full_yac":
        raise ValueError("Paper experiment requires DEFAULT_MODE='full_yac'")
    if DEFAULT_PAPER_COMPARISON and not DEFAULT_EVALUATION:
        raise ValueError("Paper comparison requires DEFAULT_EVALUATION=True")

    pathology_path = project_root / "models/humans/patients/left_knee_weakness/moderate.yaml"
    pathology = yaml.safe_load(pathology_path.read_text(encoding="utf-8"))
    ratios = {
        muscle: float(pathology["damaged_max_isometric_force_N"][muscle])
        / float(pathology["healthy_max_isometric_force_N"][muscle])
        for muscle in ("vasti_l", "vasti_r")
    }
    if not all(math.isclose(value, 0.6, abs_tol=1.0e-12) for value in ratios.values()):
        raise ValueError("Final bilateral vasti pathology is not exactly 60%")

    output_root = project_root / "results/runs/YAC_paper_experiment_v1"
    event_parameters = dict(DEFAULT_EVENT_PARAMETERS)
    baseline_validation: dict[str, Any] | None = None
    captured = io.StringIO()
    if DEFAULT_PAPER_COMPARISON:
        print("Running paper baseline:")
        print("Constant activation a=1")
        print("Viewer OFF")
        print("Video OFF", flush=True)
        with contextlib.redirect_stdout(captured):
            baseline_validation = run_human_exo(
                human_condition="moderate60",
                exoskeleton="flexible",
                assistance="yac",
                duration_s=duration,
                video=False,
                evaluation=False,
                run_name="YAC_PAPER_BASELINE_CONSTANT_A1",
                healthy_reference_result=DEFAULT_HEALTHY_REFERENCE_RESULT,
                controller_mode="bilateral_full_yac",
                active_scale=1.0,
                adp_scale=1.0,
                event_trigger=True,
                event_parameters=event_parameters,
                intention_mode="constant_one",
                output_root=output_root,
                enforce_regression=False,
                viewer=False,
                viewer_keep_open=False,
                record_git_metadata=False,
            )
        print("Baseline complete.\n")

    print("Running proposed Full YAC:")
    print("Live muscle-activation intention")
    print(f"Viewer {'ON' if viewer else 'OFF'}")
    print(f"Video {'ON' if DEFAULT_VIDEO else 'OFF'}", flush=True)
    captured.seek(0)
    captured.truncate(0)
    with contextlib.redirect_stdout(captured):
        yac_validation = run_human_exo(
            human_condition="moderate60",
            exoskeleton="flexible",
            assistance="yac",
            duration_s=duration,
            video=DEFAULT_VIDEO,
            evaluation=False,
            run_name="YAC_PAPER_FULL_INTENTION",
            healthy_reference_result=DEFAULT_HEALTHY_REFERENCE_RESULT,
            controller_mode="bilateral_full_yac",
            active_scale=1.0,
            adp_scale=1.0,
            event_trigger=True,
            event_parameters=event_parameters,
            intention_mode="live_activation",
            output_root=output_root,
            enforce_regression=False,
            viewer=viewer,
            viewer_keep_open=viewer,
            record_git_metadata=False,
        )

    yac_manifest = _artifact_path(project_root, yac_validation, "manifest")
    if yac_manifest is None:
        raise RuntimeError("Full YAC manifest path is unavailable")
    yac_directory = yac_manifest.parent
    video_path = _artifact_path(project_root, yac_validation, "video")
    if DEFAULT_VIDEO and (
        video_path is None or not video_path.is_file() or video_path.stat().st_size == 0
    ):
        raise RuntimeError("Full YAC video artifact was not created")

    if not DEFAULT_PAPER_COMPARISON or baseline_validation is None:
        summary = _direct_run_summary(project_root, yac_validation)
        print(f"Result directory: {yac_directory}")
        print(f"Video: {video_path}")
        print(json.dumps(summary, indent=2))
        return 0

    baseline_manifest = _artifact_path(project_root, baseline_validation, "manifest")
    if baseline_manifest is None:
        raise RuntimeError("Baseline manifest path is unavailable")
    baseline_directory = baseline_manifest.parent
    fairness = _verify_paper_fairness(baseline_directory, yac_directory)
    common_end = min(
        float(baseline_validation["actual_duration_s"]),
        float(yac_validation["actual_duration_s"]),
    )
    baseline = _paper_run_metrics(project_root, baseline_directory, common_end)
    yac = _paper_run_metrics(project_root, yac_directory, common_end)
    both_complete = bool(
        baseline["survival_s"] >= duration - 1.0e-6
        and yac["survival_s"] >= duration - 1.0e-6
    )
    ja_baseline = (
        baseline["Ja_full_run"] if both_complete else baseline["Ja_common_window"]
    )
    ja_yac = yac["Ja_full_run"] if both_complete else yac["Ja_common_window"]
    ja_reduction = _percent_reduction(ja_baseline, ja_yac)
    hri_reduction = _percent_reduction(
        baseline["E_HRI_M5_N2m2s"], yac["E_HRI_M5_N2m2s"]
    )
    paper_hri_reduction = _percent_reduction(
        baseline["E_HRI_paper_N2m2s"], yac["E_HRI_paper_N2m2s"]
    )
    tracking_change = _percent_reduction(
        baseline["q_rms_overall_rad"], yac["q_rms_overall_rad"]
    )
    intention_audit, intention_summary_path, intention_figure_path = (
        _run_intention_mapping_audit(
            baseline_directory,
            yac_directory,
            common_end,
            baseline,
            yac,
            ja_baseline,
            ja_yac,
        )
    )

    paper_summary = {
        "schema_version": "YAC_ONE_CLICK_PAPER_V1",
        "baseline_result_dir": str(baseline_directory),
        "yac_result_dir": str(yac_directory),
        "survival": {
            "baseline_s": baseline["survival_s"],
            "yac_s": yac["survival_s"],
        },
        "q_rms": {
            "baseline_overall_rad": baseline["q_rms_overall_rad"],
            "yac_overall_rad": yac["q_rms_overall_rad"],
            "baseline_per_joint_rad": baseline["q_rmse_rad"],
            "yac_per_joint_rad": yac["q_rmse_rad"],
            "tracking_change_percent": tracking_change,
        },
        "Ja": {"baseline": ja_baseline, "yac": ja_yac},
        "Ja_reduction_percent": ja_reduction,
        "full_run_effort_comparison_valid": both_complete,
        "E_HRI_M5": {
            "baseline_N2m2s": baseline["E_HRI_M5_N2m2s"],
            "yac_N2m2s": yac["E_HRI_M5_N2m2s"],
            "evaluation_window_s": [YAC_HRI_TRANSIENT_CUTOFF_S, common_end],
        },
        "HRI_reduction_percent": hri_reduction,
        "E_HRI_paper": {
            "baseline_N2m2s": baseline["E_HRI_paper_N2m2s"],
            "yac_N2m2s": yac["E_HRI_paper_N2m2s"],
            "evaluation_window_s": [YAC_HRI_TRANSIENT_CUTOFF_S, common_end],
            "evaluation_only": True,
        },
        "Paper_HRI_reduction_percent": paper_hri_reduction,
        "HRI_RMS": {
            "baseline_total_Nm": baseline["HRI_RMS_total_Nm"],
            "yac_total_Nm": yac["HRI_RMS_total_Nm"],
            "baseline_per_joint_Nm": baseline["HRI_RMS_per_joint_Nm"],
            "yac_per_joint_Nm": yac["HRI_RMS_per_joint_Nm"],
        },
        "Bellman_RMS": yac["Bellman_RMS"],
        "critic_norm_L": yac["critic_norm_L"],
        "critic_norm_R": yac["critic_norm_R"],
        "event_count": yac["event_count"],
        "event_fraction": yac["event_fraction"],
        "policy_refresh_reduction_percent": yac[
            "policy_refresh_reduction_percent"
        ],
        "torque_rms": yac["torque_rms_Nm"],
        "torque_peak": yac["torque_peak_Nm"],
        "saturation": yac["saturation_fraction"],
        "video_path": str(video_path) if video_path is not None else None,
        "intention_mapping_audit": {
            "decision": intention_audit["decision"],
            "paper_mechanism_reproduced": intention_audit[
                "paper_mechanism_reproduced"
            ],
            "mapping_selected": intention_audit["mapping_selected"],
            "gamma_selected": intention_audit["gamma_selected"],
            "summary_path": str(intention_summary_path),
            "figure_path": str(intention_figure_path),
        },
        "fairness": fairness,
        "scientific_labels": {
            "muscle_activation": "MyoAssist data.act simulation muscle activation; not measured sEMG",
            "Ja": "Human Effort Proxy; not metabolic cost",
            "E_HRI_M5": "cumulative squared M5 interaction cost; not mechanical energy",
            "E_HRI_paper": "evaluation-only integral of ||Kr(a)e||^2; never injected into plant",
            "paper_40_67_percent": "not used as a MuJoCo target",
        },
    }
    summary_path = yac_directory / "yac_paper_summary.json"
    summary_path.write_text(
        json.dumps(paper_summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    figure_path = yac_directory / "yac_paper_summary.png"
    _save_paper_figure(figure_path, baseline, yac, ja_baseline, ja_yac)
    if not figure_path.is_file() or figure_path.stat().st_size == 0:
        raise RuntimeError("Paper summary figure was not created")

    print("==========================================")
    print("YAC PAPER EXPERIMENT RESULT")
    print("==========================================")
    print("\nPathology:\nBilateral vasti 60%")
    print("\n[STABILITY]")
    print(f"Baseline survival: {_format_metric(baseline['survival_s'], 2, ' s')}")
    print(f"Full YAC survival: {_format_metric(yac['survival_s'], 2, ' s')}")
    print(f"Full YAC average speed: {_format_metric(yac['average_speed_mps'], 4, ' m/s')}")
    print(f"Full YAC max pelvis pitch: {_format_metric(yac['max_abs_pelvis_pitch_deg'], 3, ' deg')}")
    print(f"Full YAC min pelvis height: {_format_metric(yac['min_pelvis_height_m'], 4, ' m')}")
    print("\n[TRACKING]")
    print(f"Baseline q RMS: {_format_metric(baseline['q_rms_overall_rad'], 6, ' rad')}")
    print(f"Full YAC q RMS: {_format_metric(yac['q_rms_overall_rad'], 6, ' rad')}")
    print(f"Tracking change: {_format_metric(tracking_change, 3, '%')}")
    print("\n[HUMAN EFFORT PROXY]")
    if not both_complete:
        print("EARLY TERMINATION — FULL-RUN EFFORT COMPARISON INVALID")
    print(f"Baseline Ja: {_format_metric(ja_baseline, 6)}")
    print(f"Full YAC Ja: {_format_metric(ja_yac, 6)}")
    print(f"Ja reduction: {_format_metric(ja_reduction, 3, '%')}")
    print("\n[pHRI INTERACTION COST]")
    print(f"Evaluation window: 2.0 s -> {_format_metric(common_end, 2, ' s')}")
    print(f"Baseline E_HRI_M5: {_format_metric(baseline['E_HRI_M5_N2m2s'], 3)}")
    print(f"Full YAC E_HRI_M5: {_format_metric(yac['E_HRI_M5_N2m2s'], 3)}")
    if hri_reduction is not None and hri_reduction < 0.0:
        print(f"HRI burden increased by {_format_metric(-hri_reduction, 3, '%')}")
    else:
        print(f"HRI reduction: {_format_metric(hri_reduction, 3, '%')}")
    print(f"Baseline HRI RMS: {_format_metric(baseline['HRI_RMS_total_Nm'], 4, ' Nm')}")
    print(f"Full YAC HRI RMS: {_format_metric(yac['HRI_RMS_total_Nm'], 4, ' Nm')}")
    print("\n[ORIGINAL PAPER-EQUIVALENT INTERACTION COST]")
    print(
        "Baseline E_HRI_paper: "
        f"{_format_metric(baseline['E_HRI_paper_N2m2s'], 3)}"
    )
    print(
        "Full YAC E_HRI_paper: "
        f"{_format_metric(yac['E_HRI_paper_N2m2s'], 3)}"
    )
    print(f"Paper HRI reduction: {_format_metric(paper_hri_reduction, 3, '%')}")
    print("\n[ADP / EVENT]")
    print(f"Global Bellman RMS: {_format_metric(yac['Bellman_RMS'], 6)}")
    print(f"Critic norm L: {_format_metric(yac['critic_norm_L'], 6)}")
    print(f"Critic norm R: {_format_metric(yac['critic_norm_R'], 6)}")
    print(f"Event count: {yac['event_count']}")
    print(f"Event fraction: {_format_metric(yac['event_fraction'], 6)}")
    print(f"Policy refresh reduction: {_format_metric(yac['policy_refresh_reduction_percent'], 3, '%')}")
    print(f"Median inter-event: {_format_metric(yac['median_inter_event_s'], 4, ' s')}")
    print("\n[ACTUATION]")
    print(f"Torque RMS: {_format_metric(yac['torque_rms_Nm'], 4, ' Nm')}")
    print(f"Torque peak: {_format_metric(yac['torque_peak_Nm'], 4, ' Nm')}")
    for motor, label in (("HipExo_L", "HipL"), ("KneeExo_L", "KneeL"), ("HipExo_R", "HipR"), ("KneeExo_R", "KneeR")):
        fraction = yac["saturation_fraction"][motor]
        print(
            f"{label} saturation: "
            f"{_format_metric(None if fraction is None else 100.0 * fraction, 3, '%')}"
        )
    print("\n[ARTIFACTS]")
    print(f"Baseline: {baseline_directory}")
    print(f"Full YAC: {yac_directory}")
    print(f"Video: {video_path}")
    print(f"Summary: {summary_path}")
    print("==========================================")
    _print_intention_mapping_audit(intention_audit)
    print(f"Intention audit: {intention_summary_path}")
    print(f"Intention figure: {intention_figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
