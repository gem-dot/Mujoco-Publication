"""Minimal shared M0 workflow behind the standard one-click entry points."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from evaluation import metrics, standard_protocols


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "results/experiments"
HEALTHY_MODEL_XML = (
    PROJECT_ROOT
    / "models/humans/myoassist/source/models/22muscle_2D/myoLeg22_2D_BASELINE.xml"
)
HEALTHY_CONTROLLER = (
    PROJECT_ROOT
    / "results/healthy_baseline/healthyhuman/healthy_controller.txt"
)
HEALTHY_PATIENT_CONFIG = PROJECT_ROOT / "models/humans/patients/healthy.yaml"
HEALTHY_REFERENCE_VALIDATION = (
    PROJECT_ROOT / "results/healthy_baseline/healthyhuman/validation.json"
)
STANDALONE_EXO_XML = (
    PROJECT_ROOT
    / "models/exoskeletons/simple_hip_knee_exo/simple_hip_knee_exo.xml"
)
FLEXIBLE_XML = (
    PROJECT_ROOT
    / "models/coupled/flexible_exo/healthy_exo_nature_flexible_v1.xml"
)
FLEXIBLE_CONTROLLER = (
    PROJECT_ROOT
    / "results/healthy_flexible_exo/controller/healthy_flexible_controller_v1.txt"
)
FLEXIBLE_XML_SHA256 = (
    "4b3190bd40a40be9c26613d99d2e32e2737ce5650df88bfba96e4935f3f83892"
)
FLEXIBLE_CONTROLLER_SHA256 = (
    "0538dfbe31bfe35cca07e2b01eea11cf5a4219c09f09dfce1ad588810ba3e536"
)
HEALTHY_FLEXIBLE_POINTER = OUTPUT_ROOT / "latest_healthy_flexible_zero.json"
YAC_HEALTHY_REFERENCE_RESULT = (
    OUTPUT_ROOT / "20260822_214246__HUMAN_EXO_HEALTHY100_ZERO"
)
FLEXIBLE_INTERFACES = ("waist", "thigh_l", "thigh_r", "shank_l", "shank_r")
REAL_EXO_CHANNELS = ("HipExo_R", "HipExo_L", "KneeExo_R", "KneeExo_L")


@dataclass(frozen=True)
class HumanCondition:
    """One public condition name mapped to one existing Patient YAML."""

    name: str
    patient_config: Path
    retained_fmax_fraction: float


HUMAN_CONDITIONS = {
    "healthy100": HumanCondition(
        "healthy100", HEALTHY_PATIENT_CONFIG, 1.0
    ),
    "mild80": HumanCondition(
        "mild80",
        PROJECT_ROOT / "models/humans/patients/left_knee_weakness/mild.yaml",
        0.8,
    ),
    "moderate60": HumanCondition(
        "moderate60",
        PROJECT_ROOT / "models/humans/patients/left_knee_weakness/moderate.yaml",
        0.6,
    ),
    "severe40": HumanCondition(
        "severe40",
        PROJECT_ROOT / "models/humans/patients/left_knee_weakness/severe.yaml",
        0.4,
    ),
}


def _safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
    if not result:
        raise ValueError("Experiment name must contain a safe character")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _condition(value: str) -> HumanCondition:
    name = value.strip().lower()
    try:
        condition = HUMAN_CONDITIONS[name]
    except KeyError as exc:
        raise ValueError(
            "HUMAN must be one of: " + ", ".join(HUMAN_CONDITIONS)
        ) from exc
    if not condition.patient_config.is_file():
        raise FileNotFoundError(
            f"Patient YAML is missing for {name}: {condition.patient_config}"
        )
    return condition


def _result_directory(validation: dict[str, Any]) -> Path:
    result = Path(validation["artifacts"]["manifest"]).parent
    if not result.is_absolute():
        result = PROJECT_ROOT / result
    return result.resolve()


def _verify_damage(
    validation: dict[str, Any], condition: HumanCondition
) -> dict[str, Any]:
    damage = validation["damage_verification"]
    retained = float(damage["retained_force_fraction"])
    expected_muscles = (
        ("vasti_l", "vasti_r")
        if condition.name == "moderate60"
        else ("vasti_l",)
    )
    actual_muscles = tuple(damage["affected_muscles"])
    muscle_details = damage["muscles"]
    checks = {
        "condition_matches": math.isclose(
            retained, condition.retained_fmax_fraction, abs_tol=1e-12
        ),
        "targets_match": actual_muscles == expected_muscles,
        "only_configured_muscles_scaled": bool(
            damage["only_configured_muscle_force_scales_changed"]
        ),
        "all_fmax_ratios_match": all(
            math.isclose(
                float(muscle_details[name]["gain_ratio"]),
                condition.retained_fmax_fraction,
                abs_tol=1e-12,
            )
            for name in expected_muscles
        ),
    }
    return {
        "human_condition": condition.name,
        "target_muscles": list(expected_muscles),
        "retained_fmax_fraction": retained,
        "retained_fmax_percent": 100.0 * retained,
        "healthy_fmax_by_muscle_N": {
            name: float(muscle_details[name]["healthy_gain_force_N"])
            for name in expected_muscles
        },
        "actual_fmax_by_muscle_N": {
            name: float(muscle_details[name]["applied_gain_force_N"])
            for name in expected_muscles
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _metric(summary: dict[str, Any], category: str, name: str) -> Any:
    return metrics.metric_value(summary["metrics"][category], name)


def _impulse_asymmetry_percent(summary: dict[str, Any]) -> float:
    left = float(_metric(summary, "grf", "left_vertical_impulse_bw_s"))
    right = float(_metric(summary, "grf", "right_vertical_impulse_bw_s"))
    denominator = abs(left) + abs(right)
    return 0.0 if denominator <= 1e-12 else 200.0 * abs(left - right) / denominator


def _m5_bounded(
    validation: dict[str, Any], summary: dict[str, Any]
) -> dict[str, Any]:
    actual_substeps = int(validation["coupling_substep_count"])
    expected_substeps = int(
        round(
            float(validation["actual_duration_s"])
            / float(validation["mujoco_timestep_s"])
        )
    )
    interfaces = set(validation["coupling_summary"]["interfaces"])
    interaction = summary["metrics"].get("interaction")
    finite_statistics = bool(
        interaction
        and not interaction.get("unavailable", False)
        and all(
            item.get("available", False)
            and isinstance(item.get("value"), (int, float))
            and math.isfinite(float(item["value"]))
            for item in interaction["metrics"].values()
        )
    )
    checks = {
        "finite_state": bool(validation["finite_state"]),
        "physics_substeps_complete": actual_substeps == expected_substeps,
        "five_interfaces_present": interfaces == set(FLEXIBLE_INTERFACES),
        "interaction_statistics_finite": finite_statistics,
    }
    return {
        "actual_physics_substeps": actual_substeps,
        "expected_physics_substeps": expected_substeps,
        "interfaces": sorted(interfaces),
        "checks": checks,
        "passed": all(checks.values()),
        "interpretation": "Numerical boundedness check, not a comfort or hardware-safety threshold.",
    }


def _verify_frozen_flexible_inputs() -> None:
    for path in (FLEXIBLE_XML, FLEXIBLE_CONTROLLER):
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen input is missing: {path}")
    if _sha256(FLEXIBLE_XML) != FLEXIBLE_XML_SHA256:
        raise ValueError("Frozen Flexible XML SHA-256 mismatch")
    if _sha256(FLEXIBLE_CONTROLLER) != FLEXIBLE_CONTROLLER_SHA256:
        raise ValueError("Healthy Flexible controller SHA-256 mismatch")


def _resolve_healthy_flexible_reference(
    value: str | Path | None,
) -> Path:
    if value is None:
        if not HEALTHY_FLEXIBLE_POINTER.is_file():
            raise FileNotFoundError(
                "No verified Healthy100 Flexible Zero reference pointer exists. "
                "Run run_04 with HUMAN='healthy100' first, or set "
                "HEALTHY_REFERENCE_RESULT explicitly."
            )
        pointer = _read_json(HEALTHY_FLEXIBLE_POINTER)
        result_dir = Path(pointer["run_directory"])
    else:
        result_dir = Path(value).expanduser()
    if not result_dir.is_absolute():
        result_dir = PROJECT_ROOT / result_dir
    result_dir = result_dir.resolve()
    manifest = _read_json(result_dir / "manifest.json")
    conditions_match = (
        manifest.get("experiment_type") == "human_exo_control"
        and manifest.get("human_condition") == "healthy100"
        and manifest.get("coupled_xml_sha256") == FLEXIBLE_XML_SHA256
        and manifest.get("human_controller_parameter_sha256")
        == FLEXIBLE_CONTROLLER_SHA256
        and manifest.get("assist_controller") == "zero"
    )
    if not conditions_match:
        raise ValueError(
            "Healthy reference must be Healthy100 Flexible Zero with the same "
            "frozen plant and Human controller"
        )
    if not (result_dir / "timeseries.csv").is_file():
        raise FileNotFoundError(
            f"Healthy reference timeseries is missing: {result_dir}"
        )
    return result_dir


def _plan_simple(experiment_name: str) -> dict[str, Any]:
    timestamp = datetime.now()
    stem = f"{timestamp:%Y%m%d_%H%M%S}__{_safe_name(experiment_name)}"
    output_directory = OUTPUT_ROOT / stem
    suffix = 1
    while output_directory.exists():
        output_directory = OUTPUT_ROOT / f"{stem}__{suffix:02d}"
        suffix += 1
    return {
        "timestamp": timestamp,
        "run_id": output_directory.name,
        "output_directory": output_directory,
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _attach_postprocessing(
    output_directory: Path,
    *,
    evaluation_requested: bool,
    evaluation_protocol: str,
    evaluation_completed: bool,
    regression: dict[str, Any] | None = None,
    pipeline_checks: dict[str, Any] | None = None,
    healthy_reference_result: Path | None = None,
) -> None:
    evaluation_dir = output_directory / "evaluation"
    evaluation = {
        "requested": evaluation_requested,
        "protocol": evaluation_protocol,
        "status": "completed" if evaluation_completed else "not_requested",
        "summary": (
            str((evaluation_dir / "summary.json").relative_to(PROJECT_ROOT))
            if evaluation_completed
            else None
        ),
        "metrics": (
            str((evaluation_dir / "evaluation.csv").relative_to(PROJECT_ROOT))
            if evaluation_completed
            else None
        ),
        "overview": (
            str((evaluation_dir / "00_overview.png").relative_to(PROJECT_ROOT))
            if evaluation_completed
            and (evaluation_dir / "00_overview.png").is_file()
            else None
        ),
        "figures": (
            str((evaluation_dir / "figures").relative_to(PROJECT_ROOT))
            if evaluation_completed and (evaluation_dir / "figures").is_dir()
            else None
        ),
        "healthy_reference_result": (
            str(healthy_reference_result.relative_to(PROJECT_ROOT))
            if healthy_reference_result is not None
            else None
        ),
    }
    for filename in ("manifest.json", "validation.json"):
        path = output_directory / filename
        document = _read_json(path)
        document["evaluation"] = evaluation
        if regression is not None:
            document["regression"] = regression
        if pipeline_checks is not None:
            document["pipeline_checks"] = pipeline_checks
        _write_json(path, document)


def _healthy_regression(output_directory: Path) -> dict[str, Any]:
    table = metrics.load_timeseries(output_directory / "timeseries.csv")
    time, _ = metrics.numeric_column(table, ["time", "time_s"])
    velocity, _ = metrics.numeric_column(
        table, ["forward_velocity", "forward_velocity_mps"]
    )
    pelvis_pitch, _ = metrics.numeric_column(
        table, ["pelvis_pitch_deg", "pelvis_pitch"]
    )
    if time is None or velocity is None or pelvis_pitch is None:
        raise ValueError("Healthy regression signals are missing from timeseries.csv")
    current = {
        "walking_duration_s": float(time[-1] - time[0]),
        "mean_forward_velocity_mps": float(np.nanmean(velocity)),
        "max_abs_pelvis_pitch_deg": float(np.nanmax(np.abs(pelvis_pitch))),
    }
    historical = _read_json(HEALTHY_REFERENCE_VALIDATION)
    historical_walking = historical["walking_metrics"]
    reference = {
        "walking_duration_s": float(historical["walking_duration_seconds"]),
        "mean_forward_velocity_mps": float(
            historical_walking["mean_forward_velocity_m_per_s"]
        ),
        "max_abs_pelvis_pitch_deg": float(
            historical_walking["max_abs_pelvis_pitch_deg"]
        ),
    }
    tolerance = {
        "walking_duration_s": 0.02,
        "mean_forward_velocity_mps": 0.10,
        "max_abs_pelvis_pitch_deg": 5.0,
    }
    differences = {
        name: abs(current[name] - reference[name]) for name in reference
    }
    checks = {
        name: differences[name] <= tolerance[name] for name in reference
    }
    return {
        "protocol": "healthy_bare_20s_regression_v1",
        "reference_validation": str(
            HEALTHY_REFERENCE_VALIDATION.relative_to(PROJECT_ROOT)
        ),
        "reference": reference,
        "current": current,
        "absolute_difference": differences,
        "tolerance": tolerance,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def run_human_only(
    *,
    human_condition: str,
    duration_s: float,
    video: bool,
    evaluation: bool,
    run_name: str | None = None,
    output_root: str | Path | None = None,
    enforce_regression: bool = True,
) -> dict[str, Any]:
    """Resolve and run the formal bare-Human L1 workflow."""
    condition = _condition(human_condition)
    config = SimpleNamespace(
        EXPERIMENT_NAME=(
            run_name or f"HUMAN_ONLY_{condition.name.upper()}"
        ),
        EXPERIMENT_TYPE="human_only",
        HUMAN_CONDITION=condition.name,
        MODEL_XML=HEALTHY_MODEL_XML,
        HUMAN_CONTROLLER_PATH=HEALTHY_CONTROLLER,
        PATIENT_CONFIG_PATH=condition.patient_config,
        HUMAN_MODEL="MyoAssist 22-muscle 2D bare Human",
        EXO_ENABLED=False,
        EXO_MODEL="none",
        EXO_MODEL_VERSION="not_applicable",
        EXO_MASS_KG=0.0,
        ASSIST_CONTROLLER=None,
        ASSIST_PARAMS={},
        DURATION_SECONDS=float(duration_s),
        TARGET_VELOCITY_MPS=1.25,
        SEED=0,
        SAVE_VIDEO=bool(video),
        VIDEO_FPS=25,
        VIDEO_FILENAME="video.mp4",
        OUTPUT_ROOT=(
            OUTPUT_ROOT
            if output_root is None
            else Path(output_root).expanduser().resolve()
        ),
        COUPLING_MODE="rigid",
        RECORD_GIT_METADATA=True,
    )
    from simulation import runner

    plan = runner.plan_experiment(config)
    runner.print_experiment_summary(config, plan)
    validation = runner.run_experiment(config, plan)
    output_directory = _result_directory(validation)
    damage = _verify_damage(validation, condition)
    if not damage["passed"]:
        raise RuntimeError(
            f"Patient damage verification failed; inspect {output_directory}"
        )

    evaluation_completed = False
    if evaluation:
        standard_protocols.evaluate_human_only(output_directory, PROJECT_ROOT)
        evaluation_completed = True
    regression = None
    if condition.name == "healthy100" and abs(float(duration_s) - 20.0) <= 1e-9:
        regression = _healthy_regression(output_directory)
    _attach_postprocessing(
        output_directory,
        evaluation_requested=bool(evaluation),
        evaluation_protocol="human_only_v1",
        evaluation_completed=evaluation_completed,
        regression=regression,
        pipeline_checks={"patient_damage": damage},
    )
    if (
        enforce_regression
        and regression is not None
        and not regression["passed"]
    ):
        raise RuntimeError(
            f"Healthy bare regression differs materially; inspect {output_directory}"
        )
    validation = _read_json(output_directory / "validation.json")
    return validation


def run_exo_only(
    *,
    test_mode: str,
    duration_s: float,
    video: bool,
    evaluation: bool,
) -> dict[str, Any]:
    """Resolve and run the formal standalone Exo L2 workflow."""
    config = SimpleNamespace(
        EXPERIMENT_NAME="EXO_ONLY_ZERO",
        EXPERIMENT_TYPE="exo_only",
        MODEL_XML=STANDALONE_EXO_XML,
        EXO_MODEL="8 kg bilateral hip-knee exoskeleton",
        EXO_MODEL_VERSION="simple_hip_knee_exo_v1",
        TEST_MODE=test_mode,
        DURATION_SECONDS=float(duration_s),
        SAVE_VIDEO=bool(video),
        VIDEO_FPS=25,
        OUTPUT_ROOT=OUTPUT_ROOT,
    )
    plan = _plan_simple(config.EXPERIMENT_NAME)
    print("=" * 40)
    print("EXPERIMENT")
    print("=" * 40)
    print(f"Experiment:       {config.EXPERIMENT_NAME}")
    print(f"Type:             {config.EXPERIMENT_TYPE}")
    print(f"XML:              {config.MODEL_XML}")
    print("Fixture:          fixed pelvis/root; four hip/knee hinges free")
    print(f"Test mode:        {config.TEST_MODE}")
    print(f"Duration:         {config.DURATION_SECONDS:g} s")
    print(f"Save video:       {config.SAVE_VIDEO}")
    print(f"Output:           {plan['output_directory']}")
    print("=" * 40)

    from simulation import exo_only_runner

    validation = exo_only_runner.run_experiment(config, plan)
    output_directory = Path(plan["output_directory"]).resolve()
    evaluation_completed = False
    if evaluation:
        standard_protocols.evaluate_exo_only(output_directory)
        evaluation_completed = True
    _attach_postprocessing(
        output_directory,
        evaluation_requested=bool(evaluation),
        evaluation_protocol="exo_only_test_bench_v1",
        evaluation_completed=evaluation_completed,
    )
    return _read_json(output_directory / "validation.json")


def _make_flexible_config(
    condition: HumanCondition,
    *,
    assistance: str,
    duration_s: float,
    video: bool,
    run_name: str | None,
    healthy_reference_result: str | Path | None = None,
    controller_mode: str | None = None,
    active_scale: float | None = None,
    adp_scale: float | None = None,
    event_trigger: bool = False,
    event_parameters: dict[str, float] | None = None,
    intention_mode: str = "live_activation",
    output_root: str | Path | None = None,
    viewer: bool = False,
    viewer_keep_open: bool = False,
    record_git_metadata: bool = True,
    ppc_rho0: float | None = None,
    ppc_rho_inf: float | None = None,
    ppc_t_rho_s: float | None = None,
    pt_tp_s: float | None = None,
) -> Any:
    if assistance not in ("zero", "yac", "icra"):
        if assistance == "basic":
            raise ValueError(
                "ASSISTANCE='basic' is not READY on the frozen Flexible plant; "
                "the existing Basic controller is verified only on the legacy "
                "Rigid configuration."
            )
        raise ValueError("ASSISTANCE must be 'zero', 'yac', or 'icra'")
    _verify_frozen_flexible_inputs()
    from experiments.m5_waist_ap_calibration import make_ap_candidate
    from simulation.waist_rotation_parameters import (
        OUR_HIP_KNEE_WAIST_ROTATION_V1_1,
        OUR_HIP_KNEE_WAIST_TRANSLATION_V1,
    )

    rotation = OUR_HIP_KNEE_WAIST_ROTATION_V1_1
    translation = OUR_HIP_KNEE_WAIST_TRANSLATION_V1
    frozen_parameters = (
        rotation.alpha_pitch_nm_per_rad,
        rotation.beta_pitch_nms_per_rad,
        translation.k_ap_n_per_m,
        translation.c_ap_ns_per_m,
    )
    if frozen_parameters != (165.0, 13.5, 1500.0, 60.0):
        raise ValueError(f"Frozen M5 parameter mismatch: {frozen_parameters}")

    config = make_ap_candidate(
        translation.k_ap_n_per_m,
        translation.c_ap_ns_per_m,
        float(duration_s),
        parameter_set=translation.name,
        output_root=(
            OUTPUT_ROOT
            if output_root is None
            else Path(output_root).expanduser().resolve()
        ),
        save_video=bool(video),
    )
    config.EXPERIMENT_NAME = (
        run_name
        or f"HUMAN_EXO_{condition.name.upper()}_{assistance.upper()}"
    )
    config.EXPERIMENT_TYPE = "human_exo_control"
    config.HUMAN_CONDITION = condition.name
    config.PATIENT_CONFIG_PATH = condition.patient_config
    config.HUMAN_CONTROLLER_PATH = FLEXIBLE_CONTROLLER
    config.HUMAN_CONTROLLER_TYPE = "reflex"
    config.HUMAN_CONTROLLER_OPTIMIZED_FOR = (
        "healthy_flexible_exo_zero_torque"
    )
    config.SOURCE_INITIALIZATION = "healthy_exo_corrected_controller"
    if assistance == "zero":
        config.ASSISTANCE_NAME = "zero"
        config.ASSIST_CONTROLLER = None
        config.ASSIST_PARAMS = {}
        config.CONTROLLER_MODE = None
        config.ACTIVE_ASSISTANCE = False
        config.CONTROLLED_DOFS = ()
        config.REFERENCE_SOURCE = None
        config.INTENTION_SOURCE = None
    elif assistance == "icra":
        normalized_mode = str(controller_mode or "").strip().lower()
        allowed_icra_modes = (
            "shadow_plant_scheduler",
            "fixed_k_shadow",
            "fixed_k_nominal",
            "fixed_k_adp",
            "game_shadow",
            "performance_shadow",
            "icra_impedance_baseline",
            "critic_nash_shadow",
            "continuous_active_game",
            "calibrated_scalar_game_shadow",
            "physical_actuation_game_shadow",
            "physical_actuation_game_active",
            "intention_target_game_shadow",
            "intention_target_game_active",
            "event_triggered_intention_target_game_active",
            "conservative_event_triggered_intention_target_game_active",
            "dwell_limited_p25_event_triggered_intention_target_game_active",
            "residual_event_triggered_intention_target_game_active",
            "ppc_timing_characterization",
            "pt_horizon_characterization",
        )
        if normalized_mode not in allowed_icra_modes:
            raise ValueError(
                "Unsupported explicit ICRA controller_mode"
            )
        if healthy_reference_result is None:
            raise ValueError("ICRA requires the fixed Healthy reference")
        reference_result = Path(healthy_reference_result).expanduser().resolve()
        expected_reference = YAC_HEALTHY_REFERENCE_RESULT.resolve()
        if reference_result != expected_reference or not reference_result.is_dir():
            raise ValueError(
                "ICRA reference must be the fixed validated Healthy result: "
                f"{expected_reference}"
            )
        from controllers.ICRA import ICRAController

        config.ASSISTANCE_NAME = {
            "shadow_plant_scheduler": "ICRA",
            "fixed_k_shadow": "Fixed-K Shadow",
            "fixed_k_nominal": "Fixed-K Nominal",
            "fixed_k_adp": "Fixed-K ADP",
            "game_shadow": "ICRA Game Shadow",
            "performance_shadow": "ICRA Performance Shadow",
            "icra_impedance_baseline": "ICRA Impedance Baseline",
            "critic_nash_shadow": "ICRA Critic Nash Shadow",
            "continuous_active_game": "ICRA Continuous Active IS-CGF",
            "calibrated_scalar_game_shadow": (
                "ICRA Calibrated Scalar Game Shadow"
            ),
            "physical_actuation_game_shadow": (
                "ICRA Physical-Actuation Game Shadow"
            ),
            "physical_actuation_game_active": (
                "ICRA Physical-Actuation Game Active"
            ),
            "intention_target_game_shadow": "ICRA Intention-Target Game Shadow",
            "intention_target_game_active": "ICRA Intention-Target Game Active",
            "event_triggered_intention_target_game_active": (
                "ICRA Event-Triggered Intention-Target Game Active"
            ),
            "conservative_event_triggered_intention_target_game_active": (
                "ICRA Conservative Event-Triggered Intention-Target Game Active"
            ),
            "dwell_limited_p25_event_triggered_intention_target_game_active": (
                "ICRA Dwell-Limited p25 Event-Triggered Game Active"
            ),
            "residual_event_triggered_intention_target_game_active": (
                "ICRA Residual Event-Triggered Game Active"
            ),
            "ppc_timing_characterization": "ICRA Stage8A PPC Timing Characterization",
            "pt_horizon_characterization": "ICRA Stage8A PT Horizon Characterization",
        }[normalized_mode]
        config.ASSIST_CONTROLLER = ICRAController
        config.ASSIST_REQUIRES_MUJOCO_MODEL = True
        config.ASSIST_PARAMS = {
            "reference_result": str(reference_result),
            "mode": normalized_mode,
        }
        for key, value in {
            "ppc_rho0": ppc_rho0, "ppc_rho_inf": ppc_rho_inf,
            "ppc_t_rho_s": ppc_t_rho_s, "pt_tp_s": pt_tp_s,
        }.items():
            if value is not None:
                config.ASSIST_PARAMS[key] = float(value)
        config.CONTROLLER_MODE = normalized_mode
        config.YAC_MODE = None
        config.ACTIVE_ASSISTANCE = normalized_mode in (
            "fixed_k_nominal",
            "fixed_k_adp",
            "game_shadow",
            "performance_shadow",
            "icra_impedance_baseline",
            "critic_nash_shadow",
            "continuous_active_game",
            "physical_actuation_game_active",
            "intention_target_game_active",
            "event_triggered_intention_target_game_active",
            "conservative_event_triggered_intention_target_game_active",
            "dwell_limited_p25_event_triggered_intention_target_game_active",
            "residual_event_triggered_intention_target_game_active",
            "ppc_timing_characterization",
            "pt_horizon_characterization",
        )
        config.CONTROLLED_DOFS = ("HipExo_L", "KneeExo_L")
        config.REFERENCE_SOURCE = str(reference_result.relative_to(PROJECT_ROOT))
        config.INTENTION_SOURCE = "MyoAssist_data.act_simulation_activation"
        config.TECHNICAL_MIGRATION_STAGE = {
            "shadow_plant_scheduler": "ICRA-STAGE-1",
            "fixed_k_shadow": "ICRA-STAGE-2R",
            "fixed_k_nominal": "ICRA-STAGE-2R",
            "fixed_k_adp": "ICRA-STAGE-2R",
            "game_shadow": "ICRA-STAGE-3",
            "performance_shadow": "ICRA-STAGE-4R",
            "icra_impedance_baseline": "ICRA-STAGE-5R",
            "critic_nash_shadow": "ICRA-STAGE-6A",
            "continuous_active_game": "ICRA-STAGE-6B",
            "calibrated_scalar_game_shadow": "ICRA-STAGE-6E",
            "physical_actuation_game_shadow": "ICRA-STAGE-6F-V2",
            "physical_actuation_game_active": "ICRA-STAGE-6F-V2",
            "intention_target_game_shadow": "ICRA-STAGE-6G",
            "intention_target_game_active": "ICRA-STAGE-6G",
            "event_triggered_intention_target_game_active": "ICRA-STAGE-7",
            "conservative_event_triggered_intention_target_game_active": "ICRA-STAGE-7R",
            "dwell_limited_p25_event_triggered_intention_target_game_active": "ICRA-STAGE-7S",
            "residual_event_triggered_intention_target_game_active": "ICRA-STAGE-7T",
            "ppc_timing_characterization": "ICRA-STAGE-8A-PPC",
            "pt_horizon_characterization": "ICRA-STAGE-8A-PT",
        }[normalized_mode]
        config.INTERACTION_SEMANTICS = "M5_decomposed_once_in_reduced_plant"
        if normalized_mode.startswith("fixed_k_") or normalized_mode in (
            "game_shadow",
            "performance_shadow",
            "icra_impedance_baseline",
            "critic_nash_shadow",
            "continuous_active_game",
            "calibrated_scalar_game_shadow",
            "physical_actuation_game_shadow",
            "physical_actuation_game_active",
            "intention_target_game_shadow",
            "intention_target_game_active",
            "event_triggered_intention_target_game_active",
            "conservative_event_triggered_intention_target_game_active",
            "dwell_limited_p25_event_triggered_intention_target_game_active",
            "residual_event_triggered_intention_target_game_active",
            "ppc_timing_characterization",
            "pt_horizon_characterization",
        ):
            from controllers.ICRA import (
                FIXED_ADP_KAPPA,
                FIXED_ADP_Q,
                FIXED_ADP_R,
                FIXED_ADP_Z_SCALE_RAD_S,
            )

            config.Z_DAMPING_RATE_S_INV = 5.0
            config.ENGAGEMENT_RAMP_DURATION_S = 0.25
            config.TOTAL_TORQUE_LIMIT_NM = None
            config.ADP_RESIDUAL_LIMIT_NM = None
            config.ADP_RESIDUAL_SOFTWARE_CLIPPING = False
            config.SOFTWARE_TOTAL_TORQUE_CLIPPING = False
            fixed_b3_critic_enabled = normalized_mode in (
                "fixed_k_adp",
                "game_shadow",
                "performance_shadow",
            )
            stage6b_critic_enabled = normalized_mode in (
                "continuous_active_game",
                "calibrated_scalar_game_shadow",
                "physical_actuation_game_shadow",
                "physical_actuation_game_active",
                "intention_target_game_shadow",
                "intention_target_game_active",
                "event_triggered_intention_target_game_active",
                "conservative_event_triggered_intention_target_game_active",
                "dwell_limited_p25_event_triggered_intention_target_game_active",
                "residual_event_triggered_intention_target_game_active",
                "ppc_timing_characterization",
                "pt_horizon_characterization",
            )
            config.CRITIC_BASIS = (
                "complete_quadratic_78"
                if stage6b_critic_enabled
                else "B3" if fixed_b3_critic_enabled else None
            )
            config.CRITIC_INPUT = (
                "normalized_chi_12"
                if stage6b_critic_enabled
                else "normalized_z" if fixed_b3_critic_enabled else None
            )
            config.CRITIC_WEIGHT_DIMENSION = (
                [78, 2]
                if stage6b_critic_enabled
                else 3 if fixed_b3_critic_enabled else None
            )
            config.CRITIC_Z_SCALE_RAD_S = (
                None
                if stage6b_critic_enabled
                else FIXED_ADP_Z_SCALE_RAD_S.tolist()
            )
            config.CRITIC_Q = (
                "scheduled_QA_QI_SA_SI_plus_Qepsilon"
                if stage6b_critic_enabled
                else FIXED_ADP_Q.tolist()
            )
            config.CRITIC_R = (
                "theory_scheduled_RN_alpha"
                if stage6b_critic_enabled
                else FIXED_ADP_R.tolist()
            )
            config.CRITIC_KAPPA = (
                None if stage6b_critic_enabled else FIXED_ADP_KAPPA
            )
            config.CRITIC_ETA_C = 0.05 if stage6b_critic_enabled else None
            config.CRITIC_WEIGHT_RADIUS = (
                3.66457347596e-4 if stage6b_critic_enabled else None
            )
            config.CRITIC_D_CHI_DIAGONAL = (
                [
                    .0468633048914, .0468633048914,
                    .488526929055, .488526929055,
                    .199772901089, .199772901089,
                    3.17281678936, 3.17281678936,
                    .143612797730, .143612797730,
                    .138624223405, .138624223405,
                ]
                if stage6b_critic_enabled else None
            )
            config.CRITIC_UPDATE_MODE = (
                "continuous_100hz_causal_projected_euler"
                if stage6b_critic_enabled
                else "continuous_100hz_pending_aligned"
                if fixed_b3_critic_enabled
                else None
            )
            config.CRITIC_UPDATE_SEMANTICS = (
                "Wk_actor_and_command_then_project_Wk_plus_1"
                if stage6b_critic_enabled
                else "qacc_at_k_completes_interval_k_minus_1"
                if fixed_b3_critic_enabled
                else None
            )
            config.EXPERIENCE_REPLAY = False
            config.EVENT_TRIGGER = normalized_mode in (
                "event_triggered_intention_target_game_active",
                "conservative_event_triggered_intention_target_game_active",
                "dwell_limited_p25_event_triggered_intention_target_game_active",
                "residual_event_triggered_intention_target_game_active",
            )
            config.EVENT_TRIGGER_PARAMETERS = (
                {
                    "candidate": (
                        "p25_Td20ms_residual"
                        if normalized_mode == "residual_event_triggered_intention_target_game_active"
                        else "p25_Td20ms"
                        if normalized_mode == "dwell_limited_p25_event_triggered_intention_target_game_active"
                        else "p50_Td20ms"
                        if normalized_mode == "conservative_event_triggered_intention_target_game_active"
                        else "p75_Td20ms"
                    ),
                    "theta0": (
                        0.0003968219500891493
                        if normalized_mode == "residual_event_triggered_intention_target_game_active"
                        else 3.649642896138322e-06
                        if normalized_mode == "dwell_limited_p25_event_triggered_intention_target_game_active"
                        else 0.003704956407942091
                        if normalized_mode == "conservative_event_triggered_intention_target_game_active"
                        else 0.03785510066288972
                    ),
                    "dwell_s": 0.02,
                    "lambda_v_s_inv": 1.0 / 3.45,
                    "cv": 1.0,
                    "kappa_e": (
                        0.25 * 0.0003968219500891493 / (1.041452234785662 + 1.0e-12)
                        if normalized_mode == "residual_event_triggered_intention_target_game_active"
                        else 0.25 * 3.649642896138322e-06 / (1.041452234785662 + 1.0e-12)
                        if normalized_mode == "dwell_limited_p25_event_triggered_intention_target_game_active"
                        else 0.000889372619355183
                        if normalized_mode == "conservative_event_triggered_intention_target_game_active"
                        else 0.009087094779398164
                    ),
                    "c_chi_evt": (
                        (1.0 / 3.45) * 0.25 * 0.0003968219500891493 / (1.041452234785662 + 1.0e-12)
                        if normalized_mode == "residual_event_triggered_intention_target_game_active"
                        else (1.0 / 3.45) * 0.25 * 3.649642896138322e-06 / (1.041452234785662 + 1.0e-12)
                        if normalized_mode == "dwell_limited_p25_event_triggered_intention_target_game_active"
                        else 0.0002577891650304878
                        if normalized_mode == "conservative_event_triggered_intention_target_game_active"
                        else 0.0026339405157675844
                    ),
                    "strategy_scale_nm": (
                        [
                            0.014950184223183957, 0.0024861161589295063,
                            0.00364147970186807, 0.0027713121094356035,
                        ]
                        if normalized_mode == "residual_event_triggered_intention_target_game_active"
                        else [
                            1.2107171177701537, 0.4557594590456351,
                            1.2570709247766185, 0.4852210652602563,
                        ]
                    ),
                }
                if config.EVENT_TRIGGER else {}
            )
            config.EVENT_TRIGGER_POLICY_SEMANTICS = (
                "continuous_affine_hold_only_learned_nash_residual"
                if normalized_mode == "residual_event_triggered_intention_target_game_active"
                else "hold_only_accepted_nash_strategy_mu0_and_critic_continuous"
                if config.EVENT_TRIGGER else None
            )
            config.STRICT_FULL_TORQUE_ZOH = False
    else:
        normalized_mode = str(controller_mode or "").strip().lower()
        allowed_modes = (
            "active_nominal_flexible_hri",
            "continuous_adp",
            "full_yac",
            "bilateral_nominal",
            "bilateral_full_yac",
        )
        if normalized_mode not in allowed_modes:
            raise ValueError("Unsupported explicit YAC controller_mode")
        if healthy_reference_result is None:
            raise ValueError("YAC Stage D requires the fixed Healthy reference result")
        if active_scale is None:
            raise ValueError("YAC Stage D requires an explicit active_scale")
        reference_result = Path(healthy_reference_result).expanduser().resolve()
        expected_reference = YAC_HEALTHY_REFERENCE_RESULT.resolve()
        if reference_result != expected_reference:
            raise ValueError(
                "YAC Stage D reference must be the fixed validated Healthy result: "
                f"{expected_reference}"
            )
        if not reference_result.is_dir():
            raise FileNotFoundError(
                f"YAC Healthy reference result is missing: {reference_result}"
            )
        from controllers.YAC import (
            CRITIC_UPDATE_SEMANTICS,
            EVENT_ETA_DISCRETIZATION,
            EVENT_ETA_INITIAL,
            EVENT_FORMULA,
            EVENT_PARAMETER_SOURCE,
            EVENT_POLICY_SEMANTICS,
            STRICT_FULL_TORQUE_ZOH,
            YACController,
        )

        config.ASSISTANCE_NAME = "YAC"
        config.ASSIST_CONTROLLER = YACController
        config.ASSIST_REQUIRES_MUJOCO_MODEL = True
        normalized_event_parameters = dict(event_parameters or {})
        if normalized_mode in ("full_yac", "bilateral_full_yac"):
            required_event = {
                "lambda_eta",
                "sigma_eta",
                "xi",
                "rho_delta",
                "epsilon_delta",
            }
            missing_event = sorted(
                required_event - set(normalized_event_parameters)
            )
            if not event_trigger or missing_event:
                raise ValueError(
                    "Full YAC requires event_trigger=True and parameters: "
                    + ", ".join(sorted(required_event))
                )
        elif event_trigger:
            raise ValueError("Event Trigger is active only in full_yac mode")
        config.ASSIST_PARAMS = {
            "reference_result": str(reference_result),
            "mode": normalized_mode,
            "active_scale": float(active_scale),
            "adp_scale": 1.0 if adp_scale is None else float(adp_scale),
            "event_trigger": bool(event_trigger),
            "event_lambda_eta": normalized_event_parameters.get("lambda_eta"),
            "event_sigma_eta": normalized_event_parameters.get("sigma_eta"),
            "event_xi": normalized_event_parameters.get("xi"),
            "event_rho_delta": normalized_event_parameters.get("rho_delta"),
            "event_epsilon_delta": normalized_event_parameters.get(
                "epsilon_delta"
            ),
            "intention_mode": str(intention_mode),
        }
        config.CONTROLLER_MODE = normalized_mode
        config.YAC_MODE = normalized_mode
        config.ACTIVE_ASSISTANCE = True
        config.CONTROLLED_DOFS = (
            "HipExo_L",
            "KneeExo_L",
            "HipExo_R",
            "KneeExo_R",
        )
        config.REFERENCE_SOURCE = str(reference_result.relative_to(PROJECT_ROOT))
        config.INTENTION_SOURCE = "simulation_muscle_activation_proxy"
        config.YAC_STAGE = {
            "active_nominal_flexible_hri": "D-R",
            "continuous_adp": "E1-UNCLIPPED",
            "full_yac": "FULL-YAC-DEMO",
            "bilateral_nominal": "BILATERAL-NOMINAL",
            "bilateral_full_yac": "BILATERAL-FULL-YAC",
        }[normalized_mode]
        config.TECHNICAL_MIGRATION_STAGE = config.YAC_STAGE
        config.INTERACTION_SEMANTICS = "internal_m5_sensed_not_cancelled"
        config.PAPER_EXACT_TAU_CANCELLATION = False
        config.PAPER_CANDIDATE_RECORDED = True
        config.Z_DAMPING_RATE_S_INV = 5.0
        config.YAC_ACTIVE_SCALE = float(active_scale)
        config.YAC_ADP_SCALE = (
            1.0 if adp_scale is None else float(adp_scale)
        )
        config.ENGAGEMENT_RAMP_DURATION_S = 0.25
        config.STAGE_D_TORQUE_LIMIT_NM = None
        config.TOTAL_TORQUE_LIMIT_NM = None
        config.ADP_RESIDUAL_LIMIT_NM = None
        config.ADP_RESIDUAL_SOFTWARE_CLIPPING = False
        config.SOFTWARE_TOTAL_TORQUE_CLIPPING = False
        config.CRITIC_BASIS = "B6"
        config.CRITIC_INPUT = "normalized_z_plus_activation"
        config.CRITIC_WEIGHT_DIMENSION = 6
        config.CRITIC_Z_SCALE_RAD_S = [
            3.534755545283957,
            5.203347831091454,
        ]
        config.CRITIC_Q = [
            [0.08003523621759472, 0.0],
            [0.0, 0.03693467511385685],
        ]
        # Keep manifest metadata identical to controllers/YAC.py calibration.
        config.CRITIC_R = [[0.04, 0.0], [0.0, 0.25]]
        config.CRITIC_KAPPA = 1.0
        config.CRITIC_UPDATE_MODE = "continuous_100hz"
        config.CRITIC_UPDATE_SEMANTICS = CRITIC_UPDATE_SEMANTICS
        config.EXPERIENCE_REPLAY = False
        config.EVENT_TRIGGER = bool(event_trigger)
        config.EVENT_TRIGGER_POLICY_SEMANTICS = (
            EVENT_POLICY_SEMANTICS if event_trigger else None
        )
        config.EVENT_TRIGGER_PARAMETERS = normalized_event_parameters
        config.EVENT_FORMULA = EVENT_FORMULA if event_trigger else None
        config.EVENT_LAMBDA_ETA = normalized_event_parameters.get("lambda_eta")
        config.EVENT_SIGMA_ETA = normalized_event_parameters.get("sigma_eta")
        config.EVENT_XI = normalized_event_parameters.get("xi")
        config.EVENT_RHO_DELTA = normalized_event_parameters.get("rho_delta")
        config.EVENT_EPSILON_DELTA = normalized_event_parameters.get(
            "epsilon_delta"
        )
        config.EVENT_ETA_INITIAL = EVENT_ETA_INITIAL if event_trigger else None
        config.EVENT_ETA_DISCRETIZATION = (
            EVENT_ETA_DISCRETIZATION if event_trigger else None
        )
        config.EVENT_PARAMETER_SOURCE = (
            EVENT_PARAMETER_SOURCE if event_trigger else None
        )
        config.STRICT_FULL_TORQUE_ZOH = STRICT_FULL_TORQUE_ZOH
        config.PHASE_EWMA_BETA = 0.2
    config.WAIST_ROTATION_PARAMETER_SET = rotation.name
    config.WAIST_ROTATION_PARAMETER_SOURCE = rotation.parameter_source
    config.WAIST_TRANSLATION_PARAMETER_SOURCE = translation.parameter_source
    config.STRAP_VISUALIZATION = bool(video)
    config.STRAP_HUD = bool(video)
    config.STRAP_FORCE_VISUAL_REFERENCE_N = 200.0
    config.VIDEO_FILENAME = "video.mp4"
    config.VIEWER = bool(viewer)
    config.VIEWER_KEEP_OPEN = bool(viewer_keep_open)
    config.RECORD_GIT_METADATA = bool(record_git_metadata)
    return config


def _verify_flexible_result(
    validation: dict[str, Any],
    condition: HumanCondition,
    assistance: str,
    output_directory: Path,
) -> dict[str, Any]:
    damage = _verify_damage(validation, condition)
    exo_zero = validation["exoskeleton_channels_all_zero"]
    zero_required = assistance == "zero" or not bool(
        validation.get("active_assistance", True)
    )
    checks = {
        "patient_damage": damage["passed"],
        "frozen_flexible_xml": (
            validation["coupled_xml_sha256"] == FLEXIBLE_XML_SHA256
        ),
        "frozen_human_controller": (
            validation["human_controller_parameter_sha256"]
            == FLEXIBLE_CONTROLLER_SHA256
        ),
        "five_interface_topology": (
            set(validation["coupling_summary"]["interfaces"])
            == set(FLEXIBLE_INTERFACES)
        ),
        "zero_command_and_actual_torque": (
            not zero_required
            or all(exo_zero[name] is True for name in REAL_EXO_CHANNELS)
        ),
    }
    required_artifacts = (
        "manifest.json",
        "timeseries.csv",
        "validation.json",
        "physics_timeseries_1khz.csv",
    )
    checks["required_result_artifacts"] = all(
        (output_directory / name).is_file()
        and (output_directory / name).stat().st_size > 0
        for name in required_artifacts
    )
    if assistance == "yac":
        diagnostics = validation.get("artifacts", {}).get(
            "controller_diagnostics"
        )
        expected_stage = {
            "active_nominal_flexible_hri": "D-R",
            "continuous_adp": "E1-UNCLIPPED",
            "full_yac": "FULL-YAC-DEMO",
            "bilateral_nominal": "BILATERAL-NOMINAL",
            "bilateral_full_yac": "BILATERAL-FULL-YAC",
        }.get(validation.get("controller_mode"))
        checks["yac_controller_manifest"] = bool(
            validation.get("assist_controller") == "YAC"
            and validation.get("controller_mode")
            in (
                "active_nominal_flexible_hri",
                "continuous_adp",
                "full_yac",
                "bilateral_nominal",
                "bilateral_full_yac",
            )
            and validation.get("active_assistance") is True
            and validation.get("yac_stage") == expected_stage
            and validation.get("interaction_semantics")
            == "internal_m5_sensed_not_cancelled"
            and validation.get("paper_exact_tau_cancellation") is False
            and validation.get("paper_candidate_recorded") is True
        )
        checks["controller_diagnostics"] = bool(
            diagnostics
            and (PROJECT_ROOT / diagnostics).is_file()
            and (PROJECT_ROOT / diagnostics).stat().st_size > 0
        )
    elif assistance == "icra":
        diagnostics = validation.get("artifacts", {}).get(
            "controller_diagnostics"
        )
        icra_mode = validation.get("controller_mode")
        expected_active = icra_mode in (
            "fixed_k_nominal",
            "fixed_k_adp",
            "game_shadow",
            "performance_shadow",
            "icra_impedance_baseline",
            "critic_nash_shadow",
            "continuous_active_game",
            "physical_actuation_game_active",
            "intention_target_game_active",
            "event_triggered_intention_target_game_active",
            "conservative_event_triggered_intention_target_game_active",
            "dwell_limited_p25_event_triggered_intention_target_game_active",
            "residual_event_triggered_intention_target_game_active",
            "ppc_timing_characterization",
            "pt_horizon_characterization",
        )
        expected_assistance = {
            "shadow_plant_scheduler": "ICRA",
            "fixed_k_shadow": "Fixed-K Shadow",
            "fixed_k_nominal": "Fixed-K Nominal",
            "fixed_k_adp": "Fixed-K ADP",
            "game_shadow": "ICRA Game Shadow",
            "performance_shadow": "ICRA Performance Shadow",
            "icra_impedance_baseline": "ICRA Impedance Baseline",
            "critic_nash_shadow": "ICRA Critic Nash Shadow",
            "continuous_active_game": "ICRA Continuous Active IS-CGF",
            "calibrated_scalar_game_shadow": (
                "ICRA Calibrated Scalar Game Shadow"
            ),
            "physical_actuation_game_shadow": (
                "ICRA Physical-Actuation Game Shadow"
            ),
            "physical_actuation_game_active": (
                "ICRA Physical-Actuation Game Active"
            ),
            "intention_target_game_shadow": "ICRA Intention-Target Game Shadow",
            "intention_target_game_active": "ICRA Intention-Target Game Active",
            "event_triggered_intention_target_game_active": (
                "ICRA Event-Triggered Intention-Target Game Active"
            ),
            "conservative_event_triggered_intention_target_game_active": (
                "ICRA Conservative Event-Triggered Intention-Target Game Active"
            ),
            "dwell_limited_p25_event_triggered_intention_target_game_active": (
                "ICRA Dwell-Limited p25 Event-Triggered Game Active"
            ),
            "residual_event_triggered_intention_target_game_active": (
                "ICRA Residual Event-Triggered Game Active"
            ),
            "ppc_timing_characterization": "ICRA Stage8A PPC Timing Characterization",
            "pt_horizon_characterization": "ICRA Stage8A PT Horizon Characterization",
        }.get(icra_mode)
        checks["icra_shadow_manifest"] = bool(
            validation.get("assist_controller") == expected_assistance
            and icra_mode
            in (
                "shadow_plant_scheduler",
                "fixed_k_shadow",
                "fixed_k_nominal",
                "fixed_k_adp",
                "game_shadow",
                "performance_shadow",
                "icra_impedance_baseline",
                "critic_nash_shadow",
                "continuous_active_game",
                "calibrated_scalar_game_shadow",
                "physical_actuation_game_shadow",
                "physical_actuation_game_active",
                "intention_target_game_shadow",
                "intention_target_game_active",
                "event_triggered_intention_target_game_active",
                "conservative_event_triggered_intention_target_game_active",
                "dwell_limited_p25_event_triggered_intention_target_game_active",
                "residual_event_triggered_intention_target_game_active",
                "ppc_timing_characterization",
                "pt_horizon_characterization",
            )
            and validation.get("active_assistance") is expected_active
            and validation.get("technical_migration_stage")
            == {
                "shadow_plant_scheduler": "ICRA-STAGE-1",
                "fixed_k_shadow": "ICRA-STAGE-2R",
                "fixed_k_nominal": "ICRA-STAGE-2R",
                "fixed_k_adp": "ICRA-STAGE-2R",
                "game_shadow": "ICRA-STAGE-3",
                "performance_shadow": "ICRA-STAGE-4R",
                "icra_impedance_baseline": "ICRA-STAGE-5R",
                "critic_nash_shadow": "ICRA-STAGE-6A",
                "continuous_active_game": "ICRA-STAGE-6B",
                "calibrated_scalar_game_shadow": "ICRA-STAGE-6E",
                "physical_actuation_game_shadow": "ICRA-STAGE-6F-V2",
                "physical_actuation_game_active": "ICRA-STAGE-6F-V2",
                "intention_target_game_shadow": "ICRA-STAGE-6G",
                "intention_target_game_active": "ICRA-STAGE-6G",
                "event_triggered_intention_target_game_active": "ICRA-STAGE-7",
                "conservative_event_triggered_intention_target_game_active": "ICRA-STAGE-7R",
                "dwell_limited_p25_event_triggered_intention_target_game_active": "ICRA-STAGE-7S",
                "residual_event_triggered_intention_target_game_active": "ICRA-STAGE-7T",
                "ppc_timing_characterization": "ICRA-STAGE-8A-PPC",
                "pt_horizon_characterization": "ICRA-STAGE-8A-PT",
            }.get(icra_mode)
            and validation.get("event_trigger")
            in (
                (True, None)
                if icra_mode in (
                    "event_triggered_intention_target_game_active",
                    "conservative_event_triggered_intention_target_game_active",
                    "dwell_limited_p25_event_triggered_intention_target_game_active",
                    "residual_event_triggered_intention_target_game_active",
                )
                else (False, None)
            )
        )
        checks["controller_diagnostics"] = bool(
            diagnostics
            and (PROJECT_ROOT / diagnostics).is_file()
            and (PROJECT_ROOT / diagnostics).stat().st_size > 0
        )
    if validation["save_video"]:
        checks["video"] = (
            (output_directory / "video.mp4").is_file()
            and (output_directory / "video.mp4").stat().st_size > 0
        )
    return {
        "patient_damage": damage,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _human_exo_regression(
    condition: HumanCondition,
    validation: dict[str, Any],
    summary: dict[str, Any],
    m5: dict[str, Any],
) -> dict[str, Any] | None:
    if not math.isclose(
        float(validation["requested_duration_s"]), 20.0, abs_tol=1e-12
    ):
        return None
    duration = float(validation["actual_duration_s"])
    speed = float(_metric(summary, "stability", "mean_forward_velocity_mps"))
    exo_zero = validation["exoskeleton_channels_all_zero"]
    zero_verified = all(exo_zero[name] is True for name in REAL_EXO_CHANNELS)
    current = {
        "duration_s": duration,
        "mean_forward_velocity_mps": speed,
        "exo_zero_torque_verified": zero_verified,
        "m5_bounded": m5["passed"],
    }
    checks = {
        "exo_zero_torque": zero_verified,
        "m5_bounded": m5["passed"],
    }
    reference: dict[str, Any]
    tolerance: dict[str, Any]
    if condition.name == "healthy100":
        reference = {
            "duration_s": 20.00000000000146,
            "mean_forward_velocity_mps": 1.2531296055321168,
        }
        tolerance = {"duration_s": 0.02, "mean_forward_velocity_mps": 0.10}
        checks.update(
            {
                "duration": abs(duration - reference["duration_s"])
                <= tolerance["duration_s"],
                "speed": abs(speed - reference["mean_forward_velocity_mps"])
                <= tolerance["mean_forward_velocity_mps"],
                "completed_20s": bool(validation["success"]),
            }
        )
    elif condition.name == "mild80":
        impulse_asymmetry = _impulse_asymmetry_percent(summary)
        step_symmetry = float(
            _metric(summary, "gait", "step_length_symmetry_index_percent")
        )
        current.update(
            {
                "vertical_impulse_asymmetry_percent": impulse_asymmetry,
                "step_length_symmetry_index_percent": step_symmetry,
            }
        )
        reference = {
            "duration_s": 20.00000000000146,
            "vertical_impulse_asymmetry_percent": 29.340617417968687,
            "step_length_symmetry_index_percent": 13.972260998657045,
        }
        tolerance = {
            "duration_s": 0.02,
            "vertical_impulse_asymmetry_percent": 10.0,
            "step_length_symmetry_index_percent": 8.0,
        }
        checks.update(
            {
                "duration": abs(duration - reference["duration_s"])
                <= tolerance["duration_s"],
                "support_asymmetry_present": impulse_asymmetry >= 15.0,
                "step_asymmetry_present": step_symmetry >= 7.0,
                "support_regression": abs(
                    impulse_asymmetry
                    - reference["vertical_impulse_asymmetry_percent"]
                )
                <= tolerance["vertical_impulse_asymmetry_percent"],
                "step_regression": abs(
                    step_symmetry
                    - reference["step_length_symmetry_index_percent"]
                )
                <= tolerance["step_length_symmetry_index_percent"],
                "completed_20s": bool(validation["success"]),
            }
        )
    else:
        return None
    return {
        "protocol": f"{condition.name}_flexible_zero_20s_regression_v1",
        "reference": reference,
        "current": current,
        "tolerance": tolerance,
        "checks": checks,
        "passed": all(checks.values()),
    }


def run_human_exo(
    *,
    human_condition: str,
    exoskeleton: str,
    assistance: str,
    duration_s: float,
    video: bool,
    evaluation: bool,
    run_name: str | None = None,
    healthy_reference_result: str | Path | None = None,
    controller_mode: str | None = None,
    active_scale: float | None = None,
    adp_scale: float | None = None,
    event_trigger: bool = False,
    event_parameters: dict[str, float] | None = None,
    intention_mode: str = "live_activation",
    output_root: str | Path | None = None,
    enforce_regression: bool = True,
    viewer: bool = False,
    viewer_keep_open: bool = False,
    record_git_metadata: bool = True,
    ppc_rho0: float | None = None,
    ppc_rho_inf: float | None = None,
    ppc_t_rho_s: float | None = None,
    pt_tp_s: float | None = None,
) -> dict[str, Any]:
    """Run the unified Human/Patient and optional frozen Flexible Exo workflow."""
    condition = _condition(human_condition)
    exo_mode = exoskeleton.strip().lower()
    assist_mode = assistance.strip().lower()
    if exo_mode == "none":
        if assist_mode != "zero":
            raise ValueError("Human-only mode requires ASSISTANCE='zero'")
        return run_human_only(
            human_condition=condition.name,
            duration_s=duration_s,
            video=video,
            evaluation=evaluation,
            run_name=run_name,
            output_root=output_root,
            enforce_regression=enforce_regression,
        )
    if exo_mode != "flexible":
        raise ValueError("EXOSKELETON must be 'none' or 'flexible'")

    config = _make_flexible_config(
        condition,
        assistance=assist_mode,
        duration_s=duration_s,
        video=video,
        run_name=run_name,
        healthy_reference_result=healthy_reference_result,
        controller_mode=controller_mode,
        active_scale=active_scale,
        adp_scale=adp_scale,
        event_trigger=event_trigger,
        event_parameters=event_parameters,
        intention_mode=intention_mode,
        output_root=output_root,
        viewer=viewer,
        viewer_keep_open=viewer_keep_open,
        record_git_metadata=record_git_metadata,
        ppc_rho0=ppc_rho0,
        ppc_rho_inf=ppc_rho_inf,
        ppc_t_rho_s=ppc_t_rho_s,
        pt_tp_s=pt_tp_s,
    )
    from simulation import runner

    plan = runner.plan_experiment(config)
    runner.print_experiment_summary(config, plan)
    validation = runner.run_experiment(config, plan)
    output_directory = Path(plan["output_directory"]).resolve()
    pipeline_checks = _verify_flexible_result(
        validation, condition, assist_mode, output_directory
    )
    if not pipeline_checks["passed"]:
        raise RuntimeError(
            f"Flexible pipeline validation failed; inspect {output_directory}"
        )

    healthy_reference: Path | None = None
    summary: dict[str, Any] | None = None
    if evaluation:
        if condition.name != "healthy100":
            healthy_reference = _resolve_healthy_flexible_reference(
                healthy_reference_result
            )
        summary = standard_protocols.evaluate_human_exo(
            output_directory,
            PROJECT_ROOT,
            healthy_reference_candidates=(
                None
                if healthy_reference is None
                else (healthy_reference / "timeseries.csv",)
            ),
        )
        if len(summary["figures"]) != 14 or not all(
            item["generated"] for item in summary["figures"].values()
        ):
            raise RuntimeError(
                f"M6 did not generate all 14 Human-Exo artifacts: {output_directory}"
            )

    m5 = (
        _m5_bounded(validation, summary)
        if summary is not None
        else {
            "passed": bool(validation["finite_state"]),
            "interpretation": "Evaluation disabled; finite-state check only.",
        }
    )
    pipeline_checks["m5_bounded"] = m5
    pipeline_checks["passed"] = bool(
        pipeline_checks["passed"] and m5["passed"]
    )
    regression = (
        None
        if summary is None or assist_mode != "zero"
        else _human_exo_regression(condition, validation, summary, m5)
    )

    if condition.name == "healthy100" and assist_mode == "zero":
        _write_json(
            HEALTHY_FLEXIBLE_POINTER,
            {
                "run_id": plan["run_id"],
                "run_directory": str(output_directory.relative_to(PROJECT_ROOT)),
                "coupled_xml_sha256": FLEXIBLE_XML_SHA256,
                "human_controller_sha256": FLEXIBLE_CONTROLLER_SHA256,
            },
        )
    _attach_postprocessing(
        output_directory,
        evaluation_requested=bool(evaluation),
        evaluation_protocol="human_exoskeleton_v1",
        evaluation_completed=summary is not None,
        regression=regression,
        pipeline_checks=pipeline_checks,
        healthy_reference_result=healthy_reference,
    )
    if (
        enforce_regression
        and regression is not None
        and not regression["passed"]
    ):
        raise RuntimeError(
            f"Flexible regression differs materially; inspect {output_directory}"
        )
    return _read_json(output_directory / "validation.json")
