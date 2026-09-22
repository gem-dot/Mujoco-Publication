#!/usr/bin/env python3
"""One-click formal 3 Human Conditions x 3 Exoskeleton Conditions benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PYTHON = PROJECT_ROOT / ".venv/bin/python"

# ==================================================
# USER SETTINGS
# ==================================================

DURATION = 20.0

VIDEO = True

RUN_EXPERIMENTS = True

RUN_EVALUATION = True

BENCHMARK_YAC_MODE = "full_yac"

HUMAN_CONDITIONS = [
    "healthy100",
    "mild80",
    "moderate60",
]

EXO_CONDITIONS = [
    "no_exo",
    "exo_zero",
    "exo_yac",
]

# ==================================================


# VS Code "Run Python File" always enters the single formal project runtime.
if Path(sys.executable).resolve() != PROJECT_PYTHON.resolve():
    if not PROJECT_PYTHON.is_file():
        raise FileNotFoundError(f"Project Python is missing: {PROJECT_PYTHON}")
    os.execve(
        str(PROJECT_PYTHON),
        [str(PROJECT_PYTHON), str(Path(__file__).resolve())],
        os.environ.copy(),
    )

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mujoco  # noqa: E402

from evaluation import benchmark_comparison, metrics, standard_protocols  # noqa: E402
from orchestration import run_human_exo  # noqa: E402
from orchestration import standard_workflow  # noqa: E402
from scripts import run_YAC as validated_yac_entry  # noqa: E402


BENCHMARK_NAME = "YAC_3x3"
BENCHMARK_OUTPUT_ROOT = PROJECT_ROOT / "results/benchmarks/YAC_3x3"
FULL_YAC_EVIDENCE_RESULT = (
    PROJECT_ROOT
    / "results/algorithms/YAC/20260901_103643__MILD80_FULL_YAC_DEMO"
)
YAC_ACTIVE_SCALE = validated_yac_entry.ACTIVE_SCALE
YAC_ADP_SCALE = validated_yac_entry.ADP_SCALE
YAC_HEALTHY_REFERENCE_RESULT = validated_yac_entry.HEALTHY_REFERENCE_RESULT
YAC_EVENT_TRIGGER = bool(validated_yac_entry.EVENT_TRIGGER)
YAC_EVENT_PARAMETERS = dict(validated_yac_entry.EVENT_PARAMETERS)


def _explicit_yac_evidence(mode: str) -> Path | None:
    if mode != "full_yac":
        return None
    return FULL_YAC_EVIDENCE_RESULT / "validation.json"


def _select_yac_mode(
    user_setting: str,
    evidence_lookup: Any,
) -> tuple[str, Path | None]:
    """Honor only the explicit benchmark setting; never scan for alternatives."""
    return user_setting, evidence_lookup(user_setting)


YAC_MODE, YAC_VALIDATION_EVIDENCE = _select_yac_mode(
    BENCHMARK_YAC_MODE,
    _explicit_yac_evidence,
)
YAC_MODE_SELECTION = "explicit_user_setting"
YAC_MODE_AUTO_SELECTION = False

HUMAN_LABELS = {
    "healthy100": "HEALTHY100",
    "mild80": "MILD80",
    "moderate60": "MODERATE60",
}
EXO_LABELS = {
    "no_exo": "NO_EXO",
    "exo_zero": "EXO_ZERO",
    "exo_yac": "EXO_YAC",
}


@dataclass(frozen=True)
class RunSpec:
    index: int
    human_condition: str
    exo_condition: str
    exoskeleton: str
    assistance: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _relative_or_absolute(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _artifact_path(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _result_directory(validation: dict[str, Any]) -> Path:
    manifest = _artifact_path(validation["artifacts"]["manifest"])
    if manifest is None:
        raise ValueError("Run validation does not contain a manifest artifact")
    return manifest.parent


def _fall_from_reason(reason: str) -> bool:
    normalized = reason.strip().lower()
    return normalized.startswith("pelvis_height_") or normalized.startswith(
        "pelvis_pitch_"
    ) or "fall" in normalized


def _walking_distance(result_directory: Path) -> float | None:
    table = metrics.load_timeseries(result_directory / "timeseries.csv")
    position, _ = metrics.numeric_column(
        table, ["forward_position", "forward_position_m"]
    )
    if position is None:
        return None
    finite = position[np.isfinite(position)]
    if len(finite) < 2:
        return None
    return abs(float(finite[-1] - finite[0]))


# Imported after the project-runtime guard; used only for a recorded distance.
import numpy as np  # noqa: E402


def build_run_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    index = 0
    for human in HUMAN_CONDITIONS:
        for exo in EXO_CONDITIONS:
            index += 1
            specs.append(
                RunSpec(
                    index=index,
                    human_condition=human,
                    exo_condition=exo,
                    exoskeleton="none" if exo == "no_exo" else "flexible",
                    assistance="yac" if exo == "exo_yac" else "zero",
                )
            )
    return specs


def _resolved_yac_configuration(config: Any) -> dict[str, Any]:
    return {
        "mode": getattr(config, "CONTROLLER_MODE", None),
        "active_scale": getattr(config, "YAC_ACTIVE_SCALE", None),
        "adp_scale": getattr(config, "YAC_ADP_SCALE", None),
        "event_trigger": getattr(config, "EVENT_TRIGGER", None),
        "event_parameters": dict(
            getattr(config, "EVENT_TRIGGER_PARAMETERS", {}) or {}
        ),
        "R": getattr(config, "CRITIC_R", None),
        "adp_residual_software_clipping": getattr(
            config, "ADP_RESIDUAL_SOFTWARE_CLIPPING", None
        ),
        "software_total_torque_clipping": getattr(
            config, "SOFTWARE_TOTAL_TORQUE_CLIPPING", None
        ),
    }


def _validate_full_yac_evidence() -> dict[str, Any]:
    if YAC_VALIDATION_EVIDENCE is None:
        raise FileNotFoundError(
            f"No explicit evidence is configured for YAC mode {YAC_MODE!r}"
        )
    result_directory = YAC_VALIDATION_EVIDENCE.parent.resolve()
    manifest_path = result_directory / "manifest.json"
    validation_path = result_directory / "validation.json"
    missing = [
        str(path) for path in (manifest_path, validation_path) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Full YAC evidence is missing: " + ", ".join(missing))
    manifest = _read_json(manifest_path)
    validation = _read_json(validation_path)
    actual_duration = float(validation.get("actual_duration_s", 0.0))
    termination = str(validation.get("termination_reason", "unknown"))
    checks = {
        "manifest_yac_mode": manifest.get("yac_mode") == "full_yac",
        "validation_yac_mode": validation.get("yac_mode") == "full_yac",
        "duration_20s": actual_duration >= 20.0 - 0.02,
        "termination_time_limit": termination == "time_limit",
        "fall_false": not _fall_from_reason(termination),
        "terminated_early_false": validation.get("terminated_early") is False,
        "success": validation.get("success") is True,
        "event_trigger": validation.get("event_trigger") is True,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(
            "Full YAC evidence validation failed: " + ", ".join(failed)
        )
    return {
        "status": "PASS",
        "result_directory": str(result_directory),
        "manifest_path": str(manifest_path),
        "validation_path": str(validation_path),
        "checks": checks,
        "actual_duration_s": actual_duration,
        "termination_reason": termination,
        "manifest": manifest,
    }


def validate_configuration() -> dict[str, Any]:
    """Resolve all nine configurations without constructing or stepping physics."""
    if Path(sys.executable).resolve() != PROJECT_PYTHON.resolve():
        raise RuntimeError(f"Benchmark must use {PROJECT_PYTHON}")
    if DURATION <= 0.0:
        raise ValueError("DURATION must be positive")
    if HUMAN_CONDITIONS != ["healthy100", "mild80", "moderate60"]:
        raise ValueError("Formal benchmark Human matrix was changed")
    if EXO_CONDITIONS != ["no_exo", "exo_zero", "exo_yac"]:
        raise ValueError("Formal benchmark Exoskeleton matrix was changed")
    if YAC_MODE != "full_yac":
        raise ValueError(f"Unsupported explicit BENCHMARK_YAC_MODE: {YAC_MODE!r}")
    if validated_yac_entry.YAC_MODE != YAC_MODE:
        raise RuntimeError(
            "Benchmark and scripts/run_YAC.py mode mismatch: "
            f"{YAC_MODE!r} != {validated_yac_entry.YAC_MODE!r}"
        )
    if not YAC_EVENT_TRIGGER:
        raise ValueError("Validated full_yac benchmark requires Event Trigger")
    if not YAC_HEALTHY_REFERENCE_RESULT.is_dir():
        raise FileNotFoundError(
            f"Validated YAC reference is missing: {YAC_HEALTHY_REFERENCE_RESULT}"
        )
    if not os.access(PROJECT_ROOT / "results", os.W_OK):
        raise PermissionError(f"Results directory is not writable: {PROJECT_ROOT / 'results'}")

    expected_scales = {"healthy100": 1.0, "mild80": 0.8, "moderate60": 0.6}
    resolved: list[dict[str, Any]] = []
    benchmark_yac_configurations: list[dict[str, Any]] = []
    for spec in build_run_specs():
        condition = standard_workflow._condition(spec.human_condition)
        if condition.retained_fmax_fraction != expected_scales[spec.human_condition]:
            raise ValueError(f"Unexpected Fmax scale for {spec.human_condition}")
        if spec.exoskeleton == "flexible":
            config = standard_workflow._make_flexible_config(
                condition,
                assistance=spec.assistance,
                duration_s=DURATION,
                video=VIDEO,
                run_name=f"BENCHMARK_{HUMAN_LABELS[spec.human_condition]}_{EXO_LABELS[spec.exo_condition]}",
                healthy_reference_result=(
                    YAC_HEALTHY_REFERENCE_RESULT
                    if spec.assistance == "yac"
                    else None
                ),
                controller_mode=YAC_MODE if spec.assistance == "yac" else None,
                active_scale=YAC_ACTIVE_SCALE if spec.assistance == "yac" else None,
                adp_scale=YAC_ADP_SCALE if spec.assistance == "yac" else None,
                event_trigger=(
                    YAC_EVENT_TRIGGER if spec.assistance == "yac" else False
                ),
                event_parameters=(
                    YAC_EVENT_PARAMETERS if spec.assistance == "yac" else None
                ),
                output_root=BENCHMARK_OUTPUT_ROOT / "DRY_RUN_NOT_CREATED",
            )
            resolved_yac = (
                _resolved_yac_configuration(config)
                if spec.assistance == "yac"
                else None
            )
            if resolved_yac is not None:
                benchmark_yac_configurations.append(resolved_yac)
            resolved.append(
                {
                    **asdict(spec),
                    "vasti_fmax_scale": condition.retained_fmax_fraction,
                    "model_xml": str(Path(config.MODEL_XML).resolve()),
                    "human_controller": str(
                        Path(config.HUMAN_CONTROLLER_PATH).resolve()
                    ),
                    "controller_mode": getattr(config, "CONTROLLER_MODE", None),
                    "resolved_yac_configuration": resolved_yac,
                }
            )
        else:
            resolved.append(
                {
                    **asdict(spec),
                    "vasti_fmax_scale": condition.retained_fmax_fraction,
                    "model_xml": str(standard_workflow.HEALTHY_MODEL_XML.resolve()),
                    "human_controller": str(
                        standard_workflow.HEALTHY_CONTROLLER.resolve()
                    ),
                    "controller_mode": None,
                }
            )
    source_condition = standard_workflow._condition(validated_yac_entry.HUMAN)
    source_config = standard_workflow._make_flexible_config(
        source_condition,
        assistance="yac",
        duration_s=validated_yac_entry.DURATION,
        video=validated_yac_entry.VIDEO,
        run_name="DRY_RUN_RUN_YAC_SOURCE_CONFIGURATION",
        healthy_reference_result=validated_yac_entry.HEALTHY_REFERENCE_RESULT,
        controller_mode=validated_yac_entry.YAC_MODE,
        active_scale=validated_yac_entry.ACTIVE_SCALE,
        adp_scale=validated_yac_entry.ADP_SCALE,
        event_trigger=validated_yac_entry.EVENT_TRIGGER,
        event_parameters=validated_yac_entry.EVENT_PARAMETERS,
        output_root=BENCHMARK_OUTPUT_ROOT / "DRY_RUN_NOT_CREATED",
    )
    source_yac_configuration = _resolved_yac_configuration(source_config)
    config_match = bool(
        len(benchmark_yac_configurations) == 3
        and all(
            item == source_yac_configuration
            for item in benchmark_yac_configurations
        )
    )
    if not config_match:
        raise RuntimeError(
            "Benchmark Full YAC configuration differs from scripts/run_YAC.py"
        )

    evidence = _validate_full_yac_evidence()
    evidence_manifest = evidence["manifest"]
    evidence_configuration = {
        "mode": evidence_manifest.get("controller_mode"),
        "active_scale": evidence_manifest.get("active_scale"),
        "adp_scale": evidence_manifest.get("adp_scale"),
        "event_trigger": evidence_manifest.get("event_trigger"),
        "event_parameters": dict(
            evidence_manifest.get("event_trigger_parameters", {}) or {}
        ),
        "R": evidence_manifest.get("R"),
        "adp_residual_software_clipping": evidence_manifest.get(
            "adp_residual_software_clipping"
        ),
        "software_total_torque_clipping": evidence_manifest.get(
            "software_total_torque_clipping"
        ),
    }
    evidence_config_match = evidence_configuration == source_yac_configuration
    if not evidence_config_match:
        raise RuntimeError(
            "Current Full YAC source configuration differs from validated evidence"
        )
    return {
        "python_executable": sys.executable,
        "mujoco_version": mujoco.__version__,
        "benchmark_yac_mode": YAC_MODE,
        "yac_mode": YAC_MODE,
        "yac_mode_selection": YAC_MODE_SELECTION,
        "yac_mode_auto_selection": YAC_MODE_AUTO_SELECTION,
        "yac_mode_evidence_path": evidence["validation_path"],
        "yac_20s_validation": evidence["result_directory"],
        "full_yac_evidence": {
            key: value for key, value in evidence.items() if key != "manifest"
        },
        "resolved_full_yac": source_yac_configuration,
        "full_yac_config_match": config_match,
        "full_yac_evidence_config_match": evidence_config_match,
        "run_count": len(resolved),
        "runs": resolved,
        "m6_single_result_entry": str(
            Path(standard_protocols.__file__).resolve()
        ),
        "physics_constructed_or_stepped": False,
    }


def _source_manifest(validation_check: dict[str, Any]) -> dict[str, Any]:
    paths = {
        "bare_human_model": standard_workflow.HEALTHY_MODEL_XML,
        "bare_human_controller": standard_workflow.HEALTHY_CONTROLLER,
        "flexible_model": standard_workflow.FLEXIBLE_XML,
        "flexible_human_controller": standard_workflow.FLEXIBLE_CONTROLLER,
        "healthy100_pathology_config": standard_workflow.HUMAN_CONDITIONS[
            "healthy100"
        ].patient_config,
        "mild80_pathology_config": standard_workflow.HUMAN_CONDITIONS[
            "mild80"
        ].patient_config,
        "moderate60_pathology_config": standard_workflow.HUMAN_CONDITIONS[
            "moderate60"
        ].patient_config,
        "yac_controller": PROJECT_ROOT / "controllers/YAC.py",
        "yac_entry": PROJECT_ROOT / "scripts/run_YAC.py",
        "yac_healthy_reference": YAC_HEALTHY_REFERENCE_RESULT,
    }
    result: dict[str, Any] = {}
    for name, path in paths.items():
        resolved = Path(path).expanduser().resolve()
        result[name] = {
            "path": _relative_or_absolute(resolved),
            "sha256": _sha256(resolved) if resolved.is_file() else None,
        }
    result["yac_validated_20s_result"] = validation_check["yac_20s_validation"]
    return result


def _initial_manifest(
    benchmark_root: Path,
    timestamp: datetime,
    validation_check: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "benchmark_name": BENCHMARK_NAME,
        "timestamp": timestamp.isoformat(timespec="seconds"),
        "benchmark_root": str(benchmark_root.resolve()),
        "planned_duration_s": DURATION,
        "human_conditions": list(HUMAN_CONDITIONS),
        "exo_conditions": list(EXO_CONDITIONS),
        "pathology_semantics": {
            "target": "vasti_l",
            "quantity": "maximum muscle force capability / Fmax",
            "not_activation_scaling": True,
            "scales": {"healthy100": 1.0, "mild80": 0.8, "moderate60": 0.6},
        },
        "benchmark_yac_mode": YAC_MODE,
        "yac_mode": YAC_MODE,
        "yac_mode_selection": YAC_MODE_SELECTION,
        "yac_mode_auto_selection": YAC_MODE_AUTO_SELECTION,
        "yac_mode_evidence_path": validation_check["yac_mode_evidence_path"],
        "yac_active_scale": YAC_ACTIVE_SCALE,
        "yac_adp_scale": YAC_ADP_SCALE,
        "yac_event_trigger": YAC_EVENT_TRIGGER,
        "resolved_full_yac": validation_check["resolved_full_yac"],
        "full_yac_evidence": validation_check["full_yac_evidence"],
        "full_yac_config_match": validation_check["full_yac_config_match"],
        "full_yac_evidence_config_match": validation_check[
            "full_yac_evidence_config_match"
        ],
        "reference_noexo_condition": "HEALTHY100/NO_EXO",
        "reference_exo_condition": "HEALTHY100/EXO_ZERO",
        "python_executable": sys.executable,
        "mujoco_version": mujoco.__version__,
        "model_controller_sources": _source_manifest(validation_check),
        "run_order": [asdict(spec) for spec in build_run_specs()],
        "runs": [],
        "status": "RUNNING",
    }


def _record_completed_run(
    spec: RunSpec,
    validation: dict[str, Any],
) -> dict[str, Any]:
    result_directory = _result_directory(validation)
    run_manifest = _read_json(result_directory / "manifest.json")
    reason = str(validation.get("termination_reason", "unknown"))
    actual_duration = float(validation.get("actual_duration_s", 0.0))
    fall = _fall_from_reason(reason)
    survival = benchmark_comparison.survival_semantics(
        DURATION,
        actual_duration,
        fall,
        "FALL" if fall else "DONE",
    )
    distance = _walking_distance(result_directory)
    video = _artifact_path(validation.get("artifacts", {}).get("video"))
    evaluation_path = result_directory / "evaluation"
    return {
        **asdict(spec),
        "status": "FALL" if fall else "DONE",
        "planned_duration_s": DURATION,
        "actual_duration_s": actual_duration,
        "analysis_duration_s": actual_duration,
        "analysis_window_status": (
            "EARLY_TERMINATED"
            if fall or actual_duration < DURATION - 0.02
            else "FULL_DURATION"
        ),
        "termination_reason": reason,
        **survival,
        "walking_distance_m": distance,
        "distance_before_fall_m": distance if fall else None,
        "human_model": run_manifest.get("human_model"),
        "human_controller_source": run_manifest.get("human_controller"),
        "human_controller_file": run_manifest.get(
            "human_controller_parameter_path"
        ),
        "human_controller_sha256": run_manifest.get(
            "human_controller_parameter_sha256"
        ),
        "patient_config_file": run_manifest.get("patient_config_path"),
        "patient_config_sha256": run_manifest.get("patient_config_sha256"),
        "exo_model": run_manifest.get("exo_model"),
        "coupled_model_file": run_manifest.get("coupled_xml_path"),
        "coupled_model_sha256": run_manifest.get("coupled_xml_sha256"),
        "pathology_type": run_manifest.get("patient_type"),
        "pathology_target": run_manifest.get("pathology_target"),
        "vasti_fmax_scale": run_manifest.get("strength_scale"),
        "healthy_fmax_N": run_manifest.get("healthy_fmax_N"),
        "actual_fmax_N": run_manifest.get("actual_fmax_N"),
        "assist_mode": run_manifest.get("assist_controller"),
        "yac_mode": run_manifest.get("controller_mode"),
        "event_trigger": (
            run_manifest.get("event_trigger")
            if spec.exo_condition == "exo_yac"
            else None
        ),
        "controller_diagnostics_path": (
            validation.get("artifacts", {}).get("controller_diagnostics")
            if spec.exo_condition == "exo_yac"
            else None
        ),
        "result_directory": str(result_directory.resolve()),
        "video_path": None if video is None else str(video),
        "evaluation_path": (
            str(evaluation_path.resolve()) if evaluation_path.is_dir() else None
        ),
    }


def _record_error(
    spec: RunSpec, exc: Exception, benchmark_root: Path
) -> dict[str, Any]:
    condition = standard_workflow.HUMAN_CONDITIONS[spec.human_condition]
    condition_root = (
        benchmark_root
        / HUMAN_LABELS[spec.human_condition]
        / EXO_LABELS[spec.exo_condition]
    )
    recovered_result: Path | None = None
    latest_path = condition_root / "latest_run.json"
    if latest_path.is_file():
        try:
            latest = _read_json(latest_path)
            recovered_result = _artifact_path(latest.get("run_directory"))
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            recovered_result = None
    record = {
        **asdict(spec),
        "status": "RUN_ERROR",
        "planned_duration_s": DURATION,
        "actual_duration_s": None,
        "analysis_duration_s": None,
        "analysis_window_status": "RUN_ERROR",
        "termination_reason": "RUN_ERROR",
        "fall": None,
        "time_to_fall_s": None,
        "survived_to_horizon": None,
        "right_censored": None,
        "walking_distance_m": None,
        "distance_before_fall_m": None,
        "pathology_type": "left_knee_extensor_weakness",
        "pathology_target": "vasti_l",
        "vasti_fmax_scale": condition.retained_fmax_fraction,
        "assist_mode": spec.assistance,
        "yac_mode": YAC_MODE if spec.exo_condition == "exo_yac" else None,
        "event_trigger": (
            YAC_EVENT_TRIGGER if spec.exo_condition == "exo_yac" else None
        ),
        "result_directory": (
            None if recovered_result is None else str(recovered_result)
        ),
        "video_path": None,
        "evaluation_path": (
            str((recovered_result / "evaluation").resolve())
            if recovered_result is not None
            and (recovered_result / "evaluation").is_dir()
            else None
        ),
        "result_recovered_after_error": recovered_result is not None,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }
    if recovered_result is not None:
        validation_path = recovered_result / "validation.json"
        if validation_path.is_file():
            try:
                validation = _read_json(validation_path)
                reason = str(validation.get("termination_reason", "RUN_ERROR"))
                actual = validation.get("actual_duration_s")
                recovered_fall = _fall_from_reason(reason)
                recovered_survival = benchmark_comparison.survival_semantics(
                    DURATION,
                    actual,
                    recovered_fall,
                    "FALL" if recovered_fall else "DONE",
                )
                recovered_video = _artifact_path(
                    validation.get("artifacts", {}).get("video")
                )
                record.update(
                    {
                        "actual_duration_s": actual,
                        "analysis_duration_s": actual,
                        "analysis_window_status": (
                            "EARLY_TERMINATED"
                            if recovered_fall
                            or (
                                actual is not None
                                and float(actual) < DURATION - 0.02
                            )
                            else "FULL_DURATION"
                        ),
                        "termination_reason": reason,
                        **recovered_survival,
                        "video_path": (
                            None
                            if recovered_video is None
                            else str(recovered_video)
                        ),
                    }
                )
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass
    return record


def _run_one(spec: RunSpec, benchmark_root: Path) -> dict[str, Any]:
    condition_root = (
        benchmark_root
        / HUMAN_LABELS[spec.human_condition]
        / EXO_LABELS[spec.exo_condition]
    )
    kwargs: dict[str, Any] = {
        "human_condition": spec.human_condition,
        "exoskeleton": spec.exoskeleton,
        "assistance": spec.assistance,
        "duration_s": DURATION,
        "video": VIDEO,
        "evaluation": RUN_EVALUATION,
        "run_name": (
            f"BENCHMARK_{HUMAN_LABELS[spec.human_condition]}_"
            f"{EXO_LABELS[spec.exo_condition]}"
        ),
        "output_root": condition_root,
        "enforce_regression": False,
    }
    if spec.exo_condition == "exo_yac":
        kwargs.update(
            {
                "healthy_reference_result": YAC_HEALTHY_REFERENCE_RESULT,
                "controller_mode": YAC_MODE,
                "active_scale": YAC_ACTIVE_SCALE,
                "adp_scale": YAC_ADP_SCALE,
                "event_trigger": YAC_EVENT_TRIGGER,
                "event_parameters": YAC_EVENT_PARAMETERS,
            }
        )
    validation = run_human_exo(**kwargs)
    return _record_completed_run(spec, validation)


def _print_startup(validation_check: dict[str, Any]) -> None:
    print(f"Python executable: {sys.executable}", flush=True)
    print("=" * 60)
    print("               FULL YAC 3 × 3 BENCHMARK")
    print("=" * 60)
    print(f"Duration          : {DURATION:.1f} s")
    print(f"Video             : {'ON' if VIDEO else 'OFF'}")
    print(f"Evaluation        : {'ON' if RUN_EVALUATION else 'OFF'}")
    print("")
    print("Benchmark YAC Mode:")
    print(f"  {YAC_MODE}")
    print("")
    print("YAC Mode Source:")
    print("  EXPLICIT USER SETTING")
    print("")
    print("Auto Mode Switch:")
    print("  OFF")
    print("")
    print("Full YAC Evidence:")
    print(f"  {validation_check['full_yac_evidence']['status']}")
    print("")
    print("Evidence Result:")
    print(f"  {validation_check['yac_20s_validation']}")
    print("")
    resolved = validation_check["resolved_full_yac"]
    event = resolved["event_parameters"]
    print("Resolved Full YAC:")
    print("")
    print("  Continuous Critic : ON")
    print(
        "  Event Trigger     : "
        + ("ON" if resolved["event_trigger"] else "OFF")
    )
    print("")
    print(f"  lambda_eta        : {event['lambda_eta']}")
    print(f"  sigma_eta         : {event['sigma_eta']}")
    print(f"  xi                : {event['xi']}")
    print(f"  rho_delta         : {event['rho_delta']}")
    print(f"  epsilon_delta     : {event['epsilon_delta']}")
    print("")
    print(
        "  Residual SW Clip  : "
        + ("ON" if resolved["adp_residual_software_clipping"] else "OFF")
    )
    print(
        "  Total SW Clip     : "
        + ("ON" if resolved["software_total_torque_clipping"] else "OFF")
    )
    print("")
    print("Humans:")
    print("  Healthy100")
    print("  Mild80")
    print("  Moderate60")
    print("")
    print("Conditions:")
    print("  NO_EXO")
    print("  EXO_ZERO")
    print("  EXO_YAC_FULL")
    print("")
    print("Total Runs:")
    print(f"  {validation_check['run_count']}")
    print("")
    print("Pathology: left vasti Fmax scaling (not activation scaling)")
    print("=" * 60, flush=True)


def _print_completion(
    benchmark_root: Path, manifest: dict[str, Any]
) -> None:
    lookup = {
        (run["human_condition"], run["exo_condition"]): run
        for run in manifest["runs"]
    }
    print("=" * 60)
    print("           FULL YAC 3 × 3 BENCHMARK COMPLETE")
    print("=" * 60)
    for human in HUMAN_CONDITIONS:
        print(f"{HUMAN_LABELS[human].title()}:")
        for exo in EXO_CONDITIONS:
            record = lookup.get((human, exo), {})
            status = record.get("status", "MISSING")
            duration = record.get("actual_duration_s")
            detail = status if duration is None else f"{status} @ {duration:.3f} s"
            print(f"  {EXO_LABELS[exo]:9s}: {detail}")
        print("")
    print("-" * 60)
    print("")
    print("Benchmark Root:")
    print(benchmark_root.resolve())
    print("")
    print("Summary CSV:")
    print((benchmark_root / "benchmark_summary.csv").resolve())
    print("")
    print("Summary JSON:")
    print((benchmark_root / "benchmark_summary.json").resolve())
    print("")
    print("FINAL FIGURES:")
    figure_names = (
        "00_benchmark_overview.png",
        "01_survival_and_stability.png",
        "02a_healthy_gait.png",
        "02b_mild_gait.png",
        "02c_moderate_gait.png",
        "03_tracking_and_waveform.png",
        "04_human_effort.png",
        "05_exo_and_hri.png",
    )
    for index, filename in enumerate(figure_names, start=1):
        print("")
        print(f"[{index}]")
        print(filename)
        print((benchmark_root / "evaluation" / filename).resolve())
    print("=" * 60)


def main() -> int:
    validation_check = validate_configuration()
    _print_startup(validation_check)
    if not RUN_EXPERIMENTS:
        print("RUN_EXPERIMENTS=False; configuration is valid and no directory was created.")
        return 0

    timestamp = datetime.now()
    benchmark_root = BENCHMARK_OUTPUT_ROOT / timestamp.strftime("%Y%m%d_%H%M%S")
    benchmark_root.mkdir(parents=True, exist_ok=False)
    manifest_path = benchmark_root / "benchmark_manifest.json"
    manifest = _initial_manifest(benchmark_root, timestamp, validation_check)
    _write_json(manifest_path, manifest)

    for spec in build_run_specs():
        print(
            f"[{spec.index}/9] {HUMAN_LABELS[spec.human_condition]} / "
            f"{EXO_LABELS[spec.exo_condition]}",
            flush=True,
        )
        try:
            record = _run_one(spec, benchmark_root)
        except Exception as exc:
            record = _record_error(spec, exc, benchmark_root)
        manifest["runs"].append(record)
        _write_json(manifest_path, manifest)
        print(f"[{spec.index}/9] {record['status']}", flush=True)

    manifest["status"] = "COMPLETE_WITH_ERRORS" if any(
        run["status"] == "RUN_ERROR" for run in manifest["runs"]
    ) else "COMPLETE"
    manifest["completed_at"] = datetime.now().isoformat(timespec="seconds")
    _write_json(manifest_path, manifest)

    if RUN_EVALUATION:
        benchmark_comparison.generate_benchmark_outputs(benchmark_root)
    _print_completion(benchmark_root, manifest)
    return 1 if manifest["status"] == "COMPLETE_WITH_ERRORS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
