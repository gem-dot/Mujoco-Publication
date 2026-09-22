"""Experiment-aware M6 protocols for saved MuJoCo result directories."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from evaluation import metrics, plotting


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXO_JOINTS = ("HipExo_R", "HipExo_L", "KneeExo_R", "KneeExo_L")
SUPPORTED_EXPERIMENT_TYPES = {
    "human_only",
    "exo_only",
    "exo_tracking",
    "human_exo_control",
}
HEALTHY_REFERENCE_CANDIDATES = (
    PROJECT_ROOT / "results/healthy_baseline/healthyhuman/timeseries.csv",
    PROJECT_ROOT / "results/patients/left_knee_weakness/healthy/timeseries.csv",
)


def _category() -> dict[str, Any]:
    return {"available": [], "unavailable": {}, "metrics": {}}


def _put(
    category: dict[str, Any],
    name: str,
    value: Any,
    unit: str,
    definition: str = "",
) -> None:
    category["available"].append(name)
    category["metrics"][name] = {
        "available": True,
        "value": value,
        "unit": unit,
        "definition": definition,
    }


def _unavailable(category: dict[str, Any], name: str, reason: str) -> None:
    category["unavailable"][name] = reason
    category["metrics"][name] = {
        "available": False,
        "value": None,
        "unit": "",
        "reason": reason,
    }


def _has_finite(table: dict[str, np.ndarray], names: list[str]) -> bool:
    values, _ = metrics.numeric_column(table, names)
    return bool(values is not None and len(metrics._finite(values)))


def detect_experiment_type(
    experiment_dir: Path,
    manifest: dict[str, Any],
    table: dict[str, np.ndarray],
) -> tuple[str, str]:
    """Route from explicit metadata first, then from genuinely populated signals."""
    declared = str(manifest.get("experiment_type") or "").strip().lower()
    if declared in SUPPORTED_EXPERIMENT_TYPES:
        return declared, "manifest.experiment_type"
    if "joint_position_HipExo_R_rad" in table and "time_s" in table:
        return "exo_only", "standalone exoskeleton test-bench schema"

    has_human = any(
        name.startswith(("muscle_activation_", "activation_"))
        for name in table
    )
    has_exo = bool(manifest.get("exo_enabled")) or any(
        _has_finite(
            table,
            [
                f"exo_actual_torque_{channel}_nm",
                f"actual_torque_{channel}_Nm",
            ],
        )
        for channel in metrics.EXO_CHANNELS
    )
    if has_human and has_exo:
        return "human_exo_control", "inferred from populated Human and Exo signals"
    if has_human:
        return "human_only", "inferred from populated muscle signals without Exo dynamics"
    if has_exo:
        return "exo_tracking", "inferred from populated Exo signals"
    raise ValueError(
        f"Cannot infer experiment type from manifest or saved signals: {experiment_dir}"
    )


def _is_healthy(manifest: dict[str, Any]) -> bool:
    condition = str(manifest.get("human_condition") or "").strip().lower()
    patient_type = str(manifest.get("patient_type") or "").strip().lower()
    if condition == "healthy" or patient_type in {"healthy", "healthy_control"}:
        return True
    try:
        return float(manifest.get("strength_scale")) >= 1.0
    except (TypeError, ValueError):
        return False


def _evaluate_human_result(
    experiment_dir: Path,
    project_root: Path,
    experiment_type: str,
    routing_reason: str,
    manifest: dict[str, Any],
    healthy_reference_candidates: tuple[Path, ...] | None = None,
) -> dict[str, Any]:
    references = (
        ()
        if _is_healthy(manifest)
        else (
            HEALTHY_REFERENCE_CANDIDATES
            if healthy_reference_candidates is None
            else tuple(Path(path).expanduser().resolve() for path in healthy_reference_candidates)
        )
    )
    summary, table = metrics.evaluate_directory(
        experiment_dir=experiment_dir,
        project_root=project_root,
        healthy_reference_candidates=references,
    )
    summary["protocol"] = (
        "human_only_v1" if experiment_type == "human_only" else "human_exoskeleton_v1"
    )
    summary["experiment"]["experiment_type"] = experiment_type
    summary["experiment"]["routing_reason"] = routing_reason
    summary["experiment"]["manifest_experiment_name"] = manifest.get(
        "experiment_name"
    )

    category_order = ["stability", "gait", "grf", "muscle", "human_effort"]
    if experiment_type != "human_only":
        category_order.append("exoskeleton")
        if "interaction" in summary["metrics"]:
            category_order.append("interaction")
    summary["metrics"] = {
        name: summary["metrics"][name]
        for name in category_order
        if name in summary["metrics"]
    }
    profile_order = ["gait", "grf", "muscle"]
    if experiment_type != "human_only":
        profile_order.append("exoskeleton")
        if "interaction" in summary["profiles"]:
            profile_order.append("interaction")
    summary["profiles"] = {
        name: summary["profiles"][name]
        for name in profile_order
        if name in summary["profiles"]
    }
    if experiment_type == "human_only":
        summary["definitions"].pop("exo_power", None)
        summary["definitions"].pop("interaction", None)
    summary["category_order"] = list(summary["metrics"])

    evaluation_dir = experiment_dir / "evaluation"
    summary["figures"] = plotting.generate_result_figures(
        summary, table, evaluation_dir
    )
    metrics.write_evaluation_csv(evaluation_dir / "evaluation.csv", summary)
    metrics.write_summary_json(evaluation_dir / "summary.json", summary)
    return summary


def _numeric_required(
    table: dict[str, np.ndarray], name: str, experiment_dir: Path
) -> np.ndarray:
    values, _ = metrics.numeric_column(table, [name])
    if values is None or not len(metrics._finite(values)):
        raise ValueError(f"Required Exo-only signal is missing: {name} in {experiment_dir}")
    return values


def _evaluate_exo_only(
    experiment_dir: Path,
    manifest: dict[str, Any],
    routing_reason: str,
) -> dict[str, Any]:
    table = metrics.load_timeseries(experiment_dir / "timeseries.csv")
    time = _numeric_required(table, "time_s", experiment_dir)
    state = _category()
    torque = _category()
    power = _category()
    joint_limits = _category()
    all_columns_finite = True

    for joint in EXO_JOINTS:
        position = _numeric_required(table, f"joint_position_{joint}_rad", experiment_dir)
        velocity = _numeric_required(table, f"joint_velocity_{joint}_rad_s", experiment_dir)
        command = _numeric_required(table, f"command_torque_{joint}_Nm", experiment_dir)
        actual = _numeric_required(table, f"actual_torque_{joint}_Nm", experiment_dir)
        joint_power = _numeric_required(table, f"power_{joint}_W", experiment_dir)
        margin = _numeric_required(table, f"joint_limit_margin_{joint}_rad", experiment_dir)
        all_columns_finite &= bool(
            np.isfinite(
                np.concatenate((position, velocity, command, actual, joint_power, margin))
            ).all()
        )

        _put(state, f"{joint}.min_position_rad", float(np.min(position)), "rad")
        _put(state, f"{joint}.max_position_rad", float(np.max(position)), "rad")
        _put(state, f"{joint}.peak_abs_velocity_rad_s", float(np.max(np.abs(velocity))), "rad/s")
        _put(torque, f"{joint}.peak_abs_command_torque_Nm", float(np.max(np.abs(command))), "N*m")
        _put(torque, f"{joint}.rms_command_torque_Nm", float(np.sqrt(np.mean(np.square(command)))), "N*m")
        _put(torque, f"{joint}.peak_abs_actual_torque_Nm", float(np.max(np.abs(actual))), "N*m")
        _put(torque, f"{joint}.rms_actual_torque_Nm", float(np.sqrt(np.mean(np.square(actual)))), "N*m")
        error = actual - command
        _put(torque, f"{joint}.command_actual_rmse_Nm", float(np.sqrt(np.mean(np.square(error)))), "N*m")
        _put(torque, f"{joint}.command_actual_mae_Nm", float(np.mean(np.abs(error))), "N*m")
        _put(torque, f"{joint}.max_abs_command_actual_error_Nm", float(np.max(np.abs(error))), "N*m")

        positive = np.maximum(joint_power, 0.0)
        negative = np.minimum(joint_power, 0.0)
        _put(power, f"{joint}.peak_positive_power_W", float(np.max(positive)), "W")
        _put(power, f"{joint}.peak_negative_power_W", float(np.min(negative)), "W")
        _put(power, f"{joint}.rms_power_W", float(np.sqrt(np.mean(np.square(joint_power)))), "W")
        _put(power, f"{joint}.positive_work_J", float(np.trapezoid(positive, time)), "J")
        _put(power, f"{joint}.negative_work_J", float(np.trapezoid(negative, time)), "J", "signed negative work")
        _put(power, f"{joint}.absolute_work_J", float(np.trapezoid(np.abs(joint_power), time)), "J")
        _put(power, f"{joint}.net_work_J", float(np.trapezoid(joint_power, time)), "J")

        _put(joint_limits, f"{joint}.minimum_limit_margin_rad", float(np.min(margin)), "rad")
        _put(joint_limits, f"{joint}.limit_violation_samples", int(np.count_nonzero(margin < -1e-8)), "count")

    _put(state, "duration_s", float(time[-1] - time[0]), "s", "last recorded time minus first recorded time")
    finite_signal, _ = metrics.bool_column(table, ["finite_state"])
    finite_state = bool(all_columns_finite and (finite_signal is None or np.all(finite_signal)))
    _put(state, "finite_state", finite_state, "bool")
    _put(
        joint_limits,
        "all_joint_limits_respected",
        all(
            int(joint_limits["metrics"][f"{joint}.limit_violation_samples"]["value"]) == 0
            for joint in EXO_JOINTS
        ),
        "bool",
    )

    summary: dict[str, Any] = {
        "schema_version": metrics.SCHEMA_VERSION,
        "protocol": "exo_only_test_bench_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": {
            "directory": str(experiment_dir.resolve()),
            "timeseries_csv": str((experiment_dir / "timeseries.csv").resolve()),
            "manifest_json": str((experiment_dir / "manifest.json").resolve()),
            "row_count": int(time.size),
            "experiment_type": "exo_only",
            "routing_reason": routing_reason,
            "manifest_experiment_name": manifest.get("experiment_name"),
            "test_mode": manifest.get("test_mode"),
        },
        "definitions": {
            "power": "actual actuator torque * joint angular velocity",
            "joint_limit_margin": "minimum distance to the joint lower/upper limit",
            "fixture": "pelvis/root body fixed; four hip/knee hinges remain free",
        },
        "category_order": ["state", "joint_limits", "torque", "power"],
        "metrics": {
            "state": state,
            "joint_limits": joint_limits,
            "torque": torque,
            "power": power,
        },
        "profiles": {},
        "unsupported": {
            "human_metrics": "standalone Exo-only test bench contains no Human",
            "interaction": "standalone Exo-only test bench contains no Human-exoskeleton interface",
        },
        "figures": {},
    }
    evaluation_dir = experiment_dir / "evaluation"
    summary["figures"] = plotting.generate_result_figures(summary, table, evaluation_dir)
    metrics.write_evaluation_csv(evaluation_dir / "evaluation.csv", summary)
    metrics.write_summary_json(evaluation_dir / "summary.json", summary)
    return summary


def evaluate_result(
    experiment_dir: Path,
    project_root: Path = PROJECT_ROOT,
    *,
    healthy_reference_candidates: tuple[Path, ...] | None = None,
) -> dict[str, Any]:
    """Evaluate one saved result without rerunning or importing simulation physics."""
    experiment_dir = experiment_dir.expanduser().resolve()
    timeseries_path = experiment_dir / "timeseries.csv"
    if not timeseries_path.exists():
        raise FileNotFoundError(f"Expected timeseries.csv in result directory: {experiment_dir}")
    table = metrics.load_timeseries(timeseries_path)
    manifest = metrics.load_json_if_present(experiment_dir / "manifest.json")
    experiment_type, routing_reason = detect_experiment_type(
        experiment_dir, manifest, table
    )
    if experiment_type == "exo_only":
        return _evaluate_exo_only(experiment_dir, manifest, routing_reason)
    return _evaluate_human_result(
        experiment_dir,
        project_root.resolve(),
        experiment_type,
        routing_reason,
        manifest,
        healthy_reference_candidates,
    )


def evaluate_human_only(experiment_dir: Path, project_root: Path) -> dict[str, Any]:
    """Compatibility entry used by the formal L1 workflow."""
    summary = evaluate_result(experiment_dir, project_root)
    if summary["experiment"]["experiment_type"] != "human_only":
        raise ValueError("Human-only protocol received a non-Human-only result")
    return summary


def evaluate_exo_only(experiment_dir: Path) -> dict[str, Any]:
    """Compatibility entry used by the formal L2 workflow."""
    summary = evaluate_result(experiment_dir, PROJECT_ROOT)
    if summary["experiment"]["experiment_type"] != "exo_only":
        raise ValueError("Exo-only protocol received a non-Exo-only result")
    return summary


def evaluate_human_exo(
    experiment_dir: Path,
    project_root: Path = PROJECT_ROOT,
    *,
    healthy_reference_candidates: tuple[Path, ...] | None = None,
) -> dict[str, Any]:
    """Explicit entry for existing flexible/rigid Human–Exo results."""
    summary = evaluate_result(
        experiment_dir,
        project_root,
        healthy_reference_candidates=healthy_reference_candidates,
    )
    if summary["experiment"]["experiment_type"] != "human_exo_control":
        raise ValueError("Human–Exo protocol received a result without Human and Exo signals")
    return summary
