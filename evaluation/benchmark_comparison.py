"""Minimal cross-run aggregation for the formal YAC 3x3 benchmark."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation import metrics


HUMAN_ORDER = ("healthy100", "mild80", "moderate60")
EXO_ORDER = ("no_exo", "exo_zero", "exo_yac")
HUMAN_LABELS = {
    "healthy100": "Healthy100",
    "mild80": "Mild80",
    "moderate60": "Moderate60",
}
EXO_LABELS = {
    "no_exo": "NO_EXO",
    "exo_zero": "EXO_ZERO",
    "exo_yac": "EXO_YAC",
}
COLORS = {
    "no_exo": "#4C78A8",
    "exo_zero": "#72B7B2",
    "exo_yac": "#E45756",
    "reference": "#222222",
}
SUMMARY_COLUMNS = (
    "human_condition",
    "vasti_fmax_scale",
    "exo_condition",
    "yac_mode",
    "event_trigger",
    "event_count",
    "event_fraction",
    "critic_weight_norm_final",
    "critic_weight_norm_peak",
    "bellman_rms",
    "run_status",
    "planned_duration_s",
    "actual_duration_s",
    "analysis_duration_s",
    "analysis_window_status",
    "common_overlap_duration_s",
    "fall",
    "time_to_fall_s",
    "survived_to_horizon",
    "right_censored",
    "termination_reason",
    "walking_distance_m",
    "distance_before_fall_m",
    "mean_speed_mps",
    "pelvis_pitch_rms_deg",
    "pelvis_pitch_peak_deg",
    "pelvis_pitch_before_fall_deg",
    "pelvis_height_min_m",
    "stride_count",
    "completed_strides",
    "gait_waveform_valid",
    "gait_mean_std_valid",
    "low_sample_stride_statistics",
    "waveform_reference_available",
    "waveform_reference_type",
    "waveform_reference_result_dir",
    "stride_period_mean_s",
    "hip_tracking_rmse",
    "knee_tracking_rmse",
    "combined_tracking_rmse",
    "hip_waveform_corr",
    "knee_waveform_corr",
    "human_effort_proxy_Ja",
    "exo_hip_torque_rms",
    "exo_knee_torque_rms",
    "exo_positive_work",
    "exo_negative_work",
    "exo_net_work",
    "waist_force_rms",
    "waist_moment_rms",
    "thigh_force_rms",
    "thigh_moment_rms",
    "shank_force_rms",
    "shank_moment_rms",
    "relative_hip_rms",
    "relative_knee_rms",
    "result_directory",
    "video_path",
    "evaluation_path",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metric(summary: dict[str, Any], category: str, name: str) -> Any:
    category_value = summary.get("metrics", {}).get(category)
    if category_value is None:
        return None
    return metrics.metric_value(category_value, name)


def _metric_numbers(
    summary: dict[str, Any], category: str, names: Iterable[str]
) -> list[float]:
    values = [_finite_number(_metric(summary, category, name)) for name in names]
    return [value for value in values if value is not None]


def _combined_rms(values: Iterable[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return float(np.sqrt(np.mean(np.square(finite))))


def _sum_or_none(values: Iterable[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(finite)) if finite else None


def _max_or_none(values: Iterable[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return max(finite) if finite else None


def _resolve_result(benchmark_root: Path, value: Any) -> Path | None:
    if not value:
        return None
    result = Path(str(value)).expanduser()
    if not result.is_absolute():
        result = benchmark_root / result
    return result.resolve()


def _walking_distance(result_dir: Path) -> float | None:
    table = metrics.load_timeseries(result_dir / "timeseries.csv")
    position, _ = metrics.numeric_column(
        table, ["forward_position", "forward_position_m"]
    )
    if position is None:
        return None
    finite = np.flatnonzero(np.isfinite(position))
    if len(finite) < 2:
        return None
    return abs(float(position[finite[-1]] - position[finite[0]]))


def _true_text(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _yac_diagnostics_metrics(result_dir: Path) -> dict[str, Any]:
    values: dict[str, Any] = {
        "event_count": None,
        "event_fraction": None,
        "critic_weight_norm_final": None,
        "critic_weight_norm_peak": None,
        "bellman_rms": None,
    }
    diagnostics_path = result_dir / "controller_diagnostics.csv"
    if not diagnostics_path.is_file():
        return values
    event_count = 0
    event_valid_count = 0
    critic_norms: list[float] = []
    bellman_errors: list[float] = []
    with diagnostics_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            event_count += int(_true_text(row.get("event_triggered")))
            event_valid_count += int(_true_text(row.get("event_adp_valid")))
            critic_norm = _finite_number(row.get("critic_weight_norm"))
            bellman_error = _finite_number(row.get("bellman_error"))
            if critic_norm is not None:
                critic_norms.append(critic_norm)
            if bellman_error is not None:
                bellman_errors.append(bellman_error)
    values.update(
        {
            "event_count": event_count,
            "event_fraction": (
                event_count / event_valid_count if event_valid_count else None
            ),
            "critic_weight_norm_final": (
                critic_norms[-1] if critic_norms else None
            ),
            "critic_weight_norm_peak": (
                max(critic_norms) if critic_norms else None
            ),
            "bellman_rms": (
                float(np.sqrt(np.mean(np.square(bellman_errors))))
                if bellman_errors
                else None
            ),
        }
    )
    return values


def survival_semantics(
    planned_duration_s: Any,
    actual_duration_s: Any,
    fall: Any,
    run_status: Any,
) -> dict[str, bool | float | None]:
    """Return explicit event/censoring fields without treating errors as falls."""
    planned = _finite_number(planned_duration_s)
    actual = _finite_number(actual_duration_s)
    status = str(run_status or "").upper()
    if status in {"RUN_ERROR", "EVALUATION_MISSING"} or fall is None:
        return {
            "fall": None,
            "time_to_fall_s": None,
            "survived_to_horizon": None,
            "right_censored": None,
        }
    event = bool(fall)
    survived = bool(
        not event
        and actual is not None
        and planned is not None
        and actual >= planned - 0.02
    )
    return {
        "fall": event,
        "time_to_fall_s": actual if event else None,
        "survived_to_horizon": survived,
        "right_censored": not event,
    }


def stride_validity(completed_strides: Any) -> dict[str, bool]:
    """Formal validity rules for complete-cycle waveform statistics."""
    count = _finite_number(completed_strides)
    completed = 0 if count is None else max(0, int(count))
    return {
        "gait_waveform_valid": completed >= 2,
        "gait_mean_std_valid": completed >= 3,
        "low_sample_stride_statistics": completed == 2,
    }


def waveform_reference_spec(
    human_condition: str,
    exo_condition: str,
) -> dict[str, str]:
    """Select the non-exo or donned healthy reference without fallback."""
    if exo_condition == "no_exo":
        return {
            "reference_human_condition": "healthy100",
            "reference_exo_condition": "no_exo",
            "waveform_reference_type": (
                "SOURCE_REF_NOEXO"
                if human_condition == "healthy100"
                else "REF_NOEXO"
            ),
        }
    return {
        "reference_human_condition": "healthy100",
        "reference_exo_condition": "exo_zero",
        "waveform_reference_type": (
            "SOURCE_REF_EXO"
            if human_condition == "healthy100" and exo_condition == "exo_zero"
            else "REF_EXO"
        ),
    }


def _base_row(record: dict[str, Any]) -> dict[str, Any]:
    row = {
        name: None for name in SUMMARY_COLUMNS
    } | {
        "human_condition": str(record.get("human_condition", "")),
        "vasti_fmax_scale": _finite_number(record.get("vasti_fmax_scale")),
        "exo_condition": str(record.get("exo_condition", "")),
        "yac_mode": record.get("yac_mode"),
        "event_trigger": record.get("event_trigger"),
        "run_status": str(record.get("status", "RUN_ERROR")),
        "planned_duration_s": _finite_number(record.get("planned_duration_s")),
        "actual_duration_s": _finite_number(record.get("actual_duration_s")),
        "analysis_duration_s": _finite_number(record.get("actual_duration_s")),
        "analysis_window_status": record.get("analysis_window_status"),
        "fall": record.get("fall"),
        "time_to_fall_s": _finite_number(record.get("time_to_fall_s")),
        "survived_to_horizon": record.get("survived_to_horizon"),
        "right_censored": record.get("right_censored"),
        "termination_reason": record.get("termination_reason"),
        "result_directory": record.get("result_directory"),
        "video_path": record.get("video_path"),
        "evaluation_path": record.get("evaluation_path"),
    }
    for name in (
        "human_model",
        "human_controller_source",
        "human_controller_file",
        "human_controller_sha256",
        "patient_config_file",
        "patient_config_sha256",
        "exo_model",
        "coupled_model_file",
        "coupled_model_sha256",
        "pathology_type",
        "pathology_target",
    ):
        row[name] = record.get(name)
    reference = waveform_reference_spec(
        row["human_condition"], row["exo_condition"]
    )
    row["waveform_reference_type"] = reference["waveform_reference_type"]
    row.update(stride_validity(None))
    row["waveform_reference_available"] = False
    return row


def _run_row(
    benchmark_root: Path, record: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    row = _base_row(record)
    result_dir = _resolve_result(benchmark_root, record.get("result_directory"))
    if result_dir is None:
        return row, None
    if row["exo_condition"] == "exo_yac":
        row.update(_yac_diagnostics_metrics(result_dir))
    summary_path = result_dir / "evaluation/summary.json"
    if not summary_path.is_file():
        row["run_status"] = "EVALUATION_MISSING"
        return row, None
    summary = _load_json(summary_path)
    distance = _walking_distance(result_dir)
    fall = bool(_metric(summary, "stability", "fall_detected"))
    actual = _finite_number(_metric(summary, "stability", "walking_duration_s"))
    planned = _finite_number(record.get("planned_duration_s"))
    survival = survival_semantics(
        planned,
        actual,
        fall,
        row["run_status"],
    )
    row.update(
        {
            "actual_duration_s": actual,
            "analysis_duration_s": actual,
            "analysis_window_status": (
                "FULL_DURATION"
                if actual is not None
                and planned is not None
                and actual >= planned - 0.02
                and not fall
                else "EARLY_TERMINATED"
            ),
            **survival,
            "termination_reason": _metric(
                summary, "stability", "termination_reason"
            ),
            "walking_distance_m": distance,
            "distance_before_fall_m": distance if fall else None,
            "mean_speed_mps": _finite_number(
                _metric(summary, "stability", "mean_forward_velocity_mps")
            ),
            "pelvis_pitch_rms_deg": _finite_number(
                _metric(summary, "stability", "rms_pelvis_pitch_deg")
            ),
            "pelvis_pitch_peak_deg": _finite_number(
                _metric(summary, "stability", "max_abs_pelvis_pitch_deg")
            ),
            "pelvis_pitch_before_fall_deg": (
                _finite_number(
                    _metric(summary, "stability", "max_abs_pelvis_pitch_deg")
                )
                if fall
                else None
            ),
            "pelvis_height_min_m": _finite_number(
                _metric(summary, "stability", "min_pelvis_height_m")
            ),
            "human_effort_proxy_Ja": _finite_number(
                _metric(summary, "human_effort", "activation_squared_mean")
            ),
        }
    )
    cycle_counts: list[float] = []
    cycle_periods: list[float] = []
    for side in ("left", "right"):
        profile_count = (
            summary.get("profiles", {})
            .get("gait", {})
            .get("joints", {})
            .get(side, {})
            .get("hip", {})
            .get("current_cycles")
        )
        count = _finite_number(profile_count)
        period = _finite_number(
            _metric(summary, "gait", f"{side}_mean_cycle_time_s")
        )
        if count is not None:
            cycle_counts.append(count)
        if period is not None:
            cycle_periods.append(period)
    completed_strides = int(min(cycle_counts)) if cycle_counts else None
    row["stride_count"] = completed_strides
    row["completed_strides"] = completed_strides
    row.update(stride_validity(completed_strides))
    row["stride_period_mean_s"] = (
        float(np.mean(cycle_periods)) if cycle_periods else None
    )

    if row["exo_condition"] != "no_exo":
        hip_channels = ("HipExo_R", "HipExo_L")
        knee_channels = ("KneeExo_R", "KneeExo_L")
        hip_rms = _metric_numbers(
            summary,
            "exoskeleton",
            (f"{name}.rms_torque_Nm" for name in hip_channels),
        )
        knee_rms = _metric_numbers(
            summary,
            "exoskeleton",
            (f"{name}.rms_torque_Nm" for name in knee_channels),
        )
        all_channels = (*hip_channels, *knee_channels)
        row["exo_hip_torque_rms"] = _combined_rms(hip_rms)
        row["exo_knee_torque_rms"] = _combined_rms(knee_rms)
        row["exo_positive_work"] = _sum_or_none(
            _metric_numbers(
                summary,
                "exoskeleton",
                (f"{name}.positive_work_J" for name in all_channels),
            )
        )
        row["exo_negative_work"] = _sum_or_none(
            _metric_numbers(
                summary,
                "exoskeleton",
                (f"{name}.negative_work_J" for name in all_channels),
            )
        )
        row["exo_net_work"] = _sum_or_none(
            _metric_numbers(
                summary,
                "exoskeleton",
                (f"{name}.net_work_J" for name in all_channels),
            )
        )
        for site, output_prefix in (
            ("waist", "waist"),
            ("thigh", "thigh"),
            ("shank", "shank"),
        ):
            sites = ("waist",) if site == "waist" else (f"{site}_l", f"{site}_r")
            for signal in ("force", "moment"):
                values = _metric_numbers(
                    summary,
                    "interaction",
                    (f"{name}.{signal}.rms" for name in sites),
                )
                row[f"{output_prefix}_{signal}_rms"] = _max_or_none(values)
        row["relative_hip_rms"] = _max_or_none(
            _metric_numbers(
                summary,
                "interaction",
                (f"thigh_{side}.rotation.rms" for side in ("l", "r")),
            )
        )
        row["relative_knee_rms"] = _max_or_none(
            _metric_numbers(
                summary,
                "interaction",
                (f"shank_{side}.rotation.rms" for side in ("l", "r")),
            )
        )
    return row, summary


def _waveform_metrics(
    summary: dict[str, Any], reference: dict[str, Any], joint: str
) -> tuple[float | None, float | None]:
    errors: list[np.ndarray] = []
    correlations: list[float] = []
    for side in ("left", "right"):
        current = (
            summary.get("profiles", {})
            .get("gait", {})
            .get("joints", {})
            .get(side, {})
            .get(joint, {})
            .get("current_deg")
        )
        healthy = (
            reference.get("profiles", {})
            .get("gait", {})
            .get("joints", {})
            .get(side, {})
            .get(joint, {})
            .get("current_deg")
        )
        if current is None or healthy is None:
            continue
        current_values = np.asarray(current, dtype=float)
        healthy_values = np.asarray(healthy, dtype=float)
        valid = np.isfinite(current_values) & np.isfinite(healthy_values)
        if int(np.count_nonzero(valid)) < 3:
            continue
        current_values = current_values[valid]
        healthy_values = healthy_values[valid]
        errors.append(current_values - healthy_values)
        if np.std(current_values) > 1e-12 and np.std(healthy_values) > 1e-12:
            correlations.append(
                float(np.corrcoef(current_values, healthy_values)[0, 1])
            )
    if not errors:
        return None, None
    concatenated = np.concatenate(errors)
    rmse = float(np.sqrt(np.mean(np.square(concatenated))))
    correlation = float(np.mean(correlations)) if correlations else None
    return rmse, correlation


def _apply_reference_metrics(
    rows: list[dict[str, Any]], summaries: list[dict[str, Any] | None]
) -> dict[str, dict[str, Any] | None]:
    references: dict[str, dict[str, Any] | None] = {
        "noexo": None,
        "exo": None,
    }
    for name, exo_condition in (("noexo", "no_exo"), ("exo", "exo_zero")):
        index = next(
            (
                index
                for index, row in enumerate(rows)
                if row["human_condition"] == "healthy100"
                and row["exo_condition"] == exo_condition
                and summaries[index] is not None
                and bool(row.get("gait_waveform_valid"))
            ),
            None,
        )
        if index is not None:
            references[name] = {
                "summary": summaries[index],
                "result_directory": rows[index].get("result_directory"),
            }

    for row, summary in zip(rows, summaries):
        spec = waveform_reference_spec(
            str(row["human_condition"]), str(row["exo_condition"])
        )
        reference_name = "noexo" if row["exo_condition"] == "no_exo" else "exo"
        reference = references[reference_name]
        row["waveform_reference_type"] = spec["waveform_reference_type"]
        row["waveform_reference_result_dir"] = (
            None if reference is None else reference["result_directory"]
        )
        row["waveform_reference_available"] = reference is not None
        if (
            summary is None
            or reference is None
            or not bool(row.get("gait_waveform_valid"))
        ):
            continue
        reference_summary = reference["summary"]
        assert isinstance(reference_summary, dict)
        hip_rmse, hip_corr = _waveform_metrics(summary, reference_summary, "hip")
        knee_rmse, knee_corr = _waveform_metrics(summary, reference_summary, "knee")
        row["hip_tracking_rmse"] = hip_rmse
        row["knee_tracking_rmse"] = knee_rmse
        row["combined_tracking_rmse"] = _combined_rms(
            value for value in (hip_rmse, knee_rmse) if value is not None
        )
        row["hip_waveform_corr"] = hip_corr
        row["knee_waveform_corr"] = knee_corr
    return references


def _same_nonempty(left: Any, right: Any) -> bool:
    return left not in (None, "") and right not in (None, "") and left == right


def noexo_to_exozero_semantics(
    no_exo: dict[str, Any] | None,
    exo_zero: dict[str, Any] | None,
) -> dict[str, Any]:
    no_exo = no_exo or {}
    exo_zero = exo_zero or {}
    matched = _same_nonempty(
        no_exo.get("human_controller_sha256"),
        exo_zero.get("human_controller_sha256"),
    )
    return {
        "noexo_to_exozero_label": (
            "isolated_mechanical_burden" if matched else "overall_donning_effect"
        ),
        "noexo_to_exozero_controller_matched": matched,
        "noexo_human_controller_sha256": no_exo.get("human_controller_sha256"),
        "exozero_human_controller_sha256": exo_zero.get(
            "human_controller_sha256"
        ),
        "human_controllers": {
            "no_exo": {
                "source": no_exo.get("human_controller_source"),
                "file": no_exo.get("human_controller_file"),
                "sha256": no_exo.get("human_controller_sha256"),
            },
            "exo_zero": {
                "source": exo_zero.get("human_controller_source"),
                "file": exo_zero.get("human_controller_file"),
                "sha256": exo_zero.get("human_controller_sha256"),
            },
        },
        "interpretation": (
            "isolates exoskeleton mechanical burden because the human-controller "
            "parameter SHA is matched"
            if matched
            else "mixes exoskeleton mechanics and human-controller differences; "
            "do not interpret as isolated mechanical burden"
        ),
    }


def controller_effect_pair_semantics(
    exo_zero: dict[str, Any] | None,
    exo_yac: dict[str, Any] | None,
) -> dict[str, Any]:
    exo_zero = exo_zero or {}
    exo_yac = exo_yac or {}
    fields = (
        "human_controller_sha256",
        "human_model",
        "exo_model",
        "pathology_type",
        "pathology_target",
        "vasti_fmax_scale",
        "patient_config_sha256",
        "coupled_model_sha256",
    )
    mismatches = [
        name
        for name in fields
        if not _same_nonempty(exo_zero.get(name), exo_yac.get(name))
    ]
    valid = not mismatches
    return {
        "exozero_to_exoyac_pair_valid": valid,
        "controller_effect_pair_valid": valid,
        "exozero_human_controller_sha256": exo_zero.get(
            "human_controller_sha256"
        ),
        "exoyac_human_controller_sha256": exo_yac.get(
            "human_controller_sha256"
        ),
        "human_controllers": {
            "exo_zero": {
                "source": exo_zero.get("human_controller_source"),
                "file": exo_zero.get("human_controller_file"),
                "sha256": exo_zero.get("human_controller_sha256"),
            },
            "exo_yac": {
                "source": exo_yac.get("human_controller_source"),
                "file": exo_yac.get("human_controller_file"),
                "sha256": exo_yac.get("human_controller_sha256"),
            },
        },
        "matched_fields": list(fields),
        "mismatched_or_missing_fields": mismatches,
        "interpretation": (
            "valid controller-effect pair: human controller, plant, pathology, "
            "and M5 coupled model match"
            if valid
            else "not an isolated controller-effect pair; inspect mismatched or "
            "missing provenance fields"
        ),
    }


def _comparison_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for human in HUMAN_ORDER:
        selected = {
            row["exo_condition"]: row
            for row in rows
            if row["human_condition"] == human
        }
        durations = [
            float(row["actual_duration_s"])
            for row in selected.values()
            if _finite_number(row.get("actual_duration_s")) is not None
        ]
        overlap = min(durations) if len(durations) == 3 else None
        for row in selected.values():
            row["common_overlap_duration_s"] = overlap
        noexo_pair = noexo_to_exozero_semantics(
            selected.get("no_exo"), selected.get("exo_zero")
        )
        controller_pair = controller_effect_pair_semantics(
            selected.get("exo_zero"), selected.get("exo_yac")
        )
        result[human] = {
            "common_overlap_duration_s": overlap,
            "no_padding_or_post_fall_interpolation": True,
            **noexo_pair,
            **controller_pair,
            "pairs": [
                {
                    "comparison": noexo_pair["noexo_to_exozero_label"],
                    "from": "no_exo",
                    "to": "exo_zero",
                    "controller_matched": noexo_pair[
                        "noexo_to_exozero_controller_matched"
                    ],
                    "human_controller_sha256": {
                        "no_exo": noexo_pair["noexo_human_controller_sha256"],
                        "exo_zero": noexo_pair[
                            "exozero_human_controller_sha256"
                        ],
                    },
                    "human_controllers": noexo_pair["human_controllers"],
                    "interpretation": noexo_pair["interpretation"],
                },
                {
                    "comparison": "controller_benefit_primary",
                    "from": "exo_zero",
                    "to": "exo_yac",
                    "controller_effect_pair_valid": controller_pair[
                        "controller_effect_pair_valid"
                    ],
                    "human_controller_sha256": {
                        "exo_zero": controller_pair[
                            "exozero_human_controller_sha256"
                        ],
                        "exo_yac": controller_pair[
                            "exoyac_human_controller_sha256"
                        ],
                    },
                    "human_controllers": controller_pair["human_controllers"],
                    "mismatched_or_missing_fields": controller_pair[
                        "mismatched_or_missing_fields"
                    ],
                    "interpretation": controller_pair["interpretation"],
                },
                {
                    "comparison": "overall_assisted_outcome",
                    "from": "no_exo",
                    "to": "exo_yac",
                },
            ],
        }
    return result


def _csv_value(value: Any) -> Any:
    if value is None:
        return "NaN"
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN"
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_json_ready(child) for child in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in SUMMARY_COLUMNS})


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _row_map(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["human_condition"]), str(row["exo_condition"])): row
        for row in rows
    }


def _plot_overview(rows: list[dict[str, Any]], path: Path) -> None:
    lookup = _row_map(rows)
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    for row_index, human in enumerate(HUMAN_ORDER):
        for column_index, exo in enumerate(EXO_ORDER):
            ax = axes[row_index, column_index]
            ax.set_xticks([])
            ax.set_yticks([])
            item = lookup.get((human, exo))
            if item is None:
                status, duration, speed, face = "MISSING", None, None, "#F3F4F6"
            else:
                duration = _finite_number(item.get("actual_duration_s"))
                speed = _finite_number(item.get("mean_speed_mps"))
                if item.get("run_status") in ("RUN_ERROR", "EVALUATION_MISSING"):
                    status, face = "ERROR", "#FEE2E2"
                elif bool(item.get("fall")):
                    status, face = f"Fall @ {duration:.2f} s", "#FECACA"
                elif bool(item.get("survived_to_horizon")):
                    status, face = "No Fall", "#DCFCE7"
                else:
                    status, face = "EARLY STOP", "#FEF3C7"
            ax.set_facecolor(face)
            ax.text(0.5, 0.66, status, ha="center", va="center", fontsize=15, weight="bold")
            ax.text(0.5, 0.42, "Duration: " + ("NaN" if duration is None else f"{duration:.1f} s"), ha="center")
            ax.text(0.5, 0.28, "Speed: " + ("NaN" if speed is None else f"{speed:.3f} m/s"), ha="center")
            if row_index == 0:
                ax.set_title(EXO_LABELS[exo], fontsize=13, weight="bold")
            if column_index == 0:
                ax.set_ylabel(HUMAN_LABELS[human], fontsize=13, weight="bold")
    fig.suptitle("YAC 3 × 3 Benchmark Overview", fontsize=17, weight="bold")
    _save(fig, path)


def _plot_metric_lines(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    metric_name: str,
    title: str,
    unit: str,
) -> None:
    x = np.arange(len(EXO_ORDER))
    for human in HUMAN_ORDER:
        values = []
        for exo in EXO_ORDER:
            row = next((r for r in rows if r["human_condition"] == human and r["exo_condition"] == exo), None)
            values.append(np.nan if row is None or _finite_number(row.get(metric_name)) is None else float(row[metric_name]))
        linewidth = 2.8 if human == "moderate60" else 1.8
        ax.plot(x, values, marker="o", linewidth=linewidth, label=HUMAN_LABELS[human])
    ax.set_xticks(x, [EXO_LABELS[name] for name in EXO_ORDER])
    ax.set_title(title, loc="left")
    ax.set_ylabel(unit)
    ax.grid(alpha=0.22)


def _plot_survival(rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    _plot_metric_lines(axes[0], rows, "actual_duration_s", "Walking duration", "s")
    axes[0].axhline(20.0, color="#6B7280", linestyle="--", linewidth=1)
    _plot_metric_lines(axes[1], rows, "pelvis_pitch_peak_deg", "Peak |pelvis pitch|", "deg")
    axes[1].axhline(metrics.FALL_PELVIS_PITCH_DEG, color="#D62728", linestyle="--", linewidth=1)
    _plot_metric_lines(axes[2], rows, "pelvis_height_min_m", "Minimum pelvis height", "m")
    axes[2].axhline(metrics.FALL_PELVIS_HEIGHT_M, color="#D62728", linestyle="--", linewidth=1)
    axes[0].legend(frameon=False)
    fig.suptitle("Survival and Stability — Moderate60 emphasized", fontsize=15)
    _save(fig, path)


def _plot_gait_condition(
    human: str,
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any] | None],
    references: dict[str, dict[str, Any] | None],
    path: Path,
) -> None:
    phase = np.linspace(0.0, 100.0, 101)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
    for column, exo in enumerate(EXO_ORDER):
        index = next((i for i, row in enumerate(rows) if row["human_condition"] == human and row["exo_condition"] == exo), None)
        summary = None if index is None else summaries[index]
        row = None if index is None else rows[index]
        reference_name = "noexo" if exo == "no_exo" else "exo"
        reference = references.get(reference_name)
        reference_summary = None if reference is None else reference.get("summary")
        reference_label = (
            "Healthy100 NO_EXO" if exo == "no_exo" else "Healthy100 EXO_ZERO"
        )
        for joint_index, joint in enumerate(("hip", "knee")):
            ax = axes[joint_index, column]
            entry = None if summary is None else summary.get("profiles", {}).get("gait", {}).get("joints", {}).get("left", {}).get(joint)
            reference_entry = None if reference_summary is None else reference_summary.get("profiles", {}).get("gait", {}).get("joints", {}).get("left", {}).get(joint)
            if entry is None or entry.get("current_deg") is None:
                ax.text(0.5, 0.5, "UNAVAILABLE", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            if row is None or not bool(row.get("gait_waveform_valid")):
                count = None if row is None else row.get("completed_strides")
                ax.text(
                    0.5,
                    0.5,
                    f"INSUFFICIENT COMPLETE STRIDES\nn={count if count is not None else 'NaN'}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_axis_off()
                continue
            mean = np.asarray(entry["current_deg"], dtype=float)
            std = entry.get("current_std_deg")
            if reference_entry is not None and reference_entry.get("current_deg") is not None:
                ax.plot(phase, reference_entry["current_deg"], color=COLORS["reference"], linestyle="--", label=reference_label)
            else:
                ax.text(0.98, 0.04, "REFERENCE UNAVAILABLE", ha="right", va="bottom", transform=ax.transAxes, fontsize=8, color="#B91C1C")
            if std is not None and bool(row.get("gait_mean_std_valid")):
                std_values = np.asarray(std, dtype=float)
                ax.fill_between(phase, mean - std_values, mean + std_values, color=COLORS[exo], alpha=0.18, linewidth=0)
            line_label = "Mean ± std" if bool(row.get("gait_mean_std_valid")) else "Mean (low sample)"
            ax.plot(phase, mean, color=COLORS[exo], label=line_label)
            if bool(row.get("low_sample_stride_statistics")):
                ax.text(0.98, 0.96, "LOW SAMPLE: 2 strides", ha="right", va="top", transform=ax.transAxes, fontsize=8, color="#92400E")
            ax.set_title(f"{EXO_LABELS[exo]} — Left {joint.title()}", loc="left")
            ax.set_ylabel("Angle (deg)")
            ax.grid(alpha=0.22)
            if joint_index == 0:
                ax.legend(frameon=False)
    for ax in axes[-1, :]:
        if ax.axison:
            ax.set_xlabel("Gait cycle (%)")
    fig.suptitle(f"{HUMAN_LABELS[human]} Gait Kinematics", fontsize=15)
    _save(fig, path)


def _plot_tracking(rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    specs = (
        ("hip_tracking_rmse", "Hip waveform RMSE", "deg"),
        ("knee_tracking_rmse", "Knee waveform RMSE", "deg"),
        ("hip_waveform_corr", "Hip waveform correlation", "r"),
        ("knee_waveform_corr", "Knee waveform correlation", "r"),
    )
    x = np.arange(len(HUMAN_ORDER))
    offsets = (-0.22, 0.0, 0.22)
    for ax, (metric_name, title, unit) in zip(axes.ravel(), specs):
        for offset, exo in zip(offsets, EXO_ORDER):
            values = []
            for human in HUMAN_ORDER:
                row = next((r for r in rows if r["human_condition"] == human and r["exo_condition"] == exo), None)
                value = None if row is None else _finite_number(row.get(metric_name))
                values.append(np.nan if value is None else value)
            ax.scatter(x + offset, values, s=55, color=COLORS[exo], label=EXO_LABELS[exo])
        ax.set_xticks(x, [HUMAN_LABELS[name] for name in HUMAN_ORDER])
        ax.set_title(title, loc="left")
        ax.set_ylabel(unit)
        ax.grid(axis="y", alpha=0.22)
    axes[0, 0].legend(frameon=False, ncol=3)
    fig.suptitle(
        "Waveform Metrics — NoExo ref: Healthy NO_EXO; Exo ref: Healthy EXO_ZERO",
        fontsize=15,
    )
    _save(fig, path)


def _plot_effort(rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    x = np.arange(len(HUMAN_ORDER))
    offsets = (-0.22, 0.0, 0.22)
    for offset, exo in zip(offsets, EXO_ORDER):
        values = []
        for human in HUMAN_ORDER:
            row = next((r for r in rows if r["human_condition"] == human and r["exo_condition"] == exo), None)
            value = None if row is None else _finite_number(row.get("human_effort_proxy_Ja"))
            values.append(np.nan if value is None else value)
        ax.scatter(x + offset, values, s=80, color=COLORS[exo], label=EXO_LABELS[exo])
    ax.set_xticks(x, [HUMAN_LABELS[name] for name in HUMAN_ORDER])
    ax.set_ylabel(r"$J_a=(1/T)\int\sum_i a_i^2dt$")
    ax.set_title("Human Effort Proxy — not metabolic cost", loc="left")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=3)
    _save(fig, path)


def _plot_exo_hri(rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    exo_rows = [row for row in rows if row["exo_condition"] in ("exo_zero", "exo_yac")]
    x = np.arange(len(HUMAN_ORDER))
    offsets = (-0.12, 0.12)
    for offset, exo in zip(offsets, ("exo_zero", "exo_yac")):
        selected = {row["human_condition"]: row for row in exo_rows if row["exo_condition"] == exo}
        hip = [np.nan if _finite_number(selected.get(h, {}).get("exo_hip_torque_rms")) is None else selected[h]["exo_hip_torque_rms"] for h in HUMAN_ORDER]
        knee = [np.nan if _finite_number(selected.get(h, {}).get("exo_knee_torque_rms")) is None else selected[h]["exo_knee_torque_rms"] for h in HUMAN_ORDER]
        waist_force = [np.nan if _finite_number(selected.get(h, {}).get("waist_force_rms")) is None else selected[h]["waist_force_rms"] for h in HUMAN_ORDER]
        waist_moment = [np.nan if _finite_number(selected.get(h, {}).get("waist_moment_rms")) is None else selected[h]["waist_moment_rms"] for h in HUMAN_ORDER]
        rel_hip = [np.nan if _finite_number(selected.get(h, {}).get("relative_hip_rms")) is None else math.degrees(selected[h]["relative_hip_rms"]) for h in HUMAN_ORDER]
        rel_knee = [np.nan if _finite_number(selected.get(h, {}).get("relative_knee_rms")) is None else math.degrees(selected[h]["relative_knee_rms"]) for h in HUMAN_ORDER]
        axes[0, 0].scatter(x + offset, hip, color=COLORS[exo], marker="o", s=60, label=f"{EXO_LABELS[exo]} hip")
        axes[0, 0].scatter(x + offset, knee, color=COLORS[exo], marker="^", s=60, label=f"{EXO_LABELS[exo]} knee")
        axes[0, 1].scatter(x + offset, waist_force, color=COLORS[exo], marker="o", s=60, label=EXO_LABELS[exo])
        axes[1, 0].scatter(x + offset, waist_moment, color=COLORS[exo], marker="o", s=60, label=EXO_LABELS[exo])
        axes[1, 1].scatter(x + offset, rel_hip, color=COLORS[exo], marker="o", s=60, label=f"{EXO_LABELS[exo]} thigh")
        axes[1, 1].scatter(x + offset, rel_knee, color=COLORS[exo], marker="^", s=60, label=f"{EXO_LABELS[exo]} shank")
    titles = (("Exo torque RMS", "N·m"), ("Waist force RMS", "N"), ("Waist moment RMS", "N·m"), ("Relative rotation RMS", "deg"))
    for ax, (title, unit) in zip(axes.ravel(), titles):
        ax.set_xticks(x, [HUMAN_LABELS[name] for name in HUMAN_ORDER])
        ax.set_title(title, loc="left")
        ax.set_ylabel(unit)
        ax.grid(axis="y", alpha=0.22)
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].legend(frameon=False)
    axes[1, 0].legend(frameon=False)
    axes[1, 1].legend(frameon=False, fontsize=8)
    fig.suptitle("Exoskeleton Output and Human–Exoskeleton Interaction", fontsize=15)
    _save(fig, path)


def _generate_figures(
    evaluation_dir: Path,
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any] | None],
    references: dict[str, dict[str, Any] | None],
) -> dict[str, str]:
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("00_benchmark_overview.png", lambda path: _plot_overview(rows, path)),
        ("01_survival_and_stability.png", lambda path: _plot_survival(rows, path)),
        ("02a_healthy_gait.png", lambda path: _plot_gait_condition("healthy100", rows, summaries, references, path)),
        ("02b_mild_gait.png", lambda path: _plot_gait_condition("mild80", rows, summaries, references, path)),
        ("02c_moderate_gait.png", lambda path: _plot_gait_condition("moderate60", rows, summaries, references, path)),
        ("03_tracking_and_waveform.png", lambda path: _plot_tracking(rows, path)),
        ("04_human_effort.png", lambda path: _plot_effort(rows, path)),
        ("05_exo_and_hri.png", lambda path: _plot_exo_hri(rows, path)),
    ]
    generated: dict[str, str] = {}
    for filename, function in jobs:
        destination = evaluation_dir / filename
        function(destination)
        generated[filename] = str(destination.resolve())
    return generated


def generate_benchmark_outputs(benchmark_root: Path) -> dict[str, Any]:
    """Read completed run directories and write cross-run tables and figures."""
    benchmark_root = benchmark_root.expanduser().resolve()
    manifest_path = benchmark_root / "benchmark_manifest.json"
    manifest = _load_json(manifest_path)
    records = list(manifest.get("runs", []))
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any] | None] = []
    for record in records:
        row, summary = _run_row(benchmark_root, record)
        rows.append(row)
        summaries.append(summary)
    references = _apply_reference_metrics(rows, summaries)
    comparisons = _comparison_metadata(rows)
    evaluation_dir = benchmark_root / "evaluation"
    figures = _generate_figures(evaluation_dir, rows, summaries, references)
    payload = {
        "schema_version": "1.0",
        "benchmark_name": manifest.get("benchmark_name", "YAC_3x3"),
        "benchmark_yac_mode": manifest.get("benchmark_yac_mode"),
        "yac_mode": manifest.get("yac_mode"),
        "yac_mode_selection": manifest.get(
            "yac_mode_selection", "explicit_user_setting"
        ),
        "yac_mode_auto_selection": manifest.get(
            "yac_mode_auto_selection", False
        ),
        "yac_mode_evidence_path": manifest.get("yac_mode_evidence_path"),
        "resolved_full_yac": manifest.get("resolved_full_yac"),
        "full_yac_evidence": manifest.get("full_yac_evidence"),
        "full_yac_config_match": manifest.get("full_yac_config_match"),
        "full_yac_evidence_config_match": manifest.get(
            "full_yac_evidence_config_match"
        ),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_count": len(rows),
        "reference_noexo_condition": "HEALTHY100/NO_EXO",
        "reference_exo_condition": "HEALTHY100/EXO_ZERO",
        "healthy_waveform_references": {
            "ref_noexo": {
                "human_condition": "healthy100",
                "exo_condition": "no_exo",
                "available": references["noexo"] is not None,
                "definition": "bilateral 101-point mean gait waveform from HEALTHY100 NO_EXO",
            },
            "ref_exo": {
                "human_condition": "healthy100",
                "exo_condition": "exo_zero",
                "available": references["exo"] is not None,
                "definition": "bilateral 101-point mean gait waveform from HEALTHY100 EXO_ZERO",
            },
        },
        "missing_value_semantics": {
            "csv": "NaN",
            "json": "null",
            "interpretation": "not applicable or unavailable; never imputed as physical zero",
        },
        "metric_definitions": {
            "vasti_fmax_scale": "maximum left vasti force capability scaling; not activation scaling",
            "analysis_duration_s": "actual recorded duration; no post-fall padding",
            "completed_strides": "minimum valid left/right complete same-side gait-cycle count",
            "tracking_rmse": "RMSE in degrees against condition-appropriate healthy 101-point mean waveform",
            "waveform_corr": "Pearson r against condition-appropriate healthy 101-point mean waveform",
            "right_censored": "no observed fall event before observation ended; a stable horizon completion is censored at 20 s",
            "event_fraction": "event_triggered rows divided by event_adp_valid rows in controller_diagnostics.csv",
            "bellman_rms": "RMS of finite bellman_error samples in controller_diagnostics.csv",
            "human_effort_proxy_Ja": "(1/T) integral sum_i activation_i^2 dt; not metabolic cost",
            "bilateral_exo_torque_rms": "root mean square of available left/right channel RMS values",
            "bilateral_interface_rms": "maximum left/right interface RMS; conservative site summary",
            "relative_hip_rms": "maximum thigh L/R relative-rotation RMS",
            "relative_knee_rms": "maximum shank L/R relative-rotation RMS",
        },
        "comparisons": comparisons,
        "runs": rows,
        "figures": figures,
        "composite_score": None,
    }
    csv_paths = (
        benchmark_root / "benchmark_summary.csv",
        evaluation_dir / "evaluation.csv",
    )
    json_paths = (
        benchmark_root / "benchmark_summary.json",
        evaluation_dir / "summary.json",
    )
    for path in csv_paths:
        _write_csv(path, rows)
    ready = _json_ready(payload)
    for path in json_paths:
        path.write_text(
            json.dumps(ready, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return ready
