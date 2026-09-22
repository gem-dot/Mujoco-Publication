#!/usr/bin/env python3
"""Saved-artifact-only final ICRA summary and compact plotting helper."""

from __future__ import annotations
import argparse, csv, hashlib, json, math, platform, sys
from pathlib import Path
from typing import Any
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from controllers import ICRA as icra  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    def ready(item: Any) -> Any:
        if isinstance(item, dict): return {str(k): ready(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)): return [ready(v) for v in item]
        if isinstance(item, np.generic): return item.item()
        if isinstance(item, float) and not math.isfinite(item): return None
        return item
    path.write_text(json.dumps(ready(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def number(value: Any) -> float:
    if value is True or str(value).lower() == "true": return 1.0
    if value is False or str(value).lower() == "false": return 0.0
    if value in (None, "", "None", "nan"): return math.nan
    return float(value)


def col(rows: list[dict[str, Any]], name: str) -> np.ndarray:
    return np.asarray([number(row.get(name)) for row in rows], dtype=float)


def truth(row: dict[str, Any], name: str) -> bool:
    return str(row.get(name, "")).lower() in ("1", "true")


def phase_metrics(time_s: np.ndarray, phase: np.ndarray, torque: np.ndarray) -> dict[str, float]:
    retained = (time_s >= 2.0) & np.isfinite(phase) & np.isfinite(torque).all(axis=1)
    loading = retained & (phase >= 0.0) & (phase <= 0.20)
    midswing = retained & (phase >= 0.65) & (phase <= 0.85)
    support = np.column_stack((np.maximum(-torque[:, 0], 0.0), np.maximum(torque[:, 1], 0.0)))
    s = float(np.sqrt(np.mean(np.sum(support[loading] ** 2, axis=1))))
    a = float(np.sqrt(np.mean(np.sum(torque[midswing] ** 2, axis=1))))
    return {"S_loading_nm": s, "A_midswing_nm": a, "R_phase": s / (a + 1e-9)}


def merge_rich_timeseries(path: Path, ts: list[dict[str, str]], dg: list[dict[str, str]]) -> None:
    names = list(dg[0]); collisions = set(ts[0]) & set(names)
    if set(names).issubset(ts[0]):
        return
    mapped = {name: f"controller_{name}" if name in collisions else name for name in names}
    rows = []
    for index, source in enumerate(ts):
        row: dict[str, Any] = dict(source)
        diagnostic = dg[index - 1] if 0 < index <= len(dg) else None
        row.update({mapped[name]: "" if diagnostic is None else diagnostic[name] for name in names})
        rows.append(row)
    write_csv(path, rows)


def export_events(result: Path, ts: list[dict[str, str]], dg: list[dict[str, str]]) -> Path:
    path = result / "event_diagnostics.csv"
    keys = [key for key in dg[0] if key.startswith("stage7t_")]
    context = ["phase", "scheduler_alpha", "stage6g_gamma", "stage6g_physical_hip", "stage6g_physical_knee"]
    rows = [{"time": ts[i + 1]["time"], **{key: row.get(key, "") for key in context + keys}}
            for i, row in enumerate(dg)]
    write_csv(path, rows); return path


def plot_basic(result: Path, ts0: list[dict[str, str]], dg0: list[dict[str, str]], controller: str) -> None:
    n = min(len(dg0), len(ts0) - 1); ts, dg = ts0[1:n + 1], dg0[:n]; time_s = col(ts, "time")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for key, label in (("stage5_q_ref_hip", "Hip reference"), ("stage5_q_ref_knee", "Knee reference")):
        axes[0, 0].plot(time_s, col(dg, key), label=label)
    axes[0, 0].plot(time_s, col(ts, "left_hip_angle_rad"), label="Hip actual")
    axes[0, 0].plot(time_s, col(ts, "left_knee_angle_rad"), label="Knee actual")
    axes[0, 0].set(title="A. Left reference and actual", ylabel="rad"); axes[0, 0].legend(ncol=2, fontsize=8)
    axes[0, 1].plot(time_s, col(ts, "exo_actual_torque_HipExo_L_nm"), label="Hip")
    axes[0, 1].plot(time_s, col(ts, "exo_actual_torque_KneeExo_L_nm"), label="Knee")
    axes[0, 1].set(title="B. Physical motor torque", ylabel="Nm"); axes[0, 1].legend()
    axes[1, 0].plot(time_s, col(dg, "scheduler_alpha"), label="alpha")
    axes[1, 0].plot(time_s, col(dg, "stage6g_gamma"), label="gamma")
    axes[1, 0].set(title="C. Intention scheduling", xlabel="time (s)"); axes[1, 0].legend()
    if controller == "final_event":
        axes[1, 1].plot(time_s, col(dg, "stage7t_e_bar_res_norm_sq"), label="residual error")
        axes[1, 1].plot(time_s, col(dg, "stage7t_threshold"), label="threshold")
        events = col(dg, "stage7t_event_accepted") > .5
        axes[1, 1].scatter(time_s[events], col(dg, "stage7t_threshold")[events], s=8, color="red", label="refresh")
        axes[1, 1].set_title("D. Learned-residual Event")
    else:
        axes[1, 1].plot(time_s, np.abs(col(dg, "stage6g_delta_A")), label="|Bellman A|")
        axes[1, 1].plot(time_s, np.abs(col(dg, "stage6g_delta_I")), label="|Bellman I|")
        axes[1, 1].set_title("D. Continuous Critic residual")
    axes[1, 1].set_xlabel("time (s)"); axes[1, 1].legend(fontsize=8)
    fig.suptitle("ICRA Final Benchmark"); fig.savefig(result / "basic_results.png", dpi=180); plt.close(fig)


def extend_manifest(manifest: dict[str, Any], result: Path, condition: str, controller: str,
                    benchmark_sha256: str, hashes: dict[str, str], first: dict[str, str]) -> None:
    manifest.update({
        "final_benchmark_schema_version": "1.0", "public_condition": condition,
        "public_controller": controller,
        "final_controller_mapping": {"continuous": "Stage6H final continuous ICRA",
                                     "final_event": "Stage7T decomposition-consistent residual Event ICRA"},
        "ICRA_py_sha256": hashes["controllers/ICRA.py"],
        "ICRA_benchmark_py_sha256": benchmark_sha256,
        "YAC_frozen_hashes": {k: v for k, v in hashes.items() if "YAC" in k},
        "scientific_parameters_from_controller_diagnostics": {
            "PPC": {k: number(first.get(k)) for k in ("ppc_rho0", "ppc_rho_inf", "ppc_T_rho_s", "ppc_nu")},
            "predefined_time": {k: number(first.get(k)) for k in ("pt_Tp_s", "pt_delta", "pt_p_s_inv")},
            "scheduler": {"mapping": first.get("scheduler_mapping"), "alpha_source": "scheduler_stage6c_alpha_target"},
            "gamma": {"source": "stage6g_gamma", "physical": "continuous"},
            "game": {"objective": first.get("stage6g_objective"), "strategy_order": first.get("game_strategy_order")},
            "critic": {"basis": manifest.get("critic_basis"), "input": manifest.get("critic_input"),
                       "weight_dimension": manifest.get("critic_weight_dimension"), "update_mode": manifest.get("critic_update_mode")},
            "event": None if controller == "continuous" else {k: first.get(k) for k in (
                "stage7t_candidate", "stage7t_theta0", "stage7t_dwell_s", "stage7t_lambda_v_s_inv",
                "stage7t_kappa_e", "stage7t_c_chi_evt", "stage7t_cv")},
            "D_res_nm": None if controller == "continuous" else list(np.asarray(icra.STAGE7T_RESIDUAL_SCALE_NM, dtype=float)),
        },
        "software_runtime": {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__},
        "final_artifacts": {name: str((result / name).relative_to(ROOT)) for name in (
            "manifest.json", "summary.json", "timeseries.csv", "physics_timeseries_1khz.csv",
            "controller_diagnostics.csv", "event_diagnostics.csv", "video.mp4", "basic_results.png") if (result / name).is_file()},
    })
    write_json(result / "manifest.json", manifest)


def finalize_result(result_directory: str | Path, *, condition: str, controller: str,
                    requested_duration_s: float, save_video: bool, save_physics_1khz: bool,
                    frozen_hashes: dict[str, str], benchmark_sha256: str) -> dict[str, Any]:
    result = Path(result_directory).expanduser().resolve()
    validation, manifest = read_json(result / "validation.json"), read_json(result / "manifest.json")
    ts0, dg0 = read_csv(result / "timeseries.csv"), read_csv(result / "controller_diagnostics.csv")
    n = min(len(dg0), len(ts0) - 1); ts, dg = ts0[1:n + 1], dg0[:n]
    time_s, phase = col(ts, "time"), col(dg, "phase")
    torque = np.column_stack((col(ts, "exo_actual_torque_HipExo_L_nm"), col(ts, "exo_actual_torque_KneeExo_L_nm")))
    errors = np.column_stack((col(dg, "stage5_error_hip"), col(dg, "stage5_error_knee")))
    valid = (col(dg, "reference_valid") > .5) & np.isfinite(errors).all(axis=1)
    tracking = float(np.sqrt(np.mean(np.sum(errors[valid] ** 2, axis=1))))
    phases = phase_metrics(time_s, phase, torque)
    emask = time_s >= 2.; activation_names = [k for k in ts[0] if k.startswith("muscle_activation_")]
    activations = np.asarray([[number(row[k]) for k in activation_names] for row in ts]); et = time_s[emask]
    ja = float(np.trapezoid(np.sum(activations[emask] ** 2, axis=1), et) / (et[-1] - et[0]))
    physics = read_csv(result / "physics_timeseries_1khz.csv"); pt = col(physics, "time")
    hri = np.column_stack([col(physics, f"coupling_qfrc_{name}_nm") for name in ("hip_l", "knee_l", "hip_r", "knee_r")])
    hmask = (pt >= 2.) & np.isfinite(hri).all(axis=1); hri_rms = float(np.sqrt(np.mean(np.sum(hri[hmask] ** 2, axis=1))))
    motor = {}
    for name in ("HipExo_L", "KneeExo_L", "HipExo_R", "KneeExo_R"):
        values = col(ts, f"exo_actual_torque_{name}_nm")
        motor[name] = {"RMS_nm": float(np.sqrt(np.mean(values ** 2))), "peak_abs_nm": float(np.max(np.abs(values)))}
    active = col(dg, "stage6g_game_ramp") >= 1.
    zeta = np.column_stack([col(dg, name) for name in ("zeta_track_hip", "zeta_track_knee", "zeta_int_hip", "zeta_int_knee")])
    event_summary = None; event_path = None
    if controller == "final_event":
        eligible, accepted = col(dg, "stage7t_event_logic_active") > .5, col(dg, "stage7t_event_accepted") > .5
        intervals = col(dg, "stage7t_inter_event_interval_s"); intervals = intervals[np.isfinite(intervals)]
        ec, ac = int(np.sum(eligible)), int(np.sum(accepted)); elapsed = max(time_s[eligible][-1] - time_s[eligible][0] + icra.CONTROL_DT_S, icra.CONTROL_DT_S)
        event_summary = {"semantics": "accepted learned Nash-residual refresh reduction only",
            "eligible_residual_policy_ticks": ec, "accepted_residual_refreshes": ac,
            "accepted_refresh_fraction": ac/ec, "learned_residual_refresh_reduction": 1-ac/ec,
            "accepted_event_rate_hz": ac/elapsed,
            "inter_event_interval_s": {"minimum": float(np.min(intervals)), "median": float(np.median(intervals)),
                                       "p95": float(np.quantile(intervals, .95)), "maximum": float(np.max(intervals))}}
        event_path = export_events(result, ts0, dg0)
    right = np.column_stack((col(ts, "exo_actual_torque_HipExo_R_nm"), col(ts, "exo_actual_torque_KneeExo_R_nm")))
    right_zero = bool(np.max(np.abs(right)) == 0.)
    checks = {"full_requested_duration": float(validation["actual_duration_s"])+1e-9 >= requested_duration_s,
        "time_limit": validation["termination_reason"] == "time_limit", "finite": bool(validation["finite_state"]),
        "no_native_saturation": not np.any(col(dg, "native_saturation_command") > .5), "right_motors_exact_zero": right_zero,
        "PPC_valid": not np.any(col(dg, "stage6g_abort") > .5),
        "RN_safe": float(np.nanmin(col(dg, "stage6g_RN_min_eigenvalue")[active])) > 1e-4,
        "weights_bounded": max(float(np.nanmax(col(dg, "stage6g_W_A_norm"))), float(np.nanmax(col(dg, "stage6g_W_I_norm")))) <= icra.STAGE6B_WEIGHT_RADIUS+1e-12,
        "left_torque_nonzero": bool(np.max(np.abs(torque)) > 0.),
        "physics_1khz_exported": (result/"physics_timeseries_1khz.csv").is_file(),
        "video_exported": (not save_video) or (result/"video.mp4").is_file()}
    if controller == "final_event":
        checks.update({"residual_event_active": bool(event_summary and event_summary["accepted_residual_refreshes"] > 0),
            "residual_refresh_reduction_present": bool(event_summary and event_summary["learned_residual_refresh_reduction"] > 0),
            "decomposition_identity": float(np.nanmax(col(dg, "stage7t_decomposition_contract_error_max_abs")[active])) < 1e-12,
            "residual_only_hold": all(truth(row, "stage7t_only_critic_residual_held") for row, yes in zip(dg, active) if yes)})
    summary = {"schema_version": "ICRA-final-1.0", "condition": condition,
        "controller": "Final Residual-Event ICRA" if controller == "final_event" else "Final Continuous ICRA",
        "controller_public_name": controller, "survival_time_s": float(validation["actual_duration_s"]),
        "termination_reason": validation["termination_reason"], "average_forward_speed_mps": float(np.mean(col(ts, "forward_velocity"))),
        "tracking_RMS_rad": tracking, "phase_assistance": phases, "Ja_activation_squared_proxy": ja,
        "Ja_semantics": "human muscle activation-squared effort proxy; not metabolic energy",
        "MyoAssist_activation_semantics": "data.act simulation muscle activation; not measured EMG",
        "M5_physical_HRI_RMS_total_nm": hri_rms, "M5_HRI_semantics": "physical simulation interaction metric; not human comfort",
        "motor_torque": motor, "native_saturation": bool(np.any(col(dg, "native_saturation_command") > .5)),
        "right_motor_zero_check": right_zero, "PPC_max_abs_zeta": float(np.nanmax(np.abs(zeta))),
        "RN_minimum_eigenvalue": float(np.nanmin(col(dg, "stage6g_RN_min_eigenvalue")[active])),
        "Critic_weight_maximum_norm": max(float(np.nanmax(col(dg, "stage6g_W_A_norm"))), float(np.nanmax(col(dg, "stage6g_W_I_norm")))),
        "projection_fraction": {"A": float(col(dg, "stage6g_projection_fraction_A")[-1]), "I": float(col(dg, "stage6g_projection_fraction_I")[-1])},
        "event": event_summary, "safety_checks": checks, "all_safety_checks_passed": all(checks.values()),
        "artifacts": {"manifest": "manifest.json", "timeseries": "timeseries.csv", "physics_1khz": "physics_timeseries_1khz.csv",
            "controller_diagnostics": "controller_diagnostics.csv", "event_diagnostics": event_path.name if event_path else None,
            "video": "video.mp4" if (result/"video.mp4").is_file() else None, "basic_figure": "basic_results.png"}}
    plot_basic(result, ts0, dg0, controller); merge_rich_timeseries(result/"timeseries.csv", ts0, dg0)
    write_json(result/"summary.json", summary); extend_manifest(manifest, result, condition, controller, benchmark_sha256, frozen_hashes, dg0[0])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("result_directory", type=Path); args = parser.parse_args()
    manifest = read_json(args.result_directory/"manifest.json"); mode = str(manifest.get("controller_mode"))
    controller = "final_event" if mode == "residual_event_triggered_intention_target_game_active" else "continuous"
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ("controllers/ICRA.py", "controllers/YAC.py", "controllers/YAC_benchmark.py")}
    summary = finalize_result(args.result_directory, condition=str(manifest.get("human_condition")), controller=controller,
        requested_duration_s=float(manifest["requested_duration_s"]), save_video=bool(manifest.get("save_video")), save_physics_1khz=True,
        frozen_hashes=hashes, benchmark_sha256=hashlib.sha256((ROOT/"controllers/ICRA_benchmark.py").read_bytes()).hexdigest())
    print(json.dumps(summary, indent=2)); return 0 if summary["all_safety_checks_passed"] else 1


if __name__ == "__main__": raise SystemExit(main())
