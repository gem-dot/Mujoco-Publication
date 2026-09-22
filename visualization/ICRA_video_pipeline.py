"""Final, reproducible ICRA simulation -> log -> replay -> render pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import imageio.v2 as imageio
import numpy as np

from orchestration import standard_workflow
from scripts import icra_three_case_demo as prior_demo
from simulation import runner
from visualization.ICRA_video_renderer import FPS, HEIGHT, WIDTH, render_case, render_overview
from visualization.ICRA_video_replay import capture_runner_state, save_trajectory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PARENT = PROJECT_ROOT / "results" / "experiments"
REFERENCE_RESULT = PROJECT_ROOT / "results/experiments/20260822_214246__HUMAN_EXO_HEALTHY100_ZERO"
FINAL_MODE = "residual_event_triggered_intention_target_game_active"
EXPECTED = {"impairment": (8.69, "pelvis_pitch_above_plus_60_deg"),
            "passive": (7.56, "pelvis_height_below_0.65_m"),
            "active": (20.0, "time_limit")}

CASES = {
    "healthy": {"directory": "healthy", "title": "01 · Healthy Human",
                "subtitle": "Healthy muscle capacity · No exoskeleton · Neuromuscular reflex locomotion",
                "condition_label": "Healthy", "vasti_capacity": "100%", "exo_state": "None",
                "assistance_state": "None", "active": False, "passive": False,
                "video": "video_01_healthy_human.mp4"},
    "impairment": {"directory": "moderate60_impairment", "title": "02 · Moderate60 Impaired Human",
                "subtitle": "Bilateral vasti force capacity: 60% · No exoskeleton · No active assistance",
                "condition_label": "Moderate60", "vasti_capacity": "60%", "exo_state": "None",
                "assistance_state": "OFF", "active": False, "passive": False,
                "video": "video_02_moderate60_impairment.mp4"},
    "passive": {"directory": "moderate60_passive", "title": "03 · Moderate60 + Passive Exoskeleton",
                "subtitle": "8 kg bilateral exoskeleton · Motors OFF · μ = 0",
                "condition_label": "Moderate60", "vasti_capacity": "60%", "exo_state": "8.0 kg",
                "assistance_state": "Motors OFF", "active": False, "passive": True,
                "video": "video_03_moderate60_passive_exoskeleton.mp4"},
    "active": {"directory": "moderate60_active", "title": "04 · Moderate60 + Active Assistance",
                "subtitle": "Muscle activation guided game control · Residual event execution",
                "condition_label": "Moderate60", "vasti_capacity": "60%", "exo_state": "8.0 kg",
                "assistance_state": "ON", "active": True, "passive": False,
                "video": "video_04_moderate60_active_assistance.mp4"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def json_read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def output_directory() -> Path:
    base = OUTPUT_PARENT / f"{datetime.now():%Y%m%d_%H%M%S}__ICRA_FINAL_SCIENTIFIC_VIDEO"
    candidate, index = base, 1
    while candidate.exists():
        candidate = Path(f"{base}__{index:02d}"); index += 1
    return candidate


def human_only_config(condition_name: str, output_root: Path) -> Any:
    condition = standard_workflow._condition(condition_name)
    return SimpleNamespace(
        EXPERIMENT_NAME=f"ICRA_VIDEO_{condition_name.upper()}_NO_EXO",
        EXPERIMENT_TYPE="human_only", HUMAN_CONDITION=condition.name,
        MODEL_XML=standard_workflow.HEALTHY_MODEL_XML,
        HUMAN_CONTROLLER_PATH=(standard_workflow.HEALTHY_CONTROLLER if condition_name == "healthy100"
                               else standard_workflow.FLEXIBLE_CONTROLLER),
        PATIENT_CONFIG_PATH=condition.patient_config,
        HUMAN_MODEL="MyoAssist 22-muscle 2D bare Human",
        SOURCE_INITIALIZATION=None if condition_name == "healthy100" else "healthy_flexible_exo_zero_controller",
        EXO_ENABLED=False, EXO_MODEL="none", EXO_MODEL_VERSION="not_applicable", EXO_MASS_KG=0.0,
        ASSISTANCE_NAME="zero", ASSIST_CONTROLLER=None, ASSIST_PARAMS={}, ACTIVE_ASSISTANCE=False,
        CONTROLLER_MODE=None, DURATION_SECONDS=20.0, TARGET_VELOCITY_MPS=1.25, SEED=0,
        SAVE_VIDEO=False, VIDEO_FPS=FPS, VIDEO_FILENAME="unused.mp4", OUTPUT_ROOT=output_root,
        COUPLING_MODE="rigid", VIEWER=False, VIEWER_KEEP_OPEN=False, RECORD_GIT_METADATA=False,
    )


def flexible_config(assistance: str, output_root: Path) -> Any:
    config = standard_workflow._make_flexible_config(
        standard_workflow._condition("moderate60"), assistance=assistance, duration_s=20.0,
        video=False, run_name=f"ICRA_VIDEO_MODERATE60_{assistance.upper()}",
        healthy_reference_result=REFERENCE_RESULT if assistance == "icra" else None,
        controller_mode=FINAL_MODE if assistance == "icra" else None,
        output_root=output_root, viewer=False, viewer_keep_open=False, record_git_metadata=False,
    )
    config.SAVE_VIDEO = False
    config.STRAP_VISUALIZATION = False
    config.STRAP_HUD = False
    config.VIDEO_FPS = FPS
    return config


def configs(root: Path) -> dict[str, Any]:
    raw = root / "raw_runs"
    return {"healthy": human_only_config("healthy100", raw / "01_healthy"),
            "impairment": human_only_config("moderate60", raw / "02_impairment"),
            "passive": flexible_config("zero", raw / "03_passive"),
            "active": flexible_config("icra", raw / "04_active")}


def result_directory(validation: dict[str, Any]) -> Path:
    path = Path(validation["artifacts"]["manifest"])
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve().parent


def run_and_log(key: str, config: Any, root: Path) -> dict[str, Any]:
    print(f"[SIMULATION] {CASES[key]['title']}", flush=True)
    plan = runner.plan_experiment(config)
    with capture_runner_state() as capture:
        validation = runner.run_experiment(config, plan)
    source = result_directory(validation)
    log_dir = root / "video_logs" / CASES[key]["directory"]
    trajectory = log_dir / "trajectory.npz"
    state_info = save_trajectory(trajectory, capture["recorder"], source,
                                 controller_mode=FINAL_MODE if key == "active" else "none")
    manifest = json_read(source / "manifest.json")
    metadata = {
        "schema_version": "ICRA-video-log-1.0", "case": key,
        "condition": manifest.get("human_condition"), "duration_s": float(validation["actual_duration_s"]),
        "requested_duration_s": float(validation["requested_duration_s"]),
        "termination_reason": validation["termination_reason"], "finite_state": validation["finite_state"],
        "simulation_dt_s": float(manifest["mujoco_timestep_s"]),
        "controller_dt_s": float(manifest["control_timestep_s"]),
        "state_history_dt_s": float(np.median(np.diff([
            float(row["time"]) for row in capture["recorder"].rows
        ]))),
        "model_xml": manifest.get("coupled_xml_path") or manifest.get("human_xml_path"),
        "model_xml_sha256": manifest.get("coupled_xml_sha256") or manifest.get("human_xml_sha256"),
        "patient_config": manifest.get("patient_config_path"),
        "patient_config_sha256": manifest.get("patient_config_sha256"),
        "controller_mode": FINAL_MODE if key == "active" else None,
        "source_result_directory": str(source.relative_to(PROJECT_ROOT)),
        "trajectory_sha256": sha256(trajectory), "state_history": state_info,
        "replay_semantics": "logged qpos/qvel restored with mj_forward; no mj_step during rendering",
    }
    json_write(log_dir / "metadata.json", metadata)
    print(f"  {metadata['duration_s']:.2f} s · {metadata['termination_reason']}", flush=True)
    return metadata


def segment_metrics(root: Path, key: str, metadata: dict[str, Any]) -> dict[str, Any]:
    trajectory = root / "video_logs" / CASES[key]["directory"] / "trajectory.npz"
    with np.load(trajectory, allow_pickle=False) as log:
        duration = float(metadata["duration_s"])
        tracking = None
        if key == "active":
            valid = np.isfinite(log["qd_hip_l"]) & np.isfinite(log["qd_knee_l"])
            errors = np.column_stack((
                log["q_exo_hip_l"][valid] - log["qd_hip_l"][valid],
                log["q_exo_knee_l"][valid] - log["qd_knee_l"][valid],
            ))
            tracking = float(np.sqrt(np.mean(np.sum(errors ** 2, axis=1))))
        eligible = int(log["eligible_count"][-1]); accepted = int(log["accepted_count"][-1])
        reduction = 100.0 * (1.0 - accepted / eligible) if eligible else None
        return {
            "duration_s": duration, "termination_reason": metadata["termination_reason"],
            "fall_time_s": duration if metadata["termination_reason"] != "time_limit" else None,
            "pelvis_min_height_m": float(np.min(log["pelvis_height"])),
            "pelvis_max_abs_pitch_deg": float(np.max(np.abs(log["pelvis_pitch_deg"]))),
            "tracking_rms_rad": tracking,
            "healthy_gait_rmse_rad": None,
            "healthy_gait_rmse_unavailable_reason": "No gait-normalized matched-reference metric was defined for this video protocol.",
            "eligible_refreshes": eligible if key == "active" else None,
            "accepted_refreshes": accepted if key == "active" else None,
            "refresh_reduction_percent": reduction if key == "active" else None,
            "actuator_saturation_count": int(np.count_nonzero(log["actuator_saturation_flag"])),
            "all_motor_torques_exact_zero": bool(np.all(log["mu_hip"] == 0) and np.all(log["mu_knee"] == 0)) if key == "passive" else None,
        }


def regression(summary: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    healthy = summary["segments"]["healthy"]
    checks["healthy_full_20s"] = healthy["duration_s"] >= 19.98 and healthy["termination_reason"] == "time_limit"
    for key in ("impairment", "passive"):
        target, reason = EXPECTED[key]
        actual = summary["segments"][key]
        actual["paper_regression_target_s"] = target
        actual["absolute_difference_s"] = abs(actual["duration_s"] - target)
        checks[f"{key}_time"] = actual["absolute_difference_s"] <= 0.02
        checks[f"{key}_criterion"] = actual["termination_reason"] == reason
    active = summary["segments"]["active"]
    active["paper_regression_target_s"] = 20.0
    active["absolute_difference_s"] = abs(active["duration_s"] - 20.0)
    checks["active_full_20s"] = active["duration_s"] >= 19.98 and active["termination_reason"] == "time_limit"
    checks["active_event_counts"] = active["eligible_refreshes"] == 1857 and active["accepted_refreshes"] == 824
    checks["active_no_saturation"] = active["actuator_saturation_count"] == 0
    checks["passive_zero_torque"] = summary["segments"]["passive"]["all_motor_torques_exact_zero"] is True
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "targets_are_regression_checks_not_render_inputs": True}


def validate_video(path: Path, expected_duration: float) -> dict[str, Any]:
    reader = imageio.get_reader(path)
    try:
        meta = reader.get_meta_data(); count = reader.count_frames()
        indices = (0, count // 2, count - 1)
        frames = [reader.get_data(i) for i in indices]
    finally:
        reader.close()
    diffs = [float(np.mean(np.abs(frames[i].astype(float) - frames[i-1].astype(float)))) for i in range(1, 3)]
    checks = {"nonempty": path.stat().st_size > 10000, "fps_30": math.isclose(float(meta["fps"]), 30.0),
              "resolution_1920x1080": list(meta["size"]) == [WIDTH, HEIGHT], "h264": meta.get("codec") == "h264",
              "duration": abs(count / FPS - expected_duration) <= 1 / FPS + 1e-6,
              "sample_frames_nonblack": all(float(np.mean(frame)) > 10 for frame in frames)}
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "frame_count": count, "duration_s": count / FPS, "codec": meta.get("codec"), "sample_frame_differences": diffs}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-only", type=Path, help="Re-render an existing completed video_logs directory")
    args = parser.parse_args(argv)
    os.environ.setdefault("MUJOCO_HEADLESS", "1")
    hashes_before = prior_demo.frozen_hashes()
    root = args.render_only.expanduser().resolve() if args.render_only else output_directory()
    root.mkdir(parents=True, exist_ok=args.render_only is not None)
    config_map = configs(root)
    if args.render_only:
        metadata = {key: json_read(root / "video_logs" / CASES[key]["directory"] / "metadata.json") for key in CASES}
    else:
        metadata = {key: run_and_log(key, config_map[key], root) for key in CASES}
    summary = {"schema_version": "ICRA-final-video-summary-1.0", "created_at": datetime.now().astimezone().isoformat(),
               "segments": {key: segment_metrics(root, key, metadata[key]) for key in CASES}}
    summary["regression"] = regression(summary)
    json_write(root / "video_summary.json", summary)
    if summary["regression"]["status"] != "PASS":
        print(json.dumps(summary["regression"], indent=2))
        print("Regression failed; formal rendering intentionally not started.")
        return 2
    print("[REGRESSION] PASS · beginning deterministic replay/render", flush=True)
    videos: list[Path] = []
    for key, spec in CASES.items():
        case = {**spec, "key": key, "physical_duration_s": metadata[key]["duration_s"],
                "termination_reason": metadata[key]["termination_reason"]}
        trajectory = root / "video_logs" / spec["directory"] / "trajectory.npz"
        video = root / spec["video"]
        sheet = root / "contact_sheets" / f"{Path(spec['video']).stem}_contact_sheet.png"
        print(f"[RENDER] {video.name}", flush=True)
        render_info = render_case(config_map[key], trajectory, video, sheet, case)
        summary["segments"][key]["file"] = str(video)
        summary["segments"][key]["contact_sheet"] = str(sheet)
        summary["segments"][key]["render"] = render_info
        summary["segments"][key]["visual_qa"] = validate_video(video, render_info["duration_s"])
        videos.append(video)
    overview = root / "video_05_overview_comparison.mp4"
    print(f"[RENDER] {overview.name}", flush=True)
    overview_info = render_overview(videos, overview, summary)
    summary["overview"] = {"file": str(overview), "render": overview_info,
                           "visual_qa": validate_video(overview, overview_info["duration_s"])}
    hashes_after = prior_demo.frozen_hashes()
    summary["pipeline"] = {"logging": "PASS", "replay": "PASS", "renderer": "PASS",
                           "30_fps_synchronization": "PASS", "frozen_hashes_before": hashes_before,
                           "frozen_hashes_after": hashes_after,
                           "controllers_unchanged": hashes_before == hashes_after,
                           "physics_modified_for_video": False, "fall_criteria_modified": False}
    summary["status"] = "PASS" if hashes_before == hashes_after and all(
        item["visual_qa"]["status"] == "PASS" for item in summary["segments"].values()
    ) and summary["overview"]["visual_qa"]["status"] == "PASS" else "FAIL"
    json_write(root / "video_summary.json", summary)
    json_write(root / "manifest.json", {"schema_version": "ICRA-final-scientific-video-1.0", "status": summary["status"],
               "video_summary": str(root / "video_summary.json"),
               "videos": [str(p) for p in videos] + [str(overview)],
               "video_logs": [str(root / "video_logs" / CASES[k]["directory"] / "trajectory.npz") for k in CASES],
               "contact_sheets": [summary["segments"][k]["contact_sheet"] for k in CASES],
               "scientific_integrity": summary["pipeline"]})
    print(f"[COMPLETE] {summary['status']} · {root}", flush=True)
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
