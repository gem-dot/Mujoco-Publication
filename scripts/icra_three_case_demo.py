"""Orchestration and video presentation layer for ``run_ICRA_demo.py``."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from orchestration import standard_workflow
from scripts.analyze_ICRA_final import finalize_result
from simulation import runner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "results/experiments"
REFERENCE_RESULT = (
    PROJECT_ROOT / "results/experiments/20260822_214246__HUMAN_EXO_HEALTHY100_ZERO"
)
PATIENT_ONLY_PROTOCOL_EVIDENCE = (
    PROJECT_ROOT
    / "results/runs/YAC_benchmark/20260903_150544__YAC_FINAL_BENCHMARK"
    / "01_MODERATE60_NO_EXO/20260903_150544__NO_EXO"
)
FINAL_MODE = "residual_event_triggered_intention_target_game_active"
VIDEO_WIDTH = 960
VIDEO_HEIGHT = 544
EXO_CHANNELS = ("HipExo_L", "KneeExo_L", "HipExo_R", "KneeExo_R")
EXPECTED_FROZEN_HASHES = {
    "controllers/ICRA.py": "971e34d1ad8f6af6355f2ec4ac1e16600355f8f56dbab2582b92fb4a9ddb89ef",
    "controllers/ICRA_benchmark.py": "50bcd462539bc2f77c5677df99a483b69dbe5c9b86c7b198bd9089e9f2587b49",
    "controllers/YAC.py": "92a822b86bae56810b40489a3d10b61f7329b3c8dc977cddd753144452254bd2",
    "controllers/YAC_benchmark.py": "ef3987ac50d637350eb28519dcd17c0bda11757eb3fa9177a878380dde769604",
}

CASE_SPECS = (
    {
        "key": "patient_only",
        "number": 1,
        "name": "Patient Only",
        "title": "Moderate60 - Patient Only",
        "subtitle": "No exoskeleton",
        "filename": "01_patient_no_exoskeleton.mp4",
        "result_filename": "01_patient_no_exoskeleton.json",
    },
    {
        "key": "passive_exoskeleton",
        "number": 2,
        "name": "Passive Exoskeleton",
        "title": "Moderate60 - Passive Exoskeleton",
        "subtitle": "Zero motor assistance",
        "filename": "02_patient_exoskeleton_zero_assist.mp4",
        "result_filename": "02_patient_exoskeleton_zero_assist.json",
    },
    {
        "key": "final_icra",
        "number": 3,
        "name": "Final ICRA",
        "title": "Moderate60 - Final ICRA",
        "subtitle": "Residual-Event Dynamic Fusion",
        "filename": "03_patient_exoskeleton_ICRA.mp4",
        "result_filename": "03_patient_exoskeleton_ICRA.json",
    },
)


@dataclass(frozen=True)
class DemoSettings:
    reuse_existing_results: bool = False
    duration_s: float = 20.0
    video_fps: int = 25
    show_viewer: bool = False
    auto_open_result: bool = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_hashes() -> dict[str, str]:
    current = {name: sha256(PROJECT_ROOT / name) for name in EXPECTED_FROZEN_HASHES}
    if current != EXPECTED_FROZEN_HASHES:
        raise RuntimeError(f"Frozen-controller hash mismatch: {current}")
    return current


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def result_directory(validation: dict[str, Any]) -> Path:
    path = Path(str(validation["artifacts"]["manifest"]))
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve().parent


def unique_demo_directory() -> Path:
    stem = f"{datetime.now():%Y%m%d_%H%M%S}__ICRA_THREE_CASE_DEMO"
    candidate = EXPERIMENT_ROOT / stem
    suffix = 1
    while candidate.exists():
        candidate = EXPERIMENT_ROOT / f"{stem}__{suffix:02d}"
        suffix += 1
    return candidate


def make_patient_only_config(output_root: Path, settings: DemoSettings) -> Any:
    """Validated bare model with the same reflex parameters as exo cases.

    This exact protocol was previously exercised by the formal bilateral
    Moderate60 benchmark. Only the exoskeleton model is absent.
    """
    condition = standard_workflow._condition("moderate60")
    return SimpleNamespace(
        EXPERIMENT_NAME="ICRA_DEMO_PATIENT_ONLY",
        EXPERIMENT_TYPE="human_only",
        HUMAN_CONDITION=condition.name,
        MODEL_XML=standard_workflow.HEALTHY_MODEL_XML,
        HUMAN_CONTROLLER_PATH=standard_workflow.FLEXIBLE_CONTROLLER,
        PATIENT_CONFIG_PATH=condition.patient_config,
        HUMAN_MODEL="MyoAssist 22-muscle 2D bare Human",
        SOURCE_INITIALIZATION="healthy_flexible_exo_zero_controller",
        EXO_ENABLED=False,
        EXO_MODEL="none",
        EXO_MODEL_VERSION="not_applicable",
        EXO_MASS_KG=0.0,
        ASSISTANCE_NAME="zero",
        ASSIST_CONTROLLER=None,
        ASSIST_PARAMS={},
        ACTIVE_ASSISTANCE=False,
        CONTROLLER_MODE=None,
        DURATION_SECONDS=float(settings.duration_s),
        TARGET_VELOCITY_MPS=1.25,
        SEED=0,
        SAVE_VIDEO=True,
        VIDEO_FPS=int(settings.video_fps),
        VIDEO_FILENAME="video.mp4",
        OUTPUT_ROOT=output_root,
        COUPLING_MODE="rigid",
        VIEWER=bool(settings.show_viewer),
        VIEWER_KEEP_OPEN=False,
        RECORD_GIT_METADATA=False,
    )


def validate_patient_only_protocol_evidence() -> None:
    manifest = read_json(PATIENT_ONLY_PROTOCOL_EVIDENCE / "manifest.json")
    checks = {
        "human_only": manifest.get("experiment_type") == "human_only",
        "moderate60": manifest.get("human_condition") == "moderate60",
        "bilateral_vasti": manifest.get("affected_muscles") == ["vasti_l", "vasti_r"],
        "strength_60_percent": manifest.get("strength_scale") == 0.6,
        "matched_reflex_parameters": (
            manifest.get("human_controller_parameter_sha256")
            == standard_workflow.FLEXIBLE_CONTROLLER_SHA256
        ),
        "video_exists": (PATIENT_ONLY_PROTOCOL_EVIDENCE / "video.mp4").is_file(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Historical patient-only protocol evidence failed: {checks}")


def make_flexible_config(
    output_root: Path, settings: DemoSettings, *, assistance: str
) -> Any:
    condition = standard_workflow._condition("moderate60")
    config = standard_workflow._make_flexible_config(
        condition,
        assistance=assistance,
        duration_s=settings.duration_s,
        video=True,
        run_name=(
            "ICRA_DEMO_PASSIVE_EXOSKELETON"
            if assistance == "zero"
            else "ICRA_DEMO_FINAL_RESIDUAL_EVENT"
        ),
        healthy_reference_result=REFERENCE_RESULT if assistance == "icra" else None,
        controller_mode=FINAL_MODE if assistance == "icra" else None,
        output_root=output_root,
        viewer=settings.show_viewer,
        viewer_keep_open=False,
        record_git_metadata=False,
    )
    # Presentation-only cleanup. Disabling strap arrows/HUD keeps all three raw
    # renders on the same camera without changing any simulated force or state.
    config.STRAP_VISUALIZATION = False
    config.STRAP_HUD = False
    config.VIDEO_FPS = int(settings.video_fps)
    config.VIDEO_FILENAME = "video.mp4"
    return config


def run_config(config: Any, *, assistance: str | None = None) -> dict[str, Any]:
    plan = runner.plan_experiment(config)
    with open(os.devnull, "w", encoding="utf-8") as quiet, contextlib.redirect_stdout(quiet):
        validation = runner.run_experiment(config, plan)
    output = result_directory(validation)
    condition = standard_workflow._condition("moderate60")
    if assistance is None:
        damage = standard_workflow._verify_damage(validation, condition)
        shared_controller = (
            validation.get("human_controller_parameter_sha256")
            == standard_workflow.FLEXIBLE_CONTROLLER_SHA256
        )
        pipeline = {
            "patient_damage": damage,
            "shared_human_controller_with_exo_cases": shared_controller,
            "passed": bool(damage["passed"] and shared_controller),
        }
        standard_workflow._attach_postprocessing(
            output,
            evaluation_requested=False,
            evaluation_protocol="icra_demo_human_only_v1",
            evaluation_completed=False,
            pipeline_checks=pipeline,
        )
    else:
        pipeline = standard_workflow._verify_flexible_result(
            validation, condition, assistance, output
        )
        pipeline["clean_matched_video_render"] = {
            "strap_force_arrows": False,
            "strap_hud": False,
            "physics_affected": False,
        }
        pipeline["m5_bounded"] = {
            "passed": bool(validation["finite_state"]),
            "interpretation": "Evaluation disabled; finite-state check only.",
        }
        pipeline["passed"] = bool(pipeline["passed"] and validation["finite_state"])
        if not pipeline["passed"]:
            raise RuntimeError(f"Flexible pipeline validation failed: {output}")
        standard_workflow._attach_postprocessing(
            output,
            evaluation_requested=False,
            evaluation_protocol="icra_demo_human_exoskeleton_v1",
            evaluation_completed=False,
            pipeline_checks=pipeline,
        )
    return read_json(output / "validation.json")


def run_fresh_cases(
    raw_root: Path, settings: DemoSettings, hashes: dict[str, str]
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    sources: dict[str, Path] = {}
    final_summaries: dict[str, dict[str, Any]] = {}
    configs = (
        (CASE_SPECS[0], make_patient_only_config(raw_root / "01_patient_only", settings), None),
        (CASE_SPECS[1], make_flexible_config(raw_root / "02_passive_exoskeleton", settings, assistance="zero"), "zero"),
        (CASE_SPECS[2], make_flexible_config(raw_root / "03_final_icra", settings, assistance="icra"), "icra"),
    )
    for spec, config, assistance in configs:
        print(f"[{spec['number']}/3] {spec['title']}", flush=True)
        print("Running fresh MuJoCo physics and raw video...", flush=True)
        validation = run_config(config, assistance=assistance)
        source = result_directory(validation)
        sources[str(spec["key"])] = source
        if spec["key"] == "final_icra":
            summary = finalize_result(
                source,
                condition="moderate60",
                controller="final_event",
                requested_duration_s=settings.duration_s,
                save_video=True,
                save_physics_1khz=True,
                frozen_hashes=hashes,
                benchmark_sha256=hashes["controllers/ICRA_benchmark.py"],
            )
            final_summaries[str(spec["key"])] = summary
        refreshed = read_json(source / "validation.json")
        print(
            f"Physical survival: {float(refreshed['actual_duration_s']):.2f} s "
            f"({refreshed['termination_reason']})",
            flush=True,
        )
    return sources, final_summaries


def latest_reusable_sources() -> dict[str, Path]:
    for manifest_path in sorted(
        EXPERIMENT_ROOT.glob("*__ICRA_THREE_CASE_DEMO*/manifest.json"), reverse=True
    ):
        manifest = read_json(manifest_path)
        if manifest.get("status") != "PASS":
            continue
        source_map = manifest.get("raw_result_directories", {})
        if set(source_map) != {str(spec["key"]) for spec in CASE_SPECS}:
            continue
        sources = {
            key: (PROJECT_ROOT / value).resolve()
            for key, value in source_map.items()
        }
        if all((path / "validation.json").is_file() and (path / "video.mp4").is_file() for path in sources.values()):
            return sources
    raise FileNotFoundError("No prior validated ICRA three-case demo is available for reuse")


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    path = Path("/System/Library/Fonts/Supplemental") / name
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def overlay_frame(
    frame: np.ndarray,
    title: str,
    subtitle: str,
    termination_text: str | None = None,
) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    title_font = load_font(24, bold=True)
    subtitle_font = load_font(16)
    termination_font = load_font(17, bold=True)
    lines = [(title, title_font), (subtitle, subtitle_font)]
    if termination_text:
        lines.append((termination_text, termination_font))
    boxes = [draw.textbbox((0, 0), text, font=font) for text, font in lines]
    width = max(box[2] - box[0] for box in boxes) + 28
    heights = [box[3] - box[1] for box in boxes]
    height = sum(heights) + 18 + 8 * (len(lines) - 1)
    left, top = 18, 16
    draw.rounded_rectangle(
        (left, top, left + width, top + height),
        radius=7,
        fill=(15, 22, 28, 172),
    )
    y = top + 9
    for index, ((text, font), line_height) in enumerate(zip(lines, heights)):
        color = (255, 210, 190, 255) if termination_text and index == 2 else (255, 255, 255, 255)
        draw.text((left + 14, y), text, font=font, fill=color)
        y += line_height + 8
    return np.asarray(Image.alpha_composite(image, layer).convert("RGB"))


def video_metadata(path: Path) -> dict[str, Any]:
    reader = imageio.get_reader(path)
    try:
        metadata = reader.get_meta_data()
        count = int(reader.count_frames())
        samples = [reader.get_data(index) for index in (0, count // 2, count - 1)]
    finally:
        reader.close()
    return {
        "frame_count": count,
        "fps": float(metadata["fps"]),
        "duration_s": count / float(metadata["fps"]),
        "reported_container_duration_s": float(metadata.get("duration", count / float(metadata["fps"]))),
        "width": int(metadata["size"][0]),
        "height": int(metadata["size"][1]),
        "codec": str(metadata.get("codec")),
        "pix_fmt": str(metadata.get("pix_fmt")),
        "decodable_samples": len(samples),
    }


def build_presentation_video(
    raw_video: Path,
    output: Path,
    spec: dict[str, Any],
    physical_survival_s: float,
    termination_reason: str,
    settings: DemoSettings,
) -> dict[str, Any]:
    reader = imageio.get_reader(raw_video)
    raw_meta = reader.get_meta_data()
    raw_count = int(reader.count_frames())
    raw_fps = float(raw_meta["fps"])
    raw_size = tuple(int(value) for value in raw_meta["size"])
    if not math.isclose(raw_fps, settings.video_fps, abs_tol=1e-9):
        reader.close()
        raise ValueError(f"Raw FPS mismatch: {raw_video}: {raw_fps}")
    if raw_size != (VIDEO_WIDTH, VIDEO_HEIGHT):
        reader.close()
        raise ValueError(f"Raw resolution mismatch: {raw_video}: {raw_size}")
    target_count = int(round(settings.duration_s * settings.video_fps))
    early = physical_survival_s < settings.duration_s - 1.0 / settings.video_fps
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        output,
        fps=settings.video_fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        ffmpeg_log_level="error",
    )
    genuine_count = 0
    padded_count = 0
    last_base: np.ndarray | None = None
    try:
        if early:
            genuine_count = min(
                raw_count,
                max(1, int(math.floor(physical_survival_s * settings.video_fps)) + 1),
                target_count,
            )
            for index in range(genuine_count):
                last_base = reader.get_data(index)
                writer.append_data(
                    overlay_frame(last_base, str(spec["title"]), str(spec["subtitle"]))
                )
            if last_base is None:
                raise RuntimeError(f"Raw video has no genuine frames: {raw_video}")
            label = "FALL" if termination_reason.startswith("pelvis_") else "TERMINATED"
            termination_text = f"{label} - terminated at {physical_survival_s:.2f} s"
            padded = overlay_frame(
                last_base, str(spec["title"]), str(spec["subtitle"]), termination_text
            )
            padded_count = target_count - genuine_count
            for _ in range(padded_count):
                writer.append_data(padded)
        else:
            if raw_count < target_count:
                raise RuntimeError(
                    f"Full-duration raw video is too short: {raw_count} < {target_count}"
                )
            source_indices = np.rint(
                np.linspace(0, raw_count - 1, target_count)
            ).astype(int)
            if len(np.unique(source_indices)) != target_count:
                raise RuntimeError("Full-duration frame resampling produced duplicates")
            for index in source_indices:
                frame = reader.get_data(int(index))
                writer.append_data(
                    overlay_frame(frame, str(spec["title"]), str(spec["subtitle"]))
                )
            genuine_count = target_count
    finally:
        writer.close()
        reader.close()
    final_meta = video_metadata(output)
    freeze_tail_difference = None
    if padded_count >= 2:
        check_reader = imageio.get_reader(output)
        try:
            penultimate = check_reader.get_data(target_count - 2).astype(np.int16)
            last = check_reader.get_data(target_count - 1).astype(np.int16)
            freeze_tail_difference = float(np.mean(np.abs(penultimate - last)))
        finally:
            check_reader.close()
    checks = {
        "exists_nonempty": output.is_file() and output.stat().st_size > 1000,
        "exact_frame_count": final_meta["frame_count"] == target_count,
        "duration_20s": math.isclose(final_meta["duration_s"], settings.duration_s, abs_tol=1e-9),
        "fps_25": math.isclose(final_meta["fps"], settings.video_fps, abs_tol=1e-9),
        "resolution_960x544": (final_meta["width"], final_meta["height"]) == (VIDEO_WIDTH, VIDEO_HEIGHT),
        "h264": final_meta["codec"] == "h264",
        "three_frames_decodable": final_meta["decodable_samples"] == 3,
        "padding_only_after_early_termination": (padded_count > 0) is early,
        "freeze_tail_static": freeze_tail_difference is None or freeze_tail_difference <= 0.1,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Video validation failed for {output}: {checks}")
    return {
        **final_meta,
        "raw_video": str(raw_video.relative_to(PROJECT_ROOT)),
        "raw_frame_count": raw_count,
        "raw_duration_s": raw_count / raw_fps,
        "genuine_frame_count": genuine_count,
        "padded_frame_count": padded_count,
        "padding_mode": "last_genuine_frame_freeze" if padded_count else "none",
        "fall_annotation_applied": bool(padded_count),
        "title_overlay_applied": True,
        "freeze_tail_mean_abs_pixel_difference": freeze_tail_difference,
        "sha256": sha256(output),
        "checks": checks,
    }


def numeric(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.asarray([float(row[name]) for row in rows], dtype=float)


def analyze_case(
    spec: dict[str, Any],
    source: Path,
    final_video: Path,
    video: dict[str, Any],
    final_summary: dict[str, Any] | None,
    settings: DemoSettings,
) -> dict[str, Any]:
    validation = read_json(source / "validation.json")
    manifest = read_json(source / "manifest.json")
    rows = read_csv(source / "timeseries.csv")
    exo_present = spec["key"] != "patient_only"
    active = spec["key"] == "final_icra"
    strikes = [row for row in rows if row.get("heel_strike_side") in ("left", "both")]
    complete_cycles = max(0, len(strikes) - 1)
    average_speed = float(np.mean(numeric(rows, "forward_velocity")))
    peaks: dict[str, float | None] = {"hip": None, "knee": None}
    all_zero: bool | None = None
    right_zero: bool | None = None
    saturation: bool | None = None
    if exo_present:
        peaks = {
            "hip": float(np.max(np.abs(numeric(rows, "exo_actual_torque_HipExo_L_nm")))),
            "knee": float(np.max(np.abs(numeric(rows, "exo_actual_torque_KneeExo_L_nm")))),
        }
        all_zero = all(
            bool(
                np.all(numeric(rows, f"exo_command_torque_{channel}_nm") == 0.0)
                and np.all(numeric(rows, f"exo_actual_torque_{channel}_nm") == 0.0)
            )
            for channel in EXO_CHANNELS
        )
        right_zero = bool(
            np.all(numeric(rows, "exo_actual_torque_HipExo_R_nm") == 0.0)
            and np.all(numeric(rows, "exo_actual_torque_KneeExo_R_nm") == 0.0)
        )
        saturation = False if not active else bool(final_summary["native_saturation"])
    tracking = None if final_summary is None else float(final_summary["tracking_RMS_rad"])
    item = {
        "case_name": spec["name"],
        "pathology": "Moderate60 bilateral vasti weakness",
        "patient_config_sha256": manifest.get("patient_config_sha256"),
        "affected_muscles": manifest.get("affected_muscles"),
        "strength_scale": manifest.get("strength_scale"),
        "exo_present": exo_present,
        "active_assistance": active,
        "controller": (
            "Final Residual-Event ICRA (Stage7T)"
            if active
            else "Zero motor assistance" if exo_present else "MyoAssist reflex - no exoskeleton"
        ),
        "controller_mode": FINAL_MODE if active else None,
        "requested_duration_s": float(validation["requested_duration_s"]),
        "physical_survival_s": float(validation["actual_duration_s"]),
        "video_duration_s": float(video["duration_s"]),
        "termination_reason": str(validation["termination_reason"]),
        "average_speed_mps": average_speed,
        "complete_gait_cycles": complete_cycles,
        "complete_gait_cycles_semantics": "all complete intervals between successive left heel strikes",
        "tracking_rms_rad": tracking,
        "tracking_rms_unavailable_reason": None if tracking is not None else "No assistance reference/controller tracking error exists for this case.",
        "human_gait_rmse_rad": None,
        "human_gait_rmse_unavailable_reason": "No matched gait-normalized evaluation was requested for this video-only run.",
        "hip_motor_peak_nm": peaks["hip"],
        "knee_motor_peak_nm": peaks["knee"],
        "native_saturation": saturation,
        "right_motor_zero": right_zero,
        "all_exoskeleton_motor_torques_exact_zero": all_zero,
        "scientific_result_directory": str(source.relative_to(PROJECT_ROOT)),
        "scientific_validation": str((source / "validation.json").relative_to(PROJECT_ROOT)),
        "video_file": str(final_video.relative_to(PROJECT_ROOT)),
        "video": video,
        "survival_preserved_separately_from_video_duration": math.isclose(
            float(validation["actual_duration_s"]),
            float(read_json(source / "validation.json")["actual_duration_s"]),
            abs_tol=1e-12,
        ),
    }
    if active and (
        manifest.get("controller_mode") != FINAL_MODE
        or manifest.get("active_assistance") is not True
    ):
        raise RuntimeError("Case 3 is not the frozen Final Residual-Event mode")
    if spec["key"] == "passive_exoskeleton" and all_zero is not True:
        raise RuntimeError("Passive-exoskeleton torque is not exactly zero")
    return item


def make_contact_sheet(videos: dict[str, Path], output: Path) -> None:
    columns = (2.0, 9.0, 19.0)
    images: list[list[Image.Image]] = []
    for spec in CASE_SPECS:
        reader = imageio.get_reader(videos[str(spec["key"])])
        try:
            images.append(
                [Image.fromarray(reader.get_data(int(time_s * 25))) for time_s in columns]
            )
        finally:
            reader.close()
    width, height = VIDEO_WIDTH // 2, VIDEO_HEIGHT // 2
    sheet = Image.new("RGB", (width * 3, height * 3), "white")
    for row_index, row in enumerate(images):
        for column_index, frame in enumerate(row):
            sheet.paste(frame.resize((width, height), Image.Resampling.LANCZOS), (column_index * width, row_index * height))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def write_text_summary(path: Path, cases: dict[str, dict[str, Any]], story: str) -> None:
    lines = [
        "ICRA THREE-CASE 20-SECOND VIDEO DEMO",
        "",
    ]
    for spec in CASE_SPECS:
        item = cases[str(spec["name"])]
        lines += [
            f"{spec['number']}. {spec['name']}",
            f"   Physical survival: {item['physical_survival_s']:.2f} s",
            f"   Termination: {item['termination_reason']}",
            f"   Presentation video: {item['video_duration_s']:.2f} s",
            f"   Video: {item['video_file']}",
            "",
        ]
    lines += ["Expected visual story support: " + story, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_demo(settings: DemoSettings) -> int:
    if not math.isclose(settings.duration_s, 20.0, abs_tol=1e-12):
        raise ValueError("The formal demo duration must remain 20.0 s")
    if settings.video_fps != 25:
        raise ValueError("The formal demo FPS must remain 25")
    hashes_before = frozen_hashes()
    validate_patient_only_protocol_evidence()
    if not REFERENCE_RESULT.is_dir():
        raise FileNotFoundError(f"Frozen Healthy reference is missing: {REFERENCE_RESULT}")
    demo_root = unique_demo_directory()
    videos_root = demo_root / "videos"
    results_root = demo_root / "results"
    raw_root = demo_root / "raw_runs"
    frames_root = demo_root / "frames"
    for directory in (videos_root, results_root, raw_root, frames_root):
        directory.mkdir(parents=True, exist_ok=False)

    print("=" * 40)
    print("ICRA THREE-CASE DEMO")
    print("=" * 40)
    print(f"Fresh physics: {'NO' if settings.reuse_existing_results else 'YES'}")
    print("Duration: 20.0 s per presentation video")
    print("Video: 960x544, 25 FPS, H.264")
    print("=" * 40)

    if settings.reuse_existing_results:
        sources = latest_reusable_sources()
        final_summary_path = sources["final_icra"] / "summary.json"
        final_summaries = {
            "final_icra": read_json(final_summary_path)
            if final_summary_path.is_file()
            else {}
        }
    else:
        sources, final_summaries = run_fresh_cases(raw_root, settings, hashes_before)

    cases: dict[str, dict[str, Any]] = {}
    final_videos: dict[str, Path] = {}
    for spec in CASE_SPECS:
        key = str(spec["key"])
        source = sources[key]
        validation = read_json(source / "validation.json")
        final_video = videos_root / str(spec["filename"])
        metadata = build_presentation_video(
            source / "video.mp4",
            final_video,
            spec,
            float(validation["actual_duration_s"]),
            str(validation["termination_reason"]),
            settings,
        )
        final_videos[key] = final_video
        item = analyze_case(
            spec,
            source,
            final_video,
            metadata,
            final_summaries.get(key),
            settings,
        )
        cases[str(spec["name"])] = item
        write_json(results_root / str(spec["result_filename"]), item)

    patient = cases["Patient Only"]
    passive = cases["Passive Exoskeleton"]
    icra = cases["Final ICRA"]
    expected_story = bool(
        patient["termination_reason"] != "time_limit"
        and passive["termination_reason"] != "time_limit"
        and passive["physical_survival_s"] < patient["physical_survival_s"]
        and icra["physical_survival_s"] >= settings.duration_s - 0.02
        and icra["termination_reason"] == "time_limit"
    )
    story_label = "YES" if expected_story else "PARTIAL"
    summary = {
        "schema_version": "ICRA-three-case-demo-1.0",
        "status": "PASS",
        "pathology": "Moderate60 bilateral vasti weakness",
        "requested_duration_s": settings.duration_s,
        "presentation_video_duration_s": settings.duration_s,
        "cases": cases,
        "comparison": {
            "actual_survival_order_s": {
                name: cases[name]["physical_survival_s"]
                for name in ("Patient Only", "Passive Exoskeleton", "Final ICRA")
            },
            "expected_visual_story_supported": story_label,
            "ordering_was_not_enforced": True,
        },
    }
    write_json(results_root / "three_case_summary.json", summary)
    write_text_summary(results_root / "three_case_summary.txt", cases, story_label)
    contact_sheet = frames_root / "three_case_contact_sheet.png"
    make_contact_sheet(final_videos, contact_sheet)

    hashes_after = frozen_hashes()
    quality_checks = {
        "one_click_entry_exists": (PROJECT_ROOT / "run_ICRA_demo.py").is_file(),
        "three_scientific_cases": len(cases) == 3,
        "three_videos_exist": all(path.is_file() and path.stat().st_size > 1000 for path in final_videos.values()),
        "all_videos_exactly_20s": all(math.isclose(item["video_duration_s"], 20.0, abs_tol=1e-9) for item in cases.values()),
        "early_falls_freeze_padded": all(
            item["video"]["padding_mode"] == "last_genuine_frame_freeze"
            for item in cases.values()
            if item["physical_survival_s"] < 19.96
        ),
        "survival_values_preserved": all(item["survival_preserved_separately_from_video_duration"] for item in cases.values()),
        "passive_exo_exact_zero": passive["all_exoskeleton_motor_torques_exact_zero"] is True,
        "final_icra_mode": icra["controller_mode"] == FINAL_MODE,
        "final_icra_full_20s": icra["physical_survival_s"] >= 19.98 and icra["termination_reason"] == "time_limit",
        "frozen_hashes_unchanged": hashes_before == hashes_after,
        "matched_video_geometry": len({(item["video"]["width"], item["video"]["height"], item["video"]["fps"]) for item in cases.values()}) == 1,
        "all_outputs_decodable": all(item["video"]["checks"]["three_frames_decodable"] for item in cases.values()),
        "git_write_operations": False,
    }
    status = "PASS" if all(value is not False for key, value in quality_checks.items() if key != "git_write_operations") else "PARTIAL"
    summary["status"] = status
    write_json(results_root / "three_case_summary.json", summary)
    manifest = {
        "schema_version": "ICRA-three-case-demo-manifest-1.0",
        "status": status,
        "created_at": datetime.now().astimezone().isoformat(),
        "settings": {
            "reuse_existing_results": settings.reuse_existing_results,
            "duration_s": settings.duration_s,
            "video_fps": settings.video_fps,
            "video_resolution": [VIDEO_WIDTH, VIDEO_HEIGHT],
            "camera_id": runner.CAMERA_ID,
            "show_viewer": settings.show_viewer,
        },
        "scientific_policy": {
            "fresh_matched_runs": not settings.reuse_existing_results,
            "fall_criteria_unchanged": True,
            "early_fall_video_policy": "last genuine rendered frame frozen to exactly 20 s",
            "synthetic_motion": False,
            "controller_or_physics_tuning": False,
            "patient_only_protocol": "bare Human with the same frozen flexible-human reflex parameter file used by Cases 2 and 3",
            "patient_only_protocol_evidence": str(PATIENT_ONLY_PROTOCOL_EVIDENCE.relative_to(PROJECT_ROOT)),
        },
        "raw_result_directories": {
            key: str(path.relative_to(PROJECT_ROOT)) for key, path in sources.items()
        },
        "videos": {
            key: str(path.relative_to(PROJECT_ROOT)) for key, path in final_videos.items()
        },
        "summary": str((results_root / "three_case_summary.json").relative_to(PROJECT_ROOT)),
        "contact_sheet": str(contact_sheet.relative_to(PROJECT_ROOT)),
        "quality_checks": quality_checks,
        "expected_visual_story_supported": story_label,
        "frozen_hashes_before": hashes_before,
        "frozen_hashes_after": hashes_after,
        "git_write_operations": False,
    }
    write_json(demo_root / "manifest.json", manifest)

    print("\n" + "=" * 40)
    print("DEMO COMPLETE")
    print("=" * 40)
    for name in ("Patient Only", "Passive Exoskeleton", "Final ICRA"):
        item = cases[name]
        print(f"{name}: survival = {item['physical_survival_s']:.2f} s")
    print(f"Expected visual story: {story_label}")
    print(f"Videos: {videos_root}")
    print(f"Status: {status}")
    print("=" * 40)
    if settings.auto_open_result and platform.system() == "Darwin":
        subprocess.run(["open", str(demo_root)], check=False)
    return 0 if status == "PASS" else 1
