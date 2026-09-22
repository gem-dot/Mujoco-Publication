"""Deterministic 1080p renderer for the final ICRA experiment videos."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from visualization.ICRA_video_replay import DeterministicReplay


WIDTH, HEIGHT, FPS = 1920, 1080, 30
VIEWPORT_WIDTH = 1190
PANEL_LEFT, PANEL_RIGHT = 1222, 1892
BLUE = (44, 107, 176)
GREEN = (58, 145, 106)
ORANGE = (225, 126, 45)
PURPLE = (133, 92, 171)
RED = (196, 60, 60)
GRAY = (115, 123, 132)
GRID = (220, 225, 230)
DARK = (29, 38, 48)
HEADER_HEIGHT = 64
SEGMENT_TITLE_SIZE_OLD = 31
SEGMENT_TITLE_SIZE = 34
COUNTER_HZ = 5.0


@lru_cache(maxsize=None)
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    root = Path("/System/Library/Fonts/Supplemental")
    candidate = root / ("Arial Bold.ttf" if bold else "Arial.ttf")
    return ImageFont.truetype(str(candidate), size) if candidate.is_file() else ImageFont.load_default()


@lru_cache(maxsize=None)
def mono_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = ("DejaVuSansMono-Bold.ttf", "Menlo.ttc") if bold else ("DejaVuSansMono.ttf", "Menlo.ttc")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return font(size, bold)


def finite_limits(values: np.ndarray, fallback: tuple[float, float]) -> tuple[float, float]:
    good = values[np.isfinite(values)]
    if not len(good):
        return fallback
    low, high = float(np.min(good)), float(np.max(good))
    span = max(high - low, 1e-6)
    return low - 0.08 * span, high + 0.08 * span


def active_plot_constants(log: Any) -> dict[str, Any]:
    """Precompute immutable geometry/scales before the first rendered frame."""
    torque = np.concatenate((np.asarray(log["mu_hip"], float), np.asarray(log["mu_knee"], float)))
    torque_limit = max(6.0, math.ceil(float(np.nanmax(np.abs(torque))) + 0.5))
    s_limits = finite_limits(np.asarray(log["s"], float), (0.0, 0.5))
    schedule = np.column_stack((np.asarray(log["alpha"], float), np.asarray(log["gamma"], float)))
    schedule_limits = (
        min(0.0, s_limits[0], float(np.nanmin(schedule))),
        max(1.0, s_limits[1], float(np.nanmax(schedule))),
    )
    ratios = np.asarray(log["event_ratio"], float)
    finite_ratios = ratios[np.isfinite(ratios)]
    ratio_limit = max(1.25, min(3.0, float(np.nanpercentile(finite_ratios, 99.5)) if finite_ratios.size else 1.25))
    accepted_times = np.asarray(log["time"], float)[np.asarray(log["event_accepted"], bool)]
    return {
        "rects": ((70, 210, 675, 355), (70, 430, 675, 575),
                  (70, 650, 675, 795), (70, 870, 675, 970)),
        "x_limits_s": (0.0, 20.0),
        "tracking_limits_deg": (-55.0, 105.0),
        "torque_limits_nm": (-torque_limit, torque_limit),
        "schedule_limits": schedule_limits,
        "event_ratio_limits": (0.0, ratio_limit),
        "accepted_times_s": accepted_times,
    }


def _points(
    times: np.ndarray,
    values: np.ndarray,
    rect: tuple[int, int, int, int],
    x_max: float,
    y_limits: tuple[float, float],
) -> list[tuple[int, int]]:
    left, top, right, bottom = rect
    y0, y1 = y_limits
    if len(times) > 1:
        sample_dt = float(np.median(np.diff(times)))
        total_samples = int(round(x_max / sample_dt)) + 1 if sample_dt > 0 else len(times)
        stride = max(1, int(math.ceil(total_samples / 700)))
        if stride > 1:
            indices = np.arange(0, len(times), stride, dtype=int)
            if indices[-1] != len(times) - 1:
                indices = np.append(indices, len(times) - 1)
            times, values = times[indices], values[indices]
    valid = np.isfinite(times) & np.isfinite(values)
    x = left + np.clip(times[valid] / max(x_max, 1e-9), 0, 1) * (right - left)
    y = bottom - np.clip((values[valid] - y0) / max(y1 - y0, 1e-9), 0, 1) * (bottom - top)
    return [(int(a), int(b)) for a, b in zip(x, y)]


def draw_plot(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    title: str,
    times: np.ndarray,
    series: list[tuple[str, np.ndarray, tuple[int, int, int], bool]],
    *,
    current_time: float,
    x_max: float,
    y_limits: tuple[float, float],
    unit: str,
    thresholds: tuple[float, ...] = (),
    event_times: np.ndarray | None = None,
) -> None:
    left, top, right, bottom = rect
    draw.rounded_rectangle((left - 10, top - 30, right + 8, bottom + 25), 10, fill="white", outline=(208, 214, 220), width=2)
    draw.text((left, top - 27), title, font=font(18, True), fill=DARK)
    for fraction in (0.0, 0.5, 1.0):
        y = int(bottom - fraction * (bottom - top))
        draw.line((left, y, right, y), fill=GRID, width=1)
        value = y_limits[0] + fraction * (y_limits[1] - y_limits[0])
        draw.text((left - 4, y - 7), f"{value:.2g}", font=font(13), fill=GRAY, anchor="ra")
    for threshold in thresholds:
        if y_limits[0] <= threshold <= y_limits[1]:
            y = int(bottom - (threshold - y_limits[0]) / (y_limits[1] - y_limits[0]) * (bottom - top))
            for x in range(left, right, 12):
                draw.line((x, y, min(x + 6, right), y), fill=GRAY, width=2)
    for name, values, color, dashed in series:
        pts = _points(times, values, rect, x_max, y_limits)
        if len(pts) >= 2:
            if dashed:
                for j in range(0, len(pts) - 1, 8):
                    draw.line(pts[j : min(j + 5, len(pts))], fill=color, width=3)
            else:
                draw.line(pts, fill=color, width=3, joint="curve")
    if event_times is not None:
        for event_time in event_times[event_times <= current_time]:
            x = int(left + event_time / max(x_max, 1e-9) * (right - left))
            draw.ellipse((x - 3, top + 4, x + 3, top + 10), fill=ORANGE)
    cursor = int(left + min(current_time, x_max) / max(x_max, 1e-9) * (right - left))
    draw.line((cursor, top, cursor, bottom), fill=(45, 48, 52), width=2)
    draw.text((right, bottom + 6), f"{x_max:.1f} s", font=font(13), fill=GRAY, anchor="ra")
    draw.text((left, bottom + 6), unit, font=font(13), fill=GRAY)
    lx = right
    for name, _, color, dashed in reversed(series):
        text_w = draw.textbbox((0, 0), name, font=font(13))[2]
        lx -= text_w + 31
        draw.line((lx, top - 16, lx + 18, top - 16), fill=color, width=2)
        if dashed:
            draw.line((lx + 7, top - 16, lx + 11, top - 16), fill="white", width=3)
        draw.text((lx + 22, top - 24), name, font=font(13), fill=DARK)


def _slice(log: Any, key: str, end: int, scale: float = 1.0) -> np.ndarray:
    return np.asarray(log[key][: end + 1], dtype=float) * scale


def telemetry_image(
    replay: DeterministicReplay,
    case: dict[str, Any],
    index: int,
    current_time: float,
    display_time: float,
    ending: bool,
) -> Image.Image:
    log = replay.log
    image = Image.new("RGB", (WIDTH - VIEWPORT_WIDTH, HEIGHT), (246, 248, 250))
    draw = ImageDraw.Draw(image)
    times = _slice(log, "time", index)
    x_max = round(float(case["physical_duration_s"]), 6)
    exo = case["exo_state"]
    assistance = case["assistance_state"]
    draw.text((32, 18), case["title"], font=font(32, True), fill=DARK)
    draw.text((32, 58), case["subtitle"], font=font(17), fill=GRAY)
    status = [
        f"Condition: {case['condition_label']}",
        f"Time: {min(current_time, x_max):05.2f} s",
        f"Vasti capacity: {case['vasti_capacity']}",
        f"Exoskeleton: {exo}",
        f"Assistance: {assistance}",
    ]
    for row, value in enumerate(status):
        draw.text((32 + (row % 2) * 355, 101 + (row // 2) * 27), value, font=font(16, row == 1), fill=DARK)

    active = bool(case["active"])
    if active:
        constants = case["active_plot_constants"]
        rects = constants["rects"]
        draw_plot(draw, rects[0], "A · Tracking · solid actual / dashed ref", times,
                  [("H", _slice(log, "q_exo_hip_l", index, 180/math.pi), BLUE, False),
                   ("Hd", _slice(log, "qd_hip_l", index, 180/math.pi), BLUE, True),
                   ("K", _slice(log, "q_exo_knee_l", index, 180/math.pi), GREEN, False),
                   ("Kd", _slice(log, "qd_knee_l", index, 180/math.pi), GREEN, True)],
                  current_time=current_time, x_max=x_max,
                  y_limits=constants["tracking_limits_deg"], unit="deg")
        draw_plot(draw, rects[1], "B · Physical Motor Torque", times,
                  [("hip", _slice(log, "mu_hip", index), BLUE, False), ("knee", _slice(log, "mu_knee", index), GREEN, False)],
                  current_time=current_time, x_max=x_max,
                  y_limits=constants["torque_limits_nm"], unit="N·m")
        draw.line((rects[1][0], rects[1][1] + 2, rects[1][2], rects[1][1] + 2), fill=GRAY, width=1)
        draw.line((rects[1][0], rects[1][3] - 2, rects[1][2], rects[1][3] - 2), fill=GRAY, width=1)
        draw.text((rects[1][2] - 5, rects[1][1] + 7), "native limits ±100 N·m (off-scale)",
                  font=font(12), fill=GRAY, anchor="ra")
        draw_plot(draw, rects[2], "C · Activation Scheduling", times,
                  [("s(t)", _slice(log, "s", index), BLUE, False),
                   ("alpha", _slice(log, "alpha", index), ORANGE, False),
                  ("gamma", _slice(log, "gamma", index), PURPLE, False)],
                  current_time=current_time, x_max=x_max,
                  y_limits=constants["schedule_limits"], unit="native values")
        ratio = _slice(log, "event_ratio", index)
        draw_plot(draw, rects[3], "D · Residual Event Execution", times,
                  [("ratio", ratio, PURPLE, False)], current_time=current_time, x_max=x_max,
                  y_limits=constants["event_ratio_limits"], unit="||ēr||² / Θr",
                  thresholds=(1.0,), event_times=constants["accepted_times_s"])

        counter_time = math.floor((current_time + 1e-9) * COUNTER_HZ) / COUNTER_HZ
        counter_index = max(0, int(np.searchsorted(log["time"], counter_time, side="right")) - 1)
        eligible = int(log["eligible_count"][counter_index])
        accepted = int(log["accepted_count"][counter_index])
        reduction = 100.0 * (1.0 - accepted / eligible) if eligible else 0.0
        draw.rounded_rectangle((60, 997, 683, 1060), 8, fill=(239, 242, 245), outline=(208, 214, 220), width=2)
        labels = (("Eligible", f"{eligible:4d}"), ("Accepted", f"{accepted:4d}"),
                  ("Reduction", f"{reduction:5.1f}%"))
        for column, (label, value) in enumerate(labels):
            x = 93 + column * 202
            draw.text((x, 1005), label, font=font(13, True), fill=GRAY)
            draw.text((x + 132, 1033), value, font=mono_font(17, True), fill=DARK, anchor="ra")
    else:
        rects = [(70, 230, 675, 380), (70, 470, 675, 620), (70, 710, 675, 860), (70, 945, 675, 1015)]
        draw_plot(draw, rects[0], "A1 · Pelvis Height", times,
                  [("height", _slice(log, "pelvis_height", index), BLUE, False)], current_time=current_time,
                  x_max=x_max, y_limits=(0.5, 1.15), unit="m", thresholds=(0.65,))
        draw_plot(draw, rects[1], "A2 · Pelvis Pitch", times,
                  [("pitch", _slice(log, "pelvis_pitch_deg", index), BLUE, False)], current_time=current_time,
                  x_max=x_max, y_limits=(-70, 70), unit="deg", thresholds=(-60, 60))
        draw_plot(draw, rects[2], "B · Left Lower-Limb Kinematics", times,
                  [("hip", _slice(log, "q_hip_l", index, 180/math.pi), BLUE, False),
                   ("knee", _slice(log, "q_knee_l", index, 180/math.pi), GREEN, False)],
                  current_time=current_time, x_max=x_max, y_limits=(-55, 105), unit="deg")
        draw_plot(draw, rects[3], "C · Muscle Activation RMS", times,
                  [("s(t)", _slice(log, "activation_rms", index), BLUE, False)], current_time=current_time,
                  x_max=x_max, y_limits=(0, max(0.45, float(np.nanmax(log["activation_rms"])) * 1.08)), unit="simulation activation")
        if case["passive"]:
            draw.rounded_rectangle((425, 166, 680, 206), 8, fill=(233, 237, 241))
            draw.text((552, 186), "MOTORS OFF   μ = 0", font=font(17, True), fill=GRAY, anchor="mm")

    if ending:
        fall = case["termination_reason"] != "time_limit"
        box = (60, 420, 675, 660)
        draw.rounded_rectangle(box, 18, fill=(255, 247, 244) if fall else (240, 249, 245), outline=RED if fall else GREEN, width=4)
        if fall:
            reason = {
                "pelvis_height_below_0.65_m": "Pelvis height threshold reached",
                "pelvis_pitch_above_plus_60_deg": "Pelvis pitch threshold exceeded",
                "pelvis_pitch_below_minus_60_deg": "Pelvis pitch threshold exceeded",
            }.get(case["termination_reason"], case["termination_reason"])
            draw.text((367, 485), "FALL DETECTED", font=font(35, True), fill=RED, anchor="mm")
            draw.text((367, 545), f"{case['physical_duration_s']:.2f} s", font=font(27, True), fill=DARK, anchor="mm")
            draw.text((367, 598), reason, font=font(19), fill=DARK, anchor="mm")
        else:
            headline = "Stable healthy gait" if case["key"] == "healthy" else "20.00 s  EVALUATION COMPLETED"
            sub = "Evaluation completed" if case["key"] == "healthy" else "No fall criterion triggered"
            draw.text((367, 500), headline, font=font(31, True), fill=GREEN, anchor="mm")
            draw.text((367, 565), sub, font=font(21), fill=DARK, anchor="mm")
    return image


def render_case(
    config: Any,
    trajectory: Path,
    output: Path,
    contact_sheet: Path,
    case: dict[str, Any],
) -> dict[str, Any]:
    replay = DeterministicReplay(config, trajectory)
    case = dict(case)
    if bool(case["active"]):
        case["active_plot_constants"] = active_plot_constants(replay.log)
    renderer = mujoco.Renderer(replay.model, height=HEIGHT, width=VIEWPORT_WIDTH)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = 90.0
    camera.elevation = -4.0
    camera.distance = 4.25
    physical = float(case["physical_duration_s"])
    physical_for_timeline = round(physical, 6)
    tail = 1.2 if case["termination_reason"] != "time_limit" else 1.0
    frames = int(math.ceil((physical_for_timeline + tail) * FPS))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output, fps=FPS, codec="libx264", pixelformat="yuv420p", macro_block_size=1,
                                ffmpeg_log_level="error", output_params=["-crf", "18", "-preset", "medium", "-movflags", "+faststart"])
    samples: list[Image.Image] = []
    sample_indices = set(np.linspace(0, frames - 1, 6).astype(int).tolist())
    try:
        for frame_index in range(frames):
            display_time = frame_index / FPS
            sim_time = min(display_time, physical_for_timeline)
            pose_index = replay.nearest_index(sim_time)
            plot_index = max(0, int(np.searchsorted(replay.log["time"], sim_time, side="right")) - 1)
            replay.restore(pose_index)
            pelvis_x = float(replay.log["pelvis_x"][pose_index])
            camera.lookat[:] = (pelvis_x, 0.0, 0.95)
            renderer.update_scene(replay.data, camera=camera)
            viewport = renderer.render()
            base = Image.new("RGB", (WIDTH, HEIGHT), (246, 248, 250))
            base.paste(Image.fromarray(viewport), (0, 0))
            ending = display_time >= physical_for_timeline
            panel = telemetry_image(replay, case, plot_index, sim_time, display_time, ending)
            base.paste(panel, (VIEWPORT_WIDTH, 0))
            draw = ImageDraw.Draw(base)
            draw.rectangle((0, 0, VIEWPORT_WIDTH, HEADER_HEIGHT), fill=(22, 31, 40))
            draw.text((26, 3), case["title"], font=font(SEGMENT_TITLE_SIZE, True), fill="white")
            draw.text((26, 42), case["subtitle"], font=font(16), fill=(210, 217, 224))
            draw.text((VIEWPORT_WIDTH - 28, HEADER_HEIGHT // 2), f"MuJoCo · t = {sim_time:05.2f} s",
                      font=font(17, True), fill="white", anchor="rm")
            arr = np.asarray(base)
            writer.append_data(arr)
            if frame_index in sample_indices:
                samples.append(base.resize((960, 540), Image.Resampling.LANCZOS))
    finally:
        writer.close()
        renderer.close()
    sheet = Image.new("RGB", (1920, 1620), "white")
    for i, sample in enumerate(samples):
        sheet.paste(sample, ((i % 2) * 960, (i // 2) * 540))
    contact_sheet.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(contact_sheet)
    count, seconds = imageio_ffmpeg.count_frames_and_secs(str(output))
    return {"frame_count": int(count), "duration_s": float(seconds), "fps": FPS,
            "resolution": [WIDTH, HEIGHT], "contact_sheet": str(contact_sheet), "freeze_tail_s": tail}


def card(lines: list[tuple[str, int, tuple[int, int, int], bool]]) -> np.ndarray:
    image = Image.new("RGB", (WIDTH, HEIGHT), (247, 249, 251))
    draw = ImageDraw.Draw(image)
    y = 310
    for text, size, color, bold in lines:
        draw.text((WIDTH // 2, y), text, font=font(size, bold), fill=color, anchor="mm")
        y += size + 35
    return np.asarray(image)


def render_overview(videos: list[Path], output: Path, summary: dict[str, Any]) -> dict[str, Any]:
    segment_lengths = [12, 12, 12, 20]
    title_frames = 3 * FPS
    summary_frames = 8 * FPS
    writer = imageio.get_writer(output, fps=FPS, codec="libx264", pixelformat="yuv420p", macro_block_size=1,
                                ffmpeg_log_level="error", output_params=["-crf", "18", "-preset", "medium", "-movflags", "+faststart"])
    title = card([
        ("Muscle Activation Guided Admittance and Impedance Control of", 52, DARK, True),
        ("Lower Limb Exoskeletons via Differential Games with", 52, DARK, True),
        ("Neuromusculoskeletal Evaluation", 48, BLUE, True),
    ])
    try:
        for _ in range(title_frames): writer.append_data(title)
        for path, length in zip(videos, segment_lengths):
            reader = imageio.get_reader(path)
            count = reader.count_frames()
            wanted = length * FPS
            try:
                last_frame: np.ndarray | None = None
                for j in range(wanted):
                    if j < count:
                        last_frame = reader.get_data(j)
                    if last_frame is None:
                        raise RuntimeError(f"Overview source has no frames: {path}")
                    frame = last_frame
                    writer.append_data(frame)
            finally:
                reader.close()
        event = summary["segments"]["active"]
        reduction = event["refresh_reduction_percent"]
        final = card([("Experiment Summary", 43, DARK, True),
                      ("Healthy Human · Stable gait", 28, BLUE, False),
                      (f"Moderate60 · Fall at {summary['segments']['impairment']['duration_s']:.2f} s", 28, RED, False),
                      (f"Passive Wear · Fall at {summary['segments']['passive']['duration_s']:.2f} s", 28, RED, False),
                      ("Active Assistance · 20 s · No fall", 30, GREEN, True),
                      (f"Residual event execution · {reduction:.1f}% fewer learned correction refreshes", 24, ORANGE, False)])
        for _ in range(summary_frames): writer.append_data(final)
    finally:
        writer.close()
    count, seconds = imageio_ffmpeg.count_frames_and_secs(str(output))
    return {"frame_count": int(count), "duration_s": float(seconds), "fps": FPS, "resolution": [WIDTH, HEIGHT]}
