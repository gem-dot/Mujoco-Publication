"""Experiment-aware scientific figures for saved MuJoCo results."""

from __future__ import annotations

from pathlib import Path
import textwrap
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation import metrics


COLORS = {
    "right": "#2878B5",
    "left": "#C44E52",
    "reference": "#222222",
    "accent": "#5B4B8A",
    "neutral": "#6B7280",
}
INTERFACE_COLORS = {
    "waist": "#4C78A8",
    "thigh_l": "#E45756",
    "thigh_r": "#72B7B2",
    "shank_l": "#F2CF5B",
    "shank_r": "#B279A2",
}
MANAGED_FIGURE_NAMES = {
    "01_stability.png",
    "02_joint_gait_cycles.png",
    "03_left_knee_tracking.png",
    "04_spatiotemporal_gait.png",
    "05_muscle_activation_heatmap.png",
    "06_key_muscle_profiles.png",
    "07_grf_bw.png",
    "08_exo_torque.png",
    "09_exo_power_work.png",
    "10_summary_metrics.png",
    "02_gait_kinematics.png",
    "03_gait_spatiotemporal.png",
    "04_grf_touch_force_bw.png",
    "06_muscle_activation_key_profiles.png",
    "07_muscle_force_heatmap.png",
    "08_muscle_force_key_profiles.png",
    "09_human_effort_proxy.png",
    "10_exo_torque.png",
    "11_exo_power_work.png",
    "12_interface_relative_motion.png",
    "13_interface_load.png",
    "01_exo_joint_state.png",
    "02_exo_torque.png",
    "03_exo_power.png",
    "04_exo_joint_limits.png",
}

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "lines.linewidth": 1.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _unavailable(ax: plt.Axes, title: str, reason: str) -> None:
    ax.set_title(title)
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        f"UNAVAILABLE\n\n{reason}",
        ha="center",
        va="center",
        transform=ax.transAxes,
        color="#555555",
        bbox={
            "boxstyle": "round,pad=0.8",
            "facecolor": "#F3F4F6",
            "edgecolor": "#D1D5DB",
        },
    )


def _value(summary: dict[str, Any], category: str, name: str) -> Any:
    if category not in summary.get("metrics", {}):
        return None
    return metrics.metric_value(summary["metrics"][category], name)


def _raw(table: dict[str, np.ndarray], aliases: list[str]) -> np.ndarray | None:
    values, _ = metrics.numeric_column(table, aliases)
    return values


def _format(value: Any, unit: str = "", digits: int = 3) -> str:
    if value is None:
        return "Unavailable"
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if isinstance(value, str):
        return value
    number = float(value)
    text = f"{number:.{digits}g}"
    return f"{text} {unit}".strip()


def _max_metric(summary: dict[str, Any], category: str, suffix: str) -> float | None:
    values: list[float] = []
    for name, item in summary.get("metrics", {}).get(category, {}).get("metrics", {}).items():
        if name.endswith(suffix) and item.get("available"):
            value = item.get("value")
            if isinstance(value, (int, float)) and np.isfinite(value):
                values.append(float(value))
    return max(values) if values else None


def _plot_overview(summary: dict[str, Any], path: Path) -> None:
    experiment = summary.get("experiment", {})
    experiment_type = str(experiment.get("experiment_type") or "inferred")
    cards: list[tuple[str, Any, str]] = []
    if "stability" in summary.get("metrics", {}):
        cards.extend(
            [
                ("Walking duration", _value(summary, "stability", "walking_duration_s"), "s"),
                ("Mean forward speed", _value(summary, "stability", "mean_forward_velocity_mps"), "m/s"),
                ("Minimum pelvis height", _value(summary, "stability", "min_pelvis_height_m"), "m"),
                ("Peak pelvis pitch", _value(summary, "stability", "max_abs_pelvis_pitch_deg"), "deg"),
            ]
        )
    if "gait" in summary.get("metrics", {}):
        cards.append(
            ("Step-length symmetry", _value(summary, "gait", "step_length_symmetry_index_percent"), "%")
        )
    if "human_effort" in summary.get("metrics", {}):
        cards.append(
            ("Human effort proxy", _value(summary, "human_effort", "activation_squared_mean"), "")
        )
    if "exoskeleton" in summary.get("metrics", {}):
        cards.extend(
            [
                ("Peak actual exo torque", _max_metric(summary, "exoskeleton", ".peak_abs_torque_Nm"), "N·m"),
                ("Absolute mechanical work", _value(summary, "exoskeleton", "available_channels_total_absolute_work_J"), "J"),
            ]
        )
    if "interaction" in summary.get("metrics", {}):
        cards.extend(
            [
                ("Peak interface force", _max_metric(summary, "interaction", ".force.peak"), "N"),
                ("Peak relative displacement", _max_metric(summary, "interaction", ".translation.peak"), "m"),
            ]
        )
    if "state" in summary.get("metrics", {}):
        cards.extend(
            [
                ("Bench duration", _value(summary, "state", "duration_s"), "s"),
                ("Finite state", _value(summary, "state", "finite_state"), ""),
                ("Peak actual torque", _max_metric(summary, "torque", ".peak_abs_actual_torque_Nm"), "N·m"),
                ("Peak positive power", _max_metric(summary, "power", ".peak_positive_power_W"), "W"),
                ("Joint limits respected", _value(summary, "joint_limits", "all_joint_limits_respected"), ""),
            ]
        )

    columns = 3
    rows = max(1, int(np.ceil(len(cards) / columns)))
    fig, axes = plt.subplots(rows, columns, figsize=(13, 2.6 * rows + 1.2))
    axes_array = np.atleast_1d(axes).ravel()
    for ax, (label, value, unit) in zip(axes_array, cards):
        ax.axis("off")
        ax.text(0.5, 0.62, _format(value, unit), ha="center", va="center", fontsize=19, weight="bold")
        ax.text(0.5, 0.28, label, ha="center", va="center", color="#4B5563")
        ax.add_patch(
            plt.Rectangle(
                (0.03, 0.08), 0.94, 0.82, transform=ax.transAxes,
                facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.0,
            )
        )
    for ax in axes_array[len(cards):]:
        ax.axis("off")
    name = str(
        experiment.get("manifest_experiment_name")
        or Path(str(experiment.get("directory", "experiment"))).name
    ).replace("_", " ")
    status = _value(summary, "stability", "termination_reason")
    if status is None:
        status = experiment.get("test_mode", "")
    title = textwrap.fill(f"Experiment Overview — {name}", width=72)
    fig.suptitle(title, fontsize=16, weight="bold")
    fig.text(0.5, 0.01, f"Type: {experiment_type}   |   Termination/Test: {status}", ha="center", color="#4B5563")
    _save(fig, path)


def _plot_stability(summary: dict[str, Any], table: dict[str, np.ndarray], path: Path) -> None:
    time = _raw(table, ["time", "simulation_time"])
    signals = [
        ("Forward speed", _raw(table, ["forward_velocity", "forward_velocity_mps"]), "m/s"),
        ("Pelvis height", _raw(table, ["pelvis_height", "pelvis_height_m"]), "m"),
        ("Pelvis pitch", _raw(table, ["pelvis_pitch_deg", "pelvis_pitch"]), "deg"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    for ax, (title, values, unit) in zip(axes, signals):
        if time is None or values is None:
            _unavailable(ax, title, "required time-series signal is missing")
            continue
        ax.plot(time, values, color=COLORS["right"])
        ax.set_ylabel(unit)
        ax.set_title(title, loc="left")
        ax.grid(alpha=0.22)
    target = summary.get("evaluation_context", {}).get("target_velocity_mps")
    if target is not None:
        axes[0].axhline(float(target), color=COLORS["neutral"], linestyle="--", label=f"Target {float(target):.2f} m/s")
        axes[0].legend()
    axes[1].axhline(metrics.FALL_PELVIS_HEIGHT_M, color="#D62728", linestyle="--", label="Fall threshold")
    axes[1].legend()
    axes[2].axhline(metrics.FALL_PELVIS_PITCH_DEG, color="#D62728", linestyle="--")
    axes[2].axhline(-metrics.FALL_PELVIS_PITCH_DEG, color="#D62728", linestyle="--", label="Fall thresholds")
    axes[2].legend()
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Stability and Walking State", fontsize=15)
    _save(fig, path)


def _plot_joint_cycles(summary: dict[str, Any], path: Path) -> None:
    profiles = summary["profiles"]["gait"]
    phase = np.asarray(profiles["phase_percent"])
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    for row, joint in enumerate(metrics.JOINTS):
        for col, side in enumerate(("right", "left")):
            ax = axes[row, col]
            entry = profiles["joints"][side][joint]
            current = entry.get("current_deg")
            current_std = entry.get("current_std_deg")
            reference = entry.get("healthy_reference_deg")
            if current is None:
                _unavailable(ax, f"{side.title()} {joint}", "valid gait cycles are unavailable")
                continue
            if reference is not None:
                ax.plot(phase, reference, color=COLORS["reference"], linestyle="--", label="Healthy reference")
            if current_std is not None:
                mean_values = np.asarray(current, dtype=float)
                std_values = np.asarray(current_std, dtype=float)
                ax.fill_between(
                    phase,
                    mean_values - std_values,
                    mean_values + std_values,
                    color=COLORS[side],
                    alpha=0.16,
                    linewidth=0,
                    label="Mean ± std",
                )
            ax.plot(phase, current, color=COLORS[side], label="Current")
            ax.set_title(f"{side.title()} {joint.title()} — {entry['current_cycles']} cycles", loc="left")
            ax.set_ylabel("Angle (deg)")
            ax.grid(alpha=0.22)
            if reference is not None:
                ax.legend()
    for ax in axes[-1, :]:
        ax.set_xlabel("Gait cycle (%)")
    fig.suptitle("Gait-normalized Joint Kinematics", fontsize=15)
    _save(fig, path)


def _paired_dots(ax: plt.Axes, labels: tuple[str, str], values: tuple[Any, Any], title: str, unit: str) -> None:
    valid = [(label, float(value)) for label, value in zip(labels, values) if value is not None]
    if not valid:
        _unavailable(ax, title, "required gait metric is unavailable")
        return
    for index, (label, value) in enumerate(valid):
        color = COLORS["right"] if label.startswith("R") else COLORS["left"]
        ax.plot([0, value], [index, index], color="#CBD5E1", linewidth=2)
        ax.scatter(value, index, s=70, color=color, zorder=3)
        ax.text(value, index + 0.14, f"{value:.3g}", ha="center", fontsize=9)
    ax.set_yticks(range(len(valid)), [item[0] for item in valid])
    ax.set_xlabel(unit)
    ax.set_title(title, loc="left")
    ax.grid(axis="x", alpha=0.22)


def _plot_spatiotemporal(summary: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    _paired_dots(
        axes[0, 0], ("Right", "Left"),
        (_value(summary, "gait", "right_mean_step_length_m"), _value(summary, "gait", "left_mean_step_length_m")),
        "Mean step length", "m",
    )
    _paired_dots(
        axes[0, 1], ("Right", "Left"),
        (_value(summary, "gait", "right_mean_cycle_time_s"), _value(summary, "gait", "left_mean_cycle_time_s")),
        "Mean gait-cycle time", "s",
    )
    _paired_dots(
        axes[1, 0], ("R stance", "L stance"),
        (_value(summary, "gait", "right_mean_stance_time_s"), _value(summary, "gait", "left_mean_stance_time_s")),
        "Mean stance duration", "s",
    )
    axes[1, 1].axis("off")
    symmetry_rows = [
        ("Step length", _value(summary, "gait", "step_length_symmetry_index_percent")),
        ("Cycle time", _value(summary, "gait", "cycle_time_symmetry_index_percent")),
    ]
    axes[1, 1].text(0.0, 0.92, "Bilateral symmetry", fontsize=12, weight="bold")
    for index, (name, value) in enumerate(symmetry_rows):
        axes[1, 1].text(0.05, 0.68 - 0.28 * index, name, color="#4B5563")
        axes[1, 1].text(0.95, 0.68 - 0.28 * index, _format(value, "%"), ha="right", fontsize=16, weight="bold")
    axes[1, 1].text(0.05, 0.08, "0% is symmetric; components are reported separately.", color="#6B7280", fontsize=9)
    fig.suptitle("Spatiotemporal Gait Summary", fontsize=15)
    _save(fig, path)


def _plot_grf(summary: dict[str, Any], path: Path) -> None:
    profiles = summary["profiles"]["grf"]
    fig, (curve_ax, metric_ax) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [2.2, 1]})
    if not profiles.get("available"):
        _unavailable(curve_ax, "Foot touch force / body weight", profiles.get("reason", "touch-force signals unavailable"))
        _unavailable(metric_ax, "Contact summary", "touch-force signals unavailable")
        _save(fig, path)
        return
    phase = np.asarray(profiles["phase_percent"])
    for side in ("right", "left"):
        profile = profiles[side].get("gait_profile_bw")
        if profile is not None:
            curve_ax.plot(phase, profile, color=COLORS[side], label=side.title())
    curve_ax.axhline(metrics.CONTACT_THRESHOLD_BW, color=COLORS["neutral"], linestyle="--", linewidth=1, label="Contact threshold")
    curve_ax.set_xlabel("Gait cycle (%)")
    curve_ax.set_ylabel("Heel + toe touch normal force / total-system BW")
    curve_ax.set_title("Gait-normalized foot touch force", loc="left")
    curve_ax.grid(alpha=0.22)
    curve_ax.legend()
    peak_values = [
        _value(summary, "grf", "peak_right_grf_bw"),
        _value(summary, "grf", "peak_left_grf_bw"),
    ]
    for index, (label, value, color) in enumerate(zip(("Right", "Left"), peak_values, (COLORS["right"], COLORS["left"]))):
        if value is not None:
            metric_ax.scatter(float(value), index, s=80, color=color)
            metric_ax.plot([0, float(value)], [index, index], color="#CBD5E1")
            metric_ax.text(float(value), index + 0.14, f"{float(value):.3g} BW", ha="center")
    metric_ax.set_yticks((0, 1), ("Right peak", "Left peak"))
    metric_ax.set_xlabel("Body weights")
    metric_ax.set_title("Recorded peak", loc="left")
    metric_ax.grid(axis="x", alpha=0.22)
    fig.suptitle("Foot Touch Normal Force — not 3D GRF or plantar pressure", fontsize=14)
    _save(fig, path)


def _plot_muscle_heatmap(summary: dict[str, Any], path: Path, signal: str) -> None:
    phase = np.asarray(summary["profiles"]["muscle"]["phase_percent"])
    source = summary["profiles"]["muscle"][signal]
    profile_key = "profile" if signal == "activation" else "profile_N"
    available = [(name, source.get(name, {}).get(profile_key)) for name in metrics.MUSCLES]
    available = [(name, values) for name, values in available if values is not None]
    fig, ax = plt.subplots(figsize=(12, 8.5))
    if not available:
        _unavailable(ax, f"Muscle {signal} heatmap", "muscle signal or valid gait cycles are unavailable")
        _save(fig, path)
        return
    names = [item[0] for item in available]
    matrix = np.asarray([item[1] for item in available])
    if signal == "activation":
        image = ax.imshow(matrix, aspect="auto", extent=[phase[0], phase[-1], len(names) - 0.5, -0.5], cmap="viridis", vmin=0.0, vmax=1.0)
        label = "MuJoCo activation (0–1)"
        title = "22-muscle MuJoCo Activation — not measured EMG"
    else:
        image = ax.imshow(matrix, aspect="auto", extent=[phase[0], phase[-1], len(names) - 0.5, -0.5], cmap="magma", vmin=0.0)
        label = "Absolute actuator force (N)"
        title = "22-muscle Actuator Force"
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Gait cycle (%)")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=label)
    _save(fig, path)


def _plot_key_muscles(summary: dict[str, Any], path: Path, signal: str) -> None:
    phase = np.asarray(summary["profiles"]["muscle"]["phase_percent"])
    source = summary["profiles"]["muscle"][signal]
    profile_key = "profile" if signal == "activation" else "profile_N"
    fig, axes = plt.subplots(len(metrics.KEY_MUSCLE_GROUPS), 1, figsize=(10, 12), sharex=True)
    for ax, muscle in zip(axes, metrics.KEY_MUSCLE_GROUPS):
        plotted = False
        for side, color in (("r", COLORS["right"]), ("l", COLORS["left"])):
            name = f"{muscle}_{side}"
            values = source.get(name, {}).get(profile_key)
            if values is None:
                continue
            ax.plot(phase, values, color=color, label="Right" if side == "r" else "Left")
            plotted = True
        if not plotted:
            _unavailable(ax, muscle, f"{signal} gait profile unavailable")
            continue
        ax.set_title(muscle, loc="left")
        ax.set_ylabel("Activation" if signal == "activation" else "Force (N)")
        ax.grid(alpha=0.22)
        ax.legend(ncol=2)
    axes[-1].set_xlabel("Gait cycle (%)")
    suffix = "MuJoCo activation, not EMG" if signal == "activation" else "absolute actuator force"
    fig.suptitle(f"Key Muscle Profiles — {suffix}", fontsize=14)
    _save(fig, path)


def _plot_human_effort(summary: dict[str, Any], table: dict[str, np.ndarray], path: Path) -> None:
    time = _raw(table, ["time", "simulation_time"])
    activation_arrays = []
    contributions = []
    names = []
    for muscle in metrics.MUSCLES:
        values, _ = metrics._muscle_column(table, muscle, "activation")
        if values is not None and len(metrics._finite(values)):
            activation_arrays.append(values)
            contributions.append(float(np.nanmean(np.square(values))))
            names.append(muscle)
    fig, (time_ax, contribution_ax) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.7, 1]})
    if time is None or len(activation_arrays) != len(metrics.MUSCLES):
        _unavailable(time_ax, "Human effort proxy", "all 22 activation signals are required")
        _unavailable(contribution_ax, "Muscle contributions", "all 22 activation signals are required")
        _save(fig, path)
        return
    matrix = np.column_stack(activation_arrays)
    total = np.sum(np.square(matrix), axis=1)
    time_ax.plot(time, total, color=COLORS["accent"])
    time_ax.axhline(float(np.nanmean(total)), color=COLORS["neutral"], linestyle="--", label="Recorded mean")
    time_ax.set_xlabel("Time (s)")
    time_ax.set_ylabel(r"$\sum_i a_i^2$")
    time_ax.set_title("Instantaneous activation-squared sum", loc="left")
    time_ax.grid(alpha=0.22)
    time_ax.legend()
    y = np.arange(len(names))
    colors = [COLORS["right"] if name.endswith("_r") else COLORS["left"] for name in names]
    contribution_ax.scatter(contributions, y, c=colors, s=28)
    contribution_ax.set_yticks(y, names)
    contribution_ax.set_xlabel(r"Mean $a_i^2$")
    contribution_ax.set_title("Per-muscle contribution", loc="left")
    contribution_ax.grid(axis="x", alpha=0.22)
    figure_value = _value(summary, "human_effort", "activation_squared_mean")
    fig.suptitle(f"Human Effort Proxy — mean Σa² = {_format(figure_value)}; not metabolic cost", fontsize=14)
    _save(fig, path)


def _plot_exo_torque(summary: dict[str, Any], path: Path) -> None:
    profiles = summary["profiles"]["exoskeleton"]
    time = np.asarray(profiles["time_s"])
    channels = ("HipExo_R", "HipExo_L", "KneeExo_R", "KneeExo_L")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True, sharey=True)
    for ax, channel in zip(axes.flat, channels):
        entry = profiles["channels"][channel]
        command = entry.get("command_torque_Nm")
        actual = entry.get("actual_torque_Nm")
        if command is None and actual is None:
            _unavailable(ax, channel, "command and actual torque are unavailable")
            continue
        if command is not None:
            ax.plot(time, command, color=COLORS["reference"], linestyle="--", label="Command")
        if actual is not None:
            ax.plot(time, actual, color=COLORS["accent"], label="Actual")
        combined = np.asarray((command or []) + (actual or []), dtype=float)
        finite = combined[np.isfinite(combined)]
        if len(finite) and np.max(np.abs(finite)) <= 1e-12:
            ax.set_ylim(-1.0, 1.0)
            ax.text(0.5, 0.78, "0 N·m throughout", transform=ax.transAxes, ha="center", color="#166534")
        ax.axhline(0.0, color="#9CA3AF", linewidth=0.8)
        ax.set_title(channel, loc="left")
        ax.set_ylabel("Torque (N·m)")
        ax.grid(alpha=0.22)
        ax.legend()
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)")
    fig.suptitle("Exoskeleton Command and Actual Torque", fontsize=15)
    _save(fig, path)


def _plot_exo_power(summary: dict[str, Any], path: Path) -> None:
    profiles = summary["profiles"]["exoskeleton"]
    time = np.asarray(profiles["time_s"])
    channels = ("HipExo_R", "HipExo_L", "KneeExo_R", "KneeExo_L")
    fig, (ax, table_ax) = plt.subplots(2, 1, figsize=(12, 8.5), gridspec_kw={"height_ratios": [2.1, 1]})
    rows: list[list[str]] = []
    all_zero = True
    for channel in channels:
        power = profiles["channels"][channel].get("power_W")
        if power is None:
            continue
        values = np.asarray(power, dtype=float)
        ax.plot(time, values, label=channel)
        finite = values[np.isfinite(values)]
        all_zero &= bool(len(finite) and np.max(np.abs(finite)) <= 1e-12)
        rows.append(
            [
                channel,
                _format(_value(summary, "exoskeleton", channel + ".positive_work_J")),
                _format(_value(summary, "exoskeleton", channel + ".negative_work_J")),
                _format(_value(summary, "exoskeleton", channel + ".absolute_work_J")),
                _format(_value(summary, "exoskeleton", channel + ".net_work_J")),
            ]
        )
    if not rows:
        _unavailable(ax, "Exoskeleton mechanical power", "torque and joint velocity are unavailable")
        _unavailable(table_ax, "Mechanical work", "power is unavailable")
        _save(fig, path)
        return
    if all_zero:
        ax.set_ylim(-1.0, 1.0)
        ax.text(0.5, 0.82, "0 W motor mechanical power throughout", transform=ax.transAxes, ha="center", color="#166534")
    ax.axhline(0.0, color="#9CA3AF", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mechanical power (W)")
    ax.set_title(r"Motor mechanical power $P=\tau_{actual}\dot{q}$", loc="left")
    ax.grid(alpha=0.22)
    ax.legend(ncol=2)
    table_ax.axis("off")
    table = table_ax.table(cellText=rows, colLabels=["Joint", "W+ (J)", "W− (J)", "|W| (J)", "Wnet (J)"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)
    fig.suptitle("Exoskeleton Mechanical Output", fontsize=15)
    _save(fig, path)


def _plot_interaction(summary: dict[str, Any], path: Path, *, load: bool) -> None:
    profiles = summary["profiles"]["interaction"]
    time = np.asarray(profiles["time_s"])
    vectors = ("force", "moment") if load else ("translation", "rotation")
    units = ("N", "N·m") if load else ("mm", "deg")
    scales = (1.0, 1.0) if load else (1000.0, 180.0 / np.pi)
    titles = ("Interface force norm", "Interface moment norm") if load else ("Relative displacement norm", "Relative rotation norm")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, vector, unit, scale, title in zip(axes, vectors, units, scales, titles):
        plotted = False
        for interface in profiles["active_interfaces"]:
            item = profiles["interfaces"].get(interface, {}).get(vector)
            if not item:
                continue
            values = np.asarray(item["norm"], dtype=float) * scale
            ax.plot(time, values, color=INTERFACE_COLORS[interface], label=interface)
            plotted = True
        if not plotted:
            _unavailable(ax, title, f"{vector} endpoint profile is unavailable")
            continue
        ax.set_ylabel(unit)
        ax.set_title(title, loc="left")
        ax.grid(alpha=0.22)
        ax.legend(ncol=3)
    axes[-1].set_xlabel("Time (s)")
    source = profiles.get("statistics_source", "recorded interaction diagnostics")
    heading = "Human–Exoskeleton Interface Load" if load else "Human–Exoskeleton Relative Motion"
    fig.suptitle(f"{heading}\nSummary peak/RMS/p95 source: {source}", fontsize=14)
    _save(fig, path)


def _exo_only_array(table: dict[str, np.ndarray], name: str) -> np.ndarray:
    return np.asarray(table[name], dtype=float)


def _plot_exo_only_joint_state(table: dict[str, np.ndarray], path: Path) -> None:
    time = _exo_only_array(table, "time_s")
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for joint in ("HipExo_R", "HipExo_L", "KneeExo_R", "KneeExo_L"):
        axes[0].plot(time, _exo_only_array(table, f"joint_position_{joint}_rad"), label=joint)
        axes[1].plot(time, _exo_only_array(table, f"joint_velocity_{joint}_rad_s"), label=joint)
    axes[0].set_ylabel("Position (rad)")
    axes[1].set_ylabel("Velocity (rad/s)")
    axes[1].set_xlabel("Time (s)")
    for ax in axes:
        ax.grid(alpha=0.22)
        ax.legend(ncol=2)
    fig.suptitle("Standalone Exoskeleton Joint State", fontsize=15)
    _save(fig, path)


def _plot_exo_only_torque(table: dict[str, np.ndarray], path: Path) -> None:
    time = _exo_only_array(table, "time_s")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True, sharey=True)
    for ax, joint in zip(axes.flat, ("HipExo_R", "HipExo_L", "KneeExo_R", "KneeExo_L")):
        command = _exo_only_array(table, f"command_torque_{joint}_Nm")
        actual = _exo_only_array(table, f"actual_torque_{joint}_Nm")
        ax.plot(time, command, "--", color=COLORS["reference"], label="Command")
        ax.plot(time, actual, color=COLORS["accent"], label="Actual")
        if np.max(np.abs(np.concatenate((command, actual)))) <= 1e-12:
            ax.set_ylim(-1.0, 1.0)
            ax.text(0.5, 0.78, "0 N·m throughout", transform=ax.transAxes, ha="center", color="#166534")
        ax.set_title(joint, loc="left")
        ax.set_ylabel("Torque (N·m)")
        ax.grid(alpha=0.22)
        ax.legend()
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)")
    fig.suptitle("Standalone Exoskeleton Torque", fontsize=15)
    _save(fig, path)


def _plot_exo_only_power(table: dict[str, np.ndarray], path: Path) -> None:
    time = _exo_only_array(table, "time_s")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    maximum = 0.0
    for joint in ("HipExo_R", "HipExo_L", "KneeExo_R", "KneeExo_L"):
        power = _exo_only_array(table, f"power_{joint}_W")
        maximum = max(maximum, float(np.max(np.abs(power))))
        ax.plot(time, power, label=joint)
    if maximum <= 1e-12:
        ax.set_ylim(-1.0, 1.0)
        ax.text(0.5, 0.82, "0 W motor mechanical power throughout", transform=ax.transAxes, ha="center", color="#166534")
    ax.axhline(0.0, color="#9CA3AF", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mechanical power (W)")
    ax.set_title("Standalone Exoskeleton Motor Power")
    ax.grid(alpha=0.22)
    ax.legend(ncol=2)
    _save(fig, path)


def _plot_exo_only_limits(table: dict[str, np.ndarray], path: Path) -> None:
    time = _exo_only_array(table, "time_s")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for joint in ("HipExo_R", "HipExo_L", "KneeExo_R", "KneeExo_L"):
        ax.plot(time, _exo_only_array(table, f"joint_limit_margin_{joint}_rad"), label=joint)
    ax.axhline(0.0, color="#D62728", linestyle="--", label="Limit boundary")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Minimum joint-limit margin (rad)")
    ax.set_title("Standalone Exoskeleton Joint-limit Validation")
    ax.grid(alpha=0.22)
    ax.legend(ncol=2)
    _save(fig, path)


def _generate(
    summary: dict[str, Any],
    table: dict[str, np.ndarray],
    evaluation_dir: Path,
    jobs: list[tuple[str, Callable[[Path], None]]],
) -> dict[str, dict[str, Any]]:
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = evaluation_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for filename in MANAGED_FIGURE_NAMES:
        managed_path = figures_dir / filename
        if managed_path.is_file():
            managed_path.unlink()
    results: dict[str, dict[str, Any]] = {}
    overview_path = evaluation_dir / "00_overview.png"
    try:
        _plot_overview(summary, overview_path)
        results["00_overview.png"] = {"generated": True, "path": str(overview_path.resolve())}
    except Exception as exc:
        results["00_overview.png"] = {"generated": False, "path": None, "reason": str(exc)}
    for filename, function in jobs:
        destination = figures_dir / filename
        try:
            function(destination)
            results[filename] = {"generated": True, "path": str(destination.resolve())}
        except Exception as exc:
            results[filename] = {"generated": False, "path": None, "reason": str(exc)}
    return results


def generate_result_figures(
    summary: dict[str, Any], table: dict[str, np.ndarray], evaluation_dir: Path
) -> dict[str, dict[str, Any]]:
    """Generate only figures supported by the routed experiment type."""
    if summary.get("protocol") == "exo_only_test_bench_v1":
        jobs = [
            ("01_exo_joint_state.png", lambda p: _plot_exo_only_joint_state(table, p)),
            ("02_exo_torque.png", lambda p: _plot_exo_only_torque(table, p)),
            ("03_exo_power.png", lambda p: _plot_exo_only_power(table, p)),
            ("04_exo_joint_limits.png", lambda p: _plot_exo_only_limits(table, p)),
        ]
        return _generate(summary, table, evaluation_dir, jobs)

    jobs: list[tuple[str, Callable[[Path], None]]] = []
    if "stability" in summary.get("metrics", {}):
        jobs.append(("01_stability.png", lambda p: _plot_stability(summary, table, p)))
    if "gait" in summary.get("metrics", {}):
        jobs.extend(
            [
                ("02_gait_kinematics.png", lambda p: _plot_joint_cycles(summary, p)),
                ("03_gait_spatiotemporal.png", lambda p: _plot_spatiotemporal(summary, p)),
            ]
        )
    if "grf" in summary.get("metrics", {}):
        jobs.append(("04_grf_touch_force_bw.png", lambda p: _plot_grf(summary, p)))
    if "muscle" in summary.get("metrics", {}):
        jobs.extend(
            [
                ("05_muscle_activation_heatmap.png", lambda p: _plot_muscle_heatmap(summary, p, "activation")),
                ("06_muscle_activation_key_profiles.png", lambda p: _plot_key_muscles(summary, p, "activation")),
                ("07_muscle_force_heatmap.png", lambda p: _plot_muscle_heatmap(summary, p, "force")),
                ("08_muscle_force_key_profiles.png", lambda p: _plot_key_muscles(summary, p, "force")),
            ]
        )
    if "human_effort" in summary.get("metrics", {}):
        jobs.append(("09_human_effort_proxy.png", lambda p: _plot_human_effort(summary, table, p)))
    if "exoskeleton" in summary.get("metrics", {}):
        jobs.extend(
            [
                ("10_exo_torque.png", lambda p: _plot_exo_torque(summary, p)),
                ("11_exo_power_work.png", lambda p: _plot_exo_power(summary, p)),
            ]
        )
    if "interaction" in summary.get("metrics", {}):
        jobs.extend(
            [
                ("12_interface_relative_motion.png", lambda p: _plot_interaction(summary, p, load=False)),
                ("13_interface_load.png", lambda p: _plot_interaction(summary, p, load=True)),
            ]
        )
    return _generate(summary, table, evaluation_dir, jobs)


def generate_all_figures(
    summary: dict[str, Any], table: dict[str, np.ndarray], figures_dir: Path
) -> dict[str, dict[str, Any]]:
    """Backward-compatible wrapper for older callers."""
    evaluation_dir = figures_dir.parent if figures_dir.name == "figures" else figures_dir
    return generate_result_figures(summary, table, evaluation_dir)


def generate_human_figures(
    summary: dict[str, Any], table: dict[str, np.ndarray], figures_dir: Path
) -> dict[str, dict[str, Any]]:
    """Backward-compatible wrapper for the original L1 protocol."""
    return generate_all_figures(summary, table, figures_dir)
