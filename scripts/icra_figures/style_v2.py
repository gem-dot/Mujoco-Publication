"""Central visual language for ICRA Core Physical V2 figures."""

from __future__ import annotations

import matplotlib.pyplot as plt


COLORS = {
    "healthy": "#20364D",
    "exozero": "#777777",
    "impedance": "#3A6EA5",
    "admittance": "#D47A35",
    "dynamic": "#16827C",
    "event": "#B84A2B",
    "alpha": "#6A51A3",
    "gamma": "#E38B2C",
    "mu0": "#444444",
    "target": "#8CC3BA",
    "loading": "#EAF2F8",
    "midswing": "#FBF1E6",
    "grid": "#D7DCE1",
    "text": "#222222",
}

METHODS = ("Pure Impedance", "Pure Admittance", "Dynamic Fusion")
METHOD_COLORS = {
    "Pure Impedance": COLORS["impedance"],
    "Pure Admittance": COLORS["admittance"],
    "Dynamic Fusion": COLORS["dynamic"],
}
METHOD_LINEWIDTHS = {
    "Pure Impedance": 1.15,
    "Pure Admittance": 1.15,
    "Dynamic Fusion": 1.85,
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.3,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.3,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.25,
            "grid.linewidth": 0.42,
            "grid.alpha": 0.34,
            "grid.color": COLORS["grid"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(-0.12, 1.045, f"({label})", transform=ax.transAxes, fontweight="bold", va="bottom")
    ax.set_title(title, pad=5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.65)


def phase_bands(ax: plt.Axes, labels: bool = False) -> None:
    ax.axvspan(0, 20, color=COLORS["loading"], alpha=0.70, linewidth=0, zorder=0)
    ax.axvspan(65, 85, color=COLORS["midswing"], alpha=0.70, linewidth=0, zorder=0)
    if labels:
        low, high = ax.get_ylim()
        y = high - 0.055 * (high - low)
        ax.text(10, y, "Loading", ha="center", va="top", color="#4D6E8A", fontsize=6.6)
        ax.text(75, y, "Midswing", ha="center", va="top", color="#96633D", fontsize=6.6)
METHOD_COLORS = {
    "Pure Impedance": COLORS["impedance"],
    "Pure Admittance": COLORS["admittance"],
    "Dynamic Fusion": COLORS["dynamic"],
}
METHOD_LINEWIDTHS = {
    "Pure Impedance": 1.15,
    "Pure Admittance": 1.15,
    "Dynamic Fusion": 1.85,
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.1,
            "axes.linewidth": 0.72,
            "lines.linewidth": 1.3,
            "grid.linewidth": 0.42,
            "grid.alpha": 0.35,
            "grid.color": COLORS["grid"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": True,
        }
    )


def panel(ax, label: str, title: str) -> None:
    ax.text(
        -0.12,
        1.04,
        f"({label})",
        transform=ax.transAxes,
        fontweight="bold",
        va="bottom",
    )
    ax.set_title(title, pad=5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.65)


def phase_bands(ax, labels: bool = False) -> None:
    ax.axvspan(0, 20, color=COLORS["loading"], linewidth=0, zorder=0)
    ax.axvspan(65, 85, color=COLORS["midswing"], linewidth=0, zorder=0)
    if labels:
        ymin, ymax = ax.get_ylim()
        y = ymax - 0.055 * (ymax - ymin)
        ax.text(10, y, "Loading", color="#4D6E8A", ha="center", va="top", fontsize=6.8)
        ax.text(75, y, "Midswing", color="#96633D", ha="center", va="top", fontsize=6.8)
