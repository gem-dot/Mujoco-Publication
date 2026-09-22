"""One stable CSV schema for all unified MuJoCo experiments."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any


TIMESERIES_SCHEMA_VERSION = "3.1"
EXO_CHANNELS = (
    "HipExo_R",
    "HipExo_L",
    "KneeExo_R",
    "KneeExo_L",
    "Exo_R",
    "Exo_L",
)
JOINT_KEYS = ("hip_r", "hip_l", "knee_r", "knee_l", "ankle_r", "ankle_l")
INTERFACE_NAMES = ("waist", "thigh_l", "thigh_r", "shank_l", "shank_r")
INTERFACE_VECTORS = ("translation", "rotation", "force", "moment")
VECTOR_AXES = ("x", "y", "z")


class ExperimentRecorder:
    """Collect rows in memory and write one deterministic CSV schema."""

    def __init__(self, muscle_names: tuple[str, ...]) -> None:
        self.muscle_names = muscle_names
        self.rows: list[dict[str, Any]] = []
        self.fieldnames = self._fieldnames()

    def _fieldnames(self) -> list[str]:
        fields = [
            "step",
            "time",
            "forward_position",
            "forward_velocity",
            "pelvis_height",
            "pelvis_pitch",
            "is_done",
            "termination_reason",
        ]
        for joint in JOINT_KEYS:
            side = "right" if joint.endswith("_r") else "left"
            name = joint.rsplit("_", 1)[0]
            fields.extend(
                [f"{side}_{name}_angle_rad", f"{side}_{name}_velocity_rad_s"]
            )
        fields.extend(
            [
                "left_heel_contact",
                "left_toe_contact",
                "right_heel_contact",
                "right_toe_contact",
                "left_stance",
                "right_stance",
                "left_heel_force_n",
                "left_toe_force_n",
                "right_heel_force_n",
                "right_toe_force_n",
                "heel_strike_side",
                "right_step_length_m",
                "left_step_length_m",
                "right_cycle_index",
                "left_cycle_index",
            ]
        )
        for muscle in self.muscle_names:
            fields.extend(
                [
                    f"muscle_ctrl_{muscle}",
                    f"muscle_activation_{muscle}",
                    f"muscle_force_{muscle}",
                ]
            )
        for channel in EXO_CHANNELS:
            fields.extend(
                [
                    f"exo_command_torque_{channel}_nm",
                    f"exo_actual_torque_{channel}_nm",
                    f"exo_joint_velocity_{channel}_rad_s",
                ]
            )
        fields.append("coupling_physics_substeps")
        for interface in INTERFACE_NAMES:
            for vector in INTERFACE_VECTORS:
                fields.extend(
                    f"coupling_{interface}_{vector}_{axis}"
                    for axis in VECTOR_AXES
                )
                norm = f"coupling_{interface}_{vector}_norm"
                fields.extend(
                    (norm, f"{norm}_peak", f"{norm}_rms", f"{norm}_p95")
                )
        return fields

    def record(
        self,
        *,
        step: int,
        observation: dict[str, Any],
        exo_commands: dict[str, float],
        exo_actual: dict[str, float],
        exo_joint_velocity: dict[str, float],
        heel_strike_side: str,
        right_step_length_m: float | None,
        left_step_length_m: float | None,
        cycle_index: dict[str, int],
        is_done: bool,
        termination_reason: str,
        interaction_diagnostics: dict[str, Any] | None = None,
    ) -> None:
        pelvis = observation["pelvis"]
        joints = observation["joints"]
        contact = observation["contact"]
        touch = observation["touch_force_n"]
        muscles = observation["muscles"]
        row: dict[str, Any] = {
            "step": step,
            "time": observation["time"],
            "forward_position": pelvis["forward_position"],
            "forward_velocity": pelvis["forward_velocity"],
            "pelvis_height": pelvis["height"],
            "pelvis_pitch": pelvis["pitch"],
            "is_done": bool(is_done),
            "termination_reason": termination_reason,
        }
        for joint in JOINT_KEYS:
            side = "right" if joint.endswith("_r") else "left"
            name = joint.rsplit("_", 1)[0]
            row[f"{side}_{name}_angle_rad"] = joints[joint]["angle"]
            row[f"{side}_{name}_velocity_rad_s"] = joints[joint]["velocity"]
        row.update(
            {
                "left_heel_contact": contact["left_heel"],
                "left_toe_contact": contact["left_toe"],
                "right_heel_contact": contact["right_heel"],
                "right_toe_contact": contact["right_toe"],
                "left_stance": contact["left_stance"],
                "right_stance": contact["right_stance"],
                "left_heel_force_n": touch["left_heel"],
                "left_toe_force_n": touch["left_toe"],
                "right_heel_force_n": touch["right_heel"],
                "right_toe_force_n": touch["right_toe"],
                "heel_strike_side": heel_strike_side,
                "right_step_length_m": (
                    math.nan if right_step_length_m is None else right_step_length_m
                ),
                "left_step_length_m": (
                    math.nan if left_step_length_m is None else left_step_length_m
                ),
                "right_cycle_index": cycle_index["right"],
                "left_cycle_index": cycle_index["left"],
            }
        )
        for muscle in self.muscle_names:
            row[f"muscle_ctrl_{muscle}"] = muscles["ctrl"][muscle]
            row[f"muscle_activation_{muscle}"] = muscles["activation"][muscle]
            row[f"muscle_force_{muscle}"] = muscles["force"][muscle]
        for channel in EXO_CHANNELS:
            row[f"exo_command_torque_{channel}_nm"] = exo_commands[channel]
            row[f"exo_actual_torque_{channel}_nm"] = exo_actual[channel]
            row[f"exo_joint_velocity_{channel}_rad_s"] = exo_joint_velocity[channel]
        row["coupling_physics_substeps"] = (
            math.nan
            if interaction_diagnostics is None
            else interaction_diagnostics["physics_substeps"]
        )
        for interface in INTERFACE_NAMES:
            values = (
                None
                if interaction_diagnostics is None
                else interaction_diagnostics["interfaces"].get(interface)
            )
            for vector in INTERFACE_VECTORS:
                components = None if values is None else values[vector]
                for index, axis in enumerate(VECTOR_AXES):
                    row[f"coupling_{interface}_{vector}_{axis}"] = (
                        math.nan if components is None else components[index]
                    )
                norm = f"coupling_{interface}_{vector}_norm"
                source_norm = norm.removeprefix(f"coupling_{interface}_")
                row[norm] = math.nan if values is None else values[source_norm]
                for statistic in ("peak", "rms", "p95"):
                    key = f"{norm}_{statistic}"
                    source = f"{vector}_norm_{statistic}"
                    row[key] = math.nan if values is None else values[source]
        self.rows.append(row)

    def write_csv(self, path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
