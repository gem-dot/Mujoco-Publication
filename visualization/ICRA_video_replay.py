"""State-history capture and deterministic MuJoCo replay for ICRA videos.

This module is deliberately outside the controller and runner.  It records the
exact state seen by the existing runner, then restores those states without
advancing physics during video rendering.
"""

from __future__ import annotations

import csv
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import mujoco
import numpy as np

from simulation import model_setup, runner
from simulation.recorder import ExperimentRecorder


class StateHistoryRecorder(ExperimentRecorder):
    """Existing CSV recorder plus lossless 1 kHz MuJoCo state snapshots."""

    def __init__(self, muscle_names: tuple[str, ...], prepared: Any) -> None:
        super().__init__(muscle_names)
        self.prepared = prepared
        self.qpos: list[np.ndarray] = []
        self.qvel: list[np.ndarray] = []
        self.act: list[np.ndarray] = []
        self.ctrl: list[np.ndarray] = []

    def record(self, **kwargs: Any) -> None:
        super().record(**kwargs)
        data = self.prepared.data
        self.qpos.append(np.asarray(data.qpos, dtype=np.float64).copy())
        self.qvel.append(np.asarray(data.qvel, dtype=np.float64).copy())
        self.act.append(np.asarray(data.act, dtype=np.float64).copy())
        self.ctrl.append(np.asarray(data.ctrl, dtype=np.float64).copy())


@contextmanager
def capture_runner_state() -> Iterator[dict[str, Any]]:
    """Instrument the unmodified runner for one call to ``run_experiment``."""

    original_prepare = runner.model_setup.prepare_model
    original_recorder = runner.ExperimentRecorder
    holder: dict[str, Any] = {}

    def prepare(config: Any) -> Any:
        prepared = original_prepare(config)
        holder["prepared"] = prepared
        return prepared

    def recorder_factory(muscle_names: tuple[str, ...]) -> StateHistoryRecorder:
        recorder = StateHistoryRecorder(muscle_names, holder["prepared"])
        holder["recorder"] = recorder
        return recorder

    runner.model_setup.prepare_model = prepare
    runner.ExperimentRecorder = recorder_factory
    try:
        yield holder
    finally:
        runner.model_setup.prepare_model = original_prepare
        runner.ExperimentRecorder = original_recorder


def read_csv_columns(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    result: dict[str, np.ndarray] = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        parsed: list[float] = []
        numeric_column = True
        has_number = False
        for value in values:
            if value.strip() == "":
                parsed.append(np.nan)
                continue
            try:
                parsed.append(float(value))
                has_number = True
            except ValueError:
                numeric_column = False
                break
        if numeric_column and has_number:
            result[key] = np.asarray(parsed, dtype=np.float64)
        else:
            result[key] = np.asarray(values, dtype=str)
    return result


def aligned_diagnostics(path: Path, samples: int) -> dict[str, np.ndarray]:
    """Align step diagnostics to state rows; row zero is the pre-control state."""

    if not path.is_file():
        return {}
    raw = read_csv_columns(path)
    result: dict[str, np.ndarray] = {}
    for key, values in raw.items():
        if len(values) == samples:
            result[key] = values
        elif len(values) == samples - 1:
            if np.issubdtype(values.dtype, np.number):
                result[key] = np.concatenate((values.astype(np.float64), [np.nan]))
            else:
                result[key] = np.concatenate((values, np.asarray([""], dtype=str)))
        else:
            raise ValueError(
                f"Diagnostic/state length mismatch for {key}: {len(values)} vs {samples}"
            )
    return result


def numeric(columns: dict[str, np.ndarray], key: str, samples: int) -> np.ndarray:
    value = columns.get(key)
    if value is None:
        return np.full(samples, np.nan, dtype=np.float64)
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return np.full(samples, np.nan, dtype=np.float64)


def boolean(columns: dict[str, np.ndarray], key: str, samples: int) -> np.ndarray:
    value = columns.get(key)
    if value is None:
        return np.zeros(samples, dtype=bool)
    if np.issubdtype(value.dtype, np.number):
        return np.nan_to_num(value.astype(float), nan=0.0) != 0.0
    return np.asarray([str(v).strip().lower() in ("1", "true", "yes") for v in value])


def save_trajectory(
    output: Path,
    recorder: StateHistoryRecorder,
    result_directory: Path,
    *,
    controller_mode: str,
) -> dict[str, Any]:
    """Merge runner, controller and full-state data into one reusable NPZ."""

    ts = read_csv_columns(result_directory / "timeseries.csv")
    samples = len(ts["time"])
    diag = aligned_diagnostics(result_directory / "controller_diagnostics.csv", samples)
    if len(recorder.qpos) != samples:
        raise ValueError(f"Captured state count mismatch: {len(recorder.qpos)} vs {samples}")

    muscle_keys = sorted(k for k in ts if k.startswith("muscle_activation_"))
    muscle_names = [k.removeprefix("muscle_activation_") for k in muscle_keys]
    muscle_activation = np.column_stack([numeric(ts, k, samples) for k in muscle_keys])
    activation_rms = np.sqrt(np.nanmean(np.square(muscle_activation), axis=1))
    scheduler_s = numeric(diag, "scheduler_s", samples)
    scheduler_s = np.where(np.isfinite(scheduler_s), scheduler_s, activation_rms)

    eligible_count = numeric(diag, "stage7t_eligible_tick_count", samples)
    accepted_count = numeric(diag, "stage7t_event_count", samples)
    for values in (eligible_count, accepted_count):
        for index in range(1, len(values)):
            if not np.isfinite(values[index]):
                values[index] = values[index - 1]
    eligible_count = np.nan_to_num(eligible_count, nan=0.0).astype(np.int64)
    accepted_count = np.nan_to_num(accepted_count, nan=0.0).astype(np.int64)
    eligible = np.diff(eligible_count, prepend=eligible_count[0]) > 0
    accepted = boolean(diag, "stage7t_event_accepted", samples)

    gait_phase = numeric(diag, "phase", samples)
    if not np.isfinite(gait_phase).any():
        gait_phase = np.full(samples, np.nan, dtype=np.float64)
        strike_text = np.asarray(ts.get("heel_strike_side", np.full(samples, "")), dtype=str)
        strike_indices = np.flatnonzero(np.asarray([v in ("left", "both") for v in strike_text]))
        times_for_phase = numeric(ts, "time", samples)
        for start, stop in zip(strike_indices[:-1], strike_indices[1:]):
            period = times_for_phase[stop] - times_for_phase[start]
            if period > 0:
                gait_phase[start:stop] = (times_for_phase[start:stop] - times_for_phase[start]) / period

    threshold = numeric(diag, "stage7t_threshold", samples)
    error_norm_sq = numeric(diag, "stage7t_e_bar_res_norm_sq", samples)
    ratio = np.divide(
        error_norm_sq,
        threshold,
        out=np.full(samples, np.nan),
        where=np.isfinite(threshold) & (threshold > 0),
    )

    exo_hip = np.full(samples, np.nan, dtype=np.float64)
    exo_knee = np.full(samples, np.nan, dtype=np.float64)
    prepared = getattr(recorder, "prepared", None)
    if prepared is not None:
        model = prepared.model.ptr
        hip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "exo_hip_l")
        knee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "exo_knee_l")
        if hip_id >= 0 and knee_id >= 0:
            qpos = np.stack(recorder.qpos)
            exo_hip = qpos[:, int(model.jnt_qposadr[hip_id])]
            exo_knee = qpos[:, int(model.jnt_qposadr[knee_id])]

    payload: dict[str, Any] = {
        "time": numeric(ts, "time", samples),
        "qpos": np.stack(recorder.qpos),
        "qvel": np.stack(recorder.qvel),
        "act": np.stack(recorder.act),
        "ctrl": np.stack(recorder.ctrl),
        "pelvis_x": numeric(ts, "forward_position", samples),
        "pelvis_height": numeric(ts, "pelvis_height", samples),
        "pelvis_pitch_deg": numeric(ts, "pelvis_pitch", samples),
        "q_hip_l": numeric(ts, "left_hip_angle_rad", samples),
        "q_knee_l": numeric(ts, "left_knee_angle_rad", samples),
        "q_exo_hip_l": exo_hip,
        "q_exo_knee_l": exo_knee,
        "qd_hip_l": numeric(diag, "q_ref_hip", samples),
        "qd_knee_l": numeric(diag, "q_ref_knee", samples),
        "muscle_names": np.asarray(muscle_names, dtype=str),
        "muscle_activation": muscle_activation,
        "activation_rms": activation_rms,
        "s": scheduler_s,
        "alpha": numeric(diag, "scheduler_alpha", samples),
        "gamma": numeric(diag, "stage6g_gamma", samples),
        "mu_hip": numeric(ts, "exo_actual_torque_HipExo_L_nm", samples),
        "mu_knee": numeric(ts, "exo_actual_torque_KneeExo_L_nm", samples),
        "mu0_hip": numeric(diag, "stage6g_mu0_hip", samples),
        "mu0_knee": numeric(diag, "stage6g_mu0_knee", samples),
        "U_aff_current": np.column_stack([
            numeric(diag, "stage7t_Uaffine_current_muA_hip", samples),
            numeric(diag, "stage7t_Uaffine_current_muA_knee", samples),
            numeric(diag, "stage7t_Uaffine_current_muI_hip", samples),
            numeric(diag, "stage7t_Uaffine_current_muI_knee", samples),
        ]),
        "U_critic_current": np.column_stack([
            numeric(diag, "stage7t_Ucritic_current_muA_hip", samples),
            numeric(diag, "stage7t_Ucritic_current_muA_knee", samples),
            numeric(diag, "stage7t_Ucritic_current_muI_hip", samples),
            numeric(diag, "stage7t_Ucritic_current_muI_knee", samples),
        ]),
        "U_h": np.column_stack([
            numeric(diag, "stage7t_Ucritic_held_muA_hip", samples),
            numeric(diag, "stage7t_Ucritic_held_muA_knee", samples),
            numeric(diag, "stage7t_Ucritic_held_muI_hip", samples),
            numeric(diag, "stage7t_Ucritic_held_muI_knee", samples),
        ]),
        "U_exec": np.column_stack([
            numeric(diag, "stage7t_Uexec_muA_hip", samples),
            numeric(diag, "stage7t_Uexec_muA_knee", samples),
            numeric(diag, "stage7t_Uexec_muI_hip", samples),
            numeric(diag, "stage7t_Uexec_muI_knee", samples),
        ]),
        "event_error_norm": np.sqrt(np.maximum(error_norm_sq, 0.0)),
        "event_error_norm_sq": error_norm_sq,
        "event_threshold": threshold,
        "event_ratio": ratio,
        "event_eligible": eligible,
        "event_accepted": accepted,
        "accepted_count": accepted_count,
        "eligible_count": eligible_count,
        "heel_strike": np.asarray(ts.get("heel_strike_side", np.full(samples, "")), dtype=str),
        "gait_cycle_phase": gait_phase,
        "termination_flag": boolean(ts, "is_done", samples),
        "termination_reason": np.asarray(ts.get("termination_reason", np.full(samples, "")), dtype=str),
        "actuator_saturation_flag": boolean(diag, "stage6b_native_saturation", samples)
        | boolean(diag, "native_saturation_command", samples),
        "controller_mode": np.asarray([controller_mode] * samples, dtype=str),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    return {
        "samples": samples,
        "time_start_s": float(payload["time"][0]),
        "time_end_s": float(payload["time"][-1]),
        "qpos_shape": list(payload["qpos"].shape),
        "qvel_shape": list(payload["qvel"].shape),
        "muscle_count": len(muscle_names),
        "keys": sorted(payload),
    }


class DeterministicReplay:
    """Restore logged states; no ``mj_step`` is performed."""

    def __init__(self, config: Any, trajectory: Path) -> None:
        self.prepared = model_setup.prepare_model(config)
        self.model = self.prepared.model.ptr
        self.data = self.prepared.data.ptr
        self.log = np.load(trajectory, allow_pickle=False)
        if self.log["qpos"].shape[1] != self.model.nq:
            raise ValueError("Replay qpos/model mismatch")
        if self.log["qvel"].shape[1] != self.model.nv:
            raise ValueError("Replay qvel/model mismatch")

    def nearest_index(self, time_s: float) -> int:
        times = self.log["time"]
        pos = int(np.searchsorted(times, time_s))
        if pos <= 0:
            return 0
        if pos >= len(times):
            return len(times) - 1
        return pos if abs(times[pos] - time_s) < abs(times[pos - 1] - time_s) else pos - 1

    def restore(self, index: int) -> None:
        self.data.qpos[:] = self.log["qpos"][index]
        self.data.qvel[:] = self.log["qvel"][index]
        if self.data.act.size == self.log["act"].shape[1]:
            self.data.act[:] = self.log["act"][index]
        if self.data.ctrl.size == self.log["ctrl"].shape[1]:
            self.data.ctrl[:] = self.log["ctrl"][index]
        self.data.time = float(self.log["time"][index])
        mujoco.mj_forward(self.model, self.data)
