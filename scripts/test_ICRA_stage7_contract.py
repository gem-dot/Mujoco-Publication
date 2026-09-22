#!/usr/bin/env python3
"""Deterministic unit contract for Stage-7 policy-execution holding."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.ICRA import (
    CONSERVATIVE_EVENT_TRIGGERED_INTENTION_TARGET_GAME_ACTIVE_MODE,
    DWELL_LIMITED_P25_EVENT_TRIGGERED_INTENTION_TARGET_GAME_ACTIVE_MODE,
    EVENT_TRIGGERED_INTENTION_TARGET_GAME_ACTIVE_MODE,
    ICRAController,
    STAGE7R_C_CHI_EVT,
    STAGE7R_KAPPA_E,
    STAGE7R_THETA0,
    STAGE7S_C_CHI_EVT,
    STAGE7S_KAPPA_E,
    STAGE7S_THETA0,
)


def _controller_without_plant(
    mode: str = EVENT_TRIGGERED_INTENTION_TARGET_GAME_ACTIVE_MODE,
) -> ICRAController:
    controller = object.__new__(ICRAController)
    controller.mode = mode
    controller._stage7_held_strategy = None
    controller._stage7_vartheta = 0.0
    controller._stage7_event_age_s = 0.0
    controller._stage7_event_count = 0
    controller._stage7_eligible_ticks = 0
    controller._stage7_critic_updates = 0
    controller._stage7_last_event_time_s = None
    controller.termination_request = ""
    return controller


def _game(strategy: np.ndarray) -> dict[str, float | bool]:
    return {
        "stage6g_learning_active": True,
        "stage6g_muA_hip": float(strategy[0]),
        "stage6g_muA_knee": float(strategy[1]),
        "stage6g_muI_hip": float(strategy[2]),
        "stage6g_muI_knee": float(strategy[3]),
    }


def main() -> None:
    controller = _controller_without_plant()
    performance = {f"chi_{index:02d}": 0.0 for index in range(12)}
    first = np.asarray((1.0, -0.2, 0.3, -0.1))
    command, first_diag = controller._stage7_event_step(
        performance=performance,
        stage6g=_game(first),
        mu0=np.asarray((2.0, 3.0)),
        continuous_command=np.asarray((2.3, 2.9)),
        time_s=1.0,
        dt=0.01,
    )
    assert first_diag["stage7_event_accepted"]
    assert first_diag["stage7_event_index"] == 0
    assert np.allclose(command, np.asarray((2.3, 2.9)), atol=1e-12, rtol=0.0)

    changed = np.asarray((-1.0, 0.8, -0.8, 0.7))
    command, held_diag = controller._stage7_event_step(
        performance=performance,
        stage6g=_game(changed),
        mu0=np.asarray((2.1, 3.2)),
        continuous_command=np.asarray((1.3, 3.9)),
        time_s=1.01,
        dt=0.01,
    )
    assert not held_diag["stage7_event_accepted"]
    assert held_diag["stage7_held_strategy_exact_between_events"]
    assert np.allclose(command, np.asarray((2.4, 3.1)), atol=1e-12, rtol=0.0)

    command, event_diag = controller._stage7_event_step(
        performance=performance,
        stage6g=_game(changed),
        mu0=np.asarray((2.2, 3.4)),
        continuous_command=np.asarray((1.4, 4.1)),
        time_s=1.02,
        dt=0.01,
    )
    assert event_diag["stage7_event_accepted"]
    assert event_diag["stage7_event_index"] == 1
    assert abs(float(event_diag["stage7_inter_event_interval_s"]) - 0.02) < 1e-12
    assert np.allclose(command, np.asarray((1.4, 4.1)), atol=1e-12, rtol=0.0)
    assert event_diag["stage7_e_bar_norm_sq"] == 0.0
    assert event_diag["stage7_critic_update_count"] == 3
    assert event_diag["stage7_eligible_tick_count"] == 3
    assert event_diag["stage7_physical_contract_error_max_abs"] == 0.0
    assert not event_diag["stage7_dwell_violation"]
    assert not event_diag["stage7_trigger_uses_phase"]
    conservative = _controller_without_plant(
        CONSERVATIVE_EVENT_TRIGGERED_INTENTION_TARGET_GAME_ACTIVE_MODE
    )
    parameters = conservative._event_execution_parameters()
    assert parameters["candidate"] == "p50_Td20ms"
    assert parameters["theta0"] == STAGE7R_THETA0
    assert parameters["kappa_e"] == STAGE7R_KAPPA_E
    assert parameters["c_chi_evt"] == STAGE7R_C_CHI_EVT
    dwell_limited = _controller_without_plant(
        DWELL_LIMITED_P25_EVENT_TRIGGERED_INTENTION_TARGET_GAME_ACTIVE_MODE
    )
    p25 = dwell_limited._event_execution_parameters()
    assert p25["candidate"] == "p25_Td20ms"
    assert p25["theta0"] == STAGE7S_THETA0
    assert p25["kappa_e"] == STAGE7S_KAPPA_E
    assert p25["c_chi_evt"] == STAGE7S_C_CHI_EVT
    print("ICRA_STAGE7_EVENT_CONTRACT_PASS")


if __name__ == "__main__":
    main()
