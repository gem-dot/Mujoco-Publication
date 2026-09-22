#!/usr/bin/env python3
"""Deterministic Stage-7T continuous-affine/residual-hold contract test."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.ICRA import (
    ICRAController,
    RESIDUAL_EVENT_TRIGGERED_INTENTION_TARGET_GAME_ACTIVE_MODE,
    STAGE7T_DWELL_S,
    STAGE7T_RESIDUAL_SCALE_NM,
)


def controller_without_plant() -> ICRAController:
    controller = object.__new__(ICRAController)
    controller.mode = RESIDUAL_EVENT_TRIGGERED_INTENTION_TARGET_GAME_ACTIVE_MODE
    controller._stage7t_held_critic = None
    controller._stage7t_vartheta = 0.0
    controller._stage7t_event_age_s = 0.0
    controller._stage7t_event_count = 0
    controller._stage7t_eligible_ticks = 0
    controller._stage7t_critic_updates = 0
    controller._stage7t_last_event_time_s = None
    controller.termination_request = ""
    return controller


def game(affine: np.ndarray, critic: np.ndarray) -> dict[str, float | bool]:
    total = affine + critic
    result: dict[str, float | bool] = {"stage6g_learning_active": True}
    for index, (player, joint) in enumerate(
        (("muA", "hip"), ("muA", "knee"), ("muI", "hip"), ("muI", "knee"))
    ):
        result[f"stage6g_{player}_{joint}"] = float(total[index])
        result[f"stage6g_{player}_affine_{joint}"] = float(affine[index])
        result[f"stage6g_{player}_critic_{joint}"] = float(critic[index])
    return result


def step(
    controller: ICRAController, affine: np.ndarray, critic: np.ndarray,
    mu0: np.ndarray, time_s: float,
) -> tuple[np.ndarray, dict[str, object]]:
    performance = {f"chi_{index:02d}": 0.0 for index in range(12)}
    return controller._stage7t_residual_event_step(
        performance=performance,
        stage6g=game(affine, critic),
        mu0=mu0,
        continuous_command=mu0 + affine[2:] + critic[2:],
        time_s=time_s,
        dt=0.01,
    )


def main() -> None:
    controller = controller_without_plant()
    affine0 = np.asarray((0.4, -0.1, -0.5, 0.2))
    critic0 = np.asarray((0.01, -0.001, 0.002, -0.001))
    mu0 = np.asarray((2.0, 3.0))
    command0, diag0 = step(controller, affine0, critic0, mu0, 1.0)
    assert diag0["stage7t_event_accepted"]
    assert np.allclose(command0, mu0 + affine0[2:] + critic0[2:])

    # Affine changes immediately while the residual remains exactly held.
    affine1 = affine0 + np.asarray((0.3, -0.2, 0.4, -0.3))
    command1, diag1 = step(controller, affine1, critic0, mu0 + 0.1, 1.01)
    assert not diag1["stage7t_event_accepted"]
    assert diag1["stage7t_held_critic_exact_between_events"]
    assert float(diag1["stage7t_e_bar_res_norm_sq"]) == 0.0
    assert np.allclose(command1, mu0 + 0.1 + affine1[2:] + critic0[2:])

    # Even after dwell, affine-only motion cannot trigger a residual event.
    affine2 = affine1 + np.asarray((-0.5, 0.4, -0.2, 0.25))
    command2, diag2 = step(controller, affine2, critic0, mu0 + 0.2, 1.02)
    assert not diag2["stage7t_event_accepted"]
    assert np.allclose(command2, mu0 + 0.2 + affine2[2:] + critic0[2:])

    # A sufficiently large Critic residual change triggers at the next tick.
    critic1 = critic0 + 10.0 * STAGE7T_RESIDUAL_SCALE_NM
    command3, diag3 = step(controller, affine2, critic1, mu0 + 0.3, 1.03)
    assert diag3["stage7t_event_accepted"]
    assert float(diag3["stage7t_inter_event_interval_s"]) >= STAGE7T_DWELL_S
    assert np.allclose(command3, mu0 + 0.3 + affine2[2:] + critic1[2:])
    assert diag3["stage7t_decomposition_contract_error_max_abs"] < 1e-12
    assert diag3["stage7t_affine_execution_contract_error_max_abs"] < 1e-15
    assert diag3["stage7t_physical_contract_error_max_abs"] < 1e-15
    assert diag3["stage7t_critic_update_count"] == 4
    assert diag3["stage7t_eligible_tick_count"] == 4
    assert not diag3["stage7t_trigger_uses_phase"]
    assert not diag3["stage7t_trigger_uses_affine_error"]
    assert not diag3["stage7t_trigger_uses_mu0_error"]
    print("ICRA_STAGE7T_RESIDUAL_EVENT_CONTRACT_PASS")


if __name__ == "__main__":
    main()
