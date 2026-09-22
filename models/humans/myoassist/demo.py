#!/usr/bin/env python3
"""Load, run, and validate the configured MyoAssist model."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml


MODEL_DIR = Path(__file__).resolve().parent
CONFIG_PATH = MODEL_DIR / "config.yaml"
SIMULATION_STEPS = 10


def model_name(model: mujoco.MjModel) -> str:
    return bytes(model.names).split(b"\0", maxsplit=1)[0].decode("utf-8")


def muscle_count(model: mujoco.MjModel) -> int:
    return int(
        np.count_nonzero(
            model.actuator_gaintype == mujoco.mjtGain.mjGAIN_MUSCLE
        )
    )


def main() -> int:
    try:
        with CONFIG_PATH.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)

        model_config = config["model"]
        xml_path = (MODEL_DIR / model_config["xml_path"]).resolve()
        if not xml_path.is_file():
            raise FileNotFoundError(f"MyoAssist XML not found: {xml_path}")

        model = mujoco.MjModel.from_xml_path(str(xml_path))
        actual = {
            "nq": model.nq,
            "nv": model.nv,
            "nu": model.nu,
            "muscle_count": muscle_count(model),
        }
        expected = {
            **model_config["dimensions"],
            "muscle_count": model_config["muscle_count"],
        }
        if actual != expected:
            raise ValueError(f"Model dimensions do not match config: {actual} != {expected}")
        if model_name(model) != model_config["mujoco_name"]:
            raise ValueError(
                f"MuJoCo model name does not match config: {model_name(model)!r}"
            )

        data = mujoco.MjData(model)
        if model.nkey:
            mujoco.mj_resetDataKeyframe(model, data, 0)
        for _ in range(SIMULATION_STEPS):
            mujoco.mj_step(model, data)

        state = np.concatenate((data.qpos, data.qvel, data.qacc))
        if not np.isfinite(state).all():
            raise FloatingPointError("Simulation produced non-finite state values")

        print(f"Model: {model_config['name']}")
        print(f"XML: {xml_path}")
        print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}")
        print(f"muscles={actual['muscle_count']}")
        print(f"simulation_time={data.time:.3f} s")
        print("MyoAssist validation passed.")
        return 0
    except Exception as exc:
        print(f"ERROR: MyoAssist validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
