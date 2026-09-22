#!/usr/bin/env python3
"""Load an MJCF/XML model and print its main dimensions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np


def model_name(model: mujoco.MjModel) -> str:
    return bytes(model.names).split(b"\0", maxsplit=1)[0].decode("utf-8")


def muscle_count(model: mujoco.MjModel) -> int:
    return int(
        np.count_nonzero(
            model.actuator_gaintype == mujoco.mjtGain.mjGAIN_MUSCLE
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect an MJCF/XML model.")
    parser.add_argument("xml", help="Path to the model XML file.")
    args = parser.parse_args()
    xml_path = Path(args.xml).expanduser().resolve()

    try:
        if not xml_path.is_file():
            raise FileNotFoundError(f"Model XML not found: {xml_path}")

        model = mujoco.MjModel.from_xml_path(str(xml_path))
        print(f"file: {xml_path}")
        print(f"model: {model_name(model)}")
        print(f"nq: {model.nq}")
        print(f"nv: {model.nv}")
        print(f"nu: {model.nu}")
        print(f"body: {model.nbody}")
        print(f"joint: {model.njnt}")
        print(f"actuator: {model.nu}")
        print(f"tendon: {model.ntendon}")
        print(f"muscle: {muscle_count(model)}")
        return 0
    except Exception as exc:
        print(f"ERROR: Could not inspect model: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
