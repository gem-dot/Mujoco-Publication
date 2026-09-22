"""Fetch the pinned external MyoAssist checkout required by the models."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "models" / "humans" / "myoassist" / "source"
URL = "https://github.com/neumovelab/myoassist.git"
COMMIT = "c10883728bbe20894d1ab59b848d3f4ba57c2a29"


def main() -> int:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.exists() and (TARGET / ".git").exists():
        subprocess.run(["git", "-C", str(TARGET), "fetch", "--depth", "1", "origin", COMMIT], check=True)
        subprocess.run(["git", "-C", str(TARGET), "checkout", "--detach", COMMIT], check=True)
    else:
        if TARGET.exists():
            raise RuntimeError(
                f"Refusing to overwrite an incomplete external checkout: {TARGET}. "
                "Remove that directory manually and retry."
            )
        with tempfile.TemporaryDirectory(prefix="myoassist-fetch-", dir=TARGET.parent) as staging:
            staging_path = Path(staging) / "source"
            subprocess.run(["git", "init", "--quiet", str(staging_path)], check=True)
            subprocess.run(["git", "-C", str(staging_path), "remote", "add", "origin", URL], check=True)
            subprocess.run(["git", "-C", str(staging_path), "sparse-checkout", "init", "--cone"], check=True)
            subprocess.run(
                [
                    "git", "-C", str(staging_path), "sparse-checkout", "set",
                    "LICENSE", "README.md", "requirements.txt", "setup.py",
                    "ctrl_optim", "myoassist_utils", "myosuite",
                    "models/22muscle_2D", "models/terrain_config.xml",
                ],
                check=True,
            )
            subprocess.run(["git", "-C", str(staging_path), "fetch", "--depth", "1", "origin", COMMIT], check=True)
            subprocess.run(["git", "-C", str(staging_path), "checkout", "--detach", "FETCH_HEAD"], check=True)
            staging_path.rename(TARGET)
    model = TARGET / "models" / "22muscle_2D" / "myoLeg22_2D_BASELINE.xml"
    if not model.is_file():
        raise FileNotFoundError(f"Pinned MyoAssist model is missing: {model}")
    print(f"External MyoAssist ready at {TARGET}")
    print(f"Pinned commit: {COMMIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
