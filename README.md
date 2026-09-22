# MuJoCo Lower-Limb Exoskeleton — Private Research Preview

This is a clean, private publication snapshot of the MuJoCo lower-limb
human/exoskeleton simulation code. It contains the frozen YAC baselines,
the current ICRA controller modules, model/configuration glue, and bounded
reproduction entry points.

**Repository status: Private research preview.** This snapshot is not a
clinical or hardware validation package. It is intended for selected
collaborators/reviewers and must remain private during the associated review.

## Environment

The source project was developed with Python 3.11 and the versions pinned in
`requirements.txt` (including MuJoCo 3.3.3). macOS/Linux with a working
MuJoCo installation is expected. GUI rendering additionally requires a display;
video rendering requires an ffmpeg-capable `imageio` installation.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python setup_external_models.py
```

The repository includes two small frozen human-controller parameter files and
one validated Healthy100 reference trajectory required by the YAC/ICRA
controllers. The reference is intentionally retained at its audited path; it
is not a general results archive.

## Required external model

The 22-muscle MyoAssist model is not vendored in this repository. Run
`setup_external_models.py` to fetch the pinned upstream checkout into
`models/humans/myoassist/source/`. It is distributed under upstream licenses;
see `THIRD_PARTY_LICENSES.md`. Do not replace it with an unreviewed model.

## Layout

```text
controllers/     frozen YAC and current ICRA controller modules
coupling/        isolated M5 coupling implementation and fixtures
evaluation/      metrics and standard evaluation protocols
models/          project XML/configuration plus patient definitions
orchestration/   model/controller experiment assembly
simulation/      MuJoCo stepping, recording, and model setup
visualization/   replay and video pipeline
scripts/         bounded launchers and contract checks
results/         compact summaries only; raw output is ignored
```

## Reproduction entry points

All commands write generated output below `results/`, which is ignored by
default. Start with a short smoke run and inspect the generated manifest.

```bash
# fast model/setup check
python models/humans/myoassist/demo.py

# short healthy human baseline
python scripts/run_01_human_only.py --duration 1 --no-video --no-evaluation

# short moderate impairment + passive flexible exoskeleton
python scripts/run_04_human_exo_control.py \
  --human moderate60 --assistance zero --duration 1 \
  --no-video --no-evaluation

# inspect frozen YAC benchmark options
python scripts/run_YAC_benchmark.py --help

# controller contract checks (no CMA-ES optimization)
python scripts/test_ICRA_stage7_contract.py
python scripts/test_ICRA_stage7T_contract.py
```

The active-assistance scripts use the included fixed Healthy100 reference at
`results/experiments/20260822_214246__HUMAN_EXO_HEALTHY100_ZERO/`. Use the
top-of-file configuration in `run_ICRA_demo.py` or the explicit YAC launchers
only after the two baseline smoke commands above pass. Do not run CMA-ES as a
setup test.

## Videos and results

No historical raw-result collection, frame dumps, debug videos, CMA-ES
populations, or private manuscript figures are included. The single reference
trajectory is required at runtime and documented in its own README. If final supplementary videos are later
approved, distribute them as a private release asset rather than adding large
generated files to normal Git history.

## License and citation

The repository-level `LICENSE` is a private-preview notice pending the owners'
final licensing decision. Third-party terms are recorded in
`THIRD_PARTY_LICENSES.md`; those terms remain independent of project code.
Add the final paper citation only after the review/anonymity period and
publication metadata are approved.
