# MyoAssist

This folder contains the local integration for the official
[NeuMove Lab MyoAssist repository](https://github.com/neumovelab/myoassist).

## Files

- `source/`: unmodified third-party repository checkout.
- `config.yaml`: selected model path and verified dimensions.
- `demo.py`: loads, runs, and validates the selected MyoAssist model.

The selected model is the 22-muscle 2D baseline model:

```text
source/models/22muscle_2D/myoLeg22_2D_BASELINE.xml
```

The checkout is pinned to commit
`c10883728bbe20894d1ab59b848d3f4ba57c2a29`. To keep the local download
focused on the current work, it uses Git sparse checkout for the official
README, license, controller code, utilities, 22-muscle models, meshes, and
terrain file. No file under `source/` is modified by this project.

## Validate

From the project root:

```bash
.venv/bin/python models/humans/myoassist/demo.py
.venv/bin/python scripts/inspect_model.py \
  models/humans/myoassist/source/models/22muscle_2D/myoLeg22_2D_BASELINE.xml
```

Open the model in MuJoCo Viewer:

```bash
.venv/bin/python scripts/view_mujoco.py \
  models/humans/myoassist/source/models/22muscle_2D/myoLeg22_2D_BASELINE.xml
```

Run the official human-only optimized reflex controller:

```bash
.venv/bin/python scripts/run_human_walk.py
```

MyoAssist is distributed under its upstream licenses. See `source/LICENSE`
and the attribution comments in the selected XML.
