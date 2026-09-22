# Frozen Healthy100 reference

This directory is the fixed validated Healthy100 zero-assistance reference
required by `controllers/YAC.py`, `controllers/ICRA.py`, and
`orchestration/standard_workflow.py`.

Only the runtime-required files are retained: the control-rate trajectory, the
1 kHz physics trajectory, manifest, and validation record. Video, figures,
evaluation exports, and other generated artifacts are intentionally excluded.
The directory name and file contents must not be changed without a new
controller/reference validation.
