# Knee Extensor Weakness

This patient family keeps unilateral `vasti_l` weakness for Mild80 and
Severe40. Moderate60 is the bilateral YAC benchmark and weakens both `vasti_l`
and `vasti_r` to the same retained strength.
The official MyoAssist baseline XML, reflex controller, and healthy
77-parameter controller are not modified.

The healthy `vasti_l` and `vasti_r` maximum isometric force is 9500.94 N per
muscle. MuJoCo stores this force scale in element 2 of the actuator muscle
`gainprm` and `biasprm` arrays. The patient runner loads the healthy model and
scales only the configured muscle force entries at runtime:

- `mild.yaml`: 80 percent, 7600.752 N
- `moderate.yaml`: bilateral `vasti_l` and `vasti_r`, each 60 percent,
  5700.564 N
- `severe.yaml`: 40 percent, 3800.376 N

Scaling both entries reduces active and passive force consistently with the
same maximum-force parameter. It does not change neural excitation, activation
dynamics, tendon geometry, any other muscle, the fall conditions, or the model
XML on disk.

Run all four conditions by opening `scripts/run_patient_sweep.py` in VS Code
and selecting **Run Python File**. The sweep uses the unchanged healthy-human
controller, no exoskeleton, and a maximum duration of 20 seconds per condition.
