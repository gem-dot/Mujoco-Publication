# Healthy MyoAssist With Simple Hip-Knee Exoskeleton

Mechanical version: `corrected_hip_knee_exo_v1` (frozen). Geometry placement,
component masses, inertias, joint structure, collision settings, and timestep
must not be changed while optimizing or validating its 77-parameter controller.

`healthy_exo.xml` is a local copy of the official MyoAssist 22-muscle 2D
baseline with an ideal rigid bilateral hip-knee exoskeleton attached to the
pelvis, femurs, and tibias. It adds 8.0 kg of inertial mass and no independent
exoskeleton degrees of freedom. Collision is disabled for the ideal alignment
geometry.

The frozen 8 kg mechanical distribution is proximal: 3.6 kg on the pelvis,
0.7 kg per hip module, 1.2 kg per thigh bar, 0.15 kg per knee module, and
0.15 kg per shank bar. The pre-freeze diagnostic variants and their results are
kept under `mechanical_diagnosis/` and `results/healthy_exo/mechanical_diagnosis/`.

The pelvis battery is mounted behind the human pelvis: its COM is 80 mm
posterior to the human pelvis COM along the pelvis local x direction. The waist
belt wraps toward both sides, and the hip modules, thigh bars, knee modules,
and shank bars are offset 55 mm laterally outward. Their collision geometry
remains disabled for ideal rigid alignment.

The project channels are `HipExo_R`, `HipExo_L`, `KneeExo_R`, and `KneeExo_L`.
The reserved third-party-style ankle channels `Exo_R` and `Exo_L` remain in the
coupled model but are dormant. All six non-muscle channels are commanded to
zero in the zero-assistance validation.

The human XML, official reflex controller, and original healthy 77-parameter
controller are not modified. The controller optimized before the
installation-position correction remains at
`results/healthy_exo/healthyexo/healthy_exo_controller.txt`; it walks 19.11
seconds on this corrected model. The corrected-model warm-start optimization
result is stored separately at
`results/healthy_exo/corrected_model/healthy_exo_corrected_controller.txt`.
It passes the 20-second validation and completes the 30-second extension while
all six non-muscle actuator controls and forces remain exactly zero.
