# Simple Hip-Knee Exoskeleton

This is an idealized bilateral hip and knee engineering model. It contains a
pelvis frame, two hip modules, two thigh bars, two knee modules, and two shank
bars. It intentionally has no ankle actuator or foot structure and does not
represent a specific commercial device.

The frozen mechanical mass is 8.0 kg with a proximal distribution: 3.6 kg
pelvis frame, 0.7 kg per hip module, 1.2 kg per thigh bar, 0.15 kg per knee
module, and 0.15 kg per shank bar. This distribution was selected after the
zero-assistance mechanical diagnosis showed that the original distal-heavy
8 kg distribution caused an early fall while this proximal distribution ran
for 20 seconds. Dimensions, local centers of mass, and recalculated diagonal
inertias are recorded in `config.yaml`.

The 3.6 kg pelvis-frame COM and main battery geometry are positioned behind
the hip axis. Additional belt geoms wrap laterally around the waist instead of
placing a large box in front of the abdomen.

The standalone model uses the four actuator names `HipExo_R`, `HipExo_L`,
`KneeExo_R`, and `KneeExo_L`. The coupled zero-assistance experiment commands
all four to zero.
