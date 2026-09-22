# Healthy Human with Nature Flexible Hip-Knee Exoskeleton

Mechanical version: `healthy_exo_nature_flexible_v1`.

This model is generated from the frozen `corrected_hip_knee_exo_v1` rigid
baseline. Human mechanics and the exoskeleton's 8 kg component mass, COM,
inertia, geometry and appearance are preserved. Only the Human-Exo topology is
changed: the exoskeleton has an independent three-DoF planar root and bilateral
hip/knee joints.

Active M5 interfaces in this version are waist, left thigh and right thigh.
Shank sites are present for `OUR_HIP_KNEE_EXTENSION` but coupling is disabled.
The XML contains no bushing force implementation; M3 applies the existing M5
LinearBushing at each MuJoCo physics substep.

Regenerate and audit the XML with:

```bash
.venv/bin/python models/coupled/flexible_exo/build_flexible_model.py
```
