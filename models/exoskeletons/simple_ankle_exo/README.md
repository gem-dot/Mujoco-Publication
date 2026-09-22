# Simple Ankle Exoskeleton

This is a first-pass engineering model, not a model of a specific commercial device.
It contains a rigid right and left shank support, foot support, ankle hinge, and
one actuator per side. Each side is 1.5 kg, for a 3.0 kg total.

The standalone XML is only for geometry and mass validation. The coupled model
attaches equivalent rigid support bodies to the MyoAssist tibia and calcaneus
bodies and keeps the human ankle degrees of freedom.

The coupled zero-assistance test commands `Exo_R=0` and `Exo_L=0`.
