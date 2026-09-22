# Third-party licenses and external assets

This file records the provenance relevant to the private publication snapshot.
It is not a substitute for the upstream license texts.

| Component | Upstream source | License evidence | Included here | Required action |
|---|---|---|---|---|
| MyoAssist 22-muscle model and utilities | [NeuMove Lab MyoAssist](https://github.com/neumovelab/myoassist), pinned commit `c10883728bbe20894d1ab59b848d3f4ba57c2a29` | Upstream `LICENSE` (Apache-2.0) and upstream notices | No; fetched by `setup_external_models.py` | Preserve upstream license/notice files and verify current upstream terms before redistribution. |
| OpenSim-derived human model lineage referenced by coupled XML comments | OpenSim/Stanford gait model lineage identified in the XML attribution comments | XML comments state CC BY 3.0 and name original contributors | No human meshes/textures are copied here | Obtain the pinned MyoAssist asset and retain attribution; verify each asset's coverage. |
| MyoSuite components | Included inside the external MyoAssist checkout | Component license files in upstream checkout | No | Do not copy the nested upstream Git history into this repository. |
| MuJoCo Python package | Official MuJoCo distribution | Package terms apply at installation | No binaries vendored | Install from the declared dependency; do not commit binaries. |

Redistribution of the external human model is intentionally deferred to the
upstream fetch step. If permission or access changes, do not substitute a
different model silently; update the setup instructions and validation.
