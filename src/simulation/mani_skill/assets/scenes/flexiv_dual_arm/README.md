# Flexiv dual-arm stand assets

Derived from the user-supplied `非夕双臂立柱机械臂底座-20260523.stp`.
The source SHA256, coordinate transform, CAD face identifiers and mounting frames
are recorded in `assembly.json`. No external model download is needed at runtime.

- `stand.glb`: 36 CAD solids, triangulated and transformed from millimeters to meters.
- `stand_collision.stl`: static triangle collision geometry for the stand.
- `import_stand.py`: reproducible converter using gmsh, trimesh and scipy.
- `assembly.json`: world frame, mount poses and default arm joint configurations.

World +X is forward, +Y is the robot's left, +Z is up. Left/right are defined
from the robot's perspective, not the front-facing observer's perspective.
The left connector points down, and the right connector points up.

See `src/simulation/DUAL_FLEXIV.md` for usage and control examples.
