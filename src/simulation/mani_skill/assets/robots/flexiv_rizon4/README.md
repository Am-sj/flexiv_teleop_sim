# Flexiv Rizon4 model

Source: https://github.com/flexivrobotics/flexiv_description
Revision: see `UPSTREAM_REVISION`. License: Apache-2.0, included in `LICENSE`.
The meshes and physical/kinematic parameters originate from Flexiv Robotics.

`rizon4.urdf` is expanded from upstream `flexiv_arm.xacro` with the Rizon4
configuration plus `grav_macro.xacro` / `flexiv_GN01`, without the world joint
or external wrist force sensor.
The fixed base is handled by ManiSkill; `grav_tcp` is the TCP. Mesh paths are
relative, so running the simulation requires neither ROS nor xacro nor a download.
The upstream mass, inertia and inertial origin elements are wrapped in URDF
`inertial` elements, and the custom `workspace` element is removed. Meshes are
copied unchanged, including OBJ material files. Grav visual STL meshes with
URDF black material are also converted to GLB with embedded colors, allowing
physics-only CPU loading without initializing Vulkan.

Grav's URDF mimic tags are replaced with the agent's flattened mimic controller
so nested relations do not create conflicting PhysX tendons. The six driven
linkage joints have a simulation effort limit of 5 Nm instead of the ROS
passive-joint value of zero. Their targets retain the official multipliers and
offsets: outer/inner knuckles = `9.404 * width - 0.155`, finger mounts = the
negative of that angle. The virtual width ranges from 0 (closed) to 0.1 (open).
Internal gripper link collisions are filtered; finger/object and arm collisions
remain enabled. The initial gripper configuration is fully open.
Each arm uses a separate internal gripper filter bit in dual-arm environments,
so collisions between the two grippers remain enabled. `rizon4.srdf` filters
only the rigid `link7` / `grav_base_link` mounting seam, whose meshes overlap
slightly at their fixed connection.

To regenerate, install xacro and trimesh in a development environment and run:

```bash
python import_model.py /path/to/flexiv_description
```

Use the revision recorded in `UPSTREAM_REVISION` for reproducible regeneration.
Controller gains in the ManiSkill agent are simulation parameters; this model
provides position control and does not implement Flexiv RDK force control.
