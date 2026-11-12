"""Import Rizon4 from a local flexivrobotics/flexiv_description checkout.

Requires xacro and trimesh. Usage: python import_model.py /path/to/flexiv_description
"""

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET


def import_model(source: Path):
    import xacro
    import trimesh

    target = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        for name in ("flexiv_arm.xacro", "flexiv_common.xacro", "grav_macro.xacro"):
            content = (source / "urdf/common" / name).read_text()
            content = content.replace("$(find flexiv_description)/urdf/common", str(temporary))
            content = content.replace("$(find flexiv_description)", str(source))
            (temporary / name).write_text(content)
        config = source / "config/Rizon4"
        wrapper = temporary / "rizon4.xacro"
        wrapper.write_text(f'''<robot name="flexiv_rizon4" xmlns:xacro="http://wiki.ros.org/xacro">
<xacro:include filename="{temporary}/flexiv_arm.xacro"/>
<xacro:include filename="{temporary}/grav_macro.xacro"/>
<xacro:flexiv_arm prefix="" add_world_link="false" create_base_joint="false"
 joint_limits_parameters_file="{config}/joint_limits.yaml"
 kinematics_parameters_file="{config}/default_kinematics.yaml"
 inertia_parameters_file="{config}/inertia_parameters.yaml"
 visual_parameters_file="{config}/visual_parameters.yaml"/>
<xacro:flexiv_GN01 prefix="" mesh_prefix_path="package://flexiv_description/"/>
</robot>''')
        root = ET.fromstring(xacro.process_file(str(wrapper)).toxml())
    # Upstream emits inertia children directly under link. URDF requires a wrapper.
    for link in root.findall("link"):
        mass = link.find("mass")
        if mass is not None:
            inertial = ET.Element("inertial")
            for tag in ("origin", "mass", "inertia"):
                child = link.find(tag)
                link.remove(child)
                inertial.append(child)
            link.insert(0, inertial)
        workspace = link.find("workspace")
        if workspace is not None:
            link.remove(workspace)
    for mesh in root.iter("mesh"):
        mesh.set("filename", mesh.get("filename").replace(
            "package://flexiv_description/meshes/Rizon4/", "meshes/").replace(
            "package://flexiv_description/meshes/Grav/", "meshes/Grav/"))
    # The ROS description leaves mimic joints unactuated (effort=0). Give
    # PhysX drives a finite effort budget; ManiSkill supplies the mimic targets.
    for joint in root.findall("joint"):
        if joint.find("mimic") is not None:
            joint.find("limit").set("effort", "5")
            # Use the agent's flattened mimic controller instead of PhysX
            # tendons, whose nested offset handling differs from ROS mimic.
            joint.remove(joint.find("mimic"))
    # Embed black material in visual GLBs. URDF material elements cause SAPIEN
    # to initialize Vulkan even when running physics-only CPU tests.
    for visual in root.findall(".//visual"):
        material = visual.find("material")
        if material is not None:
            visual.remove(material)
            mesh = visual.find("geometry/mesh")
            filename = Path(mesh.get("filename"))
            output = target / filename.with_suffix(".glb")
            output.parent.mkdir(parents=True, exist_ok=True)
            geometry = trimesh.load(source / "meshes/Grav/visual" / filename.name, force="mesh")
            geometry.visual = trimesh.visual.TextureVisuals(
                material=trimesh.visual.material.PBRMaterial(baseColorFactor=[30, 30, 30, 255]))
            geometry.export(output)
            mesh.set("filename", str(filename.with_suffix(".glb")))
    ET.indent(root)
    (target / "rizon4.urdf").write_text(
        '<?xml version="1.0"?>\n<!-- Adapted from Flexiv flexiv_description; see README.md and LICENSE. -->\n'
        + ET.tostring(root, encoding="unicode") + "\n")
    shutil.copytree(source / "meshes/Rizon4", target / "meshes", dirs_exist_ok=True)
    shutil.copytree(source / "meshes/Grav", target / "meshes/Grav", dirs_exist_ok=True)
    shutil.copyfile(source / "LICENSE", target / "LICENSE")
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    (target / "UPSTREAM_REVISION").write_text(revision + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    import_model(parser.parse_args().source.resolve())
