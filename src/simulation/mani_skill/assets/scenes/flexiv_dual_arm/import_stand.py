"""Convert the supplied STEP stand and extract the two robot mounting frames.

Development-only dependencies: gmsh, trimesh, scipy. Runtime uses bundled meshes.
Usage: python import_stand.py '/path/to/非夕双臂立柱机械臂底座-20260523.stp'
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


def import_stand(source, target):
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.importShapes(str(source.resolve()))
        gmsh.model.occ.synchronize()
        mounts = []
        base_center = None
        for dim, tag in gmsh.model.getEntities(2):
            if gmsh.model.getType(dim, tag) != "Plane":
                continue
            area = gmsh.model.occ.getMass(dim, tag)
            bounds = np.array(gmsh.model.getBoundingBox(dim, tag)).reshape(2, 3)
            center = bounds.mean(axis=0)
            lower, upper = gmsh.model.getParametrizationBounds(dim, tag)
            normal = gmsh.model.getNormal(tag, (np.array(lower) + upper) / 2)
            if area > 700000 and np.allclose(normal, [0, 0, -1]):
                base_center = np.round(center[:2], 2)
            if (20000 < area < 30000 and center[2] > 1000
                    and np.allclose(normal, [0, -1, 0])
                    and np.allclose(bounds[1, [0, 2]] - bounds[0, [0, 2]], [180, 180], atol=0.01)):
                mounts.append(dict(face=tag, center=np.round(center, 3), normal=np.array(normal)))
        if len(mounts) != 2 or base_center is None:
            raise ValueError("Expected two 180 mm vertical mounting disks and a base plate in this STEP file")
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.8)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 15)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.model.mesh.generate(2)
        tags, coordinates, _ = gmsh.model.mesh.getNodes()
        coordinates = np.array(coordinates).reshape(-1, 3)
        order = np.argsort(tags)
        tags = np.array(tags)[order]
        coordinates = coordinates[order]
        floor = float(coordinates[:, 2].min())
        # World +X is forward (CAD -Y), +Y is the robot's left (CAD +X).
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        origin = np.r_[base_center, floor]
        vertices = ((coordinates - origin) @ rotation.T) * 0.001
        scene = trimesh.Scene()
        collisions = []
        for _, volume in gmsh.model.getEntities(3):
            triangles = []
            for _, surface in gmsh.model.getBoundary([(3, volume)], oriented=False):
                types, _, elements = gmsh.model.mesh.getElements(2, surface)
                for kind, nodes in zip(types, elements):
                    if kind == 2:
                        triangles.append(np.searchsorted(tags, nodes).reshape(-1, 3))
            faces = np.concatenate(triangles)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            mesh.remove_unreferenced_vertices()
            collisions.append(mesh.copy())
            # Keep the supplied assembly's caster and wheel parts visibly distinct.
            color = [60, 62, 66, 255] if mesh.bounds[1, 2] < 0.11 else [195, 200, 205, 255]
            mesh.visual = trimesh.visual.TextureVisuals(
                material=trimesh.visual.material.PBRMaterial(baseColorFactor=color, metallicFactor=0.25, roughnessFactor=0.65))
            scene.add_geometry(mesh, node_name=f"cad_solid_{volume}")
        target.mkdir(parents=True, exist_ok=True)
        scene.export(target / "stand.glb")
        collision_mesh = trimesh.util.concatenate(collisions)
        collision_mesh.export(target / "stand_collision.stl")
        frames = {}
        # Facing forward with the robot: +Y is left. The base mesh's connector
        # opening is on local +Y; local +Z is normal to its mounting surface.
        for name, mount in zip(("left", "right"), sorted(mounts, key=lambda x: -x["center"][0])):
            position = rotation @ (mount["center"] - origin) * 0.001
            normal = rotation @ mount["normal"]
            connector = np.array([0., 0., -1. if name == "left" else 1.])
            orientation = np.column_stack([np.cross(connector, normal), connector, normal])
            xyzw = Rotation.from_matrix(orientation).as_quat()
            frames[name] = dict(position=position.tolist(), quaternion_wxyz=xyzw[[3, 0, 1, 2]].tolist(),
                                connector_direction=connector.tolist(), normal=normal.tolist(),
                                cad_center_mm=mount["center"].tolist(), cad_face=mount["face"],
                                # Approximate the supplied photo's elbow-out,
                                # forearms-forward, grippers-down posture.
                                arm_qpos=([-0.323465, 1.963841, 0.553907, 2.263673, 2.451158, 0.345103, 0.]
                                          if name == "left" else
                                          [0.180512, 1.993543, -0.712481, 2.561961, -2.593623, 0.773324, -1.348068]))
        manifest = dict(source=source.name, sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        cad_units="mm", world_units="m", cad_origin_mm=origin.tolist(),
                        cad_to_world_rotation=rotation.tolist(), bounds_m=collision_mesh.bounds.tolist(),
                        solid_count=len(collisions), triangle_count=len(collision_mesh.faces), mounts=frames,
                        pose_reference="Approximate elbow-out, forearms-forward, grippers-down posture from reference.jpg; not measured robot joint angles.")
        (target / "assembly.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2))
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    import_stand(args.source, args.output)
