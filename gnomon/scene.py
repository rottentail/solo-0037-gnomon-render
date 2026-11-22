from copy import deepcopy
import numpy as np
from .errors import require
from .math3d import pose_matrix, transform_points, transform_normals, world_bounds, unit
from .camera import camera_matrices


def _face_materials(node, face_count, key):
    """Resolve explicit face groups on top of the legacy single surface.

    `surfaces[0]` is always the node's `surface`, so a node without groups
    behaves exactly as before and keeps its single-material meaning.
    """
    surfaces = [deepcopy(node["surface"])]
    names = ["default"]
    face_surface = np.zeros(face_count, dtype=np.int64)
    owner = np.full(face_count, -1, dtype=np.int64)
    for position, group in enumerate(node["material_groups"]):
        require(group["name"] not in names, "material_group",
                f"material group name '{group['name']}' is duplicated on node '{key}'")
        faces = np.asarray(group["faces"], dtype=np.int64)
        require(bool(np.all(faces >= 0)) and int(faces.max(initial=-1)) < face_count, "material_face",
                f"material group '{group['name']}' references a missing face on node '{key}'")
        require(len(np.unique(faces)) == len(faces), "material_group",
                f"material group '{group['name']}' lists a face more than once on node '{key}'")
        require(bool(np.all(owner[faces] < 0)), "material_overlap",
                f"material group '{group['name']}' overlaps another group on node '{key}'")
        owner[faces] = position + 1
        face_surface[faces] = len(surfaces)
        surfaces.append(deepcopy(group["surface"]))
        names.append(group["name"])
    return surfaces, face_surface, names


def compile_scene(payload, resource_getter, allow_retired=False):
    nodes = {node["key"]:node for node in payload["nodes"]}
    require(len(nodes) == len(payload["nodes"]),"node_duplicate","node keys must be unique")
    matrices, visibility, visiting = {}, {}, set()
    def visit(key):
        if key in matrices:
            return matrices[key]
        require(key in nodes,"parent_missing","node parent does not exist")
        require(key not in visiting,"parent_cycle","node hierarchy contains a cycle")
        visiting.add(key)
        node = nodes[key]
        matrix = pose_matrix(node["pose"])
        visible = node["visible"]
        if node["parent"]:
            matrix = visit(node["parent"]) @ matrix
            visible = visible and visibility[node["parent"]]
        require(np.isfinite(matrix).all() and abs(np.linalg.det(matrix[:3,:3])) > 1e-15,"pose_singular","world transform is singular")
        matrices[key],visibility[key] = matrix,visible
        visiting.remove(key)
        return matrix
    for key in nodes:
        visit(key)
    resources, objects, points = {}, [], []
    def resource(identity, kind):
        if identity not in resources:
            item = resource_getter(identity,kind)
            require(allow_retired or item["state"] == "active","resource_retired","new scene references a retired resource",409)
            resources[identity] = item
        require(resources[identity]["kind"] == kind,"resource_kind","resource kind mismatch")
        return resources[identity]["payload"]
    for index,node in enumerate(payload["nodes"],1):
        if not node["mesh"]:
            require(not node["material_groups"],"material_group",
                    f"material groups require a mesh on node '{node['key']}'")
            if node["surface"]["texture"]:
                resource(node["surface"]["texture"],"texture")
            continue
        mesh = resource(node["mesh"],"mesh")
        triangles = np.asarray(mesh["triangles"])
        surfaces, face_surface, group_names = _face_materials(node, len(triangles), node["key"])
        for surface in surfaces:
            if surface["texture"]:
                resource(surface["texture"],"texture")
        textured = any(surface["texture"] for surface in surfaces)
        require(not textured or mesh["uv"] is not None or mesh.get("corner_uv") is not None,
                "texture_uv","textured node requires UV coordinates")
        world = transform_points(np.asarray(mesh["positions"]),matrices[node["key"]])
        corner_normals = mesh.get("corner_normals")
        if corner_normals is None:
            corner_normals = np.asarray(mesh["normals"])[triangles]
        else:
            corner_normals = np.asarray(corner_normals)
        shape = corner_normals.shape
        normals = transform_normals(corner_normals.reshape(-1,3),matrices[node["key"]]).reshape(shape)
        corner_uv = mesh.get("corner_uv")
        if corner_uv is None and mesh["uv"] is not None:
            corner_uv = np.asarray(mesh["uv"])[triangles]
        if visibility[node["key"]]:
            points.append(world)
            objects.append({"id":index,"key":node["key"],"mesh_id":node["mesh"],
                            "positions":world,"corner_normals":normals,
                            "triangles":triangles,
                            "corner_uv":None if corner_uv is None else np.asarray(corner_uv),
                            "surfaces":surfaces,"face_surface":face_surface,"group_names":group_names,
                            "mirrored":np.linalg.det(matrices[node["key"]][:3,:3])<0})
    camera_matrices(payload["lens"],1)
    unit(payload["light"]["toward"])
    bounds = world_bounds(np.concatenate(points)) if points else None
    return {"scene":deepcopy(payload),"objects":objects,"resources":resources,"bounds":bounds,
            "node_map":{str(obj["id"]):obj["key"] for obj in objects},
            "triangles":sum(len(obj["triangles"]) for obj in objects),
            "material_groups":sum(max(len(obj["group_names"])-1,0) for obj in objects)}


def scene_summary(compiled):
    return {"bounds":compiled["bounds"],"visible_objects":len(compiled["objects"]),
            "triangles":compiled["triangles"],"node_map":compiled["node_map"],
            "material_groups":compiled["material_groups"],
            "resource_digests":{key:value["digest"] for key,value in compiled["resources"].items()}}
