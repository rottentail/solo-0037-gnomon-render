from copy import deepcopy
import numpy as np
from .errors import require
from .math3d import pose_matrix, transform_points, transform_normals, world_bounds, unit
from .camera import camera_matrices

MAX_OBJECTS = 256


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
    arrays = {}
    def mesh_arrays(identity):
        # Instances of one node (and sibling nodes) share these converted arrays.
        if identity not in arrays:
            mesh = resource(identity,"mesh")
            arrays[identity] = {"positions":np.asarray(mesh["positions"]),"normals":np.asarray(mesh["normals"]),
                                "triangles":np.asarray(mesh["triangles"]),
                                "uv":None if mesh["uv"] is None else np.asarray(mesh["uv"])}
        return arrays[identity]
    total = 0
    for index,node in enumerate(payload["nodes"],1):
        if not node["mesh"]:
            texture = node["surface"]["texture"]
            if texture:
                resource(texture,"texture")
            continue
        mesh = mesh_arrays(node["mesh"])
        instances = node["instances"]
        if instances is not None:
            keys = [item["key"] for item in instances]
            require(len(set(keys)) == len(keys),"instance_duplicate","instance keys must be unique within a node")
        variants = [None] if instances is None else instances
        for variant in variants:
            total += 1
            require(total <= MAX_OBJECTS,"instance_budget",f"scene expands beyond {MAX_OBJECTS} renderable objects")
            if variant is None:
                object_id,matrix,surface,instance = index,matrices[node["key"]],node["surface"],None
                visible = visibility[node["key"]]
            else:
                object_id,instance = len(payload["nodes"])+total,variant["key"]
                matrix = matrices[node["key"]] @ pose_matrix(variant["pose"])
                require(np.isfinite(matrix).all() and abs(np.linalg.det(matrix[:3,:3])) > 1e-15,"pose_singular","instance world transform is singular")
                surface = variant["surface"] if variant["surface"] is not None else node["surface"]
                visible = visibility[node["key"]] and variant["visible"]
            texture = surface["texture"]
            if texture:
                resource(texture,"texture")
            require(not texture or mesh["uv"] is not None,"texture_uv","textured node requires UV coordinates")
            world = transform_points(mesh["positions"],matrix)
            normals = transform_normals(mesh["normals"],matrix)
            if visible:
                points.append(world)
                objects.append({"id":object_id,"key":node["key"],"instance":instance,"positions":world,"normals":normals,
                                "triangles":mesh["triangles"],"uv":mesh["uv"],
                                "surface":surface,"mirrored":np.linalg.det(matrix[:3,:3])<0})
    camera_matrices(payload["lens"],1)
    unit(payload["light"]["toward"])
    bounds = world_bounds(np.concatenate(points)) if points else None
    return {"scene":deepcopy(payload),"objects":objects,"resources":resources,"bounds":bounds,
            "node_map":{str(obj["id"]):(obj["key"] if obj["instance"] is None else f'{obj["key"]}/{obj["instance"]}') for obj in objects},
            "instance_map":{str(obj["id"]):obj["instance"] for obj in objects if obj["instance"] is not None},
            "triangles":sum(len(obj["triangles"]) for obj in objects)}


def scene_summary(compiled):
    return {"bounds":compiled["bounds"],"visible_objects":len(compiled["objects"]),
            "instances":len(compiled["instance_map"]),
            "triangles":compiled["triangles"],"node_map":compiled["node_map"],
            "resource_digests":{key:value["digest"] for key,value in compiled["resources"].items()}}
