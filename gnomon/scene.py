from copy import deepcopy
import numpy as np
from .errors import require
from .math3d import pose_matrix, transform_points, transform_normals, transform_directions, world_bounds, unit
from .mesh import vertex_frames
from .camera import camera_matrices


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
    def texture_for(identity, expected):
        if not identity:
            return None
        texture = resource(identity,"texture")
        # Records created before texture roles existed carry no role and were
        # always diffuse textures.
        require(texture.get("role","color") == expected,"texture_role",f"{expected} slot requires a {expected} texture")
        return texture
    for index,node in enumerate(payload["nodes"],1):
        texture = texture_for(node["surface"]["texture"],"color")
        normal_map = texture_for(node["surface"]["normal_texture"],"normal")
        if not node["mesh"]:
            require(not node["surface"]["texture"] and not node["surface"]["normal_texture"],
                    "texture_orphan","surface texture requires a mesh on the same node")
            continue
        mesh = resource(node["mesh"],"mesh")
        require(not (texture or normal_map) or mesh["uv"] is not None,"texture_uv","textured node requires UV coordinates")
        world = transform_points(np.asarray(mesh["positions"]),matrices[node["key"]])
        normals = transform_normals(mesh["normals"],matrices[node["key"]])
        tangent = bitangent = None
        if normal_map is not None:
            raw_tangent,raw_bitangent = vertex_frames(np.asarray(mesh["positions"]),np.asarray(mesh["normals"]),
                                                      np.asarray(mesh["uv"]),mesh["triangles"])
            # Tangents are surface directions, so they follow the linear part of
            # the world transform, then are re-orthogonalized against the world
            # normal. That keeps the frame perpendicular under non-uniform scale
            # and lets the bitangent mirror together with the winding.
            tangent = transform_directions(raw_tangent,matrices[node["key"]])
            bitangent = transform_directions(raw_bitangent,matrices[node["key"]])
            # Re-orthonormalize in world space against the world normal.
            tangent = tangent - normals*np.sum(normals*tangent,axis=1,keepdims=True)
            length = np.linalg.norm(tangent,axis=1)
            require(bool(np.all(length > 1e-10)),"normal_tangent",
                    "world transform collapses the tangent frame; normal map cannot be oriented")
            tangent = tangent/length[:,None]
            bitangent = bitangent - normals*np.sum(normals*bitangent,axis=1,keepdims=True)
            bitangent = bitangent - tangent*np.sum(tangent*bitangent,axis=1,keepdims=True)
            length = np.linalg.norm(bitangent,axis=1)
            require(bool(np.all(length > 1e-10)),"normal_tangent",
                    "world transform collapses the bitangent frame; normal map cannot be oriented")
            bitangent = bitangent/length[:,None]
        if visibility[node["key"]]:
            points.append(world)
            objects.append({"id":index,"key":node["key"],"positions":world,"normals":normals,
                            "triangles":np.asarray(mesh["triangles"]),"uv":None if mesh["uv"] is None else np.asarray(mesh["uv"]),
                            "tangent":tangent,"bitangent":bitangent,
                            "surface":node["surface"],"normal_texture":normal_map,
                            "mirrored":np.linalg.det(matrices[node["key"]][:3,:3])<0})
    camera_matrices(payload["lens"],1)
    unit(payload["light"]["toward"])
    bounds = world_bounds(np.concatenate(points)) if points else None
    return {"scene":deepcopy(payload),"objects":objects,"resources":resources,"bounds":bounds,
            "node_map":{str(obj["id"]):obj["key"] for obj in objects},
            "triangles":sum(len(obj["triangles"]) for obj in objects)}


def scene_summary(compiled):
    return {"bounds":compiled["bounds"],"visible_objects":len(compiled["objects"]),
            "triangles":compiled["triangles"],"node_map":compiled["node_map"],
            "resource_digests":{key:value["digest"] for key,value in compiled["resources"].items()}}
