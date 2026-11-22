from copy import deepcopy
import numpy as np
from .errors import require
from .math3d import pose_matrix, transform_points, transform_normals, world_bounds, unit
from .camera import camera_matrices
from .section import apply_sections


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
        texture = node["surface"]["texture"]
        if texture:
            resource(texture,"texture")
        if not node["mesh"]:
            continue
        mesh = resource(node["mesh"],"mesh")
        require(not texture or mesh["uv"] is not None,"texture_uv","textured node requires UV coordinates")
        world = transform_points(np.asarray(mesh["positions"]),matrices[node["key"]])
        normals = transform_normals(mesh["normals"],matrices[node["key"]])
        if visibility[node["key"]]:
            points.append(world)
            objects.append({"id":index,"key":node["key"],"positions":world,"normals":normals,
                            "triangles":np.asarray(mesh["triangles"]),"uv":None if mesh["uv"] is None else np.asarray(mesh["uv"]),
                            "surface":node["surface"],"mirrored":np.linalg.det(matrices[node["key"]][:3,:3])<0})
    requested = payload.get("sections",[])
    require(len(set(requested)) == len(requested),"section_duplicate","scene lists a section more than once")
    sections = [(index,identity,resource(identity,"section")) for index,identity in enumerate(requested,1)]
    node_map = {str(obj["id"]):obj["key"] for obj in objects}
    section_reports = []
    if sections:
        objects,section_reports = apply_sections(objects,sections)
    camera_matrices(payload["lens"],1)
    unit(payload["light"]["toward"])
    bounds = world_bounds(np.concatenate(points)) if points else None
    return {"scene":deepcopy(payload),"objects":objects,"resources":resources,"bounds":bounds,
            "node_map":node_map,
            "section_map":{str(index):{"id":identity,"label":payload_["label"]} for index,identity,payload_ in sections},
            "section_reports":section_reports,
            "triangles":sum(len(obj["triangles"]) for obj in objects)}


def scene_summary(compiled):
    return {"bounds":compiled["bounds"],"visible_objects":len(compiled["objects"]),
            "triangles":compiled["triangles"],"node_map":compiled["node_map"],
            "sections":[{"index":report["index"],"id":report["id"],"label":report["label"],
                         "status":report["status"],
                         "cap_triangles":sum(obj["cap_triangles"] for obj in report["objects"]),
                         "open_chains":sum(len(obj["open_chains"]) for obj in report["objects"])}
                        for report in compiled["section_reports"]],
            "resource_digests":{key:value["digest"] for key,value in compiled["resources"].items()}}
