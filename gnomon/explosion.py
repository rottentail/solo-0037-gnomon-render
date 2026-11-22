"""Exploded-view planning for hierarchical assembly scenes.

The source scene is never modified. An explosion is a separate scene payload
plus a queryable plan of before/after positions. Author-selected component
groups move as rigid units: every node owned by a group receives exactly one
mesh-relative translation, so parent nesting and shared meshes can never move
a part twice. Connector lines anchor at the original assembly join positions.
"""
from copy import deepcopy
from io import BytesIO
import math
import numpy as np
from PIL import Image, ImageDraw
from .camera import camera_matrices
from .errors import require
from .math3d import pose_matrix, transform_points, unit, world_bounds


def _node_index(scene):
    nodes = {node["key"]: node for node in scene["nodes"]}
    require(len(nodes) == len(scene["nodes"]), "node_duplicate", "node keys must be unique")
    return nodes


def _root_keys(nodes):
    roots = [key for key, node in nodes.items() if not node["parent"]]
    require(roots, "scene_forest", "scene must contain a root node")
    return roots


def _world_matrices(nodes):
    matrices, visiting = {}, set()

    def visit(key):
        if key in matrices:
            return matrices[key]
        require(key in nodes, "parent_missing", "node parent does not exist")
        require(key not in visiting, "parent_cycle", "node hierarchy contains a cycle")
        visiting.add(key)
        node = nodes[key]
        matrix = pose_matrix(node["pose"])
        if node["parent"]:
            matrix = visit(node["parent"]) @ matrix
        visiting.remove(key)
        matrices[key] = matrix
        return matrix

    for key in nodes:
        visit(key)
    return matrices


def resolve_groups(nodes, roots, selected):
    """Map every node to its nearest selected ancestor (None for the static remainder)."""
    selected_set = set(selected)
    require(selected_set, "explode_group", "at least one component group is required")
    unknown = selected_set.difference(nodes)
    require(not unknown, "explode_group", "unknown component group keys: " + ",".join(sorted(unknown)))
    children = {}
    for key, node in nodes.items():
        if node["parent"]:
            children.setdefault(node["parent"], []).append(key)
    anchor = {}
    for root in roots:
        stack = [(root, root if root in selected_set else None)]
        while stack:
            key, owner = stack.pop()
            if key in selected_set:
                owner = key
            anchor[key] = owner
            for child in children.get(key, ()):
                stack.append((child, owner))
    return selected_set, anchor


def _node_world_bounds(nodes, matrices, resource_getter):
    bounds = {}
    for key, node in nodes.items():
        if not node["mesh"]:
            bounds[key] = None
            continue
        mesh = resource_getter(node["mesh"], "mesh")["payload"]
        world = transform_points(np.asarray(mesh["positions"], dtype=np.float64), matrices[key])
        bounds[key] = world_bounds(world)
    return bounds


def _union_bounds(boxes):
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return None
    return {"min": [min(box["min"][i] for box in boxes) for i in range(3)],
            "max": [max(box["max"][i] for box in boxes) for i in range(3)]}


def _shift_box(box, delta):
    if box is None:
        return None
    return {"min": [box["min"][i] + delta[i] for i in range(3)],
            "max": [box["max"][i] + delta[i] for i in range(3)]}


def _group_parts(nodes, anchor, bounds):
    members = {}
    for key, owner in anchor.items():
        if owner is not None:
            members.setdefault(owner, []).append(key)
    group_boxes = {owner: _union_bounds([bounds[key] for key in keys])
                   for owner, keys in members.items()}
    return members, group_boxes


def _default_direction(group_key, nodes, world_matrices):
    parent = nodes[group_key]["parent"]
    if parent is None:
        return np.array([0.0, 0.0, 1.0])
    radial = world_matrices[group_key][:3, 3] - world_matrices[parent][:3, 3]
    if np.linalg.norm(radial) <= 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return unit(radial)


def _projected_gap(box_a, box_b, axis):
    """Gap between two AABBs projected onto axis; zero while they overlap."""
    if box_a is None or box_b is None:
        return 0.0

    def interval(box):
        center = np.array([(box["min"][i] + box["max"][i]) / 2 for i in range(3)])
        half = sum(abs(axis[i]) * (box["max"][i] - box["min"][i]) / 2 for i in range(3))
        projection = float(center @ axis)
        return projection - half, projection + half

    lo_a, hi_a = interval(box_a)
    lo_b, hi_b = interval(box_b)
    return max(0.0, max(lo_a, lo_b) - min(hi_a, hi_b))


def _scene_radius(box):
    if box is None:
        return 1.0
    center = np.array([(box["min"][i] + box["max"][i]) / 2 for i in range(3)])
    return float(max(np.linalg.norm(np.asarray(box["max"]) - center),
                     np.linalg.norm(np.asarray(box["min"]) - center), 1e-9))


def plan_explosion(scene, request, resource_getter):
    """Return a queryable explosion plan and an independent exploded scene payload."""
    nodes = _node_index(scene)
    roots = _root_keys(nodes)
    world_matrices = _world_matrices(nodes)
    selected, anchor = resolve_groups(nodes, roots, request["groups"])
    bounds = _node_world_bounds(nodes, world_matrices, resource_getter)
    members, group_boxes = _group_parts(nodes, anchor, bounds)
    empty_groups = sorted(group for group in selected if group_boxes.get(group) is None)
    require(not empty_groups, "explode_group",
            "component groups must contain at least one mesh node: " + ",".join(empty_groups))

    gap_factor = float(request["gap_factor"])
    spacing = float(request["spacing"])
    require(gap_factor >= 0.0 and spacing >= 0.0, "explode_param",
            "gap_factor and spacing must be non-negative")
    base_radius = _scene_radius(_union_bounds(list(bounds.values())))

    directions = {}
    for group in selected:
        explicit = request["directions"].get(group)
        directions[group] = unit(np.asarray(explicit, dtype=np.float64)) if explicit \
            else _default_direction(group, nodes, world_matrices)

    # One axial distance per group: requested clearance plus a scene-scaled
    # spread, preserving the worst projected gap already present on that axis.
    distances = {}
    for group in selected:
        axis = directions[group]
        projected = [_projected_gap(group_boxes.get(group), group_boxes.get(other), axis)
                     for other in selected if other != group]
        distances[group] = spacing + gap_factor * base_radius + (max(projected) if projected else 0.0)

    # The group root carries the whole shift once, in its own parent frame.
    # Descendants stay at their authored local poses and are carried by the
    # hierarchy, so parent nesting can never move a part twice.
    offsets = {}
    for key, node in nodes.items():
        owner = anchor.get(key)
        if owner is None or key != owner:
            offsets[key] = np.zeros(3)
            continue
        parent_rotation = world_matrices[node["parent"]][:3, :3] if node["parent"] else np.eye(3)
        offsets[key] = parent_rotation.T @ (directions[owner] * distances[owner])

    exploded_scene = deepcopy(scene)
    exploded_nodes = {node["key"]: node for node in exploded_scene["nodes"]}
    for key, node in nodes.items():
        offset = offsets[key]
        if np.any(offset != 0.0):
            pose = exploded_nodes[key]["pose"]
            pose["translate"] = [value + part for value, part in zip(pose["translate"], offset)]

    node_entries = []
    for key in nodes:
        owner = anchor.get(key)
        world_delta = np.zeros(3) if owner is None else directions[owner] * distances[owner]
        origin_before = world_matrices[key][:3, 3]
        origin_after = origin_before + world_delta
        require(bool(np.all(np.isfinite(origin_after)) and np.all(np.abs(origin_after) <= 1e6)),
                "explode_range", "exploded position leaves supported coordinates")
        # local_offset is non-zero only on the group root; descendants are
        # carried by the hierarchy and record a zero local edit.
        node_entries.append({"key": key, "parent": nodes[key]["parent"], "mesh": nodes[key]["mesh"],
                             "group": owner, "moved": owner is not None,
                             "local_offset": offsets[key].tolist(),
                             "world_shift": world_delta.tolist(),
                             "origin_before": origin_before.tolist(),
                             "origin_after": origin_after.tolist()})

    connectors = []
    for group in sorted(selected):
        join = world_matrices[group][:3, 3].copy()
        world_delta = directions[group] * distances[group]
        connectors.append({"group": group, "parent": nodes[group]["parent"],
                           "direction": directions[group].tolist(), "distance": distances[group],
                           "join_before": join.tolist(), "join_after": (join + world_delta).tolist(),
                           "line": {"start": join.tolist(), "end": (join + world_delta).tolist()}})

    group_entries = []
    for group in sorted(selected):
        box_before = group_boxes.get(group)
        group_entries.append({"key": group, "members": sorted(members.get(group, [])),
                              "direction": directions[group].tolist(),
                              "distance": distances[group],
                              "bounds_before": box_before,
                              "bounds_after": _shift_box(box_before, directions[group] * distances[group])})

    after_boxes = []
    for key, box in bounds.items():
        if box is None:
            continue
        owner = anchor.get(key)
        delta = np.zeros(3) if owner is None else directions[owner] * distances[owner]
        after_boxes.append(_shift_box(box, delta))

    plan = {"format": "gnomon-explosion/1",
            "parameters": {"groups": sorted(selected),
                           "directions": {group: directions[group].tolist() for group in sorted(selected)},
                           "gap_factor": gap_factor, "spacing": spacing,
                           "base_radius": base_radius},
            "groups": group_entries, "nodes": node_entries, "connectors": connectors,
            "bounds_before": _union_bounds(list(bounds.values())),
            "bounds_after": _union_bounds(after_boxes),
            "scene": exploded_scene}
    return plan


def proof_lens(reference_lens, box, angle=0.0):
    """Square-framed orthographic lens looking at box from the reference view orbit."""
    center = np.array([(box["min"][i] + box["max"][i]) / 2 for i in range(3)])
    radius = max(float(np.linalg.norm(np.asarray(box["max"]) - center)), 1e-6)
    view_dir = unit(np.asarray(reference_lens["eye"]) - np.asarray(reference_lens["target"]))
    theta = np.radians(angle)
    rotation = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0],
                         [-np.sin(theta), 0, np.cos(theta)]])
    up = np.array([0.0, 0.0, 1.0]) if abs(float(view_dir[1])) > 0.99 else np.array([0.0, 1.0, 0.0])
    distance = radius * 3.0 + 1.0
    eye = center + rotation @ view_dir * distance
    return {"projection": "orthographic", "eye": eye.tolist(), "target": center.tolist(),
            "up": up.tolist(), "near": max(0.01, distance - radius * 1.6),
            "far": distance + radius * 1.6, "span": radius * 2.3}


def overlay_guides(png_bytes, connectors, lens, size):
    """Draw connector lines between original join positions and exploded anchors."""
    view, projection = camera_matrices(lens, 1.0)
    matrix = projection @ view
    with Image.open(BytesIO(png_bytes)) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    amber = (235, 188, 96)
    for connector in connectors:
        points = []
        for point in (connector["line"]["start"], connector["line"]["end"]):
            clip = matrix @ np.array([point[0], point[1], point[2], 1.0])
            if clip[3] <= 1e-9:
                break
            ndc = clip[:3] / clip[3]
            if ndc[0] < -1.05 or ndc[0] > 1.05 or ndc[1] < -1.05 or ndc[1] > 1.05:
                break
            points.append(((ndc[0] + 1) * size / 2, (1 - ndc[1]) * size / 2))
        if len(points) != 2:
            continue
        draw.line([points[0], points[1]], fill=amber, width=2)
        for x, y in points:
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], outline=amber, width=2)
        label_x = min(points[1][0] + 5, size - 7 * len(connector["group"]) - 2)
        label_y = max(2, min(points[1][1] - 6, size - 12))
        draw.text((max(2, label_x), label_y), connector["group"], fill=amber)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def build_proof_sheet(cells, size, columns, labels):
    """Lay orthographic proof views out as one proof sheet; cells are PNG bytes."""
    rows = math.ceil(len(cells) / columns)
    canvas = Image.new("RGB", (columns * size, rows * (size + 24)), (18, 23, 33))
    draw = ImageDraw.Draw(canvas)
    for index, (raw, label) in enumerate(zip(cells, labels)):
        x, y = (index % columns) * size, (index // columns) * (size + 24)
        with Image.open(BytesIO(raw)) as image:
            canvas.paste(image.convert("RGB"), (x, y))
        draw.text((x + 6, y + size + 5), label, fill=(211, 224, 238))
    output = BytesIO()
    canvas.save(output, format="PNG")
    return output.getvalue()
