"""Planar mirror support: plane fitting, reflected camera, reflection pass layout.

A mirror is a scene node whose surface carries a ``mirror`` block. The reflection
is computed, not pasted: the scene is rendered a second time from the main camera
mirrored across the mirror plane. Because the mirrored camera shares the main
camera's projection, a mirror fragment's reflection sample lands on the fragment's
own screen position, so reflection perspective, mirror edges and pixel picking stay
aligned with the primary channels.
"""
import numpy as np
from .clipping import clip_triangle
from .errors import require


def oriented_plane(positions, triangles, eye):
    """Fit the single plane of a mirror surface; the normal faces the given eye.

    Mirror meshes must be genuinely planar so that one reflection matrix describes
    the whole surface. Frames, bevels and other non-coplanar parts belong on
    separate nodes.
    """
    points = np.asarray(positions, dtype=np.float64)
    corners = points[np.asarray(triangles)]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    length = np.linalg.norm(cross, axis=1)
    directions = cross / length[:, None]
    alignment = directions @ directions[0]
    require(bool(np.all(np.abs(alignment) >= 1 - 1e-6)), "mirror_not_planar",
            "mirror surface triangles must share one plane")
    normal = (cross * np.sign(alignment)[:, None]).sum(axis=0)
    normal /= np.linalg.norm(normal)
    offsets = points @ normal
    span = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    require(float(offsets.max() - offsets.min()) <= 1e-6 * max(1.0, span), "mirror_not_planar",
            "mirror surface vertices must be coplanar")
    offset = -float(offsets.mean())
    if float(np.asarray(eye, dtype=np.float64) @ normal + offset) < 0:
        normal, offset = -normal, -offset
    return normal.tolist(), offset


def reflection_matrix(normal, offset):
    """Homogeneous reflection across the plane n·x + offset = 0."""
    n = np.asarray(normal, dtype=np.float64)
    matrix = np.eye(4)
    matrix[:3, :3] -= 2 * np.outer(n, n)
    matrix[:3, 3] = -2 * offset * n
    return matrix


def screen_bbox(positions, triangles, view, projection, width, height):
    """Float pixel bounds (x0, y0, x1, y1) of a mesh in the frame, or None."""
    homogeneous = np.column_stack([positions, np.ones(len(positions))])
    clip = homogeneous @ (projection @ view).T
    packed = np.column_stack([clip, np.zeros((len(clip), 8))])
    xs, ys = [], []
    for face in np.asarray(triangles):
        for triangle in clip_triangle(packed[face]):
            corners = triangle[:, :4]
            if np.any(corners[:, 3] <= 1e-12):
                continue
            ndc = corners[:, :3] / corners[:, 3, None]
            xs.extend(ndc[:, 0])
            ys.extend(ndc[:, 1])
    if not xs:
        return None
    return ((min(xs) + 1) * width / 2, (1 - max(ys)) * height / 2,
            (max(xs) + 1) * width / 2, (1 - min(ys)) * height / 2)


def plan_reflections(objects, eye, view, projection, width, height, scale, region):
    """Per-mirror reflection pass layout in the scaled reflection frame.

    The reflection pass re-renders only the mirror's screen footprint (plus a
    one-pixel sampling margin) at ``scale`` of the main resolution. Returns
    {object_id: {"width", "height", "scale", "region"}}; mirrors outside the
    frame or viewed edge-on are omitted.
    """
    plans = {}
    scaled_w = max(1, int(round(width * scale)))
    scaled_h = max(1, int(round(height * scale)))
    kx, ky = scaled_w / width, scaled_h / height
    rx, ry, rw, rh = region
    frame = (rx * kx, ry * ky, (rx + rw) * kx, (ry + rh) * ky)
    for obj in objects:
        spec = obj.get("mirror")
        if spec is None:
            continue
        normal = np.asarray(spec["normal"], dtype=np.float64)
        if float(eye @ normal + spec["offset"]) <= 1e-9:
            continue  # Camera on the mirror plane: no reflection to sample.
        bbox = screen_bbox(obj["positions"], obj["triangles"], view, projection, width, height)
        if bbox is None:
            continue
        lo_x = max(bbox[0] * kx, frame[0]) - 1
        lo_y = max(bbox[1] * ky, frame[1]) - 1
        hi_x = min(bbox[2] * kx, frame[2]) + 1
        hi_y = min(bbox[3] * ky, frame[3]) + 1
        x0, y0 = max(0, int(np.floor(lo_x))), max(0, int(np.floor(lo_y)))
        x1, y1 = min(scaled_w, int(np.ceil(hi_x))), min(scaled_h, int(np.ceil(hi_y)))
        if x1 <= x0 or y1 <= y0:
            continue
        plans[obj["id"]] = {"width": scaled_w, "height": scaled_h, "scale": [kx, ky],
                            "region": [x0, y0, x1 - x0, y1 - y0]}
    return plans
