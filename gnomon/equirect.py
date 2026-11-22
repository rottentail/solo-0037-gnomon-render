"""Equirectangular (360x180) ray-cast imaging.

Unlike the perspective/orthographic pipeline, a full-view still frame is
produced by casting one ray per uniform sub-sample through an equirectangular
projection. Rows are latitude (pitch) and columns are longitude (yaw), so the
output is a receiver-ready 2:1 longitude/latitude map: the left and right
edges are the same yaw seam and the first/last rows touch the poles. No
perspective image is stretched, and no triangle is split across the seam
because intersection happens in world space.
"""
import numpy as np
from .camera import lens_axes
from .color import srgb_to_linear
from .texture import texture_array, sample_texture
from .shading import shade
from .errors import require

# Rays are traced in chunks so the (rays x triangles) vectorised workspace is
# bounded independently of output resolution.
RAY_CHUNK = 65536
# Reject clearly unbounded requests before allocating any tracing workspace.
RAY_BUDGET = 1024 * 1024 * 4


def equirect_ray_count(width, height, samples):
    return width * height * samples * samples


def preflight_equirect(width, height, samples):
    rays = equirect_ray_count(width, height, samples)
    return {"ray_count": rays, "ray_chunk": RAY_CHUNK,
            "estimated_bytes": rays * 128,
            "admissible": rays <= RAY_BUDGET}


def ray_directions(width, height, samples):
    """Unit camera-space ray directions for every uniform sub-sample.

    x indexes columns left-to-right, y rows top-to-bottom. Yaw spans 360
    degrees with the seam at +/-180 degrees; pitch spans +90 (top row) to
    -90 (bottom row). Sub-sample order matches raster.allocate: row-major
    with shape (height*samples, width*samples).
    """
    cols = (np.arange(width * samples) + 0.5) / (width * samples)
    rows = (np.arange(height * samples) + 0.5) / (height * samples)
    yaw = (2 * cols - 1) * np.pi
    pitch = (1 - 2 * rows) * np.pi / 2
    pitch_grid, yaw_grid = np.meshgrid(pitch, yaw, indexing="ij")
    cos_pitch = np.cos(pitch_grid)
    # Camera convention: forward is -z, side/right is +x. With
    # yaw = atan2(ray.x, -ray.z), the first column is yaw -180 (the seam),
    # the centre column yaw 0 (forward) and the last column approaches +180.
    return np.stack([cos_pitch * np.sin(yaw_grid),
                     np.sin(pitch_grid),
                     -cos_pitch * np.cos(yaw_grid)], axis=2)


def prepare_objects(compiled):
    prepared = []
    textures = {key: texture_array(value["payload"])
                for key, value in compiled["resources"].items() if value["kind"] == "texture"}
    for obj in compiled["objects"]:
        positions = np.asarray(obj["positions"], dtype=np.float64)
        triangles = np.asarray(obj["triangles"], dtype=np.int64)
        p0, p1, p2 = positions[triangles[:, 0]], positions[triangles[:, 1]], positions[triangles[:, 2]]
        geometric_normal = np.cross(p1 - p0, p2 - p0)
        prepared.append({"id": obj["id"], "positions": positions, "triangles": triangles,
                         "normals": np.asarray(obj["normals"], dtype=np.float64),
                         "uv": None if obj["uv"] is None else np.asarray(obj["uv"], dtype=np.float64),
                         "surface": obj["surface"], "texture": textures.get(obj["surface"]["texture"]),
                         "geometric_normal": geometric_normal})
    return prepared


def _barycentric_attributes(hit, obj):
    """Interpolate world position, normal and uv at ray-triangle hits."""
    triangles = obj["triangles"]
    a = obj["positions"][triangles[:, 0]][hit["face"]]
    b = obj["positions"][triangles[:, 1]][hit["face"]]
    c = obj["positions"][triangles[:, 2]][hit["face"]]
    u, v = hit["u"], hit["v"]
    weights = np.stack([1 - u - v, u, v], axis=1)
    world = weights[:, 0, None] * a + weights[:, 1, None] * b + weights[:, 2, None] * c
    na = obj["normals"][triangles[:, 0]][hit["face"]]
    nb = obj["normals"][triangles[:, 1]][hit["face"]]
    nc = obj["normals"][triangles[:, 2]][hit["face"]]
    normals = weights[:, 0, None] * na + weights[:, 1, None] * nb + weights[:, 2, None] * nc
    if obj["uv"] is None:
        uv = np.zeros((len(hit["face"]), 2))
    else:
        ua = obj["uv"][triangles[:, 0]][hit["face"]]
        ub = obj["uv"][triangles[:, 1]][hit["face"]]
        uc = obj["uv"][triangles[:, 2]][hit["face"]]
        uv = weights[:, 0, None] * ua + weights[:, 1, None] * ub + weights[:, 2, None] * uc
    return world, normals, uv


def _trace_chunk(origins, directions, objects, near, far):
    """Return per-ray nearest surface hits for one bounded ray chunk."""
    hit_face = np.full(len(origins), -1, dtype=np.int64)
    hit_object = np.zeros(len(origins), dtype=np.int32)
    hit_distance = np.full(len(origins), np.inf)
    hit_u = np.zeros(len(origins))
    hit_v = np.zeros(len(origins))
    for obj in objects:
        triangles = obj["triangles"]
        p0 = obj["positions"][triangles[:, 0]]
        p1 = obj["positions"][triangles[:, 1]]
        p2 = obj["positions"][triangles[:, 2]]
        edge1, edge2 = p1 - p0, p2 - p0
        double_sided = obj["surface"]["double_sided"]
        alpha_cutoff = obj["surface"]["alpha_cutoff"]
        textured = obj["texture"] is not None
        origin, direction = origins, directions
        # Moller-Trumbore against every (ray, triangle) pair.
        h = np.cross(direction[:, None, :], edge2[None, :, :])
        determinant = np.einsum("rtd,td->rt", h, edge1)
        if not double_sided:
            facing = np.einsum("td,rd->rt", obj["geometric_normal"], direction) < -1e-12
            determinant = np.where(facing, determinant, 0.0)
        safe = np.abs(determinant) > 1e-12
        inverse = np.divide(1.0, determinant, out=np.zeros_like(determinant), where=safe)
        rel = origin[:, None, :] - p0[None, :, :]
        u = np.einsum("rtd,rtd->rt", rel, h) * inverse
        q = np.cross(rel, edge1[None, :, :])
        v = np.einsum("rtd,rtd->rt", direction[:, None, :], q) * inverse
        distance = np.einsum("td,rtd->rt", edge2, q) * inverse
        valid = safe & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (distance >= near) & (distance <= far)
        candidate = np.where(valid, distance, np.inf)
        # Respect scene traversal order on exact ties: only strict wins.
        better = candidate < (hit_distance[:, None] - 1e-12)
        choices = np.argmax(np.where(better, candidate, -np.inf), axis=1)
        row = np.arange(len(origin))
        chosen_distance = candidate[row, choices]
        chosen_valid = np.isfinite(chosen_distance)
        if not chosen_valid.any():
            continue
        chosen_u = u[row, choices][chosen_valid]
        chosen_v = v[row, choices][chosen_valid]
        chosen_face = choices[chosen_valid]
        keep = np.ones(chosen_valid.sum(), dtype=bool)
        if textured:
            provisional = {"face": chosen_face, "u": chosen_u, "v": chosen_v}
            _, _, uv = _barycentric_attributes(provisional, obj)
            texels = sample_texture(obj["texture"], uv, obj["surface"]["wrap"])
            keep = texels[:, 3] >= alpha_cutoff
        if not keep.any():
            continue
        valid_rows = np.flatnonzero(chosen_valid)[keep]
        hit_face[valid_rows] = chosen_face[keep]
        hit_object[valid_rows] = obj["id"]
        hit_distance[valid_rows] = chosen_distance[valid_rows]
        hit_u[valid_rows] = chosen_u[keep]
        hit_v[valid_rows] = chosen_v[keep]
    return {"face": hit_face, "object_id": hit_object, "distance": hit_distance, "u": hit_u, "v": hit_v}


def render_equirect(compiled, request):
    width, height, samples = request["width"], request["height"], request["samples"]
    require(request.get("region") is None, "panorama_region",
            "equirectangular imaging covers the full sphere and does not accept a region")
    require(height * 2 == width, "panorama_aspect",
            "equirectangular output must be 2:1 longitude/latitude, use width = 2 * height")
    budget = preflight_equirect(width, height, samples)
    require(budget["admissible"], "render_budget",
            f"panorama ray count exceeds {RAY_BUDGET} ray budget")
    scene = compiled["scene"]
    lens = scene["lens"]
    side, up, forward = lens_axes(lens)
    eye = np.asarray(lens["eye"], dtype=np.float64)
    camera_rays = ray_directions(width, height, samples)
    # Camera-to-world for row-vector rays is the transpose of the planar view
    # rotation: world = cx*side + cy*up - cz*forward because camera forward
    # is -z. This keeps the panorama centre ray aligned with the lens forward.
    basis = np.column_stack([side, up, -forward])
    world_rays = camera_rays @ basis.T
    world_rays = world_rays / np.linalg.norm(world_rays, axis=2, keepdims=True)
    flat_rays = world_rays.reshape(-1, 3)
    origins = np.broadcast_to(eye, flat_rays.shape)
    objects = prepare_objects(compiled)
    ray_count = flat_rays.shape[0]
    buffers = {"color": np.broadcast_to(srgb_to_linear(scene["background"]), (ray_count, 3)).copy(),
               "depth": np.full(ray_count, np.inf), "normal": np.zeros((ray_count, 3)),
               "object_id": np.zeros(ray_count, dtype=np.int32),
               "world": np.zeros((ray_count, 3)), "ray": flat_rays.copy()}
    traced_faces = 0
    for start in range(0, len(flat_rays), RAY_CHUNK):
        end = min(start + RAY_CHUNK, len(flat_rays))
        hit = _trace_chunk(origins[start:end], flat_rays[start:end], objects,
                           lens["near"], lens["far"])
        covered = hit["object_id"] > 0
        if not covered.any():
            continue
        for obj in objects:
            own = covered & (hit["object_id"] == obj["id"])
            if not own.any():
                continue
            local = {"face": hit["face"][own], "u": hit["u"][own], "v": hit["v"][own]}
            world, normals, uv = _barycentric_attributes(local, obj)
            rgb, keep, shaded_normals = shade(obj["surface"], normals, uv, obj["texture"], scene["light"])
            own_index = np.flatnonzero(own)[keep]
            buffers["depth"][start + own_index] = hit["distance"][own_index]
            buffers["color"][start + own_index] = rgb[keep]
            buffers["normal"][start + own_index] = shaded_normals[keep]
            buffers["world"][start + own_index] = world[keep]
            buffers["object_id"][start + own_index] = obj["id"]
            traced_faces += len(local["face"])
    frame_shape = (height * samples, width * samples)
    for key in ("color", "normal", "world", "ray"):
        buffers[key] = buffers[key].reshape(*frame_shape, 3)
    for key in ("depth", "object_id"):
        buffers[key] = buffers[key].reshape(*frame_shape)
    metadata = {"view": None, "projection": "equirectangular",
                "lens_projection": "equirectangular",
                "region": [0, 0, width, height], "width": width, "height": height,
                "samples": samples, "lens": {"projection": "equirectangular",
                                             "eye": list(map(float, eye)),
                                             "target": list(map(float, lens["target"])),
                                             "up": list(map(float, lens["up"])),
                                             "near": lens["near"], "far": lens["far"]},
                "depth_convention": "ray distance in world units from lens eye; +inf background",
                "stats": {"ray_count": len(flat_rays), "ray_chunks": int(np.ceil(len(flat_rays) / RAY_CHUNK)),
                          "surface_ray_hits": int(traced_faces),
                          "visible_triangles": sum(len(o["triangles"]) for o in objects),
                          "estimated_bytes": budget["estimated_bytes"]}}
    return buffers, metadata
