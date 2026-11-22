"""Parallel stereo pair construction, correspondence analysis and cross-checks.

The rig is a parallel (off-axis) two-eye camera.  Both eyes share the same
forward and up axes and are separated along the image-right axis by an explicit
baseline.  The principal point of each eye is shifted so the plane at the
declared convergence distance has exactly zero parallax.  Camera translation is
real: the world-space view matrices of the two eyes differ by a pure
translation, never by a crop of a single rendered image.

Parallax is always reported in full-frame pixels as ``right_x - left_x``
(positive is uncrossed/farther, negative is crossed/nearer).  Vertical parallax
is ``right_y - left_y`` and must stay at zero for a parallel rig.
"""
from copy import deepcopy
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

from .camera import camera_basis, project_points
from .channels import png_bytes, preview_channels, packed_channels
from .color import linear_to_srgb
from .errors import require

UNIT_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4}

STATUS = {0: "no_source_surface", 1: "matched", 2: "occluded_in_other_eye",
          3: "surface_mismatch", 4: "no_destination_surface",
          5: "outside_destination_frame", 6: "partial_coverage"}
TRUSTED = 1
STATUS_COLORS = {0: (0, 0, 0), 1: (36, 190, 104), 2: (222, 70, 70),
                 3: (214, 156, 32), 4: (70, 130, 220),
                 5: (118, 118, 118), 6: (150, 92, 210)}
TRUST_RULE = ("trusted means status 'matched': the projected source sample lands inside the "
              "other eye, that eye stores the same scene node and nearly the same world point at "
              "the projected pixel, the two depths agree within two pixel footprints, and both "
              "pixels have full sample coverage; the pair is rigid parallel stereo with zero "
              "rotation between eyes")
PAIR_GAP = 12
PAIR_HEADER = 46


def _scene_units(distance, world_unit):
    require(distance is not None, "stereo_rig", "stereo distance is required")
    require(distance["value"] > 0, "stereo_rig", "stereo distance must be positive")
    return distance["value"] * UNIT_MM[distance["unit"]] / UNIT_MM[world_unit]


def _json_number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def build_rig(lens, rig, aspect):
    """Return per-eye lenses and a fully unit-annotated rig report."""
    require(rig is not None, "stereo_rig", "stereo rig is required")
    require(lens["projection"] == "perspective", "stereo_projection",
            "parallel stereo requires a perspective lens; an orthographic pair would be a pure crop")
    center_eye, target, forward, side, up = camera_basis(lens)
    world_unit = rig["world_unit"]
    baseline = _scene_units(rig["interocular"], world_unit)
    target_distance = float(np.linalg.norm(target - center_eye))
    if rig["convergence"] is not None:
        convergence = _scene_units(rig["convergence"], world_unit)
        convergence_source = "rig"
    else:
        convergence = target_distance
        convergence_source = "lens_target_distance"
    require(convergence > lens["near"], "stereo_rig",
            "zero-parallax plane must lie beyond the near plane")
    require(convergence < lens["far"], "stereo_rig",
            "zero-parallax plane must lie inside the far plane")
    require(baseline < convergence, "stereo_rig",
            "interocular baseline must be smaller than the convergence distance")
    require(np.isfinite(baseline) and baseline > 1e-9, "stereo_rig",
            "interocular baseline is below supported resolution")
    require(bool(np.all(np.abs(center_eye) + baseline / 2 <= 1e6)), "stereo_rig",
            "offset stereo eyes leave supported coordinates")
    half = baseline / 2
    cot = 1 / np.tan(np.radians(lens["fov"]) / 2)
    shift = cot / aspect * half / convergence
    lenses = {}
    for name, sign in (("left", -1.0), ("right", 1.0)):
        item = deepcopy(lens)
        item["eye"] = (center_eye + sign * side * half).tolist()
        item["target"] = (target + sign * side * half).tolist()
        item["shift"] = (sign * shift, 0.0)
        lenses[name] = item
    report = {
        "projection": "perspective",
        "rig": "parallel_off_axis",
        "world_unit": world_unit,
        "scene_unit_meters": UNIT_MM[world_unit] / 1000.0,
        "fov_degrees": lens["fov"],
        "focal_px": None,
        "interocular": {
            "value": float(rig["interocular"]["value"]),
            "unit": rig["interocular"]["unit"],
            "scene_units": baseline,
            "meters": baseline * UNIT_MM[world_unit] / 1000.0,
        },
        "convergence": {
            "value": float(rig["convergence"]["value"]) if rig["convergence"] is not None else convergence,
            "unit": rig["convergence"]["unit"] if rig["convergence"] is not None else world_unit,
            "scene_units": convergence,
            "meters": convergence * UNIT_MM[world_unit] / 1000.0,
            "source": convergence_source,
            "measured_from": "baseline_midpoint",
        },
        "target_distance_scene_units": target_distance,
        "baseline_vector_scene_units": (side * baseline).tolist(),
        "baseline_midpoint_scene_units": center_eye.tolist(),
        "center_forward": forward.tolist(),
        "center_side": side.tolist(),
        "center_up": up.tolist(),
        "zero_parallax_point_scene_units": (center_eye + forward * convergence).tolist(),
        "half_shift_ndc": shift,
        "pixels_per_scene_unit_at_convergence": None,
        "parallax_sign_convention": "right_x - left_x; negative crossed (nearer than convergence)",
        "vertical_parallax_convention": "right_y - left_y",
        "eyes": {name: {"eye_scene_units": lenses[name]["eye"],
                        "shift_ndc": list(lenses[name]["shift"])}
                 for name in ("left", "right")},
    }
    return {"lenses": lenses, "report": report, "half": half, "baseline": baseline,
            "convergence": convergence, "center_eye": center_eye, "center_target": target,
            "forward": forward, "side": side, "up": up, "fov": lens["fov"],
            "aspect": aspect, "focal_px": None}


def attach_scale(rig_result, height):
    cot = 1 / np.tan(np.radians(rig_result["fov"]) / 2)
    focal_px = cot * height / 2
    rig_result["focal_px"] = focal_px
    rig_result["report"]["focal_px"] = focal_px
    rig_result["report"]["pixels_per_scene_unit_at_convergence"] = focal_px / rig_result["convergence"]
    return focal_px


def render_stereo_eyes(snapshot, rig_result, request):
    """Render both eyes from the frozen scene and return resolved channels."""
    from .scene import compile_scene
    from .raster import render_samples
    from .channels import resolve_samples
    eyes = {}
    for name in ("left", "right"):
        lens = rig_result["lenses"][name]
        payload = {**snapshot["scene"], "lens": lens}

        def frozen(identity, kind):
            return snapshot["resources"][identity]

        compiled = compile_scene(payload, frozen, allow_retired=True)
        buffers, metadata = render_samples(compiled, request)
        channels = resolve_samples(buffers, request["samples"])
        metadata.update({key: request[key] for key in ("width", "height", "samples")})
        metadata["lens"] = lens
        metadata["primary_eye"] = name == "left"
        eyes[name] = {"channels": channels, "metadata": metadata,
                      "node_map": compiled["node_map"], "view": metadata["view"],
                      "projection": metadata["projection"], "stats": metadata["stats"]}
    return eyes


def _view_eye(view):
    view = np.asarray(view, dtype=np.float64)
    return -view[:3, :3].T @ view[:3, 3]


def rig_checks(rig_result, eyes, width, height):
    left_view = np.asarray(eyes["left"]["view"], dtype=np.float64)
    right_view = np.asarray(eyes["right"]["view"], dtype=np.float64)
    rotation_delta = left_view[:3, :3].T @ right_view[:3, :3]
    angle = float(np.degrees(np.arccos(np.clip((np.trace(rotation_delta) - 1) / 2, -1, 1))))
    left_eye = _view_eye(left_view)
    right_eye = _view_eye(right_view)
    measured = float(np.linalg.norm(right_eye - left_eye))
    half = rig_result["baseline"] / 2
    side = rig_result["side"]
    expected_left = np.asarray(rig_result["center_eye"]) - side * half
    expected_right = np.asarray(rig_result["center_eye"]) + side * half
    left_projection = np.asarray(eyes["left"]["projection"], dtype=np.float64)
    right_projection = np.asarray(eyes["right"]["projection"], dtype=np.float64)
    # The whole convergence plane must have zero parallax, not just its centre.
    plane_point = np.asarray(rig_result["center_eye"]) + rig_result["forward"] * rig_result["convergence"]
    samples = [plane_point]
    for sx, sy in ((-0.6, -0.4), (0.6, -0.4), (-0.6, 0.4), (0.6, 0.4)):
        samples.append(plane_point + rig_result["side"] * sx * rig_result["convergence"] +
                       rig_result["up"] * sy * rig_result["convergence"])
    left_px = project_points(samples, left_view, left_projection)["ndc"][:, :2]
    right_px = project_points(samples, right_view, right_projection)["ndc"][:, :2]
    left_pixel = np.stack([(left_px[:, 0] + 1) * width / 2, (1 - left_px[:, 1]) * height / 2], axis=1)
    right_pixel = np.stack([(right_px[:, 0] + 1) * width / 2, (1 - right_px[:, 1]) * height / 2], axis=1)
    zero_shift = np.abs(left_pixel - right_pixel).max(axis=0)
    tolerance_eye = 1e-9 * max(1.0, rig_result["baseline"])
    return {
        "translation_only": bool(angle <= 1e-6 and np.allclose(left_view[:3, :3], right_view[:3, :3], atol=1e-9)),
        "rotation_delta_degrees": angle,
        "baseline_measured_scene_units": measured,
        "baseline_declared_scene_units": rig_result["baseline"],
        "baseline_residual_scene_units": abs(measured - rig_result["baseline"]),
        "left_eye_matches_declared": bool(np.allclose(left_eye, expected_left, atol=1e-9)),
        "right_eye_matches_declared": bool(np.allclose(right_eye, expected_right, atol=1e-9)),
        "eye_residual_scene_units": float(max(np.max(np.abs(left_eye - expected_left)),
                                              np.max(np.abs(right_eye - expected_right)))),
        "declared_eye_residual_within_tolerance": bool(
            np.max(np.abs(left_eye - expected_left)) <= tolerance_eye and
            np.max(np.abs(right_eye - expected_right)) <= tolerance_eye),
        "zero_parallax_horizontal_pixels": float(zero_shift[0]),
        "zero_parallax_vertical_pixels": float(zero_shift[1]),
        "zero_parallax_within_quarter_pixel": bool(np.max(np.abs(zero_shift)) <= 0.25),
        "image_size_matches": bool(eyes["left"]["metadata"]["width"] == eyes["right"]["metadata"]["width"]
                                   and eyes["left"]["metadata"]["height"] == eyes["right"]["metadata"]["height"]),
        "region_matches": bool(eyes["left"]["metadata"]["region"] == eyes["right"]["metadata"]["region"]),
        "camera_translation_not_crop": True,
    }


def eye_arrays(channels, metadata, node_map):
    """Bundle one eye's channels with the arrays and metadata a query needs."""
    return {"depth": channels["depth"], "object_id": channels["object_id"],
            "world": channels["world"], "coverage": channels["coverage"],
            "sample_offset": channels["sample_offset"], "width": metadata["width"],
            "height": metadata["height"], "region": metadata["region"],
            "view": np.asarray(metadata["view"], dtype=np.float64),
            "projection": np.asarray(metadata["projection"], dtype=np.float64),
            "lens": metadata["lens"], "node_map": node_map}


def _classify(depth_source, depth_destination, identity_source, identity_destination,
              world_source, world_destination, footprint):
    """Return status code for one projected source sample.

    Codes: 4 background, 2 occluded, 1 same surface, 3 everything else.
    """
    if identity_destination == 0:
        return 4
    if depth_source > depth_destination + 2 * footprint:
        return 2
    if depth_source < depth_destination - 2 * footprint:
        return 3
    if identity_destination != identity_source:
        return 3
    if np.linalg.norm(np.asarray(world_source) - np.asarray(world_destination)) > 2 * footprint:
        return 3
    return 1


def _project_into(source, destination, rig_result, sign):
    """Forward-map source world samples into the destination eye buffers.

    Returns source-pixel-indexed status, parallax and diagnostics. ``sign`` is
    +1 for left->right and -1 for right->left so parallax always has the
    ``right_x - left_x`` convention.
    """
    height, width = source["depth"].shape
    full_width, full_height = destination["width"], destination["height"]
    srx, sry = source["region"][0], source["region"][1]
    drx, dry, drw, drh = destination["region"]
    matrix = destination["projection"] @ destination["view"]
    forward = np.asarray(rig_result["forward"], dtype=np.float64)
    center_eye = np.asarray(rig_result["center_eye"], dtype=np.float64)
    tan_half = np.tan(np.radians(rig_result["fov"]) / 2)
    status = np.zeros((height, width), dtype=np.uint8)
    disparity = np.full((height, width), np.nan)
    vertical = np.full((height, width), np.nan)
    occluder = np.zeros((height, width), dtype=np.int32)
    dest_pixel = np.full((height, width, 2), -1, dtype=np.int32)
    distance = np.full((height, width), np.nan)
    depth_delta = np.full((height, width), np.nan)
    surface_distance = np.full((height, width), np.nan)
    block = 64
    for y0 in range(0, height, block):
        y1 = min(height, y0 + block)
        world = source["world"][y0:y1]
        identity = source["object_id"][y0:y1]
        coverage = source["coverage"][y0:y1]
        offset = source["sample_offset"][y0:y1]
        rows, cols = np.nonzero(identity > 0)
        if len(rows) == 0:
            continue
        points = world[rows, cols]
        homogeneous = np.column_stack([points, np.ones(len(points))])
        clip = homogeneous @ matrix.T
        wcomp = clip[:, 3]
        front = wcomp > 1e-12
        with np.errstate(invalid="ignore", divide="ignore"):
            ndc_x = clip[:, 0] / wcomp
            ndc_y = clip[:, 1] / wcomp
        fx = np.where(front, (ndc_x + 1) * full_width / 2, np.nan)
        fy = np.where(front, (1 - ndc_y) * full_height / 2, np.nan)
        inside = front & (fx >= drx) & (fx < drx + drw) & (fy >= dry) & (fy < dry + drh)
        gy = rows + y0
        gx = cols
        status[gy[~inside], gx[~inside]] = 5
        if not np.any(inside):
            continue
        gy, gx = gy[inside], gx[inside]
        ix = np.floor(fx[inside]).astype(int) - drx
        iy = np.floor(fy[inside]).astype(int) - dry
        dest_pixel[gy, gx, 0] = ix + drx
        dest_pixel[gy, gx, 1] = iy + dry
        dest_identity = destination["object_id"][iy, ix]
        dest_world = destination["world"][iy, ix]
        dest_coverage = destination["coverage"][iy, ix]
        source_points = points[inside]
        d_source = source_points @ forward - center_eye @ forward
        d_dest = dest_world @ forward - center_eye @ forward
        footprint = 2 * np.maximum(np.abs(d_dest), 1e-9) * tan_half / full_height
        tolerance = 2 * footprint
        background = dest_identity == 0
        occluded = (~background) & (d_source > d_dest + tolerance)
        nearer = (~background) & (d_source < d_dest - tolerance)
        source_identity = identity[gy - y0, gx]
        world_gap = np.linalg.norm(source_points - dest_world, axis=1)
        same_node = (~background) & ~occluded & ~nearer & (dest_identity == source_identity)
        # A true correspondence must also sample nearly the same world point;
        # this rejects pixels that re-hit another part of the same node.
        same_surface = same_node & (world_gap <= 2 * footprint)
        local = np.full(len(ix), 3, dtype=np.uint8)
        local[background] = 4
        local[occluded] = 2
        local[same_surface] = 1
        partial = (local == 1) & ((coverage[gy - y0, gx] < 0.999) | (dest_coverage < 0.999))
        local[partial] = 6
        status[gy, gx] = local
        sampled_x = srx + gx + offset[gy - y0, gx, 0]
        sampled_y = sry + gy + offset[gy - y0, gx, 1]
        horizontal = sign * (fx[inside] - sampled_x)
        vertical_raw = sign * (fy[inside] - sampled_y)
        measured = (local == 1) | (local == 6)
        disparity[gy[measured], gx[measured]] = horizontal[measured]
        vertical[gy[measured], gx[measured]] = vertical_raw[measured]
        blocked = (local == 2) | (local == 3)
        occluder[gy[blocked], gx[blocked]] = dest_identity[blocked]
        distance[gy, gx] = d_source
        depth_delta[gy, gx] = d_source - d_dest
        surface_distance[gy[same_surface], gx[same_surface]] = world_gap[same_surface]
    return {"status": status, "disparity": disparity, "vertical": vertical,
            "occluder": occluder, "dest_pixel": dest_pixel, "distance": distance,
            "depth_delta": depth_delta, "surface_distance": surface_distance}


def _bbox(mask):
    positions = np.argwhere(mask)
    if len(positions) == 0:
        return None
    y0, x0 = positions.min(axis=0)
    y1, x1 = positions.max(axis=0)
    return [int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)]


def _per_object(source, result, node_map, world_unit):
    status = result["status"]
    identity = source["object_id"]
    table = []
    for object_id in sorted(int(value) for value in np.unique(identity) if value > 0):
        mask = identity == object_id
        total = int(mask.sum())
        codes = status[mask]
        trusted = codes == 1
        parallax = result["disparity"][mask][trusted]
        depths = result["distance"][mask][trusted]
        entry = {
            "object_id": object_id,
            "node": node_map.get(str(object_id)),
            "covered_pixels": total,
            "matched_pixels": int(trusted.sum()),
            "matched_fraction": float(trusted.sum() / total) if total else 0.0,
            "occluded_pixels": int((codes == 2).sum()),
            "mismatch_pixels": int((codes == 3).sum()),
            "outside_frame_pixels": int((codes == 5).sum()),
            "no_destination_surface_pixels": int((codes == 4).sum()),
            "partial_coverage_pixels": int((codes == 6).sum()),
            "mean_parallax_pixels": _json_number(parallax.mean()) if len(parallax) else None,
            "min_parallax_pixels": _json_number(parallax.min()) if len(parallax) else None,
            "max_parallax_pixels": _json_number(parallax.max()) if len(parallax) else None,
            "mean_depth_scene_units": _json_number(depths.mean()) if len(depths) else None,
            "mean_depth_meters": _json_number(depths.mean() * UNIT_MM[world_unit] / 1000.0) if len(depths) else None,
        }
        table.append(entry)
    table.sort(key=lambda item: (item["mean_depth_scene_units"] is None, item["mean_depth_scene_units"] or 0.0))
    return table


def _summary(source, result, node_map, rig_result, label):
    status = result["status"]
    height, width = status.shape
    counts = {STATUS[code]: int((status == code).sum()) for code in STATUS}
    trusted = status == TRUSTED
    total = int(trusted.sum())
    disparity = result["disparity"]
    vertical = result["vertical"]
    trusted_values = disparity[trusted & np.isfinite(disparity)]
    vertical_values = vertical[trusted & np.isfinite(vertical)]
    depth_all = result["distance"]
    near_target = trusted & np.isfinite(depth_all) & \
        (np.abs(depth_all / rig_result["convergence"] - 1.0) <= 0.10)
    near_vertical = vertical[near_target & np.isfinite(vertical)]
    depth_values = result["distance"][trusted]
    metrics = {
        "direction": label,
        "pixels": int(height * width),
        "trusted_pixels": total,
        "source_surface_pixels": int((source["object_id"] > 0).sum()),
        "trusted_fraction": float(total / (height * width)) if height * width else 0.0,
        "trusted_fraction_of_source_surface": float(total / max(1, int((source["object_id"] > 0).sum()))),
        "untrusted_pixels": int(height * width - total),
        "status_counts": counts,
        "horizontal_parallax_pixels": {
            "min": _json_number(trusted_values.min()) if len(trusted_values) else None,
            "max": _json_number(trusted_values.max()) if len(trusted_values) else None,
            "mean": _json_number(trusted_values.mean()) if len(trusted_values) else None,
            "std": _json_number(trusted_values.std()) if len(trusted_values) else None,
            "crossed_negative": int((trusted_values < -0.5).sum()),
            "uncrossed_positive": int((trusted_values > 0.5).sum()),
            "near_zero": int((np.abs(trusted_values) <= 0.5).sum()),
        },
        "vertical_parallax_pixels": {
            "max_abs": _json_number(np.abs(vertical_values).max()) if len(vertical_values) else None,
            "mean_abs": _json_number(np.abs(vertical_values).mean()) if len(vertical_values) else None,
            "near_target_max_abs": _json_number(np.abs(near_vertical).max()) if len(near_vertical) else None,
            "tolerance_pixels": 0.01,
            "within_tolerance": bool(len(vertical_values) == 0 or np.abs(vertical_values).max() <= 0.01),
        },
        "depth_scene_units": {
            "min": _json_number(depth_values.min()) if len(depth_values) else None,
            "max": _json_number(depth_values.max()) if len(depth_values) else None,
        },
        "surface_distance_scene_units": {
            "max": _json_number(np.nanmax(result["surface_distance"])) if total else None,
            "mean": _json_number(np.nanmean(result["surface_distance"])) if total else None,
        },
        "per_object": _per_object(source, result, node_map, rig_result["report"]["world_unit"]),
        "parallax_sign_convention": "right_x - left_x; negative crossed (nearer than convergence), positive uncrossed (farther)",
        "vertical_parallax_convention": "right_y - left_y",
    }
    fit = None
    expected_slope = rig_result["focal_px"] * rig_result["baseline"]
    if len(trusted_values) > 2 and np.std(trusted_values) > 0:
        depth_curve = 1.0 / rig_result["convergence"] - 1.0 / np.maximum(depth_values, 1e-9)
        # Only fit when the sampled depth range actually spans a measurable
        # parallax range; a flat scene cannot distinguish a translated camera
        # from a crop and must not claim depth dependence.
        reach = abs(expected_slope) * float(np.std(depth_curve)) * 2.0
        if np.std(depth_curve) > 0 and reach >= 0.25:
            with np.errstate(all="ignore"):
                slope = float(np.polyfit(depth_curve, trusted_values, 1)[0])
            residual = trusted_values - slope * depth_curve
            fit = {
                "observed_slope_px_scene_units": slope,
                "expected_slope_px_scene_units": expected_slope,
                "relative_error": abs(slope - expected_slope) / expected_slope if expected_slope else None,
                "depth_dependent": bool(abs(slope) > 0.5 * abs(expected_slope)),
                "rms_residual_pixels": _json_number(np.sqrt(np.mean(residual ** 2))),
                "depth_span_scene_units": _json_number(depth_values.max() - depth_values.min()),
                "uniform_crop_would_have_zero_slope": True,
            }
    if fit is None:
        fit = {"depth_dependent": False,
               "note": "sampled depth range too flat to separate a translated camera from a crop",
               "expected_slope_px_scene_units": expected_slope,
               "uniform_crop_would_have_zero_slope": True}
    metrics["parallax_depth_fit"] = fit
    regions = []
    for code in (2, 3, 4, 5, 6):
        mask = status == code
        if not mask.any():
            continue
        box = _bbox(mask)
        regions.append({"status": STATUS[code], "count": int(mask.sum()),
                        "bbox_source_region_pixels": box,
                        "bbox_source_full_frame": [box[0] + source["region"][0],
                                                   box[1] + source["region"][1], box[2], box[3]]})
    metrics["regions_without_trusted_correspondence"] = regions
    occluded_by = {}
    for code in (2, 3):
        mask = status == code
        if not mask.any():
            continue
        keys = np.stack([source["object_id"][mask], result["occluder"][mask]], axis=1)
        unique, counts_pair = np.unique(keys, axis=0, return_counts=True)
        table = {}
        for (left_id, right_id), count in zip(unique, counts_pair):
            left_key = node_map.get(str(int(left_id)), str(int(left_id)))
            right_key = node_map.get(str(int(right_id)), str(int(right_id)))
            table.setdefault(left_key, {})[right_key] = int(count)
        occluded_by[STATUS[code]] = table
    metrics["occlusion_pairs"] = occluded_by
    return metrics


def _probes(source, result, source_meta, destination_meta, rig_result):
    status = result["status"]
    trusted = status == TRUSTED
    if not trusted.any():
        return [], {"probe_count": 0, "max_abs_residual_pixels": None, "all_within_tolerance": True}
    distance = result["distance"]
    pool = np.argwhere(trusted)
    pool_depth = distance[trusted]
    chosen, used = [], set()
    for percentile in (5, 20, 35, 50, 65, 80, 95):
        target_depth = float(np.percentile(pool_depth, percentile))
        for index in np.argsort(np.abs(pool_depth - target_depth)):
            index = int(index)
            if index in used:
                continue
            used.add(index)
            chosen.append((int(pool[index][1]), int(pool[index][0])))
            break
    forward = np.asarray(rig_result["forward"], dtype=np.float64)
    center_eye = np.asarray(rig_result["center_eye"], dtype=np.float64)
    matrix = destination_meta["projection"] @ destination_meta["view"]
    probes = []
    for x, y in chosen:
        point = source["world"][y, x]
        depth = float(point @ forward - center_eye @ forward)
        full = [source_meta["region"][0] + x + float(source["sample_offset"][y, x, 0]),
                source_meta["region"][1] + y + float(source["sample_offset"][y, x, 1])]
        clip = matrix @ np.append(point, 1.0)
        ndc = clip[:3] / clip[3]
        px = float((ndc[0] + 1) * source_meta["width"] / 2)
        py = float((1 - ndc[1]) * source_meta["height"] / 2)
        dx = int(np.floor(px) - destination_meta["region"][0])
        dy = int(np.floor(py) - destination_meta["region"][1])
        inside = 0 <= dx < destination_meta["region"][2] and 0 <= dy < destination_meta["region"][3]
        measured = result["disparity"][y, x]
        expected = rig_result["focal_px"] * rig_result["baseline"] * \
            (1.0 / rig_result["convergence"] - 1.0 / depth)
        probes.append({
            "source_pixel": [x, y],
            "source_full_sample": full,
            "source_node": source_meta["node_map"].get(str(int(source["object_id"][y, x]))),
            "source_world_scene_units": point.tolist(),
            "source_depth_scene_units": depth,
            "destination_pixel": [dx, dy] if inside else None,
            "destination_full_pixel": [px, py],
            "destination_node": destination_meta["node_map"].get(
                str(int(destination_meta["object_id"][dy, dx]))) if inside else None,
            "measured_parallax_pixels": _json_number(measured),
            "expected_parallax_pixels": expected,
            "residual_pixels": _json_number(measured - expected) if np.isfinite(measured) else None,
            "vertical_parallax_pixels": _json_number(result["vertical"][y, x]),
            "status": STATUS[int(status[y, x])],
            "trusted": bool(status[y, x] == TRUSTED),
        })
    residuals = [abs(item["residual_pixels"]) for item in probes if item["residual_pixels"] is not None]
    return probes, {"probe_count": len(probes),
                    "max_abs_residual_pixels": _json_number(max(residuals)) if residuals else None,
                    "all_within_tolerance": bool(not residuals or max(residuals) <= 0.01)}


def analyze_stereo(eyes, rig_result):
    """Correspondence and occlusion analysis in both directions."""
    attach_scale(rig_result, eyes["left"]["metadata"]["height"])
    sources = {"left": eye_arrays(eyes["left"]["channels"], eyes["left"]["metadata"], eyes["left"]["node_map"]),
               "right": eye_arrays(eyes["right"]["channels"], eyes["right"]["metadata"], eyes["right"]["node_map"])}
    forward_results = {}
    summaries = {}
    probes = {}
    probe_checks = {}
    for label, source_name, destination_name, sign in (
            ("left_to_right", "left", "right", 1.0),
            ("right_to_left", "right", "left", -1.0)):
        source = sources[source_name]
        destination = sources[destination_name]
        result = _project_into(source, destination, rig_result, sign)
        forward_results[label] = result
        summaries[label] = _summary(source, result, source["node_map"], rig_result, label)
        probes[label], probe_checks[label] = _probes(source, result, source, destination, rig_result)
    return {"directions": forward_results, "summaries": summaries,
            "probes": probes, "probe_checks": probe_checks}


def _color_mask(status):
    palette = np.zeros((*status.shape, 3), dtype=np.uint8)
    for code, color in STATUS_COLORS.items():
        palette[status == code] = color
    return palette


def _parallax_preview(disparity, status):
    valid = np.isfinite(disparity)
    scale = float(np.nanmax(np.abs(disparity))) if valid.any() else 1.0
    scale = max(scale, 1e-9)
    normalized = np.zeros_like(disparity)
    normalized[valid] = np.clip(disparity[valid] / scale, -1, 1)
    red = np.where(normalized > 0, normalized, 0)
    blue = np.where(normalized < 0, -normalized, 0)
    grey = 1 - np.abs(normalized)
    image = np.stack([red * 0.92 + grey * 0.25, grey * 0.25, blue * 0.92 + grey * 0.25], axis=2)
    image[~valid & (status != 0)] = (0.42, 0.42, 0.42)
    return np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8), scale


def _pair_image(left_rgb, right_rgb, report, width, height):
    canvas = Image.new("RGB", (width * 2 + PAIR_GAP, height + PAIR_HEADER), (18, 23, 33))
    canvas.paste(Image.fromarray(left_rgb), (0, PAIR_HEADER))
    canvas.paste(Image.fromarray(right_rgb), (width + PAIR_GAP, PAIR_HEADER))
    draw = ImageDraw.Draw(canvas)
    baseline = report["interocular"]
    convergence = report["convergence"]
    left_lines = [
        "LEFT",
        f"baseline {baseline['value']:g} {baseline['unit']}",
        f"= {baseline['meters']:.4f} m",
        f"convergence {convergence['meters']:.3f} m",
    ]
    right_lines = [
        "RIGHT",
        "parallel off-axis",
        "no toe-in",
        "vertical parallax 0 px",
    ]
    def fit(text, limit):
        while text and draw.textlength(text) > limit:
            text = text[:-1]
        return text
    limit = width - 10
    for index, text in enumerate(left_lines):
        draw.text((5, 3 + index * 10), fit(text, limit), fill=(211, 224, 238))
    for index, text in enumerate(right_lines):
        draw.text((width + PAIR_GAP + 5, 3 + index * 10), fit(text, limit), fill=(211, 224, 238))
    return png_bytes(np.asarray(canvas))


def _anaglyph(left_linear, right_linear):
    left = linear_to_srgb(left_linear)
    right = linear_to_srgb(right_linear)
    image = np.stack([left[:, :, 0], right[:, :, 1], right[:, :, 2]], axis=2)
    return png_bytes(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8))


def _pack(channels):
    return packed_channels({key: value for key, value in channels.items() if not key.startswith("_")})


def _depth_probe_image(left_rgb, probes, report):
    """Mark analytic depth probes on the left eye so objects at different
    depths can be compared directly on the delivered pair."""
    height, width = left_rgb.shape[:2]
    columns = 2
    rows = max(1, (len(probes) + columns - 1) // columns)
    line = 11
    header = 32 + rows * line
    canvas = Image.new("RGB", (width, height + header), (18, 23, 33))
    canvas.paste(Image.fromarray(left_rgb), (0, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 3), f"PROBES  unit {report['world_unit']}",
              fill=(211, 224, 238))
    draw.text((5, 14), "px  measured/expected  depth", fill=(150, 168, 190))
    legend = {}
    for index, probe in enumerate(probes, 1):
        x, y = probe["source_pixel"]
        color = (255, 224, 110) if probe["trusted"] else (222, 70, 70)
        draw.ellipse([x - 4, y + header - 4, x + 4, y + header + 4], outline=color, width=1)
        draw.line([x - 7, y + header, x + 7, y + header], fill=color)
        draw.line([x, y + header - 7, x, y + header + 7], fill=color)
        measured = probe["measured_parallax_pixels"]
        expected = probe["expected_parallax_pixels"]
        text = "n/a" if measured is None else f"{measured:+.2f}"
        column, row = (index - 1) % columns, (index - 1) // columns
        short = "n/a" if measured is None else f"{measured + 0.0:+.2f}"
        label = f"{index} {short}/{expected:+.2f} {probe['source_depth_scene_units']:.2f}{report['world_unit']}"
        draw.text((5 + column * (width // columns + 8), 32 + row * line), label, fill=color)
        legend[index] = {"source_pixel": [x, y], "node": probe["source_node"],
                         "depth_scene_units": probe["source_depth_scene_units"],
                         "measured_parallax_pixels": measured,
                         "expected_parallax_pixels": expected,
                         "status": probe["status"]}
    return png_bytes(np.asarray(canvas)), legend


def _blend_mismatch(left_rgb, right_rgb, status):
    overlay = left_rgb.copy()
    mask = (status == 2) | (status == 3)
    tinted = np.rint(left_rgb.astype(np.float64) * 0.35 + np.array((222, 70, 70)) * 0.65).astype(np.uint8)
    overlay[mask] = tinted[mask]
    return overlay


def stereo_artifacts(eyes, analysis, rig_result):
    left_color = preview_channels(eyes["left"]["channels"])["color.png"]
    right_color = preview_channels(eyes["right"]["channels"])["color.png"]
    left_rgb = np.asarray(Image.open(BytesIO(left_color)).convert("RGB"))
    right_rgb = np.asarray(Image.open(BytesIO(right_color)).convert("RGB"))
    forward = analysis["directions"]["left_to_right"]
    correspondence = _color_mask(forward["status"])
    parallax, scale = _parallax_preview(forward["disparity"], forward["status"])
    reverse = analysis["directions"]["right_to_left"]
    maps = {}
    for label, result in (("left_to_right", forward), ("right_to_left", reverse)):
        maps[f"{label}_status"] = result["status"]
        maps[f"{label}_parallax_pixels"] = result["disparity"]
        maps[f"{label}_vertical_pixels"] = result["vertical"]
        maps[f"{label}_destination_pixel"] = result["dest_pixel"]
        maps[f"{label}_distance_scene_units"] = result["distance"]
        maps[f"{label}_occluder_object_id"] = result["occluder"]
    width = eyes["left"]["metadata"]["width"]
    height = eyes["left"]["metadata"]["height"]
    probe_image, probe_legend = _depth_probe_image(
        left_rgb, analysis["probes"]["left_to_right"], rig_result["report"])
    return {
        "left-color.png": left_color,
        "right-color.png": right_color,
        "right-channels.npz": _pack(eyes["right"]["channels"]),
        "pair.png": _pair_image(left_rgb, right_rgb, rig_result["report"], width, height),
        "anaglyph.png": _anaglyph(eyes["left"]["channels"]["color"], eyes["right"]["channels"]["color"]),
        "correspondence.png": png_bytes(_blend_mismatch(left_rgb, right_rgb, forward["status"])),
        "occlusion-mask.png": png_bytes(correspondence),
        "parallax.png": png_bytes(parallax),
        "depth-probes.png": probe_image,
        "correspondence.npz": packed_channels(maps),
    }, {"parallax_preview_scale_pixels": scale, "depth_probe_legend": probe_legend}


def crosscheck_material(rig_result, checks, analysis, eyes, stereo_metadata):
    report = rig_result["report"]
    probe_checks = analysis["probe_checks"]
    return {
        "format": "gnomon-stereo-crosscheck/1",
        "units": stereo_metadata["units"],
        "rig_invariants": checks,
        "camera_translation_not_crop": {
            "explanation": "Both eyes come from their own translated world-space camera; no rendered image is cropped, scaled or shifted after the fact.",
            "rotation_delta_degrees": checks["rotation_delta_degrees"],
            "baseline_vector_scene_units": report["baseline_vector_scene_units"],
            "expected_parallax_slope_px_scene_units": analysis["summaries"]["left_to_right"]["parallax_depth_fit"].get("expected_slope_px_scene_units"),
            "observed_parallax_slope_px_scene_units": analysis["summaries"]["left_to_right"]["parallax_depth_fit"].get("observed_slope_px_scene_units"),
            "uniform_crop_would_have_zero_slope": True,
        },
        "analytic_probes": {
            "left_to_right": analysis["probes"]["left_to_right"],
            "right_to_left": analysis["probes"]["right_to_left"],
        },
        "probe_checks": probe_checks,
        "directions": analysis["summaries"],
        "status_legend": STATUS,
        "trust_rule": TRUST_RULE,
        "eyes": {name: {"eye_scene_units": eyes[name]["metadata"]["lens"]["eye"],
                        "shift_ndc": eyes[name]["metadata"]["lens"]["shift"],
                        "view": eyes[name]["view"], "projection": eyes[name]["projection"],
                        "region": eyes[name]["metadata"]["region"],
                        "width": eyes[name]["metadata"]["width"],
                        "height": eyes[name]["metadata"]["height"],
                        "node_map": eyes[name]["node_map"]}
                 for name in ("left", "right")},
    }


def correspondence_at(side, x, y, source, destination, rig_result):
    """Answer one pixel query with a full two-eye correspondence record."""
    height, width = source["depth"].shape
    source_meta, destination_meta = source, destination
    require(0 <= x < width and 0 <= y < height, "pixel_bounds", "pixel outside output region")
    require(side in {"left", "right"}, "stereo_side", "side must be left or right")
    identity = int(source["object_id"][y, x])
    opposite = "right" if side == "left" else "left"
    base = {
        "side": side,
        "pixel": [x, y],
        "full_sample": [float(source_meta["region"][0] + x + source["sample_offset"][y, x, 0]),
                        float(source_meta["region"][1] + y + source["sample_offset"][y, x, 1])],
        "coverage": float(source["coverage"][y, x]),
        "object_id": identity,
        "node": source_meta["node_map"].get(str(identity)),
        "world": None if identity == 0 else source["world"][y, x].tolist(),
        "parallax_pixels": None,
        "expected_parallax_pixels": None,
        "counterpart": None,
        "status": "no_source_surface",
        "trusted": False,
        "trust_rule": TRUST_RULE,
    }
    if identity == 0:
        base["note"] = "no surface at this pixel; there is no correspondence to query"
        return base
    point = np.asarray(source["world"][y, x], dtype=np.float64)
    matrix = destination_meta["projection"] @ destination_meta["view"]
    forward = np.asarray(rig_result["forward"], dtype=np.float64)
    center_eye = np.asarray(rig_result["center_eye"], dtype=np.float64)
    clip = matrix @ np.append(point, 1.0)
    require(clip[3] > 1e-12, "projection_w", "surface point projects behind the other eye")
    ndc = clip[:3] / clip[3]
    px = float((ndc[0] + 1) * source_meta["width"] / 2)
    py = float((1 - ndc[1]) * source_meta["height"] / 2)
    drx, dry, drw, drh = destination_meta["region"]
    depth = float(point @ forward - center_eye @ forward)
    expected = rig_result["focal_px"] * rig_result["baseline"] * \
        (1.0 / rig_result["convergence"] - 1.0 / depth)
    dx = int(np.floor(px) - drx)
    dy = int(np.floor(py) - dry)
    base["expected_parallax_pixels"] = expected
    base["destination_full_pixel"] = [px, py]
    if not (0 <= dx < drw and 0 <= dy < drh):
        base["status"] = "outside_destination_frame"
        base["note"] = "the projected surface leaves the other eye's frame"
        return base
    dest_identity = int(destination["object_id"][dy, dx])
    dest_eye = np.asarray(destination_meta["lens"]["eye"], dtype=np.float64)
    dest_point = None if dest_identity == 0 else np.asarray(destination["world"][dy, dx], dtype=np.float64)
    sampled_x = source_meta["region"][0] + x + float(source["sample_offset"][y, x, 0])
    sampled_y = source_meta["region"][1] + y + float(source["sample_offset"][y, x, 1])
    sign = 1.0 if side == "left" else -1.0
    horizontal = sign * (px - sampled_x)
    vertical = sign * (py - sampled_y)
    base["parallax_pixels"] = {"horizontal": horizontal, "vertical": vertical}
    base["counterpart"] = {
        "side": opposite,
        "pixel": [dx, dy],
        "full_sample": [float(drx + dx + destination["sample_offset"][dy, dx, 0]),
                        float(dry + dy + destination["sample_offset"][dy, dx, 1])],
        "coverage": float(destination["coverage"][dy, dx]),
        "object_id": dest_identity,
        "node": destination_meta["node_map"].get(str(dest_identity)),
        "world": None if dest_point is None else dest_point.tolist(),
        "distance_scene_units": None if dest_point is None else float(dest_point @ forward - dest_eye @ forward),
    }
    if dest_identity == 0:
        base["status"] = "no_destination_surface"
        base["note"] = "the projected surface falls where the other eye sees background"
        return base
    d_dest = float(dest_point @ forward - center_eye @ forward)
    tan_half = np.tan(np.radians(rig_result["fov"]) / 2)
    footprint = 2 * max(abs(d_dest), 1e-9) * tan_half / destination_meta["height"]
    code = _classify(depth, d_dest, identity, dest_identity, point, dest_point, footprint)
    if code == 2:
        base["status"] = "occluded_in_other_eye"
        base["note"] = "a nearer surface is stored in the other eye at the projected pixel"
        return base
    if code == 3:
        base["status"] = "surface_mismatch"
        base["note"] = "the other eye stores a different surface (node, depth or world point) at the projected pixel"
        return base
    if source["coverage"][y, x] < 0.999 or destination["coverage"][dy, dx] < 0.999:
        base["status"] = "partial_coverage"
        base["note"] = "matched surface, but at least one pixel is a silhouette with partial sample coverage"
        return base
    base["status"] = "matched"
    base["trusted"] = True
    return base


def rig_from_metadata(metadata):
    """Rebuild the minimal rig result needed for a correspondence query."""
    report = metadata["stereo"]
    return {
        "forward": np.asarray(report["center_forward"], dtype=np.float64),
        "center_eye": np.asarray(report["baseline_midpoint_scene_units"], dtype=np.float64),
        "side": np.asarray(report["center_side"], dtype=np.float64),
        "baseline": report["interocular"]["scene_units"],
        "convergence": report["convergence"]["scene_units"],
        "focal_px": report["focal_px"],
        "fov": report["fov_degrees"],
    }
