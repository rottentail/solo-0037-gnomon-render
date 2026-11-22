"""Geometry-anchored contour line layer.

The line layer is derived from *projected mesh edges*, never from texture or
shaded color, so decorative patterns never become geometric lines.  Each edge
is classified as one of three independently deliverable structures:

* ``silhouette`` — outer contour separating geometry from background
  (includes open-mesh boundary edges);
* ``occlusion``  — internal boundary where one object hides a different
  object at a depth discontinuity;
* ``crease``     — internal fold where two triangles of the same object meet
  at a dihedral angle above the authored threshold.

Visibility is resolved per sub-pixel sample against the exact depth and
object-id buffers produced by the same render pass.  A sample on an edge is
kept only when that edge is the nearest surface, which prevents a small
foreground object from being joined to a contour lying behind it.

All geometry is projected with the full-frame camera and full-frame pixel
dimensions.  When only a region is requested, the resulting line masks are
sliced out of the full-frame grid afterwards; therefore a line occupies the
same full-frame coordinate in a crop as in the whole image, and rendering at
a larger output size only changes rasterization resolution, not where the
projected edge sits relative to the original composition.  Stroke widths are
applied in output pixels, independent of the sub-sample density.
"""
import math
import numpy as np
from .camera import camera_matrices

# NDC depth bias when comparing an edge sample with the z-buffer.  Edges sit
# exactly on their own surface, so the bias only rejects genuinely nearer
# surfaces; depth_gap additionally separates merely touching geometry.
EDGE_DEPTH_BIAS = 1e-4

LINE_KINDS = ("silhouette", "occlusion", "crease")


def _edge_faces(objects):
    """Map every undirected edge to the triangles that share it, per object.

    Also returns geometric face normals (area-weighted cross products in
    world space) so crease detection reflects actual surface folding rather
    than authored smooth vertex normals, and stays correct under non-uniform
    scale because the compiled scene already carries transformed normals.
    """
    result, face_normals = [], []
    for obj in objects:
        faces = np.asarray(obj["triangles"])
        world = np.asarray(obj["positions"])
        mapping = {}
        for face_index, face in enumerate(faces):
            for corner in range(3):
                a, b = int(face[corner]), int(face[(corner + 1) % 3])
                key = (a, b) if a < b else (b, a)
                mapping.setdefault(key, []).append(face_index)
        result.append(mapping)
        triangles = world[faces]
        geometric = np.cross(triangles[:, 1] - triangles[:, 0],
                             triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(geometric, axis=1, keepdims=True)
        face_normals.append(geometric / np.maximum(lengths, 1e-30))
    return result, face_normals


def _project_edge(a, b, width, height):
    """Project two clip-space vertices to full-frame sub-pixel coordinates."""
    if a[3] <= 1e-12 or b[3] <= 1e-12:
        return None
    ndc_a, ndc_b = a[:3] / a[3], b[:3] / b[3]
    sa = np.array([(ndc_a[0] + 1) * width / 2, (1 - ndc_a[1]) * height / 2])
    sb = np.array([(ndc_b[0] + 1) * width / 2, (1 - ndc_b[1]) * height / 2])
    if not np.isfinite(sa).all() or not np.isfinite(sb).all():
        return None
    return sa, sb, ndc_a[2], ndc_b[2]


def _clip_2d(start, end, bounds):
    """Liang-Barsky clip of a 2-D segment to the sub-pixel viewport rectangle."""
    x0, y0 = start
    x1, y1 = end
    bx0, by0, bx1, by1 = bounds
    dx, dy = x1 - x0, y1 - y0
    t_lo, t_hi = 0.0, 1.0
    for p, q in ((-dx, x0 - bx0), (dx, bx1 - x0), (-dy, y0 - by0), (dy, by1 - y0)):
        if abs(p) < 1e-15:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            t_lo = max(t_lo, t)
        else:
            t_hi = min(t_hi, t)
    if t_lo > t_hi:
        return None
    return (np.array([x0 + dx * t_lo, y0 + dy * t_lo]),
            np.array([x0 + dx * t_hi, y0 + dy * t_hi]), t_lo, t_hi)


def _raster_segment(start, end, bounds):
    """Sub-pixel grid cells touched by a segment, plus a per-cell t in [0,1].

    A 2-D DDA in grid-cell units gives one entry per crossed cell so a long
    edge stays cheap while its centerline remains anchored to the true
    projected segment rather than to pixel-center approximations.
    """
    x0, y0 = start
    x1, y1 = end
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length < 1e-12:
        return np.empty((0, 2), dtype=np.int64), np.empty(0)
    ux, uy = dx / length, dy / length
    steps = max(int(math.ceil(length)) + 1, 2)
    ts = np.linspace(0.0, 1.0, steps)
    xs = x0 + dx * ts
    ys = y0 + dy * ts
    cells = np.unique(np.floor(np.column_stack([xs, ys])).astype(np.int64), axis=0)
    if bounds is not None:
        bx0, by0, bx1, by1 = bounds
        keep = (cells[:, 0] >= bx0) & (cells[:, 0] < bx1) & (cells[:, 1] >= by0) & (cells[:, 1] < by1)
        cells = cells[keep]
    if not len(cells):
        return cells, np.empty(0)
    # Parameter of each kept cell center, used for interpolating edge depth.
    cx, cy = cells[:, 0] + 0.5, cells[:, 1] + 0.5
    t = np.clip(((cx - x0) * ux + (cy - y0) * uy) / length, 0.0, 1.0)
    return cells, t


def _distance_squared(mask):
    """Approximate squared Euclidean distance (sub-pixel cells) to nearest True.

    Fully vectorized two-pass chamfer transform (Borgefors 3-4 weights over a
    3x3 neighborhood, accumulated in squared units).  The relative distance
    error is under 2%, far below the output-pixel stroke width it drives.
    An all-False grid has no nearest centerline and returns infinities.
    """
    if not mask.any():
        return np.full(mask.shape, np.inf, dtype=np.float64)
    sentinel = float(mask.shape[0] ** 2 + mask.shape[1] ** 2 + 1)
    work = np.where(mask, 0.0, sentinel)
    one, diag = 1.0, 2.0  # squared weights for unit and diagonal steps
    kernels = [(-1, -1, diag), (-1, 0, one), (-1, 1, diag), (0, -1, one),
               (1, 1, diag), (1, 0, one), (1, -1, diag), (0, 1, one)]
    for dr, dc, weight in kernels:
        shifted = np.full_like(work, sentinel)
        src_r = slice(max(-dr, 0), work.shape[0] - max(dr, 0))
        dst_r = slice(max(dr, 0), work.shape[0] - max(-dr, 0))
        src_c = slice(max(-dc, 0), work.shape[1] - max(dc, 0))
        dst_c = slice(max(dc, 0), work.shape[1] - max(-dc, 0))
        shifted[dst_r, dst_c] = work[src_r, src_c] + weight
        np.minimum(work, shifted, out=work)
    work[work >= sentinel] = np.inf
    return work


def build_outline(compiled, buffers, request, options, buffer_region=None):
    """Return per-kind sub-pixel centerline masks and pixel-space coverages.

    The line grid is always built at *full-frame* sub-pixel resolution; a
    region request simply slices the resulting coverage.  This mirrors the
    color channel's region invariant and guarantees a line keeps the exact
    same full-frame coordinate in a crop as in the whole image, including
    strokes whose width crosses the crop boundary.

    ``buffers`` may be region-sized (the normal render path); in that case
    ``buffer_region`` states which full-frame rectangle they cover.  Passing
    full-frame buffers with ``buffer_region=(0,0,width,height)`` lets a region
    request still resolve edge visibility across the whole frame.
    """
    width, height, samples = request["width"], request["height"], request["samples"]
    region = request.get("region") or (0, 0, width, height)
    rx, ry, rw, rh = region
    brx, bry, brw, brh = buffer_region or region
    fsh, fsw = height * samples, width * samples
    view, projection = camera_matrices(compiled["scene"]["lens"], width / height)
    depth = buffers["depth"]
    object_id = buffers["object_id"]
    centerlines = {kind: np.zeros((fsh, fsw), dtype=bool) for kind in LINE_KINDS}
    edge_tables, normal_tables = _edge_faces(compiled["objects"])

    def classify_cells(kind, outward, gx, gy, z_edge, obj_index):
        """Vectorized classification of all sub-pixel cells of one edge."""
        gx = np.asarray(gx, dtype=np.int64)
        gy = np.asarray(gy, dtype=np.int64)
        keep = (gx >= 0) & (gx < fsw) & (gy >= 0) & (gy < fsh)
        gx, gy, z_edge = gx[keep], gy[keep], np.asarray(z_edge)[keep]
        if not len(gx):
            return
        # Full-frame -> supplied buffer coordinates.
        lx = gx - brx * samples
        ly = gy - bry * samples
        keep = (lx >= 0) & (lx < brw * samples) & (ly >= 0) & (ly < brh * samples)
        gx, gy, lx, ly, z_edge = gx[keep], gy[keep], lx[keep], ly[keep], z_edge[keep]
        if not len(gx):
            return
        stored = depth[ly, lx]
        # Hidden behind a nearer surface (bias guards against self z-fighting).
        visible = ~(np.isfinite(stored) & (z_edge > stored + EDGE_DEPTH_BIAS))
        gx, gy, lx, ly, z_edge, stored = (a[visible] for a in (gx, gy, lx, ly, z_edge, stored))
        if not len(gx):
            return
        if kind == "crease":
            if options.crease:
                centerlines["crease"][gy, gx] = True
            return
        # Boundary/silhouette edge: inspect the outward neighbour sample.
        nx = np.rint(gx + outward[0]).astype(np.int64)
        ny = np.rint(gy + outward[1]).astype(np.int64)
        nlx, nly = nx - brx * samples, ny - bry * samples
        in_buf = (nlx >= 0) & (nlx < brw * samples) & (nly >= 0) & (nly < brh * samples)
        outside_id = np.zeros(len(gx), dtype=np.int32)
        outside_depth = np.full(len(gx), np.inf)
        outside_id[in_buf] = object_id[nly[in_buf], nlx[in_buf]]
        outside_depth[in_buf] = depth[nly[in_buf], nlx[in_buf]]
        to_background = outside_id == 0
        to_other = (outside_id != obj_index) & (outside_id != 0)
        depth_separated = np.isfinite(outside_depth) & (np.abs(z_edge - outside_depth) >= options.depth_gap)
        if options.silhouette:
            sil = to_background
            # Occlusion already claimed here: keep the masks disjoint.
            sil &= ~centerlines["occlusion"][gy, gx]
            centerlines["silhouette"][gy[sil], gx[sil]] = True
        if options.occlusion:
            occ = to_other & depth_separated
            centerlines["silhouette"][gy[occ], gx[occ]] = False
            centerlines["occlusion"][gy[occ], gx[occ]] = True

    full_bounds = (0, 0, fsw, fsh)
    eye = np.asarray(compiled["scene"]["lens"]["eye"], dtype=np.float64)
    for obj, table, geometric_normals in zip(compiled["objects"], edge_tables, normal_tables):
        obj_index = int(obj["id"])
        positions = obj["positions"]
        homogeneous = np.column_stack([positions, np.ones(len(positions))])
        clip = homogeneous @ (projection @ view).T
        faces = np.asarray(obj["triangles"])
        # Per-face orientation toward the camera in world space.  A face is
        # front-facing when its geometric normal points toward the eye.
        face_centers = positions[faces].mean(axis=1)
        view_vecs = eye - face_centers
        view_vecs /= np.maximum(np.linalg.norm(view_vecs, axis=1, keepdims=True), 1e-30)
        facing = np.sign(np.einsum("ij,ij->i", geometric_normals, view_vecs))
        # Classify every original mesh edge once from topology, then project
        # the edge directly. 2-D viewport clipping keeps on-screen fragments,
        # and the depth test rejects anything behind the near plane or
        # occluded by another face.
        for (a, b), shared in table.items():
            outward = np.zeros(2)
            if len(shared) == 1:
                # Open-mesh boundary: always an outer-contour candidate.
                kind = "silhouette"
                face = faces[shared[0]]
                third = next(int(v) for v in face if v not in (a, b))
            else:
                f0, f1 = shared
                if facing[f0] * facing[f1] < 0:
                    # A front/back transition is the view-dependent silhouette
                    # of a closed solid; it takes priority over any crease
                    # label so the occluding outline stays a boundary line.
                    kind = "silhouette"
                    front_face = f0 if facing[f0] > 0 else f1
                    face = faces[front_face]
                    third = next(int(v) for v in face if v not in (a, b))
                else:
                    n0, n1 = geometric_normals[f0], geometric_normals[f1]
                    cosine = float(np.clip(np.dot(n0, n1), -1, 1))
                    if math.degrees(math.acos(cosine)) < options.crease_angle:
                        continue  # Smooth interior edge: shading, not a line.
                    if not options.crease:
                        continue  # Author chose not to retain internal folds.
                    kind = "crease"  # Geometric fold between two visible faces.
            if kind == "silhouette" and not (options.silhouette or options.occlusion):
                continue
            projected = _project_edge(clip[a], clip[b], fsw, fsh)
            if projected is None:
                continue
            sa, sb, za, zb = projected
            if kind == "silhouette":
                # Outward screen normal points away from the emitting face,
                # i.e. opposite the direction toward the face's third vertex.
                st = _project_edge(clip[third], clip[third], fsw, fsh)
                if st is not None:
                    edge_vec = sb - sa
                    normal = np.array([-edge_vec[1], edge_vec[0]])
                    nlen = np.linalg.norm(normal)
                    if nlen > 1e-12:
                        normal /= nlen
                        if np.dot(st[0] - (sa + sb) / 2, normal) > 0:
                            normal = -normal
                        outward = normal
            clipped = _clip_2d(sa, sb, full_bounds)
            if clipped is None:
                continue
            ca, cb, t_lo, t_hi = clipped
            cells, t = _raster_segment(ca, cb, full_bounds)
            if not len(cells):
                continue
            # Map local segment parameters back onto the full edge for depth.
            z_edge = za + (zb - za) * (t_lo + (t_hi - t_lo) * t)
            classify_cells(kind, outward, cells[:, 0], cells[:, 1], z_edge, obj_index)

    full_coverages = {}
    for kind in LINE_KINDS:
        radius_px = getattr(options, f"width_{kind}") / 2.0
        radius_cells = radius_px * samples
        # Coverage is computed on the full-frame grid so a crop is the exact
        # sub-array of the whole-frame line image, including strokes that
        # cross the crop edge.
        squared = _distance_squared(centerlines[kind])
        distances = np.sqrt(squared, out=np.full_like(squared, 1e30), where=np.isfinite(squared))
        sub_coverage = np.clip(radius_cells + 0.5 - distances, 0.0, 1.0)
        # Reduce full-frame sub-pixel coverage to output-pixel coverage.
        full_coverages[kind] = sub_coverage.reshape(height, samples, width, samples).mean(axis=(1, 3))
    coverages = {kind: full_coverages[kind][ry:ry + rh, rx:rx + rw] for kind in LINE_KINDS}
    return centerlines, coverages


def compose_line_layer(coverages, options, background_color):
    """Alpha-composite the enabled line kinds over a transparent RGBA layer."""
    h, w = coverages["silhouette"].shape
    rgba = np.zeros((h, w, 4), dtype=np.float64)
    rgba[:, :, :3] = background_color
    for kind in LINE_KINDS:
        if not getattr(options, kind):
            continue
        color = np.asarray(getattr(options, f"color_{kind}"), dtype=np.float64)
        alpha = coverages[kind]
        rgba[:, :, :3] = rgba[:, :, :3] * (1 - alpha[:, :, None]) + color * alpha[:, :, None]
        rgba[:, :, 3] = np.maximum(rgba[:, :, 3], alpha)
    return rgba


def overlay_on_color(coverages, options, linear_color):
    """Composite enabled lines over the rendered linear color (same framing)."""
    composed = compose_line_layer(coverages, options, np.zeros(3))
    alpha = composed[:, :, 3:4]
    return linear_color * (1 - alpha) + composed[:, :, :3] * alpha


def line_artifacts(coverages, options, linear_color):
    """Build the independently deliverable line-layer files.

    Returns the three per-structure grayscale masks, a transparent combined
    RGBA layer (authors' enabled structures only), the overlay composite over
    the original linear color, and a packed NPZ of fractional coverages.
    """
    from .channels import mask_png_bytes, rgba_png_bytes, png_bytes, packed_channels
    from .color import encode_rgb
    files = {}
    for kind in LINE_KINDS:
        files[f"line-{kind}.png"] = mask_png_bytes(coverages[kind])
    layer = compose_line_layer(coverages, options, np.zeros(3))
    files["line-layer.png"] = rgba_png_bytes(layer)
    overlay = overlay_on_color(coverages, options, linear_color)
    files["line-overlay.png"] = png_bytes(encode_rgb(overlay))
    npz_channels = {f"line_{kind}": coverages[kind] for kind in LINE_KINDS}
    npz_channels["line_layer_rgba_linear"] = layer
    files["lines.npz"] = packed_channels(npz_channels)
    return files, layer, overlay
