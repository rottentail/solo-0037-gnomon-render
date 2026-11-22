"""World-plane sectioning: half-space clipping, welded cap generation, sealing report.

Caps share their boundary vertices with the kept geometry (per-edge intersection
cache plus on-plane vertex snapping), so adjacent faces meet the cut without
cracks. Closed loops become triangulated caps (holes supported); chains that
cannot close are reported as uncertain instead of being capped silently.
"""
import numpy as np
from .errors import require
from .math3d import unit

SNAP = 1e-9
AREA_EPS = 1e-12


def prepare_section(payload):
    plane = payload["plane"]
    normal = unit(np.asarray(plane["normal"], dtype=np.float64))
    offset = float(plane["offset"])
    require(abs(offset) <= 1e6, "section_plane", "plane offset outside supported coordinates")
    cap = payload["cap"]
    return {"label": payload["label"],
            "plane": {"normal": normal.tolist(), "offset": offset, "keep": plane["keep"]},
            "cap": {"color": list(cap["color"]), "unlit": bool(cap["unlit"]),
                    "double_sided": bool(cap["double_sided"])}}


def cap_surface(cap):
    return {"color": tuple(cap["color"]), "texture": None, "wrap": "repeat",
            "alpha_cutoff": 0.5, "unlit": cap["unlit"], "double_sided": cap["double_sided"]}


def plane_basis(normal):
    # Right-handed (u, v, normal) with orthonormal u, v spanning the plane.
    axis = int(np.argmin(np.abs(normal)))
    other = np.zeros(3)
    other[axis] = 1.0
    u = unit(np.cross(normal, other))
    return u, np.cross(normal, u)


def _signed_area(points):
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5*np.sum(x*np.roll(y, -1)-np.roll(x, -1)*y))


def _contains(loop, point):
    # Ray casting in 2D; loops of one cross-section never cross each other.
    x, y = point
    inside = False
    for i in range(len(loop)):
        x1, y1 = loop[i]
        x2, y2 = loop[(i+1) % len(loop)]
        if (y1 > y) != (y2 > y):
            if x1+(x2-x1)*(y-y1)/(y2-y1) > x:
                inside = not inside
    return inside


def _segments_cross(a, b, c, d):
    # Proper intersection: segments cross in their interiors (shared endpoints allowed).
    def orient(p, q, r):
        return (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
    return (orient(a, b, c)*orient(a, b, d) < -AREA_EPS
            and orient(c, d, a)*orient(c, d, b) < -AREA_EPS)


def _bridge(merged, hole, pending, pts):
    # Splice a CW hole into the CCW merged polygon through a visible vertex pair.
    hi = max(range(len(hole)), key=lambda i: (pts[hole[i]][0], -pts[hole[i]][1]))
    h = pts[hole[hi]]
    edges = [(merged[i], merged[(i+1) % len(merged)]) for i in range(len(merged))]
    edges += [(hole[i], hole[(i+1) % len(hole)]) for i in range(len(hole))]
    for other in pending:
        edges += [(other[i], other[(i+1) % len(other)]) for i in range(len(other))]
    best, best_dist = None, np.inf
    for mi, m in enumerate(merged):
        if m in hole:
            continue
        dist = float(np.hypot(*(h-pts[m])))
        if dist >= best_dist:
            continue
        if any(m not in (a, b) and hole[hi] not in (a, b)
               and _segments_cross(h, pts[m], pts[a], pts[b]) for a, b in edges):
            continue
        best, best_dist = mi, dist
    if best is None:  # Pathological nesting; nearest vertex keeps output deterministic.
        best = min((mi for mi, m in enumerate(merged) if m not in hole),
                   key=lambda mi: float(np.hypot(*(h-pts[merged[mi]]))))
    ordered = [hole[(hi+k) % len(hole)] for k in range(len(hole))]
    return merged[:best+1]+ordered+[hole[hi]]+merged[best:]


def _ear_clip(poly, pts):
    if len(poly) < 3:
        return []
    work = list(poly)
    triangles = []

    def turn(a, b, c):
        return ((pts[b][0]-pts[a][0])*(pts[c][1]-pts[a][1])
                - (pts[b][1]-pts[a][1])*(pts[c][0]-pts[a][0]))

    def signs(a, b, c, p):
        pa, pb, pc = pts[a], pts[b], pts[c]
        return ((pb[0]-pa[0])*(p[1]-pa[1])-(pb[1]-pa[1])*(p[0]-pa[0]),
                (pc[0]-pb[0])*(p[1]-pb[1])-(pc[1]-pb[1])*(p[0]-pb[0]),
                (pa[0]-pc[0])*(p[1]-pc[1])-(pa[1]-pc[1])*(p[0]-pc[0]))

    def vertex_free(a, b, c, on_boundary):
        for other in work:
            if other in (a, b, c):
                continue
            low = -AREA_EPS if on_boundary else AREA_EPS
            if all(s > low for s in signs(a, b, c, pts[other])):
                return False
        return True

    def chord_clear(a, c, edges):
        # Collinear vertices can sit on an ear edge while the polygon itself
        # crosses the closing chord; the vertex test alone misses that.
        for e0, e1 in edges:
            if a in (e0, e1) or c in (e0, e1):
                continue
            if _segments_cross(pts[a], pts[c], pts[e0], pts[e1]):
                return False
        return True

    guard = 0
    while len(work) > 3 and guard < 8*len(poly)**2:
        guard += 1
        edges = [(work[i], work[(i+1) % len(work)]) for i in range(len(work))]
        clipped = False
        # Convex ears must not touch other vertices at all; degenerate ears
        # (spikes and collinear runs) only avoid strictly interior vertices.
        # A last resort keeps the clipper moving on weakly simple input.
        for mode in ("strict", "degenerate", "resort"):
            for i in range(len(work)):
                a, b, c = work[(i-1) % len(work)], work[i], work[(i+1) % len(work)]
                value = turn(a, b, c)
                if mode == "strict":
                    if value <= AREA_EPS or not vertex_free(a, b, c, True):
                        continue
                elif mode == "degenerate":
                    if abs(value) > AREA_EPS or not vertex_free(a, b, c, False):
                        continue
                elif value <= AREA_EPS or not vertex_free(a, b, c, False):
                    continue
                if not chord_clear(a, c, edges):
                    continue
                triangles.append((a, b, c))
                del work[i]
                clipped = True
                break
            if clipped:
                break
        if not clipped:
            break
    if len(work) == 3:
        triangles.append(tuple(work))
    elif len(work) > 3:  # Numerically stuck; a fan keeps output deterministic.
        triangles += [(work[0], work[i], work[i+1]) for i in range(1, len(work)-1)]
    return [t for t in triangles if turn(*t) > AREA_EPS]


def _triangulate_face(outer, holes, pts):
    merged = list(outer)
    pending = sorted(holes, key=lambda h: -max(pts[i][0] for i in h))
    while pending:
        merged = _bridge(merged, pending.pop(0), pending, pts)
    return _ear_clip(merged, pts)


def _chain_loops(chords, positions, basis):
    # Chain cap chords into closed loops or open chains via topological anchors.
    # Anchors are mesh vertices/edges, so welded points join without float matching.
    at_anchor = {}
    for ci, (ia, ib, anchor_a, anchor_b) in enumerate(chords):
        at_anchor.setdefault(anchor_a, []).append(2*ci)
        at_anchor.setdefault(anchor_b, []).append(2*ci+1)
    u, v = basis

    def vertex(e):
        return chords[e//2][e % 2]

    def other_vertex(e):
        return chords[e//2][1-e % 2]

    pair = {}
    for ends in at_anchor.values():
        if len(ends) == 2:
            pair[ends[0]], pair[ends[1]] = ends[1], ends[0]
            continue
        # Plane through a shared vertex or an open boundary: pair opposite
        # directions; an odd remainder stays unpaired and ends an open chain.
        anchor_point = positions[vertex(ends[0])]

        def angle(e):
            direction = positions[other_vertex(e)]-anchor_point
            return float(np.arctan2(direction @ v, direction @ u))

        ordered = sorted(ends, key=angle)
        half = len(ordered)//2
        for k in range(half):
            pair[ordered[k]] = ordered[k+half]
            pair[ordered[k+half]] = ordered[k]

    visited = set()
    loops, chains = [], []
    for start in range(2*len(chords)):
        if start in visited:
            continue
        # Walk backward to a dead end; cycling back to start means a closed loop.
        origin = start
        seen = set()
        while origin in pair and pair[origin]^1 not in seen:
            seen.add(origin)
            origin = pair[origin]^1
            if origin == start:
                break
        walk, e, closed = [], origin, False
        while True:
            visited.add(e)
            walk.append(vertex(e))
            other = e ^ 1
            visited.add(other)
            nxt = pair.get(other)
            if nxt is None:
                walk.append(vertex(other))
                break
            if nxt == origin:
                closed = True
                break
            e = nxt
        (loops if closed else chains).append(walk)
    return loops, chains


def _clip_and_cap(soup, plane, part):
    normal = np.asarray(plane["normal"], dtype=np.float64)
    offset = float(plane["offset"])
    if plane["keep"] == "above":
        normal, offset = -normal, -offset
    cap_normal = normal  # Caps face the removed half-space.
    positions = [np.asarray(p, dtype=np.float64) for p in soup["positions"]]
    normals = [np.asarray(n, dtype=np.float64) for n in soup["normals"]]
    uvs = [np.asarray(t, dtype=np.float64) for t in soup["uv"]]
    signed = np.asarray(positions) @ normal-offset
    tol = SNAP*max(1.0, float(np.abs(signed).max(initial=0.0)), abs(offset))
    on_plane = set()
    for i in np.nonzero(np.abs(signed) <= tol)[0]:
        positions[i] = positions[i]-signed[i]*normal  # Snap exactly onto the plane.
        signed[i] = 0.0
        on_plane.add(int(i))
    edge_cache = {}
    anchor_of = {}

    def intersect(i, j):
        si, sj = signed[i], signed[j]
        if si == 0.0:
            return i
        if sj == 0.0:
            return j
        key = (int(i), int(j)) if i < j else (int(j), int(i))
        if key not in edge_cache:
            t = si/(si-sj)
            edge_cache[key] = len(positions)
            positions.append(positions[i]+t*(positions[j]-positions[i]))
            normals.append(normals[i]+t*(normals[j]-normals[i]))
            uvs.append(uvs[i]+t*(uvs[j]-uvs[i]))
            anchor_of[edge_cache[key]] = ("e", key)
            on_plane.add(edge_cache[key])
        return edge_cache[key]

    kept, chords = [], []
    removed = 0
    seen_chords = set()
    for a, b, c, tag in soup["triangles"]:
        idx = (a, b, c)
        if all(signed[v] <= 0.0 for v in idx):
            kept.append((a, b, c, tag))
            continue
        if all(signed[v] > 0.0 for v in idx):
            removed += 1
            continue
        poly = []
        for k in range(3):
            cur, nxt = idx[k], idx[(k+1) % 3]
            if signed[cur] <= 0.0:
                poly.append(cur)
            if (signed[cur] <= 0.0) != (signed[nxt] <= 0.0):
                poly.append(intersect(cur, nxt))
        deduped = []
        for vtx in poly:
            if not deduped or vtx != deduped[-1]:
                deduped.append(vtx)
        if len(deduped) > 1 and deduped[-1] == deduped[0]:
            deduped.pop()
        if len(deduped) >= 3:
            kept.append((deduped[0], deduped[1], deduped[2], tag))
            if len(deduped) == 4:
                kept.append((deduped[0], deduped[2], deduped[3], tag))
        for k in range(len(deduped)):
            p, q = deduped[k], deduped[(k+1) % len(deduped)]
            if p in on_plane and q in on_plane and p != q:
                key = (p, q) if p < q else (q, p)
                if key not in seen_chords:
                    seen_chords.add(key)
                    chords.append((p, q, anchor_of.get(p, ("v", int(p))),
                                   anchor_of.get(q, ("v", int(q)))))

    kept_count = len(kept)
    basis = plane_basis(cap_normal)
    loops, chains = _chain_loops(chords, positions, basis) if chords else ([], [])
    pts2d, lo, extent = {}, np.zeros(2), 0.0
    for loop in loops:
        for vtx in loop:
            if vtx not in pts2d:
                w = positions[vtx]
                pts2d[vtx] = np.array([float(w @ basis[0]), float(w @ basis[1])])
    if pts2d:
        stacked = np.stack(list(pts2d.values()))
        lo = stacked.min(axis=0)
        extent = float(np.maximum(stacked.max(axis=0)-lo,1e-300).max())
        if extent > 0.0:
            pts2d = {key: (value-lo)/extent for key, value in pts2d.items()}
    cap_triangles, cap_loops = [], []
    if loops:
        areas = [_signed_area(np.array([pts2d[vtx] for vtx in loop])) for loop in loops]
        depth = [sum(1 for j, other in enumerate(loops) if j != i and
                     _contains([pts2d[vtx] for vtx in other], pts2d[loops[i][0]]))
                 for i in range(len(loops))]
        groups = {}
        for i, loop in enumerate(loops):
            if abs(areas[i]) <= AREA_EPS:
                continue  # Plane tangent along a face: no measurable section.
            if depth[i] % 2 == 0:
                groups.setdefault(i, [])
            else:
                hosts = [j for j in range(len(loops)) if depth[j] == depth[i]-1
                         and abs(areas[j]) > AREA_EPS
                         and _contains([pts2d[vtx] for vtx in loops[j]], pts2d[loop[0]])]
                if hosts:
                    groups.setdefault(min(hosts, key=lambda j: abs(areas[j])), []).append(i)
        for outer, hole_ids in groups.items():
            if abs(areas[outer]) <= AREA_EPS:
                continue
            outer_loop = loops[outer] if areas[outer] > 0 else loops[outer][::-1]
            hole_loops = [loops[h][::-1] if areas[h] > 0 else loops[h] for h in hole_ids]
            triangles = _triangulate_face(outer_loop, hole_loops, pts2d)
            cap_triangles += triangles
            cap_loops.append({"vertices": len(loops[outer]), "holes": len(hole_ids),
                              "area": abs(areas[outer])*extent*extent,
                              "triangles": len(triangles)})
    for a, b, c in cap_triangles:
        kept.append((a, b, c, part))

    used = sorted({vtx for tri in kept for vtx in tri[:3]})
    remap = {old: new for new, old in enumerate(used)}
    out = {"positions": [positions[i] for i in used],
           "normals": [normals[i] for i in used],
           "uv": [uvs[i] for i in used],
           "triangles": [(remap[a], remap[b], remap[c], tag) for a, b, c, tag in kept]}
    open_chains = [{"vertices": len(chain),
                    "endpoints": [positions[chain[0]].tolist(), positions[chain[-1]].tolist()]}
                   for chain in chains]
    status = ("open" if open_chains else
              "removed" if removed and not kept_count else
              "sealed" if removed or cap_loops else "empty")
    report = {"status": status, "kept_triangles": kept_count, "removed_triangles": removed,
              "cap_loops": cap_loops, "cap_triangles": len(cap_triangles),
              "open_chains": open_chains}
    return out, report, cap_normal


def apply_sections(objects, sections):
    """Clip every visible object by each section in order and build caps.

    sections is a list of (index, record_id, payload). Returns (entries,
    reports): entries are render objects tagged with a "section" index
    (0 = original surface); reports describe sealing per cut.
    """
    entries = []
    per_object = []
    for obj in objects:
        soup = {"positions": [tuple(p) for p in obj["positions"]],
                "normals": [tuple(n) for n in obj["normals"]],
                "uv": [tuple(t) for t in obj["uv"]] if obj["uv"] is not None
                      else [(0.0, 0.0)]*len(obj["positions"]),
                "triangles": [(int(a), int(b), int(c), 0) for a, b, c in obj["triangles"]]}
        cap_normals = {}
        object_reports = []
        for index, record_id, payload in sections:
            soup, report, cap_normal = _clip_and_cap(soup, payload["plane"], index)
            report.update({"node": obj["key"], "section": index})
            cap_normals[index] = cap_normal
            object_reports.append(report)
        per_object.append(object_reports)
        tags = {}
        for a, b, c, tag in soup["triangles"]:
            tags.setdefault(tag, []).append((a, b, c))
        positions = np.asarray(soup["positions"])
        if tags.get(0):
            entries.append({**obj, "positions": positions,
                            "normals": np.asarray(soup["normals"]),
                            "uv": np.asarray(soup["uv"]) if obj["uv"] is not None else None,
                            "triangles": np.asarray(tags[0], dtype=np.int64).reshape(-1, 3),
                            "section": 0})
        for index, record_id, payload in sections:
            cap = tags.get(index)
            if not cap:
                continue
            entries.append({"id": obj["id"], "key": obj["key"], "positions": positions,
                            "normals": np.tile(cap_normals[index], (len(positions), 1)),
                            "uv": None,
                            "triangles": np.asarray(cap, dtype=np.int64).reshape(-1, 3),
                            "surface": cap_surface(payload["cap"]),
                            "mirrored": False, "section": index})
    reports = []
    for index, record_id, payload in sections:
        objects_report = [{"node": obj["key"], **{key: report[key] for key in
                           ("status", "kept_triangles", "removed_triangles",
                            "cap_triangles", "cap_loops", "open_chains")}}
                          for obj, object_reports in zip(objects, per_object)
                          for report in object_reports if report["section"] == index]
        status = ("open" if any(r["status"] == "open" for r in objects_report) else
                  "removed" if objects_report and all(r["status"] == "removed" for r in objects_report)
                  else "sealed" if any(r["status"] in {"sealed", "removed"} for r in objects_report)
                  else "empty")
        reports.append({"index": index, "id": record_id, "label": payload["label"],
                        "plane": payload["plane"], "status": status,
                        "objects": objects_report})
    return entries, reports
