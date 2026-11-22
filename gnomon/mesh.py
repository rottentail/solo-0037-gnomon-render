import numpy as np
from .errors import RenderError, require
from .math3d import world_bounds


def vertex_frames(positions, normals, uv, triangles):
    """Mesh-space tangent frames for a tangent-space normal map.

    Returns (tangent, bitangent), each (V,3), orthonormal with the vertex
    normal. Both vectors come from the triangle UV derivatives, weighted by
    triangle area, so the frame follows the texture layout rather than the
    geometry alone.

    A UV seam is authored the way per-vertex UV attributes always require:
    duplicate the ridge vertices so each chart owns its UV values. The two
    copies then accumulate their own frames and the seam is rendered as a
    crisp change of tangent direction. If a seam is instead expressed by
    wrapping UVs across one shared vertex, the +u and -u contributions cancel
    and no usable tangent exists; that is reported instead of silently
    zeroing the normal.
    """
    faces = np.asarray(triangles, dtype=np.int64)
    p, t = positions[faces], uv[faces]
    d1, d2 = t[:,1]-t[:,0], t[:,2]-t[:,0]
    det = d1[:,0]*d2[:,1]-d1[:,1]*d2[:,0]
    scale = np.linalg.norm(d1,axis=1)*np.linalg.norm(d2,axis=1)
    degenerate = np.abs(det) <= np.maximum(1e-15,1e-9*scale)
    if np.any(degenerate):
        listed = [int(i) for i in np.flatnonzero(degenerate)[:16]]
        extra = int(degenerate.sum())-len(listed)
        tail = "" if extra == 0 else f" (+{extra} more)"
        raise RenderError("normal_tangent",
            f"normal map requires a nonzero UV area on every face; faces with no usable tangent: {listed}{tail}")
    e1, e2 = p[:,1]-p[:,0], p[:,2]-p[:,0]
    face_tangent = (e1*d2[:,1:2]-e2*d1[:,1:2])/det[:,None]
    face_bitangent = (e2*d1[:,0:1]-e1*d2[:,0:1])/det[:,None]
    weight = 0.5*np.abs(det)
    tangent_sum = np.zeros((len(positions),3))
    bitangent_sum = np.zeros((len(positions),3))
    for corner in range(3):
        np.add.at(tangent_sum,faces[:,corner],face_tangent*weight[:,None])
        np.add.at(bitangent_sum,faces[:,corner],face_bitangent*weight[:,None])
    tangent = tangent_sum - normals*np.sum(normals*tangent_sum,axis=1,keepdims=True)
    length = np.linalg.norm(tangent,axis=1)
    broken = length <= 1e-10
    if np.any(broken):
        listed = [int(i) for i in np.flatnonzero(broken)[:16]]
        extra = int(broken.sum())-len(listed)
        tail = "" if extra == 0 else f" (+{extra} more)"
        raise RenderError("normal_tangent",
            f"normal map requires a tangent direction on every vertex; vertices without one: {listed}{tail} "
            "(author a UV seam as duplicated vertices sharing one position)")
    tangent /= length[:,None]
    # Gram-Schmidt the bitangent against the tangent while keeping the sign of
    # the UV v axis, which is what encodes the chart's handedness.
    bitangent = bitangent_sum - normals*np.sum(normals*bitangent_sum,axis=1,keepdims=True)
    bitangent -= tangent*np.sum(tangent*bitangent,axis=1,keepdims=True)
    length = np.linalg.norm(bitangent,axis=1)
    collapsed = length <= 1e-10
    if np.any(collapsed):
        # A chart that is degenerate along v still has a tangent-derived
        # handedness; complete the frame around the vertex normal.
        bitangent[collapsed] = np.cross(normals[collapsed],tangent[collapsed])
        length[collapsed] = 1
    return tangent, bitangent/length[:,None]


def prepare_mesh(payload):
    positions = np.asarray(payload["positions"], dtype=np.float64)
    faces = np.asarray(payload["triangles"], dtype=np.int64)
    require(np.isfinite(positions).all() and bool(np.all(np.abs(positions) <= 1e6)), "mesh_nonfinite", "positions must be finite and within supported coordinates")
    require(faces.min() >= 0 and faces.max() < len(positions), "mesh_index", "triangle references missing vertex")
    p = positions[faces]
    cross = np.cross(p[:,1]-p[:,0], p[:,2]-p[:,0])
    area2 = np.linalg.norm(cross, axis=1)
    require(bool(np.all(area2 > 1e-12)), "mesh_degenerate", "zero-area triangles are not accepted")
    raw = payload.get("normals")
    if raw is None:
        normals = np.zeros_like(positions)
        for corner in range(3):
            np.add.at(normals, faces[:,corner], cross)
        length = np.linalg.norm(normals, axis=1)
        require(bool(np.all(length > 1e-12)), "mesh_normal", "unused or cancelling vertices need explicit normals")
        normals /= length[:,None]
    else:
        normals = np.asarray(raw, dtype=np.float64)
        require(normals.shape == positions.shape, "mesh_normal", "one normal per vertex required")
        length = np.linalg.norm(normals, axis=1)
        require(np.isfinite(normals).all() and bool(np.all(np.abs(normals)<=1e6)) and bool(np.all(length > 1e-12)), "mesh_normal", "normals must be finite and nonzero")
        normals /= length[:,None]
    uv = payload.get("uv")
    if uv is not None:
        uv = np.asarray(uv, dtype=np.float64)
        require(uv.shape == (len(positions),2) and np.isfinite(uv).all() and bool(np.all(np.abs(uv)<=1e6)), "mesh_uv", "one finite UV pair per vertex required")
        uv = uv.tolist()
    return {"label":payload["label"], "positions":positions.tolist(), "triangles":faces.tolist(),
            "normals":normals.tolist(), "uv":uv, "bounds":world_bounds(positions),
            "area":float(area2.sum()/2), "vertex_count":len(positions), "triangle_count":len(faces)}
