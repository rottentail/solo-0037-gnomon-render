import numpy as np
from .errors import require
from .math3d import world_bounds


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
    vertex_colors = payload.get("vertex_colors")
    if vertex_colors is not None:
        vertex_colors = np.asarray(vertex_colors, dtype=np.float64)
        require(vertex_colors.shape == (len(positions),3) and np.isfinite(vertex_colors).all(), "mesh_vertex_color", "one finite vertex color triplet per vertex required")
        require(bool(np.all(vertex_colors >= 0) and np.all(vertex_colors <= 1)), "mesh_vertex_color", "vertex colors must be sRGB in [0,1]")
        vertex_colors = vertex_colors.tolist()
    return {"label":payload["label"], "positions":positions.tolist(), "triangles":faces.tolist(),
            "normals":normals.tolist(), "uv":uv, "vertex_colors":vertex_colors,
            "bounds":world_bounds(positions),
            "area":float(area2.sum()/2), "vertex_count":len(positions), "triangle_count":len(faces)}
