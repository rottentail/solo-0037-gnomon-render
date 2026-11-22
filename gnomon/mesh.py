import numpy as np
from .errors import require
from .math3d import world_bounds


def _corner_normals(raw, count):
    corners = np.asarray(raw, dtype=np.float64)
    require(corners.shape == (count, 3, 3), "mesh_normal", "one normal per triangle corner required")
    require(np.isfinite(corners).all() and bool(np.all(np.abs(corners) <= 1e6)), "mesh_normal",
            "corner normals must be finite and within supported coordinates")
    length = np.linalg.norm(corners, axis=2)
    require(bool(np.all(length > 1e-12)), "mesh_normal", "corner normals must be nonzero")
    return corners / length[:, :, None]


def _corner_uv(raw, count):
    corners = np.asarray(raw, dtype=np.float64)
    require(corners.shape == (count, 3, 2), "mesh_uv", "one UV pair per triangle corner required")
    require(np.isfinite(corners).all() and bool(np.all(np.abs(corners) <= 1e6)), "mesh_uv",
            "corner UVs must be finite and within supported coordinates")
    return corners


def prepare_mesh(payload):
    positions = np.asarray(payload["positions"], dtype=np.float64)
    faces = np.asarray(payload["triangles"], dtype=np.int64)
    require(np.isfinite(positions).all() and bool(np.all(np.abs(positions) <= 1e6)), "mesh_nonfinite", "positions must be finite and within supported coordinates")
    require(faces.min() >= 0 and faces.max() < len(positions), "mesh_index", "triangle references missing vertex")
    p = positions[faces]
    cross = np.cross(p[:,1]-p[:,0], p[:,2]-p[:,0])
    area2 = np.linalg.norm(cross, axis=1)
    require(bool(np.all(area2 > 1e-12)), "mesh_degenerate", "zero-area triangles are not accepted")
    # Per-corner attributes are optional. When present they are authoritative
    # for shading and let a welded mesh carry UV and normal seams along
    # material boundaries without duplicating vertices, so triangle and face
    # identity used by picking is unchanged.
    seam_normals = None if payload.get("corner_normals") is None else _corner_normals(payload["corner_normals"], len(faces))
    seam_uv = None if payload.get("corner_uv") is None else _corner_uv(payload["corner_uv"], len(faces))
    raw = payload.get("normals")
    if raw is None:
        normals = np.zeros_like(positions)
        for corner in range(3):
            np.add.at(normals, faces[:,corner], cross)
        length = np.linalg.norm(normals, axis=1)
        if seam_normals is None:
            require(bool(np.all(length > 1e-12)), "mesh_normal", "unused or cancelling vertices need explicit normals")
        normals = np.divide(normals, length[:,None], out=np.zeros_like(normals), where=length[:,None]>1e-12)
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
    return {"label":payload["label"], "positions":positions.tolist(), "triangles":faces.tolist(),
            "normals":normals.tolist(), "uv":None if uv is None else uv.tolist(),
            # Stored only when the mesh actually carries a seam; renderers fall
            # back to vertex attributes otherwise, keeping old payloads small.
            "corner_normals":None if seam_normals is None else seam_normals.tolist(),
            "corner_uv":None if seam_uv is None else seam_uv.tolist(),
            "bounds":world_bounds(positions),
            "area":float(area2.sum()/2), "vertex_count":len(positions), "triangle_count":len(faces),
            "seams":{"normal":seam_normals is not None,"uv":seam_uv is not None}}
