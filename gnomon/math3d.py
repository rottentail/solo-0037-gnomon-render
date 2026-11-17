import numpy as np
from .errors import require


def unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    length = np.linalg.norm(vector, axis=-1, keepdims=True)
    require(bool(np.all(length > 1e-12)), "zero_vector", "direction has zero length")
    return vector / length


def pose_matrix(pose):
    x, y, z = np.radians(pose["rotate"])
    cx, cy, cz = np.cos([x, y, z])
    sx, sy, sz = np.sin([x, y, z])
    rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    matrix = np.eye(4)
    matrix[:3,:3] = rz @ ry @ rx @ np.diag(pose["scale"])
    matrix[:3,3] = pose["translate"]
    return matrix


def transform_points(points, matrix):
    homogeneous = np.column_stack([points, np.ones(len(points))])
    return (homogeneous @ matrix.T)[:,:3]


def transform_normals(normals, matrix):
    normal_matrix = np.linalg.inv(matrix[:3,:3]).T
    return unit(np.asarray(normals) @ normal_matrix.T)


def world_bounds(points):
    return {"min": points.min(axis=0).tolist(), "max": points.max(axis=0).tolist()}


def orbit_lens(lens, angle):
    from copy import deepcopy
    result = deepcopy(lens)
    delta = np.asarray(lens["eye"]) - np.asarray(lens["target"])
    theta = np.radians(angle)
    matrix = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
    result["eye"] = (np.asarray(lens["target"]) + matrix @ delta).tolist()
    return result
