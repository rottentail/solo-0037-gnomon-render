import bisect
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


def unwrap_degrees(values):
    """Lift degree samples onto a continuous branch so +179 -> -179 stays short.

    Naive interpolation across the sign boundary would sweep 358 degrees; adding
    whole 360-degree turns keeps the authored shortest rotation. The lifted
    branch is equivalent modulo 360 but numerically distinct, which is what the
    pose matrix consumes.
    """
    lifted = [float(values[0])]
    for value in values[1:]:
        delta = (float(value) - lifted[-1] + 180.0) % 360.0 - 180.0
        lifted.append(lifted[-1] + delta)
    return lifted


def smoothstep(alpha):
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _catmull_rom(p0, p1, p2, p3, t0, t1, t2, t3, t):
    # Centripetal (alpha=0.5) non-uniform Catmull-Rom: passes through every
    # keyframe with a continuous tangent while avoiding loops and cusps when
    # keyframe times are uneven.
    if abs(t1 - t0) < 1e-12 or abs(t2 - t1) < 1e-12 or abs(t3 - t2) < 1e-12:
        return p1
    m1 = (p2 - p0) / (t2 - t0)
    m2 = (p3 - p1) / (t3 - t1)
    span = t2 - t1
    a = p1
    b = m1 * span
    c = 3.0 * (p2 - p1) / span - 2.0 * m1 - m2
    d = 2.0 * (p1 - p2) / span + m1 + m2
    u = (t - t1) / span
    return a + b * u + c * u * u + d * u * u * u


def sample_channel(times, values, at, mode):
    """Sample one scalar channel at time ``at`` with the chosen continuity."""
    times = list(times)
    values = [float(v) for v in values]
    if at <= times[0]:
        return values[0]
    if at >= times[-1]:
        return values[-1]
    index = bisect.bisect_right(times, at) - 1
    t0, t1 = times[index], times[index + 1]
    p1, p2 = values[index], values[index + 1]
    if mode == "step":
        return p1
    alpha = (at - t0) / (t1 - t0)
    if mode == "linear":
        return p1 + (p2 - p1) * alpha
    if mode == "smooth":
        return p1 + (p2 - p1) * smoothstep(alpha)
    p0 = values[index - 1] if index > 0 else 2.0 * p1 - p2
    p3 = values[index + 2] if index + 2 < len(values) else 2.0 * p2 - p1
    tt = (times[index - 1] if index > 0 else t0 - (t1 - t0), t0, t1,
          times[index + 2] if index + 2 < len(times) else t1 + (t1 - t0))
    return float(_catmull_rom(np.array(p0), np.array(p1), np.array(p2), np.array(p3),
                              tt[0], tt[1], tt[2], tt[3], at))


INTERPOLATION_MODES = ("step", "linear", "smooth", "smooth_tangent")

