import numpy as np
from .math3d import unit
from .errors import require


def camera_basis(lens):
    # Eye, target, forward, image-right side and up in world space.
    # A parallel stereo rig offsets both eye and target along side so the two
    # eyes keep exactly the same forward and up axes.
    eye = np.asarray(lens["eye"], dtype=np.float64)
    target = np.asarray(lens["target"], dtype=np.float64)
    forward = unit(target - eye)
    side = unit(np.cross(forward, np.asarray(lens["up"], dtype=np.float64)))
    up = np.cross(side, forward)
    return eye, target, forward, side, up


def camera_matrices(lens, aspect):
    eye, target, forward, side, up = camera_basis(lens)
    view = np.eye(4)
    view[:3, :3] = np.stack([side, up, -forward])
    view[:3, 3] = -view[:3, :3] @ eye
    n, f = lens["near"], lens["far"]
    # Principal-point shift in normalised device coordinates. Stereo uses it to
    # put the zero-parallax plane at the convergence distance without rotating
    # (toeing in) the eyes.
    shift_x, shift_y = (float(value) for value in lens.get("shift", (0.0, 0.0)))
    if lens["projection"] == "perspective":
        cot = 1 / np.tan(np.radians(lens["fov"]) / 2)
        proj = np.array([[cot / aspect, 0, -shift_x, 0], [0, cot, -shift_y, 0],
                         [0, 0, -(f + n) / (f - n), -2 * f * n / (f - n)], [0, 0, -1, 0]])
    else:
        h = lens["span"]
        proj = np.array([[2 / (h * aspect), 0, 0, shift_x], [0, 2 / h, 0, shift_y],
                         [0, 0, -2 / (f - n), -(f + n) / (f - n)], [0, 0, 0, 1]])
    return view, proj


def project_points(points, view, projection):
    # World points -> clip, NDC and pixel-centre coordinates.
    homogeneous = np.column_stack([np.asarray(points, dtype=np.float64),
                                   np.ones(len(points))])
    clip = homogeneous @ (projection @ view).T
    w = clip[:, 3]
    require(bool(np.all(np.abs(w) > 1e-12)), "projection_w", "point projects to infinity")
    ndc = clip[:, :3] / w[:, None]
    return {"clip": clip, "ndc": ndc, "w": w}


def unproject(x, y, depth, width, height, view, projection):
    ndc = np.array([2 * (x + 0.5) / width - 1, 1 - 2 * (y + 0.5) / height, depth, 1])
    point = np.linalg.inv(projection @ view) @ ndc
    require(abs(point[3]) > 1e-12, "unproject", "point at infinity")
    return (point[:3] / point[3]).tolist()
