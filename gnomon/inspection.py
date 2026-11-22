import numpy as np
from .errors import require


def inspect_pixel(channels, metadata, x, y):
    h, w = channels["depth"].shape
    require(0 <= x < w and 0 <= y < h, "pixel_bounds", "pixel outside output region")
    identity = int(channels["object_id"][y, x])
    offset = channels["sample_offset"][y, x]
    rx, ry, _, _ = metadata["region"]
    point = None if identity == 0 else channels["world"][y, x].tolist()
    normal = None if identity == 0 else channels["normal"][y, x].tolist()
    result = {"pixel": [x, y], "full_sample": [float(rx+x+offset[0]), float(ry+y+offset[1])],
              "coverage": float(channels["coverage"][y, x]), "object_id": identity,
              "node": metadata["node_map"].get(str(identity)), "world": point, "normal": normal,
              "linear_rgb": channels["color"][y, x].tolist(),
              "selection": "nearest covered subpixel"}
    if metadata.get("lens_projection") == "equirectangular":
        # Depth is ray distance, not OpenGL NDC; the ray is the actual
        # direction traced from the lens eye into the original scene.
        direction = channels["ray"][y, x].tolist()
        result["ndc_depth"] = None
        result["ray_direction"] = direction
        result["distance"] = None if identity == 0 else float(channels["depth"][y, x])
        yaw = float(np.degrees(np.arctan2(direction[0], -direction[2])))
        pitch = float(np.degrees(np.arcsin(np.clip(direction[1], -1, 1))))
        result["yaw_degrees"] = yaw
        result["pitch_degrees"] = pitch
        result["depth_convention"] = metadata["depth_convention"]
    else:
        result["ndc_depth"] = None if identity == 0 else float(channels["depth"][y, x])
    return result
