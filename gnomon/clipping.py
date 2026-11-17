import numpy as np

# Clip the packed vertex [clip xyzw, world xyz, normal xyz, uv xy].
PLANES = np.array([[1,0,0,1],[-1,0,0,1],[0,1,0,1],[0,-1,0,1],[0,0,1,1],[0,0,-1,1]])


def clip_triangle(vertices):
    polygon = list(vertices)
    for plane in PLANES:
        if not polygon:
            return []
        clipped = []
        previous = polygon[-1]
        dp = float(previous[:4] @ plane)
        for current in polygon:
            dc = float(current[:4] @ plane)
            if (dp >= 0) != (dc >= 0):
                ratio = dp/(dp-dc)
                clipped.append(previous+(current-previous)*ratio)
            if dc >= 0:
                clipped.append(current)
            previous,dp = current,dc
        polygon = clipped
    return [np.stack([polygon[0],polygon[i],polygon[i+1]]) for i in range(1,len(polygon)-1)]
