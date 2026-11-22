"""景深：光圈采样、薄透镜清晰范围与多通道颜色平均。

不做图像空间模糊。每个光圈样本偏移视点并重新对准清晰平面，独立成像后
在线性空间平均颜色；几何通道（深度、对象、法线、世界坐标、覆盖率）始终
来自零偏移的针孔通道，拾取语义不受散景影响。
"""
import numpy as np
from .math3d import unit
from .raster import render_samples

GOLDEN_ANGLE = 2.399963229728653  # Vogel 圆盘黄金角，确定性采样无需随机数
PATTERN = "vogel-disk"


def aperture_offsets(count):
    """光圈圆盘上的确定性采样点（半径≤1）；首个样本恒为圆心针孔。"""
    offsets = [(0.0, 0.0)]
    for k in range(1, count):
        radius = np.sqrt((k - 0.5) / (count - 1)) if count > 2 else 0.5**0.5
        theta = (k - 1) * GOLDEN_ANGLE
        offsets.append((float(radius * np.cos(theta)), float(radius * np.sin(theta))))
    return offsets


def aperture_diameter(lens, f_stop):
    """光圈直径（世界单位）。透视：传感器高度固定为 1 世界单位，焦距由 fov 推出；
    正交：span 即传感器覆盖的世界高度，f_stop 表示 span 与光圈直径之比。"""
    if lens["projection"] == "perspective":
        focal_length = 0.5 / np.tan(np.radians(lens["fov"]) / 2)
        return float(focal_length / f_stop)
    return float(lens["span"] / f_stop)


def focus_report(lens, focus, height):
    """作者可读的清晰范围：以 1 输出像素为容差，沿视线方向的世界距离区间。

    透视的弥散圆（占传感器高度比例）为 f·D·|1/s − 1/d|，正交为 D·|1 − d/s|/span；
    令其在容差内解出近/远清晰距离。sharp_far 为 None 表示清晰到无穷远。
    """
    distance, f_stop = focus["distance"], focus["f_stop"]
    diameter = aperture_diameter(lens, f_stop)
    coc_limit = 1.0 / height
    if lens["projection"] == "perspective":
        focal_length = 0.5 / np.tan(np.radians(lens["fov"]) / 2)
        epsilon = coc_limit * f_stop / focal_length**2
        sharp_near = distance / (1 + distance * epsilon)
        sharp_far = None if distance * epsilon >= 1 else distance / (1 - distance * epsilon)
    else:
        epsilon = f_stop * coc_limit
        sharp_near = max(0.0, distance * (1 - epsilon))
        sharp_far = distance * (1 + epsilon)
    return {"distance": distance, "f_stop": f_stop, "samples": focus["samples"],
            "projection": lens["projection"], "aperture_diameter": diameter,
            "coc_limit": coc_limit, "sharp_near": sharp_near, "sharp_far": sharp_far,
            "pattern": PATTERN}


def jittered_lens(lens, focus, offset):
    """视点沿光圈偏移并重新对准清晰平面上的同一点；投影与裁剪面保持不变。"""
    eye = np.asarray(lens["eye"], dtype=np.float64)
    forward = unit(np.asarray(lens["target"], dtype=np.float64) - eye)
    side = unit(np.cross(forward, np.asarray(lens["up"], dtype=np.float64)))
    up = np.cross(side, forward)
    radius = aperture_diameter(lens, focus["f_stop"]) / 2
    shift = (offset[0] * side + offset[1] * up) * radius
    focal_point = eye + forward * focus["distance"]
    return {**lens, "eye": (eye + shift).tolist(), "target": focal_point.tolist()}


def render_with_focus(compiled, request):
    """光圈样本逐次成像并平均颜色；其余通道保留零偏移针孔通道的结果。"""
    focus = request["focus"]
    lens = compiled["scene"]["lens"]
    offsets = aperture_offsets(focus["samples"])
    buffers = metadata = total = None
    written = clipped = 0
    for offset in offsets:
        if offset == (0.0, 0.0):
            pass_compiled = compiled
        else:
            pass_lens = jittered_lens(lens, focus, offset)
            pass_compiled = {**compiled, "scene": {**compiled["scene"], "lens": pass_lens}}
        pass_buffers, pass_metadata = render_samples(pass_compiled, request)
        written += pass_metadata["stats"]["written_samples"]
        clipped += pass_metadata["stats"]["clipped_triangles"]
        if buffers is None:
            buffers, metadata = pass_buffers, pass_metadata
            total = buffers["color"].copy()
        else:
            total += pass_buffers["color"]
    buffers["color"] = total / len(offsets)
    metadata["stats"] = {**metadata["stats"], "written_samples": written,
                         "clipped_triangles": clipped, "lens_passes": len(offsets)}
    return buffers, metadata
