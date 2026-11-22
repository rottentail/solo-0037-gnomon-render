"""Complex-occlusion contour line layer demonstration.

Builds a scene with several overlapping objects at different depths, then
writes the independently deliverable line structures (outer silhouette,
inter-object occlusion boundary, internal crease), the transparent combined
line layer, and the line/color overlay composite.  Texture patterns never
become geometric lines.  Runs entirely offline through the production API.
"""
import argparse
import json
import tempfile
from pathlib import Path

from gnomon.service import RendererService


def box_mesh(cx, cy, cz, sx, sy, sz, label):
    """Axis-aligned closed cube centered at (cx, cy, cz)."""
    x0, x1 = cx - sx / 2, cx + sx / 2
    y0, y1 = cy - sy / 2, cy + sy / 2
    z0, z1 = cz - sz / 2, cz + sz / 2
    positions = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                 [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]
    triangles = [[0, 3, 2], [0, 2, 1], [4, 5, 6], [4, 6, 7],
                 [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                 [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3]]
    return {"label": label, "positions": positions, "triangles": triangles}


def wedge_mesh():
    return {"label": "斜楔",
            "positions": [[-1, -0.7, -0.5], [1, -0.7, -0.5], [0.5, 0.9, -0.5],
                          [-1, -0.7, 0.5], [1, -0.7, 0.5], [0.5, 0.9, 0.5]],
            "triangles": [[0, 2, 1], [3, 4, 5], [0, 1, 4], [0, 4, 3],
                          [1, 2, 5], [1, 5, 4], [2, 0, 3], [2, 3, 5]]}


def produce(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gnomon-contour-") as root:
        service = RendererService(root)
        try:
            back = service.add_mesh(box_mesh(0.0, 0.0, -0.4, 2.4, 2.0, 0.5, "背景板"))
            wedge = service.add_mesh(wedge_mesh())
            small = service.add_mesh(box_mesh(1.05, 0.95, 0.9, 0.35, 0.35, 0.35, "近景小盒"))
            far = service.add_mesh(box_mesh(-1.15, -0.9, 0.2, 0.5, 0.5, 0.5, "远角小盒"))
            scene_input = {
                "label": "复杂遮挡轮廓演示",
                "nodes": [
                    {"key": "root", "pose": {"rotate": [0, 12, 0]}},
                    {"key": "back", "parent": "root", "mesh": back["id"],
                     "surface": {"color": [0.78, 0.8, 0.86], "unlit": True}},
                    {"key": "wedge", "parent": "root", "mesh": wedge["id"],
                     "pose": {"translate": [-0.15, -0.1, 0.2], "rotate": [0, -18, 0]},
                     "surface": {"color": [0.86, 0.42, 0.2]}},
                    {"key": "small", "parent": "root", "mesh": small["id"],
                     "surface": {"color": [0.85, 0.25, 0.2], "unlit": True}},
                    {"key": "far", "parent": "root", "mesh": far["id"],
                     "surface": {"color": [0.25, 0.55, 0.8], "unlit": True}},
                ],
                "lens": {"eye": [0, 0.6, 5], "target": [0, 0, 0], "near": 0.1, "far": 25},
            }
            scene = service.add_scene(scene_input)
            line_options = {
                "silhouette": True, "occlusion": True, "crease": True,
                "width_silhouette": 1.6, "width_occlusion": 1.3, "width_crease": 1.0,
                "color_silhouette": [0.02, 0.02, 0.03],
                "color_occlusion": [0.8, 0.16, 0.12],
                "color_crease": [0.12, 0.32, 0.8],
            }
            rendered = service.render({"scene": scene["id"], "width": 320, "height": 320,
                                       "samples": 3, "lines": line_options})
            names = ["color.png", "line-silhouette.png", "line-occlusion.png",
                     "line-crease.png", "line-layer.png", "line-overlay.png", "lines.npz"]
            for name in names:
                (output / name).write_bytes(service.artifacts.read(rendered, name))
            (output / "contour-report.json").write_text(
                json.dumps(rendered["details"]["lines"], ensure_ascii=False, indent=2))
            # Demonstrate author control: silhouette-only variant.
            only_sil = service.render({"scene": scene["id"], "width": 320, "height": 320,
                                       "samples": 3,
                                       "lines": {**line_options, "occlusion": False, "crease": False}})
            (output / "line-layer-silhouette-only.png").write_bytes(
                service.artifacts.read(only_sil, "line-layer.png"))
            print(json.dumps({"render": rendered["id"], "output": str(output),
                              "structures": rendered["details"]["lines"]["structures"]},
                             ensure_ascii=False, indent=2))
        finally:
            service.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("run-output/contours"))
    produce(parser.parse_args().output)
