"""Keyframe still-frame sequence example.

Builds a small parent/child hierarchy, animates both nodes and the camera
across the +/-180 degree boundary, and renders one sequence per interpolation
mode so the continuity differences can be inspected as contact sheets.

    python examples/keyframe_sequence.py --output run-output/sequences
"""
import argparse
import json
import tempfile
from pathlib import Path

from gnomon.service import RendererService
from gnomon.samples import wedge_mesh, screen_mesh


def build_scene(service):
    mesh = service.add_mesh(wedge_mesh())["id"]
    screen = service.add_mesh(screen_mesh())["id"]
    payload = {
        "label": "关键帧序列示例",
        "nodes": [
            {"key": "turntable", "pose": {}},
            {"key": "pendulum", "parent": "turntable", "mesh": mesh,
             "surface": {"color": [0.87, 0.38, 0.19]},
             "pose": {"translate": [0.6, 0.0, 0.0]}},
            {"key": "card", "parent": "turntable", "mesh": screen,
             "surface": {"color": [0.2, 0.7, 0.8], "unlit": True},
             "pose": {"translate": [-0.7, 0.3, -0.6], "scale": [0.5, 0.5, 0.5]}},
        ],
        "lens": {"eye": [3, 2.0, 4], "target": [0, 0, 0], "near": 0.1, "far": 30},
        "light": {"toward": [1, 3, 4], "energy": 0.85, "ambient": 0.18},
    }
    return service.add_scene(payload)["id"]


def sequence_request(scene_id, mode):
    # 179 deg -> -179 deg is a 2-degree short turn, not a 358-degree sweep.
    return {
        "scene": scene_id,
        "width": 128, "height": 112, "samples": 1,
        "duration": 1.0,
        "frame_times": [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
        "columns": 3,
        "node_tracks": [
            {"node": "turntable", "interpolation": mode,
             "keyframes": [{"time": 0.0, "rotate": [0, 179, 0]},
                           {"time": 0.5, "rotate": [0, -179, 0]},
                           {"time": 1.0, "rotate": [0, 179, 0]}]},
            {"node": "pendulum", "interpolation": mode,
             "keyframes": [{"time": 0.0, "rotate": [0, 0, -25]},
                           {"time": 0.5, "rotate": [0, 0, 25]},
                           {"time": 1.0, "rotate": [0, 0, -25]}]},
        ],
        "camera_track": {"interpolation": mode,
                         "keyframes": [{"time": 0.0, "eye": [3, 2.0, 4]},
                                       {"time": 0.5, "eye": [4.5, 1.0, 2.5]},
                                       {"time": 1.0, "eye": [3, 2.0, 4]}]},
    }


def produce(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gnomon-sequence-") as root:
        service = RendererService(root)
        try:
            scene_id = build_scene(service)
            summary = {}
            for mode in ("step", "linear", "smooth", "smooth_tangent"):
                record = service.sequence(sequence_request(scene_id, mode))
                folder = output / mode
                folder.mkdir(exist_ok=True)
                for name in record["details"]["artifacts"]:
                    (folder / name).write_bytes(service.artifacts.read(record, name))
                # Re-render one frame on its own and confirm byte equality.
                middle = len(record["details"]["manifest"]["frames"]) // 2
                redo = service.rerender_frame(record["id"], middle)
                for name in redo["details"]["artifacts"]:
                    assert service.artifacts.read(redo, name) == service.artifacts.read(record, name)
                summary[mode] = {"sequence": record["id"], "frames": record["details"]["frames"],
                                 "contact_sheet": str(folder / "contact-sheet.png"),
                                 "single_frame_redo": "byte-identical"}
            (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
            print(json.dumps({mode: item["contact_sheet"] for mode, item in summary.items()},
                             ensure_ascii=False, indent=2))
        finally:
            service.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("run-output/sequences"))
    produce(parser.parse_args().output)
