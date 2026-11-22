"""Produce a real stereo delivery: pair, correspondence maps and cross-check.

Run: .venv/bin/python examples/stereo_observation.py --output run-output/stereo
"""
import argparse
import json
import tempfile
from io import BytesIO
from pathlib import Path
import numpy as np
from gnomon.service import RendererService
from gnomon.samples import wedge_mesh, screen_mesh, stereo_scene, stereo_rig


def produce(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gnomon-stereo-") as root:
        service = RendererService(root)
        try:
            mesh = service.add_mesh(wedge_mesh())
            screen = service.add_mesh(screen_mesh())
            payload = stereo_scene(mesh["id"], screen["id"])
            scene = service.add_scene(payload)
            rendered = service.render({"scene": scene["id"], "width": 192, "height": 144,
                                       "samples": 2, "stereo": stereo_rig()})
            for name in rendered["details"]["artifacts"]:
                (output / name).write_bytes(service.artifacts.read(rendered, name))
            metadata = rendered["details"]["metadata"]
            with np.load(BytesIO(service.artifacts.read(rendered, "correspondence.npz")),
                         allow_pickle=False) as archive:
                occluded = np.argwhere(archive["left_to_right_status"] == 2)
            queries = {}
            if len(occluded):
                y, x = (int(value) for value in occluded[0])
                queries["occluded_left"] = service.stereo_correspondence(rendered["id"], "left", x, y)
                counterpart = queries["occluded_left"]["counterpart"]["pixel"]
                queries["same_occluded_pixel_seen_from_right"] = service.stereo_correspondence(
                    rendered["id"], "right", counterpart[0], counterpart[1])
            summary = metadata["correspondence"]["directions"]
            (output / "observation.json").write_text(json.dumps({
                "scene": payload,
                "rig": metadata["units"],
                "rig_checks": metadata["rig_checks"],
                "left_to_right": {key: summary["left_to_right"][key] for key in
                                  ("status_counts", "horizontal_parallax_pixels",
                                   "vertical_parallax_pixels", "parallax_depth_fit", "per_object")},
                "right_to_left": {key: summary["right_to_left"][key] for key in
                                  ("status_counts", "horizontal_parallax_pixels",
                                   "vertical_parallax_pixels")},
                "probe_checks": metadata["correspondence"]["probe_checks"],
                "queries": queries,
            }, ensure_ascii=False, indent=2, allow_nan=False))
            print(json.dumps({"render": rendered["id"], "state": rendered["state"],
                              "artifacts": sorted(rendered["details"]["artifacts"]),
                              "output": str(output)}, ensure_ascii=False))
        finally:
            service.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("run-output/stereo"))
    produce(parser.parse_args().output)
