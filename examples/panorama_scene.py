"""Reproducible equirectangular 360x180 observation from an enclosed scene.

The output is a receiver-ready 2:1 longitude/latitude map with ray distance
depth and a per-pixel world ray, plus inspection probes that map clicks back
onto the original scene. Runs entirely offline through the service API.
"""
import argparse
import json
import tempfile
from pathlib import Path

from gnomon.service import RendererService
from gnomon.samples import wedge_mesh, checker_texture, room_mesh, panorama_scene
from gnomon.channels import unpack_channels


def produce(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gnomon-panorama-") as root:
        service = RendererService(root)
        try:
            wedge = service.add_mesh(wedge_mesh())
            room = service.add_mesh(room_mesh())
            texture = service.add_texture(checker_texture())
            source = service.add_scene(panorama_scene(room["id"], wedge["id"], texture["id"]))
            request = {"scene": source["id"], "width": 512, "height": 256, "samples": 2}
            preflight = service.preflight(request)
            rendered = service.render(request)
            for artifact in rendered["details"]["artifacts"]:
                (output / artifact).write_bytes(service.artifacts.read(rendered, artifact))
            channels = unpack_channels(service.artifacts.read(rendered, "channels.npz"))
            metadata = rendered["details"]["metadata"]
            probes = [service.inspect(rendered["id"], x, y)
                      for x, y in ((256, 128), (0, 128), (511, 128), (256, 0), (256, 255))]
            report = {"projection": metadata["lens_projection"],
                      "depth_convention": metadata["depth_convention"],
                      "preflight": preflight, "stats": metadata["stats"],
                      "seam_id_agreement": float((channels["object_id"][:, 0]
                                                  == channels["object_id"][:, -1]).mean()),
                      "probes": probes}
            (output / "panorama-report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        finally:
            service.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("run-output/panorama"))
    produce(parser.parse_args().output)
