"""Translucency observations: intersecting color sheets, solid wedge, occluder.

Writes real service artifacts (color/coverage/object-id PNG, channels and pixel
inspection JSON) so the blending, coverage and selection rules are auditable
without running the HTTP service. All inputs are generated in-process.
"""
import json
import tempfile
from pathlib import Path
import numpy as np
from gnomon.service import RendererService
from gnomon.samples import wedge_mesh, screen_mesh, transparency_scene, dense_stack_scene
from gnomon.channels import unpack_channels
from gnomon.raster import MAX_TRANSLUCENT_LAYERS


def produce(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gnomon-translucency-") as root:
        service = RendererService(root)
        try:
            wedge = service.add_mesh(wedge_mesh())["id"]
            screen = service.add_mesh(screen_mesh())["id"]
            config = {"width":160,"height":160,"samples":2}
            observations = {}
            for name,payload in [
                ("translucent-sheets",transparency_scene(wedge,screen)),
                ("reversed-node-order",transparency_scene(wedge,screen,reversed_order=True)),
                (f"dense-stack-{MAX_TRANSLUCENT_LAYERS+4}-layers",dense_stack_scene(screen,MAX_TRANSLUCENT_LAYERS+4)),
            ]:
                source = service.add_scene(payload)
                preflight = service.preflight({**config,"scene":source["id"]})
                record = service.render({**config,"scene":source["id"]})
                folder = output/name
                folder.mkdir(exist_ok=True)
                (folder/"scene.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2))
                for artifact in record["details"]["artifacts"]:
                    (folder/artifact).write_bytes(service.artifacts.read(record,artifact))
                channels = unpack_channels(service.artifacts.read(record,"channels.npz"))
                picked = np.argwhere(channels["object_id"] > 0)
                probes = []
                for index in (0,len(picked)//2,len(picked)-1):
                    y,x = picked[index]
                    probes.append(service.inspect(record["id"],int(x),int(y)))
                observations[name] = {"preflight":{key:preflight[key] for key in
                                  ("translucent","translucent_layer_cap","estimated_buffer_bytes","admissible")},
                                  "stats":record["details"]["metadata"]["stats"],
                                  "pixel_probes":probes,
                                  "max_stored_layers":int(channels["layer_count"].max()) if "layer_count" in channels else 0}
            (output/"translucency-observations.json").write_text(
                json.dumps(observations,ensure_ascii=False,indent=2,allow_nan=False))
        finally:
            service.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("run-output/translucency"))
    produce(parser.parse_args().output)
