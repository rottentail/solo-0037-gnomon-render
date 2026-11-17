"""Two reproducible observations from valid scenes; writes actual service outputs."""
import json
import tempfile
from copy import deepcopy
from pathlib import Path
from gnomon.service import RendererService
from gnomon.samples import screen_mesh


def material_scene(mesh):
    return {"label":"表面观测","nodes":[{"key":"panel","mesh":mesh,"surface":{"color":[0.8,0.45,0.2],"unlit":True}}],
            "lens":{"eye":[0,0,4],"target":[0,0,0],"near":0.1,"far":20}}


def produce(output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gnomon-observation-") as root:
        service=RendererService(root)
        try:
            mesh=service.add_mesh(screen_mesh())["id"]
            scene=material_scene(mesh)
            scene["nodes"].append({**deepcopy(scene["nodes"][0]),"key":"overlay"})
            reverse=deepcopy(scene);reverse["nodes"].reverse()
            tilted=material_scene(mesh)
            tilted["nodes"][0]["pose"]={"rotate":[0,35,0]}
            extra=deepcopy(tilted)
            extra["nodes"].append({"key":"remote","mesh":mesh,"pose":{"translate":[1.5,0,-3],"scale":[0.4,0.4,0.4]},"surface":{"unlit":True}})
            observations={}
            for name,payload in [("overlap-a",scene),("overlap-b",reverse),("depth-a",tilted),("depth-b",extra)]:
                source=service.add_scene(payload)
                rendered=service.render({"scene":source["id"],"width":128,"height":128,"samples":2})
                folder=output/name;folder.mkdir(exist_ok=True)
                (folder/"scene.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2))
                for artifact in rendered["details"]["artifacts"]:
                    (folder/artifact).write_bytes(service.artifacts.read(rendered,artifact))
                observations[name]={"center":service.inspect(rendered["id"],64,64),"metadata":rendered["details"]["metadata"]}
            (output/"observations.json").write_text(json.dumps(observations,ensure_ascii=False,indent=2))
        finally:
            service.close()


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("run-output/observations"))
    produce(parser.parse_args().output)
