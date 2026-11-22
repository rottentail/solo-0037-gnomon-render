"""Run after starting uvicorn; uses actual HTTP endpoints."""
from pathlib import Path
import httpx
from gnomon.samples import wedge_mesh,screen_mesh,checker_texture,compose_scene,obj_proof_sample

with httpx.Client(base_url="http://127.0.0.1:8737",timeout=120) as client:
    def post(path,payload):
        response=client.post(path,json=payload)
        response.raise_for_status()
        return response.json()
    mesh=post("/v1/meshes",wedge_mesh())
    screen=post("/v1/meshes",screen_mesh())
    texture=post("/v1/textures",checker_texture())
    scene=post("/v1/scenes",compose_scene(mesh["id"],screen["id"],texture["id"]))
    render=post("/v1/renders",{"scene":scene["id"],"width":256,"height":256,"samples":2})
    response=client.get(f"/v1/artifacts/{render['id']}/color.png")
    response.raise_for_status()
    output=Path("run-output")
    output.mkdir(exist_ok=True)
    (output/"http-color.png").write_bytes(response.content)
    print({"scene":scene["id"],"render":render["id"],"state":render["state"],"output":str(output/"http-color.png")})
    # OBJ/MTL 导入:预览解释计划与诊断,提交后得到可追溯的网格、纹理和场景。
    preview=post("/v1/imports/preview",obj_proof_sample())
    print({"import_preview":preview["plan"]["stats"],"diagnostics":preview["diagnostics"]})
    imported=post("/v1/imports",obj_proof_sample())
    imported_render=post("/v1/renders",{"scene":imported["details"]["scene"]["id"],"width":160,"height":100,"samples":1})
    response=client.get(f"/v1/artifacts/{imported_render['id']}/color.png")
    response.raise_for_status()
    (output/"http-import-color.png").write_bytes(response.content)
    print({"import":imported["id"],"meshes":list(imported["details"]["meshes"]),
           "render":imported_render["id"],"output":str(output/"http-import-color.png")})
