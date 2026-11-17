"""Run after starting uvicorn; uses actual HTTP endpoints."""
from pathlib import Path
import httpx
from gnomon.samples import wedge_mesh,screen_mesh,checker_texture,compose_scene

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
