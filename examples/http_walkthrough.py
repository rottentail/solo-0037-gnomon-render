"""Run after starting uvicorn; uses actual HTTP endpoints."""
from pathlib import Path
import httpx
from gnomon.samples import wedge_mesh,screen_mesh,checker_texture,compose_scene,stereo_scene,stereo_rig

with httpx.Client(base_url="http://127.0.0.1:8737",timeout=120) as client:
    def post(path,payload):
        response=client.post(path,json=payload)
        response.raise_for_status()
        return response.json()
    output=Path("run-output")
    output.mkdir(exist_ok=True)
    mesh=post("/v1/meshes",wedge_mesh())
    screen=post("/v1/meshes",screen_mesh())
    texture=post("/v1/textures",checker_texture())
    scene=post("/v1/scenes",compose_scene(mesh["id"],screen["id"],texture["id"]))
    render=post("/v1/renders",{"scene":scene["id"],"width":256,"height":256,"samples":2})
    response=client.get(f"/v1/artifacts/{render['id']}/color.png")
    response.raise_for_status()
    (output/"http-color.png").write_bytes(response.content)
    pair=post("/v1/renders",{"scene":scene["id"],"width":192,"height":144,"samples":1,"stereo":stereo_rig()})
    for name in ("pair.png","parallax.png","crosscheck.json"):
        artifact=client.get(f"/v1/artifacts/{pair['id']}/{name}")
        artifact.raise_for_status()
        (output/f"http-{name}").write_bytes(artifact.content)
    maps=client.get(f"/v1/artifacts/{pair['id']}/correspondence.npz")
    maps.raise_for_status()
    (output/"http-correspondence.npz").write_bytes(maps.content)
    query=client.get(f"/v1/renders/{pair['id']}/correspondence/left/96/72")
    query.raise_for_status()
    print({"scene":scene["id"],"render":render["id"],"state":render["state"],
           "stereo":pair["id"],"stereo_state":pair["state"],
           "stereo_query_status":query.json()["status"],"output":str(output)})
