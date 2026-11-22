"""Bounded production walkthroughs. No test framework or test suite is installed."""
import argparse
import json
import tempfile
from copy import deepcopy
from pathlib import Path
import numpy as np
from .service import RendererService
from .samples import (wedge_mesh, screen_mesh, checker_texture, compose_scene,
                      bevel_block_mesh, emblem_texture, decal_scene)
from .errors import RenderError, require
from .channels import unpack_channels
from .decal import projection_frame, project_uv

FLOWS = ["mesh-ingest","texture-ingest","scene-revise","scene-preflight","still-render","pixel-inspect","image-compare","view-sheet","render-bundle","resource-retire","projected-decal"]


def expect_rejection(operation,code):
    try:
        operation()
    except RenderError as exc:
        require(exc.code==code,"walkthrough_result",f"expected {code}, received {exc.code}")
        return {"rejected":exc.code}
    raise RuntimeError("walkthrough expected a rejected operation")


def decal_walkthrough(renderer, output):
    block = renderer.add_mesh(bevel_block_mesh())
    occluder = renderer.add_mesh(screen_mesh())
    emblem = renderer.add_texture(emblem_texture())
    scene_input = decal_scene(block["id"], emblem["id"], occluder["id"])
    scene = renderer.add_scene(scene_input)
    width, height, samples = 160, 120, 2
    request = {"scene": scene["id"], "width": width, "height": height, "samples": samples}
    record = renderer.render(request)
    channels = unpack_channels(renderer.artifacts.read(record, "channels.npz"))
    object_id, decal, world = channels["object_id"], channels["decal"], channels["world"]
    node_map = record["details"]["metadata"]["node_map"]
    id_by_key = {key: int(oid) for oid, key in node_map.items()}
    block_id, plain_id, mirror_id, occluder_id = (id_by_key[key] for key in
                                                  ("block", "plain_twin", "mirror_twin", "occluder"))
    covered = decal > 0.01

    # 1. The emblem lands and is an overlay: object identity stays the surface node.
    require(covered.any(), "decal_empty", "projection produced no emblem coverage")
    require(set(np.unique(object_id[covered]).tolist()) <= {block_id, mirror_id},
            "decal_identity", "decal must not change the picked node identity")

    # 2. Shared mesh isolation: the twin without a decal is completely untouched.
    require(not np.any(covered & (object_id == plain_id)), "decal_shared_mesh",
            "decal leaked onto another node sharing the same mesh")

    # 3. The mirrored instance receives its own projector and faces the camera.
    mirror_pixels = covered & (object_id == mirror_id)
    require(mirror_pixels.any(), "decal_mirror", "mirrored instance shows no emblem")
    mirror_forward = np.array(scene_input["nodes"][2]["decal"]["direction"], dtype=float)
    mirror_forward /= np.linalg.norm(mirror_forward)
    facing = -(channels["normal"][mirror_pixels] @ mirror_forward)
    require(bool(np.all(facing > 0)), "decal_backface", "emblem reached a back-facing surface")

    # 4. No penetration: nothing is painted where the occluder owns the depth.
    require(not np.any(covered & (object_id == occluder_id)), "decal_occluder_id",
            "decal overwrote the occluder node")
    occluder_zone = object_id == occluder_id
    require(not np.any(decal[occluder_zone] > 0.01), "decal_penetration",
            "emblem penetrated through the occluding surface")

    # 5. One world-space scale across adjacent faces. The emblem texture itself
    #    has transparent margins, and the block is narrower than the 3.0 projector
    #    frame, so scale is verified in projector UV space: UV is a linear map of
    #    world distance with one fixed scale for every face the projection crosses.
    frame = projection_frame(scene_input["nodes"][0]["decal"])
    block_pts = world[covered & (object_id == block_id)]
    puv, psu, psv, _ = project_uv(frame, block_pts)
    expect_u = (psu / 1.5 + 1.0) / 2.0
    expect_v = (psv / 1.5 + 1.0) / 2.0
    require(bool(np.allclose(puv[:, 0], expect_u, atol=1e-12))
            and bool(np.allclose(puv[:, 1], expect_v, atol=1e-12)),
            "decal_scale", "projector UV is not a single linear world-space map")
    extent_u = block_pts[:, 0].max() - block_pts[:, 0].min()
    extent_v = block_pts[:, 1].max() - block_pts[:, 1].min()

    # 6. Cross-face coverage: front wall (z close to +1) and the 45-degree
    #    chamfer both carry the same projector.
    block_cover = covered & (object_id == block_id)
    on_front = block_cover & (world[:, :, 2] > 0.95)
    on_chamfer = block_cover & (world[:, :, 2] <= 0.95)
    require(on_front.any() and on_chamfer.any(), "decal_cross_face",
            "emblem must continue from the front wall onto the chamfer")

    # 7. User-controlled grazing termination: hard cuts at the cutoff, fade
    #    ramps instead of a binary edge. Facing cosine is -n.forward and is
    #    positive for surfaces pointing back toward the projector.
    projector_forward = np.array(scene_input["nodes"][0]["decal"]["direction"], dtype=float)
    projector_forward /= np.linalg.norm(projector_forward)
    hard_facing = -(channels["normal"][covered] @ projector_forward)
    require(bool(np.all(hard_facing >= 0.25 - 1e-9)), "decal_grazing_hard",
            "hard grazing cutoff accepted a fragment beyond the limit")
    faded_input = deepcopy(scene_input)
    faded_input["nodes"][0]["decal"] = {**faded_input["nodes"][0]["decal"],
                                        "grazing": 0.25, "grazing_mode": "fade"}
    faded_scene = renderer.add_scene(faded_input, scene["id"])
    faded_record = renderer.render({**request, "scene": faded_scene["id"]})
    faded = unpack_channels(renderer.artifacts.read(faded_record, "channels.npz"))
    faded_mask = faded["decal"]
    fade_vals = faded_mask[faded_mask > 0]
    require(bool(np.any((fade_vals > 0.02) & (fade_vals < 0.98))), "decal_grazing_fade",
            "fade mode produced no intermediate edge weights")

    # 8. Pixel inspection reports the actual decal coverage on the surface.
    ys, xs = np.nonzero(covered & (object_id == block_id))
    inspection = renderer.inspect(record["id"], int(xs[len(xs) // 2]), int(ys[len(ys) // 2]))
    require(inspection["decal_coverage"] > 0 and inspection["node"] == "block", "decal_inspect",
            "pixel inspection must expose decal coverage and the owning node")

    # 9. A decal without a mesh target is a domain rejection.
    bad = deepcopy(scene_input)
    bad["nodes"].append({"key": "floating", "decal": bad["nodes"][0]["decal"]})
    expect_rejection(lambda: renderer.add_scene(bad), "decal_target")

    summary = {"state": record["state"],
               "covered_pixels": int(covered.sum()),
               "block_pixels": int((covered & (object_id == block_id)).sum()),
               "mirror_pixels": int(mirror_pixels.sum()),
               "plain_twin_pixels": int((covered & (object_id == plain_id)).sum()),
               "occluded_pixels": int(occluder_zone.sum()),
               "projected_world_extent": [round(float(extent_u), 4), round(float(extent_v), 4)],
               "fade_intermediate_pixels": int(np.sum((fade_vals > 0.02) & (fade_vals < 0.98))),
               "inspection": inspection,
               "stats": record["details"]["metadata"]["stats"],
               **expect_rejection(lambda: renderer.add_scene(bad), "decal_target")}
    if output:
        folder = Path(output, "decal")
        folder.mkdir(parents=True, exist_ok=True)
        for name in record["details"]["artifacts"]:
            (folder / name).write_bytes(renderer.artifacts.read(record, name))
        (folder / "faded-decal.png").write_bytes(renderer.artifacts.read(faded_record, "decal.png"))
        (folder / "scene.json").write_text(json.dumps(scene_input, ensure_ascii=False, indent=2))
    return summary


def walkthrough(root,flow="all",output=None):
    renderer = RendererService(root)
    results = {}
    try:
        mesh = renderer.add_mesh(wedge_mesh())
        screen = renderer.add_mesh(screen_mesh())
        texture = renderer.add_texture(checker_texture())
        scene_input = compose_scene(mesh["id"],screen["id"],texture["id"])
        scene = renderer.add_scene(scene_input)
        request = {"scene":scene["id"],"width":96,"height":80,"samples":2}
        if flow in {"all","mesh-ingest"}:
            bad = wedge_mesh(); bad["triangles"][0] = [0,0,1]
            results["mesh-ingest"] = {"triangles":mesh["payload"]["triangle_count"],**expect_rejection(lambda:renderer.add_mesh(bad),"mesh_degenerate")}
        if flow in {"all","texture-ingest"}:
            results["texture-ingest"] = {"dimensions":[texture["payload"]["width"],texture["payload"]["height"]],**expect_rejection(lambda:renderer.add_texture({"label":"bad","png_base64":"broken"}),"texture_decode")}
        if flow in {"all","scene-revise"}:
            revised = deepcopy(scene_input); revised["nodes"][1]["pose"]["rotate"][1]=25
            revision = renderer.add_scene(revised,scene["id"])
            bad = deepcopy(scene_input); bad["nodes"][0]["parent"]="wedge"
            results["scene-revise"] = {"independent_digests":scene["digest"]!=revision["digest"],**expect_rejection(lambda:renderer.add_scene(bad),"parent_cycle")}
        if flow in {"all","scene-preflight"}:
            normal = renderer.preflight(request)
            large = renderer.preflight({**request,"width":1024,"height":1024,"samples":3})
            results["scene-preflight"] = {"normal":normal,"oversized_admissible":large["admissible"]}
        rendered = renderer.render(request)
        if flow in {"all","still-render"}:
            crop = renderer.render({**request,"region":[16,12,48,40]})
            full = unpack_channels(renderer.artifacts.read(rendered,"channels.npz"))
            cut = unpack_channels(renderer.artifacts.read(crop,"channels.npz"))
            import numpy as np
            require(np.array_equal(full["color"][12:52,16:64],cut["color"]),"crop_semantics","region differs from full frame")
            results["still-render"] = {"state":rendered["state"],"stats":rendered["details"]["metadata"]["stats"],"region_matches":True}
        if flow in {"all","pixel-inspect"}:
            channels = unpack_channels(renderer.artifacts.read(rendered,"channels.npz"))
            import numpy as np
            positions = np.argwhere(channels["object_id"]>0)
            require(len(positions)>0,"empty_demo","demo must contain visible geometry")
            y,x = positions[len(positions)//2]
            results["pixel-inspect"] = {"pixel":renderer.inspect(rendered["id"],int(x),int(y)),**expect_rejection(lambda:renderer.inspect(rendered["id"],999,0),"pixel_bounds")}
        if flow in {"all","image-compare"}:
            revised = deepcopy(scene_input); revised["nodes"][1]["surface"]["color"]=[0.1,0.3,0.9]
            next_scene = renderer.add_scene(revised,scene["id"])
            other = renderer.render({**request,"scene":next_scene["id"]})
            comparison = renderer.compare({"left":rendered["id"],"right":other["id"]})
            require(comparison["details"]["metrics"]["changed_pixels"]>0,"comparison_demo","color revision must be visible")
            differently_sized = renderer.render({**request,"width":64})
            results["image-compare"] = {"metrics":comparison["details"]["metrics"],**expect_rejection(lambda:renderer.compare({"left":rendered["id"],"right":differently_sized["id"]}),"comparison_camera")}
        if flow in {"all","view-sheet"}:
            sheet = renderer.sheet({"scene":scene["id"],"size":64,"angles":[0,60,150,240],"columns":2})
            results["view-sheet"] = {"state":sheet["state"],"views":sheet["details"]["views"]}
            if output:
                Path(output,"sheet.png").write_bytes(renderer.artifacts.read(sheet,"sheet.png"))
        if flow in {"all","render-bundle"}:
            bundle = renderer.bundle(rendered["id"])
            results["render-bundle"] = bundle["details"]
            if output:
                Path(output,"render.zip").write_bytes(renderer.artifacts.read(bundle,"render.zip"))
        if flow in {"all","resource-retire"}:
            renderer.resource_state(mesh["id"],"retired")
            again = renderer.render(request)
            results["resource-retire"] = {"old_revision_render":again["state"],**expect_rejection(lambda:renderer.add_scene(scene_input),"resource_retired")}
            renderer.resource_state(mesh["id"],"active")
            renderer.scene_state(scene["id"],"archived")
            expect_rejection(lambda:renderer.render(request),"scene_archived")
            renderer.scene_state(scene["id"],"active")
        if flow in {"all","projected-decal"}:
            results["projected-decal"] = decal_walkthrough(renderer,output)
        if output:
            for name in rendered["details"]["artifacts"]:
                Path(output,name).write_bytes(renderer.artifacts.read(rendered,name))
        return results
    finally:
        renderer.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow",choices=["all",*FLOWS],default="all")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.output:
        args.output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gnomon-walkthrough-") as root:
        results=walkthrough(root,args.flow,args.output)
    print(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
