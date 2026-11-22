"""Bounded production walkthroughs. No test framework or test suite is installed."""
import argparse
import json
import tempfile
from copy import deepcopy
from pathlib import Path
from .service import RendererService
from .samples import wedge_mesh,screen_mesh,checker_texture,compose_scene,obj_proof_sample,obj_flawed_sample
from .errors import RenderError,require
from .channels import unpack_channels

FLOWS = ["mesh-ingest","texture-ingest","scene-revise","scene-preflight","still-render","pixel-inspect","image-compare","view-sheet","render-bundle","resource-retire","obj-import"]


def expect_rejection(operation,code):
    try:
        operation()
    except RenderError as exc:
        require(exc.code==code,"walkthrough_result",f"expected {code}, received {exc.code}")
        return {"rejected":exc.code}
    raise RuntimeError("walkthrough expected a rejected operation")


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
        if flow in {"all","obj-import"}:
            import numpy as np
            sample = obj_proof_sample()
            preview = renderer.preview_import(sample)
            require(not any(item["severity"]=="warning" for item in preview["diagnostics"]),
                    "walkthrough_result","proof sample must preview without warnings")
            imported = renderer.import_model(sample)
            details = imported["details"]
            proof = renderer.render({"scene":details["scene"]["id"],"width":160,"height":100,"samples":1})
            metadata = proof["details"]["metadata"]
            view = np.asarray(metadata["view"]); projection = np.asarray(metadata["projection"])
            def probe(point):
                clip = projection @ view @ np.array([*point,1.0])
                ndc = clip[:3]/clip[3]
                return renderer.inspect(proof["id"],int((ndc[0]+1)/2*160),int((1-ndc[1])/2*100))
            dart = probe([-1.0,-0.2,0])
            require(dart["node"]=="dart-matte_red","walkthrough_result",f"dart node mismatch: {dart['node']}")
            require(abs(dart["world"][0]+1.0)<0.03 and abs(dart["world"][1]+0.2)<0.03 and abs(dart["world"][2])<1e-9,
                    "walkthrough_result",f"dart world coordinates drifted: {dart['world']}")
            require(np.allclose(dart["normal"],[0,0,1],atol=1e-6),"walkthrough_result","CCW dart must face +Z")
            notch = probe([-1.35,0.35,0])
            require(notch["object_id"]==0,"walkthrough_result","concave notch must stay background (fan triangulation would cover it)")
            winder = probe([0,0.9,0])
            require(np.allclose(winder["normal"],[0,0,-1],atol=1e-6),"walkthrough_result","clockwise winding must keep its -Z normal")
            flat = probe([-0.275,0,0]); tilted = probe([0.275,0,0])
            require(np.allclose(flat["normal"],[0,0,1],atol=1e-6),"walkthrough_result","explicit flat normal lost")
            require(np.allclose(tilted["normal"],[0.7071,0,0.7071],atol=1e-3),"walkthrough_result","independent vn indices not honored")
            red,blue = [0.5776,0.0113,0.0113],[0.0113,0.0113,0.5776]
            for point,expected in [([0.825,0,0],red),([1.075,0,0],blue),([1.325,0,0],red),([1.575,0,0],blue)]:
                got = probe(point)
                require(got["node"]=="seams-striped" and np.allclose(got["linear_rgb"],expected,atol=0.03),
                        "walkthrough_result",f"UV seam sampling wrong at {point}: {got['linear_rgb']}")
            meshes = details["meshes"]
            require(len(meshes)==5,"walkthrough_result","expected five group/material meshes")
            for entry in meshes.values():
                renderer.catalog.get(entry["id"],"mesh")
            bound = {node.get("mesh") for node in renderer.catalog.get(details["scene"]["id"],"scene")["payload"]["nodes"]} - {None}
            require(bound == {entry["id"] for entry in meshes.values()},"walkthrough_result","scene must bind the imported mesh records")
            texture_ids = {entry["id"] for entry in details["textures"].values()}
            require(len(texture_ids)==1 and all(renderer.catalog.get(identity,"texture") for identity in texture_ids),
                    "walkthrough_result","imported texture record missing")
            flawed = renderer.preview_import(obj_flawed_sample())
            codes = {item["code"] for item in flawed["diagnostics"]}
            require({"mtl_transparency","mtl_unsupported","mtl_missing_texture"} <= codes,
                    "walkthrough_result",f"flawed material diagnostics incomplete: {sorted(codes)}")
            committed = renderer.import_model(obj_flawed_sample())
            require(committed["state"]=="complete" and committed["details"]["diagnostics"],
                    "walkthrough_result","flawed import must commit with recorded diagnostics")
            no_library = renderer.preview_import({**sample,"mtl_text":None,"images":{}})
            require("obj_mtllib_unprovided" in {item["code"] for item in no_library["diagnostics"]},
                    "walkthrough_result","missing material library must be explained")
            results["obj-import"] = {"meshes":len(meshes),"triangles":details["plan"]["stats"]["triangles"],
                                     "vertices_out":details["plan"]["stats"]["vertices_out"],
                                     "positions_in":details["plan"]["stats"]["positions_in"],
                                     "diagnostics":sorted(codes),
                                     "rejections":[
                                         expect_rejection(lambda:renderer.preview_import({"label":"坏索引","obj_text":"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 9\n"}),"obj_index"),
                                         expect_rejection(lambda:renderer.preview_import({"label":"缺角","obj_text":"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2\n"}),"obj_face"),
                                         expect_rejection(lambda:renderer.preview_import({"label":"半UV","obj_text":"v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nf 1/1 2 3\n"}),"obj_uv_partial")]}
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
