"""Bounded production walkthroughs. No test framework or test suite is installed."""
import argparse
import json
import tempfile
from copy import deepcopy
from pathlib import Path
from .service import RendererService
from .samples import wedge_mesh,screen_mesh,checker_texture,compose_scene,arc_array_scene,refined_mesh
from .errors import RenderError,require
from .channels import unpack_channels

FLOWS = ["mesh-ingest","texture-ingest","scene-revise","scene-preflight","still-render","pixel-inspect","image-compare","view-sheet","render-bundle","resource-retire","instance-array"]


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
        if flow in {"all","instance-array"}:
            import numpy as np
            from .scene import compile_scene
            from .math3d import pose_matrix, transform_points
            arc_payload = arc_array_scene(mesh["id"])
            arc = renderer.add_scene(arc_payload)
            validated = arc["payload"]  # defaults filled by schema validation
            # Shared geometry; instance pose composes with the parent chain; surface override/inheritance.
            compiled = compile_scene(validated,renderer.catalog.get)
            require(len(compiled["objects"])==12,"instance_array","arc expands to one object per instance")
            require(all(obj["triangles"] is compiled["objects"][0]["triangles"] for obj in compiled["objects"]),"instance_array","instances must share the same mesh arrays")
            require(compiled["objects"][0]["surface"]["color"]==[0.87,0.38,0.19],"instance_array","instance without surface inherits the node surface")
            parent = pose_matrix(validated["nodes"][0]["pose"]) @ pose_matrix(validated["nodes"][1]["pose"])
            local = np.asarray(mesh["payload"]["positions"])
            for obj in compiled["objects"]:
                slot = int(obj["instance"].split("-")[1])-1
                expected = transform_points(local,parent @ pose_matrix(validated["nodes"][1]["instances"][slot]["pose"]))
                require(np.allclose(obj["positions"],expected),"instance_array","instance world pose must combine array-local and parent transforms")
            # Preflight reflects the expanded scene, not the single source node.
            arc_request = {"scene":arc["id"],"width":96,"height":80,"samples":1}
            pre = renderer.preflight(arc_request)
            require(pre["visible_objects"]==12 and pre["instances"]==12 and pre["triangles"]==96,"instance_array","preflight must count expanded instances")
            shot = renderer.render(arc_request)
            ch_a = unpack_channels(renderer.artifacts.read(shot,"channels.npz"))
            names = {int(key):value for key,value in shot["details"]["metadata"]["node_map"].items()}
            id7 = next(key for key,value in names.items() if value=="ring/blade-07")
            pixels = np.argwhere(ch_a["object_id"]==id7)
            require(len(pixels)>0,"instance_array","instance must be pickable in the frame")
            y,x = pixels[len(pixels)//2]
            picked = renderer.inspect(shot["id"],int(x),int(y))
            require(picked["node"]=="ring/blade-07" and picked["instance"]=="blade-07","instance_array","pick must identify the exact instance")
            # A revision hides blade-03 and recolors blade-05; everything else must stay identical.
            modified = deepcopy(arc_payload)
            modified["nodes"][1]["instances"][2]["visible"] = False
            modified["nodes"][1]["instances"][4]["surface"] = {"color":[0.05,0.2,0.85]}
            revision = renderer.add_scene(modified,arc["id"])
            shot_b = renderer.render({**arc_request,"scene":revision["id"]})
            ch_b = unpack_channels(renderer.artifacts.read(shot_b,"channels.npz"))
            id3 = next(key for key,value in names.items() if value=="ring/blade-03")
            id5 = next(key for key,value in names.items() if value=="ring/blade-05")
            names_b = shot_b["details"]["metadata"]["node_map"]
            require(names_b.get(str(id7))=="ring/blade-07" and "ring/blade-03" not in names_b.values(),"instance_array","hiding one instance must not renumber others")
            require(not np.any(ch_b["object_id"]==id3),"instance_array","hidden instance must vanish from the frame")
            untouched = (ch_a["object_id"]!=id3)&(ch_a["object_id"]!=id5)
            require(np.array_equal(ch_a["color"][untouched],ch_b["color"][untouched]),"instance_array","editing one instance must not alter other instances")
            require(np.array_equal(ch_a["object_id"][untouched],ch_b["object_id"][untouched]),"instance_array","editing one instance must not move other instances")
            mask5 = ch_a["object_id"]==id5
            require(mask5.sum()>0 and not np.array_equal(ch_a["color"][mask5],ch_b["color"][mask5]),"instance_array","recolored instance must actually change")
            metrics = renderer.compare({"left":shot["id"],"right":shot_b["id"]})["details"]["metrics"]
            require(metrics["changed_pixels"]>0,"instance_array","instance revision must be measurable")
            # Historical frames stay frozen; re-rendering the original scene reproduces them exactly.
            frozen = unpack_channels(renderer.artifacts.read(shot,"channels.npz"))
            require(np.array_equal(frozen["color"],ch_a["color"]),"instance_array","stored historical render must stay intact")
            again = renderer.render(arc_request)
            ch_c = unpack_channels(renderer.artifacts.read(again,"channels.npz"))
            require(np.array_equal(ch_c["color"],ch_a["color"]) and np.array_equal(ch_c["object_id"],ch_a["object_id"]),"instance_array","re-render of the original scene must be identical")
            # At scale, preflight reports the real expanded cost and the budget verdict follows it.
            dense = renderer.add_mesh(refined_mesh(wedge_mesh(),3))
            large = renderer.add_scene(arc_array_scene(dense["id"],100))
            pre_large = renderer.preflight({**arc_request,"scene":large["id"]})
            require(pre_large["instances"]==100 and pre_large["triangles"]==51200 and pre_large["admissible"],"instance_array","large array within budget must be admissible")
            over = renderer.add_scene(arc_array_scene(dense["id"],128))
            pre_over = renderer.preflight({**arc_request,"scene":over["id"]})
            require(pre_over["triangles"]==65536 and not pre_over["admissible"],"instance_array","over-budget array must be inadmissible")
            reject_over = expect_rejection(lambda:renderer.render({**arc_request,"scene":over["id"]}),"triangle_budget")
            duplicate = arc_array_scene(mesh["id"],4)
            duplicate["nodes"][1]["instances"][1]["key"] = "blade-01"
            reject_dup = expect_rejection(lambda:renderer.add_scene(duplicate),"instance_duplicate")
            crowded = arc_array_scene(mesh["id"],128)
            wing = deepcopy(crowded["nodes"][1]); wing["key"] = "ring-b"
            crowded["nodes"] += [wing,{"key":"single","mesh":mesh["id"]}]
            reject_cap = expect_rejection(lambda:renderer.add_scene(crowded),"instance_budget")
            results["instance-array"] = {"instances":pre["instances"],"triangles":pre["triangles"],"picked":picked["node"],
                "revision_changed_pixels":metrics["changed_pixels"],"others_untouched":True,"history_reproduced":True,
                "large_array":{"instances":pre_large["instances"],"triangles":pre_large["triangles"],"admissible":pre_large["admissible"]},
                "over_budget":{"triangles":pre_over["triangles"],"admissible":pre_over["admissible"],**reject_over},
                "rejections":[reject_dup["rejected"],reject_cap["rejected"]]}
            if output:
                Path(output,"arc-color.png").write_bytes(renderer.artifacts.read(shot,"color.png"))
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
