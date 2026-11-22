"""Bounded production walkthroughs. No test framework or test suite is installed."""
import argparse
import json
import tempfile
from copy import deepcopy
from pathlib import Path
from .service import RendererService
from .samples import wedge_mesh,screen_mesh,checker_texture,compose_scene,box_mesh,face_material_scene
from .errors import RenderError,require
from .channels import unpack_channels

FLOWS = ["mesh-ingest","texture-ingest","scene-revise","scene-preflight","still-render","pixel-inspect","image-compare","view-sheet","render-bundle","resource-retire","face-material"]


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
        if flow in {"all","face-material"}:
            results["face-material"] = face_material_flow(renderer,output)
        if output:
            for name in rendered["details"]["artifacts"]:
                Path(output,name).write_bytes(renderer.artifacts.read(rendered,name))
        return results
    finally:
        renderer.close()


def face_material_flow(renderer,output=None):
    """Grouped, seam-aware faces on one closed mesh, plus their rejections."""
    import numpy as np
    box = renderer.add_mesh(box_mesh())
    texture = renderer.add_texture(checker_texture())
    scene_input = face_material_scene(box["id"],texture["id"])
    scene = renderer.add_scene(scene_input)
    rendered = renderer.render({"scene":scene["id"],"width":200,"height":200,"samples":1})
    channels = unpack_channels(renderer.artifacts.read(rendered,"channels.npz"))
    covered = channels["object_id"]>0
    require(box["payload"]["triangle_count"]==12 and box["payload"]["seams"]=={"normal":True,"uv":True},"face_material","box must keep welded topology with corner seams")
    require(rendered["details"]["metadata"]["stats"]["input_triangles"]==12,"face_material","face groups must not split triangles")
    group = {}
    for wanted in (7,8,10):
        mask = covered&(channels["face_id"]==wanted)
        require(bool(mask.any()),"face_material",f"face {wanted} is not visible")
        y,x = map(int,np.argwhere(mask)[len(np.argwhere(mask))//2])
        group[wanted] = renderer.inspect(rendered["id"],x,y)
    require(group[8]["material"]=="grille" and group[10]["material"]=="cavity","face_material","face groups must drive the material shown at a pixel")
    require(group[7]["material"]=="default","face_material","unlisted faces must keep the legacy surface")
    require(group[8]["node"]=="box" and group[8]["mesh"]==box["id"],"face_material","picking must return the original node, mesh and face")
    require(set(np.unique(channels["material_id"][covered]).tolist())=={0,1,2},"face_material","all three materials must be visible")
    group_scene = face_material_scene(box["id"],texture["id"])
    out_of_range = deepcopy(group_scene); out_of_range["nodes"][0]["material_groups"][0]["faces"]=[8,99]
    overlap = deepcopy(group_scene); overlap["nodes"][0]["material_groups"][1]["faces"]=[8,11]
    duplicate = deepcopy(group_scene); duplicate["nodes"][0]["material_groups"][1]["name"]="grille"
    floating = deepcopy(group_scene)
    floating["nodes"].append({"key":"orphan","material_groups":[{"name":"loose","faces":[0],"surface":{}}]})
    untextured = renderer.add_mesh({k:v for k,v in box_mesh().items() if k!="corner_uv"})
    missing_uv = face_material_scene(untextured["id"],texture["id"])
    if output:
        Path(output,"face-material-color.png").write_bytes(renderer.artifacts.read(rendered,"color.png"))
    return {"scene":scene["id"],"render":rendered["state"],"materials":rendered["details"]["metadata"]["materials"],
            "pixels":group,
            **expect_rejection(lambda:renderer.add_scene(out_of_range),"material_face"),
            **expect_rejection(lambda:renderer.add_scene(overlap),"material_overlap"),
            **expect_rejection(lambda:renderer.add_scene(duplicate),"material_group"),
            **expect_rejection(lambda:renderer.add_scene(floating),"material_group"),
            **expect_rejection(lambda:renderer.add_scene(missing_uv),"texture_uv")}


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
