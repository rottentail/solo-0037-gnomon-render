"""Bounded production walkthroughs. No test framework or test suite is installed."""
import argparse
import json
import tempfile
from copy import deepcopy
from pathlib import Path
from pydantic import ValidationError
from .service import RendererService
from .samples import wedge_mesh,screen_mesh,checker_texture,compose_scene,slats_mesh,lattice_texture,depth_scene
from .errors import RenderError,require
from .channels import unpack_channels
from .color import srgb_to_linear

FLOWS = ["mesh-ingest","texture-ingest","scene-revise","scene-preflight","still-render","pixel-inspect","image-compare","view-sheet","render-bundle","resource-retire","focus-render"]


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
        if flow in {"all","focus-render"}:
            import numpy as np
            def erode(mask,rounds):
                mask = mask.copy()
                for _ in range(rounds):
                    mask[[0,-1],:] = False; mask[:,[0,-1]] = False
                    mask = mask & np.roll(mask,1,0) & np.roll(mask,-1,0) & np.roll(mask,1,1) & np.roll(mask,-1,1)
                return mask
            def dilate(mask,rounds):
                for _ in range(rounds):
                    mask = mask | np.roll(mask,1,0) | np.roll(mask,-1,0) | np.roll(mask,1,1) | np.roll(mask,-1,1)
                return mask
            slats = renderer.add_mesh(slats_mesh())
            lattice = renderer.add_texture(lattice_texture())
            depth = renderer.add_scene(depth_scene(mesh["id"],screen["id"],slats["id"],lattice["id"]))
            base = {"scene":depth["id"],"width":128,"height":96,"samples":2}
            pinhole = renderer.render(base)
            focused = {key:renderer.render({**base,"focus":{"distance":dist,"f_stop":2.8,"samples":16}})
                       for key,dist in (("near",5.8),("subject",9.0),("far",14.0))}
            again = renderer.render({**base,"focus":{"distance":9.0,"f_stop":2.8,"samples":16}})
            single = renderer.render({**base,"focus":{"distance":9.0,"f_stop":2.8,"samples":1}})
            crop = renderer.render({**base,"region":[16,12,64,48],"focus":{"distance":9.0,"f_stop":2.8,"samples":16}})
            frames = {key:unpack_channels(renderer.artifacts.read(item,"channels.npz"))
                      for key,item in (("pinhole",pinhole),("again",again),("single",single),("crop",crop),*focused.items())}
            # 可复现：无景深走原通道；单光圈样本精确等于针孔；同参数重渲染逐位一致。
            for key in ("color","depth","object_id","normal","world","coverage","sample_offset"):
                require(np.array_equal(frames["pinhole"][key],frames["single"][key]),"focus_repro","single lens sample must match the pinhole frame")
                require(np.array_equal(frames["subject"][key],frames["again"][key]),"focus_repro","focused render must be deterministic")
            # 景深只平均颜色：几何通道与针孔完全一致，拾取不会把散开的颜色当成新几何面。
            for key in ("depth","object_id","normal","world","coverage","sample_offset"):
                require(np.array_equal(frames["pinhole"][key],frames["subject"][key]),"focus_geometry","depth of field must not alter geometry channels")
            require(not np.array_equal(frames["pinhole"]["color"],frames["subject"]["color"]),"focus_blur","focus must change the color channel")
            require(np.array_equal(frames["subject"]["color"][12:60,16:80],frames["crop"]["color"]),"focus_region","focused region must match the full-frame crop")
            ids = frames["pinhole"]["object_id"]
            key_to_id = {key:int(ident) for ident,key in pinhole["details"]["metadata"]["node_map"].items()}
            slat_id,subject_id = key_to_id["near-slats"],key_to_id["subject"]
            lattice_id,block_id = key_to_id["far-lattice"],key_to_id["far-block"]
            # 清晰平面上的主体保持锐利。
            keep = erode(ids==subject_id,2) & ~dilate(ids==slat_id,10)
            require(keep.any(),"focus_scene","subject interior must be visible")
            subject_delta = float(np.abs(frames["subject"]["color"]-frames["pinhole"]["color"])[keep].max())
            require(subject_delta<0.02,"focus_sharp","subject on the focal plane must stay sharp")
            # 远景镂空纹理离焦后局部对比下降。
            lattice_region = erode(dilate(ids==lattice_id,3) & ~dilate(ids==slat_id,12) & ~dilate(ids==subject_id,6) & ~dilate(ids==block_id,6),2)
            require(int(lattice_region.sum())>50,"focus_scene","lattice must keep a visible patch")
            pin_std = float(frames["pinhole"]["color"][lattice_region].std())
            dof_std = float(frames["subject"]["color"][lattice_region].std())
            require(dof_std<0.8*pin_std,"focus_blur","defocused lattice must lose local contrast")
            # 细条压在纯色背景上的行：前景离焦后边缘过渡带变宽，且过渡色是前景与背景的凸组合（合理显露）。
            others = np.isin(ids,[subject_id,lattice_id,block_id])
            rows = [y for y in range(ids.shape[0]) if (ids[y]==slat_id).any() and not others[y].any()]
            require(rows,"focus_scene","slats must rise above the lattice onto plain background")
            row = rows[len(rows)//2]
            background = srgb_to_linear([0.04,0.05,0.08])
            slat_color = srgb_to_linear([0.82,0.84,0.9])
            segment = slat_color-background
            def blend_band(frame):
                px = frame["color"][row]
                t = ((px-background)@segment)/(segment@segment)
                distance = np.linalg.norm(px-(background+t[:,None]*segment),axis=1)
                return (t>0.02)&(t<0.98),distance
            pin_blended,_ = blend_band(frames["pinhole"])
            far_blended,far_distance = blend_band(frames["far"])
            require(int(far_blended.sum())>int(pin_blended.sum()),"focus_reveal","defocused slats must widen the reveal band")
            require(float(far_distance[far_blended].max())<0.03,"focus_reveal","blended pixels must stay on the background-slat segment")
            # 拾取使用针孔几何：过渡带像素的颜色是混合色，但拾取仍返回真实表面。
            column = int(np.where(far_blended)[0][0])
            hit = renderer.inspect(focused["far"]["id"],column,row)
            plain = renderer.inspect(pinhole["id"],column,row)
            require(hit["object_id"]==int(ids[row,column])==plain["object_id"] and hit["world"]==plain["world"],
                    "focus_pick","pick must reflect pinhole geometry, not blended color")
            require(hit["color_source"]=="depth-of-field lens average","focus_pick","color source must be disclosed")
            meta = focused["subject"]["details"]["metadata"]["focus"]
            stats = focused["subject"]["details"]["metadata"]["stats"]
            require(stats["lens_passes"]==16,"focus_metadata","stats must report lens passes")
            require(meta["sharp_near"]>5.8<meta["distance"]<meta["sharp_far"]<14.0,"focus_range","sharp range must bracket the subject only")
            preflight = renderer.preflight({**base,"focus":{"distance":9.0,"f_stop":2.8,"samples":16}})
            require(preflight["lens_passes"]==16 and preflight["admissible"],"focus_preflight","preflight must report lens passes")
            try:
                renderer.preflight({**base,"focus":{"distance":9.0,"f_stop":0.2,"samples":16}})
            except ValidationError:
                schema_rejected = "f_stop below 0.5 rejected"
            else:
                raise RuntimeError("walkthrough expected a rejected focus parameter")
            same = renderer.compare({"left":focused["subject"]["id"],"right":again["id"]})
            require(same["details"]["metrics"]["changed_pixels"]==0,"focus_repro","identical focus settings must compare equal")
            if output:
                for name,item in (("pinhole",pinhole),*focused.items()):
                    Path(output,f"focus-{name}.png").write_bytes(renderer.artifacts.read(item,"color.png"))
                Path(output,"focus-metadata.json").write_text(json.dumps(
                    {name:item["details"]["metadata"]["focus"] for name,item in focused.items()},ensure_ascii=False,indent=2,allow_nan=False))
            results["focus-render"] = {"state":focused["subject"]["state"],"lens_passes":stats["lens_passes"],
                "sharp_range":[meta["sharp_near"],meta["sharp_far"]],"single_sample_matches_pinhole":True,
                "geometry_channels_pinhole":True,"region_matches":True,"subject_sharp_delta":subject_delta,
                "reveal_band_pixels":{"pinhole":int(pin_blended.sum()),"far_focus":int(far_blended.sum())},
                "lattice_contrast_ratio":dof_std/pin_std,
                "pick":{key:hit[key] for key in ("pixel","object_id","node","color_source")},
                "same_focus_changed_pixels":same["details"]["metrics"]["changed_pixels"],
                "schema_rejection":schema_rejected,
                **expect_rejection(lambda:renderer.compare({"left":focused["subject"]["id"],"right":pinhole["id"]}),"comparison_camera")}
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
