"""Bounded production walkthroughs. No test framework or test suite is installed."""
import argparse
import json
import tempfile
from copy import deepcopy
from pathlib import Path
import numpy as np
from .service import RendererService
from .samples import (wedge_mesh, screen_mesh, checker_texture, compose_scene,
                      room_mesh, panorama_scene)
from .errors import RenderError, require
from .channels import unpack_channels

FLOWS = ["mesh-ingest","texture-ingest","scene-revise","scene-preflight","still-render","pixel-inspect","image-compare","view-sheet","render-bundle","resource-retire","panorama-render"]


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
            require(np.array_equal(full["color"][12:52,16:64],cut["color"]),"crop_semantics","region differs from full frame")
            results["still-render"] = {"state":rendered["state"],"stats":rendered["details"]["metadata"]["stats"],"region_matches":True}
        if flow in {"all","pixel-inspect"}:
            channels = unpack_channels(renderer.artifacts.read(rendered,"channels.npz"))
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
        if flow in {"all","panorama-render"}:
            room = renderer.add_mesh(room_mesh())
            pano_input = panorama_scene(room["id"],mesh["id"],texture["id"])
            pano_scene = renderer.add_scene(pano_input)
            pano_request = {"scene":pano_scene["id"],"width":256,"height":128,"samples":2}
            # Rejection paths: no stretched perspective, no region crop, no over-budget rays.
            expect_rejection(lambda:renderer.render({**pano_request,"height":96}),"panorama_aspect")
            expect_rejection(lambda:renderer.render({**pano_request,"region":[0,0,64,32]}),"panorama_region")
            oversized = renderer.preflight({"scene":pano_scene["id"],"width":1024,"height":512,"samples":3})
            require(oversized["admissible"] is False,"panorama_budget","oversized panorama must be inadmissible")
            expect_rejection(lambda:renderer.render({"scene":pano_scene["id"],"width":1024,"height":512,"samples":3}),"render_budget")
            pre = renderer.preflight(pano_request)
            pano = renderer.render(pano_request)
            pano_channels = unpack_channels(renderer.artifacts.read(pano,"channels.npz"))
            pano_meta = pano["details"]["metadata"]
            require(pano_meta["lens_projection"]=="equirectangular","panorama_metadata","projection must be recorded")
            require(pano_meta["view"] is None,"panorama_metadata","panorama has no planar view matrix")
            require("ray" in pano_channels,"panorama_channels","ray channel must be stored")
            covered = pano_channels["object_id"] > 0
            require(bool(covered.all()),"panorama_coverage","enclosed room must cover every ray")
            depth = pano_channels["depth"]
            require(bool(np.isfinite(depth).all() and depth.min() >= pano_input["lens"]["near"]
                         and depth.max() <= pano_input["lens"]["far"]),
                    "panorama_depth","depth must be bounded ray distance, not NDC")
            # Left/right seam: both edge columns look along the same yaw line.
            seam_ids = pano_channels["object_id"][:,0] == pano_channels["object_id"][:,-1]
            seam_color = np.abs(pano_channels["color"][:,0]-pano_channels["color"][:,-1]).mean()
            require(bool(seam_ids.mean() >= 0.95) and seam_color < 0.02,
                    "panorama_seam","seam edges must show the same surfaces")
            # Poles: first and last rows sample straight up/down, missing no surface.
            require(bool(pano_channels["coverage"][0].mean() > 0.99
                         and pano_channels["coverage"][-1].mean() > 0.99),
                    "panorama_poles","polar rows must stay covered")
            name_to_id = {name:int(identity) for identity,name in pano_meta["node_map"].items()}
            def rows_for(node):
                ys,xs = np.where(pano_channels["object_id"]==name_to_id[node])
                return ys,xs
            cap_rows,_ = rows_for("ceiling-cap")
            require(len(cap_rows) > 0 and float(90-(cap_rows.min()+0.5)/128*180) > 45,
                    "panorama_overhead","overhead surface must appear near the top pole")
            occ_rows,occ_cols = rows_for("near-occluder")
            room_rows,room_cols = rows_for("room")
            require(len(occ_rows) > 0 and len(room_rows) > 0,
                    "panorama_occlusion","foreground occluder must be visible")
            near_dist = depth[occ_rows,occ_cols].max()
            far_dist = np.median(depth[room_rows,room_cols])
            require(near_dist < far_dist,"panorama_occlusion","near occluder must win depth over the room")
            seam_rows,_ = rows_for("seam-column")
            edge_hits = int(any(1 in (pano_channels["object_id"][y,0],pano_channels["object_id"][y,-1]) for y in seam_rows))
            require(edge_hits > 0,"panorama_seam","seam-crossing object must touch both longitude edges")
            # Click direction must map back onto the original scene surface.
            eye = np.asarray(pano_input["lens"]["eye"])
            probes = []
            for y,x in [(32,128),(64,10),(96,250),(2,128),(126,128)]:
                info = renderer.inspect(pano["id"],x,y)
                ray = np.asarray(info["ray_direction"])
                rebuilt = eye + info["distance"]*ray
                require(np.linalg.norm(rebuilt-np.asarray(info["world"])) < 1e-6,
                        "panorama_roundtrip","eye + distance*ray must equal the hit point")
                # The stored ray belongs to the nearest covered subpixel, whose
                # full-frame coordinate the inspection response reports.
                sample_x,sample_y = info["full_sample"]
                col = (np.degrees(np.arctan2(ray[0],-ray[2]))/360+0.5)*256-0.5
                row = (0.5-np.degrees(np.arcsin(np.clip(ray[1],-1,1)))/180)*128-0.5
                require(abs(col-sample_x) <= 0.51 and abs(row-sample_y) <= 0.51,
                        "panorama_roundtrip","ray angles must map back to the nearest subpixel")
                probes.append({"pixel":[x,y],"node":info["node"],"distance":info["distance"],
                               "yaw":info["yaw_degrees"],"pitch":info["pitch_degrees"]})
            # A planar perspective frame is semantically incompatible with the panorama.
            planar = renderer.render({"scene":scene["id"],"width":256,"height":128,"samples":1})
            expect_rejection(lambda:renderer.compare({"left":pano["id"],"right":planar["id"]}),"comparison_camera")
            pano_bundle = renderer.bundle(pano["id"])
            import io, zipfile
            with zipfile.ZipFile(io.BytesIO(renderer.artifacts.read(pano_bundle,"render.zip"))) as archive:
                manifest = json.loads(archive.read("manifest.json"))
            require("ray" in manifest["semantics"] and "equirectangular" in manifest["semantics"]["layout"],
                    "panorama_bundle","bundle must explain panorama channels")
            require("360x180" in manifest["imaging"]["difference_from_perspective_lens"],
                    "panorama_bundle","bundle must describe how this lens differs from existing lenses")
            results["panorama-render"] = {"state":pano["state"],"preflight":pre,
                "stats":pano_meta["stats"],"seam_color_delta":float(seam_color),
                "seam_id_agreement":float(seam_ids.mean()),
                "occluder_max_distance":float(near_dist),"room_median_distance":float(far_dist),
                "probes":probes,"bundle_format":manifest["format"]}
            if output:
                for name in pano["details"]["artifacts"]:
                    Path(output,f"panorama-{name}").write_bytes(renderer.artifacts.read(pano,name))
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
