"""Bounded production walkthroughs. No test framework or test suite is installed."""
import argparse
import json
import tempfile
from copy import deepcopy
from pathlib import Path
from .service import RendererService
from .samples import wedge_mesh,screen_mesh,checker_texture,compose_scene,stereo_scene,stereo_rig
from .errors import RenderError,require
from .channels import unpack_channels

FLOWS = ["mesh-ingest","texture-ingest","scene-revise","scene-preflight","still-render","pixel-inspect","image-compare","stereo-pair","view-sheet","render-bundle","resource-retire"]


def expect_rejection(operation,code):
    try:
        operation()
    except RenderError as exc:
        require(exc.code==code,"walkthrough_result",f"expected {code}, received {exc.code}")
        return {"rejected":exc.code}
    raise RuntimeError("walkthrough expected a rejected operation")


def _status_pixels(renderer, record, label, code):
    """Yield source pixels with a given correspondence status, from the NPZ map."""
    import numpy as np
    maps = unpack_channels(renderer.artifacts.read(record, "correspondence.npz"))
    positions = np.argwhere(maps[f"{label}_status"] == code)
    for y, x in positions:
        yield int(x), int(y)


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
        if flow in {"all","stereo-pair"}:
            rig = stereo_rig()
            layered = renderer.add_scene(stereo_scene(mesh["id"], screen["id"]))
            pair = renderer.render({"scene": layered["id"], "width": 160, "height": 120,
                                    "samples": 2, "stereo": rig})
            metadata = pair["details"]["metadata"]
            summary = metadata["correspondence"]["directions"]["left_to_right"]
            require(summary["trusted_pixels"] > 0, "stereo_demo", "stereo pair must have trusted matches")
            require(metadata["rig_checks"]["translation_only"], "stereo_demo", "rig must be translation only")
            require(metadata["rig_checks"]["zero_parallax_within_quarter_pixel"], "stereo_demo",
                    "convergence plane must have zero parallax")
            require(summary["vertical_parallax_pixels"]["within_tolerance"], "stereo_demo",
                    "parallel rig must not introduce vertical parallax")
            probe = pair["details"]["metadata"]["correspondence"]["probe_checks"]["left_to_right"]
            require(probe["all_within_tolerance"], "stereo_demo", "analytic parallax must match measurement")
            require(summary["status_counts"]["occluded_in_other_eye"] > 0, "stereo_demo",
                    "layered demo must show a genuine occlusion")
            occluded = next(iter(_status_pixels(renderer, pair, "left_to_right", 2)), None)
            require(occluded is not None, "stereo_demo", "layered demo must have occluded left pixels")
            query = renderer.stereo_correspondence(pair["id"], "left", occluded[0], occluded[1])
            require(query["status"] == "occluded_in_other_eye", "stereo_demo",
                    "query at an occluded pixel must report the occlusion")
            ortho_input = stereo_scene(mesh["id"], screen["id"])
            ortho_input["lens"] = {**ortho_input["lens"], "projection": "orthographic"}
            orthographic = renderer.add_scene(ortho_input)
            results["stereo-pair"] = {
                "artifacts": sorted(pair["details"]["artifacts"]),
                "units": metadata["units"],
                "rig_checks": metadata["rig_checks"],
                "left_to_right_trusted_pixels": summary["trusted_pixels"],
                "occlusion_pairs": summary["occlusion_pairs"],
                "probe_checks": probe,
                "query_status": query["status"],
                "query_occluded_pixel": list(occluded),
                "zero_parallax_horizontal_pixels": metadata["rig_checks"]["zero_parallax_horizontal_pixels"],
                **expect_rejection(lambda: renderer.render({"scene": orthographic["id"], "width": 64,
                                                            "height": 64, "stereo": rig}),
                                   "stereo_projection"),
            }
            if output:
                for name in ("pair.png", "parallax.png", "occlusion-mask.png", "crosscheck.json"):
                    Path(output, name).write_bytes(renderer.artifacts.read(pair, name))
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
