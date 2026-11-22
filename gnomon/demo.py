"""Bounded production walkthroughs. No test framework or test suite is installed."""
import argparse
import json
import tempfile
from copy import deepcopy
from pathlib import Path
import numpy as np
from .service import RendererService
from .samples import (wedge_mesh,screen_mesh,checker_texture,compose_scene,
                      hub_mesh,cap_mesh,bolt_mesh,assembly_scene)
from .errors import RenderError,require
from .channels import unpack_channels
from .explosion import plan_explosion

FLOWS = ["mesh-ingest","texture-ingest","scene-revise","scene-preflight","still-render","pixel-inspect","image-compare","view-sheet","render-bundle","resource-retire","assembly-explode"]


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
        if flow in {"all","assembly-explode"}:
            hub = renderer.add_mesh(hub_mesh()); cap = renderer.add_mesh(cap_mesh()); bolt = renderer.add_mesh(bolt_mesh())
            assembly = renderer.add_scene(assembly_scene(hub["id"],cap["id"],bolt["id"]))
            request_explosion = {"scene":assembly["id"],"groups":["carrier","bolt_a","bolt_b"],
                                 "directions":{"carrier":[1,0,0]},"gap_factor":0.4,
                                 "size":96,"samples":1,"angles":[0,60,120]}
            explosion = renderer.explode(request_explosion)
            plan = explosion["details"]["plan"]
            by_key = {entry["key"]:entry for entry in plan["nodes"]}
            # Nested cap and pin move with the carrier: exactly one world shift each.
            nested_ok = np.allclose(by_key["cap"]["world_shift"],by_key["carrier"]["world_shift"]) and \
                        np.allclose(by_key["pin"]["world_shift"],by_key["carrier"]["world_shift"]) and \
                        not by_key["hub"]["moved"]
            # The two nodes sharing the bolt mesh get distinct single shifts.
            shared_ok = by_key["bolt_a"]["moved"] and by_key["bolt_b"]["moved"] and \
                        not np.allclose(by_key["bolt_a"]["world_shift"],by_key["bolt_b"]["world_shift"])
            # Connector lines anchor at the group root's original join position.
            connector = next(item for item in plan["connectors"] if item["group"]=="carrier")
            anchor_ok = np.allclose(connector["join_before"],by_key["carrier"]["origin_before"],atol=1e-10)
            # The exploded scene is an independent payload; the source digest stays unchanged.
            intact_source = renderer.catalog.get(assembly["id"])["digest"]==assembly["digest"]
            # The exploded revision itself is renderable through the normal pipeline,
            # framed with the same orthographic proof lens used for the proof sheet.
            proof = next(item for item in explosion["details"]["index"] if item["stage"]=="after")
            framed_payload = {**plan["scene"],"lens":proof["lens"]}
            framed_scene = renderer.add_scene(framed_payload)
            exploded_render = renderer.render({"scene":framed_scene["id"],"width":192,"height":192,"samples":2})
            channels = unpack_channels(renderer.artifacts.read(exploded_render,"channels.npz"))
            rendered_parts = {value for value in np.unique(channels["object_id"]).tolist() if value}
            mesh_node_count = sum(1 for node in plan["scene"]["nodes"] if node["mesh"])
            coverage_ok = len(rendered_parts) == mesh_node_count
            # Crowding versus fidelity: a larger gap factor strictly increases separation.
            tighter = renderer.explode({**request_explosion,"gap_factor":0.05})
            span = lambda item: item["details"]["bounds_after"]["max"][0]-item["details"]["bounds_after"]["min"][0]
            tradeoff_ok = span(explosion) > span(tighter)
            bad_group = expect_rejection(lambda:renderer.explode({**request_explosion,"groups":["missing"],"directions":{}}),"explode_group")
            bad_direction = expect_rejection(
                lambda:renderer.explode({**request_explosion,"directions":{"hub":[0,0,1]}}),"explode_direction")
            empty_group_scene = {"label":"空部件组","nodes":[
                {"key":"root"},{"key":"pivot","parent":"root"},
                {"key":"body","parent":"root","mesh":hub["id"],"surface":{"color":[0.5,0.5,0.5]}}]}
            empty_source = renderer.add_scene(empty_group_scene)
            bad_empty = expect_rejection(
                lambda:renderer.explode({"scene":empty_source["id"],"groups":["pivot"],"directions":{}}),"explode_group")
            results["assembly-explode"] = {"state":explosion["state"],"groups":explosion["details"]["group_count"],
                                           "moved_nodes":explosion["details"]["moved_nodes"],
                                           "nested_single_shift":bool(nested_ok),"shared_mesh_distinct":bool(shared_ok),
                                           "connector_anchors_join":bool(anchor_ok),"source_scene_intact":bool(intact_source),
                                           "exploded_render_parts":len(rendered_parts),"coverage_ok":bool(coverage_ok),
                                           "gap_tradeoff":bool(tradeoff_ok),
                                           "rejections":[bad_group,bad_direction,bad_empty]}
            if output:
                Path(output,"proof-sheet.png").write_bytes(renderer.artifacts.read(explosion,"proof-sheet.png"))
                Path(output,"plan.json").write_bytes(renderer.artifacts.read(explosion,"plan.json"))
                Path(output,"exploded-scene.json").write_bytes(renderer.artifacts.read(explosion,"exploded-scene.json"))
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
