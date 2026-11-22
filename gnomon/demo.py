"""Bounded production walkthroughs. No test framework or test suite is installed."""
import argparse
import json
import tempfile
from copy import deepcopy
from pathlib import Path
import numpy as np
from .service import RendererService
from .samples import wedge_mesh,screen_mesh,checker_texture,compose_scene,transparency_scene,dense_stack_scene
from .errors import RenderError,require
from .channels import unpack_channels
from .color import srgb_to_linear
from .raster import MAX_TRANSLUCENT_LAYERS

FLOWS = ["mesh-ingest","texture-ingest","scene-revise","scene-preflight","still-render","pixel-inspect","image-compare","view-sheet","render-bundle","resource-retire","translucency-proof"]


def expect_rejection(operation,code):
    try:
        operation()
    except RenderError as exc:
        require(exc.code==code,"walkthrough_result",f"expected {code}, received {exc.code}")
        return {"rejected":exc.code}
    raise RuntimeError("walkthrough expected a rejected operation")


def translucency_proof(renderer,mesh_id,screen_id):
    """Order-independent blending, pick-through rules, layer and budget bounds."""
    proof = {}
    config = {"width":128,"height":128,"samples":1}

    def render(payload):
        source = renderer.add_scene(payload)
        record = renderer.render({**config,"scene":source["id"]})
        return record,unpack_channels(renderer.artifacts.read(record,"channels.npz")),source

    # 1. Node order must not decide the composited color: depth sorting does.
    scene_a = transparency_scene(mesh_id,screen_id,reversed_order=False)
    scene_b = transparency_scene(mesh_id,screen_id,reversed_order=True)
    record_a,frame_a,_ = render(scene_a)
    _,frame_b,_ = render(scene_b)
    require(np.array_equal(frame_a["color"],frame_b["color"]),"translucency_order",
            "composited color depends on node ordering")
    proof["order_independent"] = True

    # 2. Exact front-to-back source-over where red and blue sheets overlap and
    #    no opaque surface is present. The expected color is derived from the
    #    actual ordered layers, not from node order.
    node_map = record_a["details"]["metadata"]["node_map"]
    ids = {name:int(identity) for identity,name in node_map.items()}
    overlap = (np.isin(frame_a["through_ids"],[ids["red_sheet"],ids["blue_sheet"]]).sum(axis=2) >= 2)
    require(overlap.any(),"translucency_proof","overlapping region must be visible")
    background = srgb_to_linear(np.asarray(scene_a.get("background",(0.035,0.045,0.07))))
    sheet_color = {"red_sheet":srgb_to_linear(np.asarray([0.9,0.08,0.08])),
                   "blue_sheet":srgb_to_linear(np.asarray([0.08,0.2,0.95])),
                   "green_sheet":srgb_to_linear(np.asarray([0.1,0.8,0.2]))}
    sheet_alpha = {"red_sheet":0.5,"blue_sheet":0.5,"green_sheet":0.45}
    id_to_name = {identity:name for name,identity in ids.items()}
    # Independent front-to-back reference accumulated from black with the
    # background admitted through the remaining carry exactly once.
    expected_pixels = np.zeros(frame_a["color"].shape)
    carry = np.ones(frame_a["color"].shape[:2])
    for k in range(frame_a["through_ids"].shape[2]):
        slot = frame_a["through_ids"][...,k]
        for identity,name in id_to_name.items():
            if name not in sheet_color:
                continue
            hit = slot == identity
            a = sheet_alpha[name]
            expected_pixels[hit] += carry[hit,None]*a*sheet_color[name]
            carry[hit] *= 1-a
    expected_pixels += carry[...,None]*background
    measured = frame_a["color"][overlap]
    expected = expected_pixels[overlap]
    require(np.allclose(measured,expected,atol=1e-10),"translucency_math",
            "overlap color is not the Porter-Duff source-over result")
    proof["overlap_linear_rgb"] = [float(v) for v in measured[0]]
    proof["overlap_expected_linear_rgb"] = [float(v) for v in expected[0]]
    proof["overlap_layer_count"] = int(frame_a["layer_count"][overlap].max())

    # 3. Picking reaches the solid wedge through the front translucent sheets.
    wedge_pixels = np.argwhere(frame_a["object_id"] == ids["wedge"])
    require(len(wedge_pixels)>0,"translucency_proof","wedge must remain pickable through sheets")
    fronted = np.array([(y,x) for y,x in wedge_pixels
                        if frame_a["layer_count"][y,x] > 0])
    require(len(fronted)>0,"translucency_pick","some wedge pixels must sit behind translucent sheets")
    y,x = fronted[len(fronted)//2]
    inspected = renderer.inspect(record_a["id"],int(x),int(y))
    require(inspected["node"] == "wedge","translucency_pick","pick must reach the opaque wedge")
    require(len(inspected["translucent_layers_in_front"]) >= 1,"translucency_pick",
            "pick response must name the translucent front layers")
    require(inspected["selection"].startswith("nearest opaque surface"),"translucency_pick",
            "selection rule must explain pick-through")
    proof["pick_through"] = {"pixel":inspected["pixel"],"node":inspected["node"],
                             "front_layers":inspected["translucent_layers_in_front"],
                             "rule":inspected["selection"]}

    # 4. An opaque occluder hides the green sheet: nothing behind it is picked
    #    and no green contribution remains in its silhouette.
    blocked = frame_a["object_id"] == ids["blocker"]
    require(blocked.any(),"translucency_proof","occluder must cover some pixels")
    by,bx = np.argwhere(blocked)[len(np.argwhere(blocked))//2]
    blocked_inspection = renderer.inspect(record_a["id"],int(bx),int(by))
    require(blocked_inspection["node"] == "blocker","translucency_occlusion",
            "opaque occluder must win selection")
    green_rgb = srgb_to_linear(np.asarray([0.1,0.8,0.2]))
    require(not np.any(np.isclose(frame_a["color"][blocked],green_rgb,atol=1e-3)),
            "translucency_occlusion","occluded green sheet must not contribute color")
    proof["opaque_occlusion"] = {"picked":blocked_inspection["node"]}

    # 5. Where only translucent sheets exist, the nearest one is selected and
    #    reported as a translucent pick with the full ordered front stack.
    oy,ox = np.argwhere(overlap)[0]
    only = renderer.inspect(record_a["id"],int(ox),int(oy))
    require(only["node"] in {"red_sheet","blue_sheet","green_sheet"}
            and only["translucent_layer_count"] >= 2,
            "translucency_pick","translucent-only region must select nearest sheet")
    require(only["selection"].startswith("nearest translucent"),"translucency_pick",
            "translucent-only selection must state the fallback rule")
    proof["translucent_only_pick"] = {"node":only["node"],
                                      "layers":only["translucent_layers_in_front"]}

    # 6. Dense stacks hit the explainable layer cap and overflow is counted.
    within = dense_stack_scene(screen_id,MAX_TRANSLUCENT_LAYERS)
    beyond = dense_stack_scene(screen_id,MAX_TRANSLUCENT_LAYERS+4)
    record_in,frame_in,_ = render(within)
    record_over,frame_over,_ = render(beyond)
    require(int(frame_in["layer_count"].max()) == MAX_TRANSLUCENT_LAYERS,
            "layer_cap","stack at the cap must fill every layer slot")
    require(record_in["details"]["metadata"]["stats"]["translucent_overflow_samples"] == 0,
            "layer_cap","stack at the cap must not overflow")
    require(record_over["details"]["metadata"]["stats"]["translucent_overflow_samples"] > 0,
            "layer_cap","stack beyond the cap must report overflow")
    require(int(frame_over["layer_count"].max()) == MAX_TRANSLUCENT_LAYERS,
            "layer_cap","overflow must never exceed the fixed slot count")
    proof["layer_cap"] = {"max_layers":MAX_TRANSLUCENT_LAYERS,
                          "overflow_samples":record_over["details"]["metadata"]["stats"]["translucent_overflow_samples"]}

    # 7. Memory budget uses the translucent estimate and rejects oversized frames.
    preflight = renderer.preflight({**config,"scene":renderer.add_scene(scene_a)["id"]})
    require(preflight["translucent"] and preflight["translucent_layer_cap"] == MAX_TRANSLUCENT_LAYERS,
            "preflight_translucency","preflight must expose translucency bounds")
    huge = renderer.preflight({"scene":renderer.add_scene(scene_a)["id"],
                               "width":1024,"height":1024,"samples":3})
    require(not huge["admissible"],"render_budget","oversized translucent frame must be inadmissible")
    opaque_scene = deepcopy(scene_a)
    for node in opaque_scene["nodes"]:
        node["surface"].pop("opacity",None)
    opaque_preflight = renderer.preflight({**config,"scene":renderer.add_scene(opaque_scene)["id"]})
    require(not opaque_preflight["translucent"],"preflight_translucency",
            "opaque scenes must keep the baseline buffer estimate")
    proof["buffer_budget"] = {"translucent_estimated_bytes":preflight["estimated_buffer_bytes"],
                              "opaque_estimated_bytes":opaque_preflight["estimated_buffer_bytes"],
                              "oversized_admissible":huge["admissible"]}

    # 8. Binary cutout semantics survive: alpha_cutoff is still a hard mask and
    #    never becomes partial blending, so no translucent layers are allocated.
    texture = renderer.add_texture(checker_texture())
    cutout_scene = compose_scene(mesh_id,screen_id,texture["id"])
    cutout_record = renderer.render({"scene":renderer.add_scene(cutout_scene)["id"],**config})
    cutout = unpack_channels(renderer.artifacts.read(cutout_record,"channels.npz"))
    require("layer_count" not in cutout,"cutout_semantics",
            "binary cutout must stay in the opaque nearest-depth path")
    require(cutout_record["details"]["metadata"]["stats"]["translucent_overflow_samples"] == 0,
            "cutout_semantics","binary cutout must not allocate translucent layers")
    proof["binary_cutoff_preserved"] = True
    return proof


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
        if flow == "translucency-proof":
            results["translucency-proof"] = translucency_proof(renderer,mesh["id"],screen["id"])
            return results
        if flow in {"all","resource-retire"}:
            renderer.resource_state(mesh["id"],"retired")
            again = renderer.render(request)
            results["resource-retire"] = {"old_revision_render":again["state"],**expect_rejection(lambda:renderer.add_scene(scene_input),"resource_retired")}
            renderer.resource_state(mesh["id"],"active")
            renderer.scene_state(scene["id"],"archived")
            expect_rejection(lambda:renderer.render(request),"scene_archived")
            renderer.scene_state(scene["id"],"active")
        if flow == "all":
            results["translucency-proof"] = translucency_proof(renderer,mesh["id"],screen["id"])
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
