"""Bounded production walkthroughs. No test framework or test suite is installed."""
import argparse
import json
import tempfile
from copy import deepcopy
from pathlib import Path
from .service import RendererService
from .samples import wedge_mesh,screen_mesh,tube_mesh,checker_texture,compose_scene
from .errors import RenderError,require
from .channels import unpack_channels
from .color import srgb_to_linear

FLOWS = ["mesh-ingest","texture-ingest","scene-revise","scene-preflight","still-render","pixel-inspect","image-compare","view-sheet","render-bundle","resource-retire","section-render"]


def expect_rejection(operation,code):
    try:
        operation()
    except RenderError as exc:
        require(exc.code==code,"walkthrough_result",f"expected {code}, received {exc.code}")
        return {"rejected":exc.code}
    raise RuntimeError("walkthrough expected a rejected operation")


def section_walkthrough(renderer,wedge,screen,output):
    # Closed meshes cut by a saved world-plane section: caps are separately
    # colored and crack-free; open meshes report uncertain boundaries instead.
    import numpy as np
    outcome = {}

    # 1. Oblique cut through a closed wedge; cap colored independently.
    oblique = renderer.add_section({"label":"斜切剖面","plane":{"normal":[0.35,1.0,0.45],"offset":0.05},
                                    "cap":{"color":[0.9,0.2,0.25],"unlit":True}})
    wedge_scene = renderer.add_scene({"label":"楔体斜切","nodes":[{"key":"wedge","mesh":wedge}],
                                      "sections":[oblique["id"]],
                                      "lens":{"eye":[3,2.2,4],"target":[0,0,0],"near":0.1,"far":30},
                                      "light":{"toward":[1,3,4],"energy":0.85,"ambient":0.18}})
    require(wedge_scene["details"]["sections"][0]["status"]=="sealed","walkthrough_result","scene summary must report sealing")
    cut = renderer.render({"scene":wedge_scene["id"],"width":96,"height":96,"samples":2})
    channels = unpack_channels(renderer.artifacts.read(cut,"channels.npz"))
    cap_mask = channels["section_id"]==1
    require(int(cap_mask.sum())>0,"walkthrough_result","oblique cut must produce cap pixels")
    require(int(((channels["section_id"]==0)&(channels["object_id"]==1)).sum())>0,
            "walkthrough_result","kept surface must remain visible beside the cap")
    solid = cap_mask&(channels["coverage"]==1)
    # Supersampled pixels at the cap boundary blend cap and mesh colors;
    # interior pixels (all neighbors solid) carry the exact cap color.
    interior = np.zeros_like(solid)
    interior[1:-1,1:-1] = (solid[1:-1,1:-1]&solid[:-2,1:-1]&solid[2:,1:-1]
                           &solid[1:-1,:-2]&solid[1:-1,2:])
    require(int(interior.sum())>0,"walkthrough_result","cap must cover whole pixels")
    expected = srgb_to_linear([0.9,0.2,0.25])
    require(bool(np.allclose(channels["color"][interior],expected,atol=1e-12)),
            "walkthrough_result","cap pixels must carry the section cap color")
    report = cut["details"]["metadata"]["sections"][0]
    require(report["status"]=="sealed" and report["id"]==oblique["id"],"walkthrough_result","sealed section report expected")
    cy,cx = [int(v) for v in np.argwhere(cap_mask)[0]]
    picked = renderer.inspect(cut["id"],cx,cy)
    cap_normal = np.array([0.35,1.0,0.45]); cap_normal /= np.linalg.norm(cap_normal)
    require(picked["section"]=={"index":1,"id":oblique["id"],"label":"斜切剖面","surface":"section_cap"},
            "walkthrough_result","pixel selection must attribute the cap to its section")
    require(picked["node"]=="wedge" and picked["surface"]=="section_cap","walkthrough_result","cap belongs to the cut node")
    require(abs(float(np.dot(cap_normal,picked["world"]))-0.05)<1e-9,"walkthrough_result","picked cap point lies on the cut plane")
    require(np.allclose(picked["normal"],cap_normal,atol=1e-9),"walkthrough_result","cap normal faces the removed half")
    background = renderer.inspect(cut["id"],0,0)
    require(background["section"] is None and background["surface"] is None,"walkthrough_result","background has no section")
    outcome["oblique"] = {"status":report["status"],"cap_pixels":int(cap_mask.sum()),
                          "cap_color_exact":True,"inspection":picked}
    if output:
        Path(output,"section-color.png").write_bytes(renderer.artifacts.read(cut,"color.png"))
        Path(output,"section-id.png").write_bytes(renderer.artifacts.read(cut,"section-id.png"))

    # 2. Plane exactly through a mesh vertex; region render equals full-frame crop.
    vertex_normal = np.array([0.3,1.0,0.2]); vertex_normal /= np.linalg.norm(vertex_normal)
    vertex_offset = float(vertex_normal @ np.array([0.5,0.9,-0.5]))
    through_vertex = renderer.add_section({"label":"过顶点剖面","plane":{"normal":vertex_normal.tolist(),"offset":vertex_offset},
                                           "cap":{"color":[0.2,0.8,0.4],"unlit":True}})
    vertex_scene = renderer.add_scene({"label":"楔体穿顶点","nodes":[{"key":"wedge","mesh":wedge}],
                                       "sections":[through_vertex["id"]],
                                       "lens":{"eye":[3,2.2,4],"target":[0,0,0],"near":0.1,"far":30}})
    full = renderer.render({"scene":vertex_scene["id"],"width":96,"height":96,"samples":2})
    crop = renderer.render({"scene":vertex_scene["id"],"width":96,"height":96,"samples":2,"region":[16,12,48,40]})
    vertex_report = full["details"]["metadata"]["sections"][0]
    require(vertex_report["status"]=="sealed","walkthrough_result","vertex-piercing cut must still seal")
    loops = vertex_report["objects"][0]["cap_loops"]
    require(len(loops)==1 and loops[0]["vertices"]==4 and loops[0]["triangles"]==2,
            "walkthrough_result","vertex-piercing cap must be a single welded quad loop")
    require(np.array_equal(unpack_channels(renderer.artifacts.read(full,"channels.npz"))["color"][12:52,16:64],
                           unpack_channels(renderer.artifacts.read(crop,"channels.npz"))["color"]),
            "walkthrough_result","section region render differs from full frame")
    outcome["vertex_plane"] = {"status":vertex_report["status"],"cap_loops":loops,"region_matches":True}

    # 3. Holed section: square tube cut at mid height gives an annulus cap.
    tube = renderer.add_mesh(tube_mesh())
    mid = renderer.add_section({"label":"方管中截面","plane":{"normal":[0,1,0],"offset":0},
                                "cap":{"color":[0.15,0.55,0.9],"unlit":True}})
    top_view = {"projection":"orthographic","eye":[0,4,0],"target":[0,0,0],"up":[0,0,1],
                "span":3,"near":0.1,"far":20}
    tube_scene = renderer.add_scene({"label":"方管剖切","nodes":[{"key":"tube","mesh":tube["id"]}],
                                     "sections":[mid["id"]],"lens":top_view})
    closed_scene = renderer.add_scene({"label":"方管封闭对照","nodes":[{"key":"tube","mesh":tube["id"]}],"lens":top_view})
    sectioned = renderer.render({"scene":tube_scene["id"],"width":96,"height":96,"samples":1})
    closed = renderer.render({"scene":closed_scene["id"],"width":96,"height":96,"samples":1})
    tube_report = sectioned["details"]["metadata"]["sections"][0]
    tube_loops = tube_report["objects"][0]["cap_loops"]
    require(tube_report["status"]=="sealed" and len(tube_loops)==1,"walkthrough_result","tube cut must seal into one outer loop")
    require(tube_loops[0]["vertices"]==8 and tube_loops[0]["holes"]==1 and tube_loops[0]["triangles"]>0,
            "walkthrough_result","annulus cap must keep its inner hole")
    preflight = renderer.preflight({"scene":tube_scene["id"],"width":96,"height":96})
    require(preflight["sections"][0]["status"]=="sealed" and preflight["sections"][0]["cap_triangles"]==tube_loops[0]["triangles"],
            "walkthrough_result","preflight must preview the sealed section")
    sc = unpack_channels(renderer.artifacts.read(sectioned,"channels.npz"))
    cc = unpack_channels(renderer.artifacts.read(closed,"channels.npz"))
    ring = sc["section_id"]==1
    require(int(ring.sum())>0,"walkthrough_result","annulus cap pixels expected")
    ring_world = sc["world"][ring]
    spread = np.maximum(np.abs(ring_world[:,0]),np.abs(ring_world[:,2]))
    require(bool(np.all(np.abs(ring_world[:,1])<1e-12)),"walkthrough_result","cap pixels lie on the cut plane")
    require(bool(np.all(spread>=0.5-1e-9)) and bool(np.all(spread<=1+1e-9)),
            "walkthrough_result","cap pixels stay within the annulus; the hole is not capped")
    ring_color = srgb_to_linear([0.15,0.55,0.9])
    require(bool(np.allclose(sc["color"][ring],ring_color,atol=1e-12)),
            "walkthrough_result","annulus cap carries its own color")
    # The orthographic top view maps the inner square to pixels 32..63;
    # an uncapped hole shows the background straight through the pipe.
    require(int((sc["object_id"][32:64,32:64]==0).sum())==32*32,
            "walkthrough_result","the section hole must stay open")
    require(int((sc["object_id"]==0).sum())==int((cc["object_id"]==0).sum()),
            "walkthrough_result","cap and kept geometry must meet without background cracks")
    tube_pick = renderer.inspect(sectioned["id"],*[int(v) for v in np.argwhere(ring)[0]][::-1])
    require(tube_pick["section"]["id"]==mid["id"] and tube_pick["node"]=="tube",
            "walkthrough_result","annulus pixel selection attributes the section")
    outcome["holed"] = {"status":tube_report["status"],"cap_loops":tube_loops,
                        "ring_pixels":int(ring.sum()),"background_matches_closed":True}
    if output:
        Path(output,"section-tube-color.png").write_bytes(renderer.artifacts.read(sectioned,"color.png"))

    # 4. Open mesh cannot seal: the uncertain boundary is reported, not capped.
    panel = renderer.add_section({"label":"开放色片剖面","plane":{"normal":[0,1,0],"offset":0},
                                  "cap":{"color":[0.95,0.6,0.1],"unlit":True}})
    panel_scene = renderer.add_scene({"label":"开放色片剖切","nodes":[{"key":"panel","mesh":screen}],
                                      "sections":[panel["id"]],
                                      "lens":{"eye":[0,0,4],"target":[0,0,0],"near":0.1,"far":20}})
    panel_render = renderer.render({"scene":panel_scene["id"],"width":64,"height":64,"samples":1})
    panel_report = panel_render["details"]["metadata"]["sections"][0]
    panel_object = panel_report["objects"][0]
    require(panel_report["status"]=="open" and len(panel_object["open_chains"])==1,
            "walkthrough_result","open mesh must report an uncertain boundary")
    chain = panel_object["open_chains"][0]
    endpoints = sorted([round(c,9) for c in point] for point in chain["endpoints"])
    require(endpoints==[[-1.0,0.0,0.0],[1.0,0.0,0.0]],"walkthrough_result","open chain endpoints locate the unsealed boundary")
    require(panel_object["cap_triangles"]==0,"walkthrough_result","uncertain sections are not capped")
    pc = unpack_channels(renderer.artifacts.read(panel_render,"channels.npz"))
    require(int((pc["section_id"]>0).sum())==0,"walkthrough_result","no cap pixels for an open mesh")
    kept = renderer.inspect(panel_render["id"],*[int(v) for v in np.argwhere(pc["object_id"]>0)[0]][::-1])
    require(kept["surface"]=="mesh" and kept["section"] is None,"walkthrough_result","kept open-mesh surface stays ordinary")
    outcome["open_mesh"] = {"status":panel_report["status"],"open_chain":chain,"cap_triangles":0}

    # 5. Domain rejections around saved sections.
    rejections = {}
    rejections.update(expect_rejection(lambda:renderer.add_section({"label":"零法线","plane":{"normal":[0,0,0]}}),"zero_vector"))
    missing = {"label":"缺剖面","nodes":[{"key":"wedge","mesh":wedge}],"sections":["section_missing"]}
    rejections["missing"] = expect_rejection(lambda:renderer.add_scene(missing),"not_found")["rejected"]
    duplicate = {"label":"重复剖面","nodes":[{"key":"wedge","mesh":wedge}],"sections":[oblique["id"],oblique["id"]]}
    rejections["duplicate"] = expect_rejection(lambda:renderer.add_scene(duplicate),"section_duplicate")["rejected"]
    renderer.resource_state(oblique["id"],"retired")
    retired = {"label":"停用剖面","nodes":[{"key":"wedge","mesh":wedge}],"sections":[oblique["id"]]}
    rejections["retired"] = expect_rejection(lambda:renderer.add_scene(retired),"resource_retired")["rejected"]
    renderer.resource_state(oblique["id"],"active")
    again = renderer.render({"scene":wedge_scene["id"],"width":32,"height":32,"samples":1})
    require(again["state"]=="complete","walkthrough_result","old sectioned scene must keep rendering")
    outcome["rejections"] = rejections
    outcome["old_scene_render_after_retire_cycle"] = again["state"]
    return outcome


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
        if flow in {"all","section-render"}:
            results["section-render"] = section_walkthrough(renderer,mesh["id"],screen["id"],output)
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
