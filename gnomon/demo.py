"""Bounded production walkthroughs. No test framework or test suite is installed."""
import argparse
import json
import tempfile
from copy import deepcopy
from pathlib import Path
from .service import RendererService
from .samples import (wedge_mesh,screen_mesh,checker_texture,compose_scene,seam_ramp_mesh,
                      collapsed_uv_mesh,ripple_normal_texture,panel_texture,flat_normal_texture,normal_material_scene)
from .errors import RenderError,require
from .channels import unpack_channels

FLOWS = ["mesh-ingest","texture-ingest","scene-revise","scene-preflight","still-render","normal-material","pixel-inspect","image-compare","view-sheet","render-bundle","resource-retire"]


def expect_rejection(operation,code):
    try:
        operation()
    except RenderError as exc:
        require(exc.code==code,"walkthrough_result",f"expected {code}, received {exc.code}")
        return {"rejected":exc.code}
    raise RuntimeError("walkthrough expected a rejected operation")


def normal_material_walkthrough(renderer,output):
    """Normal mapping without new geometry, observed through both channels.

    Renders the seamed ramp with a tangent-space normal map and reports numbers
    a caller can re-check:

    * a flat normal map (0,0,1) leaves color unchanged, so the map alone adds
      the relief;
    * the color channel equals the unlit diffuse times the Lambert factor of
      the *world-normal channel*, tying the two channels to one surface;
    * a reflected instance keeps ridges on the same side (a dent stays a dent);
    * a non-uniformly scaled instance keeps the world frame orthonormal, so the
      light direction remains believable;
    * normal_strength=0 reproduces the unmapped shading.

    It also records how a missing UV, a normal map placed in the color slot, or
    a zero-UV-area face are rejected instead of silently rendering.
    """
    import numpy as np
    from .scene import compile_scene
    from .shading import perturbed_normals
    from .texture import normal_array
    mesh = renderer.add_mesh(seam_ramp_mesh())
    panel = renderer.add_texture(panel_texture())
    normal_map = renderer.add_texture(ripple_normal_texture())
    flat_map = renderer.add_texture(flat_normal_texture())
    lens = {"eye":[0,3.2,6],"target":[0,0.2,0],"near":0.1,"far":30}
    light = {"toward":[-1,2,1.5],"energy":0.9,"ambient":0.15}

    def render(surface,pose=None,size=128):
        node = {"key":"panel","mesh":mesh["id"],"surface":surface}
        if pose:
            node["pose"] = pose
        scene = renderer.add_scene({"label":"法线材质","nodes":[node],"lens":lens,"light":light})
        # One sample per pixel keeps color and normal describing the same
        # surface: color averages subpixels, while the normal channel reports
        # only the nearest covered subpixel.
        rendered = renderer.render({"scene":scene["id"],"width":size,"height":size,"samples":1})
        channels = unpack_channels(renderer.artifacts.read(rendered,"channels.npz"))
        return rendered,channels

    surface = {"texture":panel["id"],"color":[1,1,1]}
    plain,_ = render({**surface,"unlit":True})
    flat,_ = render({**surface,"unlit":True,"normal_texture":flat_map["id"]})
    lit,_ = render(surface)
    mapped,mapped_channels = render({**surface,"normal_texture":normal_map["id"]})
    zero,_ = render({**surface,"normal_texture":normal_map["id"],"normal_strength":0})
    reflected,_ = render({**surface,"normal_texture":normal_map["id"]},{"scale":[-1,1,1]})
    stretched,_ = render({**surface,"normal_texture":normal_map["id"]},{"scale":[0.5,1.6,1]})
    flat_plain = unpack_channels(renderer.artifacts.read(plain,"channels.npz"))
    flat_map_channels = unpack_channels(renderer.artifacts.read(flat,"channels.npz"))
    zero_channels = unpack_channels(renderer.artifacts.read(zero,"channels.npz"))
    lit_channels = unpack_channels(renderer.artifacts.read(lit,"channels.npz"))
    covered = mapped_channels["object_id"]>0
    cover = flat_plain["object_id"]>0
    flat_gap = float(np.abs(flat_map_channels["color"][cover]-flat_plain["color"][cover]).max())
    require(flat_gap <= 0.02,"normal_flat","a flat normal map must not change shading")
    zero_gap = float(np.abs(zero_channels["color"][covered]-lit_channels["color"][covered]).max())
    require(zero_gap <= 0.02,"normal_strength","normal_strength=0 must reproduce the unmapped surface")

    toward = np.asarray(light["toward"],dtype=float)
    toward /= np.linalg.norm(toward)
    normals = mapped_channels["normal"][covered]
    cosine = np.maximum(0,normals @ toward)
    factor = light["ambient"]+light["energy"]*cosine
    ratio = mapped_channels["color"][covered]/np.maximum(factor[:,None],1e-9)
    # Where the cosine is not clamped to zero, dividing the color channel by
    # the Lambert factor must return the unlit diffuse exactly. That is what
    # makes the color channel and the world-normal channel describe one face.
    roomy = cosine > 0.05
    consistency = float(np.abs(ratio[roomy]-flat_plain["color"][cover][roomy]).max()) if roomy.any() else 0.0
    require(consistency <= 0.02,"normal_lambert","lit color must equal unlit color times the Lambert factor")

    def relief(channel):
        mask = channel["object_id"]>0
        return float(channel["normal"][mask,0].min()), float(channel["normal"][mask,0].max())
    plain_relief = relief(flat_plain)
    mapped_relief = relief(mapped_channels)
    require(max(map(abs,mapped_relief)) > 1.3*max(map(abs,plain_relief)),
            "normal_relief","normal map must tilt the shading normal beyond the geometric normal")

    # Compare world normals of a reflected instance to the reflection of the
    # upright instance's world normals, vertex by vertex. This is the invariant
    # that keeps a dent from rendering as a bump.
    def compiled(pose):
        node = {"key":"panel","mesh":mesh["id"],
                "surface":{**surface,"normal_texture":normal_map["id"]}}
        if pose:
            node["pose"] = pose
        reference = renderer.add_scene({"label":"法线材质","nodes":[node],"lens":lens,"light":light})
        return compile_scene(renderer.catalog.get(reference["id"],"scene")["payload"],
                             renderer.catalog.get)["objects"][0]
    texels = normal_array(normal_map["payload"])
    upright_obj = compiled(None)
    mirror_obj = compiled({"scale":[-1,1,1]})
    mirror_matrix = np.diag([-1.0,1.0,1.0])
    reference_normals = perturbed_normals(upright_obj["normals"],upright_obj["tangent"],upright_obj["bitangent"],
                                          texels,upright_obj["uv"],"repeat",1.0)
    reflected_normals = perturbed_normals(mirror_obj["normals"],mirror_obj["tangent"],mirror_obj["bitangent"],
                                          texels,mirror_obj["uv"],"repeat",1.0)
    mirror_delta = float(np.abs(reflected_normals-reference_normals@mirror_matrix.T).max())
    require(mirror_delta <= 1e-9,"normal_mirror","reflection must flip the frame, not the ridge direction")

    # The seam itself: the two ridge copies share a position but must keep
    # opposing tangents instead of being averaged into one frame.
    ridge = [i for i,p in enumerate(upright_obj["positions"]) if np.allclose(p,[0.0,0.7,-0.8])]
    require(len(ridge) == 2,"normal_seam","seam copies must both appear in the compiled mesh")
    left,right = ridge
    seam_split = float(np.dot(upright_obj["tangent"][left],upright_obj["tangent"][right]))
    require(upright_obj["tangent"][left][0] > 0.5 and upright_obj["tangent"][right][0] < -0.5,
            "normal_seam","seam copies must keep their own opposite tangent frames")

    flat_uv = seam_ramp_mesh()
    missing = renderer.add_mesh({**flat_uv,"uv":None})
    collapsed = renderer.add_mesh(collapsed_uv_mesh())
    result = {"flat_map_max_color_delta":flat_gap,"normal_strength_zero_delta":zero_gap,
              "normal_band":mapped_relief,"unmapped_band":plain_relief,
              "normal_mapped_triangles":mapped["details"]["metadata"]["stats"]["normal_mapped_triangles"],
              "lambert_consistency":consistency,"mirror_normal_delta":mirror_delta,
              "seam_tangent_dot":seam_split,
              "stretched_frame_orthonormal":frame_orthonormal(renderer,stretched["id"]),
              "missing_uv":expect_rejection(lambda: renderer.add_scene({"label":"缺少UV","nodes":[
                  {"key":"panel","mesh":missing["id"],"surface":{"normal_texture":normal_map["id"]}}],"lens":lens}),"texture_uv"),
              "mixed_up_slot":expect_rejection(lambda: renderer.add_scene({"label":"角色错位","nodes":[
                  {"key":"panel","mesh":mesh["id"],"surface":{"texture":normal_map["id"]}}],"lens":lens}),"texture_role"),
              "degenerate_uv_frame":expect_rejection(lambda: renderer.add_scene({"label":"退化UV","nodes":[
                  {"key":"panel","mesh":collapsed["id"],"surface":{"normal_texture":normal_map["id"]}}],"lens":lens}),"normal_tangent")}

    showcase = renderer.add_scene(normal_material_scene(mesh["id"],panel["id"],normal_map["id"]))
    board = renderer.render({"scene":showcase["id"],"width":480,"height":240,"samples":1})
    board_channels = unpack_channels(renderer.artifacts.read(board,"channels.npz"))
    result["showcase_instances"] = int(len(np.unique(board_channels["object_id"]))-1)
    if output:
        for artifact in ("color.png","normal.png","channels.npz","metadata.json"):
            Path(output,"normal-material-"+artifact).write_bytes(renderer.artifacts.read(board,artifact))
        for name,rendered in (("mirror",reflected),("stretch",stretched),("flat",flat)):
            Path(output,f"normal-material-{name}.png").write_bytes(renderer.artifacts.read(rendered,"color.png"))
            Path(output,f"normal-material-{name}-normal.png").write_bytes(renderer.artifacts.read(rendered,"normal.png"))
    result["artifacts"] = sorted(board["details"]["artifacts"])
    return result


def frame_orthonormal(renderer,record):
    """Check that world frames stay orthonormal on the stretched instance."""
    import numpy as np
    from .scene import compile_scene
    scene = renderer.catalog.get(record,"render")["payload"]["snapshot"]["scene"]
    compiled = compile_scene(scene,renderer.catalog.get,allow_retired=True)
    worst = 0.0
    for obj in compiled["objects"]:
        tangent,bitangent,normals = obj["tangent"],obj["bitangent"],obj["normals"]
        if tangent is None:
            continue
        worst = max(worst,float(np.abs(np.sum(tangent*normals,axis=1)).max()),
                    float(np.abs(np.sum(bitangent*normals,axis=1)).max()),
                    float(np.abs(np.sum(tangent*bitangent,axis=1)).max()),
                    float(np.abs(np.linalg.norm(tangent,axis=1)-1).max()),
                    float(np.abs(np.linalg.norm(bitangent,axis=1)-1).max()))
    return worst


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
        if flow in {"all","normal-material"}:
            results["normal-material"] = normal_material_walkthrough(renderer,output)
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
