"""Bounded production walkthroughs. No test framework or test suite is installed."""
import argparse
import hashlib
import json
import tempfile
from copy import deepcopy
from pathlib import Path
from .service import RendererService
from .samples import wedge_mesh,screen_mesh,checker_texture,compose_scene
from .errors import RenderError,require
from .channels import unpack_channels

FLOWS = ["mesh-ingest","texture-ingest","scene-revise","scene-preflight","still-render","pixel-inspect","image-compare","view-sheet","render-bundle","resource-retire","keyframe-sequence"]


def expect_rejection(operation,code):
    try:
        operation()
    except RenderError as exc:
        require(exc.code==code,"walkthrough_result",f"expected {code}, received {exc.code}")
        return {"rejected":exc.code}
    raise RuntimeError("walkthrough expected a rejected operation")


def _sequence_request(interpolation):
    # Root turns through the +/-180 boundary (170 -> -170 deg). On the
    # continuous branch that is a short 20-degree move, never a 340 sweep.
    root_track = {"node":"root","interpolation":interpolation,
                  "keyframes":[{"time":0.0,"rotate":[0,170,0]},
                               {"time":0.5,"rotate":[0,-170,0]},
                               {"time":1.0,"rotate":[0,170,0]}]}
    # Parent and child both change: the child swings on the Z axis while the
    # parent turntable rotates, and every frame samples both at one time.
    wedge_track = {"node":"wedge","interpolation":interpolation,
                   "keyframes":[{"time":0.0,"rotate":[0,0,-30]},
                                {"time":0.5,"rotate":[0,0,30]},
                                {"time":1.0,"rotate":[0,0,-30]}]}
    camera_track = {"interpolation":interpolation,
                    "keyframes":[{"time":0.0,"eye":[3,2.2,4]},
                                 {"time":0.5,"eye":[4,1.2,3]},
                                 {"time":1.0,"eye":[3,2.2,4]}]}
    return {"scene":None,"width":96,"height":80,"samples":1,"duration":1.0,
            "frame_count":8,"columns":4,"node_tracks":[root_track,wedge_track],
            "camera_track":camera_track}


def sequence_walkthrough(renderer,scene_id,output):
    modes = ["step","linear","smooth","smooth_tangent"]
    per_mode = {}
    linear_manifest = None
    # Explicit schedule puts the middle frame exactly on the t=0.5 keyframe,
    # so keyframe agreement can be asserted numerically.
    schedule = {"frame_times":[0.0,0.125,0.25,0.375,0.5,0.625,0.75,1.0]}
    for mode in modes:
        request = _sequence_request(mode)
        request["scene"] = scene_id
        request.update(schedule)
        record = renderer.sequence(request)
        manifest = record["details"]["manifest"]
        if mode == "linear":
            linear_manifest = manifest
        per_mode[mode] = {"frames":record["details"]["frames"],"record":record["id"]}
        if output:
            folder = Path(output,"sequence-"+mode)
            folder.mkdir(parents=True,exist_ok=True)
            for name in record["details"]["artifacts"]:
                Path(folder,name).write_bytes(renderer.artifacts.read(record,name))
    frames = linear_manifest["frames"]
    require(len(frames) == 8,"sequence_frames","manifest must list every frame")
    require(frames[4]["time"] == 0.5,"sequence_frames","middle frame must sit on the authored keyframe")

    # Boundary crossing: the lifted branch stays inside [170,190] degrees and
    # never takes the 340-degree long way around.
    root_y = [frame["inputs"]["node_poses"]["root"]["rotate"][1] for frame in frames]
    require(all(169.0 <= value <= 191.0 for value in root_y[:5]),"unwrap_boundary",
            "rotation must cross the +/-180 boundary on the short branch")
    adjacent = [abs(root_y[i+1]-root_y[i]) for i in range(len(root_y)-1)]
    require(max(adjacent) < 60.0,"unwrap_boundary","no frame jump may sweep the long angle")

    # Parent and child change at the same instant: both poses are recorded on
    # the same frame with one shared time.
    middle = frames[4]
    require(middle["time"] == 0.5 and "wedge" in middle["inputs"]["node_poses"]
            and "root" in middle["inputs"]["node_poses"],"sequence_time",
            "parent and child must be sampled at one shared time")
    camera_moves = len({round(frame["inputs"]["lens"]["eye"][0],9) for frame in frames}) > 1
    require(camera_moves,"sequence_camera","camera keyframes must move the lens")

    # Single-frame redo: the dedicated endpoint replays the frozen inputs and
    # must reproduce the batch artifacts byte for byte.
    redo = renderer.rerender_frame(per_mode["linear"]["record"],4)
    batch_record = renderer.catalog.get(per_mode["linear"]["record"])
    for name in redo["details"]["artifacts"]:
        batch = renderer.artifacts.read(batch_record,name)
        single = renderer.artifacts.read(redo,name)
        require(hashlib.sha256(batch).hexdigest() == hashlib.sha256(single).hexdigest(),
                "frame_determinism","single-frame redo must match the batch frame")

    # Interpolation continuity choices must be visibly different at a
    # non-keyframe while agreeing exactly on an authored keyframe.
    step_pose = renderer.catalog.get(per_mode["step"]["record"])["details"]["manifest"]["frames"][2]["inputs"]["node_poses"]["root"]
    smooth_pose = renderer.catalog.get(per_mode["smooth"]["record"])["details"]["manifest"]["frames"][2]["inputs"]["node_poses"]["root"]
    require(step_pose != smooth_pose,"interpolation_modes","step and smooth must differ between keyframes")
    keyframe_step = renderer.catalog.get(per_mode["step"]["record"])["details"]["manifest"]["frames"][4]["inputs"]["lens"]
    keyframe_smooth = renderer.catalog.get(per_mode["smooth"]["record"])["details"]["manifest"]["frames"][4]["inputs"]["lens"]
    require(keyframe_step == keyframe_smooth,"interpolation_modes","all modes agree on authored keyframes")

    # Failures: unknown node, non-increasing keyframe times.
    bad_node = _sequence_request("linear"); bad_node["scene"] = scene_id
    bad_node["node_tracks"][0]["node"] = "ghost"
    expect_rejection(lambda:renderer.sequence(bad_node),"track_node")
    bad_time = _sequence_request("linear"); bad_time["scene"] = scene_id
    bad_time["node_tracks"][0]["keyframes"][1]["time"] = 1.0
    expect_rejection(lambda:renderer.sequence(bad_time),"track_keyframes")

    # Explicit frame schedule: every requested time gets its own frame.
    explicit = _sequence_request("linear"); explicit["scene"] = scene_id
    explicit["frame_times"] = [0.0,0.25,0.5,0.75,1.0]
    explicit_record = renderer.sequence(explicit)
    times = [frame["time"] for frame in explicit_record["details"]["manifest"]["frames"]]
    require(times == [0.0,0.25,0.5,0.75,1.0],"sequence_frames","explicit frame times must be honored")
    return {"modes":list(per_mode),"frames_per_mode":8,
            "boundary_root_rotation":root_y,"single_frame_match":True,
            "mode_difference_observed":True,"rejections":["track_node","track_keyframes"],
            "explicit_schedule":times}


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
        if flow in {"all","keyframe-sequence"}:
            results["keyframe-sequence"] = sequence_walkthrough(renderer,scene["id"],output)
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
