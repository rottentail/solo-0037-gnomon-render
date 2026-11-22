"""Normal-map observation: one mesh, relief from texture only.

Writes the color and world-normal channels of a four-instance showcase plus a
per-instance report that ties the two channels together:

* upright, reflected, and non-uniformly scaled copies of the same seamed ramp;
* a flat normal map as the control (no relief);
* color = unlit diffuse * (ambient + energy * max(0, N.L)) per pixel;
* how missing UV, a normal map in the color slot, and zero UV area are refused.
"""
import json
import tempfile
from pathlib import Path
import numpy as np
from gnomon.service import RendererService
from gnomon.samples import (seam_ramp_mesh,panel_texture,ripple_normal_texture,
                            flat_normal_texture,normal_material_scene)
from gnomon.channels import unpack_channels
from gnomon.errors import RenderError

LENS = {"eye":[0,3.2,6],"target":[0,0.2,0],"near":0.1,"far":30}
LIGHT = {"toward":[-1,2,1.5],"energy":0.9,"ambient":0.15}
CHANNELS = ("color","normal","channels","metadata")


def rejection(operation):
    try:
        operation()
    except RenderError as exc:
        return {"code":exc.code,"detail":exc.detail}
    raise RuntimeError("expected a domain rejection")


def report(channels,surface_light):
    covered = channels["object_id"]>0
    normals = channels["normal"][covered]
    cosine = np.maximum(0,normals @ surface_light)
    bands = {}
    for identity in np.unique(channels["object_id"]):
        if identity == 0:
            continue
        mask = channels["object_id"]==identity
        bands[str(int(identity))] = {"pixels":int(mask.sum()),
                                     "world_normal_x":[float(channels["normal"][mask,0].min()),
                                                       float(channels["normal"][mask,0].max())],
                                     "world_normal_y_mean":float(channels["normal"][mask,1].mean())}
    return {"covered_pixels":int(covered.sum()),
            "ndotl_range":[float(cosine.min()),float(cosine.max())],
            "color_luminance_range":[float(channels["color"][covered].min()),
                                     float(channels["color"][covered].max())],
            "instances":bands}


def produce(output):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gnomon-normal-") as root:
        service = RendererService(root)
        try:
            mesh = service.add_mesh(seam_ramp_mesh())["id"]
            panel = service.add_texture(panel_texture())["id"]
            ripple = service.add_texture(ripple_normal_texture())["id"]
            flat = service.add_texture(flat_normal_texture())["id"]

            showcase = normal_material_scene(mesh,panel,ripple)
            (output/"showcase-scene.json").write_text(json.dumps(showcase,ensure_ascii=False,indent=2))
            scene = service.add_scene(showcase)
            rendered = service.render({"scene":scene["id"],"width":480,"height":240,"samples":1})
            folder = output/"showcase";folder.mkdir(exist_ok=True)
            for artifact in rendered["details"]["artifacts"]:
                (folder/artifact).write_bytes(service.artifacts.read(rendered,artifact))

            surface = {"texture":panel,"normal_texture":ripple,"color":[1,1,1]}
            def pair(extra,pose=None):
                node = {"key":"panel","mesh":mesh,"surface":{**surface,**extra}}
                if pose:
                    node["pose"] = pose
                local = service.add_scene({"label":"法线观测","nodes":[node],"lens":LENS,"light":LIGHT})
                image = service.render({"scene":local["id"],"width":160,"height":120,"samples":1})
                return image,unpack_channels(service.artifacts.read(image,"channels.npz"))
            unlit,_ = pair({"unlit":True})
            unlit_channels = unpack_channels(service.artifacts.read(unlit,"channels.npz"))
            mapped,mapped_channels = pair({})
            reflected,reflected_channels = pair({},{"scale":[-1,1,1]})
            stretched,stretched_channels = pair({},{"scale":[0.5,1.6,1]})
            flat_image,flat_channels = pair({"unlit":True,"normal_texture":flat})
            for name,image in (("mapped",mapped),("reflected",reflected),("stretched",stretched),("flat",flat_image)):
                target = output/name;target.mkdir(exist_ok=True)
                for artifact in ("color.png","normal.png"):
                    (target/artifact).write_bytes(service.artifacts.read(image,artifact))

            toward = np.asarray(LIGHT["toward"],dtype=float)
            toward /= np.linalg.norm(toward)
            covered = mapped_channels["object_id"]>0
            factor = LIGHT["ambient"]+LIGHT["energy"]*np.maximum(0,mapped_channels["normal"][covered] @ toward)
            recovered = mapped_channels["color"][covered]/np.maximum(factor[:,None],1e-9)
            roomy = np.maximum(0,mapped_channels["normal"][covered] @ toward) > 0.05
            lambert_error = float(np.abs(recovered[roomy]-unlit_channels["color"][covered][roomy]).max())
            flat_shift = float(np.abs(flat_channels["color"][flat_channels["object_id"]>0]
                                      -unlit_channels["color"][covered]).max())

            missing = service.add_mesh({**seam_ramp_mesh(),"uv":None})["id"]
            collapsed = service.add_mesh({**seam_ramp_mesh(),
                                          "label":"退化UV",
                                          "uv":[[i*0.125,0.0] for i in range(8)]})["id"]
            def scene_for(mesh_id,surface_overrides):
                return lambda: service.add_scene({"label":"拒绝","nodes":[
                    {"key":"panel","mesh":mesh_id,"surface":surface_overrides}],"lens":LENS,"light":LIGHT})
            refusals = {"missing_uv":rejection(scene_for(missing,{"normal_texture":ripple})),
                        "normal_in_color_slot":rejection(scene_for(mesh,{"texture":ripple})),
                        "color_in_normal_slot":rejection(scene_for(mesh,{"normal_texture":panel})),
                        "zero_uv_area":rejection(scene_for(collapsed,{"normal_texture":ripple}))}

            observations = {
                "showcase":report(unpack_channels(service.artifacts.read(rendered,"channels.npz")),toward),
                "mapped":report(mapped_channels,toward),
                "reflected":report(reflected_channels,toward),
                "stretched":report(stretched_channels,toward),
                "controls":{"flat_map_color_delta_vs_unlit":flat_shift,
                            "lambert_recovered_diffuse_error":lambert_error},
                "refusals":refusals,
                "channel_relation":"color = unlit_diffuse * (ambient + energy * max(0, world_normal . light))",
                "artifacts":{"color":"linear RGB in NPZ / sRGB PNG","normal":"world-space unit normal incl. normal map"}}
            (output/"observations.json").write_text(json.dumps(observations,ensure_ascii=False,indent=2))
        finally:
            service.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("run-output/normal-material"))
    produce(parser.parse_args().output)
