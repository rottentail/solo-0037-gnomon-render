import hashlib
import json
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED


def build_bundle(render,files):
    metadata = render["details"]["metadata"]
    equirect = metadata.get("lens_projection") == "equirectangular"
    semantics = {"color":"linear RGB in NPZ; clamped sRGB in PNG",
                 "world":"nearest covered subpixel world position",
                 "coverage":"fraction of covered uniform samples",
                 "normal":"world-space unit normal at nearest sample",
                 "object_id":"scene node index, 0 background"}
    lens_contract = {"lens":metadata.get("lens"),"lens_projection":metadata.get("lens_projection")}
    if equirect:
        semantics.update({
            "depth":"ray distance in world units from lens eye, +inf for background; not NDC",
            "ray":"unit world-space ray direction per pixel; yaw=atan2(ray.x,-ray.z), pitch=asin(ray.y)",
            "layout":"equirectangular longitude/latitude map, 2:1; column edges are the same yaw seam; first/last rows touch the poles"})
        lens_contract["difference_from_perspective_lens"] = (
            "full 360x180 view cast as one ray per equirectangular sub-sample; "
            "no perspective projection, no fov/span, no stretched perspective image; "
            "seam and poles are sampled directly in world space")
    else:
        semantics.update({"depth":"OpenGL NDC [-1,1], +inf for background"})
        lens_contract["difference_from_perspective_lens"] = "planar perspective/orthographic frame with homogeneous clipping"
    manifest = {"format":"gnomon-render-bundle/1","render":render,
                "imaging":lens_contract,
                "files":{name:{"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw)} for name,raw in files.items()},
                "semantics":semantics}
    output = BytesIO()
    with ZipFile(output,"w",ZIP_DEFLATED) as archive:
        for name,raw in sorted(files.items()):
            archive.writestr(name,raw)
        archive.writestr("manifest.json",json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False))
    return output.getvalue(),manifest
