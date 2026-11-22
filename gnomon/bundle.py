import hashlib
import json
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED


def build_bundle(render,files):
    manifest = {"format":"gnomon-render-bundle/1","render":render,
                "files":{name:{"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw)} for name,raw in files.items()},
                "semantics":{"color":"linear RGB in NPZ; clamped sRGB in PNG",
                             "depth":"OpenGL NDC [-1,1], +inf for background",
                             "world":"picked subpixel world position (nearest opaque, else nearest translucent)",
                             "coverage":"fraction of subpixels hit by any surface, translucent or opaque",
                             "normal":"world-space unit normal at the picked subpixel",
                             "object_id":"scene node index of the picked surface, 0 background",
                             "layer_count":"translucent fragments composited at the picked subpixel",
                             "through_ids":"front-to-back translucent node ids in front of the picked surface",
                             "translucency":"opacity<1 fragments are depth-sorted Porter-Duff source-over; alpha_cutoff stays binary"}}
    output = BytesIO()
    with ZipFile(output,"w",ZIP_DEFLATED) as archive:
        for name,raw in sorted(files.items()):
            archive.writestr(name,raw)
        archive.writestr("manifest.json",json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False))
    return output.getvalue(),manifest
