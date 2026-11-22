import hashlib
import json
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED


def build_bundle(render,files):
    manifest = {"format":"gnomon-render-bundle/1","render":render,
                "files":{name:{"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw)} for name,raw in files.items()},
                "semantics":{"color":"linear RGB in NPZ; clamped sRGB in PNG",
                             "depth":"OpenGL NDC [-1,1], +inf for background",
                             "world":"nearest covered subpixel world position",
                             "coverage":"fraction of covered uniform samples",
                             "normal":"world-space unit normal at nearest sample",
                             "object_id":"scene node index, 0 background",
                             "reflection_id":"mirror pixels only: node index seen in the reflection, 0 none",
                             "reflection_world":"mirror pixels only: world position of the reflected surface"}}
    output = BytesIO()
    with ZipFile(output,"w",ZIP_DEFLATED) as archive:
        for name,raw in sorted(files.items()):
            archive.writestr(name,raw)
        archive.writestr("manifest.json",json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False))
    return output.getvalue(),manifest
