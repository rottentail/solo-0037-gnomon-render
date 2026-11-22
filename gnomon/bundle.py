import hashlib
import json
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED


def build_bundle(render,files):
    semantics={"color":"linear RGB in NPZ; clamped sRGB in PNG",
               "depth":"OpenGL NDC [-1,1], +inf for background",
               "world":"nearest covered subpixel world position",
               "coverage":"fraction of covered uniform samples",
               "normal":"world-space unit normal at nearest sample",
               "object_id":"scene node index, 0 background"}
    if any(name.startswith("line-") for name in files):
        semantics.update({
            "line-silhouette":"grayscale coverage of outer contours against background",
            "line-occlusion":"grayscale coverage of inter-object depth-discontinuity boundaries",
            "line-crease":"grayscale coverage of internal dihedral folds above crease_angle",
            "line-layer":"transparent RGBA composite of enabled structures; geometry-anchored, width in output pixels",
            "line-overlay":"enabled lines composited over the rendered color",
            "lines.npz":"fractional per-pixel line coverages and linear RGBA layer"})
    manifest = {"format":"gnomon-render-bundle/1","render":render,
                "files":{name:{"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw)} for name,raw in files.items()},
                "semantics":semantics}
    output = BytesIO()
    with ZipFile(output,"w",ZIP_DEFLATED) as archive:
        for name,raw in sorted(files.items()):
            archive.writestr(name,raw)
        archive.writestr("manifest.json",json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False))
    return output.getvalue(),manifest
