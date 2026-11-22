import hashlib
import json
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED


SEMANTICS = {"color": "linear RGB in NPZ; clamped sRGB in PNG",
             "depth": "OpenGL NDC [-1,1], +inf for background",
             "world": "nearest covered subpixel world position",
             "coverage": "fraction of covered uniform samples",
             "normal": "world-space unit normal at nearest sample",
             "object_id": "scene node index, 0 background"}

STEREO_SEMANTICS = {
    "left-color.png / right-color.png": "each eye rendered from its own translated world-space camera",
    "right-channels.npz": "right-eye channels with the same layout as channels.npz",
    "correspondence.npz": "per-pixel status, horizontal/vertical parallax, destination pixel, depth and occluder for left-to-right and right-to-left",
    "correspondence.png": "left-eye color with occluded or mismatched left pixels tinted red",
    "occlusion-mask.png": "per-status color mask for the left-to-right direction",
    "parallax.png": "red uncrossed / blue crossed horizontal parallax, grey untrusted",
    "pair.png": "left and right eyes side by side with explicit baseline and convergence units",
    "anaglyph.png": "red from the left eye, green and blue from the right",
    "stereo.json": "rig geometry, unit conversions and per-direction correspondence summary",
    "crosscheck.json": "independent analytic parallax probes and rig invariants",
}


def build_bundle(render, files):
    semantics = dict(SEMANTICS)
    metadata = render.get("details", {}).get("metadata", {})
    if metadata.get("kind") == "stereo-render":
        semantics.update(STEREO_SEMANTICS)
        semantics["stereo_units"] = metadata.get("units")
        semantics["parallax_convention"] = "full-frame pixels, right_x - left_x; negative crossed"
        semantics["trust_rule"] = metadata.get("correspondence", {}).get("trust_rule")
    manifest = {"format": "gnomon-render-bundle/1", "render": render,
                "files": {name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
                          for name, raw in files.items()},
                "semantics": semantics}
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for name, raw in sorted(files.items()):
            archive.writestr(name, raw)
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))
    return output.getvalue(), manifest
