from io import BytesIO
import numpy as np
from PIL import Image
from .color import encode_rgb


def resolve_samples(buffers,samples):
    sh,sw = buffers["depth"].shape
    h,w = sh//samples,sw//samples
    color = buffers["color"].reshape(h,samples,w,samples,3).mean(axis=(1,3))
    depth = buffers["depth"].reshape(h,samples,w,samples).transpose(0,2,1,3).reshape(h,w,-1)
    closest = depth.argmin(axis=2)
    yy,xx = np.indices((h,w))
    resolved = {"color":color,"coverage":(buffers["object_id"]>0).reshape(h,samples,w,samples).mean(axis=(1,3))}
    for key in ("depth","object_id","normal","world","reflection_id","reflection_world"):
        if key not in buffers:
            continue
        source = buffers[key]
        if source.ndim == 2:
            arranged = source.reshape(h,samples,w,samples).transpose(0,2,1,3).reshape(h,w,-1)
        else:
            arranged = source.reshape(h,samples,w,samples,3).transpose(0,2,1,3,4).reshape(h,w,-1,3)
        resolved[key] = arranged[yy,xx,closest]
    # Pixel inspection uses the nearest subpixel, not the pixel center.
    resolved["sample_offset"] = np.stack([(closest%samples+0.5)/samples,(closest//samples+0.5)/samples],axis=2)
    return resolved


def png_bytes(rgb):
    output = BytesIO()
    Image.fromarray(rgb).save(output,format="PNG")
    return output.getvalue()


def preview_channels(channels):
    valid = channels["object_id"]>0
    depth = channels["depth"]
    grey = np.zeros(depth.shape)
    if valid.any():
        lo,hi = depth[valid].min(),depth[valid].max()
        grey[valid] = 0.25+0.75*(1-(depth[valid]-lo)/max(hi-lo,1e-12))
    identifiers = channels["object_id"]
    colors = np.stack([(identifiers*73)%255,(identifiers*151)%255,(identifiers*199)%255],axis=2).astype(np.uint8)
    normal = np.rint(np.clip(channels["normal"]*0.5+0.5,0,1)*255).astype(np.uint8)
    normal[~valid] = 0
    return {"color.png":png_bytes(encode_rgb(channels["color"])),
            "depth.png":png_bytes(np.rint(grey*255).astype(np.uint8)),
            "normal.png":png_bytes(normal),"object-id.png":png_bytes(colors),
            "coverage.png":png_bytes(np.rint(channels["coverage"]*255).astype(np.uint8))}


def packed_channels(channels):
    output = BytesIO()
    np.savez_compressed(output,**channels)
    return output.getvalue()


def unpack_channels(raw):
    with np.load(BytesIO(raw),allow_pickle=False) as archive:
        return {name:archive[name] for name in archive.files}
