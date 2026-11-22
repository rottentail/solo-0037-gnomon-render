from io import BytesIO
import numpy as np
from PIL import Image
from .color import encode_rgb
from .raster import MAX_TRANSLUCENT_LAYERS


def _arrange(source,h,w,samples):
    if source.ndim == 2:
        return source.reshape(h,samples,w,samples).transpose(0,2,1,3).reshape(h,w,-1)
    # Per-subpixel records (e.g. K translucent layers) keep their trailing axes.
    trailing = source.shape[2:]
    return source.reshape(h,samples,w,samples,*trailing).transpose(0,2,1,3,*range(4,4+len(trailing))).reshape(h,w,-1,*trailing)


def resolve_samples(buffers,samples):
    sh,sw = buffers["depth"].shape
    h,w = sh//samples,sw//samples
    if not buffers.get("has_translucent"):
        # Opaque/cutout-only path: baseline semantics, unchanged.
        color = buffers["color"].reshape(h,samples,w,samples,3).mean(axis=(1,3))
        depth = _arrange(buffers["depth"],h,w,samples)
        closest = depth.argmin(axis=2)
        yy,xx = np.indices((h,w))
        resolved = {"color":color,
                    "coverage":(buffers["object_id"]>0).reshape(h,samples,w,samples).mean(axis=(1,3))}
        for key in ("depth","object_id","normal","world"):
            arranged = _arrange(buffers[key],h,w,samples)
            resolved[key] = arranged[yy,xx,closest]
        resolved["sample_offset"] = np.stack([(closest%samples+0.5)/samples,(closest//samples+0.5)/samples],axis=2)
        return resolved

    background = buffers["background"]
    count = _arrange(buffers["layer_count"],h,w,samples)                       # (h,w,s2)
    opaque_id = _arrange(buffers["object_id"],h,w,samples)
    opaque_depth = _arrange(buffers["depth"],h,w,samples)
    opaque_world = _arrange(buffers["world"],h,w,samples)
    opaque_normal = _arrange(buffers["normal"],h,w,samples)
    layer_id = _arrange(buffers["layer_id"],h,w,samples)                       # (h,w,s2,K)
    layer_depth = _arrange(buffers["layer_depth"],h,w,samples)
    layer_rgb = _arrange(buffers["layer_rgb"],h,w,samples)
    layer_alpha = _arrange(buffers["layer_alpha"],h,w,samples)
    layer_world = _arrange(buffers["layer_world"],h,w,samples)
    layer_normal = _arrange(buffers["layer_normal"],h,w,samples)

    # Stable depth sort makes the result independent of scene node order.
    # Empty slots carry infinite depth, so one shared order also packs them last.
    order = np.argsort(layer_depth,axis=3,kind="stable")
    layer_id = np.take_along_axis(layer_id,order,axis=3)
    layer_rgb = np.take_along_axis(layer_rgb,order[...,None],axis=3)
    layer_alpha = np.take_along_axis(layer_alpha,order,axis=3)
    layer_world = np.take_along_axis(layer_world,order[...,None],axis=3)
    layer_normal = np.take_along_axis(layer_normal,order[...,None],axis=3)
    layer_depth = np.take_along_axis(layer_depth,order,axis=3)

    # Exact front-to-back Porter-Duff source-over in linear light. Accumulation
    # starts from black with zero covered weight; the background and the opaque
    # winner enter through the remaining carry exactly once each.
    accum = np.zeros((*count.shape,3))
    carry = np.ones(count.shape)
    for k in range(MAX_TRANSLUCENT_LAYERS):
        active = k < count
        a = np.where(active,layer_alpha[...,k],0.0)[...,None]
        accum += carry[...,None]*a*layer_rgb[...,k,:]
        carry *= np.where(active,1-layer_alpha[...,k],1.0)
    has_opaque = opaque_id > 0
    accum += carry[...,None]*np.where(has_opaque[...,None],
                                      _arrange(buffers["color"],h,w,samples),background)
    pixel_color = accum.reshape(h,w,samples*samples,3).mean(axis=2)
    # Coverage means geometric coverage: any surface (translucent or opaque).
    covered = has_opaque | (count > 0)
    coverage = covered.reshape(h,w,samples*samples).mean(axis=2)

    # Picking: nearest opaque sub-sample surface wins; translucent front layers
    # never block selection. With no opaque surface behind, the nearest
    # translucent layer is selected and reported as such.
    pick_depth = np.where(has_opaque,opaque_depth,layer_depth[...,0])
    pickable = has_opaque | (count > 0)
    pick_depth = np.where(pickable,pick_depth,np.inf)
    closest = pick_depth.argmin(axis=2)
    yy,xx = np.indices((h,w))
    def pick(array):
        return array[yy,xx,closest]
    picked_opaque = pick(has_opaque)
    picked_count = pick(count).astype(np.int32)
    picked_layer_id = pick(layer_id)
    final_id = np.where(picked_opaque,pick(opaque_id),picked_layer_id[...,0]).astype(np.int32)
    final_depth = np.where(picked_opaque,pick(opaque_depth),pick(layer_depth)[...,0])
    final_world = np.where(picked_opaque[...,None],pick(opaque_world),pick(layer_world)[...,0,:])
    final_normal = np.where(picked_opaque[...,None],pick(opaque_normal),pick(layer_normal)[...,0,:])
    final_world = np.where((final_id>0)[...,None],final_world,0.0)
    final_normal = np.where((final_id>0)[...,None],final_normal,0.0)
    slot = np.arange(MAX_TRANSLUCENT_LAYERS)[None,None,:]
    # Only real stored layers are reported; id 0 marks an empty slot.
    through_ids = np.where((slot < picked_count[...,None]) & (picked_layer_id > 0),
                           picked_layer_id,0).astype(np.int32)
    return {"color":pixel_color,"coverage":coverage,"depth":final_depth,
            "object_id":final_id,"normal":final_normal,"world":final_world,
            "layer_count":picked_count,"through_ids":through_ids,
            "sample_offset":np.stack([(closest%samples+0.5)/samples,(closest//samples+0.5)/samples],axis=2)}


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
