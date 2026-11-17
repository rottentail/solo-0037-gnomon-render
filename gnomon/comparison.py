import numpy as np
from .errors import require
from .channels import png_bytes


def compare_frames(left,right,left_meta,right_meta,threshold):
    for key in ("view","projection","region","width","height","samples"):
        require(left_meta[key]==right_meta[key],"comparison_camera",f"comparison requires matching {key}",409)
    require(left["color"].shape==right["color"].shape,"comparison_shape","output shapes differ",409)
    delta = np.abs(left["color"]-right["color"])
    magnitude = delta.max(axis=2)
    lm,rm = left["object_id"]>0,right["object_id"]>0
    common = lm & rm
    union = lm | rm
    depth_delta = np.abs(left["depth"][common]-right["depth"][common])
    # Node identity is semantic; compact numeric IDs can differ between revisions.
    left_names = np.empty(lm.shape,dtype=object)
    right_names = np.empty(rm.shape,dtype=object)
    for channel,names,meta in ((left,left_names,left_meta),(right,right_names,right_meta)):
        names[:] = ""
        for key,value in meta["node_map"].items():
            names[channel["object_id"]==int(key)] = value
    heat = np.stack([np.clip(magnitude*4,0,1),np.zeros_like(magnitude),np.clip((lm^rm).astype(float),0,1)],axis=2)
    metrics = {"linear_mae":float(delta.mean()),"changed_pixels":int((magnitude>threshold).sum()),
               "silhouette_iou":float((lm&rm).sum()/union.sum()) if union.any() else 1.0,
               "node_changes":int((left_names!=right_names).sum()),
               "max_ndc_depth_delta":float(depth_delta.max()) if len(depth_delta) else None,
               "threshold":threshold,"pixels":int(magnitude.size)}
    return metrics,png_bytes(np.rint(heat*255).astype(np.uint8))
