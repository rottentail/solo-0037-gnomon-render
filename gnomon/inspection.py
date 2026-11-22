import numpy as np
from .errors import require


def inspect_pixel(channels,metadata,x,y):
    h,w = channels["depth"].shape
    require(0<=x<w and 0<=y<h,"pixel_bounds","pixel outside output region")
    identity = int(channels["object_id"][y,x])
    offset = channels["sample_offset"][y,x]
    rx,ry,_,_ = metadata["region"]
    point = None if identity==0 else channels["world"][y,x].tolist()
    normal = None if identity==0 else channels["normal"][y,x].tolist()
    reflected = None
    if "reflection_id" in channels:
        target = int(channels["reflection_id"][y,x])
        if target:
            reflected = {"object_id":target,"node":metadata["node_map"].get(str(target)),
                         "world":channels["reflection_world"][y,x].tolist()}
    return {"pixel":[x,y],"full_sample":[float(rx+x+offset[0]),float(ry+y+offset[1])],
            "coverage":float(channels["coverage"][y,x]),"object_id":identity,
            "node":metadata["node_map"].get(str(identity)),"world":point,"normal":normal,
            "ndc_depth":None if identity==0 else float(channels["depth"][y,x]),
            "linear_rgb":channels["color"][y,x].tolist(),"selection":"nearest covered subpixel",
            "mirror":metadata.get("mirrors",{}).get(str(identity)),"reflection":reflected}
