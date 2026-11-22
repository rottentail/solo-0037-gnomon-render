import numpy as np
from .errors import require


def inspect_pixel(channels,metadata,x,y):
    h,w = channels["depth"].shape
    require(0<=x<w and 0<=y<h,"pixel_bounds","pixel outside output region")
    identity = int(channels["object_id"][y,x])
    offset = channels["sample_offset"][y,x]
    rx,ry,_,_ = metadata["region"]
    background = identity==0
    point = None if background else channels["world"][y,x].tolist()
    normal = None if background else channels["normal"][y,x].tolist()
    # Renders made before face-level materials existed have no face/material
    # channels; they still resolve to the node and its single default surface.
    face = None if background or "face_id" not in channels else int(channels["face_id"][y,x])
    group = None if background or "material_id" not in channels else int(channels["material_id"][y,x])
    names = metadata.get("materials",{}).get(str(identity)) or ["default"]
    material = None if group is None else names[min(group,len(names)-1)]
    return {"pixel":[x,y],"full_sample":[float(rx+x+offset[0]),float(ry+y+offset[1])],
            "coverage":float(channels["coverage"][y,x]),"object_id":identity,
            "node":metadata["node_map"].get(str(identity)),
            "mesh":metadata.get("meshes",{}).get(str(identity)) if not background else None,
            "face":face,"material":material,"world":point,"normal":normal,
            "ndc_depth":None if background else float(channels["depth"][y,x]),
            "linear_rgb":channels["color"][y,x].tolist(),"selection":"nearest covered subpixel"}
