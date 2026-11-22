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
    section_index = int(channels["section_id"][y,x]) if identity and "section_id" in channels else 0
    section = None
    if section_index:
        owner = metadata.get("section_map",{}).get(str(section_index),{})
        section = {"index":section_index,"id":owner.get("id"),"label":owner.get("label"),
                   "surface":"section_cap"}
    return {"pixel":[x,y],"full_sample":[float(rx+x+offset[0]),float(ry+y+offset[1])],
            "coverage":float(channels["coverage"][y,x]),"object_id":identity,
            "node":metadata["node_map"].get(str(identity)),"world":point,"normal":normal,
            "surface":None if identity==0 else "section_cap" if section_index else "mesh",
            "section":section,
            "ndc_depth":None if identity==0 else float(channels["depth"][y,x]),
            "linear_rgb":channels["color"][y,x].tolist(),"selection":"nearest covered subpixel"}
