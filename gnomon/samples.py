import base64
from io import BytesIO
import numpy as np
from PIL import Image


def wedge_mesh():
    # A closed asymmetric prism with authored smooth vertex normals.
    return {"label":"折光楔体","positions":[[-1,-0.7,-0.5],[1,-0.7,-0.5],[0.5,0.9,-0.5],[-1,-0.7,0.5],[1,-0.7,0.5],[0.5,0.9,0.5]],
            "triangles":[[0,2,1],[3,4,5],[0,1,4],[0,4,3],[1,2,5],[1,5,4],[2,0,3],[2,3,5]],
            "uv":[[0,0],[1,0],[0.75,1],[0,0],[1,0],[0.75,1]]}


def screen_mesh():
    return {"label":"镂空色片","positions":[[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],
            "triangles":[[0,1,2],[0,2,3]],"normals":[[0,0,1]]*4,"uv":[[0,0],[1,0],[1,1],[0,1]]}


def _cylinder_x(label,radius,length,segments):
    # Closed cylinder whose symmetry axis is +X, with explicit per-face normals.
    positions,normals,triangles = [],[],[]
    ring = [(radius*np.cos(2*np.pi*i/segments),radius*np.sin(2*np.pi*i/segments)) for i in range(segments)]
    for x in (-length/2,length/2):
        for y,z in ring:
            positions.append([x,y,z]); normals.append([0,y/radius,z/radius])
    for i in range(segments):
        j = (i+1)%segments
        triangles.append([i,j,segments+j]); triangles.append([i,segments+j,segments+i])
    left_center = len(positions); positions.append([-length/2,0,0]); normals.append([-1,0,0])
    left_ring = len(positions)
    for y,z in ring:
        positions.append([-length/2,y,z]); normals.append([-1,0,0])
    for i in range(segments):
        triangles.append([left_center,left_ring+i,left_ring+(i+1)%segments])
    right_center = len(positions); positions.append([length/2,0,0]); normals.append([1,0,0])
    right_ring = len(positions)
    for y,z in ring:
        positions.append([length/2,y,z]); normals.append([1,0,0])
    for i in range(segments):
        triangles.append([right_center,right_ring+(i+1)%segments,right_ring+i])
    return {"label":label,"positions":positions,"triangles":triangles,"normals":normals}


def hub_mesh():
    return _cylinder_x("轮毂筒体",0.8,1.6,12)


def cap_mesh():
    return _cylinder_x("端盖",0.55,0.5,10)


def bolt_mesh():
    return _cylinder_x("六角螺栓",0.22,0.5,6)


def checker_texture():
    image = Image.new("RGBA",(32,32))
    for y in range(32):
        for x in range(32):
            hole = (x-16)**2+(y-16)**2 < 45
            color = (242,171,76,0 if hole else 255) if (x//8+y//8)%2 else (67,182,180,0 if hole else 255)
            image.putpixel((x,y),color)
    output = BytesIO()
    image.save(output,format="PNG")
    return {"label":"青金镂空纹理","png_base64":base64.b64encode(output.getvalue()).decode()}


def compose_scene(mesh,screen,texture):
    return {"label":"折面色片静帧","nodes":[
        {"key":"root","pose":{"rotate":[0,15,0]}},
        {"key":"wedge","parent":"root","mesh":mesh,"surface":{"color":[0.87,0.38,0.19]},"pose":{"rotate":[0,-20,0]}},
        {"key":"foil","parent":"root","mesh":screen,"pose":{"translate":[-0.7,0.4,-0.9],"scale":[0.7,0.7,0.7]},"surface":{"texture":texture,"color":[1,1,1],"alpha_cutoff":0.5}},
        {"key":"echo","mesh":mesh,"pose":{"translate":[1.0,-0.6,-0.8],"scale":[0.35,0.35,0.35]},"surface":{"color":[0.2,0.7,0.8]}}
    ],"lens":{"eye":[3,2.2,4],"target":[0,0,0],"near":0.1,"far":30},
    "light":{"toward":[1,3,4],"energy":0.85,"ambient":0.18}}


def assembly_scene(hub_mesh,cap_mesh,bolt_mesh):
    # Nested carrier/cap hierarchy; the bolt mesh is shared by two sibling nodes.
    return {"label":"同轴组件装配","nodes":[
        {"key":"rig","pose":{}},
        {"key":"hub","parent":"rig","mesh":hub_mesh,
         "surface":{"color":[0.45,0.55,0.68]},"pose":{"rotate":[0,20,0]}},
        {"key":"carrier","parent":"hub","pose":{"translate":[1.1,0.0,0.0],"rotate":[0,15,0]}},
        {"key":"cap","parent":"carrier","mesh":cap_mesh,
         "surface":{"color":[0.83,0.42,0.25]},"pose":{"translate":[0.9,0.0,0.0]}},
        {"key":"pin","parent":"carrier","mesh":bolt_mesh,
         "surface":{"color":[0.85,0.72,0.35]},"pose":{"translate":[0.25,0.55,0.0],"scale":[0.45,0.45,0.45]}},
        {"key":"bolt_a","parent":"rig","mesh":bolt_mesh,
         "surface":{"color":[0.35,0.7,0.55]},"pose":{"translate":[-0.9,0.65,0.65],"scale":[0.4,0.4,0.4]}},
        {"key":"bolt_b","parent":"rig","mesh":bolt_mesh,
         "surface":{"color":[0.35,0.7,0.55]},"pose":{"translate":[-0.9,-0.65,-0.65],"scale":[0.4,0.4,0.4]}}
    ],"lens":{"eye":[4.5,3.0,5.5],"target":[0.6,0,0],"near":0.1,"far":40},
    "light":{"toward":[2,4,3],"energy":0.85,"ambient":0.2}}
