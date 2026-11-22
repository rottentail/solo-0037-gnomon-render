import base64
from io import BytesIO
from PIL import Image


def wedge_mesh():
    # A closed asymmetric prism with authored smooth vertex normals.
    return {"label":"折光楔体","positions":[[-1,-0.7,-0.5],[1,-0.7,-0.5],[0.5,0.9,-0.5],[-1,-0.7,0.5],[1,-0.7,0.5],[0.5,0.9,0.5]],
            "triangles":[[0,2,1],[3,4,5],[0,1,4],[0,4,3],[1,2,5],[1,5,4],[2,0,3],[2,3,5]],
            "uv":[[0,0],[1,0],[0.75,1],[0,0],[1,0],[0.75,1]]}


def screen_mesh():
    return {"label":"镂空色片","positions":[[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],
            "triangles":[[0,1,2],[0,2,3]],"normals":[[0,0,1]]*4,"uv":[[0,0],[1,0],[1,1],[0,1]]}


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


def plate_pair_scene(mesh,gap,projection="orthographic"):
    # A small plate hovering over a larger floor plate, seen straight from
    # above: with a detail radius larger than `gap` the floor darkens around
    # the hover footprint; with a gap beyond the radius the projection overlap
    # must not read as contact.
    return {"label":"贴合板遮蔽观测","nodes":[
        {"key":"floor","mesh":mesh,"pose":{"scale":[1.4,1.4,1.4]},"surface":{"color":[0.75,0.72,0.7]}},
        {"key":"hover","mesh":mesh,"pose":{"translate":[0,0,gap],"scale":[0.35,0.35,0.35]},"surface":{"color":[0.72,0.7,0.75]}}
    ],"lens":{"projection":projection,"eye":[0,0,4],"target":[0,0,0],"near":1.0,"far":30,"span":3.2,"fov":45},
    "light":{"toward":[1,2,4],"energy":0.7,"ambient":0.35}}
