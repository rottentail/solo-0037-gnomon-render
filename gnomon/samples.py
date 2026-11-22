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


def transparency_scene(mesh,screen,reversed_order=False):
    """Intersecting translucent color sheets in front of and behind a solid wedge.

    The red and blue sheets overlap in image space; a green sheet sits behind
    the wedge. Unlit colors keep the composited color math explainable.
    """
    wedge = {"key":"wedge","mesh":mesh,"surface":{"color":[0.85,0.5,0.15],"unlit":True},
             "pose":{"rotate":[0,-20,0],"scale":[0.8,0.8,0.8]}}
    red = {"key":"red_sheet","mesh":screen,
           "pose":{"translate":[-0.45,0.15,-1.1],"rotate":[0,25,0],"scale":[0.85,0.85,0.85]},
           "surface":{"color":[0.9,0.08,0.08],"opacity":0.5,"unlit":True}}
    blue = {"key":"blue_sheet","mesh":screen,
            "pose":{"translate":[0.45,0.05,-1.25],"rotate":[0,-25,0],"scale":[0.85,0.85,0.85]},
            "surface":{"color":[0.08,0.2,0.95],"opacity":0.5,"unlit":True}}
    green = {"key":"green_sheet","mesh":screen,
             "pose":{"translate":[0.0,-0.05,1.15],"rotate":[0,10,0],"scale":[1.0,1.0,1.0]},
             "surface":{"color":[0.1,0.8,0.2],"opacity":0.45,"unlit":True}}
    occluder = {"key":"blocker","mesh":mesh,
                "pose":{"translate":[1.15,-0.55,-1.4],"scale":[0.3,0.3,0.3]},
                "surface":{"color":[0.25,0.25,0.28],"unlit":True}}
    nodes = [wedge,red,blue,green,occluder]
    if reversed_order:
        nodes.reverse()
    return {"label":"半透明交叠色片观测","nodes":nodes,
            "lens":{"eye":[0,0.2,4.2],"target":[0,0,0],"near":0.1,"far":25},
            "light":{"toward":[1,3,4],"energy":0.85,"ambient":0.18}}


def dense_stack_scene(screen,layers):
    """A dense stack of co-located translucent sheets to exercise the layer cap."""
    nodes = []
    for index in range(layers):
        nodes.append({"key":f"sheet_{index:02d}","mesh":screen,
                      "pose":{"translate":[0,0,1.2-index*0.06],"scale":[1.1,1.1,1.1]},
                      "surface":{"color":[0.9,0.4,0.7],"opacity":0.5,"unlit":True}})
    return {"label":"密集叠层边界观测","nodes":nodes,
            "lens":{"eye":[0,0,4.2],"target":[0,0,0],"near":0.1,"far":25},
            "light":{"toward":[1,3,4],"energy":0.85,"ambient":0.18}}
