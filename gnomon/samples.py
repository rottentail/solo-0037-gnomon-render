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


def stereo_scene(mesh,screen):
    """Three unlit panels at distinct depths plus a near prism for occlusion.

    Depths are chosen so the pair has crossed and uncrossed parallax together
    with genuine occlusion and silhouette disagreement between the eyes.
    """
    return {"label":"立体深度层","nodes":[
        {"key":"backdrop","mesh":screen,"pose":{"translate":[0,0,-2.2],"scale":[1.2,1.2,1.2]},
         "surface":{"color":[0.22,0.36,0.72],"unlit":True}},
        {"key":"middle","mesh":screen,"pose":{"translate":[-0.15,0.1,0.0],"scale":[0.55,0.55,0.55]},
         "surface":{"color":[0.28,0.78,0.34],"unlit":True}},
        {"key":"foreground","mesh":mesh,"pose":{"translate":[0.5,-0.15,1.1],"scale":[0.32,0.32,0.32]},
         "surface":{"color":[0.88,0.48,0.14],"unlit":True}},
        {"key":"edge","mesh":screen,"pose":{"translate":[-1.05,0.55,-0.6],"scale":[0.4,0.4,0.4]},
         "surface":{"color":[0.72,0.3,0.68],"unlit":True}},
    ],"lens":{"eye":[0,0,5],"target":[0,0,0],"near":0.1,"far":40,"fov":40},
    "light":{"toward":[1,3,4],"energy":0.85,"ambient":0.18}}


def stereo_rig(interocular_value=6.5,interocular_unit="cm",convergence=None,convergence_unit="m"):
    rig={"interocular":{"value":interocular_value,"unit":interocular_unit},"world_unit":"m"}
    if convergence is not None:
        rig["convergence"]={"value":convergence,"unit":convergence_unit}
    return rig
