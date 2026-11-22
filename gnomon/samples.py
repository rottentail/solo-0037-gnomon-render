import base64
import math
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


def arc_array_scene(mesh,count=12,radius=3.0,arc_degrees=150):
    # One wedge repeated along an arc; each instance keeps its own pose and color.
    step = arc_degrees/max(count-1,1)
    instances = []
    for i in range(count):
        angle = -arc_degrees/2+step*i
        theta = math.radians(angle)
        tint = i/max(count-1,1)
        instances.append({"key":f"blade-{i+1:02d}",
                          "pose":{"translate":[round(radius*math.sin(theta),6),0,round(radius*math.cos(theta),6)],
                                  "rotate":[0,round(angle,6),0],"scale":[0.3,0.3,0.3]},
                          "surface":{"color":[round(0.85-0.55*tint,6),round(0.25+0.5*tint,6),round(0.2+0.6*tint,6)]}})
    del instances[0]["surface"]  # blade-01 inherits the node surface.
    return {"label":"楔体弧形阵列","nodes":[
        {"key":"carousel","pose":{"translate":[0,0.2,0],"rotate":[0,30,0]}},
        {"key":"ring","parent":"carousel","mesh":mesh,"pose":{"rotate":[8,0,0]},
         "surface":{"color":[0.87,0.38,0.19]},"instances":instances}
    ],"lens":{"eye":[0,3,9],"target":[0,0,2],"near":0.1,"far":40},
    "light":{"toward":[1,3,4],"energy":0.85,"ambient":0.18}}


def refined_mesh(base,iterations=1):
    # Midpoint subdivision with welded vertices; normals and UV are regenerated downstream.
    positions = [list(p) for p in base["positions"]]
    triangles = [list(t) for t in base["triangles"]]
    for _ in range(iterations):
        cache = {}
        def midpoint(a,b):
            key = (a,b) if a<b else (b,a)
            if key not in cache:
                pa,pb = positions[key[0]],positions[key[1]]
                cache[key] = len(positions)
                positions.append([(pa[0]+pb[0])/2,(pa[1]+pb[1])/2,(pa[2]+pb[2])/2])
            return cache[key]
        refined = []
        for a,b,c in triangles:
            ab,bc,ca = midpoint(a,b),midpoint(b,c),midpoint(c,a)
            refined.extend([[a,ab,ca],[ab,b,bc],[ca,bc,c],[ab,bc,ca]])
        triangles = refined
    return {"label":f"{base['label']}细分x{iterations}","positions":positions,"triangles":triangles}
