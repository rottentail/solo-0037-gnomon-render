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


def box_mesh():
    # A closed box with per-corner normals and UVs. Each side owns a full
    # atlas tile, so UV and normal values jump across a shared edge while the
    # vertex indices, triangle indices and face identity stay unchanged.
    positions = [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                 [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]
    triangles = [[4,5,6],[4,6,7],
                 [1,0,3],[1,3,2],
                 [5,1,2],[5,2,6],
                 [0,4,7],[0,7,3],
                 [7,6,2],[7,2,3],
                 [0,1,5],[0,5,4]]
    axis = [[0,0,1],[0,0,-1],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]]
    corner_normals = [[side]*3 for side in axis for _ in range(2)]
    tile = [[[0,0],[1,0],[1,1]],[[0,0],[1,1],[0,1]]]
    return {"label":"闭合分面盒","positions":positions,"triangles":triangles,
            "corner_normals":corner_normals,"corner_uv":[tile[i%2] for i in range(len(triangles))]}


def face_material_scene(mesh,texture):
    # The legacy node surface is the default for every unlisted side. Two
    # groups override the top and the bottom only; the grille texture's holes
    # reveal the solid cavity behind it.
    return {"label":"镂空与实心分面盒","nodes":[
        {"key":"box","mesh":mesh,
         "surface":{"color":[0.82,0.42,0.16]},
         "material_groups":[
             {"name":"grille","faces":[8,9],
              "surface":{"texture":texture,"color":[1,1,1],"alpha_cutoff":0.5}},
             {"name":"cavity","faces":[10,11],
              "surface":{"color":[0.1,0.16,0.3],"unlit":True}}]}],
        "lens":{"eye":[2.4,4.4,2.4],"target":[0,0,0],"near":0.1,"far":30},
        "light":{"toward":[1,3,4],"energy":0.85,"ambient":0.18}}


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
