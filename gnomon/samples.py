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


def panel_texture():
    """Opaque sRGB color panel: two-tone tiles plus a diagonal band.

    Kept fully opaque so a lit/unlit pair of renders can be divided per pixel
    without alpha-cutout gaps complicating the comparison.
    """
    image = Image.new("RGBA",(64,64))
    for y in range(64):
        for x in range(64):
            band = (x+y) % 32 < 6
            color = (235,206,120,255) if band else ((71,158,196,255) if (x//16+y//16)%2 else (196,92,64,255))
            image.putpixel((x,y),color)
    output = BytesIO()
    image.save(output,format="PNG")
    return {"label":"双色面板纹理","png_base64":base64.b64encode(output.getvalue()).decode()}


def seam_ramp_mesh():
    """A two-panel pitched ramp whose ridge is duplicated to author a UV seam.

    Vertices 0..3 are the left panel; 4..7 repeat the same ridge positions for
    the right panel so each chart owns its UVs. Each side samples a mirrored
    half of the shared texture, which is the usual way one image covers both
    panels of a seam. The tangent frame must stay per-chart here: blending the
    two ridge copies would tilt the normals into the gap between the panels.
    """
    ridge_back,ridge_front = (0.0,0.7,-0.8),(0.0,0.7,0.8)
    left_normal,right_normal = (-0.573,0.819,0),(0.573,0.819,0)
    positions = [[-1.0,0.0,-0.8],ridge_back,ridge_front,[-1.0,0.0,0.8],
                 [0.0,0.7,-0.8],[0.0,0.7,0.8],[1.0,0.0,-0.8],[1.0,0.0,0.8]]
    triangles = [[0,2,1],[0,3,2],[4,7,6],[4,5,7]]
    normals = [left_normal]*4+[right_normal]*4
    uv = [[0,0],[0.5,0],[0.5,1],[0,1],
          [0.5,0],[0.5,1],[0,0],[0,1]]
    return {"label":"接缝斜面","positions":positions,"triangles":triangles,
            "normals":normals,"uv":uv}


def collapsed_uv_mesh():
    """The ramp with every UV on one line: valid UV storage, zero UV area.

    A color texture can still sample this (it lands on a single texel row),
    but no tangent direction follows from it, so a normal map must be refused
    instead of shading with an invented frame.
    """
    mesh = seam_ramp_mesh()
    mesh["label"] = "退化UV斜面"
    mesh["uv"] = [[i*0.125,0.0] for i in range(len(mesh["positions"]))]
    return mesh


def ripple_normal_texture():
    """Tangent-space ridge field: +u and -u slopes encoded as RGB bytes.

    128 is a flat normal, so a mid-gray pixel means "no tilt". The pattern
    repeats across u, so charts meeting at the seam keep shading continuous.
    """
    size = 64
    image = Image.new("RGBA",(size,size))
    for y in range(size):
        for x in range(size):
            slope = 0.8*math.cos(2*math.pi*4*x/size)
            vector = (slope,0.0,1.0)
            scale = math.sqrt(sum(c*c for c in vector))
            r,g,b = [int(round((c/scale*0.5+0.5)*255)) for c in vector]
            image.putpixel((x,y),(r,g,b,255))
    output = BytesIO()
    image.save(output,format="PNG")
    return {"label":"脊线法线图","png_base64":base64.b64encode(output.getvalue()).decode(),"role":"normal"}


def flat_normal_texture():
    """A flat (0,0,1) tangent-space normal map: bumps nothing by construction."""
    image = Image.new("RGBA",(8,8),(128,128,255,255))
    output = BytesIO()
    image.save(output,format="PNG")
    return {"label":"平直法线图","png_base64":base64.b64encode(output.getvalue()).decode(),"role":"normal"}


def normal_material_scene(mesh,color_texture,normal_texture):
    """Four instances in a row: upright, reflected, stretched, and no map.

    The reflected instance scales X by -1, which flips the projected winding;
    the stretched instance applies non-uniform scale. Both must keep the
    ridges on the same side, so a dent in the color channel stays a dent in
    the world-normal channel. The fourth instance is the unmapped control.
    """
    surface = {"texture":color_texture,"normal_texture":normal_texture,"color":[1,1,1]}
    spread = 1.55
    return {"label":"法线材质校样","nodes":[
        {"key":"upright","mesh":mesh,"surface":surface,"pose":{"translate":[-3*spread,0,0]}},
        {"key":"reflected","mesh":mesh,"surface":surface,"pose":{"translate":[-spread,0,0],"scale":[-1,1,1]}},
        {"key":"stretched","mesh":mesh,"surface":surface,"pose":{"translate":[spread,0,0],"scale":[0.75,1.5,1]}},
        {"key":"unmapped","mesh":mesh,"surface":{**surface,"normal_texture":None},
         "pose":{"translate":[3*spread,0,0]}}
    ],"lens":{"eye":[0,3.2,8.6],"target":[0,0.15,0],"near":0.1,"far":40},
    "light":{"toward":[-1,2,1.5],"energy":0.9,"ambient":0.15}}


def compose_scene(mesh,screen,texture):
    return {"label":"折面色片静帧","nodes":[
        {"key":"root","pose":{"rotate":[0,15,0]}},
        {"key":"wedge","parent":"root","mesh":mesh,"surface":{"color":[0.87,0.38,0.19]},"pose":{"rotate":[0,-20,0]}},
        {"key":"foil","parent":"root","mesh":screen,"pose":{"translate":[-0.7,0.4,-0.9],"scale":[0.7,0.7,0.7]},"surface":{"texture":texture,"color":[1,1,1],"alpha_cutoff":0.5}},
        {"key":"echo","mesh":mesh,"pose":{"translate":[1.0,-0.6,-0.8],"scale":[0.35,0.35,0.35]},"surface":{"color":[0.2,0.7,0.8]}}
    ],"lens":{"eye":[3,2.2,4],"target":[0,0,0],"near":0.1,"far":30},
    "light":{"toward":[1,3,4],"energy":0.85,"ambient":0.18}}
