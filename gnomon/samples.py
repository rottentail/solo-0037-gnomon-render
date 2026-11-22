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


def bevel_block_mesh():
    # Pentagonal prism extruded along X: front wall, a 45-degree top chamfer,
    # rear wall and two caps. Side winding follows the CCW polygon order so the
    # geometric normals point outward.
    polygon = [[-1, -1], [1, -1], [1, 1], [0, 2], [-1, 2]]
    positions = [[x, y, z] for x in (-1, 1) for (z, y) in polygon]
    triangles = []
    # -X cap (normal -X).
    for i in range(1, 4):
        triangles.append([0, i, i + 1])
    # +X cap (normal +X).
    for i in range(1, 4):
        triangles.append([5, 6 + i, 5 + i])
    # Side quads, vertices i (-X) and i+5 (+X); CCW around the pentagon.
    for i in range(5):
        j = (i + 1) % 5
        triangles.append([i, 5 + i, 5 + j])
        triangles.append([i, 5 + j, j])
    return {"label": "斜面五棱柱", "positions": positions,
            "triangles": triangles}


def emblem_texture():
    # RGBA emblem on a transparent field: ring, diagonal slash and center mark.
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    gold, ink = (240, 190, 92, 255), (30, 36, 52, 255)
    for y in range(64):
        for x in range(64):
            dx, dy = x - 31.5, y - 31.5
            radius = (dx * dx + dy * dy) ** 0.5
            if 22 <= radius <= 27:
                image.putpixel((x, y), gold)
            elif radius < 10 and abs(dx) < 4:
                image.putpixel((x, y), ink)
            elif radius < 10 and abs(dy) < 4:
                image.putpixel((x, y), ink)
            elif radius < 22 and abs(dx + dy) < 2.5:
                image.putpixel((x, y), gold)
    output = BytesIO()
    image.save(output, format="PNG")
    return {"label": "金墨徽记", "png_base64": base64.b64encode(output.getvalue()).decode()}


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


def decal_scene(block, emblem, occluder):
    # One projector crosses the front wall and the 45-degree chamfer at a single
    # world-space scale. A small plate occludes part of the projection volume.
    decal = {"texture": emblem, "origin": [0, 0, 4], "direction": [0, 0, -1],
             "width": 3.0, "height": 3.0, "grazing": 0.25, "grazing_mode": "hard"}
    return {"label": "投影徽记校样", "nodes": [
        {"key": "block", "mesh": block, "surface": {"color": [0.72, 0.74, 0.78], "unlit": True},
         "decal": decal},
        # Another node sharing the same mesh: no decal, proving the overlay is
        # node-level and never rewrites the shared mesh.
        {"key": "plain_twin", "mesh": block,
         "pose": {"translate": [-3.4, 0, 0], "scale": [0.8, 0.8, 0.8]},
         "surface": {"color": [0.35, 0.55, 0.75], "unlit": True}},
        # Mirrored instance carrying its own emblem: winding flips, projection
        # still resolves through world-space depth and facing tests. The mirror
        # faces +Z after the negative X scale, so its projector looks toward -Z
        # from the +Z side in world coordinates.
        {"key": "mirror_twin", "mesh": block,
         "pose": {"translate": [3.4, 0, 0], "scale": [-0.8, 0.8, 0.8]},
         "surface": {"color": [0.62, 0.45, 0.68], "unlit": True},
         "decal": {**decal, "origin": [3.4, 0, 4], "direction": [0, 0, -1]}},
        {"key": "occluder", "mesh": occluder,
         "pose": {"translate": [0.9, -0.2, 1.6], "scale": [0.45, 0.45, 0.45]},
         "surface": {"color": [0.12, 0.14, 0.18], "unlit": True}}
    ], "lens": {"projection": "perspective", "eye": [0, 1.4, 8.5], "target": [0, 0.2, 0],
               "near": 0.1, "far": 40},
        "light": {"toward": [1, 3, 4], "energy": 0.8, "ambient": 0.25}}
