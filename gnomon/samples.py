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


def room_mesh():
    # Interior surfaces of a 6x4x6 axis-aligned room centred on the origin.
    # Faces wind so their geometric normals face the centre, and authored
    # vertex normals point inward as well, since the panoramic eye is inside.
    x0,x1,y0,y1,z0,z1 = -3,3,-2,2,-3,3
    positions = [[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                 [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]]
    faces = [[0,3,2],[0,2,1],   # back wall z=z0, normal +z
             [4,5,6],[4,6,7],   # front wall z=z1, normal -z
             [0,1,5],[0,5,4],   # floor y=y0, normal +y
             [3,7,6],[3,6,2],   # ceiling y=y1, normal -y
             [0,4,7],[0,7,3],   # left wall x=x0, normal +x
             [1,2,6],[1,6,5]]   # right wall x=x1, normal -x
    # Each corner belongs to three walls; any inward wall normal is valid
    # because the walls meet at right angles and the eye never sees an edge-on.
    vertex_normals = [[1,1,1],[-1,1,1],[-1,-1,1],[1,-1,1],
                      [1,1,-1],[-1,1,-1],[-1,-1,-1],[1,-1,-1]]
    uv = [[(p[0]+3)/6,(p[2]+3)/6] for p in positions]
    return {"label":"环视内室盒面","positions":positions,"triangles":faces,
            "normals":vertex_normals,"uv":uv}


def panorama_scene(room, wedge, texture):
    # The eye is inside the room and looks toward -z, so the longitude seam
    # (yaw +/-180) is the +z wall behind the eye. A column straddles that
    # seam, the ceiling provides an overhead surface and the wedge sits close
    # to the eye as a foreground occluder.
    return {"label":"三百六十度内室环视","nodes":[
        {"key":"room","mesh":room,"surface":{"color":[0.78,0.8,0.84],"double_sided":True}},
        {"key":"seam-column","mesh":wedge,
         "pose":{"translate":[0,0,3.0],"rotate":[0,0,0],"scale":[1.1,2.2,1.1]},
         "surface":{"color":[0.9,0.55,0.2],"double_sided":True}},
        {"key":"ceiling-cap","mesh":wedge,
         "pose":{"translate":[0.6,1.75,0.9],"rotate":[0,40,0],"scale":[0.9,0.25,0.9]},
         "surface":{"color":[0.35,0.55,0.85],"double_sided":True}},
        {"key":"near-occluder","mesh":wedge,
         "pose":{"translate":[0.55,-0.25,0.95],"rotate":[0,-35,0],"scale":[0.45,0.45,0.45]},
         "surface":{"texture":texture,"color":[1,1,1],"alpha_cutoff":0.5,"double_sided":True}}
    ],"lens":{"projection":"equirectangular","eye":[0,0,0],"target":[0,0,-1],
              "up":[0,1,0],"near":0.05,"far":20},
    "light":{"toward":[0.4,1,0.6],"energy":0.9,"ambient":0.25}}
