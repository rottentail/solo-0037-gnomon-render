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


def slats_mesh():
    # 三根细竖条薄盒，间隙可透视远景；用于前景遮挡的显露关系校样。
    positions, triangles = [], []
    for cx in (-0.55, 0.0, 0.55):
        base = len(positions)
        x0, x1, y0, y1, z0, z1 = cx-0.045, cx+0.045, -1.6, 1.6, -0.03, 0.03
        positions += [[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                      [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]]
        for a,b,c,d in [(0,1,2,3),(5,4,7,6),(4,0,3,7),(1,5,6,2),(3,2,6,7),(4,5,1,0)]:
            triangles += [[base+a,base+b,base+c],[base+a,base+c,base+d]]
    return {"label":"近景细条","positions":positions,"triangles":triangles}


def lattice_texture():
    # 细镂空窗棂：暖色细条与透明方孔，alpha_cutoff 裁切后可透视。
    image = Image.new("RGBA",(64,64))
    for y in range(64):
        for x in range(64):
            bar = x%8<2 or y%8<2
            color = (226,188,104,255) if (x//8+y//8)%2 else (196,150,86,255)
            image.putpixel((x,y),color if bar else (30,40,60,0))
    output = BytesIO()
    image.save(output,format="PNG")
    return {"label":"细镂空窗棂纹理","png_base64":base64.b64encode(output.getvalue()).decode()}


def depth_scene(wedge,screen,slats,lattice):
    # 近景细条(≈5.8)、中景楔体(≈9.0)、远景镂空纹理(≈14.0)与远块(≈12.5)，
    # 沿视线拉开层次，配合 focus 请求展示清晰平面前后的自然模糊。
    return {"label":"景深校样","nodes":[
        {"key":"subject","mesh":wedge,"pose":{"translate":[-1.2,0.15,0],"rotate":[0,-25,0]},"surface":{"color":[0.87,0.45,0.2]}},
        {"key":"near-slats","mesh":slats,"pose":{"translate":[0.8,0.9,3.2]},"surface":{"color":[0.82,0.84,0.9],"unlit":True}},
        {"key":"far-lattice","mesh":screen,"pose":{"translate":[-0.5,0.0,-5],"scale":[3.6,3.6,3.6]},"surface":{"texture":lattice,"color":[1,1,1],"alpha_cutoff":0.5,"unlit":True}},
        {"key":"far-block","mesh":wedge,"pose":{"translate":[2.3,-0.5,-3.5],"rotate":[0,30,0],"scale":[0.5,0.5,0.5]},"surface":{"color":[0.25,0.65,0.75]}}
    ],"lens":{"eye":[0,1.2,9],"target":[0,0.9,0],"near":0.1,"far":40,"fov":32},
    "light":{"toward":[2,4,5],"energy":0.9,"ambient":0.25},"background":[0.04,0.05,0.08]}
