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


def _png_base64(image):
    output = BytesIO()
    image.save(output,format="PNG")
    return base64.b64encode(output.getvalue()).decode()


def stripes_texture():
    # 左半红、右半蓝,用于验证 UV 接缝两侧的采样结果。
    image = Image.new("RGBA",(16,16))
    for y in range(16):
        for x in range(16):
            image.putpixel((x,y),(200,30,30,255) if x < 8 else (30,30,200,255))
    return _png_base64(image)


# 导入验证样本的正交镜头与光照:世界坐标到像素是线性映射,法线余弦即亮度。
PROOF_LENS = {"projection":"orthographic","eye":[0,0,5],"target":[0,0,0],"up":[0,1,0],
              "near":0.1,"far":20,"span":2.4}
PROOF_LIGHT = {"toward":[0,0,1],"energy":1.0,"ambient":0.0}

PROOF_OBJ = """# gnomon 导入证明样本:凹多边形、共享位置的 UV 接缝、独立法线索引、反向绕序。
mtllib proof.mtl
o proof
v -1.5 -0.8 0
v -0.6 0 0
v -1.5 0.8 0
v -1.1 0 0
v -0.55 -0.55 0
v 0 -0.55 0
v 0 0.55 0
v -0.55 0.55 0
v 0.55 -0.55 0
v 0.55 0.55 0
v 0.7 -0.5 0
v 1.2 -0.5 0
v 1.2 0.5 0
v 0.7 0.5 0
v 1.7 -0.5 0
v 1.7 0.5 0
v 0.3 0.8 0
v -0.3 0.8 0
v 0 1.05 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vn 0 0 1
vn 0.7 0 0.7
g dart
usemtl matte_red
f 1 2 3 4
g pair
usemtl flat_teal
f 5//1 6//1 7//1 8//1
usemtl tilted_teal
f 6//2 9//2 10//2 7//2
g seams
usemtl striped
f 11/1 12/2 13/3 14/4
f 12/5 15/6 16/7 13/8
g winder
usemtl matte_green
f 17 18 19
"""

PROOF_MTL = """newmtl matte_red
Kd 0.8 0.15 0.15
newmtl flat_teal
Kd 0.15 0.7 0.7
newmtl tilted_teal
Kd 0.15 0.7 0.7
newmtl striped
Kd 1 1 1
map_Kd stripes.png
newmtl matte_green
Kd 0.2 0.75 0.2
"""

FLAWED_OBJ = """mtllib flawed.mtl
o flawed
v -1 -1 0
v 1 -1 0
v 1 1 0
v -1 1 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
g panel
usemtl ghost
f 1/1 2/2 3/3 4/4
"""

FLAWED_MTL = """newmtl ghost
Kd 0.9 0.9 0.9
d 0.4
Ks 0.9 0.9 0.9
Ns 50
map_Kd missing.png
map_Bump bump.png
Pr 0.5
illum 4
"""


def obj_proof_sample():
    return {"label":"导入证明样本","obj_text":PROOF_OBJ,"mtl_text":PROOF_MTL,
            "images":{"stripes.png":stripes_texture()},
            "lens":dict(PROOF_LENS),"light":dict(PROOF_LIGHT)}


def obj_flawed_sample():
    # 无法表示的透明度/高光/凹凸与缺失贴图:导入应解释而不是拒绝。
    return {"label":"诊断样本","obj_text":FLAWED_OBJ,"mtl_text":FLAWED_MTL,"images":{}}
