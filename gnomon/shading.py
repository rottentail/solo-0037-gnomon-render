import numpy as np
from .color import srgb_to_linear
from .math3d import unit
from .texture import sample_texture


def shade(surface, normals, uv, texture, light):
    base = np.tile(srgb_to_linear(surface["color"]),(len(normals),1))
    alpha = np.ones(len(normals))
    if texture is not None:
        texels = sample_texture(texture,uv,surface["wrap"])
        base *= texels[:,:3]
        alpha = texels[:,3]
    length = np.linalg.norm(normals,axis=1,keepdims=True)
    normals = np.divide(normals,length,out=np.zeros_like(normals),where=length>1e-12)
    if not surface["unlit"]:
        cosine = np.maximum(0,normals @ unit(light["toward"]))
        base *= (light["ambient"]+light["energy"]*cosine)[:,None]
    return base,alpha>=surface["alpha_cutoff"],normals
