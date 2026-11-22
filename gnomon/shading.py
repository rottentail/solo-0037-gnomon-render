import numpy as np
from .color import srgb_to_linear
from .math3d import unit
from .texture import sample_texture


def shade(surface, normals, uv, texture, light):
    base = np.tile(srgb_to_linear(surface["color"]),(len(normals),1))
    # Texture alpha remains a binary cutout mask; material opacity is the
    # continuous translucency control. Effective alpha is their product.
    mask = np.ones(len(normals))
    if texture is not None:
        texels = sample_texture(texture,uv,surface["wrap"])
        base *= texels[:,:3]
        mask = (texels[:,3] >= surface["alpha_cutoff"]).astype(np.float64)
    length = np.linalg.norm(normals,axis=1,keepdims=True)
    normals = np.divide(normals,length,out=np.zeros_like(normals),where=length>1e-12)
    if not surface["unlit"]:
        cosine = np.maximum(0,normals @ unit(light["toward"]))
        base *= (light["ambient"]+light["energy"]*cosine)[:,None]
    alpha = mask*surface["opacity"]
    return base,alpha,normals
