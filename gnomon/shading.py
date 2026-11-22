import numpy as np
from .color import srgb_to_linear
from .errors import RenderError, require
from .math3d import unit
from .texture import sample_texture, sample_normal


def perturbed_normals(normals, tangent, bitangent, normal_map, uv, wrap, strength):
    """Apply a tangent-space normal map using the per-vertex world frame.

    The sampled vector is normalized, then rebuilt in world space with the
    interpolated tangent frame. Handedness is carried by the world bitangent:
    on a mirrored node the bitangent follows the mirror too, so slopes keep the
    side they had in mesh space instead of flipping. The strength scales only
    the tangential components, so 0 leaves the geometric normal intact.
    """
    sample = sample_normal(normal_map,uv,wrap)
    sample[:,0] *= strength
    sample[:,1] *= strength
    length = np.linalg.norm(sample,axis=1,keepdims=True)
    if np.any(length <= 1e-12):
        raise RenderError("normal_sample","normal texture contains a zero-length normal vector")
    sample /= length
    # Perspective-correct interpolation of frame vectors is only approximately
    # orthonormal; renormalizing keeps dents and bumps symmetric.
    tangent = unit(tangent)
    bitangent = unit(bitangent)
    world = sample[:,0:1]*tangent + sample[:,1:2]*bitangent + sample[:,2:3]*normals
    return unit(world)


def shade(surface, normals, uv, texture, light, tangent=None, bitangent=None, normal_map=None):
    base = np.tile(srgb_to_linear(surface["color"]),(len(normals),1))
    alpha = np.ones(len(normals))
    if texture is not None:
        texels = sample_texture(texture,uv,surface["wrap"])
        base *= texels[:,:3]
        alpha = texels[:,3]
    length = np.linalg.norm(normals,axis=1,keepdims=True)
    normals = np.divide(normals,length,out=np.zeros_like(normals),where=length>1e-12)
    if normal_map is not None:
        require(tangent is not None and bitangent is not None,"normal_tangent",
                "normal mapped surface requires per-vertex tangent and bitangent attributes")
        # The shading normal is perturbed by texture data alone; no new geometry
        # is introduced. It is also what the world-normal channel reports, so a
        # color-channel dent and a world-normal-channel dent describe one face.
        normals = perturbed_normals(normals,tangent,bitangent,normal_map,uv,
                                    surface["wrap"],surface["normal_strength"])
    if not surface["unlit"]:
        cosine = np.maximum(0,normals @ unit(light["toward"]))
        base *= (light["ambient"]+light["energy"]*cosine)[:,None]
    return base,alpha>=surface["alpha_cutoff"],normals
