"""Projected-decal math.

The emblem is carried by an orthographic projector defined in world space.
Every shaded fragment keeps its own material; the decal only contributes an
overlay whose scale is the projector's world-space size on every surface.
"""
import numpy as np
from .color import srgb_to_linear
from .math3d import unit
from .texture import sample_texture


# Candidates used to build a stable right-handed projector basis.
_FRAME_HINTS = (np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0]))


def projection_frame(decal):
    """Return (origin, u, v, forward, half_width, half_height) world frame."""
    forward = unit(np.asarray(decal["direction"], dtype=np.float64))
    if decal.get("axis_u") is not None:
        u = unit(np.asarray(decal["axis_u"], dtype=np.float64))
        u = unit(u - forward * float(np.dot(u, forward)))
    else:
        hint = next((h for h in _FRAME_HINTS if abs(float(np.dot(h, forward))) < 0.95), _FRAME_HINTS[0])
        u = unit(np.cross(forward, hint))
    v = unit(np.cross(u, forward))
    if decal.get("axis_v") is not None:
        requested = unit(np.asarray(decal["axis_v"], dtype=np.float64))
        if float(np.dot(requested, v)) < 0:
            u, v = -u, -v
    origin = np.asarray(decal["origin"], dtype=np.float64)
    return {"origin": origin, "u": u, "v": v, "forward": forward,
            "half_width": decal["width"] / 2.0, "half_height": decal["height"] / 2.0}


def project_uv(frame, world):
    """World positions -> projector-plane UV in [0,1] plus signed plane extents."""
    delta = world - frame["origin"]
    su = delta @ frame["u"]
    sv = delta @ frame["v"]
    depth = delta @ frame["forward"]
    uv = np.column_stack([(su / frame["half_width"] + 1.0) / 2.0,
                          (sv / frame["half_height"] + 1.0) / 2.0])
    return uv, su, sv, depth


def grazing_weight(normals, frame, grazing, mode):
    # Projection travels along +forward; a surface accepts the emblem when its
    # normal points back toward the projector, i.e. cosine < 0.
    cosine = normals @ frame["forward"]
    facing = -cosine
    if mode == "hard":
        return (facing >= grazing).astype(np.float64)
    # Linear ramp: full weight at head-on incidence, zero at the grazing cutoff.
    weight = np.clip((facing - grazing) / max(1.0 - grazing, 1e-12), 0.0, 1.0)
    return weight


def apply_decal(decal, frame, texture, world, normals, base_color):
    """Evaluate the overlay for fragments that already passed the depth test.

    Returns the composited linear color and an alpha mask describing the actual
    decal coverage (before tinting), plus the projection depth for diagnostics.
    """
    uv, su, sv, depth = project_uv(frame, world)
    if decal["wrap"] == "clamp":
        # Clamped emblems are confined to the projector rectangle.
        inside = (np.abs(su) <= frame["half_width"]) & (np.abs(sv) <= frame["half_height"])
    else:
        # Repeating emblems tile beyond the rectangle with the same world scale.
        inside = np.ones(len(world), dtype=bool)
    facing = grazing_weight(normals, frame, decal["grazing"], decal["grazing_mode"])
    accepted = inside & (facing > 0.0)
    mask = np.zeros(len(world))
    if not np.any(accepted):
        return base_color, mask, depth
    texels = sample_texture(texture, uv[accepted], decal["wrap"])
    alpha = texels[:, 3] * facing[accepted]
    keep = alpha >= decal["alpha_cutoff"]
    accepted_indices = np.flatnonzero(accepted)
    kept = accepted_indices[keep]
    if len(kept) == 0:
        return base_color, mask, depth
    final_alpha = np.zeros(len(world))
    final_alpha[kept] = alpha[keep]
    overlay = np.zeros_like(base_color)
    tint = srgb_to_linear(np.asarray(decal["tint"], dtype=np.float64))
    overlay[kept] = texels[keep, :3] * tint
    result = base_color.copy()
    if decal["blend"] == "over":
        result[kept] = overlay[kept] * final_alpha[kept, None] + base_color[kept] * (1.0 - final_alpha[kept, None])
    else:
        # Multiply keeps the underlying shading fully visible beneath the emblem.
        result[kept] = base_color[kept] * (1.0 - final_alpha[kept, None]) + base_color[kept] * overlay[kept] * final_alpha[kept, None]
    mask[kept] = final_alpha[kept]
    return result, mask, depth
