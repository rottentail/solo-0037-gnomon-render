import base64
import binascii
from io import BytesIO
import numpy as np
from PIL import Image, UnidentifiedImageError
from .color import srgb_to_linear
from .errors import RenderError, require


def decode_texture(payload):
    try:
        raw = base64.b64decode(payload["png_base64"], validate=True)
        require(len(raw) <= 4000000, "texture_size", "encoded texture exceeds 4 MB")
        with Image.open(BytesIO(raw)) as image:
            require(image.format == "PNG", "texture_format", "only PNG textures are accepted")
            require(image.width <= 1024 and image.height <= 1024, "texture_size", "texture dimensions exceed 1024")
            require(not image.info.get("icc_profile"), "texture_profile", "baseline expects unprofiled sRGB PNG")
            require(not getattr(image, "is_animated", False), "texture_frames", "animated PNG is not accepted")
            pixels = np.asarray(image.convert("RGBA"))
    except (binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise RenderError("texture_decode", str(exc)) from exc
    return {"label":payload["label"], "width":pixels.shape[1], "height":pixels.shape[0],
            "rgba":pixels.tolist(), "encoding":"srgb", "alpha":"straight"}


def texture_array(texture):
    rgba = np.asarray(texture["rgba"], dtype=np.float64)/255
    rgba[:,:,:3] = srgb_to_linear(rgba[:,:,:3])
    return rgba


def sample_texture(texture, uv, wrap):
    height, width = texture.shape[:2]
    coords = np.mod(uv,1) if wrap == "repeat" else np.clip(uv,0,1)
    # UV origin is bottom-left; PNG rows have top-left origin.
    x = coords[:,0] * width - 0.5
    y = (1-coords[:,1]) * height - 0.5
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x-x0)[:,None], (y-y0)[:,None]
    def at(ix, iy):
        if wrap == "repeat":
            ix, iy = ix % width, iy % height
        else:
            ix, iy = np.clip(ix,0,width-1), np.clip(iy,0,height-1)
        return texture[iy,ix]
    # Interpolate premultiplied colors to avoid dark fringes in alpha-cut textures.
    prepared = texture.copy()
    prepared[:,:,:3] *= prepared[:,:,3:4]
    texture = prepared
    result = ((1-fx)*(1-fy)*at(x0,y0) + fx*(1-fy)*at(x0+1,y0)
              + (1-fx)*fy*at(x0,y0+1) + fx*fy*at(x0+1,y0+1))
    result[:,:3] = np.divide(result[:,:3],result[:,3:4],out=np.zeros_like(result[:,:3]),where=result[:,3:4]>1e-12)
    return result
