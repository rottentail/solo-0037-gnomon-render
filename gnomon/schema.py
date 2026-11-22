import math
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class MeshInput(StrictModel):
    label: str = Field(min_length=1, max_length=120)
    positions: list[tuple[float, float, float]] = Field(min_length=3, max_length=20000)
    triangles: list[tuple[int, int, int]] = Field(min_length=1, max_length=30000)
    normals: list[tuple[float, float, float]] | None = None
    uv: list[tuple[float, float]] | None = None


class TextureInput(StrictModel):
    label: str = Field(min_length=1, max_length=120)
    png_base64: str = Field(max_length=6000000)
    role: Literal["color", "normal"] = "color"


class Transform(StrictModel):
    translate: tuple[float, float, float] = (0, 0, 0)
    rotate: tuple[float, float, float] = (0, 0, 0)
    scale: tuple[float, float, float] = (1, 1, 1)

    @model_validator(mode="after")
    def invertible(self):
        if any(abs(x) < 1e-6 or abs(x) > 10000 for x in self.scale):
            raise ValueError("scale must be invertible and bounded")
        if any(abs(x) > 1e6 for x in self.translate):
            raise ValueError("translation outside supported coordinates")
        return self


class Surface(StrictModel):
    color: tuple[float, float, float] = (0.6, 0.6, 0.6)
    texture: str | None = None
    normal_texture: str | None = None
    normal_strength: float = Field(default=1, ge=0, le=4)
    wrap: Literal["repeat", "clamp"] = "repeat"
    alpha_cutoff: float = Field(default=0.5, ge=0, le=1)
    unlit: bool = False
    double_sided: bool = True

    @model_validator(mode="after")
    def color_range(self):
        if any(x < 0 or x > 1 for x in self.color):
            raise ValueError("surface color must be sRGB in [0,1]")
        return self


class Node(StrictModel):
    key: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$")
    parent: str | None = None
    mesh: str | None = None
    pose: Transform = Field(default_factory=Transform)
    surface: Surface = Field(default_factory=Surface)
    visible: bool = True


class Lens(StrictModel):
    projection: Literal["perspective", "orthographic"] = "perspective"
    eye: tuple[float, float, float] = (3, 2, 4)
    target: tuple[float, float, float] = (0, 0, 0)
    up: tuple[float, float, float] = (0, 1, 0)
    near: float = Field(default=0.1, ge=0.001)
    far: float = Field(default=100, le=1e6)
    fov: float = Field(default=45, ge=5, le=150)
    span: float = Field(default=4, ge=0.001, le=1e6)

    @model_validator(mode="after")
    def depth_range(self):
        if any(abs(v)>1e6 for vector in (self.eye,self.target,self.up) for v in vector):
            raise ValueError("camera coordinates outside supported range")
        if self.far <= self.near:
            raise ValueError("far must exceed near")
        return self


class Light(StrictModel):
    toward: tuple[float, float, float] = (2, 4, 3)
    energy: float = Field(default=0.8, ge=0, le=8)
    ambient: float = Field(default=0.2, ge=0, le=2)

    @model_validator(mode="after")
    def bounded_direction(self):
        if any(abs(v)>1e6 for v in self.toward):
            raise ValueError("light direction outside supported range")
        return self


class SceneInput(StrictModel):
    label: str = Field(min_length=1, max_length=120)
    nodes: list[Node] = Field(min_length=1, max_length=100)
    lens: Lens = Field(default_factory=Lens)
    light: Light = Field(default_factory=Light)
    background: tuple[float, float, float] = (0.035, 0.045, 0.07)

    @model_validator(mode="after")
    def background_range(self):
        if any(x < 0 or x > 1 for x in self.background):
            raise ValueError("background must be sRGB in [0,1]")
        return self


class RenderInput(StrictModel):
    scene: str
    width: int = Field(default=256, ge=16, le=1024)
    height: int = Field(default=256, ge=16, le=1024)
    samples: Literal[1, 2, 3] = 1
    region: tuple[int, int, int, int] | None = None

    @model_validator(mode="after")
    def region_bounds(self):
        if self.region:
            x, y, w, h = self.region
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > self.width or y+h > self.height:
                raise ValueError("region is x,y,width,height in full-frame coordinates")
        return self


class CompareInput(StrictModel):
    left: str
    right: str
    threshold: float = Field(default=0.02, ge=0, le=1)


class SheetInput(StrictModel):
    scene: str
    angles: list[float] = Field(default_factory=lambda: [0, 90, 180, 270], min_length=2, max_length=12)
    size: int = Field(default=128, ge=32, le=256)
    columns: int = Field(default=4, ge=1, le=6)
