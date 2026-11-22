import math
from typing import Literal
import numpy as np
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
    wrap: Literal["repeat", "clamp"] = "repeat"
    alpha_cutoff: float = Field(default=0.5, ge=0, le=1)
    unlit: bool = False
    double_sided: bool = True

    @model_validator(mode="after")
    def color_range(self):
        if any(x < 0 or x > 1 for x in self.color):
            raise ValueError("surface color must be sRGB in [0,1]")
        return self


class Decal(StrictModel):
    texture: str
    # Projector origin and projection direction in world coordinates.
    origin: tuple[float, float, float]
    direction: tuple[float, float, float]
    # World-space size of the emblem: one full texture span maps to this length,
    # so the scale stays identical across every surface the projection crosses.
    width: float = Field(gt=0, le=10000)
    height: float = Field(gt=0, le=10000)
    # Optional in-plane right/up hints controlling emblem orientation.
    axis_u: tuple[float, float, float] | None = None
    axis_v: tuple[float, float, float] | None = None
    # Grazing-angle termination: cosine of the angle between surface normal and
    # projection direction at which the emblem is cut (hard) or fully faded.
    grazing: float = Field(default=0.05, ge=0, le=1)
    grazing_mode: Literal["hard", "fade"] = "hard"
    blend: Literal["over", "multiply"] = "over"
    tint: tuple[float, float, float] = (1, 1, 1)
    alpha_cutoff: float = Field(default=0.5, ge=0, le=1)
    wrap: Literal["repeat", "clamp"] = "clamp"

    @model_validator(mode="after")
    def projection_frame(self):
        if all(abs(v) < 1e-12 for v in self.direction):
            raise ValueError("decal direction must be a nonzero vector")
        if any(abs(v) > 1e6 for v in (*self.origin, *self.direction)):
            raise ValueError("decal frame outside supported coordinates")
        if any(x < 0 or x > 1 for x in self.tint):
            raise ValueError("decal tint must be sRGB in [0,1]")
        if self.axis_u is not None and all(abs(v) < 1e-12 for v in self.axis_u):
            raise ValueError("decal axis_u must be a nonzero vector")
        if self.axis_v is not None and all(abs(v) < 1e-12 for v in self.axis_v):
            raise ValueError("decal axis_v must be a nonzero vector")
        if self.axis_u is not None and self.axis_v is not None:
            u, v = np.asarray(self.axis_u), np.asarray(self.axis_v)
            if abs(float(np.dot(u / np.linalg.norm(u), v / np.linalg.norm(v)))) > 1e-6:
                raise ValueError("decal axis_u and axis_v must be perpendicular")
        return self


class Node(StrictModel):
    key: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$")
    parent: str | None = None
    mesh: str | None = None
    pose: Transform = Field(default_factory=Transform)
    surface: Surface = Field(default_factory=Surface)
    visible: bool = True
    # Projected decal is a node-level overlay: the immutable mesh and other nodes
    # sharing it are never rewritten.
    decal: Decal | None = None


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
