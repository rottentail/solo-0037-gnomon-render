import json
import threading
import time
from .catalog import Catalog, canonical
from .artifacts import Artifacts
from .mesh import prepare_mesh
from .texture import decode_texture
from .scene import compile_scene, scene_summary
from .raster import render_samples
from .channels import resolve_samples, preview_channels, packed_channels, unpack_channels
from .inspection import inspect_pixel
from .comparison import compare_frames
from .sheet import build_sheet
from .bundle import build_bundle
from .errors import RenderError, require
from .schema import MeshInput, TextureInput, SceneInput, RenderInput, CompareInput, SheetInput


class RendererService:
    def __init__(self,root):
        self.catalog = Catalog(root)
        self.artifacts = Artifacts(root)
        self.operation_lock = threading.RLock()
        self.recovered_operations = self.catalog.recover_interrupted()

    def close(self):
        self.catalog.close()

    def add_mesh(self,payload):
        parsed = MeshInput.model_validate(payload).model_dump()
        return self.catalog.create("mesh",prepare_mesh(parsed))

    def add_texture(self,payload):
        parsed = TextureInput.model_validate(payload).model_dump()
        return self.catalog.create("texture",decode_texture(parsed))

    def add_scene(self,payload,predecessor=None):
        parsed = SceneInput.model_validate(payload).model_dump()
        with self.operation_lock:
            if predecessor:
                self.catalog.get(predecessor,"scene")
            compiled = compile_scene(parsed,self.catalog.get)
            return self.catalog.create("scene",parsed,details={"predecessor":predecessor,**scene_summary(compiled)})

    def scene_state(self,identity,target):
        require(target in {"active","archived"},"scene_state","unknown scene state")
        with self.operation_lock:
            self.catalog.get(identity,"scene")
            expected = {"active"} if target=="archived" else {"archived"}
            return self.catalog.transition(identity,expected,target)

    def resource_state(self,identity,target):
        require(target in {"active","retired"},"resource_state","unknown resource state")
        with self.operation_lock:
            record = self.catalog.get(identity)
            require(record["kind"] in {"mesh","texture"},"resource_kind","only mesh and texture can retire")
            expected = {"active"} if target=="retired" else {"retired"}
            return self.catalog.transition(identity,expected,target)

    def preflight(self,request):
        parsed = RenderInput.model_validate(request).model_dump()
        scene = self.catalog.get(parsed["scene"],"scene")
        require(scene["state"]=="active","scene_archived","scene is archived",409)
        compiled = compile_scene(scene["payload"],self.catalog.get,allow_retired=True)
        region = parsed["region"] or [0,0,parsed["width"],parsed["height"]]
        from .raster import sample_buffer_bytes, MAX_TRANSLUCENT_LAYERS, LAYER_RECORD_BYTES, BUFFER_BUDGET
        estimated = sample_buffer_bytes(region[2],region[3],parsed["samples"],compiled["has_translucent"])
        return {**scene_summary(compiled),"estimated_buffer_bytes":estimated,
                "translucent":compiled["has_translucent"],
                "translucent_layer_cap":MAX_TRANSLUCENT_LAYERS,
                "layer_record_bytes":LAYER_RECORD_BYTES,
                "buffer_budget_bytes":BUFFER_BUDGET,
                "admissible":estimated<=BUFFER_BUDGET and compiled["triangles"]<=60000,
                "region":region,"request":parsed}

    def _snapshot(self,scene):
        compiled = compile_scene(scene["payload"],self.catalog.get,allow_retired=True)
        snapshot = {"scene_id":scene["id"],"scene_digest":scene["digest"],"scene":scene["payload"],
                    "resources":{key:{"kind":value["kind"],"digest":value["digest"],"payload":value["payload"],"state":value["state"]} for key,value in compiled["resources"].items()}}
        return compiled,snapshot

    def _complete(self,record,files,details):
        descriptors = self.artifacts.write(record["id"],files)
        return self.catalog.transition(record["id"],{record["state"]},"complete",{**details,"artifacts":descriptors})

    def _fail(self,record,exc):
        detail = {"code":exc.code,"detail":exc.detail} if isinstance(exc,RenderError) else {"code":"operation_failed","detail":str(exc)}
        self.catalog.transition(record["id"],{record["state"]},"failed",{"error":detail})

    def render(self,request):
        parsed = RenderInput.model_validate(request).model_dump()
        with self.operation_lock:
            scene = self.catalog.get(parsed["scene"],"scene")
            require(scene["state"]=="active","scene_archived","scene is archived",409)
            compiled,snapshot = self._snapshot(scene)
            record = self.catalog.create("render",{"request":parsed,"snapshot":snapshot},"queued")
            record = self.catalog.transition(record["id"],{"queued"},"running")
            try:
                started = time.perf_counter()
                buffers,metadata = render_samples(compiled,parsed)
                channels = resolve_samples(buffers,parsed["samples"])
                metadata.update({key:parsed[key] for key in ("width","height","samples")})
                metadata.update(node_map=compiled["node_map"],scene_digest=scene["digest"],
                                elapsed_seconds=time.perf_counter()-started)
                files = preview_channels(channels)
                files["channels.npz"] = packed_channels(channels)
                files["metadata.json"] = canonical(metadata).encode()
                return self._complete(record,files,{"metadata":metadata})
            except Exception as exc:
                self._fail(record,exc)
                raise

    def inspect(self,identity,x,y):
        record = self.catalog.get(identity,"render")
        channels = unpack_channels(self.artifacts.read(record,"channels.npz"))
        return inspect_pixel(channels,record["details"]["metadata"],x,y)

    def compare(self,request):
        parsed = CompareInput.model_validate(request).model_dump()
        with self.operation_lock:
            left = self.catalog.get(parsed["left"],"render")
            right = self.catalog.get(parsed["right"],"render")
            record = self.catalog.create("comparison",parsed,"measuring")
            try:
                lc = unpack_channels(self.artifacts.read(left,"channels.npz"))
                rc = unpack_channels(self.artifacts.read(right,"channels.npz"))
                metrics,heat = compare_frames(lc,rc,left["details"]["metadata"],right["details"]["metadata"],parsed["threshold"])
                return self._complete(record,{"difference.png":heat,"metrics.json":canonical(metrics).encode()},{"metrics":metrics})
            except Exception as exc:
                self._fail(record,exc)
                raise

    def sheet(self,request):
        parsed = SheetInput.model_validate(request).model_dump()
        with self.operation_lock:
            scene = self.catalog.get(parsed["scene"],"scene")
            require(scene["state"]=="active","scene_archived","scene is archived",409)
            _,snapshot = self._snapshot(scene)
            record = self.catalog.create("sheet",{"request":parsed,"snapshot":snapshot},"building")
            try:
                def render_view(lens):
                    payload = {**snapshot["scene"],"lens":lens}
                    def frozen(identity,kind):
                        return snapshot["resources"][identity]
                    compiled = compile_scene(payload,frozen,allow_retired=True)
                    config = {"width":parsed["size"],"height":parsed["size"],"samples":1,"region":None}
                    buffers,metadata = render_samples(compiled,config)
                    channels = resolve_samples(buffers,1)
                    return preview_channels(channels)["color.png"],metadata
                png,index = build_sheet(snapshot["scene"],parsed["angles"],parsed["size"],parsed["columns"],render_view)
                return self._complete(record,{"sheet.png":png,"index.json":canonical(index).encode()},{"views":len(index),"index":index})
            except Exception as exc:
                self._fail(record,exc)
                raise

    def bundle(self,identity):
        with self.operation_lock:
            render = self.catalog.get(identity,"render")
            require(render["state"]=="complete","bundle_incomplete","render must be complete",409)
            record = self.catalog.create("bundle",{"render":identity},"packing")
            try:
                files = {name:self.artifacts.read(render,name) for name in render["details"]["artifacts"]}
                raw,manifest = build_bundle(render,files)
                return self._complete(record,{"render.zip":raw},{"file_count":len(files),"manifest_format":manifest["format"]})
            except Exception as exc:
                self._fail(record,exc)
                raise

    def record_summary(self,record):
        return {key:record[key] for key in ("id","kind","state","digest","details","created")}
